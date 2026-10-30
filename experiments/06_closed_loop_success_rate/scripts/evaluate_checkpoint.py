#!/usr/bin/env python3
"""Run the Stage-06 closed loop for the Stage-05 seed-42 checkpoint.

* loads ``best.pt`` with ``strict=True`` and refuses any checkpoint whose
  SHA-256 is not the one recorded in the Stage-05 report;
* one JSON file per episode, written atomically, plus a JSONL journal line;
* a restart re-reads the journal, verifies the checkpoint/manifest identity and
  skips every completed episode instead of recomputing it;
* ``--workers N`` runs N processes (one CUDA context each); episodes are
  independent, so parallelism does not change any result.

Usage:
  python evaluate_checkpoint.py --limit 12 --workers 1          # smoke
  python evaluate_checkpoint.py --workers 4                      # full eval
  python evaluate_checkpoint.py --split val_seen --cohort A      # subset
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from closed_loop import (DEFAULT_STEP_CAP, evaluate_task, load_model,  # noqa: E402
                         record_summary)

ROOT = HERE.parent.parent
RESULTS = HERE.parent / "results/seed42"
CHECKPOINT_SHA = "077f01a4f1a643c3db7ff6a4eb57980a401232bcc4810b05925fb1fceccc0ce2"


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_journal(results_dir: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    for path in sorted(results_dir.glob("journal*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[rec["episode_id"]] = rec
    return done


def completed(results_dir: Path, checkpoint_sha: str, manifest_sha: str) -> dict[str, dict]:
    """Only episodes whose JSON is present and whose identity still matches."""
    done = {}
    for eid, rec in load_journal(results_dir).items():
        if rec.get("checkpoint_sha") != checkpoint_sha or rec.get("manifest_sha") != manifest_sha:
            raise RuntimeError(
                f"journal entry {eid} was produced with a different checkpoint/manifest "
                f"({rec.get('checkpoint_sha')} / {rec.get('manifest_sha')}); "
                "move results aside instead of mixing runs")
        path = results_dir / "episodes" / f"{eid.replace(':', '_')}.json"
        if path.exists():
            done[eid] = rec
    return done


def _worker(job: tuple) -> dict:
    (part, tasks, checkpoint, expected_sha, manifest_sha, step_cap, device, out_dir,
     cands_root) = job
    out_dir = Path(out_dir)
    episodes = out_dir / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore

    model, _ = load_model(Path(checkpoint), device, expected_sha)
    encoder = FrozenEncoder(device)
    rasters = RasterStore()
    cands = CandidateStore(Path(cands_root)) if cands_root else CandidateStore()
    journal = out_dir / f"journal.part{part:02d}.jsonl"
    started = time.time()
    ok = 0
    total = len(tasks)
    with journal.open("a", encoding="utf-8") as jf:
        for i, task in enumerate(tasks, 1):
            task = dict(task, checkpoint_sha=expected_sha, manifest_sha=manifest_sha)
            record = evaluate_task(model, encoder, rasters, cands, task, step_cap, device)
            _atomic_json(episodes / f"{record['episode_id'].replace(':', '_')}.json", record)
            jf.write(json.dumps(record_summary(record)) + "\n")
            jf.flush()
            ok += 1
            if i % 20 == 0 or i == total:
                rate = i / max(time.time() - started, 1e-9)
                eta = (total - i) / rate if rate else 0
                print(f"[w{part}] {i}/{total} episodes | {rate:.2f} ep/s | "
                      f"elapsed {time.time() - started:.0f}s | eta {eta:.0f}s", flush=True)
    return {"part": part, "episodes": ok, "seconds": round(time.time() - started, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(
        ROOT / "05_transformer_m_training/runs_v2/seed42/best.pt"))
    parser.add_argument("--manifest", default=str(RESULTS / "manifest.json"))
    parser.add_argument("--out-dir", default=str(RESULTS))
    parser.add_argument("--split", nargs="+", default=["val_seen", "val_unseen"])
    parser.add_argument("--cohort", nargs="+", default=["A", "B"])
    parser.add_argument("--limit", type=int, default=0,
                        help="first N pending tasks only (smoke test)")
    parser.add_argument("--step-cap", type=int, default=DEFAULT_STEP_CAP)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", default="",
                        help="candidate-library directory (default: cache_v2/candidates24)")
    parser.add_argument("--task-ids", default="",
                        help="comma-separated episode ids to run (smoke selection)")
    parser.add_argument("--task-file", default="",
                        help="file with one episode id per line (overrides --task-ids)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = (manifest_path.parent / "manifest.sha256").read_text(encoding="utf-8").strip()

    if manifest.get("checkpoint_sha") != CHECKPOINT_SHA:
        raise RuntimeError(f"manifest checkpoint {manifest.get('checkpoint_sha')} "
                           f"!= expected {CHECKPOINT_SHA}")
    # verify the checkpoint bytes before loading them
    load_model(Path(args.checkpoint), "cpu", CHECKPOINT_SHA)

    tasks = [t for t in manifest["tasks"]
             if t["split"] in args.split and t["cohort"] in args.cohort]
    if args.task_file:
        wanted = {line.strip() for line in Path(args.task_file).read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.startswith("#")}
        missing = wanted - {t["episode_id"] for t in tasks}
        if missing:
            raise RuntimeError(f"task file has {len(missing)} ids not in the manifest: "
                               f"{sorted(missing)[:5]}")
        tasks = [t for t in tasks if t["episode_id"] in wanted]
    elif args.task_ids:
        wanted = {x.strip() for x in args.task_ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t["episode_id"] in wanted]
    done = completed(out_dir, CHECKPOINT_SHA, manifest_sha)
    pending = [t for t in tasks if t["episode_id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"tasks: {len(tasks)} selected | {len(done)} already done | {len(pending)} pending "
          f"| step_cap={args.step_cap} | workers={args.workers}", flush=True)
    if not pending:
        print("nothing to do", flush=True)
        return 0

    config = {"checkpoint": str(args.checkpoint), "checkpoint_sha": CHECKPOINT_SHA,
              "manifest_sha": manifest_sha, "step_cap": args.step_cap,
              "workers": args.workers, "splits": args.split, "cohorts": args.cohort,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    _atomic_json(out_dir / "eval_config.json", config)

    workers = max(1, min(args.workers, len(pending)))
    chunks = [pending[i::workers] for i in range(workers)]
    jobs = [(i, chunks[i], args.checkpoint, CHECKPOINT_SHA, manifest_sha,
             args.step_cap, args.device, str(out_dir), args.candidates)
            for i in range(workers)]
    started = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for msg in pool.imap_unordered(_worker, jobs):
            print(f"  worker done: {msg}", flush=True)
    elapsed = time.time() - started

    # merge part journals into the canonical journal (latest entry per id wins)
    merged = load_journal(out_dir)
    tmp = out_dir / "journal.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        for eid in sorted(merged):
            fh.write(json.dumps(merged[eid]) + "\n")
    os.replace(tmp, out_dir / "journal.jsonl")

    total_sr = sum(1 for r in merged.values() if r.get("success")) / max(len(merged), 1)
    print(f"finished {len(pending)} episodes in {elapsed / 60:.2f} min "
          f"({elapsed / max(len(pending), 1):.2f} s/episode); "
          f"journal {len(merged)} episodes, SR={total_sr:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
