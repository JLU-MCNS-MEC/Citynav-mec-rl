#!/usr/bin/env python3
"""Run a belief-only geometric controller over the Stage-06 task manifest.

* uses the same manifest as Stage 06 (5,324 tasks, same cohorts/length groups);
* no Transformer checkpoint is loaded, so the only identity to verify on resume
  is the manifest SHA-256 and the controller name;
* one atomic JSON per episode plus a JSONL journal; completed episodes are
  skipped after verification.

Usage:
  python evaluate_geometry.py --controller top1 --limit 12 --task-file smoke.txt
  python evaluate_geometry.py --controller top1
  python evaluate_geometry.py --controller top5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from geometry_controller import make as make_controller  # noqa: E402
from geometry_env import DEFAULT_STEP_CAP, record_summary, run_episode  # noqa: E402

STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_journal(out_dir: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    for path in sorted(out_dir.glob("journal*.jsonl")):
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


def completed(out_dir: Path, controller: str, manifest_sha: str) -> dict[str, dict]:
    out = {}
    for eid, rec in load_journal(out_dir).items():
        if rec.get("controller") != controller or rec.get("manifest_sha") != manifest_sha:
            raise RuntimeError(f"journal entry {eid} belongs to another controller/manifest "
                               f"({rec.get('controller')} / {rec.get('manifest_sha')})")
        if (out_dir / "episodes" / f"{eid.replace(':', '_')}.json").exists():
            out[eid] = rec
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["top1", "top5"], required=True)
    parser.add_argument("--manifest", default=str(STAGE06 / "manifest.json"))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--split", nargs="+", default=["val_seen", "val_unseen"])
    parser.add_argument("--cohort", nargs="+", default=["A", "B"])
    parser.add_argument("--task-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--step-cap", type=int, default=DEFAULT_STEP_CAP)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = (manifest_path.parent / "manifest.sha256").read_text(encoding="utf-8").strip()
    if sha256(manifest_path) != manifest_sha:
        raise RuntimeError("manifest bytes do not match manifest.sha256")
    out_dir = Path(args.out_dir) if args.out_dir else HERE.parent / f"results/{args.controller}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t for t in manifest["tasks"]
             if t["split"] in args.split and t["cohort"] in args.cohort]
    if args.task_file:
        wanted = {x.strip() for x in Path(args.task_file).read_text(encoding="utf-8").splitlines()
                  if x.strip() and not x.startswith("#")}
        tasks = [t for t in tasks if t["episode_id"] in wanted]
    done = completed(out_dir, args.controller, manifest_sha)
    pending = [t for t in tasks if t["episode_id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"controller={args.controller} | tasks {len(tasks)} | done {len(done)} | "
          f"pending {len(pending)} | step_cap {args.step_cap}", flush=True)
    if not pending:
        print("nothing to do", flush=True)
        return 0

    atomic_json(out_dir / "eval_config.json", {
        "controller": args.controller, "manifest": str(manifest_path),
        "manifest_sha": manifest_sha, "step_cap": args.step_cap,
        "splits": args.split, "cohorts": args.cohort,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore

    encoder = FrozenEncoder(args.device)
    rasters, cands = RasterStore(), CandidateStore()
    controller = make_controller(args.controller)
    target_cache: dict = {}
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    journal = out_dir / "journal.part00.jsonl"
    started = time.time()
    n_done = 0
    with journal.open("a", encoding="utf-8") as jf:
        for i, task in enumerate(pending, 1):
            task = dict(task, manifest_sha=manifest_sha)
            record = run_episode(task, controller, encoder, rasters, cands,
                                 args.step_cap, target_cache)
            atomic_json(episodes_dir / f"{record['episode_id'].replace(':', '_')}.json", record)
            jf.write(json.dumps(record_summary(record)) + "\n")
            jf.flush()
            n_done += 1
            if i % 50 == 0 or i == len(pending):
                rate = i / max(time.time() - started, 1e-9)
                eta = (len(pending) - i) / rate if rate else 0
                print(f"{i}/{len(pending)} | {rate:.1f} ep/s | elapsed "
                      f"{time.time() - started:.0f}s | eta {eta:.0f}s", flush=True)
    # merge part journals
    merged = load_journal(out_dir)
    tmp = out_dir / "journal.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        for eid in sorted(merged):
            fh.write(json.dumps(merged[eid]) + "\n")
    os.replace(tmp, out_dir / "journal.jsonl")
    elapsed = time.time() - started
    sr = sum(1 for r in merged.values() if r.get("success")) / max(len(merged), 1)
    print(f"finished {n_done} episodes in {elapsed / 60:.2f} min "
          f"({elapsed / max(n_done, 1):.3f} s/episode); journal {len(merged)}, SR={sr:.4f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
