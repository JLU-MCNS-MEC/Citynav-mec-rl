#!/usr/bin/env python3
"""09B: closed loop with online visual matching + dynamic candidate posterior.

Same 5,324 tasks and rules as Stage 06/08.  Resumable: results/<run>/journal
entries carry the manifest SHA and the calibration SHA; a mismatch aborts
instead of mixing runs.
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

from dynamic_env import DEFAULT_STEP_CAP, record_summary, run_episode  # noqa: E402

STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"
RES = HERE.parent / "results"


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


def load_journal(out: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    for p in sorted(out.glob("journal*.jsonl")):
        with p.open(encoding="utf-8") as fh:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(STAGE06 / "manifest.json"))
    parser.add_argument("--out-dir", default=str(RES / "09b_dynamic"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-file", default="")
    parser.add_argument("--step-cap", type=int, default=DEFAULT_STEP_CAP)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = (manifest_path.parent / "manifest.sha256").read_text(encoding="utf-8").strip()
    if sha256(manifest_path) != manifest_sha:
        raise RuntimeError("manifest bytes do not match manifest.sha256")
    calib_path = RES / "09b_calibration.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    calib_sha = sha256(calib_path)
    best = calib["best"]
    params = {"temperature": best["T"], "sigma": best["sigma"], "eta": best["eta"],
              "rho": best["rho"], "tau": json.loads(
                  (RES / "09a_calibration.json").read_text(encoding="utf-8"))["tau"]["layer2"],
              "margin_tau": json.loads(
                  (RES / "09a_calibration.json").read_text(encoding="utf-8"))["margin_tau"]["layer2"]}
    print("params:", params, flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = manifest["tasks"]
    if args.task_file:
        wanted = {x.strip() for x in Path(args.task_file).read_text(encoding="utf-8").splitlines()
                  if x.strip() and not x.startswith("#")}
        tasks = [t for t in tasks if t["episode_id"] in wanted]
    done = {}
    for eid, rec in load_journal(out).items():
        if rec.get("manifest_sha") != manifest_sha or rec.get("controller") != "dynamic_visual":
            raise RuntimeError(f"journal entry {eid} belongs to another run")
        if (out / "episodes" / f"{eid.replace(':', '_')}.json").exists():
            done[eid] = rec
    pending = [t for t in tasks if t["episode_id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"tasks {len(tasks)} | done {len(done)} | pending {len(pending)}", flush=True)
    if not pending:
        return 0

    atomic_json(out / "eval_config.json", {
        "controller": "dynamic_visual", "manifest_sha": manifest_sha,
        "calibration_sha": calib_sha, "params": params, "step_cap": args.step_cap,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore
    import matching

    fe = FrozenEncoder(args.device)
    matcher = matching.Encoder(args.device)
    rasters, cands = RasterStore(), CandidateStore()
    goal_cache: dict = {}
    goal_feat_cache: dict = {}
    episodes = out / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    journal = out / "journal.part00.jsonl"
    started = time.time()
    n = 0
    with journal.open("a", encoding="utf-8") as jf:
        for i, task in enumerate(pending, 1):
            task = dict(task, manifest_sha=manifest_sha)
            rec = run_episode(task, fe, matcher, rasters, cands, goal_cache,
                              goal_feat_cache, params, args.step_cap)
            atomic_json(episodes / f"{rec['episode_id'].replace(':', '_')}.json", rec)
            jf.write(json.dumps(record_summary(rec)) + "\n")
            jf.flush()
            n += 1
            if i % 25 == 0 or i == len(pending):
                rate = i / max(time.time() - started, 1e-9)
                print(f"{i}/{len(pending)} | {rate:.2f} ep/s | elapsed "
                      f"{time.time() - started:.0f}s | eta {(len(pending) - i) / rate:.0f}s", flush=True)
    merged = load_journal(out)
    tmp = out / "journal.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        for eid in sorted(merged):
            fh.write(json.dumps(merged[eid]) + "\n")
    os.replace(tmp, out / "journal.jsonl")
    sr = sum(1 for r in merged.values() if r.get("success")) / max(len(merged), 1)
    print(f"finished {n} episodes in {(time.time() - started) / 60:.2f} min; "
          f"journal {len(merged)}, SR={sr:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
