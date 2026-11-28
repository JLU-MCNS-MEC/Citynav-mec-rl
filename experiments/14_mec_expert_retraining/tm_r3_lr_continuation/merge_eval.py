#!/usr/bin/env python3
"""Merge parallel val_seen evaluations into one 2498-denominator summary.

Usage:
  python merge_eval.py --dirs d1 d2 --denominator 2498 --out merged_summary.json

Rows are read from `journal.part*.jsonl` in each directory, deduplicated by
episode_id (last wins), restricted to split == "val_seen", and summarised with
the requested denominator (number of val_seen tasks, NOT the number of rows
completed).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def read_rows(dirs):
    """Read per-episode rows from a finished harness directory.

    The evaluator writes journal.part*.jsonl while running and consolidates to
    journal.jsonl on completion; a final val_seen_episode_results.csv is always
    written.  All three sources are supported (journal preferred).
    """
    rows = {}
    for d in dirs:
        files = sorted(Path(d).glob("journal.part*.jsonl"))
        if not files and (Path(d) / "journal.jsonl").exists():
            files = [Path(d) / "journal.jsonl"]
        if files:
            for f in files:
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        rows[r["episode_id"]] = r      # last write wins
            continue
        csv = Path(d) / "val_seen_episode_results.csv"
        if csv.exists():
            import pandas as pd
            for r in pd.read_csv(csv).to_dict("records"):
                rows[r["episode_id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--denominator", type=int, default=2498)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    rows = read_rows(args.dirs)
    val = {k: r for k, r in rows.items() if r.get("split") == "val_seen"}
    n = len(val)
    succ = sum(1 for r in val.values() if r.get("success"))
    stopped = [r for r in val.values() if r.get("stopped")]
    fails = Counter(r.get("failure", "") for r in val.values())

    summary = {
        "completed_episodes": n,
        "expected_episodes": args.denominator,
        "complete": n >= args.denominator,
        "sr": succ / args.denominator if args.denominator else None,
        "sr_completed": succ / n if n else None,
        "osr": sum(1 for r in val.values() if r.get("osr")) / args.denominator
        if args.denominator else None,
        "spl": float(np.mean([r["spl"] for r in val.values()])) if n else None,
        "ne_m": float(np.mean([r["ne_m"] for r in val.values()])) if n else None,
        "mean_steps": float(np.mean([r["steps"] for r in val.values()])) if n else None,
        "stop_rate": len(stopped) / n if n else None,
        "stop_precision": (sum(1 for r in stopped if r.get("success")) / len(stopped))
        if stopped else None,
        "success_tasks": succ,
        "failure_tasks": n - succ,
        "failure_counts": dict(fails),
        "sources": list(args.dirs),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    if args.csv:
        cols = ["episode_id", "split", "map", "length_group", "success", "osr",
                "spl", "ne_m", "steps", "stopped", "stop_reason", "failure"]
        pd.DataFrame([{c: r.get(c) for c in cols} for r in val.values()]).to_csv(
            args.csv, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
