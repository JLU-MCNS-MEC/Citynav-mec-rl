#!/usr/bin/env python3
"""Merge sharded closed-loop JSON outputs into one split-level result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="full", choices=("full", "map_free"))
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    episodes = []
    missing = []
    for index in range(args.shards):
        path = (
            args.out_dir
            / f"{args.split}_{args.model}_{args.variant}_shard{index}.json"
        )
        if not path.exists():
            missing.append(path.name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes.extend(payload["episodes"])
    if missing:
        raise SystemExit(f"missing shard output(s): {', '.join(missing)}")
    if not episodes:
        raise SystemExit("no episodes found")

    def _mean(key: str) -> float:
        return float(np.mean([float(item[key]) for item in episodes]))

    summary = {
        "split": args.split,
        "model": args.model,
        "variant": args.variant,
        "use_retrieval": args.variant == "full",
        "episodes": len(episodes),
        "success_rate": _mean("success"),
        "oracle_success_rate": _mean("oracle_success"),
        "spl": _mean("spl"),
        "ne_m": _mean("ne_m"),
        "ndtw": _mean("ndtw"),
        "mean_steps": _mean("steps"),
        "stop_rate": _mean("stopped"),
        "mean_step_latency_s": float(
            np.mean([
                item["mean_step_latency_s"]
                for item in episodes
                if item["mean_step_latency_s"]
            ])
        ),
        "shards": args.shards,
        "test_unseen_read": False,
    }
    output = args.output or (
        args.out_dir
        / f"{args.split}_{args.model}_{args.variant}_summary.json"
    )
    output.write_text(
        json.dumps(
            {"summary": summary, "episodes": episodes},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {output} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
