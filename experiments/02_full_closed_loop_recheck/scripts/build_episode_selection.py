#!/usr/bin/env python3
"""Build disjoint, length-stratified episode lists for experiment 02.

Official CityNav has no short/medium/long definition, so this script defines
audit-local bins as tertiles of the teacher action count within each split,
computed on the official-filtered population (teacher_steps <= 500).  Each
round gets 8 episodes per bin (24 total); rounds A and B are disjoint and
attempt to spread episodes across maps.

The outputs are index lists into the original split JSON.  CityNavDataset keeps
raw record order, so an index can be used directly by closed_loop_eval.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parents[2]
ROOT = AUDIT.parent
DEFAULT_METRICS = (
    ROOT
    / "experiments/01_dataset_length_and_horizon_audit/results/episode_metrics.csv"
)


def _bin_thresholds(values: list[int]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.percentile(arr, 100.0 / 3.0)),
        float(np.percentile(arr, 200.0 / 3.0)),
    )


def _bin_of(steps: int, thresholds: tuple[float, float]) -> str:
    low, high = thresholds
    if steps <= low:
        return "short"
    if steps <= high:
        return "medium"
    return "long"


def _pick_round(pool: dict[str, list[dict]], seed: int, per_bin: int, used: set) -> list[dict]:
    rng = random.Random(seed)
    chosen: list[dict] = []
    for name in ("short", "medium", "long"):
        candidates = [row for row in pool[name] if row["index"] not in used]
        by_map: dict[str, list[dict]] = defaultdict(list)
        for row in candidates:
            by_map[row["map_name"]].append(row)
        maps = sorted(by_map)
        rng.shuffle(maps)
        for rows in by_map.values():
            rng.shuffle(rows)
        # Round-robin over maps so a round is not dominated by one map.
        cursor = {map_name: 0 for map_name in maps}
        picked = 0
        while picked < per_bin and maps:
            progressed = False
            for map_name in list(maps):
                rows = by_map[map_name]
                if cursor[map_name] >= len(rows):
                    continue
                row = rows[cursor[map_name]]
                cursor[map_name] += 1
                if row["index"] in used:
                    continue
                chosen.append(row)
                used.add(row["index"])
                picked += 1
                progressed = True
                if picked >= per_bin:
                    break
            if not progressed:
                break
        if picked < per_bin:
            raise SystemExit(f"not enough episodes for bin {name}: {picked} < {per_bin}")
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parents[1] / "selection")
    ap.add_argument("--splits", nargs="+", default=["val_seen", "val_unseen"])
    ap.add_argument("--per-bin", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.metrics.open(encoding="utf-8")))
    for row in rows:
        row["index"] = int(row["index"])
        row["teacher_steps"] = int(row["teacher_steps"])
        row["trajectory_points"] = int(row["trajectory_points"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"bins": "tertiles of teacher_steps within each split",
                     "official_population": "teacher_steps <= 500",
                     "per_bin_per_round": args.per_bin, "splits": {}}

    for split in args.splits:
        split_rows = [
            row for row in rows
            if row["split"] == split and row["teacher_steps"] <= 500
        ]
        thresholds = _bin_thresholds([row["teacher_steps"] for row in split_rows])
        for row in split_rows:
            row["bin"] = _bin_of(row["teacher_steps"], thresholds)
        pool: dict[str, list[dict]] = defaultdict(list)
        for row in split_rows:
            pool[row["bin"]].append(row)
        used: set[int] = set()
        rounds = {}
        for round_index, name in enumerate(("A", "B")):
            picked = _pick_round(pool, args.seed + 1000 * round_index, args.per_bin, used)
            picked.sort(key=lambda row: (row["bin"], row["map_name"], row["index"]))
            rounds[name] = picked
            (args.out_dir / f"{split}_round{name}_indices.json").write_text(
                json.dumps([row["index"] for row in picked]) + "\n",
                encoding="utf-8",
            )
        summary["splits"][split] = {
            "eligible": len(split_rows),
            "tertile_thresholds_teacher_steps": [round(v, 3) for v in thresholds],
            "rounds": {
                name: {
                    "n": len(picked),
                    "bins": {
                        bin_name: sum(1 for row in picked if row["bin"] == bin_name)
                        for bin_name in ("short", "medium", "long")
                    },
                    "maps": sorted({row["map_name"] for row in picked}),
                    "teacher_steps": {
                        "min": min(row["teacher_steps"] for row in picked),
                        "max": max(row["teacher_steps"] for row in picked),
                    },
                    "episodes": picked,
                }
                for name, picked in rounds.items()
            },
        }
        print(split, thresholds, {name: len(v) for name, v in rounds.items()})

    (args.out_dir / "selection.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out_dir/'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
