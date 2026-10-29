#!/usr/bin/env python3
"""Experiment 01: trajectory length, teacher-action count and horizon audit.

The official CityNav paper defines Easy/Medium/Hard by start-to-target distance
(171 m / 258 m percentiles of the training set); it does not define
short/medium/long trajectories and does not report average action counts.
The official episode generator keeps teacher action sequences with at most
500 steps.  This script therefore computes, from the local official splits:

  * raw trajectory point count (NOT the action count),
  * lookahead-teacher action count with a generous safety bound,
  * human / teacher path length in metres,
  * start-to-target distance and the official difficulty bin,
  * action distribution and coverage at candidate step caps.

Only train_seen / val_seen / val_unseen are read.  test_unseen stays sealed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TeacherConfig  # noqa: E402
from src.data.trajectory_adapter import TrajectoryAdapter, load_trajectory  # noqa: E402

ACTION_NAMES = ["Stop", "MoveForward", "TurnRight", "TurnLeft", "GoUp", "GoDown"]
CAPS = (50, 100, 150, 200, 250, 300, 400, 500)
OFFICIAL_DISTANCE_BINS = ((171.0, "easy"), (258.0, "medium"), (float("inf"), "hard"))

_RAW: dict[tuple[str, str], list] = {}
_ADAPTER: TrajectoryAdapter | None = None


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _distance_bin(distance_m: float) -> str:
    for upper, name in OFFICIAL_DISTANCE_BINS:
        if distance_m < upper:
            return name
    return "hard"


def _episode_key(item: dict) -> tuple:
    trajectory = item.get("trajectory") or [[0.0]]
    return (
        item.get("area"),
        int(item.get("block", 0)),
        tuple(item.get("object_ids", [])),
        tuple(item.get("ann_ids", [])),
        len(trajectory),
        tuple(trajectory[0][:3]),
        tuple(trajectory[-1][:3]),
    )


def _process(task: tuple[str, str, int]) -> dict:
    split, label, index = task
    raw = _RAW[(split, label)][index]
    poses = load_trajectory(raw["trajectory"])
    marker = raw["marker_positions"][-1]
    target = raw["target_positions"][-1]
    assert _ADAPTER is not None
    teacher = _ADAPTER.rollout(poses, marker, target)

    trajectory_points = len(raw["trajectory"])
    human_xyz = np.asarray([point[:3] for point in raw["trajectory"]], dtype=np.float64)
    teacher_xyz = np.asarray([pose.xyz for pose in teacher.poses], dtype=np.float64)
    start = human_xyz[0]
    target_xyz = np.asarray(target, dtype=np.float64)
    start_target_m = float(np.linalg.norm(start - target_xyz))
    actions = teacher.actions
    counts = Counter(int(a) for a in actions)

    row = {
        "split": split,
        "difficulty": raw.get("_difficulty", "unlabelled"),
        "index": index,
        "map_name": f"{raw['area']}_block_{int(raw['block'])}",
        "trajectory_points": trajectory_points,
        "teacher_steps": int(len(actions)),
        "stop_reason": teacher.stop_reason,
        "success_20m": bool(teacher.success),
        "final_target_m": float(teacher.final_target_distance_m),
        "final_marker_m": float(teacher.final_marker_distance_m),
        "start_target_m": start_target_m,
        "human_path_m": _polyline_length(human_xyz),
        "teacher_path_m": _polyline_length(teacher_xyz),
        "distance_bin": _distance_bin(start_target_m),
        "actions_per_point": float(len(actions) / max(trajectory_points, 1)),
    }
    for action, name in enumerate(ACTION_NAMES):
        row[f"action_{name}"] = int(counts.get(action, 0))
    for cap in CAPS:
        row[f"steps_gt_{cap}"] = bool(len(actions) > cap)
    return row


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 3),
        "std": round(float(arr.std(ddof=0)), 3),
        "min": round(float(arr.min()), 3),
        "p25": round(float(np.percentile(arr, 25)), 3),
        "median": round(float(np.percentile(arr, 50)), 3),
        "p75": round(float(np.percentile(arr, 75)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "max": round(float(arr.max()), 3),
    }


def _summarize(rows: list[dict], group: str) -> dict:
    fields = (
        "trajectory_points",
        "teacher_steps",
        "start_target_m",
        "human_path_m",
        "teacher_path_m",
        "actions_per_point",
    )
    out = {
        "group": group,
        "n": len(rows),
        "teacher_success_rate_20m": round(
            float(np.mean([r["success_20m"] for r in rows])), 4
        ),
        "teacher_stop_reason": dict(
            Counter(r["stop_reason"] for r in rows).most_common()
        ),
        "fields": {field: _stats([r[field] for r in rows]) for field in fields},
        "coverage": {
            str(cap): {
                "episodes_over_cap": int(sum(r[f"steps_gt_{cap}"] for r in rows)),
                "fraction_over_cap": round(
                    float(np.mean([r[f"steps_gt_{cap}"] for r in rows])), 6
                ),
            }
            for cap in CAPS
        },
        "action_distribution": {
            name: {
                "count": int(sum(r[f"action_{name}"] for r in rows)),
                "fraction": round(
                    float(
                        sum(r[f"action_{name}"] for r in rows)
                        / max(sum(r["teacher_steps"] for r in rows), 1)
                    ),
                    6,
                ),
            }
            for name in ACTION_NAMES
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "CITYNAV_DATA_ROOT",
                "/home/20261024_1/Workspace/DATA/sbfnav-data",
            )
        ),
    )
    ap.add_argument("--subdir", default="original_citynav")
    ap.add_argument(
        "--splits", nargs="+", default=["val_seen", "val_unseen", "train_seen"]
    )
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parents[1] / "results")
    args = ap.parse_args()

    global _ADAPTER
    _ADAPTER = TrajectoryAdapter(TeacherConfig(max_steps=int(args.max_steps)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    tasks: list[tuple[str, str, int]] = []
    for split in args.splits:
        # Full split plus the official difficulty files.  The easy/medium/hard
        # files partition the split, so only the full file is needed for the
        # canonical population; labels are attached from the official subsets.
        path = args.data_root / args.subdir / f"citynav_{split}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        _RAW[(split, "full")] = raw
        difficulty_of: dict[int, str] = {}
        for label in ("easy", "medium", "hard"):
            diff_path = args.data_root / args.subdir / f"citynav_{split}_{label}.json"
            if not diff_path.exists():
                continue
            for item in json.loads(diff_path.read_text(encoding="utf-8")):
                difficulty_of[_episode_key(item)] = label
        for item in raw:
            item["_difficulty"] = difficulty_of.get(_episode_key(item), "unlabelled")
        for index, item in enumerate(raw):
            tasks.append((split, "full", index))
        print(
            f"{split}: {len(raw)} episodes, labelled "
            f"{sum(1 for x in raw if x['_difficulty'] != 'unlabelled')}",
            flush=True,
        )

    rows: list[dict] = []
    if args.workers > 1:
        import multiprocessing as mp

        with mp.Pool(processes=int(args.workers)) as pool:
            for row in pool.imap_unordered(_process, tasks, chunksize=16):
                rows.append(row)
    else:
        for task in tasks:
            rows.append(_process(task))
    rows.sort(key=lambda r: (r["split"], r["index"]))
    print(f"processed {len(rows)} episodes in {time.time()-started:.1f}s", flush=True)

    csv_path = args.out_dir / "episode_metrics.csv"
    fieldnames = list(rows[0].keys())
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"official": {
        "difficulty_definition": "start-to-target distance; 33rd/66th training percentiles 171 m / 258 m",
        "short_medium_long_defined": False,
        "average_action_count_reported": False,
        "reported_average_path_length_m": {"human": 508, "shortest": 290},
        "official_episode_max_steps_filter": 500,
        "official_eval_max_timestep_default": 200,
        "success_dist_m": 20.0,
        "teacher_stop_distance_m": 5.0,
    }, "groups": []}

    for split in args.splits:
        split_rows = [r for r in rows if r["split"] == split]
        summary["groups"].append(_summarize(split_rows, f"split={split}"))
        for difficulty in ("easy", "medium", "hard"):
            part = [r for r in split_rows if r["difficulty"] == difficulty]
            if part:
                summary["groups"].append(
                    _summarize(part, f"split={split},difficulty={difficulty}")
                )
        for distance_bin in ("easy", "medium", "hard"):
            part = [r for r in split_rows if r["distance_bin"] == distance_bin]
            if part:
                summary["groups"].append(
                    _summarize(part, f"split={split},distance_bin={distance_bin}")
                )
    summary["groups"].append(_summarize(rows, "all"))

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {csv_path} and {args.out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
