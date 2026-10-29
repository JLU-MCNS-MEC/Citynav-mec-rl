#!/usr/bin/env python3
"""B. Teacher STOP label vs the official 20 m success radius.

For every Train-Seen episode, replay the official lookahead teacher, then
measure the 3-D distance between the pose that emits STOP and
``target_positions[-1]`` (the evaluation goal).  The teacher stops within 5 m of
the *marker*, which is not necessarily the evaluation target.

This is offline diagnostics only: raw labels and the official success radius
are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_data_config  # noqa: E402
from src.data.citynav_dataset import CityNavDataset  # noqa: E402

SUCCESS_RADIUS_M = 20.0


def _quantiles(values):
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train_seen")
    ap.add_argument("--out-dir", type=Path, default=AUDIT / "results")
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_config = load_data_config()
    dataset = CityNavDataset(data_config, args.split, include_processed_descriptions=False)
    records = dataset.records[: args.limit] if args.limit else dataset.records

    rows = []
    for idx, record in enumerate(records):
        result = record.teacher(dataset.teacher_config)
        stop_pose = result.poses[-1]
        target = np.asarray(record.target_xyz, dtype=np.float64)
        marker = np.asarray(record.marker_xyz, dtype=np.float64)
        dist3d = float(np.linalg.norm(np.asarray(stop_pose.xyz) - target))
        dist_xy = float(np.linalg.norm(np.asarray(stop_pose.xy) - target[:2]))
        marker_dist = float(np.linalg.norm(np.asarray(stop_pose.xyz) - marker))
        rows.append({
            "index": idx,
            "map": record.map_name,
            "episode_id": record.episode_id,
            "split": args.split,
            "stop_reason": result.stop_reason,
            "teacher_steps": int(result.steps),
            "dist_stop_to_target_3d_m": dist3d,
            "dist_stop_to_target_xy_m": dist_xy,
            "dist_stop_to_marker_3d_m": marker_dist,
            "dist_marker_to_target_m": float(record.dist_marker_to_target),
            "final_target_distance_m": float(result.final_target_distance_m),
            "success_at_20m": bool(dist3d <= SUCCESS_RADIUS_M),
            "target_xyz": list(target),
            "marker_xyz": list(marker),
            "stop_xyz": list(stop_pose.xyz),
        })

    teacher_stops = [r for r in rows if r["stop_reason"] == "teacher_stop_distance"]
    all_stops = rows
    dists = [r["dist_stop_to_target_3d_m"] for r in teacher_stops]

    summary = {
        "split": args.split,
        "records": len(rows),
        "stop_reason_counts": dict(Counter(r["stop_reason"] for r in rows)),
        "teacher_stop_records": len(teacher_stops),
        "teacher_stop_distance_to_target_3d_m": _quantiles(dists),
        "teacher_stop_distance_to_target_xy_m": _quantiles(
            [r["dist_stop_to_target_xy_m"] for r in teacher_stops]
        ),
        "dist_marker_to_target_m": _quantiles(
            [r["dist_marker_to_target_m"] for r in rows]
        ),
        "conflicts": {
            "over_20m": int(sum(d > SUCCESS_RADIUS_M for d in dists)),
            "fraction_over_20m": float(np.mean([d > SUCCESS_RADIUS_M for d in dists])) if dists else 0.0,
            "over_10m": int(sum(d > 10.0 for d in dists)),
            "over_5m": int(sum(d > 5.0 for d in dists)),
            "over_30m": int(sum(d > 30.0 for d in dists)),
        },
        "all_stops_including_max_steps": {
            "count": len(all_stops),
            "over_20m": int(sum(r["dist_stop_to_target_3d_m"] > SUCCESS_RADIUS_M for r in all_stops)),
            "fraction_over_20m": float(np.mean([r["dist_stop_to_target_3d_m"] > SUCCESS_RADIUS_M for r in all_stops])),
        },
        "candidate_rule_offline": {
            "rule": "STOP label retained only when dist(stop, target_xyz) <= 20 m",
            "stop_labels_total": len(teacher_stops),
            "stop_labels_retained": int(sum(d <= SUCCESS_RADIUS_M for d in dists)),
            "stop_labels_removed": int(sum(d > SUCCESS_RADIUS_M for d in dists)),
            "removed_fraction": float(np.mean([d > SUCCESS_RADIUS_M for d in dists])) if dists else 0.0,
            "note": "diagnostic only; never overwrites the official labels or radius",
        },
        "per_map": {},
    }

    by_map = defaultdict(list)
    for r in teacher_stops:
        by_map[r["map"]].append(r["dist_stop_to_target_3d_m"])
    for map_name, ds in sorted(by_map.items()):
        summary["per_map"][map_name] = {
            "n": len(ds),
            "median_m": float(np.median(ds)),
            "fraction_over_20m": float(np.mean([d > SUCCESS_RADIUS_M for d in ds])),
        }

    # Keep the per-record table compact: only the fields needed to reproduce the
    # statistics (no coordinates or per-step arrays).
    csv_path = args.out_dir / f"teacher_stop_{args.split}.csv"
    fields = [
        "index", "map", "episode_id", "stop_reason", "teacher_steps",
        "dist_stop_to_target_3d_m", "dist_stop_to_target_xy_m",
        "dist_stop_to_marker_3d_m", "dist_marker_to_target_m",
        "final_target_distance_m", "success_at_20m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (args.out_dir / f"teacher_stop_{args.split}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(dists, bins=np.arange(0, 205, 5), color="#3b6ea5")
    axes[0].axvline(SUCCESS_RADIUS_M, color="#c0392b", linestyle="--",
                    label="official success radius 20 m")
    axes[0].set_xlabel("3-D distance stop pose to target_positions[-1] (m)")
    axes[0].set_ylabel("episodes")
    axes[0].set_title("Teacher STOP distance to evaluation target")
    axes[0].legend(fontsize=8)
    fracs = [
        np.mean([d <= t for d in dists])
        for t in (5, 10, 20, 30, 50)
    ]
    axes[1].bar(["<=5", "<=10", "<=20", "<=30", "<=50"], fracs, color="#4a8f6d")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fraction of teacher STOPs")
    axes[1].set_title("Cumulative agreement with 20 m criterion")
    for i, v in enumerate(fracs):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(AUDIT / "figures" / "teacher_stop_distance.png", dpi=170)
    plt.close(fig)

    print(json.dumps({k: summary[k] for k in (
        "records", "stop_reason_counts", "teacher_stop_records",
        "teacher_stop_distance_to_target_3d_m", "conflicts",
        "candidate_rule_offline",
    )}, indent=2, ensure_ascii=False))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
