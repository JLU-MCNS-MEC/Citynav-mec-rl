#!/usr/bin/env python3
"""Build direct target-directed teacher labels for all development splits.

The teacher needs only the start pose (from ``trajectory[0]``) and the final
target.  Human trajectories are used for the start xy/yaw and for the offline
comparison figure - never for the action labels.

Outputs (per split):
  results/teacher_{split}.csv        one row per task
  results/teacher_{split}.npz        compact trajectories (states/actions)
  results/summary_{split}.json
  results/summary.json               combined + train_seen length groups
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from direct_teacher import ACTION_NAMES, CRUISE_TRIGGER_DEG, STOP_RADIUS, run_episode  # noqa: E402
from obstacles import load_ground, load_or_build  # noqa: E402

DATA = Path("/home/20261024_1/Workspace/DATA/sbfnav-data/original_citynav")
CACHE = HERE / "cache/obstacles"
RESULTS = HERE / "results"
SPLITS = ("train_seen", "val_seen", "val_unseen")
FIELDS = ["split", "index", "map", "success", "reason", "mode",
          "start_x", "start_y", "start_yaw", "target_x", "target_y",
          "start_target_m", "human_path_len_m", "path_len_m", "flown_m", "steps",
          "min_dist", "final_dist", "blocked_forward",
          "act_stop", "act_forward", "act_left", "act_right", "length_group"]

_OBSTACLES: dict[str, dict] = {}
_CRUISE_TRIGGER: float | None = CRUISE_TRIGGER_DEG


def _human_len(record: dict) -> float:
    xy = np.asarray(record["trajectory"], np.float64)[:, :2]
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _work(item: tuple[str, int, dict]) -> tuple[dict, dict]:
    split, index, record = item
    name = f"{record['area']}_block_{record['block']}"
    info = _OBSTACLES.get(name)
    if info is None:
        info = load_or_build(name, CACHE)
        _OBSTACLES[name] = info
    out = run_episode(record, info, cruise_trigger_deg=_CRUISE_TRIGGER)
    actions = out["actions"]
    counts = Counter(actions)
    states = np.asarray(out["states"], np.float64).reshape(-1, 3)
    flown = float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()) if len(states) > 1 else 0.0
    row = {
        "split": split, "index": index, "map": name,
        "success": out["success"], "reason": out["reason"], "mode": out["mode"],
        "start_x": round(out["start_xy"][0], 4), "start_y": round(out["start_xy"][1], 4),
        "start_yaw": round(out["start_yaw"], 6),
        "target_x": round(out["target_xy"][0], 4), "target_y": round(out["target_xy"][1], 4),
        "start_target_m": round(out["start_target_m"], 3),
        "human_path_len_m": round(_human_len(record), 3),
        "path_len_m": round(out["path_len_m"], 3),
        "flown_m": round(flown, 3),
        "steps": out["steps"], "min_dist": round(out["min_dist"], 3),
        "final_dist": round(out["final_dist"], 3),
        "blocked_forward": out["blocked_forward"],
        "act_stop": counts.get(0, 0), "act_forward": counts.get(1, 0),
        "act_left": counts.get(2, 0), "act_right": counts.get(3, 0),
        "length_group": "",
    }
    traj = {
        "states": np.asarray(out["states"], np.float32).reshape(-1, 3),
        "actions": np.asarray(actions, np.int8),
        "waypoints": np.asarray(out.get("waypoints", []), np.float32).reshape(-1, 2),
        "target": np.asarray(out["target_xy"], np.float32),
        "start": np.asarray(out["start_xy"], np.float32),
        "success": np.asarray(out["success"]),
    }
    return row, traj


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    a = np.asarray(values, np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "min": float(a.min()), "n": int(a.size)}


def _length_groups(steps: list[int]) -> dict:
    steps = np.asarray(steps, np.int64)
    q1, q2 = np.percentile(steps, [33.3333, 66.6667])
    return {"short_max": int(math.floor(q1)), "medium_max": int(math.floor(q2)),
            "n": int(steps.size),
            "counts": {"short": int((steps <= q1).sum()),
                       "medium": int(((steps > q1) & (steps <= q2)).sum()),
                       "long": int((steps > q2).sum())}}


def group_of(steps: int, groups: dict) -> str:
    if steps <= groups["short_max"]:
        return "short"
    if steps <= groups["medium_max"]:
        return "medium"
    return "long"


def run_split(split: str, workers: int, limit: int) -> dict:
    records = json.loads((DATA / f"citynav_{split}.json").read_text(encoding="utf-8"))
    items = list(enumerate(records))
    if limit:
        items = items[:limit]
    started = time.time()
    rows: list[dict] = []
    trajs: dict[str, np.ndarray] = {"keys": [], "states": [], "actions": [],
                                    "waypoints": [], "targets": [], "starts": []}
    with Pool(workers) as pool:
        for row, traj in pool.imap_unordered(_work, [(split, i, r) for i, r in items],
                                             chunksize=32):
            rows.append(row)
            trajs["keys"].append(f"{row['index']}")
            trajs["states"].append(traj["states"])
            trajs["actions"].append(traj["actions"])
            trajs["waypoints"].append(traj["waypoints"])
            trajs["targets"].append(traj["target"])
            trajs["starts"].append(traj["start"])
    rows.sort(key=lambda r: r["index"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / f"teacher_{split}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    # trajectories are stored as an object-free npz: concatenation offsets
    lengths = np.asarray([len(a) for a in trajs["actions"]], np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    np.savez_compressed(
        RESULTS / f"teacher_{split}.npz",
        keys=np.asarray(trajs["keys"]),
        offsets=offsets,
        states=np.concatenate([s.reshape(-1, 3) for s in trajs["states"]]) if rows else np.zeros((0, 3), np.float32),
        actions=np.concatenate(trajs["actions"]) if rows else np.zeros(0, np.int8),
        waypoint_offsets=np.concatenate([[0], np.cumsum([len(w) for w in trajs["waypoints"]])]),
        waypoints=np.concatenate([w.reshape(-1, 2) for w in trajs["waypoints"]]) if rows else np.zeros((0, 2), np.float32),
        targets=np.stack(trajs["targets"]) if rows else np.zeros((0, 2), np.float32),
        starts=np.stack(trajs["starts"]) if rows else np.zeros((0, 2), np.float32),
    )
    ok = [r for r in rows if r["success"]]
    summary = {
        "split": split,
        "records": len(rows),
        "success": len(ok),
        "success_rate": len(ok) / max(len(rows), 1),
        "reason_counts": dict(Counter(r["reason"] for r in rows)),
        "mode_counts": dict(Counter(r["mode"] for r in rows)),
        "steps_success": _percentiles([r["steps"] for r in ok]),
        "start_target_m": _percentiles([r["start_target_m"] for r in rows]),
        "path_len_m": _percentiles([r["path_len_m"] for r in ok]),
        "min_dist": _percentiles([r["min_dist"] for r in ok]),
        "by_map": {},
        "actions_success": dict(sum((Counter({k: r[f"act_{k}"] for k in
                                              ("stop", "forward", "left", "right")})
                                     for r in ok), Counter())),
        "seconds": round(time.time() - started, 1),
    }
    per_map: dict[str, dict] = {}
    for row in rows:
        m = per_map.setdefault(row["map"], {"records": 0, "success": 0, "steps": []})
        m["records"] += 1
        if row["success"]:
            m["success"] += 1
            m["steps"].append(row["steps"])
    for name, m in per_map.items():
        m["success_rate"] = m["success"] / m["records"]
        m["steps"] = _percentiles(m["steps"]) if m["steps"] else {}
    summary["by_map"] = per_map
    return {"summary": summary, "rows": rows}


def main() -> int:
    global RESULTS, _CRUISE_TRIGGER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", nargs="+", default=list(SPLITS))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default=str(RESULTS), help="output directory")
    parser.add_argument("--cruise-trigger", default=str(CRUISE_TRIGGER_DEG),
                        help="deadband re-trigger in degrees, or 'naive' for the plain 15 deg rule")
    args = parser.parse_args()
    RESULTS = Path(args.out_dir)
    _CRUISE_TRIGGER = None if args.cruise_trigger == "naive" else float(args.cruise_trigger)
    maps = sorted({f"{r['area']}_block_{r['block']}"
                   for r in json.loads((DATA / "citynav_val_seen.json").read_text(encoding="utf-8"))
                   } | {f"{r['area']}_block_{r['block']}"
                        for r in json.loads((DATA / "citynav_val_unseen.json").read_text(encoding="utf-8"))
                        } | {f"{r['area']}_block_{r['block']}"
                             for r in json.loads((DATA / "citynav_train_seen.json").read_text(encoding="utf-8"))})
    # Build/load obstacle grids once in the parent; fork shares them with workers.
    ground = load_ground()
    started = time.time()
    for name in maps:
        _OBSTACLES[name] = load_or_build(name, CACHE, ground)
    print(f"obstacle grids: {len(_OBSTACLES)} maps in {time.time() - started:.1f}s", flush=True)
    combined = {"teacher_config": {"cruise_trigger_deg": _CRUISE_TRIGGER}}
    train_steps: list[int] = []
    for split in args.split:
        print(f"=== {split} ===", flush=True)
        result = run_split(split, args.workers, args.limit)
        summary = result["summary"]
        combined[split] = summary
        (RESULTS / f"summary_{split}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({k: summary[k] for k in
                          ("records", "success", "success_rate", "reason_counts",
                           "mode_counts", "steps_success", "seconds")},
                         indent=2, ensure_ascii=False), flush=True)
        if split == "train_seen":
            train_steps = [r["steps"] for r in result["rows"] if r["success"]]
    if train_steps:
        groups = _length_groups(train_steps)
        # length groups always come from the Train-Seen distribution, and are
        # applied to every split so Stage 06 evaluates matched task sets
        for split in args.split:
            path = RESULTS / f"teacher_{split}.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            for row in rows:
                row["length_group"] = group_of(int(row["steps"]), groups) if row["success"] == "True" else ""
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        combined["length_groups"] = groups
        print("length groups:", json.dumps(groups, ensure_ascii=False), flush=True)
    combined["obstacle_scan"] = {name: {"blocked_ratio": i["blocked_ratio"],
                                        "max_above_ground_m": i["max_above_ground_m"],
                                        "ceiling_m": i["ceiling_m"]}
                                 for name, i in _OBSTACLES.items()}
    (RESULTS / "summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
