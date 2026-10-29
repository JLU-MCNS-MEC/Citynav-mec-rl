#!/usr/bin/env python3
"""Stage 02: audit the Train-Seen human trajectories at fixed 50 m.

For every Train-Seen episode this script

* projects the trajectory to the fixed height ``GROUND_LEVEL + 50 m`` (x/y kept),
* removes repeated horizontal points,
* computes path length, efficiency, detour, backtracking, turning, jumps,
  out-of-bounds fraction, duplicate fraction and the distance to the final
  target (``target_positions[-1]``),
* replays the cleaned path with the official lookahead teacher restricted to
  four actions (STOP / Forward 5 m / TurnLeft 30 / TurnRight 30) and an
  official-consistent 20 m STOP radius,
* writes a per-episode metrics journal (resumable), a CSV, cleaned trajectories
  and quality groups.

Val-Seen / Val-Unseen are untouched and Test-Unseen is never read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

DATA = Path("/home/20261024_1/Workspace/DATA")
SPLIT_JSON = DATA / "sbfnav-data/original_citynav/citynav_train_seen.json"
RGBD = DATA / "rgbd"
GROUND_SRC = Path("/tmp/citynav-official/gsamllavanav/mapdata.py")

FORWARD_M = 5.0
TURN_RAD = math.radians(30.0)
TURN_THRESHOLD = math.radians(30.0)
STOP_RADIUS = 20.0
DUP_EPS = 0.05
STOP, FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3
ACTION_NAMES = {STOP: "Stop", FORWARD: "Forward", TURN_LEFT: "TurnLeft", TURN_RIGHT: "TurnRight"}


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_ground() -> dict[str, float]:
    if not GROUND_SRC.exists():
        return {}
    import importlib.util

    spec = importlib.util.spec_from_file_location("official_mapdata", GROUND_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: float(v) for k, v in module.GROUND_LEVEL.items()}


class RasterIndex:
    """Lazy raster metadata and a decimated height-validity mask per map."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, dict] = {}

    def get(self, name: str) -> dict:
        if name in self._cache:
            return self._cache[name]
        import rasterio

        with rasterio.open(self.root / f"{name}.tif") as ds:
            bounds = ds.bounds
            height = ds.read(1, out_shape=(ds.height // 10, ds.width // 10))
        valid = height > -1000  # nodata is -9999; real elevations are > 0
        info = {
            "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
            "valid": valid,
            "shape": valid.shape,
        }
        self._cache[name] = info
        return info

    def coverage(self, name: str, xy: np.ndarray) -> tuple[float, float]:
        """Return (outside-raster bounds ratio, nodata ratio among inside points)."""
        info = self.get(name)
        left, bottom, right, top = info["bounds"]
        inside = (xy[:, 0] >= left) & (xy[:, 0] <= right) & (xy[:, 1] >= bottom) & (xy[:, 1] <= top)
        outside = float(1.0 - inside.mean())
        if not inside.any():
            return outside, 1.0
        h, w = info["shape"]
        cols = np.clip(((xy[:, 0] - left) / (right - left) * w).astype(int), 0, w - 1)
        rows = np.clip(((top - xy[:, 1]) / (top - bottom) * h).astype(int), 0, h - 1)
        valid = info["valid"][rows[inside], cols[inside]]
        return outside, float(1.0 - valid.mean())


def clean_path(xy: np.ndarray) -> tuple[np.ndarray, int]:
    keep = [0]
    for i in range(1, len(xy)):
        if np.hypot(*(xy[i] - xy[keep[-1]])) >= DUP_EPS:
            keep.append(i)
    return xy[keep], len(xy) - len(keep)


def build_waypoints(xy: np.ndarray, marker: np.ndarray, stride: float = 5.0) -> np.ndarray:
    """Official ``interpolated_trajectory``: keep points >stride apart along the path.

    The human polyline is walked from the start to the marker; a waypoint is
    appended when it is more than ``stride`` away from the previous waypoint,
    mirroring ``mturk_trajectory.interpolated_trajectory``.
    """
    chain = np.vstack([xy, marker[None, :]])
    way = [chain[0]]
    for src, dst in zip(chain[:-1], chain[1:]):
        delta = dst - src
        dist = float(np.linalg.norm(delta))
        if dist < 1e-9:
            continue
        n = int(dist / stride)
        candidates = [src + i * delta / dist * stride for i in range(n)] + [dst]
        for point in candidates:
            if float(np.linalg.norm(point - way[-1])) > stride:
                way.append(point)
    return np.asarray(way)


def replay(path: np.ndarray, target: np.ndarray, start_yaw: float) -> dict:
    """Lookahead teacher restricted to four actions, 20 m STOP radius."""
    n = len(path)
    pos = path[0].astype(np.float64).copy()
    yaw = float(start_yaw)
    ptr = 0
    steps = 0
    min_dist = float(np.linalg.norm(pos - target))
    actions: list[int] = []
    path_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    max_steps = int(min(2000, max(60, 1.6 * path_len / FORWARD_M + 100)))
    while steps < max_steps:
        dist = float(np.linalg.norm(pos - target))
        min_dist = min(min_dist, dist)
        if dist <= STOP_RADIUS:
            actions.append(STOP)
            return {
                "replay_success": True,
                "replay_steps": steps,
                "replay_min_dist": min_dist,
                "replay_final_dist": dist,
                "actions": actions,
            }
        # official teacher: globally closest waypoint, then look ahead by one
        j = int(np.argmin(np.linalg.norm(path - pos, axis=1)))
        ptr = j
        nxt = min(n - 1, j + 1)
        vec = path[nxt] - pos
        dyaw = wrap(math.atan2(vec[1], vec[0]) - yaw) if np.hypot(*vec) > 1e-6 else 0.0
        if dyaw > TURN_THRESHOLD:
            yaw = wrap(yaw + TURN_RAD)
            actions.append(TURN_LEFT)
        elif dyaw < -TURN_THRESHOLD:
            yaw = wrap(yaw - TURN_RAD)
            actions.append(TURN_RIGHT)
        else:
            pos = pos + FORWARD_M * np.array([math.cos(yaw), math.sin(yaw)])
            actions.append(FORWARD)
        steps += 1
    return {
        "replay_success": False,
        "replay_steps": steps,
        "replay_min_dist": min_dist,
        "replay_final_dist": float(np.linalg.norm(pos - target)),
        "actions": actions,
    }


def analyze(record: dict, idx: int, ground: dict[str, float], rasters: RasterIndex) -> dict:
    name = f"{record['area']}_block_{record['block']}"
    traj = np.asarray(record["trajectory"], dtype=np.float64)
    xy_raw = traj[:, :2]
    target = np.asarray(record["target_positions"][-1], dtype=np.float64)[:2]
    xy, removed = clean_path(xy_raw)
    if len(xy) < 2:
        return {
            "key": f"{idx}|{name}",
            "index": idx,
            "map": name,
            "object_ids": record["object_ids"],
            "ann_ids": record["ann_ids"],
            "points_raw": int(len(xy_raw)),
            "points_clean": int(len(xy)),
            "degenerate": True,
        }
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_len = float(seg.sum())
    start_dist = float(np.linalg.norm(xy[0] - target))
    final_dist = float(np.linalg.norm(xy[-1] - target))
    dists = np.linalg.norm(xy - target, axis=1)
    min_dist = float(dists.min())
    entered = bool(min_dist <= STOP_RADIUS)
    first_entry = int(np.argmax(dists <= STOP_RADIUS)) if entered else None

    direction = target - xy[0]
    norm = float(np.linalg.norm(direction))
    unit = direction / norm if norm > 1e-9 else np.zeros(2)
    progress = (xy - xy[0]) @ unit
    backward = float(np.maximum(0.0, -np.diff(progress)).sum())
    backtrack_ratio = backward / path_len if path_len > 0 else 0.0

    headings = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    dyaw_signed = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    dyaw = np.abs(dyaw_signed)
    turn_total_deg = float(np.degrees(dyaw.sum()))
    big_turns = int((dyaw > TURN_THRESHOLD).sum())
    signs = np.where(np.abs(dyaw_signed) > math.radians(15.0), np.sign(dyaw_signed), 0)
    signs = signs[signs != 0]
    reversals = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0

    start_yaw = math.atan2(traj[0, 4], traj[0, 3]) if (abs(traj[0, 3]) + abs(traj[0, 4])) > 1e-9 else float(headings[0])
    marker = np.asarray(record["marker_positions"][-1], dtype=np.float64)[:2]
    waypoints = build_waypoints(xy, marker)
    rep = replay(waypoints, target, start_yaw)
    actions = rep.pop("actions")
    counts = {ACTION_NAMES[a]: actions.count(a) for a in ACTION_NAMES}
    replay_steps = rep["replay_steps"]

    efficiency = start_dist / path_len if path_len > 0 else 0.0
    outside_ratio, nodata_ratio = rasters.coverage(name, xy)
    return {
        "key": f"{idx}|{name}",
        "index": idx,
        "map": name,
        "object_ids": record["object_ids"],
        "ann_ids": record["ann_ids"],
        "points_raw": int(len(xy_raw)),
        "points_clean": int(len(xy)),
        "duplicates_removed": int(removed),
        "duplicate_ratio": removed / max(len(xy_raw), 1),
        "path_len_m": path_len,
        "start_target_m": start_dist,
        "final_target_m": final_dist,
        "min_target_m": min_dist,
        "entered_20m": entered,
        "first_entry_point": first_entry,
        "efficiency": efficiency,
        "detour_ratio": path_len / start_dist if start_dist > 1e-9 else float("inf"),
        "backtrack_m": backward,
        "backtrack_ratio": backtrack_ratio,
        "turn_total_deg": turn_total_deg,
        "big_turns": big_turns,
        "reversals": reversals,
        "max_jump_m": float(seg.max()) if len(seg) else 0.0,
        "outside_ratio": outside_ratio,
        "nodata_ratio": nodata_ratio,
        "replay_success": rep["replay_success"],
        "replay_steps": replay_steps,
        "replay_min_dist": rep["replay_min_dist"],
        "replay_final_dist": rep["replay_final_dist"],
        "action_counts": counts,
        "teacher_actions": len(actions),
        "waypoints": int(len(waypoints)),
        "ground_level_m": ground.get(name),
        "fixed_z_m": (ground.get(name) + 50.0) if name in ground else None,
        "degenerate": False,
    }


def classify(rows: list[dict], thresholds: dict) -> None:
    for row in rows:
        if row.get("degenerate"):
            row["quality"] = "low"
            row["reasons"] = ["degenerate"]
            continue
        reasons = []
        if not row["entered_20m"]:
            reasons.append("never_entered_20m")
        if not row["replay_success"]:
            reasons.append("replay_failed")
        if row["outside_ratio"] > 0.01:
            reasons.append("out_of_bounds")
        if row["max_jump_m"] > thresholds["jump_cut"]:
            reasons.append("abnormal_jump")
        if row["path_len_m"] < 20.0:
            reasons.append("too_short")
        if reasons:
            row["quality"] = "low"
            row["reasons"] = reasons
            continue
        soft = []
        if row["efficiency"] < thresholds["eff_p25"]:
            soft.append("low_efficiency")
        if row["backtrack_ratio"] > thresholds["backtrack_p90"]:
            soft.append("high_backtrack")
        if row["turn_total_deg"] > thresholds["turn_p95"]:
            soft.append("excessive_turning")
        if row["nodata_ratio"] > 0.5:
            soft.append("heavy_nodata")
        row["quality"] = "medium" if soft else "high"
        row["reasons"] = soft
        row["score"] = round(
            100.0
            * (
                0.35 * min(row["efficiency"], 1.0)
                + 0.25 * float(row["replay_success"])
                + 0.15 * (1.0 - min(row["backtrack_ratio"], 1.0))
                + 0.15 * (1.0 - min(row["duplicate_ratio"], 1.0))
                + 0.10 * (1.0 - min(row["turn_total_deg"] / 1440.0, 1.0))
            ),
            2,
        )


def write_outputs(rows: list[dict], out_dir: Path, records: list[dict], ground: dict) -> dict:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "index", "map", "object_ids", "ann_ids", "points_raw", "points_clean",
        "duplicates_removed", "duplicate_ratio", "path_len_m", "start_target_m",
        "final_target_m", "min_target_m", "entered_20m", "first_entry_point",
        "efficiency", "detour_ratio", "backtrack_m", "backtrack_ratio",
        "turn_total_deg", "big_turns", "reversals", "max_jump_m", "outside_ratio", "nodata_ratio",
        "replay_success", "replay_steps", "replay_min_dist", "replay_final_dist",
        "teacher_actions", "ground_level_m", "fixed_z_m", "quality", "score", "reasons",
    ]
    with (out_dir / "train_quality.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["object_ids"] = ";".join(map(str, row.get("object_ids", [])))
            row["ann_ids"] = ";".join(map(str, row.get("ann_ids", [])))
            row["reasons"] = ";".join(row.get("reasons", []))
            writer.writerow(row)

    # cleaned horizontal trajectories for every episode
    pts, offsets, keys, maps = [], [0], [], []
    for row in rows:
        if row.get("degenerate"):
            keys.append(row["key"]); maps.append(row["map"]); offsets.append(offsets[-1]); continue
        # re-read from the journal? store points in memory via _points
        xy = _POINTS.get(row["key"])
        if xy is None:
            continue
        pts.append(xy.astype(np.float32))
        offsets.append(offsets[-1] + len(xy))
        keys.append(row["key"]); maps.append(row["map"])
    np.savez_compressed(
        out_dir / "cleaned_trajectories.npz",
        points=np.concatenate(pts) if pts else np.zeros((0, 2), np.float32),
        offsets=np.asarray(offsets, np.int64),
        keys=np.asarray(keys, dtype="U"),
        maps=np.asarray(maps, dtype="U"),
    )

    counts = {q: sum(1 for r in rows if r.get("quality") == q) for q in ("high", "medium", "low")}
    action_totals: dict[str, int] = {name: 0 for name in ACTION_NAMES.values()}
    for row in rows:
        for name, value in row.get("action_counts", {}).items():
            action_totals[name] += value
    by_map: dict[str, dict[str, int]] = {}
    for row in rows:
        by_map.setdefault(row["map"], {"n": 0, "high": 0, "medium": 0, "low": 0})
        by_map[row["map"]]["n"] += 1
        by_map[row["map"]][row.get("quality", "low")] += 1
    steps = np.asarray([r.get("teacher_actions", 0) for r in rows if not r.get("degenerate")])
    terciles = np.percentile(steps, [33.33, 66.67]).tolist() if len(steps) else [0, 0]
    bins = {"short": [r for r in rows if not r.get("degenerate") and r["teacher_actions"] <= terciles[0]],
            "medium": [r for r in rows if not r.get("degenerate") and terciles[0] < r["teacher_actions"] <= terciles[1]],
            "long": [r for r in rows if not r.get("degenerate") and r["teacher_actions"] > terciles[1]]}
    quality_by_bin = {
        b: {q: sum(1 for r in sub if r.get("quality") == q) for q in ("high", "medium", "low")}
        for b, sub in bins.items()
    }
    summary = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": "train_seen",
        "test_unseen_read": False,
        "n_episodes": len(rows),
        "quality_counts": counts,
        "quality_fraction": {k: v / max(len(rows), 1) for k, v in counts.items()},
        "action_totals": action_totals,
        "teacher_steps_terciles": terciles,
        "quality_by_length_bin": quality_by_bin,
        "by_map": by_map,
        "replay_success_rate": float(np.mean([r.get("replay_success", False) for r in rows])),
        "entered_20m_rate": float(np.mean([r.get("entered_20m", False) for r in rows])),
        "efficiency_mean": float(np.mean([r["efficiency"] for r in rows if not r.get("degenerate")])),
        "path_len_mean": float(np.mean([r["path_len_m"] for r in rows if not r.get("degenerate")])),
        "ground_level_known": sum(1 for r in rows if r.get("ground_level_m") is not None),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    high = [r["index"] for r in rows if r.get("quality") == "high"]
    medium = [r["index"] for r in rows if r.get("quality") == "medium"]
    mixed = high + medium
    (out_dir / "selection_high.json").write_text(json.dumps({"split": "train_seen", "indices": high}, indent=2), encoding="utf-8")
    (out_dir / "selection_high_medium_mixed.json").write_text(json.dumps({"split": "train_seen", "indices": mixed}, indent=2), encoding="utf-8")
    return summary


_POINTS: dict[str, np.ndarray] = {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, default=SPLIT_JSON)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N episodes")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    journal = args.out_dir / "train_metrics.jsonl"

    done: dict[str, dict] = {}
    if args.resume and journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
        print(f"resume: {len(done)} episodes already done", flush=True)

    print("loading train_seen JSON …", flush=True)
    records = json.loads(args.split_json.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]
    print(f"episodes: {len(records)}", flush=True)

    # cleaned polylines are kept in memory for the npz export (also on resume)
    for idx, record in enumerate(records):
        name = f"{record['area']}_block_{record['block']}"
        raw = np.asarray(record["trajectory"], dtype=np.float64)[:, :2]
        cleaned, _ = clean_path(raw) if len(raw) > 1 else (raw, 0)
        _POINTS[f"{idx}|{name}"] = cleaned

    ground = load_ground()
    rasters = RasterIndex(RGBD)
    t0 = dt.datetime.now()
    with journal.open("a", encoding="utf-8") as fh:
        for idx, record in enumerate(records):
            name = f"{record['area']}_block_{record['block']}"
            key = f"{idx}|{name}"
            if key in done:
                continue
            row = analyze(record, idx, ground, rasters)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if (idx + 1) % 2000 == 0:
                elapsed = (dt.datetime.now() - t0).total_seconds()
                print(f"  {idx+1}/{len(records)}  {elapsed:.0f}s", flush=True)

    rows = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda r: r["index"])

    finite = [r for r in rows if not r.get("degenerate")]
    eff = np.asarray([r["efficiency"] for r in finite])
    bt = np.asarray([r["backtrack_ratio"] for r in finite])
    turn = np.asarray([r["turn_total_deg"] for r in finite])
    jump = np.asarray([r["max_jump_m"] for r in finite])
    thresholds = {
        "eff_p25": float(np.percentile(eff, 25)),
        "backtrack_p90": float(np.percentile(bt, 90)),
        "turn_p95": float(np.percentile(turn, 95)),
        "jump_p99": float(np.percentile(jump, 99)),
    }
    thresholds["jump_cut"] = max(10.0, thresholds["jump_p99"])
    classify(rows, thresholds)

    summary = write_outputs(rows, args.out_dir, records, ground)
    summary["thresholds"] = thresholds
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_episodes", "quality_counts", "quality_fraction",
                                              "replay_success_rate", "entered_20m_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
