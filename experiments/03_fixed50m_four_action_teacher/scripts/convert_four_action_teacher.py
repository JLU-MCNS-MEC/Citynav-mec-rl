#!/usr/bin/env python3
"""Convert all Train-Seen trajectories to fixed-50 m four-action teacher labels.

Writes resumable per-chunk checkpoints, then merges them into a single npz plus
a per-episode CSV, summary JSON and contract checks.

    python convert_four_action_teacher.py            # full run (resumable)
    python convert_four_action_teacher.py --limit 50 # smoke
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_action_teacher import ACTION_NAMES, STOP, convert_episode, move, wrap

DATA = Path("/home/20261024_1/Workspace/DATA")
SPLIT_JSON = DATA / "sbfnav-data/original_citynav/citynav_train_seen.json"
GROUND_SRC = Path("/tmp/citynav-official/gsamllavanav/mapdata.py")
HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
CHUNK = 2000


def load_ground() -> dict[str, float]:
    if not GROUND_SRC.exists():
        return {}
    import importlib.util

    spec = importlib.util.spec_from_file_location("official_mapdata", GROUND_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: float(v) for k, v in module.GROUND_LEVEL.items()}


def chunk_path(out_dir: Path, index: int) -> Path:
    return out_dir / "chunks" / f"chunk_{index:05d}.npz"


def pack(episodes: list[dict], ground: dict[str, float], limit: int) -> dict:
    keys, maps, reasons = [], [], []
    start_xy, start_yaw, target_xy, marker_xy = [], [], [], []
    success, steps, path_len = [], [], []
    start_d, final_d, min_d = [], [], []
    entered, removed, teacher_final, teacher_min = [], [], [], []
    fixed_z = []
    states, state_offsets = [], [0]
    actions = []
    waypoints, way_offsets = [], [0]
    for idx, record in enumerate(episodes):
        if limit and idx >= limit:
            break
        name = f"{record['area']}_block_{record['block']}"
        result = convert_episode(record, ground.get(name))
        keys.append(f"{idx}|{name}")
        maps.append(name)
        reasons.append(result.get("reason", "degenerate"))
        success.append(bool(result.get("success", False)))
        steps.append(int(result.get("teacher_steps", 0)))
        path_len.append(float(result.get("path_len_m", 0.0)))
        start_d.append(float(result.get("start_target_m", 0.0)))
        final_d.append(float(result.get("final_target_m", 0.0)))
        min_d.append(float(result.get("min_target_m", 0.0)))
        entered.append(bool(result.get("entered_20m", False)))
        removed.append(int(result.get("duplicates_removed", 0)))
        teacher_final.append(float(result.get("teacher_final_dist", 0.0)))
        teacher_min.append(float(result.get("teacher_min_dist", 0.0)))
        fixed_z.append(float(result.get("fixed_z_m") or 0.0))
        start_xy.append(np.asarray(record["trajectory"], dtype=np.float64)[0, :2].astype(np.float32).tolist())
        start_yaw.append(float(result.get("start_yaw", 0.0)))
        target_xy.append(result.get("target_xy", [0.0, 0.0]))
        marker_xy.append(result.get("marker_xy", [0.0, 0.0]))
        st = result.get("states", [])
        if st:
            states.append(np.asarray(st, dtype=np.float32))
            state_offsets.append(state_offsets[-1] + len(st))
        else:
            state_offsets.append(state_offsets[-1])
        actions.append(np.asarray(result.get("actions", []), dtype=np.int8))
        wp = result.get("waypoints", np.zeros((0, 2)))
        if len(wp):
            waypoints.append(np.asarray(wp, dtype=np.float32))
            way_offsets.append(way_offsets[-1] + len(wp))
        else:
            way_offsets.append(way_offsets[-1])
    return {
        "keys": np.asarray(keys, dtype="U"),
        "maps": np.asarray(maps, dtype="U"),
        "reasons": np.asarray(reasons, dtype="U"),
        "success": np.asarray(success, dtype=bool),
        "steps": np.asarray(steps, dtype=np.int32),
        "path_len_m": np.asarray(path_len, dtype=np.float32),
        "start_target_m": np.asarray(start_d, dtype=np.float32),
        "final_target_m": np.asarray(final_d, dtype=np.float32),
        "min_target_m": np.asarray(min_d, dtype=np.float32),
        "entered_20m": np.asarray(entered, dtype=bool),
        "duplicates_removed": np.asarray(removed, dtype=np.int32),
        "teacher_final_m": np.asarray(teacher_final, dtype=np.float32),
        "teacher_min_m": np.asarray(teacher_min, dtype=np.float32),
        "fixed_z_m": np.asarray(fixed_z, dtype=np.float32),
        "start_xy": np.asarray(start_xy, dtype=np.float32).reshape(-1, 2),
        "start_yaw": np.asarray(start_yaw, dtype=np.float32),
        "target_xy": np.asarray(target_xy, dtype=np.float32).reshape(-1, 2),
        "marker_xy": np.asarray(marker_xy, dtype=np.float32).reshape(-1, 2),
        "states": np.concatenate(states) if states else np.zeros((0, 3), np.float32),
        "state_offsets": np.asarray(state_offsets, dtype=np.int64),
        "actions": np.concatenate(actions) if actions else np.zeros((0,), np.int8),
        "waypoints": np.concatenate(waypoints) if waypoints else np.zeros((0, 2), np.float32),
        "way_offsets": np.asarray(way_offsets, dtype=np.int64),
    }


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading train_seen JSON …", flush=True)
    records = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    total = min(len(records), args.limit) if args.limit else len(records)
    print(f"episodes: {total}", flush=True)
    ground = load_ground()

    start = 0
    if args.resume:
        existing = sorted((out_dir / "chunks").glob("chunk_*.npz"))
        if existing:
            last = existing[-1]
            last_idx = int(last.stem.split("_")[1])
            start = last_idx + CHUNK
            print(f"resume after chunk {last.name} (start={start})", flush=True)

    t0 = dt.datetime.now()
    for base in range(start, total, CHUNK):
        path = chunk_path(out_dir, base)
        if path.exists():
            continue
        end = min(total, base + CHUNK)
        payload = pack(records[base:end], ground, 0)
        atomic_savez(path, **payload)
        done = end
        elapsed = (dt.datetime.now() - t0).total_seconds()
        print(f"  {done}/{total}  chunk={path.name}  {elapsed:.0f}s", flush=True)

    # merge chunks
    files = sorted((out_dir / "chunks").glob("chunk_*.npz"))
    merged: dict[str, list[np.ndarray]] = {}
    for path in files:
        with np.load(path) as z:
            for key in z.files:
                merged.setdefault(key, []).append(z[key])
    labels = {
        key: (np.concatenate(parts) if key in ("keys", "maps", "reasons", "success", "steps",
                                               "path_len_m", "start_target_m", "final_target_m",
                                               "min_target_m", "entered_20m", "duplicates_removed",
                                               "teacher_final_m", "teacher_min_m", "fixed_z_m",
                                               "start_xy", "start_yaw", "target_xy", "marker_xy",
                                               "states", "actions", "waypoints")
              else parts[0])
        for key, parts in merged.items()
    }
    # offsets need shifting per chunk
    state_offsets = [np.asarray([0], np.int64)]
    way_offsets = [np.asarray([0], np.int64)]
    for path in files:
        with np.load(path) as z:
            so = z["state_offsets"]
            wo = z["way_offsets"]
            state_offsets.append(so[1:] + state_offsets[-1][-1])
            way_offsets.append(wo[1:] + way_offsets[-1][-1])
    labels["state_offsets"] = np.concatenate(state_offsets)
    labels["way_offsets"] = np.concatenate(way_offsets)

    atomic_savez(out_dir / "four_action_labels.npz", **labels)
    write_sample(labels, out_dir / "sample_labels.npz", n=200)
    write_summaries(labels, total, args.out_dir)
    print("merged labels written", flush=True)
    return 0


def write_sample(labels: dict, path: Path, n: int = 200) -> None:
    """Small stratified sample committed to git for reproducibility."""
    n = min(n, len(labels["keys"]))
    state_off = labels["state_offsets"][: n + 1] - labels["state_offsets"][0]
    way_off = labels["way_offsets"][: n + 1] - labels["way_offsets"][0]
    sample = {
        "keys": labels["keys"][:n], "maps": labels["maps"][:n],
        "reasons": labels["reasons"][:n], "success": labels["success"][:n],
        "steps": labels["steps"][:n], "path_len_m": labels["path_len_m"][:n],
        "entered_20m": labels["entered_20m"][:n],
        "start_xy": labels["start_xy"][:n], "start_yaw": labels["start_yaw"][:n],
        "target_xy": labels["target_xy"][:n], "marker_xy": labels["marker_xy"][:n],
        "fixed_z_m": labels["fixed_z_m"][:n],
        "states": labels["states"][: int(state_off[-1])],
        "state_offsets": state_off,
        "actions": labels["actions"][: int(state_off[-1])],
        "waypoints": labels["waypoints"][: int(way_off[-1])],
        "way_offsets": way_off,
    }
    atomic_savez(path, **sample)


def _episode_actions(labels: dict, i: int) -> np.ndarray:
    return labels["actions"][labels["state_offsets"][i]:labels["state_offsets"][i + 1]]


def write_summaries(labels: dict, n: int, out_dir: Path) -> None:
    keys = labels["keys"]
    maps = labels["maps"]
    success = labels["success"]
    steps = labels["steps"]
    reasons = labels["reasons"]
    entered = labels["entered_20m"]
    counts = Counter(ACTION_NAMES[int(a)] for a in labels["actions"])
    ok_steps = steps[success]
    summary = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": "train_seen",
        "test_unseen_read": False,
        "n_episodes": int(len(keys)),
        "replay_success": int(success.sum()),
        "replay_success_rate": float(success.mean()) if len(success) else 0.0,
        "entered_20m": int(entered.sum()),
        "entered_20m_rate": float(entered.mean()) if len(entered) else 0.0,
        "action_totals": dict(counts),
        "failure_reasons": dict(Counter(reasons[~success])),
        "teacher_steps": {
            "mean": float(ok_steps.mean()) if len(ok_steps) else 0.0,
            "median": float(np.median(ok_steps)) if len(ok_steps) else 0.0,
            "p95": float(np.percentile(ok_steps, 95)) if len(ok_steps) else 0.0,
            "p99": float(np.percentile(ok_steps, 99)) if len(ok_steps) else 0.0,
            "max": int(ok_steps.max()) if len(ok_steps) else 0,
        },
        "by_map": {},
        "by_length_bin": {},
        "fixed_z_range": [float(labels["fixed_z_m"].min()), float(labels["fixed_z_m"].max())],
    }
    for name in sorted(set(maps.tolist())):
        mask = maps == name
        summary["by_map"][name] = {
            "n": int(mask.sum()),
            "success": int(success[mask].sum()),
            "rate": float(success[mask].mean()) if mask.sum() else 0.0,
        }
    ok_steps_only = np.sort(steps[success])
    terc = np.percentile(ok_steps_only, [33.33, 66.67]) if len(ok_steps_only) else [0, 0]
    for tag, mask in (
        ("short", success & (steps <= terc[0])),
        ("medium", success & (steps > terc[0]) & (steps <= terc[1])),
        ("long", success & (steps > terc[1])),
    ):
        summary["by_length_bin"][tag] = {
            "n": int(mask.sum()),
            "success": int(success[mask].sum()),
            "rate": float(success[mask].mean()) if mask.sum() else 0.0,
        }
    summary["by_length_bin"]["failed"] = {
        "n": int((~success).sum()), "success": 0, "rate": 0.0,
    }
    summary["teacher_step_terciles"] = [float(x) for x in terc]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fields = ["key", "map", "success", "reason", "steps", "path_len_m", "start_target_m",
              "final_target_m", "min_target_m", "entered_20m", "duplicates_removed",
              "teacher_final_m", "teacher_min_m", "fixed_z_m"]
    with (out_dir / "teacher_episodes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in range(len(keys)):
            writer.writerow({
                "key": keys[i], "map": maps[i], "success": bool(success[i]),
                "reason": reasons[i], "steps": int(steps[i]),
                "path_len_m": float(labels["path_len_m"][i]),
                "start_target_m": float(labels["start_target_m"][i]),
                "final_target_m": float(labels["final_target_m"][i]),
                "min_target_m": float(labels["min_target_m"][i]),
                "entered_20m": bool(entered[i]),
                "duplicates_removed": int(labels["duplicates_removed"][i]),
                "teacher_final_m": float(labels["teacher_final_m"][i]),
                "teacher_min_m": float(labels["teacher_min_m"][i]),
                "fixed_z_m": float(labels["fixed_z_m"][i]),
            })


if __name__ == "__main__":
    raise SystemExit(main())
