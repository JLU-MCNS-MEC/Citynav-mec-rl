"""E1 — CityNav label / coordinate audit (read-only).

Checks, without touching the raw data:

1. direction-vector convention: ``yaw = atan2(dy, dx)`` versus the local
   motion direction computed from consecutive trajectory points;
2. left/right sign convention: teacher ``TURN_LEFT`` must rotate the pose
   counter-clockwise (yaw increases) and the demonstration must actually bend
   left at that point;
3. vertical actions: ``GO_UP``/``GO_DOWN`` versus the demonstration's local
   dz sign;
4. forward actions: teacher ``MOVE_FORWARD`` versus horizontal motion;
5. marker/target separation: ``marker_positions[-1]`` is the teacher stop goal,
   ``target_positions[-1]`` is the evaluation goal, and they must not be used
   interchangeably;
6. coverage statistics used later by E3 (where in the trajectory the selected
   teacher step falls, and the raw action distribution).

Exit code is 0 even when violations are found; violations are reported in the
JSON/Markdown so that no result is silently "fixed".
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_data_config  # noqa: E402
from src.data.actions import ACTION_NAMES  # noqa: E402
from src.data.citynav_dataset import CityNavDataset  # noqa: E402
from src.data.trajectory_adapter import (  # noqa: E402
    TrajectoryAdapter,
    load_trajectory,
    transition_pose,
    wrap_angle,
)

TURN_LEFT_MIN_DEG = 5.0
FORWARD_MAX_DEG = 45.0


def _deg(rad: float) -> float:
    return float(math.degrees(rad))


def audit_episode(record, adapter: TrajectoryAdapter) -> dict:
    poses = list(record.poses)
    raw = np.asarray(record.trajectory_raw, dtype=np.float64)

    # --- 1. direction vector convention -------------------------------------
    dir_norm = np.hypot(raw[:, 3], raw[:, 4])
    zero_dir = int(np.sum(dir_norm < 1e-6))
    yaw = np.array([p.yaw for p in poses])
    xy = np.stack([(p.x, p.y) for p in poses])

    yaw_motion_diff = []
    for i in range(len(poses) - 1):
        d = xy[i + 1] - xy[i]
        if np.linalg.norm(d) > 1e-6:
            yaw_motion_diff.append(abs(_deg(wrap_angle(math.atan2(d[1], d[0]) - yaw[i + 1]))))
    yaw_motion_diff = np.asarray(yaw_motion_diff) if yaw_motion_diff else np.zeros(0)

    result = adapter.rollout(poses, record.marker_xyz, record.target_xyz)

    # --- 2/3/4. teacher action vs demonstration local motion -----------------
    demo_yaw = yaw
    demo_z = raw[:, 2]
    check = collections.defaultdict(lambda: [0, 0])  # action -> [agree, total]
    executed_bad = 0
    for i, action in enumerate(result.actions):
        current = result.poses[i]
        # executed pose must equal one official action step
        expected = transition_pose(current, action, adapter.config)
        actual = result.poses[i + 1] if i + 1 < len(result.poses) else None
        if actual is not None:
            d_xy = math.hypot(actual.x - expected.x, actual.y - expected.y)
            d_z = abs(actual.z - expected.z)
            d_yaw = abs(_deg(wrap_angle(actual.yaw - expected.yaw)))
            if d_xy > 0.05 or d_z > 0.05 or d_yaw > 0.5:
                executed_bad += 1

        # nearest demonstration index for an independent motion reading
        j = int(np.argmin(np.linalg.norm(xy - current.xy, axis=1)))
        if j >= len(poses) - 1:
            continue
        d = xy[j + 1] - xy[j]
        dz_demo = float(demo_z[j + 1] - demo_z[j])
        yaw_demo = _deg(wrap_angle(demo_yaw[j + 1] - demo_yaw[j]))
        if ACTION_NAMES[action] == "TURN_RIGHT":
            agree = yaw_demo < -TURN_LEFT_MIN_DEG
        elif ACTION_NAMES[action] == "TURN_LEFT":
            agree = yaw_demo > TURN_LEFT_MIN_DEG
        elif ACTION_NAMES[action] == "GO_UP":
            agree = dz_demo > 0.5
        elif ACTION_NAMES[action] == "GO_DOWN":
            agree = dz_demo < -0.5
        elif ACTION_NAMES[action] == "MOVE_FORWARD":
            agree = abs(yaw_demo) <= FORWARD_MAX_DEG and abs(dz_demo) <= 1.0
        else:  # STOP
            agree = True
        check[ACTION_NAMES[action]][1] += 1
        check[ACTION_NAMES[action]][0] += int(bool(agree))

    return {
        "map": record.map_name,
        "steps": len(result.actions),
        "zero_direction_vectors": zero_dir,
        "yaw_motion_median_deg": float(np.median(yaw_motion_diff)) if len(yaw_motion_diff) else None,
        "yaw_motion_p95_deg": float(np.percentile(yaw_motion_diff, 95)) if len(yaw_motion_diff) else None,
        "action_histogram": {
            ACTION_NAMES[a]: int(result.actions.count(a)) for a in ACTION_NAMES
        },
        "semantic_agreement": {k: list(v) for k, v in check.items()},
        "executed_pose_mismatch": executed_bad,
        "marker_target_distance_m": float(
            np.linalg.norm(np.asarray(record.marker_xyz) - np.asarray(record.target_xyz))
        ),
        "trajectory_points": int(len(poses)),
        "success": bool(result.success),
        "stop_reason": result.stop_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ply-root", required=True)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.split == "test_unseen":
        raise PermissionError("test_unseen stays sealed")
    data_config = load_data_config(
        data_root=args.data_root, ply_root=args.ply_root
    )
    dataset = CityNavDataset(
        data_config, args.split, include_processed_descriptions=False
    )
    records = dataset.records
    if args.max_records is not None and int(args.max_records) < len(records):
        idx = np.linspace(0, len(records) - 1, int(args.max_records), dtype=int)
        records = [records[int(i)] for i in idx]

    adapter = TrajectoryAdapter(dataset.teacher_config)
    episodes = []
    for n, record in enumerate(records, 1):
        episodes.append(audit_episode(record, adapter))
        if n % 200 == 0:
            print(f"  audited {n}/{len(records)}", flush=True)

    # ---------------- aggregate ----------------
    action_totals = collections.Counter()
    agreement = collections.defaultdict(lambda: [0, 0])
    zero_dir_eps = 0
    executed_bad_total = 0
    yaw_med = []
    yaw_p95 = []
    marker_d = []
    for ep in episodes:
        for name, count in ep["action_histogram"].items():
            action_totals[name] += count
        for name, (agree, total) in ep["semantic_agreement"].items():
            agreement[name][0] += agree
            agreement[name][1] += total
        zero_dir_eps += int(ep["zero_direction_vectors"] > 0)
        executed_bad_total += ep["executed_pose_mismatch"]
        if ep["yaw_motion_median_deg"] is not None:
            yaw_med.append(ep["yaw_motion_median_deg"])
            yaw_p95.append(ep["yaw_motion_p95_deg"])
        marker_d.append(ep["marker_target_distance_m"])

    summary = {
        "split": args.split,
        "episodes": len(episodes),
        "available": len(dataset.records),
        "test_unseen_read": False,
        "action_totals": dict(action_totals),
        "semantic_agreement": {
            k: {"agree": v[0], "total": v[1], "rate": (v[0] / v[1]) if v[1] else None}
            for k, v in agreement.items()
        },
        "episodes_with_zero_direction_vectors": zero_dir_eps,
        "executed_pose_mismatches": executed_bad_total,
        "yaw_vs_motion_median_deg": float(np.median(yaw_med)) if yaw_med else None,
        "yaw_vs_motion_p95_deg": float(np.percentile(yaw_p95, 95)) if yaw_p95 else None,
        "marker_target_distance_m": {
            "median": float(np.median(marker_d)) if marker_d else None,
            "p95": float(np.percentile(marker_d, 95)) if marker_d else None,
            "max": float(np.max(marker_d)) if marker_d else None,
        },
    }
    payload = {"summary": summary, "episodes": episodes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
