#!/usr/bin/env python3
"""Contract checks for the fixed-50 m four-action teacher labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_action_teacher import ACTION_NAMES, FORWARD, STOP, TURN_LEFT, TURN_RIGHT, move


def check(labels_path: Path, sample: int = 500, seed: int = 0) -> dict:
    z = np.load(labels_path)
    actions = z["actions"]
    states = z["states"]
    offsets = z["state_offsets"]
    n = len(z["keys"])
    ok = {}

    ok["action_ids_valid"] = bool(np.isin(actions, [0, 1, 2, 3]).all())
    ok["no_vertical_actions"] = bool(not np.isin(actions, [4, 5]).any())

    lengths = np.diff(offsets)
    ok["state_action_aligned"] = bool(len(states) == len(actions) == int(lengths.sum()))

    success_ok = True
    failure_stop = False
    for i in range(n):
        ep_actions = actions[offsets[i]:offsets[i + 1]]
        if len(ep_actions) == 0:
            continue
        if bool(z["success"][i]):
            if ep_actions[-1] != STOP:
                success_ok = False
        elif STOP in ep_actions:
            failure_stop = True
    ok["success_ends_with_stop"] = success_ok
    ok["failure_has_no_stop"] = not failure_stop

    # STOP distance and exact action arithmetic
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(z["success"])
    picks = rng.choice(candidates, size=min(sample, len(candidates)), replace=False) if len(candidates) else []
    max_stop_dist = 0.0
    max_replay_err = 0.0
    for i in picks:
        sl = slice(offsets[i], offsets[i + 1])
        ep_states, ep_actions = states[sl], actions[sl]
        target = z["target_xy"][i]
        stop_state = ep_states[-1]
        max_stop_dist = max(max_stop_dist, float(np.linalg.norm(stop_state[:2] - target)))
        pos = z["start_xy"][i].astype(np.float64)
        yaw = float(z["start_yaw"][i])
        err = 0.0
        for j, action in enumerate(ep_actions):
            expected = ep_states[j]
            err = max(err, float(np.linalg.norm(pos - expected[:2])), abs(float((yaw - expected[2] + math.pi) % (2 * math.pi) - math.pi)))
            pos, yaw = move(pos, yaw, int(action))
        max_replay_err = max(max_replay_err, err)
    ok["stop_within_20m"] = bool(max_stop_dist <= 20.0 + 1e-3)
    ok["replay_consistent"] = bool(max_replay_err < 1e-3)

    reasons = set(z["reasons"].tolist())
    ok["reasons_valid"] = reasons <= {"success", "never_entered_20m", "step_limit",
                                      "success_without_stop", "degenerate"}

    result = {
        "labels": str(labels_path),
        "n_episodes": int(n),
        "n_states": int(len(states)),
        "success": int(z["success"].sum()),
        "checks": ok,
        "all_passed": bool(all(ok.values())),
        "max_stop_distance_m": max_stop_dist,
        "max_replay_error": max_replay_err,
        "action_counts": {ACTION_NAMES[int(a)]: int((actions == a).sum()) for a in range(4)},
    }
    z.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path(__file__).resolve().parents[1] / "results/four_action_labels.npz")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results/teacher_contract.json")
    args = parser.parse_args()
    result = check(args.labels)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
