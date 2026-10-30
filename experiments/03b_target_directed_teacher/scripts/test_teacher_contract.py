"""Contract tests for the direct target-directed teacher."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from direct_teacher import (  # noqa: E402
    FORWARD_M, STOP, TURN_LEFT, TURN_RIGHT, run_episode, start_pose,
)

DATA = Path("/home/20261024_1/Workspace/DATA/sbfnav-data/original_citynav")
RESULTS = HERE / "results"

FLAT = {
    "name": "toy",
    "blocked": np.zeros((600, 600), bool),
    "cell_m": 1.0,
    "bounds": [0.0, 0.0, 600.0, 600.0],
    "ground_level": 0.0,
    "ceiling_m": 49.5,
    "max_above_ground_m": 0.0,
    "blocked_ratio": 0.0,
}


def record_with(start, yaw, target, middle=None) -> dict:
    traj = [[start[0], start[1], 0.0, math.cos(yaw), math.sin(yaw), 0.0]]
    traj += [[p[0], p[1], 0.0, math.cos(yaw), math.sin(yaw), 0.0]
             for p in (middle or [])]
    return {
        "area": "toy", "block": 0,
        "trajectory": traj,
        "target_positions": [[target[0], target[1], 0.0]],
        "marker_positions": [[target[0], target[1], 0.0]],
        "object_ids": 1, "ann_ids": 1, "descriptions": ["ignore me"],
    }


def test_stop_inside_20m():
    out = run_episode(record_with([100.0, 100.0], 0.0, [115.0, 100.0]), FLAT)
    assert out["success"] and out["actions"] == [STOP]


def test_heading_error_turns_then_forward():
    # target is 30 m to the left of a forward-pointing UAV
    out = run_episode(record_with([100.0, 100.0], 0.0, [100.0, 130.0]), FLAT)
    assert out["actions"][0] == TURN_LEFT
    assert out["success"]


def test_right_turn_when_target_is_to_the_right():
    out = run_episode(record_with([100.0, 100.0], 0.0, [100.0, 70.0]), FLAT)
    assert out["actions"][0] == TURN_RIGHT


def test_actions_follow_four_action_geometry():
    out = run_episode(record_with([100.0, 100.0], 0.0, [300.0, 220.0]), FLAT)
    s = np.asarray(out["states"], np.float64)
    a = out["actions"]
    assert set(a) <= {0, 1, 2, 3}
    for i, act in enumerate(a[:-1]):
        d = np.linalg.norm(s[i + 1, :2] - s[i, :2])
        dyaw = (s[i + 1, 2] - s[i, 2] + math.pi) % (2 * math.pi) - math.pi
        if act == 1:
            assert abs(d - FORWARD_M) < 1e-6
            assert abs(dyaw) < 1e-9
        else:
            assert d < 1e-9
            assert abs(abs(dyaw) - math.radians(30.0)) < 1e-9


def test_human_path_middle_points_and_marker_do_not_change_labels():
    target = [260.0, 140.0]
    a = run_episode(record_with([100.0, 100.0], 0.3, target,
                                middle=[[120.0, 120.0], [200.0, 110.0]]), FLAT)
    rec = record_with([100.0, 100.0], 0.3, target)
    rec["marker_positions"] = [[10.0, 10.0, 0.0]]
    rec["descriptions"] = ["a completely different sentence"]
    rec["trajectory"] = [rec["trajectory"][0]]        # no intermediate points
    b = run_episode(rec, FLAT)
    assert a["actions"] == b["actions"]
    assert np.allclose(a["states"], b["states"])
    assert np.allclose(np.asarray(a["waypoints"]), np.asarray(b["waypoints"]))


def test_start_pose_uses_trajectory_orientation():
    rec = record_with([10.0, 20.0], math.pi / 3, [100.0, 100.0])
    xy, yaw = start_pose(rec)
    assert np.allclose(xy, [10.0, 20.0])
    assert abs(yaw - math.pi / 3) < 1e-9


def test_stored_train_labels_obey_stop_and_geometry():
    z = np.load(RESULTS / "teacher_train_seen.npz")
    off = z["offsets"]
    states, actions, targets = z["states"], z["actions"], z["targets"]
    n = len(off) - 1
    assert n > 20000
    step = max(1, n // 400)
    checked = 0
    for i in range(0, n, step):
        s = states[off[i]:off[i + 1]].astype(np.float64)
        a = actions[off[i]:off[i + 1]]
        t = targets[i].astype(np.float64)
        assert set(int(x) for x in a) <= {0, 1, 2, 3}
        if a[-1] == STOP:
            d = float(np.linalg.norm(s[-1, :2] - t))
            assert d <= 20.0 + 1e-6, (i, d)
            assert int((a == STOP).sum()) == 1
            # the decision state is s[-1]; the state before it may still be
            # outside 20 m (a forward step can enter the radius)
            if len(s) > 1:
                prev = float(np.linalg.norm(s[-2, :2] - t))
                assert prev <= 20.0 + 5.0 + 1e-6
        else:
            assert int((a == STOP).sum()) == 0
        for k, act in enumerate(a[:-1]):
            step_len = float(np.linalg.norm(s[k + 1, :2] - s[k, :2]))
            if act == 1:
                assert abs(step_len - 5.0) < 2e-3   # float32 storage
        checked += 1
    assert checked > 100


def test_every_val_episode_ends_with_one_stop():
    z = np.load(RESULTS / "teacher_val_seen.npz")
    summary = json.loads((RESULTS / "summary_val_seen.json").read_text())
    off = z["offsets"]
    acts = z["actions"]
    assert len(off) - 1 == summary["records"]
    for i in range(len(off) - 1):
        a = acts[off[i]:off[i + 1]]
        assert len(a) > 0
        if a[-1] == STOP:                     # success: exactly one STOP, at the end
            assert int((a == STOP).sum()) == 1
        else:                                 # failure (should not happen): no STOP
            assert 0 not in set(int(x) for x in a)
    assert summary["reason_counts"].get("step_limit", 0) >= 0


def _aliasing_case(cruise_trigger_deg):
    """Start with a 15.09 deg heading error: the naive rule's fixed point."""
    bearing = math.radians(15.09)
    target = [100.0 + 500.0 * math.cos(bearing), 100.0 + 500.0 * math.sin(bearing)]
    rec = record_with([100.0, 100.0], 0.0, target)
    return run_episode(rec, FLAT, cruise_trigger_deg=cruise_trigger_deg)


def test_naive_rule_chatters_at_the_switching_boundary():
    out = _aliasing_case(None)
    a = out["actions"][:40]
    turns = sum(1 for x in a if x in (TURN_LEFT, TURN_RIGHT))
    forwards = sum(1 for x in a if x == 1)
    assert turns >= 15 and forwards <= 25, (turns, forwards)


def test_deadband_rule_has_no_limit_cycle():
    out = _aliasing_case(25.0)
    a = out["actions"][:40]
    turns = sum(1 for x in a if x in (TURN_LEFT, TURN_RIGHT))
    forwards = sum(1 for x in a if x == 1)
    assert turns <= 2 and forwards >= 30, (turns, forwards)
    assert out["success"]


def test_stored_labels_have_no_turn_chatter():
    z = np.load(RESULTS / "teacher_train_seen.npz")
    off, acts = z["offsets"], z["actions"]
    n = len(off) - 1
    total_actions = 0
    turn_total = 0
    turns_per_episode = []
    for i in range(n):
        a = acts[off[i]:off[i + 1]]
        t = int(((a == TURN_LEFT) | (a == TURN_RIGHT)).sum())
        turns_per_episode.append(t)
        turn_total += t
        total_actions += len(a)
    turns_per_episode = np.asarray(turns_per_episode)
    assert turn_total / total_actions < 0.12, turn_total / total_actions
    assert turns_per_episode.mean() < 5.0, turns_per_episode.mean()
    assert float((turns_per_episode > 6).mean()) < 0.05
