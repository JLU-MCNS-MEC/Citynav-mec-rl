#!/usr/bin/env python3
"""C. Causality / leakage tests for the extended history construction.

These test ``history_arrays`` directly, independent of rendering:
  * slot k must describe the transition at absolute step ``index-1-k``;
  * history must not change when the current or future labels/actions change;
  * history must not change when future poses change;
  * masks must expose exactly min(index, 16) valid slots.

Run: python test_history_leakage.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[2]
sys.path.insert(0, str(ROOT))

from src.data.actions import ACTION_NAMES, NUM_ACTIONS  # noqa: E402
from src.data.t5_features import HISTORY_STEPS, history_arrays  # noqa: E402
from src.data.trajectory_adapter import Pose5D  # noqa: E402


def _sequence(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    poses = [
        Pose5D(float(i) * 5.0, 0.0, 10.0, 0.1 * i, 0.0) for i in range(n)
    ]
    actions = [int(v) for v in rng.integers(0, NUM_ACTIONS, size=n)]
    visuals = rng.normal(size=(n, 768)).astype(np.float32)
    return poses, actions, visuals


def _history(poses, actions, visuals, index):
    return history_arrays(poses, actions, index, visuals)


def test_slots_match_previous_actions():
    poses, actions, visuals = _sequence(40)
    index = 30
    _, _, h_action, h_mask = _history(poses, actions, visuals, index)
    assert int(h_mask.sum()) == HISTORY_STEPS
    for k in range(HISTORY_STEPS):
        source = index - 1 - k
        expected = np.zeros(NUM_ACTIONS, dtype=np.float32)
        expected[actions[source]] = 1.0
        assert np.array_equal(h_action[k], expected), (k, source)
        assert ACTION_NAMES[actions[source]]


def test_short_history_is_causal():
    poses, actions, visuals = _sequence(12)
    index = 3
    _, _, h_action, h_mask = _history(poses, actions, visuals, index)
    assert int(h_mask.sum()) == index
    for k in range(index):
        expected = np.zeros(NUM_ACTIONS, dtype=np.float32)
        expected[actions[index - 1 - k]] = 1.0
        assert np.array_equal(h_action[k], expected)


def test_future_actions_do_not_leak():
    poses, actions, visuals = _sequence(40)
    index = 30
    base = _history(poses, actions, visuals, index)
    changed = list(actions)
    for j in range(index, len(changed)):
        changed[j] = (changed[j] + 1) % NUM_ACTIONS
    after = _history(poses, changed, visuals, index)
    for a, b in zip(base, after):
        assert np.array_equal(a, b)


def test_future_poses_do_not_leak():
    poses, actions, visuals = _sequence(40)
    index = 30
    base = _history(poses, actions, visuals, index)
    changed = list(poses)
    for j in range(index + 1, len(changed)):
        p = changed[j]
        changed[j] = Pose5D(p.x + 100.0, p.y - 25.0, p.z, p.yaw + 1.0, p.pitch)
    after = _history(changed, actions, visuals, index)
    for a, b in zip(base, after):
        assert np.array_equal(a, b)


def test_current_pose_anchors_relative_features_causally():
    """The current pose is the documented reference for relative features.

    It may change the relative slots (dims 3-5) of every history entry, but the
    historical motion features (dims 0-2) and the masks must stay identical.
    """
    poses, actions, visuals = _sequence(40)
    index = 30
    base_vis, base_pose, base_action, base_mask = _history(
        poses, actions, visuals, index
    )
    changed = list(poses)
    p = changed[index]
    changed[index] = Pose5D(p.x + 3.0, p.y, p.z, p.yaw, p.pitch)
    after = _history(changed, actions, visuals, index)
    assert np.array_equal(base_vis, after[0])
    assert np.array_equal(base_action, after[2])
    assert np.array_equal(base_mask, after[3])
    # Slot 0 is the transition into the current state; moving the current pose
    # must change exactly that motion, and nothing earlier.
    assert not np.array_equal(base_pose[0, :3], after[1][0, :3])
    assert np.array_equal(base_pose[1:, :3], after[1][1:, :3])
    # Relative features are anchored to the current pose by design.
    assert not np.array_equal(base_pose[1:, 3:], after[1][1:, 3:])


def test_visual_history_is_causal():
    poses, actions, visuals = _sequence(40)
    index = 30
    base = _history(poses, actions, visuals, index)
    changed = visuals.copy()
    changed[index:] += 5.0
    after = _history(poses, actions, changed, index)
    assert np.array_equal(base[0], after[0])
    changed = visuals.copy()
    changed[index - 1] += 5.0
    after = _history(poses, actions, changed, index)
    assert not np.array_equal(base[0][0], after[0][0])


def main() -> int:
    tests = [
        test_slots_match_previous_actions,
        test_short_history_is_causal,
        test_future_actions_do_not_leak,
        test_future_poses_do_not_leak,
        test_current_pose_anchors_relative_features_causally,
        test_visual_history_is_causal,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("all history causality tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
