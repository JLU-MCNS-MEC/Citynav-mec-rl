"""Unit tests for the fixed-50 m four-action teacher."""

from __future__ import annotations

import math

import numpy as np
import pytest

from four_action_teacher import (
    FORWARD,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
    build_waypoints,
    clean_path,
    move,
    rollout,
    wrap,
)


def test_wrap_stays_in_range():
    for angle in (-7.0, -math.pi, 0.0, math.pi - 1e-9, 9.0):
        wrapped = wrap(angle)
        assert -math.pi <= wrapped < math.pi


def test_forward_moves_exactly_5m():
    pos, yaw = move(np.array([0.0, 0.0]), 0.0, FORWARD)
    assert np.allclose(pos, [5.0, 0.0])
    assert yaw == 0.0


def test_turns_are_exactly_30_degrees():
    _, yaw = move(np.array([0.0, 0.0]), 0.0, TURN_LEFT)
    assert math.isclose(yaw, math.radians(30.0), abs_tol=1e-12)
    _, yaw = move(np.array([0.0, 0.0]), 0.0, TURN_RIGHT)
    assert math.isclose(yaw, math.radians(-30.0), abs_tol=1e-12)


def test_straight_rollout_stops_at_20m():
    way = np.array([[0.0, 0.0], [40.0, 0.0], [80.0, 0.0], [100.0, 0.0]])
    result = rollout(way, np.array([100.0, 0.0]), 0.0, path_len=100.0)
    assert result.success
    assert result.actions[-1] == STOP
    assert set(result.actions[:-1]) == {FORWARD}
    assert result.steps == len(result.actions)
    assert result.steps == 17  # sixteen 5 m moves, then STOP at x=80
    assert len(result.states) == len(result.actions)


def test_turn_left_then_forward():
    way = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0]])
    result = rollout(way, np.array([0.0, 40.0]), 0.0, path_len=40.0, max_steps=200)
    assert result.success
    # heading must rotate left by 90 degrees in 30-degree steps
    assert result.actions[:3] == [TURN_LEFT, TURN_LEFT, TURN_LEFT]
    assert result.actions[-1] == STOP


def test_failed_rollout_never_appends_stop():
    way = np.array([[0.0, 0.0], [5.0, 0.0]])
    result = rollout(way, np.array([500.0, 0.0]), 0.0, path_len=5.0, max_steps=30)
    assert not result.success
    assert STOP not in result.actions
    assert result.reason in {"step_limit", "success_without_stop"}


def test_waypoints_follow_official_stride():
    xy = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
    marker = np.array([45.0, 0.0])
    way = build_waypoints(xy, marker)
    assert np.allclose(way[0], [0.0, 0.0])
    # the official rule only appends points farther than 5 m, so the marker can
    # be skipped when the last appended waypoint is within 5 m of it
    assert np.linalg.norm(way[-1] - marker) <= 5.0 + 1e-9
    gaps = np.linalg.norm(np.diff(way, axis=0), axis=1)
    assert (gaps > 5.0).all()


def test_clean_path_removes_duplicates():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [0.01, 0.0], [1.0, 0.0]])
    cleaned, removed = clean_path(xy)
    assert len(cleaned) == 2
    assert removed == 2
