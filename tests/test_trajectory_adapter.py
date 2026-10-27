"""Contract tests for the CityNav six-number trajectory adapter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import TeacherConfig
from src.data.trajectory_adapter import (
    FORWARD,
    LEFT,
    RIGHT,
    STOP,
    Pose5D,
    TrajectoryAdapter,
    load_trajectory,
    resample_polyline,
    transition_pose,
)


def test_load_trajectory_converts_direction_vectors_to_yaw_pitch() -> None:
    raw = [
        [0.0, 0.0, 10.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 10.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 10.0, 0.0, 0.0, -1.0],
    ]
    poses = load_trajectory(raw)
    assert poses[0].yaw == pytest.approx(0.0)
    assert poses[0].pitch == pytest.approx(0.0)
    assert poses[1].yaw == pytest.approx(math.pi / 2)
    assert poses[1].pitch == pytest.approx(0.0)
    # The last point looks straight down; yaw falls back to local motion.
    assert poses[2].yaw == pytest.approx(math.pi / 2)
    assert poses[2].pitch == pytest.approx(-math.pi / 2)


def test_fixed_altitude_uses_ground_level() -> None:
    poses = load_trajectory(
        [[0.0, 0.0, 99.0, 1.0, 0.0, 0.0]],
        fix_altitude_m=50.0,
        ground_level_m=10.0,
    )
    assert poses[0].z == pytest.approx(60.0)
    with pytest.raises(ValueError):
        load_trajectory([[0.0, 0.0, 99.0, 1.0, 0.0, 0.0]], fix_altitude_m=50.0)


def test_degenerate_horizontal_direction_uses_previous_yaw() -> None:
    raw = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    ]
    poses = load_trajectory(raw)
    assert [pose.yaw for pose in poses] == pytest.approx([0.0, 0.0, 0.0])


def test_rejects_malformed_points_and_non_finite_values() -> None:
    with pytest.raises(ValueError):
        load_trajectory([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        load_trajectory([[0.0, 0.0, 0.0, 1.0, 0.0, float("nan")]])


def test_transition_pose_matches_four_action_interface() -> None:
    config = TeacherConfig(forward_step_m=5.0, turn_step_deg=15.0)
    pose = Pose5D(0.0, 0.0, 50.0, 0.0, 0.0)

    forward = transition_pose(pose, FORWARD, config)
    left = transition_pose(pose, LEFT, config)
    right = transition_pose(pose, RIGHT, config)
    stop = transition_pose(pose, STOP, config)

    assert forward.x == pytest.approx(5.0)
    assert forward.y == pytest.approx(0.0)
    assert left.yaw == pytest.approx(math.radians(15.0))
    assert right.yaw == pytest.approx(math.radians(-15.0))
    assert stop == pose


def test_resample_polyline_is_uniform() -> None:
    path = resample_polyline([[0.0, 0.0], [10.0, 0.0]], step_m=2.0)
    assert path.shape == (6, 2)
    assert np.allclose(path[0], [0.0, 0.0])
    assert np.allclose(path[-1], [10.0, 0.0])
    assert np.allclose(np.linalg.norm(np.diff(path, axis=0), axis=1), 2.0)


def test_teacher_stops_inside_success_radius() -> None:
    config = TeacherConfig(
        forward_step_m=5.0,
        turn_step_deg=15.0,
        stop_radius_m=20.0,
        lookahead_m=12.0,
        max_steps=50,
    )
    result = TrajectoryAdapter(config).rollout(
        [Pose5D(0.0, 0.0, 50.0, 0.0, 0.0)],
        [30.0, 0.0],
    )
    assert result.success is True
    assert result.stop_reason == "target_radius"
    assert result.actions[-1] == STOP
    assert result.final_distance_m <= config.stop_radius_m
    assert len(result.actions) == len(result.poses)


def test_teacher_reports_max_steps_failure() -> None:
    config = TeacherConfig(max_steps=2, forward_step_m=5.0, lookahead_m=12.0)
    result = TrajectoryAdapter(config).rollout(
        [Pose5D(0.0, 0.0, 50.0, 0.0, 0.0)],
        [100.0, 0.0],
    )
    assert result.success is False
    assert result.stop_reason == "max_steps"
    assert result.actions == [FORWARD, FORWARD]
    assert len(result.poses) == 3
