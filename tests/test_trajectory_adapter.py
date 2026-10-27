"""Contract tests for the CityNav six-action trajectory adapter."""

from __future__ import annotations

import math

import pytest

from src.config import TeacherConfig
from src.data.actions import (
    GO_DOWN,
    GO_UP,
    MOVE_FORWARD,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
    move_head_index,
)
from src.data.trajectory_adapter import (
    Pose5D,
    TrajectoryAdapter,
    load_trajectory,
    transition_pose,
)


def _poses_along_x() -> list[Pose5D]:
    return [
        Pose5D(0.0, 0.0, 10.0, 0.0, 0.0),
        Pose5D(10.0, 0.0, 10.0, 0.0, 0.0),
        Pose5D(20.0, 0.0, 10.0, 0.0, 0.0),
    ]


def test_load_trajectory_preserves_true_altitude_and_direction() -> None:
    raw = [
        [0.0, 0.0, 99.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 101.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 102.0, 0.0, 0.0, -1.0],
    ]
    poses = load_trajectory(raw)
    assert poses[0].z == pytest.approx(99.0)
    assert poses[1].z == pytest.approx(101.0)
    assert poses[0].yaw == pytest.approx(0.0)
    assert poses[1].yaw == pytest.approx(math.pi / 2)
    # The last point looks straight down; yaw falls back to local motion.
    assert poses[2].yaw == pytest.approx(math.pi / 2)
    assert poses[2].pitch == pytest.approx(-math.pi / 2)


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


def test_six_action_transitions_match_official_parameters() -> None:
    config = TeacherConfig()
    pose = Pose5D(0.0, 0.0, 50.0, 0.0, 0.0)

    forward = transition_pose(pose, MOVE_FORWARD, config)
    right = transition_pose(pose, TURN_RIGHT, config)
    left = transition_pose(pose, TURN_LEFT, config)
    up = transition_pose(pose, GO_UP, config)
    down = transition_pose(pose, GO_DOWN, config)
    stop = transition_pose(pose, STOP, config)

    assert forward.x == pytest.approx(5.0)
    assert forward.y == pytest.approx(0.0)
    assert right.yaw == pytest.approx(math.radians(-30.0))
    assert left.yaw == pytest.approx(math.radians(30.0))
    assert up.z == pytest.approx(52.0)
    assert down.z == pytest.approx(48.0)
    assert stop == pose


def test_move_head_order_is_forward_right_left_up_down() -> None:
    assert [move_head_index(action) for action in (1, 2, 3, 4, 5)] == [
        0, 1, 2, 3, 4
    ]


def test_teacher_uses_marker_stop_and_target_success_separately() -> None:
    config = TeacherConfig(
        stop_distance_m=5.0,
        success_radius_m=20.0,
        lookahead=1,
        max_steps=100,
    )
    # Teacher goal is marker (30,0,10); evaluation target is far (100,0,10).
    result = TrajectoryAdapter(config).rollout(
        _poses_along_x(),
        marker_xyz=(30.0, 0.0, 10.0),
        target_xyz=(100.0, 0.0, 10.0),
    )
    assert result.stop_reason == "teacher_stop_distance"
    assert result.final_marker_distance_m < 5.0
    assert result.final_target_distance_m > 20.0
    assert result.success is False
    assert result.actions[-1] == STOP


def test_teacher_success_uses_twenty_metre_target_radius() -> None:
    config = TeacherConfig(
        stop_distance_m=5.0,
        success_radius_m=20.0,
        lookahead=1,
        max_steps=100,
    )
    result = TrajectoryAdapter(config).rollout(
        _poses_along_x(),
        marker_xyz=(30.0, 0.0, 10.0),
        target_xyz=(31.0, 0.0, 10.0),
    )
    assert result.final_marker_distance_m < 5.0
    assert result.final_target_distance_m < 20.0
    assert result.success is True


def test_teacher_emits_vertical_actions() -> None:
    config = TeacherConfig(
        stop_distance_m=5.0,
        success_radius_m=20.0,
        lookahead=1,
        max_steps=100,
    )
    up = TrajectoryAdapter(config).rollout(
        [Pose5D(0.0, 0.0, 10.0, 0.0, 0.0)],
        marker_xyz=(0.0, 0.0, 20.0),
        target_xyz=(0.0, 0.0, 20.0),
    )
    down = TrajectoryAdapter(config).rollout(
        [Pose5D(0.0, 0.0, 20.0, 0.0, 0.0)],
        marker_xyz=(0.0, 0.0, 10.0),
        target_xyz=(0.0, 0.0, 10.0),
    )
    assert GO_UP in up.actions
    assert GO_DOWN in down.actions


def test_teacher_max_steps_has_terminal_stop_and_failure() -> None:
    config = TeacherConfig(
        stop_distance_m=1.0,
        success_radius_m=20.0,
        lookahead=1,
        max_steps=2,
    )
    result = TrajectoryAdapter(config).rollout(
        [Pose5D(0.0, 0.0, 10.0, 0.0, 0.0)],
        marker_xyz=(100.0, 0.0, 10.0),
        target_xyz=(100.0, 0.0, 10.0),
    )
    assert result.stop_reason == "max_steps"
    assert result.actions[-1] == STOP
    assert result.success is False
    assert len(result.actions) == 3


def test_supervision_steps_align_poses_and_actions() -> None:
    config = TeacherConfig(max_steps=100)
    result = TrajectoryAdapter(config).rollout(
        _poses_along_x(),
        marker_xyz=(30.0, 0.0, 10.0),
        target_xyz=(31.0, 0.0, 10.0),
    )
    steps = result.supervision_steps()
    assert len(steps) == len(result.actions)
    assert len(steps) == len(result.poses)
    assert steps[-1][1] == STOP
