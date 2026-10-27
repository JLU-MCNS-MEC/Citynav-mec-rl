"""CityNav trajectory -> five-dimensional pose -> six discrete actions.

CityNav stores each trajectory point as six numbers::

    x, y, z, dx, dy, dz

Following the official ``vlnce/space.py`` conversion::

    yaw   = atan2(dy, dx)
    pitch = atan2(dz, sqrt(dx**2 + dy**2))

The six numbers are *not* six actions.  This module converts them to a
five-dimensional pose and then to the official six-action interface
``STOP / MOVE_FORWARD / TURN_RIGHT / TURN_LEFT / GO_UP / GO_DOWN`` using the
official lookahead teacher over the human demonstration.

Important separation of distances
---------------------------------
* ``TeacherConfig.stop_distance_m`` (default 5 m) is the teacher's own stop
  threshold, copied from the official ``LookaheadTeacherParams``.
* ``TeacherConfig.success_radius_m`` (default 20 m) is the navigation success
  radius, copied from the official ``--success_dist`` default.  It is used only
  for evaluation, never to decide a teacher action.

The original trajectory altitude is preserved.  No fixed 50 m altitude is
applied anywhere in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.config import TeacherConfig
from src.data.actions import (
    ACTION_NAMES,
    GO_DOWN,
    GO_UP,
    MOVE_FORWARD,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
)


_EPS = 1e-9


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose5D:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float

    @classmethod
    def from_direction_vector(
        cls,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> "Pose5D":
        yaw = math.atan2(float(dy), float(dx))
        pitch = math.atan2(float(dz), math.hypot(float(dx), float(dy)))
        return cls(float(x), float(y), float(z), yaw, pitch)

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.x, self.y, self.z, self.yaw, self.pitch)

    @property
    def xyz(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.z], dtype=np.float64)

    @property
    def xy(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float64)

    def with_pitch(self, pitch: float) -> "Pose5D":
        return Pose5D(self.x, self.y, self.z, self.yaw, float(pitch))


def _motion_yaw(raw_trajectory: Sequence[Sequence[float]], index: int) -> float | None:
    """Yaw of the local motion direction, using next or previous point.

    The direction must point *out of* the current point.  When the previous
    point is used, the displacement is therefore ``current - previous`` rather
    than ``previous - current``; otherwise the fallback yaw is reversed by 180
    degrees.
    """
    n = len(raw_trajectory)
    current = raw_trajectory[index]
    for candidate, sign in ((index + 1, 1.0), (index - 1, -1.0)):
        if 0 <= candidate < n:
            other = raw_trajectory[candidate]
            mx = sign * (float(other[0]) - float(current[0]))
            my = sign * (float(other[1]) - float(current[1]))
            if math.hypot(mx, my) > 1e-6:
                return math.atan2(my, mx)
    return None


def yaw_from_direction_or_motion(
    raw_trajectory: Sequence[Sequence[float]],
    index: int,
    previous_yaw: float = 0.0,
) -> float:
    point = raw_trajectory[index]
    dx, dy = float(point[3]), float(point[4])
    if math.hypot(dx, dy) > 1e-6:
        return math.atan2(dy, dx)
    motion_yaw = _motion_yaw(raw_trajectory, index)
    if motion_yaw is not None:
        return motion_yaw
    return float(previous_yaw)


def load_trajectory(raw_trajectory: Sequence[Sequence[float]]) -> list[Pose5D]:
    """Convert CityNav six-number points to :class:`Pose5D`.

    The original z is preserved.  No fixed altitude is applied.
    """
    if not raw_trajectory:
        raise ValueError("trajectory must not be empty")
    for index, point in enumerate(raw_trajectory):
        if len(point) != 6:
            raise ValueError(
                f"trajectory point {index} has {len(point)} values; expected 6"
            )
        if not all(math.isfinite(float(value)) for value in point):
            raise ValueError(f"trajectory point {index} contains non-finite values")

    poses: list[Pose5D] = []
    previous_yaw = 0.0
    for index, point in enumerate(raw_trajectory):
        x, y, z, dx, dy, dz = (float(value) for value in point)
        yaw = yaw_from_direction_or_motion(raw_trajectory, index, previous_yaw)
        pitch = math.atan2(dz, math.hypot(dx, dy))
        poses.append(Pose5D(x, y, z, yaw, pitch))
        previous_yaw = yaw
    return poses


def transition_pose(
    pose: Pose5D,
    action: int,
    config: TeacherConfig,
) -> Pose5D:
    """Apply one official discrete action to a pose.

    TURN_LEFT increases yaw, TURN_RIGHT decreases yaw, matching the official
    CityNav action table.  GO_UP/GO_DOWN change z by 2 m.  Camera pitch is not
    changed by these navigation actions; the teacher rollout replaces it with
    the pitch of the nearest demonstration point.
    """
    action = int(action)
    if action == int(STOP):
        return pose
    if action == int(MOVE_FORWARD):
        return Pose5D(
            pose.x + float(config.forward_step_m) * math.cos(pose.yaw),
            pose.y + float(config.forward_step_m) * math.sin(pose.yaw),
            pose.z,
            pose.yaw,
            pose.pitch,
        )
    if action == int(TURN_RIGHT):
        return Pose5D(
            pose.x, pose.y, pose.z,
            wrap_angle(pose.yaw - config.turn_step_rad), pose.pitch,
        )
    if action == int(TURN_LEFT):
        return Pose5D(
            pose.x, pose.y, pose.z,
            wrap_angle(pose.yaw + config.turn_step_rad), pose.pitch,
        )
    if action == int(GO_UP):
        return Pose5D(
            pose.x, pose.y, pose.z + float(config.vertical_step_m),
            pose.yaw, pose.pitch,
        )
    if action == int(GO_DOWN):
        return Pose5D(
            pose.x, pose.y, pose.z - float(config.vertical_step_m),
            pose.yaw, pose.pitch,
        )
    raise ValueError(f"unknown action: {action}")


def resample_polyline(
    points: np.ndarray | Sequence[Sequence[float]],
    step_m: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"expected [N,2] polyline, got {points.shape}")
    if len(points) <= 1:
        return points.copy()
    if step_m <= 0:
        raise ValueError("step_m must be positive")
    out = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        segment = float(np.linalg.norm(end - start))
        if segment < _EPS:
            continue
        count = max(1, int(math.ceil(segment / step_m)))
        for index in range(1, count + 1):
            out.append(start + (end - start) * (index / count))
    return np.asarray(out, dtype=np.float64)


def _straight_line_points(
    src: np.ndarray,
    dst: np.ndarray,
    stride_m: float = 5.0,
) -> list[np.ndarray]:
    """Official ``straight_line_trajectory`` point sampling."""
    distance = float(np.linalg.norm(dst - src))
    if distance < _EPS:
        return []
    stride_vector = (dst - src) / (distance + 0.01) * float(stride_m)
    count = int(distance / float(stride_m))
    points = [src + index * stride_vector for index in range(count)]
    points.append(dst)
    return points


def interpolate_demonstration_path(
    poses: Sequence[Pose5D],
    marker_xyz: Sequence[float],
    *,
    stride_m: float = 5.0,
) -> np.ndarray:
    """Build the official extended/interpolated demonstration path.

    The official teacher appends ``marker_positions[-1]`` to the demonstration
    and samples each segment every 5 m, keeping points more than 5 m from the
    previous kept point.  The marker, not ``target_positions[-1]``, is the
    teacher's stopping goal.
    """
    marker = np.asarray(marker_xyz, dtype=np.float64)
    if marker.shape != (3,):
        raise ValueError("marker_xyz must have shape [3]")
    extended = [pose.xyz for pose in poses] + [marker]
    out = [extended[0]]
    for src, dst in zip(extended[:-1], extended[1:]):
        for point in _straight_line_points(src, dst, stride_m=stride_m):
            if float(np.linalg.norm(point - out[-1])) > 5.0:
                out.append(point)
    return np.asarray(out, dtype=np.float64)


@dataclass
class TeacherResult:
    actions: list[int]
    poses: list[Pose5D]
    success: bool
    stop_reason: str
    final_target_distance_m: float
    final_marker_distance_m: float
    steps: int
    cross_track_median_m: float
    cross_track_p95_m: float
    cross_track_max_m: float
    yaw_error_median_deg: float
    stop_distance_m: float
    success_radius_m: float

    @property
    def final_distance_m(self) -> float:
        """Backward-compatible name for the evaluation-target distance."""
        return self.final_target_distance_m

    def supervision_steps(self) -> list[tuple[Pose5D, int]]:
        """Pair each state with the action taken from it."""
        return list(zip(self.poses, self.actions))

    def as_dict(self) -> dict:
        return {
            "success": bool(self.success),
            "stop_reason": self.stop_reason,
            "steps": int(self.steps),
            "final_target_distance_m": round(
                float(self.final_target_distance_m), 3
            ),
            "final_marker_distance_m": round(
                float(self.final_marker_distance_m), 3
            ),
            "stop_distance_m": round(float(self.stop_distance_m), 3),
            "success_radius_m": round(float(self.success_radius_m), 3),
            "cross_track_median_m": round(float(self.cross_track_median_m), 3),
            "cross_track_p95_m": round(float(self.cross_track_p95_m), 3),
            "cross_track_max_m": round(float(self.cross_track_max_m), 3),
            "yaw_error_median_deg": round(float(self.yaw_error_median_deg), 3),
            "action_histogram": {
                ACTION_NAMES[action]: int(self.actions.count(int(action)))
                for action in ACTION_NAMES
            },
        }


class TrajectoryAdapter:
    """Official CityNav lookahead teacher for the six-action interface."""

    def __init__(self, config: TeacherConfig | None = None):
        self.config = config or TeacherConfig()

    def rollout(
        self,
        poses: Sequence[Pose5D],
        marker_xyz: Sequence[float],
        target_xyz: Sequence[float] | None = None,
    ) -> TeacherResult:
        """Replay the official lookahead teacher.

        ``marker_xyz`` (``marker_positions[-1]``) is the teacher's stopping goal,
        appended to the demonstration before interpolation.  ``target_xyz``
        (``target_positions[-1]``) is the evaluation goal used only for the
        success radius.  They are intentionally not interchangeable.
        """
        if not poses:
            raise ValueError("poses must not be empty")
        cfg = self.config
        marker = np.asarray(marker_xyz, dtype=np.float64)
        if marker.shape != (3,):
            raise ValueError("marker_xyz must have shape [3]")
        evaluation_target = (
            marker
            if target_xyz is None
            else np.asarray(target_xyz, dtype=np.float64)
        )
        if evaluation_target.shape != (3,):
            raise ValueError("target_xyz must have shape [3]")

        demo_xyz = np.asarray([pose.xyz for pose in poses], dtype=np.float64)
        demo_xy = demo_xyz[:, :2]
        path = interpolate_demonstration_path(poses, marker)
        goal = path[-1]
        current = poses[0]
        trajectory = path.copy()
        offset = 0
        sim_poses: list[Pose5D] = [current]
        actions: list[int] = []
        stop_reason = "max_steps"

        for _ in range(int(cfg.max_steps)):
            distance_to_goal = float(np.linalg.norm(goal - current.xyz))
            if distance_to_goal < float(cfg.stop_distance_m):
                actions.append(int(STOP))
                stop_reason = "teacher_stop_distance"
                break
            if len(trajectory) == 0:
                actions.append(int(STOP))
                stop_reason = "empty_path"
                break

            distances = np.linalg.norm(trajectory - current.xyz, axis=1)
            closest_local = int(np.argmin(distances))
            next_local = min(closest_local + int(cfg.lookahead), len(trajectory) - 1)
            waypoint = trajectory[next_local]
            dx, dy, dz = waypoint - current.xyz
            horizontal = float(math.hypot(dx, dy))
            next_yaw = math.atan2(float(dy), float(dx))
            dyaw = (
                0.0
                if horizontal < 0.01
                else wrap_angle(next_yaw - current.yaw)
            )

            if dz > float(cfg.vertical_threshold_m):
                action = int(GO_UP)
            elif dz < -float(cfg.vertical_threshold_m):
                action = int(GO_DOWN)
            elif dyaw > float(cfg.turn_threshold_rad):
                action = int(TURN_LEFT)
            elif dyaw < -float(cfg.turn_threshold_rad):
                action = int(TURN_RIGHT)
            else:
                action = int(MOVE_FORWARD)

            if abs(dx) < _EPS and abs(dy) < _EPS and abs(dz) < _EPS:
                actions.append(int(STOP))
                stop_reason = "zero_displacement"
                break

            actions.append(action)
            moved = transition_pose(current, action, cfg)
            new_xyz = moved.xyz

            # No looking back: the next step only sees the remaining
            # interpolated demonstration from its nearest waypoint onward.
            remaining_distances = np.linalg.norm(trajectory - new_xyz, axis=1)
            nearest_local = int(np.argmin(remaining_distances))
            offset += nearest_local
            trajectory = path[offset:]
            nearest_raw = int(
                np.argmin(np.linalg.norm(demo_xyz - new_xyz, axis=1))
            )
            current = Pose5D(
                moved.x,
                moved.y,
                moved.z,
                moved.yaw,
                poses[nearest_raw].pitch,
            )
            sim_poses.append(current)

        else:
            # Safety limit reached.  The official loop is unbounded and always
            # appends a terminal STOP; mark the reason so this is not mistaken
            # for the 5 m teacher-stop rule.
            actions.append(int(STOP))
            stop_reason = "max_steps"

        if not actions:
            actions.append(int(STOP))

        final_pose = sim_poses[-1]
        final_target_distance = float(
            np.linalg.norm(evaluation_target - final_pose.xyz)
        )
        final_marker_distance = float(np.linalg.norm(goal - final_pose.xyz))
        success = final_target_distance < float(cfg.success_radius_m)

        sim_xy = np.asarray([[pose.x, pose.y] for pose in sim_poses])
        distance_matrix = np.linalg.norm(
            sim_xy[:, None, :] - demo_xy[None, :, :], axis=2
        )
        nearest_distances = distance_matrix.min(axis=1)
        nearest_indices = distance_matrix.argmin(axis=1)
        yaw_errors = [
            abs(wrap_angle(sim_poses[i].yaw - poses[int(nearest_indices[i])].yaw))
            for i in range(len(sim_poses))
        ]

        return TeacherResult(
            actions=actions,
            poses=sim_poses,
            success=bool(success),
            stop_reason=stop_reason,
            final_target_distance_m=final_target_distance,
            final_marker_distance_m=final_marker_distance,
            steps=len(actions),
            cross_track_median_m=float(np.median(nearest_distances)),
            cross_track_p95_m=float(np.percentile(nearest_distances, 95)),
            cross_track_max_m=float(np.max(nearest_distances)),
            yaw_error_median_deg=float(
                np.degrees(np.median(yaw_errors)) if yaw_errors else 0.0
            ),
            stop_distance_m=float(cfg.stop_distance_m),
            success_radius_m=float(cfg.success_radius_m),
        )


def action_names(actions: Iterable[int]) -> list[str]:
    return [ACTION_NAMES[int(action)] for action in actions]
