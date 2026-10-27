"""CityNav trajectory -> five-dimensional pose -> discrete-action adapter.

CityNav stores each trajectory point as six numbers::

    x, y, z, dx, dy, dz

The official conversion (``citynav-official/vlnce/space.py``) is::

    yaw   = atan2(dy, dx)
    pitch = atan2(dz, sqrt(dx^2 + dy^2))

The six numbers are *not* six actions.  This module converts them to a
five-dimensional pose and then to the four-action MEC interface
``Forward / Left / Right / Stop`` using a pure-pursuit teacher over the human
demonstration.  The teacher's simulated poses are compared with the original
demonstration path, so the action labels are auditable.

Degenerate directions
---------------------
Many CityNav points look almost straight down (pitch near +/-90 degrees), so
``dx`` and ``dy`` are nearly zero and yaw from the direction vector is
undefined.  In that case we fall back to the local motion direction, then to
the previous yaw.  This is a deterministic handling of the documented
degeneracy, not a change of the coordinate convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.config import TeacherConfig


FORWARD = 0
LEFT = 1
RIGHT = 2
STOP = 3

ACTION_NAMES = {
    FORWARD: "Forward",
    LEFT: "Left",
    RIGHT: "Right",
    STOP: "Stop",
}
NUM_ACTIONS = 4
NUM_MOVE_ACTIONS = 3

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
    def xy(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float64)


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


def load_trajectory(
    raw_trajectory: Sequence[Sequence[float]],
    *,
    fix_altitude_m: float | None = None,
    ground_level_m: float | None = None,
) -> list[Pose5D]:
    """Convert CityNav six-number points to :class:`Pose5D`.

    If ``fix_altitude_m`` and ``ground_level_m`` are given, all z values are set
    to ``ground_level_m + fix_altitude_m``.  This matches the fixed-altitude
    MEC navigation setting; the original z is preserved by the caller for audit.
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
        if fix_altitude_m is not None:
            if ground_level_m is None:
                raise ValueError("ground_level_m is required when fixing altitude")
            z = float(ground_level_m) + float(fix_altitude_m)
        poses.append(Pose5D(x, y, z, yaw, pitch))
        previous_yaw = yaw
    return poses


def transition_pose(
    pose: Pose5D,
    action: int,
    config: TeacherConfig,
) -> Pose5D:
    if action == STOP:
        return pose
    if action == FORWARD:
        x = pose.x + config.forward_step_m * math.cos(pose.yaw)
        y = pose.y + config.forward_step_m * math.sin(pose.yaw)
        return Pose5D(x, y, pose.z, pose.yaw, pose.pitch)
    if action == LEFT:
        return Pose5D(
            pose.x, pose.y, pose.z,
            wrap_angle(pose.yaw + config.turn_step_rad), pose.pitch,
        )
    if action == RIGHT:
        return Pose5D(
            pose.x, pose.y, pose.z,
            wrap_angle(pose.yaw - config.turn_step_rad), pose.pitch,
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


@dataclass
class TeacherResult:
    actions: list[int]
    poses: list[Pose5D]
    success: bool
    stop_reason: str
    final_distance_m: float
    steps: int
    cross_track_median_m: float
    cross_track_p95_m: float
    cross_track_max_m: float
    yaw_error_median_deg: float

    def supervision_steps(self) -> list[tuple[Pose5D, int]]:
        """Pair each executed pose with its action.

        Successful rollouts have ``len(actions) == len(poses)``; the final pair
        is the Stop action at the stopping pose.
        """
        return list(zip(self.poses, self.actions))

    def as_dict(self) -> dict:
        return {
            "success": bool(self.success),
            "stop_reason": self.stop_reason,
            "steps": int(self.steps),
            "final_distance_m": round(float(self.final_distance_m), 3),
            "cross_track_median_m": round(float(self.cross_track_median_m), 3),
            "cross_track_p95_m": round(float(self.cross_track_p95_m), 3),
            "cross_track_max_m": round(float(self.cross_track_max_m), 3),
            "yaw_error_median_deg": round(float(self.yaw_error_median_deg), 3),
            "action_histogram": {
                ACTION_NAMES[index]: int(self.actions.count(index))
                for index in range(NUM_ACTIONS)
            },
        }


class TrajectoryAdapter:
    """Pure-pursuit teacher for the four-action MEC interface."""

    def __init__(self, config: TeacherConfig | None = None):
        self.config = config or TeacherConfig()

    def rollout(
        self,
        poses: Sequence[Pose5D],
        target_xy: Sequence[float],
    ) -> TeacherResult:
        if not poses:
            raise ValueError("poses must not be empty")
        cfg = self.config
        target = np.asarray(target_xy, dtype=np.float64)
        if target.shape != (2,):
            raise ValueError("target_xy must have shape [2]")

        demo_xy = np.asarray([[pose.x, pose.y] for pose in poses], dtype=np.float64)
        path = resample_polyline(demo_xy, cfg.resample_step_m)
        segment_lengths = (
            np.linalg.norm(np.diff(path, axis=0), axis=1)
            if len(path) > 1
            else np.zeros(0, dtype=np.float64)
        )
        cumulative = np.concatenate(
            [np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths)]
        )
        total_length = float(cumulative[-1])

        start = poses[0]
        sim_poses: list[Pose5D] = [start]
        actions: list[int] = []
        progress = 0
        success = False
        stop_reason = "max_steps"

        for _ in range(int(cfg.max_steps)):
            current = sim_poses[-1]
            current_xy = current.xy
            distance_to_target = float(np.linalg.norm(target - current_xy))
            if distance_to_target <= float(cfg.stop_radius_m):
                actions.append(STOP)
                success = True
                stop_reason = "target_radius"
                break

            remaining = path[progress:]
            if len(remaining) == 0:
                stop_reason = "empty_path"
                break
            nearest_offset = int(
                np.argmin(np.linalg.norm(remaining - current_xy, axis=1))
            )
            progress = min(progress + nearest_offset, len(path) - 1)

            waypoint_index = int(
                np.searchsorted(
                    cumulative,
                    cumulative[progress] + float(cfg.lookahead_m),
                    side="left",
                )
            )
            waypoint_index = min(waypoint_index, len(path) - 1)
            if (
                cumulative[waypoint_index] >= total_length - float(cfg.lookahead_m)
                or waypoint_index == len(path) - 1
            ):
                waypoint = target
            else:
                waypoint = path[waypoint_index]

            delta = waypoint - current_xy
            if float(np.linalg.norm(delta)) < 1e-6:
                progress = min(progress + 1, len(path) - 1)
                continue
            bearing = math.atan2(float(delta[1]), float(delta[0]))
            yaw_error = wrap_angle(bearing - current.yaw)
            if abs(yaw_error) <= float(cfg.turn_threshold_rad):
                action = FORWARD
            elif yaw_error > 0:
                action = LEFT
            else:
                action = RIGHT
            actions.append(action)
            sim_poses.append(transition_pose(current, action, cfg))

        final_distance = float(
            np.linalg.norm(target - sim_poses[-1].xy)
        )

        sim_xy = np.asarray([[pose.x, pose.y] for pose in sim_poses])
        nearest_distances = np.min(
            np.linalg.norm(sim_xy[:, None, :] - demo_xy[None, :, :], axis=2),
            axis=1,
        )
        nearest_indices = np.argmin(
            np.linalg.norm(sim_xy[:, None, :] - demo_xy[None, :, :], axis=2),
            axis=1,
        )
        yaw_errors = [
            abs(wrap_angle(sim_poses[i].yaw - poses[int(nearest_indices[i])].yaw))
            for i in range(len(sim_poses))
        ]

        return TeacherResult(
            actions=actions,
            poses=sim_poses,
            success=success,
            stop_reason=stop_reason,
            final_distance_m=final_distance,
            steps=len(actions),
            cross_track_median_m=float(np.median(nearest_distances)),
            cross_track_p95_m=float(np.percentile(nearest_distances, 95)),
            cross_track_max_m=float(np.max(nearest_distances)),
            yaw_error_median_deg=float(
                np.degrees(np.median(yaw_errors)) if yaw_errors else 0.0
            ),
        )


def action_names(actions: Iterable[int]) -> list[str]:
    return [ACTION_NAMES[int(action)] for action in actions]
