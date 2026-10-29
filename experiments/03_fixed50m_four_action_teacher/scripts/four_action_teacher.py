#!/usr/bin/env python3
"""Fixed-50 m four-action teacher for CityNav human trajectories.

The teacher follows the official lookahead construction (``lookahead=1``,
turn threshold 30 deg, forward 5 m) but restricts the action space to

    STOP / Forward 5 m / TurnLeft 30 deg / TurnRight 30 deg

and, per the fixed-50 m protocol, uses the horizontal distance to
``target_positions[-1]`` with a **20 m** STOP radius (the official success
radius).  It never emits GoUp/GoDown and never appends a synthetic STOP to a
failed rollout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STOP, FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3
ACTION_NAMES = {STOP: "Stop", FORWARD: "Forward", TURN_LEFT: "TurnLeft", TURN_RIGHT: "TurnRight"}
FORWARD_M = 5.0
TURN_RAD = math.radians(30.0)
TURN_THRESHOLD = math.radians(30.0)
STOP_RADIUS = 20.0
DUP_EPS = 0.05
WAYPOINT_STRIDE = 5.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clean_path(xy: np.ndarray) -> tuple[np.ndarray, int]:
    keep = [0]
    for i in range(1, len(xy)):
        if float(np.hypot(*(xy[i] - xy[keep[-1]]))) >= DUP_EPS:
            keep.append(i)
    return xy[keep], len(xy) - len(keep)


def build_waypoints(xy: np.ndarray, marker: np.ndarray, stride: float = WAYPOINT_STRIDE) -> np.ndarray:
    """Official ``interpolated_trajectory``: keep points >stride apart, end at marker."""
    chain = np.vstack([xy, marker[None, :]])
    way = [chain[0]]
    for src, dst in zip(chain[:-1], chain[1:]):
        delta = dst - src
        dist = float(np.linalg.norm(delta))
        if dist < 1e-9:
            continue
        n = int(dist / stride)
        candidates = [src + i * delta / dist * stride for i in range(n)] + [dst]
        for point in candidates:
            if float(np.linalg.norm(point - way[-1])) > stride:
                way.append(point)
    return np.asarray(way, dtype=np.float64)


def move(pos: np.ndarray, yaw: float, action: int) -> tuple[np.ndarray, float]:
    if action == FORWARD:
        return pos + FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)]), yaw
    if action == TURN_LEFT:
        return pos, wrap(yaw + TURN_RAD)
    if action == TURN_RIGHT:
        return pos, wrap(yaw - TURN_RAD)
    if action == STOP:
        return pos, yaw
    raise ValueError(f"unknown action {action}")


@dataclass
class Rollout:
    states: list[tuple[float, float, float]]
    actions: list[int]
    success: bool
    steps: int
    final_dist: float
    min_dist: float
    reason: str


def rollout(
    waypoints: np.ndarray,
    target: np.ndarray,
    start_yaw: float,
    path_len: float,
    max_steps: int | None = None,
) -> Rollout:
    """Replay the human path with the four-action teacher."""
    n = len(waypoints)
    pos = np.asarray(waypoints[0], dtype=np.float64).copy()
    yaw = float(start_yaw)
    if max_steps is None:
        max_steps = int(min(2000, max(120, 2.5 * path_len / FORWARD_M + 100)))
    states: list[tuple[float, float, float]] = []
    actions: list[int] = []
    min_dist = float(np.linalg.norm(pos - target))
    final_dist = min_dist
    while len(actions) < max_steps:
        dist = float(np.linalg.norm(pos - target))
        min_dist = min(min_dist, dist)
        final_dist = dist
        states.append((float(pos[0]), float(pos[1]), float(yaw)))
        if dist <= STOP_RADIUS:
            actions.append(STOP)
            return Rollout(states, actions, True, len(actions), dist, min_dist, "success")
        j = int(np.argmin(np.linalg.norm(waypoints - pos, axis=1)))
        nxt = min(n - 1, j + 1)
        vec = waypoints[nxt] - pos
        dyaw = wrap(math.atan2(vec[1], vec[0]) - yaw) if np.hypot(*vec) > 1e-6 else 0.0
        if dyaw > TURN_THRESHOLD:
            action = TURN_LEFT
        elif dyaw < -TURN_THRESHOLD:
            action = TURN_RIGHT
        else:
            action = FORWARD
        actions.append(action)
        pos, yaw = move(pos, yaw, action)
    return Rollout(
        states, actions, False, len(actions), final_dist, min_dist,
        "step_limit" if dist > STOP_RADIUS else "success_without_stop",
    )


def convert_episode(record: dict, ground_level: float | None) -> dict:
    """Convert one CityNav record into fixed-height four-action labels."""
    name = f"{record['area']}_block_{record['block']}"
    traj = np.asarray(record["trajectory"], dtype=np.float64)
    xy_raw = traj[:, :2]
    target = np.asarray(record["target_positions"][-1], dtype=np.float64)[:2]
    marker = np.asarray(record["marker_positions"][-1], dtype=np.float64)[:2]
    xy, removed = clean_path(xy_raw)
    if len(xy) < 2:
        return {
            "map": name, "object_ids": record["object_ids"], "ann_ids": record["ann_ids"],
            "degenerate": True, "success": False, "reason": "degenerate",
            "states": [], "actions": [], "waypoints": xy,
            "path_len_m": 0.0, "duplicates_removed": removed,
            "start_target_m": float(np.linalg.norm(xy[0] - target)) if len(xy) else 0.0,
            "final_target_m": float(np.linalg.norm(xy[-1] - target)) if len(xy) else 0.0,
            "min_target_m": float(np.linalg.norm(xy - target, axis=1).min()) if len(xy) else 0.0,
        }
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_len = float(seg.sum())
    dists = np.linalg.norm(xy - target, axis=1)
    entered = bool(dists.min() <= STOP_RADIUS)
    start_yaw = (
        math.atan2(traj[0, 4], traj[0, 3])
        if (abs(traj[0, 3]) + abs(traj[0, 4])) > 1e-9
        else math.atan2(*(xy[1] - xy[0])[::-1])
    )
    waypoints = build_waypoints(xy, marker)
    result = rollout(waypoints, target, start_yaw, path_len)
    if not result.success:
        reason = "never_entered_20m" if not entered else result.reason
    else:
        reason = "success"
    return {
        "map": name,
        "object_ids": record["object_ids"],
        "ann_ids": record["ann_ids"],
        "degenerate": False,
        "success": result.success,
        "reason": reason,
        "states": result.states,
        "actions": result.actions,
        "waypoints": waypoints,
        "path_len_m": path_len,
        "duplicates_removed": removed,
        "start_target_m": float(np.linalg.norm(xy[0] - target)),
        "final_target_m": float(np.linalg.norm(xy[-1] - target)),
        "min_target_m": float(dists.min()),
        "entered_20m": entered,
        "teacher_steps": result.steps,
        "teacher_final_dist": result.final_dist,
        "teacher_min_dist": result.min_dist,
        "ground_level_m": ground_level,
        "fixed_z_m": (ground_level + 50.0) if ground_level is not None else None,
        "start_yaw": float(start_yaw),
        "target_xy": target.tolist(),
        "marker_xy": marker.tolist(),
    }
