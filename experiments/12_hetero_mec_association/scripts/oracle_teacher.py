#!/usr/bin/env python3
"""Privileged teacher used only by the Oracle association baseline.

Definition (aligned with mec-v8.2 Stage-2k ``select_oracle``):
the teacher action ``a*`` is computed from the *true* target position and the
elevation obstacle map at the current state.  Oracle then picks the available
node whose hosted expert outputs exactly ``a*`` with the lowest instantaneous
E2E; if no expert agrees, it picks the node assigning the highest probability
to ``a*`` (E2E as tie-break).

The action rule mirrors the Stage-03B direct teacher, including its
two-threshold anti-chatter band: the cruise threshold (25 deg) applies whenever
the current heading error is already inside the 15 deg acquisition band, and
the acquisition threshold (15 deg) applies otherwise.  ``escape_dir`` is kept
across steps so a blocked heading is rotated out of consistently.

This module must never be imported by a deployable policy path.  The closed-loop
runner imports it only under ``--strategy oracle`` and records
``oracle_agree`` so the privileged information is auditable.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

sys.path.insert(0, str(P.REPO / "experiments/03b_target_directed_teacher/scripts"))
from direct_teacher import (CRUISE_TRIGGER_DEG, FORWARD, LOOKAHEAD_CELLS,  # noqa: E402
                            STOP, TURN_LEFT, TURN_RAD, TURN_RIGHT,
                            TURN_THRESHOLD, WAYPOINT_TOL, plan_path, wrap)
from obstacles import free  # noqa: E402


def _ray_free(info: dict, x0: float, y0: float, x1: float, y1: float,
              step: float = 0.75) -> bool:
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(dist / step) + 1)
    for i in range(n + 1):
        f = i / n
        if not free(info, x0 + f * (x1 - x0), y0 + f * (y1 - y0)):
            return False
    return True


class TeacherOracle:
    """Teacher action at the current true state (privileged)."""

    def __init__(self):
        self.calls = 0
        self.plan_modes: dict[str, int] = {}
        self.fallback = 0
        self.escape_dir = 0
        self.aligned = False

    def reset(self) -> None:
        self.escape_dir = 0
        self.aligned = False

    def action(self, info: dict, pos: np.ndarray, yaw: float,
               target: np.ndarray) -> int:
        self.calls += 1
        pos = np.asarray(pos, np.float64)
        target = np.asarray(target, np.float64)
        dist = float(np.linalg.norm(pos - target))
        if dist <= P.STOP_RADIUS:
            return STOP

        waypoints, mode, path_len, reason = plan_path(info, pos, target)
        self.plan_modes[mode] = self.plan_modes.get(mode, 0) + 1
        if waypoints is None:
            self.fallback += 1
            return self._fallback(info, pos, yaw, target)

        # aim at a look-ahead point of the currently planned path
        n = len(waypoints)
        j = int(np.argmin(np.linalg.norm(waypoints - pos[None, :], axis=1)))
        target_idx = min(n - 1, j + LOOKAHEAD_CELLS)
        aim = np.asarray(waypoints[target_idx], np.float64)
        if float(np.linalg.norm(aim - pos)) < WAYPOINT_TOL and target_idx < n - 1:
            aim = np.asarray(waypoints[target_idx + 1], np.float64)
        vec = aim - pos
        if float(np.hypot(*vec)) > 1e-9:
            bearing = math.atan2(vec[1], vec[0])
            dyaw = wrap(bearing - yaw)
        else:
            bearing, dyaw = yaw, 0.0

        nxt = pos + P.FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)])
        forward_free = _ray_free(info, float(pos[0]), float(pos[1]),
                                 float(nxt[0]), float(nxt[1]), step=0.5)
        self.last_forward_free = forward_free

        if self.escape_dir:
            if forward_free:
                self.escape_dir = 0
                self.aligned = True
                return FORWARD
            action = TURN_LEFT if self.escape_dir > 0 else TURN_RIGHT
            tyaw = wrap(yaw + self.escape_dir * TURN_RAD)
            self.aligned = abs(wrap(bearing - tyaw)) <= TURN_THRESHOLD + 1e-9
            return action

        # both thresholds are radians (TURN_THRESHOLD is already radians)
        trigger = (math.radians(CRUISE_TRIGGER_DEG) if self.aligned
                   else TURN_THRESHOLD)
        self.last_dyaw = dyaw
        self.last_trigger = trigger
        if abs(dyaw) > trigger + 1e-9:
            action = TURN_LEFT if dyaw > 0 else TURN_RIGHT
            tyaw = wrap(yaw + (TURN_RAD if action == TURN_LEFT else -TURN_RAD))
            self.aligned = abs(wrap(bearing - tyaw)) <= TURN_THRESHOLD + 1e-9
            return action
        if not forward_free:
            best_err, best_heading = math.inf, yaw
            for k in range(12):
                heading = wrap(k * TURN_RAD)
                fx = pos[0] + P.FORWARD_M * math.cos(heading)
                fy = pos[1] + P.FORWARD_M * math.sin(heading)
                if not _ray_free(info, float(pos[0]), float(pos[1]),
                                 float(fx), float(fy), step=0.5):
                    continue
                err = abs(wrap(heading - bearing))
                if err < best_err:
                    best_err, best_heading = err, heading
            self.escape_dir = 1 if wrap(best_heading - yaw) >= 0 else -1
            tyaw = wrap(yaw + self.escape_dir * TURN_RAD)
            self.aligned = abs(wrap(bearing - tyaw)) <= TURN_THRESHOLD + 1e-9
            return TURN_LEFT if self.escape_dir > 0 else TURN_RIGHT
        self.aligned = True
        return FORWARD

    @staticmethod
    def _fallback(info, pos, yaw, target) -> int:
        vec = np.asarray(target, np.float64) - np.asarray(pos, np.float64)
        if float(np.hypot(*vec)) <= P.STOP_RADIUS:
            return STOP
        bearing = math.atan2(vec[1], vec[0])
        dyaw = wrap(bearing - yaw)
        if abs(dyaw) > TURN_THRESHOLD:
            return TURN_LEFT if dyaw > 0 else TURN_RIGHT
        nxt = np.asarray(pos, np.float64) + P.FORWARD_M * np.asarray(
            [math.cos(yaw), math.sin(yaw)])
        if _ray_free(info, float(pos[0]), float(pos[1]),
                     float(nxt[0]), float(nxt[1]), step=0.5):
            return FORWARD
        return TURN_LEFT if dyaw >= 0 else TURN_RIGHT
