#!/usr/bin/env python3
"""Direct target-directed four-action teacher.

The task is defined by the UAV start pose and the final target only.  Human
trajectories contribute **only** the start xy/yaw; the target xy is used only to
build offline action labels.  Nothing from the human path (no intermediate
points, no marker, no language) constrains navigation.

Obstacle avoidance comes from the elevation raster (``obstacles.py``): a
straight segment is used whenever it is free at the fixed 50 m altitude; when
the straight line crosses a cell whose surface reaches the flight altitude, an
A* detour on the free cells is used instead.

Action rules (identical in straight and detour mode):

1. horizontal distance to the target <= 20 m  -> STOP
2. |heading error| > 15 deg                    -> TurnLeft / TurnRight 30 deg
3. otherwise                                   -> Forward 5 m

Anti-chatter deadband
---------------------
The plain rule above is unstable when the turn quantum is 30 deg and the
acceptance band is 15 deg:  after a turn the residual error is |dyaw| - 30 deg,
which can land *just* inside 15 deg (e.g. 14.9 deg); one 5 m forward step moves
the bearing by a fraction of a degree, pushes the error back over 15 deg, and
the teacher turns back.  This yields the 2-step limit cycle L F R F L F ... and
inflated the turn share to 32 % of all actions (Train-Seen).

The two-threshold (Schmitt trigger) rule fixes it while keeping the 15 deg
alignment target:

* not aligned  : turn when |dyaw| > 15 deg   (acquisition, exactly the rule above)
* aligned      : turn when |dyaw| > CRUISE_TRIGGER_DEG (default 30 deg)

so a heading that is already aligned is only corrected once it has drifted
beyond the turn quantum again.  ``cruise_trigger_deg=None`` reproduces the
naive rule for comparison.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from obstacles import cell_to_world, free, world_to_cell

STOP, FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3
ACTION_NAMES = {STOP: "Stop", FORWARD: "Forward", TURN_LEFT: "TurnLeft", TURN_RIGHT: "TurnRight"}
FORWARD_M = 5.0
TURN_RAD = math.radians(30.0)
TURN_THRESHOLD = math.radians(15.0)
STOP_RADIUS = 20.0
WAYPOINT_TOL = 2.0
LOOKAHEAD_CELLS = 8      # aim 8 m further along the path (1 m cells)
CRUISE_TRIGGER_DEG = 25.0  # re-trigger a turn only after drifting this far (see docstring)

_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
               (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _octile(ar: int, ac: int, br: int, bc: int) -> float:
    dr, dc = abs(ar - br), abs(ac - bc)
    return (dr + dc) + (math.sqrt(2) - 2.0) * min(dr, dc)


def astar(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
          max_expand: int = 400_000) -> list[tuple[int, int]] | None:
    """8-connected A* on the 1 m free grid; diagonals need both cardinals free."""
    h, w = blocked.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None
    if blocked[sr, sc] or blocked[gr, gc]:
        return None
    if (sr, sc) == (gr, gc):
        return [(sr, sc)]
    open_heap = [(0.0, 0, (sr, sc))]
    g_score = {(sr, sc): 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    closed = np.zeros((h, w), bool)
    counter = 0
    expanded = 0
    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        r, c = cur
        if closed[r, c]:
            continue
        closed[r, c] = True
        expanded += 1
        if cur == (gr, gc):
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if expanded > max_expand:
            return None
        for dr, dc, cost in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or blocked[nr, nc] or closed[nr, nc]:
                continue
            if dr and dc and (blocked[r + dr, c] or blocked[r, c + dc]):
                continue
            tentative = g_score[(r, c)] + cost
            if tentative < g_score.get((nr, nc), math.inf):
                g_score[(nr, nc)] = tentative
                came[(nr, nc)] = cur
                counter += 1
                heapq.heappush(open_heap, (tentative + _octile(nr, nc, gr, gc),
                                           counter, (nr, nc)))
    return None


def _ray_free(info: dict, x0: float, y0: float, x1: float, y1: float,
              step: float = 0.75) -> bool:
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(dist / step) + 1)
    for i in range(n + 1):
        f = i / n
        if not free(info, x0 + f * (x1 - x0), y0 + f * (y1 - y0)):
            return False
    return True


def plan_path(info: dict, start_xy: np.ndarray, goal_xy: np.ndarray):
    """Return (waypoints, mode, path_len, reason).

    Straight line when free; otherwise A* detour on the elevation obstacle grid.
    """
    sx, sy = float(start_xy[0]), float(start_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    if not free(info, sx, sy):
        return None, "start_blocked", 0.0, "start_blocked"
    if not free(info, gx, gy):
        return None, "goal_blocked", 0.0, "goal_blocked"
    if _ray_free(info, sx, sy, gx, gy):
        length = math.hypot(gx - sx, gy - sy)
        return np.array([[sx, sy], [gx, gy]], np.float64), "straight", length, "ok"
    blocked = info["blocked"]
    s = world_to_cell(info, sx, sy)
    g = world_to_cell(info, gx, gy)
    path = astar(blocked, s, g)
    if path is None:
        return None, "no_path", 0.0, "no_path"
    world = np.array([cell_to_world(info, r, c) for r, c in path], np.float64)
    world[0] = [sx, sy]
    world[-1] = [gx, gy]
    seg = np.linalg.norm(np.diff(world, axis=0), axis=1)
    return world, "astar", float(seg.sum()), "ok"


def rollout(info: dict, start_xy: np.ndarray, start_yaw: float, goal_xy: np.ndarray,
            waypoints: np.ndarray, path_len: float, max_steps: int | None = None,
            cruise_trigger_deg: float | None = CRUISE_TRIGGER_DEG):
    """Generate STOP / Forward 5 m / TurnLeft 30 deg / TurnRight 30 deg labels.

    ``cruise_trigger_deg=None`` disables the anti-chatter deadband and
    reproduces the naive 15 deg rule (kept only to quantify the defect).
    """
    if max_steps is None:
        max_steps = int(min(2000, max(200, 4.0 * path_len / FORWARD_M + 200)))
    cruise = None if cruise_trigger_deg is None else math.radians(cruise_trigger_deg)
    aligned = False         # True once the heading error is inside 15 deg
    pos = np.asarray(start_xy, np.float64).copy()
    yaw = float(start_yaw)
    states: list[tuple[float, float, float]] = []
    actions: list[int] = []
    min_dist = float(np.linalg.norm(pos - goal_xy))
    blocked_forward = 0
    escape_dir = 0          # -1 / +1 while rotating out of a blocked heading
    idx = 0
    n = len(waypoints)
    while len(actions) < max_steps:
        dist = float(np.linalg.norm(pos - goal_xy))
        min_dist = min(min_dist, dist)
        states.append((float(pos[0]), float(pos[1]), float(yaw)))
        if dist <= STOP_RADIUS:
            actions.append(STOP)
            return {"success": True, "reason": "success", "states": states,
                    "actions": actions, "min_dist": min_dist,
                    "final_dist": dist, "blocked_forward": blocked_forward,
                    "steps": len(actions)}
        # follow the path: re-anchor on the nearest waypoint of a small window,
        # then aim LOOKAHEAD_M further along it (a 2-point straight path makes
        # this identical to the direct rule)
        lo = max(0, idx - 2)
        hi = min(n, idx + 20)
        j = lo + int(np.argmin(np.linalg.norm(waypoints[lo:hi] - pos[None, :], axis=1)))
        idx = max(idx, j)                     # progress anchor never goes back
        target_idx = min(n - 1, idx + LOOKAHEAD_CELLS)
        target = np.asarray(waypoints[target_idx], np.float64)
        if float(np.linalg.norm(target - pos)) < WAYPOINT_TOL and target_idx < n - 1:
            target = np.asarray(waypoints[target_idx + 1], np.float64)
        vec = target - pos
        if np.hypot(*vec) > 1e-9:
            bearing = math.atan2(vec[1], vec[0])
            dyaw = wrap(bearing - yaw)
        else:
            bearing, dyaw = yaw, 0.0
        nxt = pos + FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)])
        forward_free = _ray_free(info, float(pos[0]), float(pos[1]),
                                 float(nxt[0]), float(nxt[1]), step=0.5)
        if escape_dir:
            # keep rotating the same way until a forward step is possible;
            # re-deciding every step would oscillate around a blocked heading
            if forward_free:
                # take the step immediately: re-checking the heading error first
                # would turn the UAV back into the blocked heading and pin it
                # against the map edge (observed on 59 Train-Seen tasks)
                escape_dir = 0
                actions.append(FORWARD)
                pos = nxt
                aligned = abs(wrap(bearing - yaw)) <= TURN_THRESHOLD + 1e-9
                continue
            else:
                actions.append(TURN_LEFT if escape_dir > 0 else TURN_RIGHT)
                yaw = wrap(yaw + escape_dir * TURN_RAD)
                aligned = abs(wrap(bearing - yaw)) <= TURN_THRESHOLD + 1e-9
                continue
        trigger = cruise if (aligned and cruise is not None) else TURN_THRESHOLD
        if abs(dyaw) > trigger + 1e-9:
            action = TURN_LEFT if dyaw > 0 else TURN_RIGHT
            actions.append(action)
            yaw = wrap(yaw + (TURN_RAD if action == TURN_LEFT else -TURN_RAD))
            aligned = abs(wrap(bearing - yaw)) <= TURN_THRESHOLD + 1e-9
            continue
        elif not forward_free:
            blocked_forward += 1
            # pick the free heading closest to the target bearing and keep
            # rotating that way (see above)
            best_err, best_heading = math.inf, yaw
            for k in range(12):
                heading = wrap(k * TURN_RAD)
                fx = pos[0] + FORWARD_M * math.cos(heading)
                fy = pos[1] + FORWARD_M * math.sin(heading)
                if not _ray_free(info, float(pos[0]), float(pos[1]),
                                 float(fx), float(fy), step=0.5):
                    continue
                err = abs(wrap(heading - bearing))
                if err < best_err:
                    best_err, best_heading = err, heading
            escape_dir = 1 if wrap(best_heading - yaw) >= 0 else -1
            actions.append(TURN_LEFT if escape_dir > 0 else TURN_RIGHT)
            yaw = wrap(yaw + escape_dir * TURN_RAD)
            aligned = abs(wrap(bearing - yaw)) <= TURN_THRESHOLD + 1e-9
            continue
        actions.append(FORWARD)
        pos = nxt
        aligned = True
    return {"success": False, "reason": "step_limit", "states": states,
            "actions": actions, "min_dist": min_dist,
            "final_dist": float(np.linalg.norm(pos - goal_xy)),
            "blocked_forward": blocked_forward, "steps": len(actions)}


def start_pose(record: dict) -> tuple[np.ndarray, float]:
    traj = np.asarray(record["trajectory"], np.float64)
    xy = traj[0, :2]
    if (abs(traj[0, 3]) + abs(traj[0, 4])) > 1e-9:
        yaw = math.atan2(traj[0, 4], traj[0, 3])
    else:
        yaw = math.atan2(*(traj[1, :2] - traj[0, :2])[::-1])
    return xy, float(yaw)


def run_episode(record: dict, info: dict,
                cruise_trigger_deg: float | None = CRUISE_TRIGGER_DEG) -> dict:
    name = f"{record['area']}_block_{record['block']}"
    start_xy, start_yaw = start_pose(record)
    target = np.asarray(record["target_positions"][-1], np.float64)[:2]
    start_target = float(np.linalg.norm(start_xy - target))
    waypoints, mode, path_len, reason = plan_path(info, start_xy, target)
    if waypoints is None:
        return {"map": name, "success": False, "reason": reason, "mode": mode,
                "start_xy": start_xy.tolist(), "start_yaw": start_yaw,
                "target_xy": target.tolist(), "start_target_m": start_target,
                "path_len_m": 0.0, "states": [], "actions": [],
                "min_dist": start_target, "final_dist": start_target,
                "steps": 0, "blocked_forward": 0}
    out = rollout(info, start_xy, start_yaw, target, waypoints, path_len,
                  cruise_trigger_deg=cruise_trigger_deg)
    out.update({"map": name, "mode": mode,
                "start_xy": start_xy.tolist(), "start_yaw": start_yaw,
                "target_xy": target.tolist(), "start_target_m": start_target,
                "path_len_m": path_len, "waypoints": waypoints.tolist()})
    return out
