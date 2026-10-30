#!/usr/bin/env python3
"""09B environment: online visual matching + dynamic candidate posterior.

Difference from Stage 08 (static geometry):

* the controller acts on the MAP of a candidate posterior, not on the top-1
  similarity candidate;
* the posterior starts from the goal-image/candidate similarity and is updated
  at each step by the online visual match (layer2 dense matching, the 09A
  winner), in log space, with an evidence power and a correlated-frame discount;
* updates only happen when the match passes the calibrated confidence and is
  spatially consistent with the current map estimate.

The true target is used only to build the goal image and to score the episode.
The controller never sees it; the belief receives only candidate positions and
the visual match offset.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "06_closed_loop_success_rate/scripts"))
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))

import geom  # noqa: E402
import matching  # noqa: E402
from closed_loop import (CLASSES, FORWARD_M, STOP_RADIUS, TURN_RAD, classify,  # noqa: E402
                         obstacle_for, segment_status)
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from pipeline import TARGET_HALF_M  # noqa: E402

ALIGN_RAD = math.radians(25.0)
DEFAULT_STEP_CAP = 200
CONSISTENCY_M = 60.0


class DynamicBelief:
    """Log-space posterior over the 24 m candidates."""

    def __init__(self, positions: np.ndarray, sims: np.ndarray, temperature: float):
        self.positions = positions
        self.logp = sims / temperature
        self.logp -= self.logp.max()
        self.logp -= math.log(np.exp(self.logp).sum())
        self.last_update = -99
        self.last_conf_pos = None

    def entropy(self) -> float:
        p = np.exp(self.logp)
        return float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p)))

    def map_index(self) -> int:
        return int(np.argmax(self.logp))

    def gate(self, r_vis: np.ndarray, agree_m: float) -> bool:
        """Two consecutive confident matches must agree before an update."""
        if agree_m <= 0:
            self.last_conf_pos = r_vis
            return True
        ok = (self.last_conf_pos is not None
              and float(np.linalg.norm(r_vis - self.last_conf_pos)) <= agree_m)
        self.last_conf_pos = r_vis
        return ok

    def update(self, r_vis: np.ndarray, uav_xy: np.ndarray, sigma: float,
               eta: float, rho: float, step: int) -> bool:
        d = (self.positions - r_vis)
        d2 = np.einsum("ij,ij->i", d, d)
        ll = -0.5 * d2 / (sigma * sigma)
        w = eta * (rho if step - self.last_update <= 1 else 1.0)
        self.logp = self.logp + w * (ll - ll.max())
        self.logp -= math.log(np.exp(self.logp).sum())
        self.last_update = step
        return True


def run_episode(task: dict, fe: FrozenEncoder, matcher: matching.Encoder,
                rasters: RasterStore, cands: CandidateStore, goal_cache: dict,
                goal_feat_cache: dict, params: dict, step_cap: int = DEFAULT_STEP_CAP):
    started = time.time()
    name = task["map"]
    raster = rasters.get(name)
    lib = cands.get(name)
    info = obstacle_for(name)
    pos = np.asarray([float(task["start_x"]), float(task["start_y"])], np.float64)
    yaw = float(task["start_yaw"])
    target = np.asarray([float(task["target_x"]), float(task["target_y"])], np.float64)

    tvec = fe.embed([raster.north_crop(float(target[0]), float(target[1]),
                                       TARGET_HALF_M)])[0].astype(np.float32)
    sims = lib.embeddings @ tvec
    belief = DynamicBelief(lib.positions.astype(np.float64), sims.astype(np.float64),
                           params["temperature"])

    gkey = (name, round(float(target[0]), 2), round(float(target[1]), 2))
    if gkey not in goal_cache:
        goal_cache[gkey] = matching.prepare_goal(raster, target[0], target[1])
    if gkey not in goal_feat_cache:
        goal_feat_cache[gkey] = matcher.features(goal_cache[gkey][None], ("layer2",))["layer2"][0]

    steps, updates = [], 0
    stopped = collision = out_of_bounds = False
    stop_dist = None
    min_dist = float(np.linalg.norm(pos - target))
    min_step = 0
    entropies = []

    for step in range(step_cap):
        dist = float(np.linalg.norm(pos - target))
        if dist < min_dist:
            min_dist, min_step = dist, step

        obs = matching.prepare_observation(raster, pos[0], pos[1], yaw)
        of = matcher.features(obs[None], ("layer2",))["layer2"][0]
        col, row, score, margin = matching.match_dense(of, goal_feat_cache[gkey], 2)
        dx, dy = matching.pixel_to_offset(col, row)
        r_vis = np.array([pos[0] + dx, pos[1] + dy])
        confident = (score >= params["tau"]) and (margin >= params["margin_tau"])
        map_i = belief.map_index()
        map_pos = belief.positions[map_i]
        consistent = float(np.linalg.norm(map_pos - r_vis)) <= CONSISTENCY_M
        updated = False
        if confident and consistent and belief.gate(r_vis, params.get("agree_m", 0.0)):
            belief.update(r_vis, pos, params["sigma"], params["eta"], params["rho"], step)
            updates += 1
            updated = True
            map_i = belief.map_index()
            map_pos = belief.positions[map_i]
        ent = belief.entropy()
        entropies.append(ent)

        vec = map_pos - pos
        distance = float(np.linalg.norm(vec))
        bearing = math.atan2(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw),
                             vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
        if distance <= STOP_RADIUS:
            action = 0
        elif bearing > ALIGN_RAD:
            action = 2
        elif bearing < -ALIGN_RAD:
            action = 3
        else:
            action = 1

        steps.append({"step": step, "x": float(pos[0]), "y": float(pos[1]),
                      "yaw": float(yaw), "action": int(action),
                      "map_candidate": [float(map_pos[0]), float(map_pos[1])],
                      "est_distance_m": distance,
                      "est_bearing_deg": math.degrees(bearing),
                      "visual_score": float(score), "visual_margin": float(margin),
                      "visual_offset_m": [float(dx), float(dy)],
                      "confident": bool(confident), "consistent": bool(consistent),
                      "updated": bool(updated), "entropy": ent,
                      "true_dist_m": dist})

        if action == 0:
            stopped = True
            stop_dist = dist
            break
        if action == 1:
            nxt = pos + FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)])
            status = segment_status(info, float(pos[0]), float(pos[1]),
                                    float(nxt[0]), float(nxt[1]))
            if status == "collision":
                collision = True
                steps[-1]["blocked"] = "collision"
                break
            if status == "out_of_bounds":
                out_of_bounds = True
                steps[-1]["blocked"] = "out_of_bounds"
                break
            pos = nxt
        elif action == 2:
            yaw = (yaw + TURN_RAD + math.pi) % (2 * math.pi) - math.pi
        elif action == 3:
            yaw = (yaw - TURN_RAD + math.pi) % (2 * math.pi) - math.pi

    final_dist = float(np.linalg.norm(pos - target))
    osr = bool(min_dist <= STOP_RADIUS)
    success = bool(stopped and stop_dist is not None and stop_dist <= STOP_RADIUS
                   and final_dist <= STOP_RADIUS)
    record = {
        "episode_id": task["episode_id"], "split": task["split"], "index": int(task["index"]),
        "map": name, "length_group": task["length_group"], "cohort": task["cohort"],
        "controller": "dynamic_visual",
        "start": [float(task["start_x"]), float(task["start_y"]), float(task["start_yaw"])],
        "target": [float(task["target_x"]), float(task["target_y"])],
        "step_cap": int(step_cap), "steps": len(steps),
        "stopped": stopped, "stop_step": (len(steps) - 1) if stopped else None,
        "stop_dist": stop_dist, "final_pos": [float(pos[0]), float(pos[1]), float(yaw)],
        "final_dist": final_dist, "min_dist": min_dist, "min_dist_step": int(min_step),
        "osr": osr, "success": success, "collision": collision,
        "out_of_bounds": out_of_bounds,
        "path_len_m": FORWARD_M * sum(1 for s in steps if s["action"] == 1),
        "teacher_path_len_m": float(task.get("teacher_path_len_m") or 0.0),
        "action_counts": {CLASSES[i]: int(sum(1 for s in steps if s["action"] == i))
                          for i in range(4)},
        "goal_image_invalid": bool(task.get("goal_image_invalid", False)),
        "target_valid_fraction": float(task.get("target_valid_fraction", 1.0)),
        "belief_updates": updates,
        "mean_entropy": float(np.mean(entropies)) if entropies else None,
        "final_entropy": float(entropies[-1]) if entropies else None,
        "trajectory": steps,
        "seconds": round(time.time() - started, 3),
        "manifest_sha": task.get("manifest_sha"),
    }
    record["failure"] = classify(record)
    return record


def record_summary(record: dict) -> dict:
    keys = ("episode_id", "split", "index", "map", "length_group", "cohort", "controller",
            "step_cap", "steps", "stopped", "stop_step", "stop_dist", "final_dist",
            "min_dist", "min_dist_step", "osr", "success", "failure", "collision",
            "out_of_bounds", "path_len_m", "teacher_path_len_m", "action_counts",
            "goal_image_invalid", "target_valid_fraction", "belief_updates",
            "mean_entropy", "final_entropy", "seconds", "manifest_sha")
    return {k: record.get(k) for k in keys}
