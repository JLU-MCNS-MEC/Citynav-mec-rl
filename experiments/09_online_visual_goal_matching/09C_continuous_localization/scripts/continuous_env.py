#!/usr/bin/env python3
"""09C closed loop: continuous Gaussian-mixture belief + observability gating.

Two modes share everything except the evidence gate:

* ``c1`` - continuous localisation, gates as in Stage 09B
  (score >= 0.875, two-frame world agreement <= 10 m);
* ``c2`` - adds the Train-Seen hard-negative gate: score >= tau2, margin >= m2,
  two-frame agreement <= agree_m, raster validity at the match >= v_min,
  >=12 px from the observation border, and Mahalanobis(z, best component) <= d_max.

The estimate is the MAP component's continuous mean, so a target between two
24 m candidates keeps its sub-grid offset.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/06_closed_loop_success_rate/scripts"))
sys.path.insert(0, str(ROOT / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(ROOT / "experiments/04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "experiments/09_online_visual_goal_matching/scripts"))

import matching  # noqa: E402
from closed_loop import (CLASSES, FORWARD_M, STOP_RADIUS, TURN_RAD, classify,  # noqa: E402
                         obstacle_for, segment_status)
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from pipeline import TARGET_HALF_M  # noqa: E402

ALIGN_RAD = math.radians(25.0)
DEFAULT_STEP_CAP = 200


def run_episode(task: dict, mode: str, fe: FrozenEncoder, matcher: matching.Encoder,
                rasters: RasterStore, cands: CandidateStore, goal_cache: dict,
                goal_feat_cache: dict, params: dict, step_cap: int = DEFAULT_STEP_CAP) -> dict:
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
    belief = ContinuousBelief(lib.positions.astype(np.float64), sims.astype(np.float64),
                              params["temperature"], k=params.get("k", 8),
                              sigma_prior_m=params.get("sigma_prior_m", 12.0))

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
    gate = params["gate"]

    for step in range(step_cap):
        dist = float(np.linalg.norm(pos - target))
        if dist < min_dist:
            min_dist, min_step = dist, step

        obs = matching.prepare_observation(raster, pos[0], pos[1], yaw)
        of = matcher.features(obs[None], ("layer2",))["layer2"][0]
        m = subpixel_match(of, goal_feat_cache[gkey], 2)
        dx, dy = matching.pixel_to_offset(m["col"], m["row"])
        z = np.array([pos[0] + dx, pos[1] + dy])

        score, margin = m["score"], m["margin"]
        agree_ref = belief.last_accept
        agree = (agree_ref is not None
                 and float(np.linalg.norm(z - agree_ref)) <= gate["agree_m"])
        validity = float(raster.valid_fraction(float(z[0]), float(z[1]), 8.0))
        border_ok = min(m["col"], m["row"], 223 - m["col"], 223 - m["row"]) >= gate["border_px"]
        maha = belief.best_mahalanobis(z)
        if mode == "c1":
            conf_ok = score >= gate["tau"]
            accept = conf_ok and agree
        else:
            conf_ok = ((score >= gate["tau2"]) and (margin >= gate["margin2"])
                       and (validity >= gate["v_min"]) and border_ok
                       and (maha <= gate["d_max"]))
            accept = conf_ok and agree
        if conf_ok:                     # remember even when the pair is rejected
            belief.last_accept = z.copy()

        updated = False
        if accept:
            belief.update(z, params["sigma_vis_m"], params["eta"], params["rho"], step)
            updates += 1
            updated = True
        est = belief.estimate()
        ent = belief.entropy()
        entropies.append(ent)

        vec = est - pos
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
                      "estimate": [float(est[0]), float(est[1])],
                      "est_distance_m": distance,
                      "est_bearing_deg": math.degrees(bearing),
                      "visual_score": float(score), "visual_margin": float(margin),
                      "visual_z": [float(z[0]), float(z[1])],
                      "visual_err_to_true_m": float(np.linalg.norm(z - target)),
                      "estimate_err_to_true_m": float(np.linalg.norm(est - target)),
                      "accept": bool(accept), "updated": bool(updated),
                      "agree": bool(agree), "validity": validity,
                      "mahalanobis": maha, "n_modes": int(len(belief.weights)),
                      "entropy": ent, "true_dist_m": dist})

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
    rec = {
        "episode_id": task["episode_id"], "split": task["split"], "index": int(task["index"]),
        "map": name, "length_group": task["length_group"], "cohort": task["cohort"],
        "controller": f"continuous_{mode}", "mode": mode,
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
    rec["failure"] = classify(rec)
    return rec


def record_summary(rec: dict) -> dict:
    keys = ("episode_id", "split", "index", "map", "length_group", "cohort", "controller",
            "mode", "step_cap", "steps", "stopped", "stop_step", "stop_dist", "final_dist",
            "min_dist", "min_dist_step", "osr", "success", "failure", "collision",
            "out_of_bounds", "path_len_m", "teacher_path_len_m", "action_counts",
            "goal_image_invalid", "target_valid_fraction", "belief_updates",
            "mean_entropy", "final_entropy", "seconds", "manifest_sha")
    return {k: rec.get(k) for k in keys}
