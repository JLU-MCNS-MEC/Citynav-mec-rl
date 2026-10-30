#!/usr/bin/env python3
"""Closed-loop environment for the belief-only geometric controllers.

Identical to Stage 06's environment in everything except the policy:

* same tasks, start poses, target reference images, 24 m candidate libraries,
  obstacle grid, boundary checks, 200-step cap, 5 m forward, 30 deg turns and
  the 20 m horizontal success rule;
* **no current ``heading_crop``, no current visual pass, no history, no
  Transformer** at decision time;
* the only decision input is the 12-d belief, recomputed from the live pose at
  every step with the same ``belief_features`` call Stage 06 used.

The true target is used only to build the goal image (task input) and, after
the fact, to score the episode.  ``controller.act`` is called with the belief
array alone; ``task`` is not in its scope.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "06_closed_loop_success_rate/scripts"))
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))

from closed_loop import (CLASSES, FORWARD_M, STOP_RADIUS, TURN_RAD, classify,  # noqa: E402
                         obstacle_for, segment_status)
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from pipeline import TARGET_HALF_M, belief_features  # noqa: E402

DEFAULT_STEP_CAP = 200


def target_vector(encoder: FrozenEncoder, raster, task: dict) -> np.ndarray:
    """Goal-image embedding, exactly as in Stage 05/06 (raw FP32)."""
    return encoder.embed([raster.north_crop(float(task["target_x"]),
                                            float(task["target_y"]),
                                            TARGET_HALF_M)])[0].astype(np.float32)


def run_episode(task: dict, controller, encoder: FrozenEncoder, rasters: RasterStore,
                cands: CandidateStore, step_cap: int = DEFAULT_STEP_CAP,
                target_cache: dict | None = None) -> dict:
    started = time.time()
    name = task["map"]
    raster = rasters.get(name)
    lib = cands.get(name)
    info = obstacle_for(name)

    pos = np.asarray([float(task["start_x"]), float(task["start_y"])], np.float64)
    yaw = float(task["start_yaw"])
    target = np.asarray([float(task["target_x"]), float(task["target_y"])], np.float64)
    key = (name, round(float(target[0]), 3), round(float(target[1]), 3))
    if target_cache is not None and key in target_cache:
        tvec = target_cache[key]
    else:
        tvec = target_vector(encoder, raster, task)
        if target_cache is not None:
            target_cache[key] = tvec
    sims = lib.embeddings @ tvec

    steps: list[dict] = []
    stop_prob = None
    stop_dist = None
    stopped = collision = out_of_bounds = False
    min_dist = float(np.linalg.norm(pos - target))
    min_step = 0

    for step in range(step_cap):
        dist = float(np.linalg.norm(pos - target))      # scoring only
        if dist < min_dist:
            min_dist, min_step = dist, step

        belief, _ = belief_features(lib, tvec, pos, yaw, sims=sims)
        action = controller.act(belief)                 # <- the only input
        diag = controller.diagnostics(belief)

        steps.append({"step": step, "x": float(pos[0]), "y": float(pos[1]),
                      "yaw": float(yaw), "action": int(action),
                      "est_distance_m": diag["est_distance_m"],
                      "est_bearing_deg": diag["est_bearing_deg"],
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
    forward_steps = sum(1 for s in steps if s["action"] == 1)
    record = {
        "episode_id": task["episode_id"], "split": task["split"], "index": int(task["index"]),
        "map": name, "length_group": task["length_group"], "cohort": task["cohort"],
        "controller": controller.name,
        "start": [float(task["start_x"]), float(task["start_y"]), float(task["start_yaw"])],
        "target": [float(task["target_x"]), float(task["target_y"])],
        "step_cap": int(step_cap), "steps": len(steps),
        "stopped": stopped, "stop_step": (len(steps) - 1) if stopped else None,
        "stop_dist": stop_dist, "stop_prob": stop_prob,
        "final_pos": [float(pos[0]), float(pos[1]), float(yaw)],
        "final_dist": final_dist, "min_dist": min_dist, "min_dist_step": int(min_step),
        "osr": osr, "success": success, "collision": collision,
        "out_of_bounds": out_of_bounds,
        "path_len_m": FORWARD_M * forward_steps,
        "teacher_path_len_m": float(task.get("teacher_path_len_m") or 0.0),
        "action_counts": {CLASSES[i]: int(sum(1 for s in steps if s["action"] == i))
                          for i in range(4)},
        "goal_image_invalid": bool(task.get("goal_image_invalid", False)),
        "target_valid_fraction": float(task.get("target_valid_fraction", 1.0)),
        "target_center_valid": bool(task.get("target_center_valid", True)),
        "trajectory": steps,
        "seconds": round(time.time() - started, 3),
        "manifest_sha": task.get("manifest_sha"),
    }
    record["failure"] = classify(record)
    return record


def record_summary(record: dict) -> dict:
    keys = ("episode_id", "split", "index", "map", "length_group", "cohort", "controller",
            "step_cap", "steps", "stopped", "stop_step", "stop_dist", "stop_prob",
            "final_dist", "min_dist", "min_dist_step", "osr", "success", "failure",
            "collision", "out_of_bounds", "path_len_m", "teacher_path_len_m",
            "action_counts", "goal_image_invalid", "target_valid_fraction", "seconds",
            "manifest_sha")
    return {k: record.get(k) for k in keys}
