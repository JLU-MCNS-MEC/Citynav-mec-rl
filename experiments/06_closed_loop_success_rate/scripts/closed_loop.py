#!/usr/bin/env python3
"""Stage 06 closed-loop environment for the Stage-05 Transformer-M.

The model's own argmax action drives the UAV.  Nothing from the teacher
trajectory except the initial pose and the final target position is used:
history is built from the states and actions the model actually produced, and
the belief is recomputed from the current pose at every step.

Feature contract (must match ``05_transformer_m_training/scripts/dataset_v2.py``):

* current observation : ``MapRaster.heading_crop(x, y, yaw)`` 100 m x 100 m ->
  224 x 224 -> ``FrozenEncoder`` (uint8 -> GPU -> f32/255 -> ImageNet) ->
  (1, 49, 2048);
* history visual      : mean of the same forward over the 49 tokens, L2
  normalised, rounded to FP16 and back (the training cache stored FP16);
* target reference    : ``MapRaster.north_crop(tx, ty, 48 m)`` -> ``embed`` ->
  L2 normalised FP32.  The model receives the FP16-rounded value, the belief
  uses the raw FP32 value, exactly as ``prepare_arrays`` did;
* belief              : ``belief_features`` on the 24 m candidate library with
  the per-episode candidate scores precomputed once.

Motion: STOP ends the episode; Forward moves 5 m along the heading; TurnLeft /
TurnRight change the heading by +30 / -30 degrees.  A Forward move is rejected
if any sampled point of the 5 m segment leaves the raster (out_of_bounds) or
hits a 1 m cell whose surface reaches the 50 m flight altitude (collision).
The evaluator never rewrites an action and never calls A*.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "03b_target_directed_teacher/scripts"))

from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore, history_block  # noqa: E402
from model_v2 import TransformerM  # noqa: E402
from obstacles import load_or_build, world_to_cell  # noqa: E402
from pipeline import HISTORY_LEN, TARGET_HALF_M, belief_features, wrap  # noqa: E402

CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
FORWARD_M = 5.0
TURN_RAD = math.radians(30.0)
STOP_RADIUS = 20.0
DEFAULT_STEP_CAP = 200
OBSTACLE_CACHE = ROOT / "03b_target_directed_teacher/cache/obstacles"

#: decision keys produced by the frozen ResNet, in the order model_v2 expects
FEATURE_KEYS = ("current", "target_emb", "history_vis", "history_pose",
                "history_action", "history_mask", "belief")


def load_model(checkpoint: Path, device: str, expected_sha: str | None = None):
    """Load ``best.pt`` with strict=True and verify its identity."""
    import hashlib

    import torch

    if expected_sha:
        h = hashlib.sha256()
        with checkpoint.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        got = h.hexdigest()
        if got != expected_sha:
            raise RuntimeError(f"checkpoint SHA-256 {got} != expected {expected_sha}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TransformerM().to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


def obstacle_for(name: str) -> dict:
    return load_or_build(name, OBSTACLE_CACHE)


def segment_status(info: dict, x0: float, y0: float, x1: float, y1: float,
                   samples: int = 10) -> str:
    """'free', 'collision' or 'out_of_bounds' for a straight 5 m move."""
    blocked = info["blocked"]
    rows, cols = blocked.shape
    for i in range(1, samples + 1):
        f = i / samples
        x = x0 + (x1 - x0) * f
        y = y0 + (y1 - y0) * f
        row, col = world_to_cell(info, x, y)
        if not (0 <= row < rows and 0 <= col < cols):
            return "out_of_bounds"
        if blocked[row, col]:
            return "collision"
    return "free"


def move(pos: np.ndarray, yaw: float, action: int) -> tuple[np.ndarray, float]:
    """Apply one action ignoring the environment (tests / labels)."""
    if action == 1:
        return pos + FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)]), yaw
    if action == 2:
        return pos.copy(), wrap(yaw + TURN_RAD)
    if action == 3:
        return pos.copy(), wrap(yaw - TURN_RAD)
    return pos.copy(), yaw


def classify(record: dict) -> str:
    """Failure taxonomy.

    ``reached_but_not_stopped`` answers "did it find the target and keep
    flying", so it takes precedence over the terminal event; the collision /
    out_of_bounds flags stay in the record and are also counted separately.
    """
    if record.get("goal_image_invalid"):
        return "goal_image_invalid"
    if record["success"]:
        return "success"
    if record.get("osr"):
        return "reached_but_not_stopped"
    if record.get("collision"):
        return "collision"
    if record.get("out_of_bounds"):
        return "out_of_bounds"
    if record.get("stopped"):
        return "premature_stop"
    return "never_stop"


def _target_vectors(encoder: FrozenEncoder, raster, task: dict) -> tuple[np.ndarray, np.ndarray]:
    vec = encoder.embed([raster.north_crop(float(task["target_x"]), float(task["target_y"]),
                                           TARGET_HALF_M)])[0].astype(np.float32)
    return vec, vec.astype(np.float16).astype(np.float32)


def run_episode(model, encoder: FrozenEncoder, rasters: RasterStore, cands: CandidateStore,
                task: dict, step_cap: int = DEFAULT_STEP_CAP, device: str = "cuda",
                obstacles: dict | None = None, progress: dict | None = None) -> dict:
    """One closed-loop episode; returns the full per-step record."""
    import torch

    started = time.time()
    name = task["map"]
    raster = rasters.get(name)
    lib = cands.get(name)
    info = obstacles if obstacles is not None else obstacle_for(name)

    pos = np.asarray([float(task["start_x"]), float(task["start_y"])], np.float64)
    yaw = float(task["start_yaw"])
    target = np.asarray([float(task["target_x"]), float(task["target_y"])], np.float64)
    tvec32, tvec_model = _target_vectors(encoder, raster, task)
    sims = lib.embeddings @ tvec32

    states: list[list[float]] = []
    actions: list[int] = []
    pooled: list[np.ndarray] = []
    step_rows: list[dict] = []
    stop_prob = None
    stop_dist = None
    min_dist = float(np.linalg.norm(pos - target))
    min_step = 0
    collision = False
    out_of_bounds = False
    stopped = False

    for step in range(step_cap):
        dist = float(np.linalg.norm(pos - target))
        if dist < min_dist:
            min_dist, min_step = dist, step

        states.append([float(pos[0]), float(pos[1]), float(yaw)])
        actions.append(0)                       # placeholder; history reads only j < t
        img = raster.heading_crop(float(pos[0]), float(pos[1]), float(yaw))
        feats = encoder(np.stack([img]))        # (1,49,2048) float32 on GPU
        vec = feats.mean(dim=1)[0]
        vec = vec / vec.norm().clamp_min(1e-9)
        pooled.append(vec.detach().cpu().numpy().astype(np.float16))
        belief, _ = lib_belief(lib, tvec32, pos, yaw, sims)
        hist = history_block(np.stack(pooled), np.asarray(states, np.float32),
                             np.asarray(actions, np.int64), len(states) - 1)
        batch = {
            "current": feats,
            "target_emb": torch.from_numpy(tvec_model)[None].to(device),
            "belief": torch.from_numpy(belief[None]).to(device),
            **{k: torch.from_numpy(np.asarray(v)[None]).to(device) for k, v in hist.items()},
        }
        with torch.inference_mode():
            logits = model(batch)
            probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        action = int(probs.argmax())
        actions[-1] = action
        step_rows.append({"step": step, "x": float(pos[0]), "y": float(pos[1]),
                          "yaw": float(yaw), "dist": dist, "action": action,
                          "probs": [float(p) for p in probs],
                          "min_dist": min_dist})
        if action == 0:
            stopped = True
            stop_prob = float(probs[0])
            stop_dist = dist
            break
        if action == 1:
            nxt = pos + FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)])
            status = segment_status(info, float(pos[0]), float(pos[1]),
                                    float(nxt[0]), float(nxt[1]))
            if status == "collision":
                collision = True
                step_rows[-1]["blocked"] = "collision"
                break
            if status == "out_of_bounds":
                out_of_bounds = True
                step_rows[-1]["blocked"] = "out_of_bounds"
                break
            pos = nxt
        elif action == 2:
            yaw = wrap(yaw + TURN_RAD)
        elif action == 3:
            yaw = wrap(yaw - TURN_RAD)

    final_dist = float(np.linalg.norm(pos - target))
    osr = bool(min_dist <= STOP_RADIUS)
    success = bool(stopped and stop_dist is not None and stop_dist <= STOP_RADIUS
                   and final_dist <= STOP_RADIUS)
    forward_steps = sum(1 for a in actions if a == 1)
    record = {
        "episode_id": task["episode_id"], "split": task["split"], "index": int(task["index"]),
        "map": name, "length_group": task["length_group"], "cohort": task["cohort"],
        "start": [float(task["start_x"]), float(task["start_y"]), float(task["start_yaw"])],
        "target": [float(task["target_x"]), float(task["target_y"])],
        "step_cap": int(step_cap), "steps": len(step_rows),
        "stopped": stopped, "stop_step": (len(step_rows) - 1) if stopped else None,
        "stop_dist": stop_dist, "stop_prob": stop_prob,
        "final_pos": [float(pos[0]), float(pos[1]), float(yaw)],
        "final_dist": final_dist, "min_dist": min_dist, "min_dist_step": int(min_step),
        "osr": osr, "success": success, "collision": collision,
        "out_of_bounds": out_of_bounds,
        "path_len_m": FORWARD_M * forward_steps,
        "teacher_path_len_m": float(task.get("teacher_path_len_m") or 0.0),
        "action_counts": {CLASSES[i]: int(sum(1 for a in actions if a == i)) for i in range(4)},
        "goal_image_invalid": bool(task.get("goal_image_invalid", False)),
        "target_valid_fraction": float(task.get("target_valid_fraction", 1.0)),
        "target_center_valid": bool(task.get("target_center_valid", True)),
        "trajectory": step_rows,
        "seconds": round(time.time() - started, 3),
        "checkpoint_sha": task.get("checkpoint_sha"),
        "manifest_sha": task.get("manifest_sha"),
    }
    record["failure"] = classify(record)
    return record


def lib_belief(lib, tvec32: np.ndarray, pos: np.ndarray, yaw: float, sims: np.ndarray):
    return belief_features(lib, tvec32, pos, yaw, sims=sims)


def evaluate_task(model, encoder, rasters, cands, task: dict, step_cap: int,
                  device: str) -> dict:
    """Wrapper that turns an exception into an ``other_failure`` record."""
    if task.get("goal_image_invalid"):
        record = {
            "episode_id": task["episode_id"], "split": task["split"],
            "index": int(task["index"]), "map": task["map"],
            "length_group": task["length_group"], "cohort": task["cohort"],
            "start": [task["start_x"], task["start_y"], task["start_yaw"]],
            "target": [task["target_x"], task["target_y"]],
            "step_cap": int(step_cap), "steps": 0, "stopped": False, "stop_step": None,
            "stop_dist": None, "stop_prob": None, "final_pos": None, "final_dist": None,
            "min_dist": None, "min_dist_step": None, "osr": False, "success": False,
            "collision": False, "out_of_bounds": False, "path_len_m": 0.0,
            "teacher_path_len_m": float(task.get("teacher_path_len_m") or 0.0),
            "action_counts": {c: 0 for c in CLASSES},
            "goal_image_invalid": True,
            "target_valid_fraction": float(task.get("target_valid_fraction", 0.0)),
            "target_center_valid": bool(task.get("target_center_valid", False)),
            "trajectory": [], "seconds": 0.0,
            "checkpoint_sha": task.get("checkpoint_sha"),
            "manifest_sha": task.get("manifest_sha"),
        }
        record["failure"] = "goal_image_invalid"
        return record
    try:
        obstacles = obstacle_for(task["map"])
        return run_episode(model, encoder, rasters, cands, task, step_cap, device, obstacles)
    except Exception as exc:                      # noqa: BLE001 - recorded, never hidden
        import traceback
        return {
            "episode_id": task["episode_id"], "split": task["split"],
            "index": int(task["index"]), "map": task["map"],
            "length_group": task["length_group"], "cohort": task["cohort"],
            "success": False, "failure": "other_failure", "error": repr(exc),
            "traceback": traceback.format_exc()[-2000:],
            "checkpoint_sha": task.get("checkpoint_sha"),
            "manifest_sha": task.get("manifest_sha"),
        }


def record_summary(record: dict) -> dict:
    """Journal line: everything needed to resume/report, without the trajectory."""
    keys = ("episode_id", "split", "index", "map", "length_group", "cohort", "step_cap", "steps",
            "stopped", "stop_step", "stop_dist", "stop_prob", "final_dist", "min_dist",
            "min_dist_step", "osr", "success", "failure", "collision", "out_of_bounds",
            "path_len_m", "teacher_path_len_m", "action_counts", "goal_image_invalid",
            "target_valid_fraction", "seconds", "checkpoint_sha", "manifest_sha")
    out = {k: record.get(k) for k in keys}
    return out
