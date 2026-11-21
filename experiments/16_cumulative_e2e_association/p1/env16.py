#!/usr/bin/env python3
"""Stage 16-P1: cumulative-E2E association environment.

Independent of Stage 12-15 artifacts. Reuses the frozen CityNav navigation
protocol (matching / belief / history / kinematics) but:

  * hosts the FIVE Stage-14 frozen experts (not the Stage-11 models);
  * applies the Stage-14-R2 normalization for ts/tm/large;
  * uses the measured 10-TFLOPS inference latencies from Stage 16-P0;
  * rewards the CUMULATIVE episode E2E, not the mean step E2E.

Reward variants
  A  r_t = -e2e_ms / NORM_MS
  B  r_t = -e2e_ms / NORM_MS, plus FAIL_PENALTY on failure termination
     (failure = collision / out_of_bounds / step_cap / premature stop).
  B is reward shaping only: no SR constraint, no energy constraint, no CMDP.
"""
from __future__ import annotations

import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP16 = HERE.parent
REPO = EXP16.parents[1]
EXP14 = REPO / "experiments/14_mec_expert_retraining"
EXP12 = REPO / "experiments/12_hetero_mec_association"

# navigation protocol (frozen, read-only reuse)
sys.path.insert(0, str(EXP12 / "scripts"))
sys.path.insert(0, str(REPO / "experiments/v4_closed_loop_revalidation/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
sys.path.insert(0, str(REPO / "experiments/10_goal_query_mec_training"
                       "/10D_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
sys.path.insert(0, str(REPO / "experiments/06_closed_loop_success_rate/scripts"))
sys.path.insert(0, str(EXP14))

import protocol as P12  # noqa: E402  (frozen positions/comm/encoder model)
import matching  # noqa: E402
from closed_loop import (FORWARD_M, STOP_RADIUS, TURN_RAD, obstacle_for,  # noqa: E402
                         segment_status)
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from run_closed_loop import (AGREE_M, BEL, CAL, HISTORY_LEN,  # noqa: E402
                             TARGET_HALF_M, TAU, belief_features,
                             history_inputs, landmark_inputs,
                             pool_history_from_current)
from v5fix_contract import EMPTY_BELIEF, EMPTY_MATCH  # noqa: E402
from models import build_model, MODEL_NAMES  # noqa: E402

# ---- Stage 16 fixed configuration ------------------------------------------
MODEL_ORDER = ["small", "medium", "ts", "tm", "large"]
DEPLOY = MODEL_ORDER * 2
#: measured in P0 (Stage-14 final checkpoints, 2x MACs convention)
MODEL_GFLOPS_MEASURED = {
    "small": 4.954039, "medium": 14.978528, "ts": 39.671126,
    "tm": 100.025418, "large": 199.723981,
}
#: training-side Dev Macro-F1 (leak-free quality prior; never Val-Seen)
QUALITY_DEV_F1 = {
    "small": 0.5771, "medium": 0.6025, "ts": 0.6500,
    "tm": 0.6177, "large": 0.6105,
}
CKPTS = {
    "small": EXP14 / "checkpoints/Small/best.pt",
    "medium": EXP14 / "checkpoints/Medium/best.pt",
    "ts": EXP14 / "stability_revision/runs/ts/seed42/checkpoints/best.pt",
    "tm": EXP14 / "stability_revision/runs/tm/seed42/checkpoints/best.pt",
    "large": EXP14 / "stability_revision/runs/large/seed42/checkpoints/best.pt",
}
NORMALIZED = {"ts", "tm", "large"}
NORM_FILE = EXP14 / "stability_revision/normalization_statistics.json"
MEC_TFLOPS = 10.0
COVERAGE_M = 500.0
MAX_STEPS = 200

#: reward normalization constants (set from Train/Dev in P1; never Val-Seen)
DEFAULT_NORM_MS = 60.0
DEFAULT_FAIL_PENALTY = -50.0   # in normalized units


def load_norm():
    d = json.loads(NORM_FILE.read_text(encoding="utf-8"))["standardized"]
    return {k: (np.asarray(d[k]["mean"], np.float32),
                np.asarray(d[k]["std"], np.float32))
            for k in ("belief", "match", "history_pose")}


def load_models(device):
    models = {}
    for key in MODEL_ORDER:
        m = build_model(key).to(device)
        st = torch.load(CKPTS[key], map_location=device, weights_only=False)
        m.load_state_dict(st["model"])
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[key] = m
    return models


class NodeModel:
    def __init__(self, node, pos, enc_ms):
        self.node = node
        d = float(np.linalg.norm(np.asarray([node["x"], node["y"]])
                                 - np.asarray(pos)))
        self.distance = d
        self.available = bool(d <= COVERAGE_M)
        self.comm_ms = P12.comm_latency_ms(d)
        self.inf_ms = MODEL_GFLOPS_MEASURED[node["model_type"]] / MEC_TFLOPS
        self.e2e_ms = enc_ms + self.comm_ms + self.inf_ms

    def features(self):
        dx = self.node["x"] - self._pos[0]
        dy = self.node["y"] - self._pos[1]
        return None


class EpisodeEnv16:
    """One task, real closed loop, association action each step."""

    def __init__(self, by_map, device, models, norm, reward_variant="B",
                 norm_ms=DEFAULT_NORM_MS, fail_penalty=DEFAULT_FAIL_PENALTY,
                 step_cap=MAX_STEPS, record_steps=False):
        self.by_map = by_map
        self.device = device
        self.models = models
        self.norm = norm
        self.reward_variant = reward_variant
        self.norm_ms = float(norm_ms)
        self.fail_penalty = float(fail_penalty)
        self.step_cap = step_cap
        self.record_steps = record_steps
        self.rasters, self.cands = RasterStore(), CandidateStore()
        self.enc4 = FrozenEncoder(device)
        self.matcher_enc = matching.Encoder(device)
        enc_rec = json.loads((EXP12 / "results/protocol/encoder_flops.json")
                             .read_text())
        self.enc_ms = float(enc_rec["flops"]) / 1.5e12 * 1e3

    def start(self, task):
        name = task["map"]
        self.nodes = self.by_map[name]
        self.raster = self.rasters.get(name)
        self.lib = self.cands.get(name)
        self.info = obstacle_for(name)
        self.pos = np.asarray([task["start_x"], task["start_y"]], np.float64)
        self.yaw = float(task["start_yaw"])
        self.target = np.asarray([task["target_x"], task["target_y"]], np.float64)
        self.empty_cands = len(self.lib.positions) == 0
        self.belief = None
        if not self.empty_cands:
            tvec = self.enc4.embed([self.raster.north_crop(
                float(self.target[0]), float(self.target[1]), TARGET_HALF_M)]
            )[0].astype(np.float32)
            sims = self.lib.embeddings @ tvec
            self.belief = ContinuousBelief(
                self.lib.positions.astype(np.float64),
                sims.astype(np.float64), 0.1, k=8, sigma_prior_m=12.0)
        goal_img = matching.prepare_goal(self.raster, float(self.target[0]),
                                         float(self.target[1]))
        gl = self.matcher_enc.features(goal_img[None], ("layer2",))["layer2"][0]
        self._gl2 = gl
        self.gdesc = gl[:, 13:15, 13:15].mean(axis=(1, 2))
        self.gdesc = self.gdesc / np.maximum(
            np.linalg.norm(self.gdesc, keepdims=True), 1e-8)
        self.hist = [deque(maxlen=HISTORY_LEN) for _ in range(3)]
        self.step_idx = 0
        self.n_forward = 0
        self.episode_e2e = []
        self.min_dist = float(np.linalg.norm(self.pos - self.target))
        self.done = False
        self.success = False
        self.stop_dist = None
        self.failure = ""
        self.collision = self.oob = False
        self.steps = [] if self.record_steps else None

    def metrics(self):
        out = []
        for n in self.nodes:
            m = NodeModel(n, self.pos, self.enc_ms)
            m._pos = self.pos
            out.append(m)
        return out

    def coverage_mask(self):
        return np.asarray([m.available for m in self.metrics()], np.float32)

    def step(self, action):
        metrics = self.metrics()
        m = metrics[int(action)]
        self.episode_e2e.append(float(m.e2e_ms))
        if self.empty_cands:
            belief_feat = EMPTY_BELIEF.copy()
            match_feat = EMPTY_MATCH.copy()
        else:
            obs = matching.prepare_observation(self.raster, self.pos[0],
                                               self.pos[1], self.yaw)
            of = self.matcher_enc.features(obs[None], ("layer2",))["layer2"][0]
            mm = subpixel_match(of, self._gl2, 2)
            dx, dy = matching.pixel_to_offset(mm["col"], mm["row"])
            z = np.array([self.pos[0] + dx, self.pos[1] + dy])
            score, margin = mm["score"], mm["margin"]
            agree = (self.belief.last_accept is not None and
                     float(np.linalg.norm(z - self.belief.last_accept)) <= AGREE_M)
            if score >= TAU:
                self.belief.last_accept = z.copy()
            if score >= TAU and agree:
                self.belief.update(z, CAL["sigma_vis_m"], BEL["best"]["eta"],
                                   BEL["best"]["rho"], self.step_idx)
            est = self.belief.estimate()
            unc = float(math.sqrt(np.mean(self.belief.vars)))
            since = float(self.step_idx - self.belief.last_update)
            ent = float(self.belief.entropy())
            if not math.isfinite(ent):
                ent = 0.0
            belief_feat = belief_features(est, self.pos, self.yaw, score, unc,
                                          ent, since, dx, dy)
            match_feat = np.asarray([dx, dy, score, margin, mm["sharpness"]],
                                    np.float32)
        vis, pose_h, act_h, hmask = history_inputs(
            self.hist[0], self.hist[1], self.hist[2], self.pos, self.yaw)
        cur_img = self.raster.heading_crop(float(self.pos[0]),
                                           float(self.pos[1]), float(self.yaw))
        mtype = m.node["model_type"]
        # Stage-14-R2 normalization for ts/tm/large
        if mtype in NORMALIZED:
            belief_feat = ((belief_feat - self.norm["belief"][0]) /
                           self.norm["belief"][1]).astype(np.float32)
            match_feat = ((match_feat - self.norm["match"][0]) /
                          self.norm["match"][1]).astype(np.float32)
            pose_h = ((pose_h - self.norm["history_pose"][0]) /
                      self.norm["history_pose"][1]).astype(np.float32)
        batch = {
            "target_emb": torch.from_numpy(self.gdesc[None].copy()).to(self.device),
            "history_vis": torch.from_numpy(vis).to(self.device),
            "history_pose": torch.from_numpy(pose_h).to(self.device),
            "history_action": torch.from_numpy(act_h).to(self.device),
            "history_mask": torch.from_numpy(hmask).to(self.device),
            "belief": torch.from_numpy(belief_feat[None].copy()).to(self.device),
            "match": torch.from_numpy(match_feat[None].copy()).to(self.device),
            "landmarks": torch.from_numpy(
                landmark_inputs(self.belief, self.empty_cands)).to(self.device),
            "states": torch.from_numpy(
                np.asarray([[self.pos[0], self.pos[1], self.yaw]],
                           np.float32)).to(self.device),
        }
        with torch.no_grad():
            cur = self.enc4(np.asarray([cur_img], np.uint8))
            batch["current"] = cur
            logits = self.models[mtype](batch)
        a = int(logits.argmax(-1).item())
        self.hist[0].append(pool_history_from_current(cur)[0].astype(np.float32))
        self.hist[1].append((float(self.pos[0]), float(self.pos[1]), float(self.yaw)))
        self.hist[2].append(a)
        dist = float(np.linalg.norm(self.pos - self.target))
        self.min_dist = min(self.min_dist, dist)
        if a == 0:
            self.done = True
            self.success = bool(dist <= STOP_RADIUS)
            self.failure = "success" if self.success else "premature_stop"
            self.stop_dist = dist
        elif a == 1:
            nxt = self.pos + FORWARD_M * np.asarray([math.cos(self.yaw),
                                                     math.sin(self.yaw)])
            st = segment_status(self.info, float(self.pos[0]),
                                float(self.pos[1]), float(nxt[0]), float(nxt[1]))
            if st == "collision":
                self.done, self.failure, self.collision = True, "collision", True
            elif st == "out_of_bounds":
                self.done, self.failure, self.oob = True, "out_of_bounds", True
            else:
                self.pos = nxt
                self.n_forward += 1
        elif a == 2:
            self.yaw = (self.yaw + TURN_RAD + math.pi) % (2 * math.pi) - math.pi
        elif a == 3:
            self.yaw = (self.yaw - TURN_RAD + math.pi) % (2 * math.pi) - math.pi
        self.step_idx += 1
        if self.step_idx >= self.step_cap:
            self.done = True
            self.failure = self.failure or "step_cap"

        # ---- cumulative-E2E reward ----
        reward = -float(m.e2e_ms) / self.norm_ms
        if self.done and self.reward_variant == "B" and not self.success:
            reward += self.fail_penalty
        if self.steps is not None:
            self.steps.append({"step": self.step_idx - 1, "node": m.node["gbs_id"],
                               "model": mtype, "action": a,
                               "e2e_ms": float(m.e2e_ms),
                               "comm_ms": float(m.comm_ms),
                               "inf_ms": float(m.inf_ms),
                               "encode_ms": float(self.enc_ms),
                               "reward": float(reward),
                               "success": bool(self.success),
                               "failure": self.failure})
        info = {"e2e_ms": float(m.e2e_ms), "inf_ms": float(m.inf_ms),
                "comm_ms": float(m.comm_ms), "node": m.node["gbs_id"],
                "model": mtype, "failure": self.failure,
                "success": bool(self.success),
                "cumulative_e2e_ms": float(np.sum(self.episode_e2e))}
        return reward, self.done, info


def build_deployment_by_map():
    """10 nodes per map, five models repeated twice, homogeneous 10 TFLOPS."""
    positions = json.loads((REPO / "experiments/mec_deployment_feasibility"
                            "/results/mec_positions_k10.json").read_text())
    out = {}
    for name, xy in positions.items():
        nodes = []
        for i, (x, y) in enumerate(xy):
            nodes.append({"map": name, "gbs_id": f"G{i + 1}", "x": float(x),
                          "y": float(y), "model_type": DEPLOY[i],
                          "compute_tflops": MEC_TFLOPS})
        out[name] = nodes
    return out
