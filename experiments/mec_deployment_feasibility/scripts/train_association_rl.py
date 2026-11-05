#!/usr/bin/env python3
"""Stage-RL：三种 10-MEC 部署下的 DQN / PPO 关联智能体训练。

严格沿用历史 mec-v8.2 Stage-2r 逻辑（分支 experiment/v83-spatial-foundation）：
- 状态：Pure-MEC v2 = 4 + 10×19（候选 19-D 定义见下）
- 奖励：方案 D（纯 E2E）：u_t = 1 − e2e_t/τ；r_t = u_t/H；
        终止步再加 (1 − T/H)·J；R = J = 1 − mean_E2E/τ
- 算法：标准 DQN（MLP 194→256→256→10，γ=1.0，ε 1.0→0.05/200k 步，
        replay 50k，batch 128，warmup 2k，target 每 1k 步，lr 1e-4，grad clip 10）
        标准 PPO（MLP 194→256→256→{10,1}，γ=1.0，GAE 0.95，clip 0.2，
        lr 3e-4，rollout 2048，minibatch 256，10 epochs，value 0.5，entropy 0.01，
        grad clip 0.5，target KL 0.05）
- 部署：A=10×Small，B=10×Transformer-M，C=5×Small+5×TM（同位置同信道）
- 时延口径（用户指示：暂不使用实测推理时延）：
  T_inf = FLOPs / 0.2 TFLOPS（解析），T_comm = 1.6 Mbit / Shannon 速率，
  T_encode = 1.68 ms（官方口径）
- 数据：11A Train 18,521 → 固定顺序取 Dev-60 + Train-4000（与 Val-Seen 无交集）；
  训练只用 Train，Val-Seen 仅在最终评测使用。
- 能力先验：Dev-60 上各模型最近节点策略的 SR/OSR/SPL（nDTW 以 SPL 代用，注明）；
  不使用 Val-Seen 统计量。

用法:
  python train_association_rl.py --config A --algorithm dqn --seed 11 \
      --out-dir results/rl_A_dqn_s11
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/v4_closed_loop_revalidation/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
sys.path.insert(0, str(REPO / "experiments/06_closed_loop_success_rate/scripts"))

import matching  # noqa: E402
from closed_loop import (CLASSES, FORWARD_M, STOP_RADIUS, TURN_RAD,  # noqa: E402
                         obstacle_for, segment_status)
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from model11_small import SmallMEC4  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402
from run_closed_loop import (AGREE_M, BEL, CAL, HISTORY_LEN, TARGET_HALF_M,  # noqa: E402
                             TAU, belief_features, history_inputs,
                             landmark_inputs, pool_history_from_current)
from v5fix_contract import EMPTY_BELIEF, EMPTY_MATCH  # noqa: E402

RES = HERE.parent / "results"

# ---- 时延口径（解析，非实测推理）----
FLOP = {"small": 4.5e6, "transformer_m": 556.0e6}
CAP = 0.2e12
ENCODE_MS = 1.680
PAYLOAD_BITS = 49 * 2048 * 16
TAU_MS, H_MAX = 100.0, 400
GBS_H, UAV_H, CARRIER, BW_HZ, TXP = 10.0, 80.0, 2.0e9, 5.0e6, 0.1
NOISE_PSD, NF, LOS_A, LOS_B, EXC_LOS, EXC_NLOS = -174.0, 9.0, 9.61, 0.16, 1.0, 20.0
C_LIGHT = 299_792_458.0
POS_SCALE, COMPUTE_SCALE, BW_SCALE, RATE_SCALE = 500.0, 2.0e12, 30.0e6, 200.0e6
LAT_SCALE, E2E_CLIP, SNR_SCALE, INF_SCALE = 100.0, 3.0, 40.0, 10.0
MODEL_NAMES = ("Small", "Transformer-S", "Medium", "Transformer-M", "Large")
MKEY = {"small": "Small", "transformer_m": "Transformer-M"}
STATE_DIM = 4 + 10 * 19
N_ACT = 10


def dbm_to_watt(dbm):
    return 10.0 ** ((dbm - 30.0) / 10.0)


def uplink_rate_bps(h_m):
    h = max(float(h_m), 0.0)
    v = abs(UAV_H - GBS_H)
    d3 = max(math.sqrt(h * h + v * v), 1.0)
    elev = math.degrees(math.atan2(v, max(h, 1e-15)))
    p_los = 1.0 / (1.0 + LOS_A * math.exp(-LOS_B * (elev - LOS_A)))
    fsl = 20.0 * math.log10(4.0 * math.pi * CARRIER * d3 / C_LIGHT)
    pl = fsl + p_los * EXC_LOS + (1 - p_los) * EXC_NLOS
    noise = dbm_to_watt(NOISE_PSD + 10.0 * math.log10(BW_HZ) + NF)
    snr = TXP * 10.0 ** (-pl / 10.0) / max(noise, 1e-15)
    return max(BW_HZ * math.log2(1.0 + snr), 1e-15), snr


class NodeModel:
    """节点级时延与状态特征。"""

    def __init__(self, node: dict, pos: np.ndarray):
        self.node = node
        d = float(np.linalg.norm(np.asarray([node["x"], node["y"]]) - pos))
        rate, snr = uplink_rate_bps(d)
        self.distance = d
        self.available = d <= node["coverage_radius_m"]
        self.comm_ms = PAYLOAD_BITS / rate * 1e3
        self.inf_ms = FLOP[node["model_type"]] / node["compute_capacity_flops_per_s"] * 1e3
        self.e2e_ms = ENCODE_MS + self.comm_ms + self.inf_ms
        self.rate = rate
        self.snr = snr

    def features(self, pos: np.ndarray, prior: dict) -> np.ndarray:
        dx = self.node["x"] - pos[0]
        dy = self.node["y"] - pos[1]
        cap = prior[MKEY[self.node["model_type"]]]
        vals = [dx / POS_SCALE, dy / POS_SCALE, self.distance / POS_SCALE,
                self.node["compute_capacity_flops_per_s"] / COMPUTE_SCALE,
                self.node["bandwidth_hz"] / BW_SCALE]
        if self.available:
            snr_db = 10.0 * math.log10(max(self.snr, 1e-15))
            vals += [self.rate / RATE_SCALE, self.comm_ms / LAT_SCALE,
                     min(self.inf_ms / INF_SCALE, E2E_CLIP),
                     min(self.e2e_ms / LAT_SCALE, E2E_CLIP),
                     float(np.clip(snr_db / SNR_SCALE, -1.0, 1.0))]
        else:
            vals += [0.0] * 5
        vals += [cap["sr"], cap["osr"], cap["spl"], cap["ndtw"]]
        onehot = [0.0] * len(MODEL_NAMES)
        onehot[MODEL_NAMES.index(MKEY[self.node["model_type"]])] = 1.0
        vals += onehot
        return np.asarray(vals, np.float32)


def build_state(pos, t, running_mean_e2e, metrics, prior) -> tuple[np.ndarray, np.ndarray]:
    glob = [pos[0] / 1500.0, pos[1] / 1500.0, t / H_MAX,
            running_mean_e2e / TAU_MS]
    feats = np.stack([m.features(pos, prior) for m in metrics])
    mask = np.asarray([m.available for m in metrics], np.float32)
    if mask.sum() == 0:                     # 兜底：最近节点
        nearest = int(np.argmin([m.distance for m in metrics]))
        mask[nearest] = 1.0
    return np.concatenate([glob, feats.ravel()]).astype(np.float32), mask


class EpisodeEnv:
    """单任务闭环 + 关联动作（在线仿真）。"""

    def __init__(self, deployment: dict, prior: dict, device: str,
                 models: dict, step_cap: int = 200):
        self.by_map = {}
        for n in deployment:
            self.by_map.setdefault(n["map"], []).append(n)
        self.prior = prior
        self.device = device
        self.models = models
        self.step_cap = step_cap
        self.rasters, self.cands = RasterStore(), CandidateStore()
        self.enc4 = FrozenEncoder(device)
        self.matcher_enc = matching.Encoder(device)

    def start(self, task: dict):
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
        self.goal_img = matching.prepare_goal(
            self.raster, float(self.target[0]), float(self.target[1]))
        gl = self.matcher_enc.features(self.goal_img[None], ("layer2",))["layer2"][0]
        self._gl2 = gl
        self.gdesc = gl[:, 13:15, 13:15].mean(axis=(1, 2))
        self.gdesc = self.gdesc / np.maximum(
            np.linalg.norm(self.gdesc, keepdims=True), 1e-8)
        self.hist = [__import__("collections").deque(maxlen=HISTORY_LEN)
                     for _ in range(3)]
        self.step_idx = 0
        self.n_forward = 0
        self.episode_e2e = []
        self.min_dist = float(np.linalg.norm(self.pos - self.target))
        self.done = False
        self.success = False
        self.stop_dist = None
        self.failure = ""
        self.collision = self.oob = False

    def metrics(self) -> list[NodeModel]:
        return [NodeModel(n, self.pos) for n in self.nodes]

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        mean_e2e = (float(np.mean(self.episode_e2e))
                    if self.episode_e2e else 0.0)
        return build_state(self.pos, self.step_idx, mean_e2e,
                           self.metrics(), self.prior)

    def step(self, action: int):
        """执行关联动作 → 该节点模型推理 → 环境转移。返回 (reward, done, info)。"""
        metrics = self.metrics()
        m = metrics[int(action)]
        self.episode_e2e.append(m.e2e_ms)
        if self.empty_cands:
            belief_feat = EMPTY_BELIEF.copy()
            match_feat = EMPTY_MATCH.copy()
            est = np.asarray([self.pos[0], self.pos[1]], np.float64)
            score = 0.0
        else:
            obs = matching.prepare_observation(self.raster, self.pos[0],
                                              self.pos[1], self.yaw)
            of = self.matcher_enc.features(obs[None], ("layer2",))["layer2"][0]
            mm = subpixel_match(of, self._gl2, 2)
            dx, dy = matching.pixel_to_offset(mm["col"], mm["row"])
            z = np.array([self.pos[0] + dx, self.pos[1] + dy])
            score, margin = mm["score"], mm["margin"]
            agree = (self.belief.last_accept is not None
                     and float(np.linalg.norm(z - self.belief.last_accept)) <= AGREE_M)
            conf_ok = score >= TAU
            if conf_ok:
                self.belief.last_accept = z.copy()
            if conf_ok and agree:
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
        mtype = m.node["model_type"]
        model = self.models[mtype]
        with torch.no_grad():
            cur = self.enc4(np.asarray([np.asarray(cur_img, np.uint8)]))
            batch["current"] = cur
            logits = model(batch)
        a = int(logits.argmax(-1).item())
        self.hist[0].append(pool_history_from_current(cur)[0].astype(np.float32))
        self.hist[1].append((float(self.pos[0]), float(self.pos[1]), float(self.yaw)))
        self.hist[2].append(a)
        dist = float(np.linalg.norm(self.pos - self.target))
        self.min_dist = min(self.min_dist, dist)
        if a == 0:
            self.done, self.success = True, bool(dist <= STOP_RADIUS)
            self.failure = "success" if self.success else "premature_stop"
            self.stop_dist = dist
        elif a == 1:
            nxt = self.pos + FORWARD_M * np.asarray(
                [math.cos(self.yaw), math.sin(self.yaw)])
            st = segment_status(self.info, float(self.pos[0]), float(self.pos[1]),
                                float(nxt[0]), float(nxt[1]))
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
        # ---- 方案 D 奖励 ----
        u = 1.0 - m.e2e_ms / TAU_MS
        reward = u / H_MAX
        done = self.done
        if done:
            T = self.step_idx
            J = 1.0 - float(np.mean(self.episode_e2e)) / TAU_MS
            reward += (1.0 - T / H_MAX) * J
            if self.stop_dist is not None:
                self.success = bool(self.stop_dist <= STOP_RADIUS)
        info = {"e2e_ms": m.e2e_ms, "inf_ms": m.inf_ms, "comm_ms": m.comm_ms,
                "node": m.node["gbs_id"], "model": mtype,
                "failure": self.failure, "success": self.success,
                "return_D": (1.0 - float(np.mean(self.episode_e2e)) / TAU_MS)
                if done else None}
        return reward, done, info



class QNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_act=N_ACT, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_act))

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_act=N_ACT, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, n_act)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class Replay:
    def __init__(self, cap, state_dim, rng):
        self.cap, self.rng = cap, rng
        self.s = np.zeros((cap, state_dim), np.float32)
        self.m = np.zeros((cap, N_ACT), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, state_dim), np.float32)
        self.m2 = np.zeros((cap, N_ACT), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.size, self.ptr = 0, 0

    def add(self, *tr):
        i = self.ptr
        self.s[i], self.m[i], self.a[i], self.r[i], self.s2[i], self.m2[i], self.d[i] = tr
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs, device):
        idx = self.rng.integers(0, self.size, size=bs)
        return tuple(torch.as_tensor(v[idx], device=device) for v in
                     (self.s, self.m, self.a, self.r, self.s2, self.m2, self.d))


def masked_argmax(q, mask):
    q = q.detach().cpu().numpy().astype(np.float64).copy()
    q[mask < 0.5] = -1e18
    return int(np.argmax(q))


def epsilon_at(step, decay=200_000):
    frac = min(1.0, step / decay)
    return 1.0 + (0.05 - 1.0) * frac


def load_tasks_and_split(seed=20261022, n_train=4000, n_dev=60):
    """11A Train 18,521 → 固定顺序划分 Dev-60 与 Train-4000（无 Val-Seen）。"""
    import csv
    raw = json.loads(Path("/mnt/hdd2t/Citynav-MEC/DATA/sbfnav-data"
                          "/original_citynav/citynav_train_seen.json").read_text())
    rows = list(csv.DictReader(open(
        REPO / "experiments/11_teacher_geometry_ablation"
        "/11A_teacher_dataset_audit/results/teacher_episode_manifest.csv")))
    pool = []
    for r in rows:
        e = raw[int(r["index"])]
        traj = e["trajectory"]
        traj = json.loads(traj) if isinstance(traj, str) else traj
        tgt = e["target_positions"]
        tgt = json.loads(tgt) if isinstance(tgt, str) else tgt
        pool.append({"map": f"{e['area']}_block_{e['block']}",
                     "index": int(r["index"]),
                     "start_x": traj[0][0], "start_y": traj[0][1],
                     "start_yaw": traj[0][3] if len(traj[0]) > 3 else 0.0,
                     "target_x": tgt[0][0], "target_y": tgt[0][1]})
    assert len(pool) == 18521
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    dev = [pool[i] for i in order[:n_dev]]
    train = [pool[i] for i in order[n_dev:n_dev + n_train]]
    return train, dev


def build_models(types, device):
    ck = {"small": ("runs_small/seed42/best.pt", SmallMEC4),
          "transformer_m": ("runs_11c_v5fix/seed42/best.pt", TransformerM12Fix)}
    out = {}
    for t in types:
        rel, cls = ck[t]
        m = cls().to(device).eval()
        sd = torch.load(REPO / "experiments/11_teacher_geometry_ablation"
                        "/11B_transformer_m_v4/results" / rel,
                        map_location=device, weights_only=False)
        m.load_state_dict(sd["model"])
        for p in m.parameters():
            p.requires_grad_(False)
        out[t] = m
    return out


def evaluate(env, tasks, policy_fn) -> dict:
    e2e, succ, steps = [], 0, []
    for task in tasks:
        env.start(task)
        s, mask = env.state()
        done = False
        while not done:
            a = policy_fn(s, mask)
            _, done, info = env.step(a)
            s, mask = env.state()
        e2e.append(float(np.mean(env.episode_e2e)))
        succ += int(env.success)
        steps.append(env.step_idx)
    return {"n": len(tasks), "sr": succ / len(tasks),
            "e2e_mean_ms": float(np.mean(e2e)),
            "e2e_p95_ms": float(np.percentile(e2e, 95)),
            "steps_mean": float(np.mean(steps))}


def build_prior(envA, envB, dev_tasks) -> dict:
    """Dev-60 上最近节点策略的能力先验（SR/OSR/SPL；nDTW 以 SPL 代用）。"""
    prior = {}
    for key, env in (("Small", envA), ("Transformer-M", envB)):
        sr = o = spl = 0.0
        for task in dev_tasks:
            env.start(task)
            done = False
            while not done:
                metrics = env.metrics()
                a = int(np.argmin([m.distance for m in metrics]))
                _, done, info = env.step(a)
            sr += float(env.success)
            o += float(env.min_dist <= STOP_RADIUS)
            dist = float(np.linalg.norm(env.pos - env.target))
            path = FORWARD_M * env.step_idx
            spl += (min(1.0, float(np.linalg.norm(
                np.asarray([task["start_x"], task["start_y"]]) - env.target)) / path)
                if env.success and path > 0 else 0.0)
        n = len(dev_tasks)
        prior[key] = {"sr": sr / n, "osr": o / n, "spl": spl / n,
                      "ndtw": spl / n}
    for k in ("Transformer-S", "Medium", "Large"):
        prior[k] = {"sr": 0.0, "osr": 0.0, "spl": 0.0, "ndtw": 0.0}
    return prior


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["A", "B", "C"], required=True)
    ap.add_argument("--algorithm", choices=["dqn", "ppo"], required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--dev-every", type=int, default=200)
    ap.add_argument("--step-cap", type=int, default=200)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = Path(a.out_dir)
    if not out.is_absolute():
        out = RES / out
    out.mkdir(parents=True, exist_ok=True)
    hist_path = out / "history.jsonl"
    dev_path = out / "dev.jsonl"
    log = open(out / "train.log", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = a.device

    cfgname = {"A": "A_10xSmall", "B": "B_10xTransformerM",
               "C": "C_5Small_5TM"}[a.config]
    deployment_all = json.loads((RES / "deployment_configs.json").read_text())[cfgname]
    # 先验需要 Small 与 TM 两套权重（即使当前配置只用其一）
    types = sorted({"small", "transformer_m"}
                   | {n["model_type"] for n in deployment_all})
    models = build_models(types, device)
    train_tasks, dev_tasks = load_tasks_and_split(n_train=4000, n_dev=60)
    say(f"[start] config={a.config} algo={a.algorithm} seed={a.seed} "
        f"types={types} train={len(train_tasks)} dev={len(dev_tasks)}")

    env = EpisodeEnv(deployment_all, prior={}, device=device, models=models,
                     step_cap=a.step_cap)
    def homog(t):
        return [dict(n, model_type=t) for n in deployment_all]
    envA = EpisodeEnv(homog("small"), {}, device, models, a.step_cap)
    envB = EpisodeEnv(homog("transformer_m"), {}, device, models, a.step_cap)
    cache = RES / "capability_prior_dev60.json"
    if cache.exists():
        prior = json.loads(cache.read_text())
        say(f"[prior] reused {cache.name}")
    else:
        prior = build_prior(envA, envB, dev_tasks)
        cache.write_text(json.dumps(prior, indent=2))
        say(f"[prior] computed and cached -> {cache.name}")
    (out / "capability_prior.json").write_text(json.dumps(prior, indent=2))
    env.prior = prior
    say(f"[prior] {json.dumps(prior)}")

    best_dev = float("inf")
    best_ep = -1
    best_sr = -1.0
    start_ep = 0
    ckpt_path = out / "checkpoint.pt"      # 仅用于断点续训（当前权重）
    best_path = out / "best.pt"            # 仅保存最低 Dev E2E 的权重

    if a.algorithm == "dqn":
        online = QNet().to(device)
        target = QNet().to(device)
        target.load_state_dict(online.state_dict())
        opt = torch.optim.Adam(online.parameters(), lr=1e-4)
        replay = Replay(50_000, STATE_DIM, rng)
        total_steps = 0
    else:
        ac = ActorCritic().to(device)
        opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
        buf = {"s": [], "m": [], "a": [], "r": [], "done": [], "lp": [], "v": []}
        total_steps = 0

    def save_ckpt(ep, path=None):
        payload = {"ep": ep, "best_dev": best_dev, "best_ep": best_ep,
                   "best_sr": best_sr, "total_steps": total_steps,
                   "prior": prior, "config": a.config,
                   "algorithm": a.algorithm, "seed": a.seed,
                   "state_dim": STATE_DIM, "n_act": N_ACT}
        if a.algorithm == "dqn":
            payload.update(online=online.state_dict(), target=target.state_dict(),
                           opt=opt.state_dict())
        else:
            payload.update(ac=ac.state_dict(), opt=opt.state_dict())
        torch.save(payload, path or ckpt_path)

    def policy_greedy(s, mask):
        if a.algorithm == "dqn":
            with torch.no_grad():
                q = online(torch.as_tensor(s[None], device=device))
            return masked_argmax(q[0], mask)
        with torch.no_grad():
            logits, _ = ac(torch.as_tensor(s[None], device=device))
        logits = logits[0].cpu().numpy().astype(np.float64).copy()
        logits[mask < 0.5] = -1e18
        return int(np.argmax(logits))

    def validate(ep):
        nonlocal best_dev, best_ep, best_sr
        m = evaluate(env, dev_tasks, policy_greedy)
        rec = {"episode": ep, **m}
        with dev_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        improved = (m["e2e_mean_ms"] < best_dev - 0.002 or
                    (abs(m["e2e_mean_ms"] - best_dev) <= 0.002
                     and m["sr"] > best_sr))
        if improved:
            best_dev, best_ep, best_sr = m["e2e_mean_ms"], ep, m["sr"]
            save_ckpt(ep, best_path)
            (out / "best_meta.json").write_text(json.dumps(
                {"best_dev_e2e_ms": best_dev, "best_episode": best_ep,
                 "best_dev_sr": best_sr}, indent=2))
        say(f"[dev] ep={ep} E2E={m['e2e_mean_ms']:.3f}ms P95={m['e2e_p95_ms']:.2f} "
            f"SR={m['sr']*100:.1f}% steps={m['steps_mean']:.1f} "
            f"{'NEW_BEST' if improved else ''}"
            f" (best={best_dev:.3f}@{best_ep})")
        return m

    # ---- 训练主循环 ----
    ma_rewards, ma_e2e, ma_succ = [], [], []
    ep = start_ep
    t0 = time.time()
    while ep < a.episodes:
        task = train_tasks[ep % len(train_tasks)]
        env.start(task)
        s, mask = env.state()
        done = False
        ep_reward = 0.0
        ep_e2e = []
        while not done:
            if a.algorithm == "dqn":
                eps = epsilon_at(total_steps)
                if rng.random() < eps:
                    if mask.sum() == 0:
                        a_idx = 0
                    else:
                        cand = np.flatnonzero(mask > 0.5)
                        a_idx = int(cand[int(rng.integers(0, len(cand)))])
                else:
                    with torch.no_grad():
                        q = online(torch.as_tensor(s[None], device=device))
                    a_idx = masked_argmax(q[0], mask)
                r, done, info = env.step(a_idx)
                s2, mask2 = env.state()
                replay.add(s, mask, a_idx, r, s2, mask2, float(done))
                total_steps += 1
                if (replay.size >= 2000 and total_steps % 1 == 0):
                    bs, bm, ba, br, bs2, bm2, bd = replay.sample(128, device)
                    q = online(bs).gather(1, ba[:, None]).squeeze(1)
                    with torch.no_grad():
                        qn = target(bs2)
                        qn = qn.masked_fill(bm2 < 0.5, -1e18)
                        nxt = qn.max(dim=1).values
                        tgt = br + 1.0 * (1.0 - bd) * nxt
                    loss = F.smooth_l1_loss(q, tgt)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                    opt.step()
                    if total_steps % 1000 == 0:
                        target.load_state_dict(online.state_dict())
            else:
                with torch.no_grad():
                    logits, v = ac(torch.as_tensor(s[None], device=device))
                logits = logits[0].cpu().numpy().astype(np.float64).copy()
                logits[mask < 0.5] = -1e18
                p = np.exp(logits - logits.max()); p /= p.sum()
                a_idx = int(rng.choice(N_ACT, p=p))
                logp = float(np.log(p[a_idx] + 1e-12))
                r, done, info = env.step(a_idx)
                s2, mask2 = env.state()
                buf["s"].append(s.copy()); buf["m"].append(mask.copy())
                buf["a"].append(a_idx); buf["r"].append(r)
                buf["done"].append(float(done)); buf["lp"].append(logp)
                buf["v"].append(float(v.item()))
                total_steps += 1
                if len(buf["s"]) >= 2048:
                    # GAE + PPO update
                    S = torch.as_tensor(np.asarray(buf["s"]), device=device)
                    M = torch.as_tensor(np.asarray(buf["m"]), device=device)
                    A_ = torch.as_tensor(np.asarray(buf["a"]), device=device)
                    R_ = np.asarray(buf["r"], np.float64)
                    D_ = np.asarray(buf["done"], np.float64)
                    V = np.asarray(buf["v"], np.float64)
                    LP = np.asarray(buf["lp"], np.float64)
                    adv = np.zeros_like(R_); last = 0.0
                    for t in range(len(R_) - 1, -1, -1):
                        nv = 0.0 if t == len(R_) - 1 else V[t + 1]
                        delta = R_[t] + 1.0 * (1 - D_[t]) * nv - V[t]
                        last = delta + 1.0 * 0.95 * (1 - D_[t]) * last
                        adv[t] = last
                    ret = adv + V
                    adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
                    ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)
                    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
                    old_lp = torch.as_tensor(LP, dtype=torch.float32, device=device)
                    n = len(R_); idx = np.arange(n)
                    for _ in range(10):
                        rng.shuffle(idx)
                        for st in range(0, n, 256):
                            b = torch.as_tensor(idx[st:st + 256], device=device)
                            logits, val = ac(S[b])
                            logits = logits.masked_fill(M[b] < 0.5, -1e18)
                            logp_all = torch.log_softmax(logits, dim=-1)
                            lp = logp_all.gather(1, A_[b][:, None]).squeeze(1)
                            ratio = torch.exp(lp - old_lp[b])
                            surr1 = ratio * adv_t[b]
                            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b]
                            pi_loss = -torch.min(surr1, surr2).mean()
                            v_loss = 0.5 * (val - ret_t[b]).pow(2).mean()
                            ent = -(torch.exp(logp_all) * logp_all).sum(-1).mean()
                            loss = pi_loss + v_loss - 0.01 * ent
                            opt.zero_grad(); loss.backward()
                            torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                            opt.step()
                    for k in buf:
                        buf[k].clear()
            s, mask = s2, mask2
            ep_reward += float(r)
            ep_e2e.append(info["e2e_ms"])
        ma_rewards.append(ep_reward); ma_e2e.append(float(np.mean(ep_e2e)))
        ma_succ.append(float(env.success))
        rec = {"episode": ep, "return_D": ep_reward,
               "mean_e2e_ms": float(np.mean(ep_e2e)),
               "success": int(env.success), "steps": env.step_idx,
               "failure": env.failure, "steps_total": total_steps,
               "elapsed_s": round(time.time() - t0, 1)}
        if a.algorithm == "dqn":
            rec["epsilon"] = epsilon_at(total_steps)
        with hist_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        ep += 1
        if ep % 100 == 0:
            say(f"[train] ep={ep} MA100 reward={np.mean(ma_rewards[-100:]):.4f} "
                f"E2E={np.mean(ma_e2e[-100:]):.2f}ms "
                f"SR={np.mean(ma_succ[-100:])*100:.1f}% "
                f"elapsed={(time.time()-t0)/60:.1f}min")
        if ep % a.dev_every == 0:
            validate(ep)
            save_ckpt(ep)          # checkpoint.pt 仅作为续训快照
    # 最终验证
    final = validate(a.episodes)
    save_ckpt(a.episodes)
    assert best_path.exists(), "best.pt 未生成（无任何验证改进？）"
    say(f"[done] config={a.config} algo={a.algorithm} seed={a.seed} "
        f"best_dev={best_dev:.3f}ms@{best_ep} final_dev={final['e2e_mean_ms']:.3f}ms "
        f"SR={final['sr']*100:.1f}%")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
