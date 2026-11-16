#!/usr/bin/env python3
"""Stage 12: DQN / PPO association agents on the four frozen deployments.

Identical algorithm definitions to the Stage-11 feasibility study / mec-v8.2
Stage-2r, with two changes required by this stage:
  * node features are re-scaled for 5-30 TFLOPS and 0.17-40 ms inference;
  * the deployment hosts 2 or 5 expert types (2 nodes each in the 5-model case).

Reward (scheme D, pure E2E): u_t = 1 - e2e_t/100; r_t = u_t/400; terminal
correction (1 - T/400) * J; SR/OSR/SPL never enter the reward.

Training pool: 11A Train 18,521 -> Dev-60 + Train-4,000 (seed 20261022).
Val-Seen is never read here.

Usage:
  python train_association_rl.py --group two_homo --algorithm ppo --seed 42 \
      --episodes 4000 --out-dir rl/two_homo/ppo/seed42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# parallel jobs: keep each process from spawning a full CPU thread pool
torch.set_num_threads(int(os.environ.get("STAGE12_TORCH_THREADS", "4")))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

sys.path.insert(0, str(P.REPO / "experiments/v4_closed_loop_revalidation/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/06_closed_loop_success_rate/scripts"))

import matching  # noqa: E402
from closed_loop import (CLASSES, FORWARD_M, STOP_RADIUS, TURN_RAD,  # noqa: E402
                         obstacle_for, segment_status)
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from run_closed_loop import (AGREE_M, BEL, CAL, HISTORY_LEN, TARGET_HALF_M,  # noqa: E402
                             TAU, belief_features, build_model,
                             history_inputs, landmark_inputs,
                             pool_history_from_current)
from v5fix_contract import EMPTY_BELIEF, EMPTY_MATCH  # noqa: E402

STATE_DIM = P.STATE_DIM
N_ACT = P.N_ACT


def load_models(types, device):
    out = {}
    for mt in types:
        model = build_model(P.RUNNER_MODEL[mt], device)
        ck = torch.load(P.ckpt_path(mt), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        out[mt] = model
    return out


class NodeModel:
    """Node-level metrics and 19-D state features."""

    def __init__(self, node, pos, enc_ms):
        self.node = node
        m = P.node_metrics(node, pos, enc_ms)
        self.distance = m["distance_m"]
        self.available = m["available"]
        self.comm_ms = m["comm_ms"]
        self.inf_ms = m["inf_ms"]
        self.e2e_ms = m["e2e_ms"]
        self.rate = m["rate_bps"]
        self.snr = m["snr"]

    def features(self, pos, prior):
        dx = self.node["x"] - pos[0]
        dy = self.node["y"] - pos[1]
        cap = prior[P.MODEL_LABEL[self.node["model_type"]]]
        vals = [dx / P.POS_SCALE, dy / P.POS_SCALE, self.distance / P.POS_SCALE,
                self.node["compute_capacity_flops_per_s"] / P.COMPUTE_SCALE,
                self.node["bandwidth_hz"] / P.BW_SCALE]
        if self.available:
            snr_db = 10.0 * math.log10(max(self.snr, 1e-15))
            vals += [self.rate / P.RATE_SCALE, self.comm_ms / P.LAT_SCALE,
                     min(self.inf_ms / P.INF_SCALE, P.E2E_CLIP),
                     min(self.e2e_ms / P.LAT_SCALE, P.E2E_CLIP),
                     float(np.clip(snr_db / P.SNR_SCALE, -1.0, 1.0))]
        else:
            vals += [0.0] * 5
        vals += [cap["sr"], cap["osr"], cap["spl"], cap["ndtw"]]
        onehot = [0.0] * len(P.MODEL_ORDER)
        onehot[P.MODEL_ORDER.index(self.node["model_type"])] = 1.0
        vals += onehot
        return np.asarray(vals, np.float32)


def build_state(pos, t, running_mean_e2e, metrics, prior):
    g = [pos[0] / 1500.0, pos[1] / 1500.0, t / P.H_MAX,
         running_mean_e2e / P.TAU_MS]
    feats = np.stack([m.features(pos, prior) for m in metrics])
    mask = np.asarray([m.available for m in metrics], np.float32)
    if mask.sum() == 0:
        nearest = int(np.argmin([m.distance for m in metrics]))
        mask[nearest] = 1.0
    return np.concatenate([g, feats.ravel()]).astype(np.float32), mask


class EpisodeEnv:
    """One task, real closed loop, association action each step."""

    def __init__(self, by_map, prior, device, models, step_cap=P.MAX_STEPS):
        self.by_map = by_map
        self.prior = prior
        self.device = device
        self.models = models
        self.step_cap = step_cap
        self.rasters, self.cands = RasterStore(), CandidateStore()
        self.enc4 = FrozenEncoder(device)
        self.matcher_enc = matching.Encoder(device)
        self.enc_ms = P.encode_ms()

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
        self.goal_img = matching.prepare_goal(
            self.raster, float(self.target[0]), float(self.target[1]))
        gl = self.matcher_enc.features(self.goal_img[None],
                                       ("layer2",))["layer2"][0]
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

    def metrics(self):
        return [NodeModel(n, self.pos, self.enc_ms) for n in self.nodes]

    def state(self):
        mean_e2e = float(np.mean(self.episode_e2e)) if self.episode_e2e else 0.0
        return build_state(self.pos, self.step_idx, mean_e2e,
                           self.metrics(), self.prior)

    def step(self, action):
        metrics = self.metrics()
        m = metrics[int(action)]
        self.episode_e2e.append(m.e2e_ms)
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
            agree = (self.belief.last_accept is not None
                     and float(np.linalg.norm(z - self.belief.last_accept)) <= AGREE_M)
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
        with torch.no_grad():
            cur = self.enc4(np.asarray([cur_img], np.uint8))
            batch["current"] = cur
            logits = self.models[mtype](batch)
        a = int(logits.argmax(-1).item())
        self.hist[0].append(pool_history_from_current(cur)[0].astype(np.float32))
        self.hist[1].append((float(self.pos[0]), float(self.pos[1]),
                             float(self.yaw)))
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
            st = segment_status(self.info, float(self.pos[0]),
                                float(self.pos[1]), float(nxt[0]),
                                float(nxt[1]))
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

        u = 1.0 - m.e2e_ms / P.TAU_MS
        reward = u / P.H_MAX
        done = self.done
        if done:
            T = self.step_idx
            J = 1.0 - float(np.mean(self.episode_e2e)) / P.TAU_MS
            reward += (1.0 - T / P.H_MAX) * J
            if self.stop_dist is not None:
                self.success = bool(self.stop_dist <= STOP_RADIUS)
        info = {"e2e_ms": m.e2e_ms, "inf_ms": m.inf_ms,
                "comm_ms": m.comm_ms, "node": m.node["gbs_id"],
                "model": mtype, "failure": self.failure,
                "success": self.success,
                "return_D": (1.0 - float(np.mean(self.episode_e2e)) / P.TAU_MS)
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
        (self.s[i], self.m[i], self.a[i], self.r[i], self.s2[i],
         self.m2[i], self.d[i]) = tr
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
    import csv
    raw = json.loads(P.RAW_TRAIN.read_text())
    rows = list(csv.DictReader(open(P.TRAIN_MANIFEST)))
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
    assert len(pool) == 18521, len(pool)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    dev = [pool[i] for i in order[:n_dev]]
    train = [pool[i] for i in order[n_dev:n_dev + n_train]]
    return train, dev


def homog_by_map(group, model_type):
    dep = P.deployment(group)
    out = {}
    for name, nodes in dep.items():
        out[name] = [dict(n, model_type=model_type,
                          compute_capacity_flops_per_s=10.0e12,
                          compute_tflops=10.0) for n in nodes]
    return out


def build_prior(group, device, models, dev_tasks, step_cap):
    """Dev-60 capability prior per expert on a homogeneous deployment."""
    out = {}
    for mt in P.MODEL_ORDER:
        env = EpisodeEnv(homog_by_map(group, mt), {}, device, models, step_cap)
        sr = o = spl = 0.0
        for task in dev_tasks:
            env.start(task)
            done = False
            while not done:
                metrics = env.metrics()
                a = int(np.argmin([m.distance for m in metrics]))
                _, done, _ = env.step(a)
            sr += float(env.success)
            o += float(env.min_dist <= STOP_RADIUS)
            dist = float(np.linalg.norm(env.pos - env.target))
            path = FORWARD_M * env.step_idx
            spl += (min(1.0, float(np.linalg.norm(
                np.asarray([task["start_x"], task["start_y"]]) - env.target))
                / path) if env.success and path > 0 else 0.0)
        n = len(dev_tasks)
        out[P.MODEL_LABEL[mt]] = {"sr": sr / n, "osr": o / n,
                                  "spl": spl / n, "ndtw": spl / n}
    return out


def evaluate(env, tasks, policy_fn):
    e2e, succ, steps = [], 0, []
    for task in tasks:
        env.start(task)
        s, mask = env.state()
        done = False
        while not done:
            a = policy_fn(s, mask)
            _, done, _ = env.step(a)
            s, mask = env.state()
        e2e.append(float(np.mean(env.episode_e2e)))
        succ += int(env.success)
        steps.append(env.step_idx)
    return {"n": len(tasks), "sr": succ / len(tasks),
            "e2e_mean_ms": float(np.mean(e2e)),
            "e2e_p95_ms": float(np.percentile(e2e, 95)),
            "steps_mean": float(np.mean(steps))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=P.GROUPS, required=True)
    ap.add_argument("--algorithm", choices=["dqn", "ppo"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--dev-every", type=int, default=200)
    ap.add_argument("--step-cap", type=int, default=P.MAX_STEPS)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = Path(a.out_dir)
    if not out.is_absolute():
        out = P.RES / out
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

    types = sorted(set(P.GROUP_SPECS[a.group]["models"]))
    # all five are needed once for the Dev-60 capability prior; the deployment
    # itself only references ``types``.
    models = load_models(P.MODEL_ORDER, device)
    train_tasks, dev_tasks = load_tasks_and_split()
    say(f"[start] group={a.group} algo={a.algorithm} seed={a.seed} "
        f"types={types} train={len(train_tasks)} dev={len(dev_tasks)}")

    prior_cache = P.PROTOCOL_DIR / "capability_prior_dev60_5models.json"
    if prior_cache.exists():
        prior = json.loads(prior_cache.read_text())
        say(f"[prior] reused {prior_cache.name}")
    else:
        prior = build_prior(a.group, device, models, dev_tasks, a.step_cap)
        prior_cache.write_text(json.dumps(prior, indent=2))
        say(f"[prior] computed -> {prior_cache.name}")
    (out / "capability_prior.json").write_text(json.dumps(prior, indent=2))

    # The prior only has to be computed once; the cache above is group-agnostic
    # because it is measured on 10x homogeneous 10-TFLOPS nodes.
    env = EpisodeEnv(P.deployment(a.group), prior, device, models, a.step_cap)

    best_dev = float("inf")
    best_ep = -1
    best_sr = -1.0
    ckpt_path = out / "checkpoint.pt"
    best_path = out / "best.pt"

    if a.algorithm == "dqn":
        online = QNet().to(device)
        target = QNet().to(device)
        target.load_state_dict(online.state_dict())
        opt = torch.optim.Adam(online.parameters(), lr=1e-4)
        replay = Replay(50_000, STATE_DIM, rng)
    else:
        ac = ActorCritic().to(device)
        opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
        buf = {"s": [], "m": [], "a": [], "r": [], "done": [], "lp": [],
               "v": []}
    total_steps = 0

    def save_ckpt(ep, path=None):
        payload = {"ep": ep, "best_dev": best_dev, "best_ep": best_ep,
                   "best_sr": best_sr, "total_steps": total_steps,
                   "prior": prior, "group": a.group,
                   "algorithm": a.algorithm, "seed": a.seed,
                   "state_dim": STATE_DIM, "n_act": N_ACT,
                   "episodes": a.episodes}
        if a.algorithm == "dqn":
            payload.update(online=online.state_dict(),
                           target=target.state_dict(), opt=opt.state_dict())
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
        with dev_path.open("a") as f:
            f.write(json.dumps({"episode": ep, **m}) + "\n")
        improved = (m["e2e_mean_ms"] < best_dev - 0.002 or
                    (abs(m["e2e_mean_ms"] - best_dev) <= 0.002
                     and m["sr"] > best_sr))
        if improved:
            best_dev, best_ep, best_sr = m["e2e_mean_ms"], ep, m["sr"]
            save_ckpt(ep, best_path)
            (out / "best_meta.json").write_text(json.dumps(
                {"best_dev_e2e_ms": best_dev, "best_episode": best_ep,
                 "best_dev_sr": best_sr}, indent=2))
        say(f"[dev] ep={ep} E2E={m['e2e_mean_ms']:.3f}ms "
            f"P95={m['e2e_p95_ms']:.2f} SR={m['sr']*100:.1f}% "
            f"steps={m['steps_mean']:.1f} "
            f"{'NEW_BEST' if improved else ''} "
            f"(best={best_dev:.3f}@{best_ep})")
        return m

    ma_rewards, ma_e2e, ma_succ = [], [], []
    ep = 0
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
                if replay.size >= 2000:
                    bs, bm, ba, br, bs2, bm2, bd = replay.sample(128, device)
                    q = online(bs).gather(1, ba[:, None]).squeeze(1)
                    with torch.no_grad():
                        qn = target(bs2).masked_fill(bm2 < 0.5, -1e18)
                        nxt = qn.max(dim=1).values
                        tgt = br + (1.0 - bd) * nxt
                    loss = F.smooth_l1_loss(q, tgt)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                    opt.step()
                    if total_steps % 1000 == 0:
                        target.load_state_dict(online.state_dict())
            else:
                with torch.no_grad():
                    logits, v = ac(torch.as_tensor(s[None], device=device))
                logits = logits[0].cpu().numpy().astype(np.float64).copy()
                logits[mask < 0.5] = -1e18
                p = np.exp(logits - logits.max())
                p /= p.sum()
                a_idx = int(rng.choice(N_ACT, p=p))
                logp = float(np.log(p[a_idx] + 1e-12))
                r, done, info = env.step(a_idx)
                s2, mask2 = env.state()
                buf["s"].append(s.copy())
                buf["m"].append(mask.copy())
                buf["a"].append(a_idx)
                buf["r"].append(r)
                buf["done"].append(float(done))
                buf["lp"].append(logp)
                buf["v"].append(float(v.item()))
                total_steps += 1
                if len(buf["s"]) >= 2048:
                    S = torch.as_tensor(np.asarray(buf["s"]), device=device)
                    M = torch.as_tensor(np.asarray(buf["m"]), device=device)
                    A_ = torch.as_tensor(np.asarray(buf["a"]), device=device)
                    R_ = np.asarray(buf["r"], np.float64)
                    D_ = np.asarray(buf["done"], np.float64)
                    V = np.asarray(buf["v"], np.float64)
                    LP = np.asarray(buf["lp"], np.float64)
                    adv = np.zeros_like(R_)
                    last = 0.0
                    for t in range(len(R_) - 1, -1, -1):
                        nv = 0.0 if t == len(R_) - 1 else V[t + 1]
                        delta = R_[t] + (1 - D_[t]) * nv - V[t]
                        last = delta + 0.95 * (1 - D_[t]) * last
                        adv[t] = last
                    ret = adv + V
                    adv_t = torch.as_tensor(adv, dtype=torch.float32,
                                            device=device)
                    ret_t = torch.as_tensor(ret, dtype=torch.float32,
                                            device=device)
                    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
                    old_lp = torch.as_tensor(LP, dtype=torch.float32,
                                             device=device)
                    n = len(R_)
                    idx = np.arange(n)
                    for _ in range(10):
                        rng.shuffle(idx)
                        for st in range(0, n, 256):
                            b = torch.as_tensor(idx[st:st + 256],
                                                device=device)
                            logits_b, val = ac(S[b])
                            logits_b = logits_b.masked_fill(M[b] < 0.5, -1e18)
                            logp_all = torch.log_softmax(logits_b, dim=-1)
                            lp = logp_all.gather(1, A_[b][:, None]).squeeze(1)
                            ratio = torch.exp(lp - old_lp[b])
                            surr1 = ratio * adv_t[b]
                            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b]
                            pi_loss = -torch.min(surr1, surr2).mean()
                            v_loss = 0.5 * (val - ret_t[b]).pow(2).mean()
                            ent = -(torch.exp(logp_all) * logp_all).sum(-1).mean()
                            loss = pi_loss + v_loss - 0.01 * ent
                            opt.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                            opt.step()
                    for k in buf:
                        buf[k].clear()
            s, mask = s2, mask2
            ep_reward += float(r)
            ep_e2e.append(info["e2e_ms"])
        ma_rewards.append(ep_reward)
        ma_e2e.append(float(np.mean(ep_e2e)))
        ma_succ.append(float(env.success))
        rec = {"episode": ep, "return_D": ep_reward,
               "mean_e2e_ms": float(np.mean(ep_e2e)),
               "success": int(env.success), "steps": env.step_idx,
               "failure": env.failure, "steps_total": total_steps,
               "elapsed_s": round(time.time() - t0, 1),
               "ma100_reward": float(np.mean(ma_rewards[-100:])),
               "ma100_e2e_ms": float(np.mean(ma_e2e[-100:])),
               "ma100_success": float(np.mean(ma_succ[-100:]))}
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
            save_ckpt(ep)
    final = validate(a.episodes)
    save_ckpt(a.episodes)
    assert best_path.exists(), "best.pt was not produced"
    say(f"[done] group={a.group} algo={a.algorithm} seed={a.seed} "
        f"best_dev={best_dev:.3f}ms@{best_ep} "
        f"final_dev={final['e2e_mean_ms']:.3f}ms SR={final['sr']*100:.1f}%")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
