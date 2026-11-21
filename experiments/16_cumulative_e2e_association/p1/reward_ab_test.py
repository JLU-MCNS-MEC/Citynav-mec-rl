#!/usr/bin/env python3
"""Stage 16-P1: short-horizon A/B reward comparison with DQN.

Train pool: 11A Train 4,000 (Dev-60 held out), seed 20261022 — same leak-free
split as Stage 12, never Val-Seen.

A: r = -e2e/NORM
B: r = -e2e/NORM on every step, plus FAIL_PENALTY when the episode ends
   without success (collision / OOB / step cap / premature stop).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP16 = HERE.parent
REPO = EXP16.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/12_hetero_mec_association/scripts"))

from env16 import (EpisodeEnv16, build_deployment_by_map, load_models,  # noqa: E402
                   load_norm, MODEL_ORDER, MAX_STEPS)
import protocol as P12  # noqa: E402
from train_association_rl import load_tasks_and_split  # noqa: E402

N_NODES = 10
STATE_DIM = 4 + N_NODES * 7
N_ACT = N_NODES


def state_features(env):
    metrics = env.metrics()
    g = np.asarray([env.pos[0] / 1500.0, env.pos[1] / 1500.0,
                    env.step_idx / MAX_STEPS,
                    (float(np.mean(env.episode_e2e)) / 100.0
                     if env.episode_e2e else 0.0)], np.float32)
    feats = []
    for m in metrics:
        dx = (m.node["x"] - env.pos[0]) / 500.0
        dy = (m.node["y"] - env.pos[1]) / 500.0
        feats.append([dx, dy, m.distance / 500.0, m.comm_ms / 100.0,
                      m.inf_ms / 25.0, m.e2e_ms / 100.0, float(m.available)])
    mask = np.asarray([m.available for m in metrics], np.float32)
    if mask.sum() == 0:
        mask[int(np.argmin([m.distance for m in metrics]))] = 1.0
    return np.concatenate([g, np.asarray(feats, np.float32).ravel()]), mask


class QNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, N_ACT))

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, cap=30000, seed=42):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros((cap, STATE_DIM), np.float32)
        self.m = np.zeros((cap, N_ACT), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, STATE_DIM), np.float32)
        self.m2 = np.zeros((cap, N_ACT), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, m, a, r, s2, m2, d):
        i = self.ptr
        self.s[i] = s; self.m[i] = m; self.a[i] = a; self.r[i] = r
        self.s2[i] = s2; self.m2[i] = m2; self.d[i] = d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs, device):
        idx = self.rng.integers(0, self.size, size=bs)
        return tuple(torch.as_tensor(v[idx], device=device) for v in
                     (self.s, self.m, self.a, self.r, self.s2, self.m2, self.d))


def train_variant(tasks, dev_tasks, env, variant, episodes, device, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    q = QNet().to(device)
    qt = QNet().to(device); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    replay = Replay(seed=seed)
    gamma = 0.99
    total_steps = 0
    rec = []
    t0 = time.time()
    for ep in range(episodes):
        task = tasks[ep % len(tasks)]
        env.reward_variant = variant
        env.start(task)
        s, mask = state_features(env)
        done = False
        ep_reward, ep_e2e = 0.0, 0.0
        while not done:
            eps = max(0.05, 1.0 - total_steps / 60000)
            if random.random() < eps:
                valid = np.flatnonzero(mask > 0.5)
                a = int(np.random.choice(valid))
            else:
                with torch.no_grad():
                    qv = q(torch.as_tensor(s[None], device=device))[0].cpu().numpy()
                qv[mask < 0.5] = -1e18
                a = int(np.argmax(qv))
            r, done, info = env.step(a)
            s2, m2 = state_features(env)
            replay.add(s, mask, a, r, s2, m2, float(done))
            ep_reward += r
            ep_e2e = info["cumulative_e2e_ms"]
            s, mask = s2, m2
            total_steps += 1
            if replay.size >= 2000:
                bs, bm, ba, br, bs2, bm2, bd = replay.sample(128, device)
                with torch.no_grad():
                    qn = qt(bs2).masked_fill(bm2 < 0.5, -1e18).max(dim=1).values
                    tgt = br + gamma * (1.0 - bd) * qn
                qv = q(bs).gather(1, ba[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(qv, tgt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()
                if total_steps % 200 == 0:
                    qt.load_state_dict(q.state_dict())
        rec.append({"episode": ep + 1, "variant": variant,
                    "reward": ep_reward, "cumulative_e2e_ms": ep_e2e,
                    "success": bool(env.success), "steps": env.step_idx,
                    "failure": env.failure,
                    "mean_e2e_ms": float(np.mean(env.episode_e2e))})
        if (ep + 1) % 50 == 0:
            ma = rec[-50:]
            print(f"[{variant}] ep {ep+1} rewardMA={np.mean([r['reward'] for r in ma]):.2f} "
                  f"e2eMA={np.mean([r['cumulative_e2e_ms'] for r in ma]):.0f}ms "
                  f"SR={np.mean([r['success'] for r in ma]):.3f} "
                  f"steps={np.mean([r['steps'] for r in ma]):.1f}",
                  flush=True)
    # greedy dev evaluation
    dev = {"n": 0, "sr": 0, "cum_e2e": [], "steps": [], "premature": 0,
           "failures": {}}
    for task in dev_tasks:
        env.reward_variant = "A"
        env.start(task)
        s, mask = state_features(env)
        done = False
        while not done:
            with torch.no_grad():
                qv = q(torch.as_tensor(s[None], device=device))[0].cpu().numpy()
            qv[mask < 0.5] = -1e18
            a = int(np.argmax(qv))
            _, done, info = env.step(a)
            s, mask = state_features(env)
        dev["n"] += 1
        dev["sr"] += int(env.success)
        dev["cum_e2e"].append(float(np.sum(env.episode_e2e)))
        dev["steps"].append(env.step_idx)
        if env.failure == "premature_stop":
            dev["premature"] += 1
        dev["failures"][env.failure] = dev["failures"].get(env.failure, 0) + 1
    out = {"variant": variant, "episodes": episodes,
           "elapsed_sec": round(time.time() - t0, 1),
           "train": rec,
           "dev": {"n": dev["n"], "sr": dev["sr"] / max(dev["n"], 1),
                   "mean_cum_e2e_ms": float(np.mean(dev["cum_e2e"])),
                   "p95_cum_e2e_ms": float(np.percentile(dev["cum_e2e"], 95)),
                   "mean_steps": float(np.mean(dev["steps"])),
                   "premature_stop_rate": dev["premature"] / max(dev["n"], 1),
                   "failures": dev["failures"]}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--norm-ms", type=float, default=60.0)
    ap.add_argument("--fail-penalty", type=float, default=-50.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    train_tasks, dev_tasks = load_tasks_and_split()
    by_map = build_deployment_by_map()
    models = load_models(args.device)
    norm = load_norm()
    env = EpisodeEnv16(by_map, args.device, models, norm,
                       norm_ms=args.norm_ms, fail_penalty=args.fail_penalty,
                       record_steps=False)
    print(f"train={len(train_tasks)} dev={len(dev_tasks)} "
          f"norm_ms={args.norm_ms} fail_penalty={args.fail_penalty}", flush=True)

    results = {}
    for variant in ("A", "B"):
        res = train_variant(train_tasks, dev_tasks, env, variant,
                            args.episodes, args.device)
        results[variant] = res
        (out / f"reward_{variant}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({"variant": variant, "dev": res["dev"]}, indent=2))
    (out / "reward_ab_summary.json").write_text(json.dumps(
        {v: results[v]["dev"] for v in results}, indent=2))


if __name__ == "__main__":
    main()
