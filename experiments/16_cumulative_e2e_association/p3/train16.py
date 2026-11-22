#!/usr/bin/env python3
"""Stage 16-P3/P4: association DRL trainer on the cumulative-E2E environment.

Algorithms
  dqn  : Double-DQN with target net and replay
  ppo  : actor-critic with GAE and clipped surrogate
  td3  : discrete TD3 (twin Q + target policy smoothing via action-mix*)
  sac  : discrete SAC (soft Q, entropy-regularised policy)

* The discrete TD3 adaptation uses twin Q-networks and a target policy that
  samples from the softmax of the online actor with a small temperature
  schedule; it never rounds a continuous action into a node id.  It is
  pre-registered as "TD3-discrete (Q-target smoothing)" and must pass its own
  smoke test before being used in the paper comparison.

Reward: pure cumulative E2E (A): r_t = -e2e_t / NORM_MS.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP16 = HERE.parent
REPO = EXP16.parents[1]
sys.path.insert(0, str(EXP16 / "p1"))
sys.path.insert(0, str(REPO / "experiments/12_hetero_mec_association/scripts"))

from env16 import (EpisodeEnv16, build_deployment_by_map, load_models,  # noqa: E402
                   load_norm, MAX_STEPS)
from train_association_rl import load_tasks_and_split  # noqa: E402

N_NODES = 10
STATE_DIM = 4 + N_NODES * 7
N_ACT = N_NODES
NORM_MS = 100.0


def state_features(env):
    metrics = env.metrics()
    g = np.asarray([env.pos[0] / 1500.0, env.pos[1] / 1500.0,
                    env.step_idx / MAX_STEPS,
                    (float(np.mean(env.episode_e2e)) / NORM_MS
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
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, N_ACT))

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, N_ACT)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class Replay:
    def __init__(self, cap=50000, seed=42):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros((cap, STATE_DIM), np.float32)
        self.m = np.zeros((cap, N_ACT), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, STATE_DIM), np.float32)
        self.m2 = np.zeros((cap, N_ACT), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.ptr = self.size = 0

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


def masked_softmax(logits, mask):
    logits = logits.masked_fill(mask < 0.5, -1e18)
    logits = torch.nan_to_num(logits, nan=-1e18, posinf=1e4, neginf=-1e18)
    p = torch.softmax(logits, -1)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return p / (p.sum(-1, keepdim=True) + 1e-12)


def train(args):
    torch.set_num_threads(int(os.environ.get("STAGE16_TORCH_THREADS", "4")))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device

    train_tasks, dev_tasks = load_tasks_and_split()
    by_map = build_deployment_by_map()
    models = load_models(device)
    norm = load_norm()
    env = EpisodeEnv16(by_map, device, models, norm, reward_variant="A",
                       norm_ms=NORM_MS, record_steps=False)

    if args.algorithm == "dqn":
        q = QNet().to(device); qt = QNet().to(device)
        qt.load_state_dict(q.state_dict())
        opt = torch.optim.Adam(q.parameters(), lr=1e-4)
        replay = Replay(seed=args.seed)
        gamma = args.gamma
    elif args.algorithm in ("ppo",):
        ac = ActorCritic().to(device)
        opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
        gamma = args.gamma
    elif args.algorithm == "td3":
        q1 = QNet().to(device); q2 = QNet().to(device)
        q1t = QNet().to(device); q2t = QNet().to(device)
        q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        pi = ActorCritic().to(device); pit = ActorCritic().to(device)
        pit.load_state_dict(pi.state_dict())
        opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
        opt_pi = torch.optim.Adam(pi.parameters(), lr=3e-4)
        replay = Replay(seed=args.seed)
        gamma = args.gamma
        tau = 0.005
        policy_delay = 2
    elif args.algorithm == "ddpg":
        q = QNet().to(device); qt = QNet().to(device)
        qt.load_state_dict(q.state_dict())
        pi = ActorCritic().to(device); pit = ActorCritic().to(device)
        pit.load_state_dict(pi.state_dict())
        opt_q = torch.optim.Adam(q.parameters(), lr=3e-4)
        opt_pi = torch.optim.Adam(pi.parameters(), lr=3e-4)
        replay = Replay(seed=args.seed)
        gamma = args.gamma
        tau = 0.005
    elif args.algorithm == "sac":
        q1 = QNet().to(device); q2 = QNet().to(device)
        q1t = QNet().to(device); q2t = QNet().to(device)
        q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        ac = ActorCritic().to(device)
        opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
        opt_pi = torch.optim.Adam(ac.parameters(), lr=3e-4)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        opt_alpha = torch.optim.Adam([log_alpha], lr=3e-4)
        target_entropy = -0.5 * math.log(N_ACT)
        replay = Replay(seed=args.seed)
        gamma = args.gamma
        tau = 0.005
    else:
        raise ValueError(args.algorithm)

    hist_path = out / "history.jsonl"
    dev_path = out / "dev.jsonl"
    total_steps = 0
    t0 = time.time()

    def record(ep, ep_reward, cum_e2e, steps, ok, failure, mean_e2e, extra=None):
        rec = {"episode": ep, "reward": ep_reward,
               "cumulative_e2e_ms": cum_e2e, "mean_e2e_ms": mean_e2e,
               "success": bool(ok), "steps": steps, "failure": failure,
               "total_steps": total_steps,
               "elapsed_sec": round(time.time() - t0, 1)}
        if extra:
            rec.update(extra)
        with hist_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def greedy(s, mask):
        with torch.no_grad():
            if args.algorithm in ("dqn",):
                qv = q(torch.as_tensor(s[None], device=device))[0].cpu().numpy()
            elif args.algorithm == "ppo":
                logits, _ = ac(torch.as_tensor(s[None], device=device))
                qv = logits[0].cpu().numpy()
            elif args.algorithm in ("td3", "ddpg"):
                logits, _ = pi(torch.as_tensor(s[None], device=device))
                qv = logits[0].cpu().numpy()
            else:
                logits, _ = ac(torch.as_tensor(s[None], device=device))
                qv = logits[0].cpu().numpy()
        qv[mask < 0.5] = -1e18
        return int(np.argmax(qv))

    def evaluate(tasks, n=60):
        succ = 0; cums = []; steps = []; fails = {}
        for task in tasks[:n]:
            env.start(task)
            s, mask = state_features(env)
            done = False
            while not done:
                a = greedy(s, mask)
                _, done, info = env.step(a)
                s, mask = state_features(env)
            succ += int(env.success)
            cums.append(float(np.sum(env.episode_e2e)))
            steps.append(env.step_idx)
            fails[env.failure] = fails.get(env.failure, 0) + 1
        return {"n": len(tasks[:n]), "sr": succ / max(len(tasks[:n]), 1),
                "mean_cum_e2e_ms": float(np.mean(cums)),
                "p95_cum_e2e_ms": float(np.percentile(cums, 95)),
                "mean_steps": float(np.mean(steps)), "failures": fails}

    def policy_state():
        if args.algorithm == "dqn":
            return {"q": q.state_dict()}
        if args.algorithm == "ppo":
            return {"ac": ac.state_dict()}
        if args.algorithm == "td3":
            return {"q1": q1.state_dict(), "q2": q2.state_dict(),
                    "pi": pi.state_dict()}
        if args.algorithm == "ddpg":
            return {"q": q.state_dict(), "pi": pi.state_dict()}
        return {"q1": q1.state_dict(), "q2": q2.state_dict(),
                "ac": ac.state_dict(),
                "log_alpha": log_alpha.detach().cpu()}

    # PPO rollout buffer
    ppo_buf = {"s": [], "m": [], "a": [], "r": [], "d": [], "lp": [], "v": []}

    for ep in range(1, args.episodes + 1):
        task = train_tasks[(ep - 1) % len(train_tasks)]
        env.start(task)
        s, mask = state_features(env)
        done = False
        ep_reward = 0.0
        while not done:
            if args.algorithm == "dqn":
                eps = max(0.05, 1.0 - total_steps / 60000)
                if random.random() < eps:
                    a = int(np.random.choice(np.flatnonzero(mask > 0.5)))
                else:
                    a = greedy(s, mask)
                r, done, info = env.step(a)
                s2, m2 = state_features(env)
                replay.add(s, mask, a, r, s2, m2, float(done))
                if replay.size >= 2000:
                    bs, bm, ba, br, bs2, bm2, bd = replay.sample(128, device)
                    with torch.no_grad():
                        qn = qt(bs2).masked_fill(bm2 < 0.5, -1e18).max(1).values
                        tgt = br + gamma * (1 - bd) * qn
                    qv = q(bs).gather(1, ba[:, None]).squeeze(1)
                    loss = F.smooth_l1_loss(qv, tgt)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                    opt.step()
                    if total_steps % 200 == 0:
                        qt.load_state_dict(q.state_dict())
                s, mask = s2, m2
            elif args.algorithm == "ppo":
                st = torch.as_tensor(s[None], device=device)
                with torch.no_grad():
                    logits, val = ac(st)
                    probs = masked_softmax(logits, torch.as_tensor(mask[None], device=device))
                a = int(torch.multinomial(probs[0], 1).item())
                lp = float(torch.log(probs[0, a] + 1e-12))
                r, done, info = env.step(a)
                s2, m2 = state_features(env)
                ppo_buf["s"].append(s); ppo_buf["m"].append(mask)
                ppo_buf["a"].append(a); ppo_buf["r"].append(r)
                ppo_buf["d"].append(float(done)); ppo_buf["lp"].append(lp)
                ppo_buf["v"].append(float(val[0]))
                s, mask = s2, m2
            elif args.algorithm == "td3":
                if random.random() < max(0.05, 1.0 - total_steps / 60000):
                    a = int(np.random.choice(np.flatnonzero(mask > 0.5)))
                else:
                    a = greedy(s, mask)
                r, done, info = env.step(a)
                s2, m2 = state_features(env)
                replay.add(s, mask, a, r, s2, m2, float(done))
                if replay.size >= 2000:
                    bs, bm, ba, br, bs2, bm2, bd = replay.sample(128, device)
                    with torch.no_grad():
                        lg2, _ = pit(bs2)
                        p2 = masked_softmax(lg2, bm2)
                        # discrete target policy smoothing: sample action from p2
                        a2 = torch.multinomial(p2, 1).squeeze(1)
                        qv = torch.min(q1t(bs2), q2t(bs2)).gather(1, a2[:, None]).squeeze(1)
                        tgt = br + gamma * (1 - bd) * qv
                    qv1 = q1(bs).gather(1, ba[:, None]).squeeze(1)
                    qv2 = q2(bs).gather(1, ba[:, None]).squeeze(1)
                    loss_q = F.smooth_l1_loss(qv1, tgt) + F.smooth_l1_loss(qv2, tgt)
                    opt_q.zero_grad(); loss_q.backward()
                    nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 1.0)
                    opt_q.step()
                    if total_steps % (policy_delay * 100) == 0:
                        lg, _ = pi(bs)
                        p = masked_softmax(lg, bm)
                        a_pi = torch.multinomial(p.detach(), 1).squeeze(1)
                        loss_pi = -torch.min(q1(bs), q2(bs)).gather(
                            1, a_pi[:, None]).mean()
                        opt_pi.zero_grad(); loss_pi.backward()
                        nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
                        opt_pi.step()
                        for t_, o_ in zip(pit.parameters(), pi.parameters()):
                            t_.data.mul_(1 - tau).add_(tau * o_.data)
                        for t_, o_ in zip(q1t.parameters(), q1.parameters()):
                            t_.data.mul_(1 - tau).add_(tau * o_.data)
                        for t_, o_ in zip(q2t.parameters(), q2.parameters()):
                            t_.data.mul_(1 - tau).add_(tau * o_.data)
                s, mask = s2, m2
            elif args.algorithm == "ddpg":
                if random.random() < max(0.05, 1.0 - total_steps / 60000):
                    a = int(np.random.choice(np.flatnonzero(mask > 0.5)))
                else:
                    a = greedy(s, mask)
                r, done, info = env.step(a)
                s2, m2 = state_features(env)
                replay.add(s, mask, a, r, s2, m2, float(done))
                if replay.size >= 2000:
                    bs, bm, ba, br, bs2, bm2, bd = replay.sample(128, device)
                    with torch.no_grad():
                        lg2, _ = pit(bs2)
                        p2 = masked_softmax(lg2, bm2)
                        a2 = p2.argmax(1)
                        qn = qt(bs2).gather(1, a2[:, None]).squeeze(1)
                        tgt = br + gamma * (1 - bd) * qn
                    qv = q(bs).gather(1, ba[:, None]).squeeze(1)
                    loss_q = F.smooth_l1_loss(qv, tgt)
                    opt_q.zero_grad(); loss_q.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt_q.step()
                    lg, _ = pi(bs)
                    p = masked_softmax(lg, bm)
                    q_all = q(bs)
                    loss_pi = -(p * q_all).sum(1).mean()
                    opt_pi.zero_grad(); loss_pi.backward()
                    nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
                    for t_, o_ in zip(qt.parameters(), q.parameters()):
                        t_.data.mul_(1 - tau).add_(tau * o_.data)
                    for t_, o_ in zip(pit.parameters(), pi.parameters()):
                        t_.data.mul_(1 - tau).add_(tau * o_.data)
                s, mask = s2, m2
            else:  # sac
                st = torch.as_tensor(s[None], device=device)
                with torch.no_grad():
                    logits, _ = ac(st)
                    probs = masked_softmax(logits, torch.as_tensor(mask[None], device=device))
                a = int(torch.multinomial(probs[0], 1).item())
                r, done, info = env.step(a)
                s2, m2 = state_features(env)
                replay.add(s, mask, a, r, s2, m2, float(done))
                if replay.size >= 2000:
                    bs, bm, ba, br, bs2, bm2, bd = replay.sample(128, device)
                    with torch.no_grad():
                        lg2, _ = ac(bs2)
                        p2 = masked_softmax(lg2, bm2)
                        logp2 = torch.log(p2 + 1e-12)
                        alpha = log_alpha.exp()
                        qn = torch.min(q1t(bs2), q2t(bs2)) - alpha * logp2
                        qn = qn.masked_fill(bm2 < 0.5, -1e18).max(1).values
                        tgt = br + gamma * (1 - bd) * qn
                    qv1 = q1(bs).gather(1, ba[:, None]).squeeze(1)
                    qv2 = q2(bs).gather(1, ba[:, None]).squeeze(1)
                    loss_q = F.mse_loss(qv1, tgt) + F.mse_loss(qv2, tgt)
                    opt_q.zero_grad(); loss_q.backward(); opt_q.step()
                    lg, _ = ac(bs)
                    p = masked_softmax(lg, bm)
                    logp = torch.log(p + 1e-12)
                    qpi = torch.min(q1(bs), q2(bs))
                    loss_pi = (p * (log_alpha.exp() * logp - qpi)).sum(1).mean()
                    opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
                    loss_a = -(p.detach() * (log_alpha.exp() * logp.detach()
                                             + target_entropy)).sum(1).mean()
                    opt_alpha.zero_grad(); loss_a.backward(); opt_alpha.step()
                    for t_, o_ in zip(q1t.parameters(), q1.parameters()):
                        t_.data.mul_(1 - tau).add_(tau * o_.data)
                    for t_, o_ in zip(q2t.parameters(), q2.parameters()):
                        t_.data.mul_(1 - tau).add_(tau * o_.data)
                s, mask = s2, m2

            ep_reward += r
            total_steps += 1
            if args.max_steps and total_steps >= args.max_steps:
                break

        if args.algorithm == "ppo" and (len(ppo_buf["s"]) >= 1024 or done):
            # GAE + PPO update over the collected rollout
            S = torch.as_tensor(np.asarray(ppo_buf["s"]), device=device)
            M = torch.as_tensor(np.asarray(ppo_buf["m"]), device=device)
            A = torch.as_tensor(np.asarray(ppo_buf["a"]), device=device)
            R = torch.as_tensor(np.asarray(ppo_buf["r"]), dtype=torch.float32, device=device)
            D = torch.as_tensor(np.asarray(ppo_buf["d"]), dtype=torch.float32, device=device)
            LP = torch.as_tensor(np.asarray(ppo_buf["lp"]), dtype=torch.float32, device=device)
            V = torch.as_tensor(np.asarray(ppo_buf["v"]), dtype=torch.float32, device=device)
            adv = torch.zeros_like(R)
            gae = 0.0
            for t in range(len(R) - 1, -1, -1):
                next_v = V[t + 1] if t + 1 < len(V) else 0.0
                delta = R[t] + gamma * (1 - D[t]) * next_v - V[t]
                gae = delta + gamma * 0.95 * (1 - D[t]) * gae
                adv[t] = gae
            ret = adv + V
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            for _ in range(4):
                logits, val = ac(S)
                p = masked_softmax(logits, M)
                logp = torch.log(p.gather(1, A[:, None]).squeeze(1) + 1e-12)
                ratio = torch.exp(logp - LP)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * (val - ret).pow(2).mean()
                loss = pi_loss + v_loss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
            for k in ppo_buf:
                ppo_buf[k] = []

        record(ep, ep_reward, info["cumulative_e2e_ms"], env.step_idx,
               env.success, env.failure,
               float(np.mean(env.episode_e2e)) if env.episode_e2e else 0.0)
        if ep % 100 == 0:
            d = evaluate(dev_tasks, 60)
            with dev_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"episode": ep, **d}) + "\n")
            print(f"[{args.algorithm}] ep {ep} dev SR={d['sr']:.3f} "
                  f"cumE2E={d['mean_cum_e2e_ms']:.0f}ms steps={d['mean_steps']:.1f}",
                  flush=True)
            torch.save({"ep": ep, "algorithm": args.algorithm,
                        "seed": args.seed, "state_dim": STATE_DIM,
                        "n_act": N_ACT, "policy": policy_state()},
                       out / "checkpoint.pt")
        if args.max_steps and total_steps >= args.max_steps:
            break

    torch.save({"ep": args.episodes, "algorithm": args.algorithm,
                "seed": args.seed, "state_dim": STATE_DIM,
                "n_act": N_ACT, "policy": policy_state()},
               out / "final.pt")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=["dqn", "ppo", "td3", "sac", "ddpg"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
