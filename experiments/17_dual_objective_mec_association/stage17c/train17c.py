#!/usr/bin/env python3
"""Stage 17-C: dual-objective DRL trainers on the Stage-17 environment.

The five learners are the Stage-16 implementations (DQN / PPO / discrete TD3 /
discrete SAC / discrete DDPG).  Update equations, optimizers, learning rates,
batch size, replay capacity, warm-up, exploration and the TD3 actor delay are
copied from `experiments/16.../p3/train16.py`; only these change:

  * state         build_state17 (127 dims instead of 74)
  * action mask   feasible_mec_mask applied identically to all five algorithms
  * reward        reward17c.DualObjectiveReward (no -legacy_e2e/100)
  * environment   EpisodeEnv17 (energy / deadline hard constraints)

Stage 16's file and checkpoints are not touched.  Not executed in Stage 17-C.
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
ROOT = HERE.parent
REPO = ROOT.parents[1]
for p in [HERE, ROOT / "src", REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from config17 import load_config                                  # noqa: E402
from env16 import build_deployment_by_map, load_models, load_norm  # noqa: E402
from env17 import EpisodeEnv17                                    # noqa: E402
from state17 import build_state17                                 # noqa: E402
from train_association_rl import load_tasks_and_split             # noqa: E402
from nets17 import (N_ACT, STATE17_DIM, ActorCritic17, QNet17,    # noqa: E402
                    mask_logits, masked_softmax)
from reward17c import DualObjectiveReward, RewardConfig           # noqa: E402

GAMMA_DEFAULT = 0.99          # Stage-16 value; see docs/16 for the analysis
HIDDEN = 256
REPLAY_CAP = 50000
BATCH = 128
WARMUP = 2000
PPO_EPOCHS = 4
PPO_CLIP = 0.2
GAE_LAMBDA = 0.95
TAU = 0.005
TARGET_SYNC = 200             # DQN target net / TD3 actor delay (Stage 16)


class Replay17:
    def __init__(self, state_dim=STATE17_DIM, n_act=N_ACT, cap=REPLAY_CAP, seed=42):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros((cap, state_dim), np.float32)
        self.m = np.zeros((cap, n_act), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, state_dim), np.float32)
        self.m2 = np.zeros((cap, n_act), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.ptr = self.size = 0

    def add(self, s, m, a, r, s2, m2, d):
        i = self.ptr
        self.s[i], self.m[i], self.a[i], self.r[i] = s, m, a, r
        self.s2[i], self.m2[i], self.d[i] = s2, m2, d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs, device):
        idx = self.rng.integers(0, self.size, size=bs)
        return tuple(torch.as_tensor(v[idx], device=device) for v in
                     (self.s, self.m, self.a, self.r, self.s2, self.m2, self.d))


def masked_argmax(logits, mask):
    v = np.asarray(logits, np.float32).copy()
    v[np.asarray(mask, np.float32) < 0.5] = -1e18
    return int(np.argmax(v))


def uniform_feasible(mask, rng):
    idx = np.flatnonzero(np.asarray(mask) > 0.5)
    return int(rng.choice(idx)) if len(idx) else None


def soft_update(target, source, tau=TAU):
    for t_, o_ in zip(target.parameters(), source.parameters()):
        t_.data.mul_(1 - tau).add_(tau * o_.data)


def make_agents(algo, device):
    if algo == "dqn":
        q, qt = QNet17().to(device), QNet17().to(device)
        qt.load_state_dict(q.state_dict())
        return {"q": q, "qt": qt, "opt": torch.optim.Adam(q.parameters(), lr=1e-4)}
    if algo == "ppo":
        ac = ActorCritic17().to(device)
        return {"ac": ac, "opt": torch.optim.Adam(ac.parameters(), lr=3e-4)}
    if algo == "td3":
        q1, q2 = QNet17().to(device), QNet17().to(device)
        q1t, q2t = QNet17().to(device), QNet17().to(device)
        q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        pi, pit = ActorCritic17().to(device), ActorCritic17().to(device)
        pit.load_state_dict(pi.state_dict())
        return {"q1": q1, "q2": q2, "q1t": q1t, "q2t": q2t, "pi": pi, "pit": pit,
                "opt_q": torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4),
                "opt_pi": torch.optim.Adam(pi.parameters(), lr=3e-4)}
    if algo == "ddpg":
        q, qt = QNet17().to(device), QNet17().to(device)
        qt.load_state_dict(q.state_dict())
        pi, pit = ActorCritic17().to(device), ActorCritic17().to(device)
        pit.load_state_dict(pi.state_dict())
        return {"q": q, "qt": qt, "pi": pi, "pit": pit,
                "opt_q": torch.optim.Adam(q.parameters(), lr=3e-4),
                "opt_pi": torch.optim.Adam(pi.parameters(), lr=3e-4)}
    if algo == "sac":
        q1, q2 = QNet17().to(device), QNet17().to(device)
        q1t, q2t = QNet17().to(device), QNet17().to(device)
        q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        ac = ActorCritic17().to(device)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        return {"q1": q1, "q2": q2, "q1t": q1t, "q2t": q2t, "ac": ac,
                "log_alpha": log_alpha,
                "opt_q": torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4),
                "opt_pi": torch.optim.Adam(ac.parameters(), lr=3e-4),
                "opt_alpha": torch.optim.Adam([log_alpha], lr=3e-4),
                "target_entropy": -0.5 * math.log(N_ACT)}
    raise ValueError(algo)


def policy_state(algo, ag):
    if algo == "dqn":
        return {"q": ag["q"].state_dict()}
    if algo == "ppo":
        return {"ac": ag["ac"].state_dict()}
    if algo == "td3":
        return {"q1": ag["q1"].state_dict(), "q2": ag["q2"].state_dict(),
                "pi": ag["pi"].state_dict()}
    if algo == "ddpg":
        return {"q": ag["q"].state_dict(), "pi": ag["pi"].state_dict()}
    return {"q1": ag["q1"].state_dict(), "q2": ag["q2"].state_dict(),
            "ac": ag["ac"].state_dict(), "log_alpha": ag["log_alpha"].detach().cpu()}


def act_policy(algo, ag, s, mask, device, rng, explore=True, total_steps=0):
    st = torch.as_tensor(s[None], device=device)
    mk = torch.as_tensor(mask[None], device=device)
    eps = max(0.05, 1.0 - total_steps / 60000)
    with torch.no_grad():
        if algo == "dqn":
            if explore and rng.random() < eps:
                return uniform_feasible(mask, rng), None, None
            return masked_argmax(ag["q"](st)[0].cpu().numpy(), mask), None, None
        if algo == "ppo":
            logits, val = ag["ac"](st)
            p = masked_softmax(logits, mk)
            a = int(torch.multinomial(p[0], 1).item())
            return a, float(torch.log(p[0, a] + 1e-12)), float(val[0])
        if algo in ("td3", "ddpg"):
            if explore and rng.random() < eps:
                return uniform_feasible(mask, rng), None, None
            logits, _ = ag["pi"](st)
            return masked_argmax(logits[0].cpu().numpy(), mask), None, None
        logits, _ = ag["ac"](st)              # sac
        p = masked_softmax(logits, mk)
        return int(torch.multinomial(p[0], 1).item()), None, None


def update_offpolicy(algo, ag, replay, device, gamma, total_steps):
    bs, bm, ba, br, bs2, bm2, bd = replay.sample(BATCH, device)
    if algo == "dqn":
        with torch.no_grad():
            qn = mask_logits(ag["qt"](bs2), bm2).max(1).values
            tgt = br + gamma * (1 - bd) * qn
        qv = ag["q"](bs).gather(1, ba[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(qv, tgt)
        ag["opt"].zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(ag["q"].parameters(), 1.0); ag["opt"].step()
        if total_steps % TARGET_SYNC == 0:
            ag["qt"].load_state_dict(ag["q"].state_dict())
        return {"loss": float(loss)}

    if algo == "td3":
        with torch.no_grad():
            lg2, _ = ag["pit"](bs2)
            p2 = masked_softmax(lg2, bm2)
            a2 = torch.multinomial(p2, 1).squeeze(1)
            qn = torch.min(ag["q1t"](bs2), ag["q2t"](bs2)).gather(1, a2[:, None]).squeeze(1)
            tgt = br + gamma * (1 - bd) * qn
        qv1 = ag["q1"](bs).gather(1, ba[:, None]).squeeze(1)
        qv2 = ag["q2"](bs).gather(1, ba[:, None]).squeeze(1)
        loss_q = F.smooth_l1_loss(qv1, tgt) + F.smooth_l1_loss(qv2, tgt)
        ag["opt_q"].zero_grad(); loss_q.backward()
        nn.utils.clip_grad_norm_(list(ag["q1"].parameters()) + list(ag["q2"].parameters()), 1.0)
        ag["opt_q"].step()
        if total_steps % TARGET_SYNC == 0:      # Stage-16 actor delay
            lg, _ = ag["pi"](bs)
            p = masked_softmax(lg, bm)
            a_pi = torch.multinomial(p.detach(), 1).squeeze(1)
            loss_pi = -torch.min(ag["q1"](bs), ag["q2"](bs)).gather(
                1, a_pi[:, None]).mean()
            ag["opt_pi"].zero_grad(); loss_pi.backward()
            nn.utils.clip_grad_norm_(ag["pi"].parameters(), 1.0); ag["opt_pi"].step()
            soft_update(ag["pit"], ag["pi"])
            soft_update(ag["q1t"], ag["q1"]); soft_update(ag["q2t"], ag["q2"])
        return {"loss": float(loss_q)}

    if algo == "ddpg":
        with torch.no_grad():
            lg2, _ = ag["pit"](bs2)
            p2 = masked_softmax(lg2, bm2)
            a2 = p2.argmax(1)
            qn = ag["qt"](bs2).gather(1, a2[:, None]).squeeze(1)
            tgt = br + gamma * (1 - bd) * qn
        qv = ag["q"](bs).gather(1, ba[:, None]).squeeze(1)
        loss_q = F.smooth_l1_loss(qv, tgt)
        ag["opt_q"].zero_grad(); loss_q.backward()
        nn.utils.clip_grad_norm_(ag["q"].parameters(), 1.0); ag["opt_q"].step()
        lg, _ = ag["pi"](bs)
        p = masked_softmax(lg, bm)
        loss_pi = -(p * ag["q"](bs)).sum(1).mean()
        ag["opt_pi"].zero_grad(); loss_pi.backward()
        nn.utils.clip_grad_norm_(ag["pi"].parameters(), 1.0); ag["opt_pi"].step()
        soft_update(ag["qt"], ag["q"]); soft_update(ag["pit"], ag["pi"])
        return {"loss": float(loss_q)}

    # sac
    with torch.no_grad():
        lg2, _ = ag["ac"](bs2)
        p2 = masked_softmax(lg2, bm2)
        logp2 = torch.log(p2 + 1e-12)
        alpha = ag["log_alpha"].exp()
        qn = torch.min(ag["q1t"](bs2), ag["q2t"](bs2)) - alpha * logp2
        qn = mask_logits(qn, bm2).max(1).values
        tgt = br + gamma * (1 - bd) * qn
    qv1 = ag["q1"](bs).gather(1, ba[:, None]).squeeze(1)
    qv2 = ag["q2"](bs).gather(1, ba[:, None]).squeeze(1)
    loss_q = F.mse_loss(qv1, tgt) + F.mse_loss(qv2, tgt)
    ag["opt_q"].zero_grad(); loss_q.backward(); ag["opt_q"].step()
    lg, _ = ag["ac"](bs)
    p = masked_softmax(lg, bm)
    logp = torch.log(p + 1e-12)
    qpi = torch.min(ag["q1"](bs), ag["q2"](bs))
    loss_pi = (p * (ag["log_alpha"].exp() * logp - qpi)).sum(1).mean()
    ag["opt_pi"].zero_grad(); loss_pi.backward(); ag["opt_pi"].step()
    loss_a = -(p.detach() * (ag["log_alpha"].exp() * logp.detach()
                             + ag["target_entropy"])).sum(1).mean()
    ag["opt_alpha"].zero_grad(); loss_a.backward(); ag["opt_alpha"].step()
    soft_update(ag["q1t"], ag["q1"]); soft_update(ag["q2t"], ag["q2"])
    return {"loss": float(loss_q)}


def ppo_update(ag, buf, device, gamma):
    S = torch.as_tensor(np.asarray(buf["s"]), device=device)
    M = torch.as_tensor(np.asarray(buf["m"]), device=device)
    A = torch.as_tensor(np.asarray(buf["a"]), device=device)
    R = torch.as_tensor(np.asarray(buf["r"]), dtype=torch.float32, device=device)
    D = torch.as_tensor(np.asarray(buf["d"]), dtype=torch.float32, device=device)
    LP = torch.as_tensor(np.asarray(buf["lp"]), dtype=torch.float32, device=device)
    V = torch.as_tensor(np.asarray(buf["v"]), dtype=torch.float32, device=device)
    adv = torch.zeros_like(R); gae = 0.0
    for t in range(len(R) - 1, -1, -1):
        nv = V[t + 1] if t + 1 < len(V) else 0.0
        delta = R[t] + gamma * (1 - D[t]) * nv - V[t]
        gae = delta + gamma * GAE_LAMBDA * (1 - D[t]) * gae
        adv[t] = gae
    ret = adv + V
    if len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    for _ in range(PPO_EPOCHS):
        logits, val = ag["ac"](S)
        p = masked_softmax(logits, M)
        logp = torch.log(p.gather(1, A[:, None]).squeeze(1) + 1e-12)
        ratio = torch.exp(logp - LP)
        pi_loss = -torch.min(ratio * adv,
                             torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv).mean()
        loss = pi_loss + 0.5 * (val - ret).pow(2).mean()
        ag["opt"].zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(ag["ac"].parameters(), 1.0)
        ag["opt"].step()


def evaluate(algo, ag, env, tasks, device, n=60):
    rows = []
    for t in tasks[:n]:
        env.start(t)
        s, meta = build_state17(env, env.cfg)
        done = False
        while not done:
            a, _, _ = act_policy(algo, ag, s, meta["feasible_mec_mask"], device,
                                 np.random.default_rng(0), explore=False)
            if a is None:
                env.step(None)
                break
            _, done, _ = env.step(a)
            s, meta = build_state17(env, env.cfg)
        rows.append({"success": bool(env.success), "failure": env.failure,
                     "latency_ms": float(env.cumulative_coll_latency_ms),
                     "energy_j": float(env.cumulative_energy_j),
                     "steps": int(env.step_idx)})
    return {"n": len(rows), "sr": float(np.mean([r["success"] for r in rows])),
            "latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
            "energy_j": float(np.mean([r["energy_j"] for r in rows])),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "failures": {f: int(sum(r["failure"] == f for r in rows))
                         for f in sorted({r["failure"] for r in rows})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=["dqn", "ppo", "td3", "sac", "ddpg"],
                    required=True)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default=str(ROOT / "configs/default.json"))
    ap.add_argument("--train-config", default=str(HERE / "train_config.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-every", type=int, default=100)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("S17C_THREADS", "4")))
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = args.device

    cfg = load_config(args.config)
    tc = json.loads(Path(args.train_config).read_text())
    rc = RewardConfig(**tc["reward"])
    reward = DualObjectiveReward(rc, forward_m=cfg.forward_m)
    train_tasks, dev_tasks = load_tasks_and_split()

    env = EpisodeEnv17(build_deployment_by_map(), dev, load_models(dev),
                       load_norm(), cfg)
    ag = make_agents(args.algorithm, dev)
    replay = Replay17(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    buf = {k: [] for k in ("s", "m", "a", "r", "d", "lp", "v")}
    hist, devp = out / "history.jsonl", out / "dev.jsonl"
    total_steps, t0 = 0, time.time()

    for ep in range(1, args.episodes + 1):
        env.start(train_tasks[(ep - 1) % len(train_tasks)])
        reward.reset(env)
        s, meta = build_state17(env, cfg)
        done, ep_r = False, 0.0
        while not done:
            mask = meta["feasible_mec_mask"]
            a, lp, val = act_policy(args.algorithm, ag, s, mask, dev, rng,
                                    explore=True, total_steps=total_steps)
            if a is None:
                _, done, info = env.step(None)
                sig = reward.step(env, info, done)
                ep_r += sig.total_training_reward
                break
            _, done, info = env.step(a)
            sig = reward.step(env, info, done)
            ep_r += sig.total_training_reward
            s2, meta2 = build_state17(env, cfg)
            if args.algorithm == "ppo":
                buf["s"].append(s); buf["m"].append(mask); buf["a"].append(a)
                buf["r"].append(sig.total_training_reward)
                buf["d"].append(float(done)); buf["lp"].append(lp); buf["v"].append(val)
            else:
                replay.add(s, mask, a, sig.total_training_reward, s2,
                           meta2["feasible_mec_mask"], float(done))
                if replay.size >= WARMUP:
                    update_offpolicy(args.algorithm, ag, replay, dev, args.gamma,
                                     total_steps)
            s, meta = s2, meta2
            total_steps += 1

        if args.algorithm == "ppo" and buf["s"]:
            ppo_update(ag, buf, dev, args.gamma)
            for k in buf:
                buf[k] = []

        rec = {"episode": ep, "reward": ep_r, "success": bool(env.success),
               "failure": env.failure, "steps": int(env.step_idx),
               "d_sum_ms": float(env.cumulative_coll_latency_ms),
               "energy_j": float(env.cumulative_energy_j),
               "energy_ratio": float(env.cumulative_energy_j)
               / max(env.energy_budget_j, 1e-9),
               "attempts": int(env.service_attempt_count),
               "inferences": int(env.completed_inference_count),
               "executed": int(env.executed_navigation_action_count),
               "model_selection": env.model_selection.astype(int).tolist(),
               "total_steps": total_steps,
               "elapsed_sec": round(time.time() - t0, 1)}
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if args.eval_every and ep % args.eval_every == 0:
            d = evaluate(args.algorithm, ag, env, dev_tasks, dev)
            with devp.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"episode": ep, **d}) + "\n")
            print(f"[{args.algorithm}] ep {ep} dev SR={d['sr']:.3f} "
                  f"Dsum={d['latency_ms']:.0f}ms E={d['energy_j']:.0f}J", flush=True)
            torch.save({"ep": ep, "algorithm": args.algorithm,
                        "state_dim": STATE17_DIM, "n_act": N_ACT,
                        "policy": policy_state(args.algorithm, ag),
                        "reward": rc.__dict__}, out / "checkpoint.pt")

    torch.save({"ep": args.episodes, "algorithm": args.algorithm,
                "state_dim": STATE17_DIM, "n_act": N_ACT,
                "policy": policy_state(args.algorithm, ag),
                "reward": rc.__dict__, "gamma": args.gamma},
               out / "final.pt")
    print(json.dumps({"algorithm": args.algorithm, "episodes": args.episodes,
                      "elapsed_sec": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
