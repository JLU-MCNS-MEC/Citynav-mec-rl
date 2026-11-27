#!/usr/bin/env python3
"""Stage 17-C-R1: dual-objective DRL trainers (five algorithms).

Inheritance: update equations follow Stage 16 (`p3/train16.py`) with these
R1 changes, all of them required by the review:

  * reward        r_t = lambda_Q * 1[success] - lambda_D * D_coll / D_max
                  (D_max read from the Stage-17 env config; no D_ref, no
                  progress reward)
  * gamma         1.0 for every algorithm (undiscounted episodic objective)
  * TD3 actor     differentiable policy expectation over feasible actions
  * SAC target    soft value  sum_m pi(m|s') [minQ' - alpha log pi]
  * DQN target    masked max, bootstrap 0 on terminal / empty masks
  * masks         one feasible_mec_mask definition for all five algorithms
  * config        CLI > train_config.json > code defaults, effective config
                  printed and stored in every checkpoint
  * checkpoints   best.pt (Dev dual-objective score) and last.pt

Not executed in Stage 17-C-R1.
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
                    masked_expectation, masked_max, masked_softmax)
from reward17c import (DualObjectiveRewardR1, RewardConfigR1)     # noqa: E402

DEFAULTS = {
    "gamma": 1.0,
    "episodes": 4000,
    "seed": 42,
    "batch": 128,
    "warmup": 2000,
    "replay_cap": 50000,
    "hidden": 256,
    "lr_dqn": 1e-4,
    "lr_actor": 3e-4,
    "tau": 0.005,
    "target_sync_steps": 200,
    "ppo_epochs": 4,
    "ppo_clip": 0.2,
    "gae_lambda": 0.95,
    "eval_every": 100,
    "sac_eta": 0.5,
    "policy_delay": 2,
    "sac_critic_clip": 1.0,
    "lambda_quality": 1.0,
    "lambda_latency": 0.5,
}

CLI_KEYS = ("gamma", "episodes", "seed", "eval_every")

METRIC_KEYS = (
    "loss_q", "loss_pi", "loss_v", "grad_q", "grad_pi",
    "alpha", "entropy", "sac_target_entropy", "mean_env_reward",
    "mean_entropy_bonus", "mean_bellman_target", "mean_q1", "mean_q2",
    "mean_q_min", "critic_td_error",
    "approx_kl", "clip_frac", "policy_entropy", "actor_param_delta",
    "actor_logits_std", "actor_max_prob", "actor_entropy",
    "q_action_std", "q_action_range",
    "critic_updates", "actor_updates", "target_updates",
)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def resolve_config(cli: dict, json_path: Path) -> dict:
    """Precedence: explicit CLI value > train_config.json > DEFAULTS."""
    cfg = dict(DEFAULTS)
    if json_path.exists():
        data = json.loads(json_path.read_text())
        for k, v in data.items():
            if k in DEFAULTS and isinstance(v, (int, float, bool)):
                cfg[k] = v
            elif k == "reward":
                for rk, rv in v.items():
                    if rk in ("lambda_quality", "lambda_latency"):
                        cfg[rk] = rv
    for k in CLI_KEYS:
        if cli.get(k) is not None:
            cfg[k] = cli[k]
    return cfg


def dev_score(sr: float, mean_d_sum_ms: float, lam_q: float, lam_d: float,
              d_max_ms: float, t_max: int) -> float:
    """Model-selection score (NOT a new latency constraint)."""
    return lam_q * sr - lam_d * mean_d_sum_ms / max(d_max_ms * t_max, 1e-9)


def is_better(score: float, best: float) -> bool:
    return best is None or score > best


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------
class Replay17:
    def __init__(self, state_dim, n_act, cap, seed):
        self.cap, self.rng = cap, np.random.default_rng(seed)
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


# --------------------------------------------------------------------------
# agents
# --------------------------------------------------------------------------
def make_agents(algo, device, cfg):
    hidden = int(cfg["hidden"])
    lr = float(cfg["lr_actor"])
    if algo == "dqn":
        q, qt = QNet17(hidden=hidden).to(device), QNet17(hidden=hidden).to(device)
        qt.load_state_dict(q.state_dict())
        return {"q": q, "qt": qt,
                "opt": torch.optim.Adam(q.parameters(), lr=float(cfg["lr_dqn"]))}
    if algo == "ppo":
        ac = ActorCritic17(hidden=hidden).to(device)
        return {"ac": ac, "opt": torch.optim.Adam(ac.parameters(), lr=lr)}
    if algo == "td3":
        q1, q2 = QNet17(hidden=hidden).to(device), QNet17(hidden=hidden).to(device)
        q1t, q2t = QNet17(hidden=hidden).to(device), QNet17(hidden=hidden).to(device)
        q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        pi, pit = ActorCritic17(hidden=hidden).to(device), ActorCritic17(hidden=hidden).to(device)
        pit.load_state_dict(pi.state_dict())
        return {"q1": q1, "q2": q2, "q1t": q1t, "q2t": q2t, "pi": pi, "pit": pit,
                "opt_q": torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr),
                "opt_pi": torch.optim.Adam(pi.parameters(), lr=lr)}
    if algo == "ddpg":
        q, qt = QNet17(hidden=hidden).to(device), QNet17(hidden=hidden).to(device)
        qt.load_state_dict(q.state_dict())
        pi, pit = ActorCritic17(hidden=hidden).to(device), ActorCritic17(hidden=hidden).to(device)
        pit.load_state_dict(pi.state_dict())
        return {"q": q, "qt": qt, "pi": pi, "pit": pit,
                "opt_q": torch.optim.Adam(q.parameters(), lr=lr),
                "opt_pi": torch.optim.Adam(pi.parameters(), lr=lr)}
    if algo == "sac":
        q1, q2 = QNet17(hidden=hidden).to(device), QNet17(hidden=hidden).to(device)
        q1t, q2t = QNet17(hidden=hidden).to(device), QNet17(hidden=hidden).to(device)
        q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        ac = ActorCritic17(hidden=hidden).to(device)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        return {"q1": q1, "q2": q2, "q1t": q1t, "q2t": q2t, "ac": ac,
                "log_alpha": log_alpha,
                "opt_q": torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr),
                "opt_pi": torch.optim.Adam(ac.parameters(), lr=lr),
                "opt_alpha": torch.optim.Adam([log_alpha], lr=lr),
                # Stage-16 definition, kept and documented as a choice
                "target_entropy": -0.5 * math.log(N_ACT)}
    raise ValueError(algo)


def grad_norm(params) -> float:
    tot = 0.0
    for prm in params:
        if prm.grad is not None:
            tot += float(prm.grad.detach().pow(2).sum())
    return math.sqrt(tot)


def soft_update(target, source, tau):
    for t_, o_ in zip(target.parameters(), source.parameters()):
        t_.data.mul_(1 - tau).add_(tau * o_.data)


def policy_state(algo, ag):
    if algo == "dqn":
        return {"q": ag["q"].state_dict(), "qt": ag["qt"].state_dict()}
    if algo == "ppo":
        return {"ac": ag["ac"].state_dict()}
    if algo == "td3":
        return {"q1": ag["q1"].state_dict(), "q2": ag["q2"].state_dict(),
                "q1t": ag["q1t"].state_dict(), "q2t": ag["q2t"].state_dict(),
                "pi": ag["pi"].state_dict(), "pit": ag["pit"].state_dict()}
    if algo == "ddpg":
        return {"q": ag["q"].state_dict(), "qt": ag["qt"].state_dict(),
                "pi": ag["pi"].state_dict(), "pit": ag["pit"].state_dict()}
    return {"q1": ag["q1"].state_dict(), "q2": ag["q2"].state_dict(),
            "q1t": ag["q1t"].state_dict(), "q2t": ag["q2t"].state_dict(),
            "ac": ag["ac"].state_dict(), "log_alpha": ag["log_alpha"].detach().cpu()}


def optimizer_state(algo, ag):
    keys = [k for k in ag if k.startswith("opt")]
    return {k: ag[k].state_dict() for k in keys}


# --------------------------------------------------------------------------
# acting
# --------------------------------------------------------------------------
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
            if float(p.sum()) < 1e-12:
                return None, None, None
            if not explore:
                return int(p[0].argmax()), None, float(val[0])
            a = int(torch.multinomial(p[0], 1).item())
            return a, float(torch.log(p[0, a] + 1e-12)), float(val[0])
        if algo in ("td3", "ddpg"):
            if explore and rng.random() < eps:
                return uniform_feasible(mask, rng), None, None
            logits, _ = ag["pi"](st)
            return masked_argmax(logits[0].cpu().numpy(), mask), None, None
        logits, _ = ag["ac"](st)              # sac
        p = masked_softmax(logits, mk)
        if float(p.sum()) < 1e-12:
            return None, None, None
        if not explore:
            return int(p[0].argmax()), None, None
        return int(torch.multinomial(p[0], 1).item()), None, None


# --------------------------------------------------------------------------
# updates
# --------------------------------------------------------------------------
def update_offpolicy(algo, ag, replay, device, gamma, total_steps, cfg):
    bs, bm, ba, br, bs2, bm2, bd = replay.sample(int(cfg["batch"]), device)
    tau = float(cfg["tau"])
    sync = int(cfg["target_sync_steps"])

    if algo == "dqn":
        with torch.no_grad():
            nxt = masked_max(ag["qt"](bs2), bm2)      # 0 on empty/terminal masks
            tgt = br + gamma * (1 - bd) * nxt
        qv = ag["q"](bs).gather(1, ba[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(qv, tgt)
        ag["opt"].zero_grad(); loss.backward()
        gn = grad_norm(ag["q"].parameters())
        nn.utils.clip_grad_norm_(ag["q"].parameters(), 1.0); ag["opt"].step()
        if total_steps % sync == 0:
            ag["qt"].load_state_dict(ag["q"].state_dict())
        return {"loss_q": float(loss.detach()), "grad_q": gn}

    if algo == "td3":
        with torch.no_grad():
            lg2, _ = ag["pit"](bs2)
            p2 = masked_softmax(lg2, bm2)
            # target policy smoothing, but never sample from a zero row:
            # terminal / no-feasible samples get bootstrap 0 instead
            live = (bm2.sum(1) > 0.5) & (bd < 0.5)
            a2 = torch.zeros(p2.shape[0], dtype=torch.long, device=p2.device)
            if bool(live.any()):
                a2[live] = torch.multinomial(p2[live], 1).squeeze(1)
            qn = torch.min(ag["q1t"](bs2), ag["q2t"](bs2)).gather(
                1, a2[:, None]).squeeze(1)
            qn = torch.where(live, qn, torch.zeros_like(qn))
            tgt = br + gamma * (1 - bd) * qn
        qv1 = ag["q1"](bs).gather(1, ba[:, None]).squeeze(1)
        qv2 = ag["q2"](bs).gather(1, ba[:, None]).squeeze(1)
        loss_q = F.smooth_l1_loss(qv1, tgt) + F.smooth_l1_loss(qv2, tgt)
        ag["opt_q"].zero_grad(); loss_q.backward()
        gn_q = grad_norm(list(ag["q1"].parameters()) + list(ag["q2"].parameters()))
        nn.utils.clip_grad_norm_(list(ag["q1"].parameters()) + list(ag["q2"].parameters()), 1.0)
        ag["opt_q"].step()
        ag["critic_updates"] = ag.get("critic_updates", 0) + 1
        if ag["critic_updates"] % int(cfg.get("policy_delay", 2)) == 0:
            # P1: one actor update per `policy_delay` CRITIC updates
            # (was tied to total_steps % 200, i.e. almost never).
            lg, _ = ag["pi"](bs)
            p = masked_softmax(lg, bm)
            q = torch.min(ag["q1"](bs), ag["q2"](bs))
            loss_pi = -masked_expectation(q, p).mean()
            ag["opt_pi"].zero_grad(); loss_pi.backward()
            gn_pi = grad_norm(ag["pi"].parameters())
            nn.utils.clip_grad_norm_(ag["pi"].parameters(), 1.0); ag["opt_pi"].step()
            soft_update(ag["pit"], ag["pi"], tau)
            soft_update(ag["q1t"], ag["q1"], tau); soft_update(ag["q2t"], ag["q2"], tau)
            ag["actor_updates"] = ag.get("actor_updates", 0) + 1
            ag["target_updates"] = ag.get("target_updates", 0) + 1
            return {"loss_q": float(loss_q.detach()), "grad_q": gn_q,
                    "loss_pi": float(loss_pi.detach()), "grad_pi": gn_pi,
                    "critic_updates": ag["critic_updates"],
                    "actor_updates": ag["actor_updates"],
                    "target_updates": ag["target_updates"]}
        return {"loss_q": float(loss_q.detach()), "grad_q": gn_q,
                "critic_updates": ag["critic_updates"],
                "actor_updates": ag.get("actor_updates", 0),
                "target_updates": ag.get("target_updates", 0)}

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
        gn_q = grad_norm(ag["q"].parameters())
        nn.utils.clip_grad_norm_(ag["q"].parameters(), 1.0); ag["opt_q"].step()
        lg, _ = ag["pi"](bs)
        p = masked_softmax(lg, bm)
        q_all = ag["q"](bs)
        loss_pi = -masked_expectation(q_all, p).mean()   # differentiable
        ag["opt_pi"].zero_grad(); loss_pi.backward()
        gn_pi = grad_norm(ag["pi"].parameters())
        before = [prm.detach().clone() for prm in ag["pi"].parameters()]
        nn.utils.clip_grad_norm_(ag["pi"].parameters(), 1.0); ag["opt_pi"].step()
        delta = math.sqrt(sum(float((a - b).detach().pow(2).sum())
                              for a, b in zip(before, ag["pi"].parameters())))
        soft_update(ag["qt"], ag["q"], tau); soft_update(ag["pit"], ag["pi"], tau)
        # DDPG diagnostics (logits saturation vs flat critic)
        p_d = p.detach(); lg_d = lg.detach()
        return {"loss_q": float(loss_q.detach()), "grad_q": gn_q,
                "loss_pi": float(loss_pi.detach()), "grad_pi": gn_pi,
                "actor_logits_std": float(lg_d.std()),
                "actor_max_prob": float(p_d.max(1).values.mean()),
                "actor_entropy": float(-(p_d * torch.log(p_d + 1e-12)).sum(1).mean()),
                "q_action_std": float(q_all.detach().std(1).mean()),
                "q_action_range": float((q_all.detach().max(1).values
                                         - q_all.detach().min(1).values).mean()),
                "actor_param_delta": delta}

    # sac
    with torch.no_grad():
        lg2, _ = ag["ac"](bs2)
        p2 = masked_softmax(lg2, bm2)
        logp2 = torch.log(p2 + 1e-12)
        alpha = ag["log_alpha"].exp()
        qmin = torch.min(ag["q1t"](bs2), ag["q2t"](bs2))
        # soft value = policy expectation over feasible nodes (zero if none)
        v_next = masked_expectation(qmin - alpha * logp2, p2)
        tgt = br + gamma * (1 - bd) * v_next
    qv1 = ag["q1"](bs).gather(1, ba[:, None]).squeeze(1)
    qv2 = ag["q2"](bs).gather(1, ba[:, None]).squeeze(1)
    loss_q = F.mse_loss(qv1, tgt) + F.mse_loss(qv2, tgt)
    ag["opt_q"].zero_grad(); loss_q.backward()
    gn_q = grad_norm(list(ag["q1"].parameters()) + list(ag["q2"].parameters()))
    # P1 stability fix (recorded separately): clip the critic gradient so the
    # large TD errors seen in P0 cannot blow up the shared trunk.
    nn.utils.clip_grad_norm_(list(ag["q1"].parameters()) + list(ag["q2"].parameters()),
                             float(cfg.get("sac_critic_clip", 1.0)))
    ag["opt_q"].step()
    lg, _ = ag["ac"](bs)
    p = masked_softmax(lg, bm)
    logp = torch.log(p + 1e-12)
    qpi = torch.min(ag["q1"](bs), ag["q2"](bs))
    loss_pi = masked_expectation(ag["log_alpha"].exp() * logp - qpi, p).mean()
    ag["opt_pi"].zero_grad(); loss_pi.backward()
    gn_pi = grad_norm(ag["ac"].parameters())
    ag["opt_pi"].step()
    ent_val = float(-(p.detach() * logp.detach()).sum(1).mean())
    # temperature: L_alpha = E[log alpha * (H_pi - H_target)],
    # H_target(s) = eta * log(K_feasible(s)); rows without feasible nodes are
    # excluded from the alpha update.
    ent = -(p.detach() * logp.detach()).sum(1)
    k_feas = bm.sum(1)
    h_target = float(cfg.get("sac_eta", 0.5)) * torch.log(k_feas.clamp(min=1.0))
    valid = k_feas > 0.5
    if bool(valid.any()):
        loss_a = (ag["log_alpha"] * (ent - h_target))[valid].mean()
        ag["opt_alpha"].zero_grad(); loss_a.backward(); ag["opt_alpha"].step()
    soft_update(ag["q1t"], ag["q1"], tau); soft_update(ag["q2t"], ag["q2"], tau)
    alpha_v = float(ag["log_alpha"].exp().detach())
    return {"loss_q": float(loss_q.detach()), "grad_q": gn_q,
            "loss_pi": float(loss_pi.detach()), "grad_pi": gn_pi,
            "alpha": alpha_v, "entropy": ent_val,
            "sac_target_entropy": float(h_target.mean().detach()),
            "mean_env_reward": float(br.mean()),
            "mean_entropy_bonus": float((alpha * ent).mean()),
            "mean_bellman_target": float(tgt.mean()),
            "mean_q1": float(qv1.mean()), "mean_q2": float(qv2.mean()),
            "mean_q_min": float(torch.min(qv1, qv2).mean()),
            "critic_td_error": float((qv1 - tgt).abs().mean())}


def ppo_update(ag, buf, device, cfg):
    gamma = float(cfg["gamma"])
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
        gae = delta + gamma * float(cfg["gae_lambda"]) * (1 - D[t]) * gae
        adv[t] = gae
    ret = adv + V
    if len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    clip = float(cfg["ppo_clip"])
    for _ in range(int(cfg["ppo_epochs"])):
        logits, val = ag["ac"](S)
        p = masked_softmax(logits, M)
        logp = torch.log(p.gather(1, A[:, None]).squeeze(1) + 1e-12)
        ratio = torch.exp(logp - LP)
        pi_loss = -torch.min(ratio * adv,
                             torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
        loss = pi_loss + 0.5 * (val - ret).pow(2).mean()
        ag["opt"].zero_grad(); loss.backward()
        gn = grad_norm(ag["ac"].parameters())
        nn.utils.clip_grad_norm_(ag["ac"].parameters(), 1.0)
        ag["opt"].step()
        with torch.no_grad():
            approx_kl = float((LP - logp).mean())
            clip_frac = float(((ratio - 1.0).abs() > clip).float().mean())
            ent_pi = float(-(p * torch.log(p + 1e-12)).sum(1).mean())
        last = {"loss_pi": float(pi_loss.detach()),
                "loss_v": float(0.5 * (val - ret).pow(2).mean().detach()),
                "grad_pi": gn, "approx_kl": approx_kl,
                "clip_frac": clip_frac, "policy_entropy": ent_pi}
    return last


# --------------------------------------------------------------------------
# evaluation / checkpoints
# --------------------------------------------------------------------------
def evaluate(algo, ag, env, tasks, device, n=60, prev_actions=None):
    """Deterministic Dev evaluation with policy diagnostics and per-task rows."""
    rows, action_hist, ents, maxps = [], {}, [], []
    for t in tasks[:n]:
        env.start(t)
        s, meta = build_state17(env, env.cfg)
        done, acts = False, []
        while not done:
            mask = meta["feasible_mec_mask"]
            a, _, _ = act_policy(algo, ag, s, mask, device,
                                 np.random.default_rng(0), explore=False)
            if a is None:
                env.step(None)
                break
            acts.append(int(a))
            st = torch.as_tensor(s[None], device=device)
            with torch.no_grad():
                if algo == "dqn":
                    lg = ag["q"](st)[0]
                else:
                    lg = ag["pi"](st)[0] if algo in ("td3", "ddpg") else ag["ac"](st)[0]
                pr = masked_softmax(lg[None], torch.as_tensor(mask[None], device=device))
            if float(pr.sum()) > 1e-12:
                ents.append(float(-(pr * torch.log(pr + 1e-12)).sum()))
                maxps.append(float(pr.max()))
            _, done, _ = env.step(a)
            s, meta = build_state17(env, env.cfg)
        action_hist[int(t["index"])] = acts
        rows.append({"task_index": int(t["index"]), "success": bool(env.success),
                     "failure": env.failure,
                     "d_sum_ms": float(env.cumulative_coll_latency_ms),
                     "energy_j": float(env.cumulative_energy_j),
                     "steps": int(env.step_idx),
                     "actions": acts})
    change_ratio = None
    if prev_actions:
        same = tot = 0
        for k, acts in action_hist.items():
            if k in prev_actions:
                for x, y in zip(acts, prev_actions[k]):
                    tot += 1; same += int(x == y)
        change_ratio = 1.0 - same / max(tot, 1)
    hist = np.zeros(N_ACT, int)
    for acts in action_hist.values():
        for a in acts:
            hist[a] += 1
    return {"n": len(rows), "sr": float(np.mean([r["success"] for r in rows])),
            "d_sum_ms": float(np.mean([r["d_sum_ms"] for r in rows])),
            "energy_j": float(np.mean([r["energy_j"] for r in rows])),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "failures": {f: int(sum(r["failure"] == f for r in rows))
                         for f in sorted({r["failure"] for r in rows})},
            "policy_entropy_dev": float(np.mean(ents)) if ents else None,
            "max_prob_dev": float(np.mean(maxps)) if maxps else None,
            "action_hist": hist.tolist(),
            "action_change_ratio": change_ratio,
            "per_task": rows,
            "_actions": action_hist}


def save_checkpoint(path, algo, ag, cfg, env_cfg, episode, global_step,
                    dev_metrics, dev_score_value, best, state=None):
    torch.save({
        "algorithm": algo, "state_dim": STATE17_DIM, "n_act": N_ACT,
        "policy": policy_state(algo, ag), "optimizers": optimizer_state(algo, ag),
        "config": cfg, "gamma": float(cfg["gamma"]),
        "reward": {"lambda_quality": float(cfg["lambda_quality"]),
                   "lambda_latency": float(cfg["lambda_latency"])},
        "environment": {"d_max_ms": env_cfg.d_max_ms,
                        "battery_j": env_cfg.task_energy_budget_j,
                        "coverage_m": env_cfg.coverage_m,
                        "v_forward_mps": env_cfg.v_forward_mps},
        "episode": episode, "global_step": global_step,
        "dev_metrics": dev_metrics, "dev_score": dev_score_value,
        "best": bool(best),
        "counters": {k: ag[k] for k in ("critic_updates", "actor_updates",
                                        "target_updates") if k in ag},
        "state": state or {},
    }, path)


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=["dqn", "ppo", "td3", "sac", "ddpg"],
                    required=True)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default=str(ROOT / "configs/default.json"))
    ap.add_argument("--train-config", default=str(HERE / "train_config.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default="", help="checkpoint to continue from")
    args = ap.parse_args()

    cfg = resolve_config({"gamma": args.gamma, "episodes": args.episodes,
                          "seed": args.seed, "eval_every": args.eval_every},
                         Path(args.train_config))
    resume_sd = None
    if args.resume:
        resume_sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved = dict(resume_sd.get("config", {}))
        saved.update({k: v for k, v in {
            "episodes": args.episodes if args.episodes is not None else None,
            "gamma": args.gamma, "seed": args.seed,
            "eval_every": args.eval_every}.items() if v is not None})
        cfg = {**cfg, **saved}
        print(f"[resume] from {args.resume} at episode {resume_sd.get('episode')}",
              flush=True)
    torch.set_num_threads(int(os.environ.get("S17C_THREADS", "4")))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "effective_config.json").write_text(json.dumps(cfg, indent=2))
    print("[config] " + json.dumps(cfg), flush=True)

    env_cfg = load_config(args.config)
    reward = DualObjectiveRewardR1(
        RewardConfigR1(cfg["lambda_quality"], cfg["lambda_latency"]),
        d_max_ms=env_cfg.d_max_ms, t_max=env_cfg.max_steps)
    train_tasks, dev_tasks = load_tasks_and_split()
    env = EpisodeEnv17(build_deployment_by_map(), args.device,
                       load_models(args.device), load_norm(), env_cfg)
    ag = make_agents(args.algorithm, args.device, cfg)
    start_ep = 0
    if resume_sd is not None:
        pol = resume_sd["policy"]
        for key in ("q", "qt", "q1", "q2", "q1t", "q2t", "pi", "pit", "ac"):
            if key in pol:
                ag[key].load_state_dict(pol[key])
        if "log_alpha" in pol:
            with torch.no_grad():
                ag["log_alpha"].copy_(pol["log_alpha"].to(args.device))
        for k, sd in resume_sd.get("optimizers", {}).items():
            ag[k].load_state_dict(sd)
        for k, v in resume_sd.get("counters", {}).items():
            ag[k] = v
        start_ep = int(resume_sd.get("episode", 0))
    replay = Replay17(STATE17_DIM, N_ACT, int(cfg["replay_cap"]), int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    buf = {k: [] for k in ("s", "m", "a", "r", "d", "lp", "v")}
    hist, devp = out / "history.jsonl", out / "dev.jsonl"
    total_steps, t0 = 0, time.time()
    updates, last_metrics, best_score = 0, {}, None

    perm_rng = np.random.default_rng(int(cfg["seed"]))
    perm = perm_rng.permutation(len(train_tasks))
    ptr, cycle = 0, 0
    prev_actions = None
    if resume_sd is not None:
        st = resume_sd.get("state", {})
        total_steps = int(resume_sd.get("global_step", 0))
        if st.get("perm"):
            perm = np.asarray(st["perm"], np.int64)
            ptr = int(st.get("ptr", 0)); cycle = int(st.get("cycle", 0))
            perm_rng.bit_generator.state = st["perm_rng"]
        if st.get("torch_rng") is not None:
            torch.set_rng_state(torch.as_tensor(st["torch_rng"], dtype=torch.uint8))
        if st.get("np_rng") is not None:
            rng.bit_generator.state = st["np_rng"]
    for ep in range(start_ep + 1, int(cfg["episodes"]) + 1):
        if ptr >= len(perm):
            perm = perm_rng.permutation(len(train_tasks))
            ptr, cycle = 0, cycle + 1
        task_idx = int(perm[ptr]); ptr += 1
        env.start(train_tasks[task_idx])
        reward.reset(env)
        s, meta = build_state17(env, env_cfg)
        done, ep_r = False, 0.0
        while not done:
            mask = meta["feasible_mec_mask"]
            a, lp, val = act_policy(args.algorithm, ag, s, mask, args.device, rng,
                                    explore=True, total_steps=total_steps)
            if a is None:
                _, done, info = env.step(None)
                ep_r += reward.step(env, info, done).total_training_reward
                break
            _, done, info = env.step(a)
            sig = reward.step(env, info, done)
            ep_r += sig.total_training_reward
            s2, meta2 = build_state17(env, env_cfg)
            if args.algorithm == "ppo":
                buf["s"].append(s); buf["m"].append(mask); buf["a"].append(a)
                buf["r"].append(sig.total_training_reward)
                buf["d"].append(float(done)); buf["lp"].append(lp); buf["v"].append(val)
            else:
                replay.add(s, mask, a, sig.total_training_reward, s2,
                           meta2["feasible_mec_mask"], float(done))
                if replay.size >= int(cfg["warmup"]):
                    last_metrics = update_offpolicy(
                        args.algorithm, ag, replay, args.device,
                        float(cfg["gamma"]), total_steps, cfg)
                    updates += 1
            s, meta = s2, meta2
            total_steps += 1

        if args.algorithm == "ppo" and buf["s"]:
            before = [prm.detach().clone() for prm in ag["ac"].parameters()]
            last_metrics = ppo_update(ag, buf, args.device, cfg)
            last_metrics["actor_param_delta"] = math.sqrt(sum(
                float((a - b).detach().pow(2).sum())
                for a, b in zip(before, ag["ac"].parameters())))
            updates += 1
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
               "updates": updates,
               "replay_size": replay.size,
               **{k: last_metrics.get(k) for k in METRIC_KEYS},
               "elapsed_sec": round(time.time() - t0, 1)}
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        state = {"perm": perm.tolist(), "ptr": ptr, "cycle": cycle,
                 "perm_rng": perm_rng.bit_generator.state,
                 "torch_rng": torch.get_rng_state().tolist(),
                 "np_rng": rng.bit_generator.state}
        save_checkpoint(out / "last.pt", args.algorithm, ag, cfg, env_cfg, ep,
                        total_steps, None, None, False, state)

        if cfg["eval_every"] and ep % int(cfg["eval_every"]) == 0:
            d = evaluate(args.algorithm, ag, env, dev_tasks, args.device,
                         prev_actions=prev_actions)
            prev_actions = d.pop("_actions")
            score = dev_score(d["sr"], d["d_sum_ms"], float(cfg["lambda_quality"]),
                              float(cfg["lambda_latency"]), env_cfg.d_max_ms,
                              env_cfg.max_steps)
            d["score"] = score
            with devp.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"episode": ep, **d}) + "\n")
            print(f"[{args.algorithm}] ep {ep} dev SR={d['sr']:.3f} "
                  f"Dsum={d['d_sum_ms']:.0f}ms score={score:.4f}", flush=True)
            if is_better(score, best_score):
                best_score = score
                save_checkpoint(out / "best.pt", args.algorithm, ag, cfg, env_cfg,
                                ep, total_steps, d, score, True, state)

    print(json.dumps({"algorithm": args.algorithm, "episodes": cfg["episodes"],
                      "gamma": cfg["gamma"], "best_dev_score": best_score,
                      "elapsed_sec": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
