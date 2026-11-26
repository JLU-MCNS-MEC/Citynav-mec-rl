#!/usr/bin/env python3
"""Stage 17-D-P1-A checks T1-T9 (pure tensors; no checkpoint, no environment)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from nets17 import N_ACT, STATE17_DIM, ActorCritic17, masked_expectation, masked_softmax  # noqa: E402
import train17c  # noqa: E402


class FakeReplay:
    def __init__(self, B):
        self.t = (torch.randn(B, STATE17_DIM), torch.ones(B, N_ACT),
                  torch.randint(0, N_ACT, (B,)), torch.randn(B),
                  torch.randn(B, STATE17_DIM), torch.ones(B, N_ACT),
                  torch.zeros(B))
        self.t[1][:, 0] = 0.0
        self.t[5][:, 0] = 0.0

    def sample(self, bs, device):
        return self.t


def main():
    res = {}
    cfg_base = dict(train17c.DEFAULTS)
    B = 8

    # T1 TD3 actor/critic ratio -----------------------------------------
    ag = train17c.make_agents("td3", "cpu", {"hidden": 32, "lr_actor": 1e-3,
                                             "lr_dqn": 1e-4})
    rp = FakeReplay(B)
    cfg = dict(cfg_base); cfg.update({"batch": B, "policy_delay": 2})
    for i in range(10):
        train17c.update_offpolicy("td3", ag, rp, "cpu", 1.0, i, cfg)
    res["T1_td3_delay"] = {"critic": ag.get("critic_updates"),
                           "actor": ag.get("actor_updates"),
                           "target": ag.get("target_updates"),
                           "pass": ag.get("critic_updates") == 10
                                   and ag.get("actor_updates") == 5
                                   and ag.get("target_updates") == 5}

    # T2 TD3 actor gradient ---------------------------------------------
    pi = ActorCritic17(hidden=32)
    s = torch.randn(4, STATE17_DIM)
    m = torch.ones(4, N_ACT); m[:, 2] = 0.0
    p = masked_softmax(pi(s)[0], m)
    q = torch.randn(4, N_ACT)
    (-masked_expectation(q, p).mean()).backward()
    g = sum(float(prm.grad.abs().sum()) for prm in pi.parameters()
            if prm.grad is not None)
    res["T2_td3_actor_grad"] = {"grad_sum": g, "pass": g > 1e-10 and math.isfinite(g)}

    # T3 DDPG diagnostics ------------------------------------------------
    ag_d = train17c.make_agents("ddpg", "cpu", {"hidden": 32, "lr_actor": 1e-3,
                                                "lr_dqn": 1e-4})
    out = train17c.update_offpolicy("ddpg", ag_d, FakeReplay(B), "cpu", 1.0, 1,
                                    dict(cfg_base, batch=B))
    keys = ("actor_logits_std", "actor_max_prob", "actor_entropy", "q_action_std",
            "q_action_range", "actor_param_delta", "grad_pi")
    res["T3_ddpg_diag"] = {"keys": {k: out.get(k) for k in keys},
                           "pass": all(out.get(k) is not None
                                       and math.isfinite(float(out[k])) for k in keys)}

    # T4 SAC soft target ------------------------------------------------
    alpha = 0.2
    p2 = torch.tensor([[0.5, 0.3, 0.2, 0.0]])
    q1 = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    q2 = torch.tensor([[1.5, 1.5, 2.0, 100.0]])
    qmin = torch.min(q1, q2)
    v = float(masked_expectation(qmin - alpha * torch.log(p2 + 1e-12), p2))
    manual = (0.5 * (1.0 - alpha * math.log(0.5))
              + 0.3 * (1.5 - alpha * math.log(0.3))
              + 0.2 * (2.0 - alpha * math.log(0.2)))
    res["T4_sac_target"] = {"code": v, "manual": manual,
                            "pass": abs(v - manual) < 1e-6}

    # T5 SAC diagnostics ------------------------------------------------
    ag_s = train17c.make_agents("sac", "cpu", {"hidden": 32, "lr_actor": 1e-3,
                                               "lr_dqn": 1e-4})
    out = train17c.update_offpolicy("sac", ag_s, FakeReplay(B), "cpu", 1.0, 1,
                                    dict(cfg_base, batch=B))
    keys = ("mean_env_reward", "mean_entropy_bonus", "mean_bellman_target",
            "mean_q1", "mean_q2", "mean_q_min", "critic_td_error",
            "alpha", "entropy", "sac_target_entropy")
    res["T5_sac_diag"] = {"keys": {k: out.get(k) for k in keys},
                          "pass": all(out.get(k) is not None for k in keys)}

    # T6 PPO diagnostics ------------------------------------------------
    ag_p = train17c.make_agents("ppo", "cpu", {"hidden": 32, "lr_actor": 1e-3,
                                               "lr_dqn": 1e-4})
    buf = {"s": [np.zeros(STATE17_DIM, np.float32)] * 4,
           "m": [np.ones(N_ACT, np.float32)] * 4,
           "a": [0, 1, 2, 3], "r": [0.1, -0.2, 0.3, 0.0],
           "d": [0.0, 0.0, 0.0, 1.0], "lp": [-2.3] * 4, "v": [0.0] * 4}
    out = train17c.ppo_update(ag_p, buf, "cpu", dict(cfg_base))
    keys = ("loss_pi", "loss_v", "grad_pi", "approx_kl", "clip_frac",
            "policy_entropy")
    res["T6_ppo_diag"] = {"keys": {k: out.get(k) for k in keys},
                          "pass": all(out.get(k) is not None for k in keys)}

    # T7 reproducible task permutation ----------------------------------
    def first(seed, k=5):
        r = np.random.default_rng(seed)
        out, ptr, perm = [], 0, r.permutation(4000)
        while len(out) < k:
            out.append(int(perm[ptr])); ptr += 1
        return out
    a, b, c = first(42), first(42), first(43)
    res["T7_task_order"] = {"seed42_a": a, "seed42_b": b, "seed43": c,
                            "pass": a == b and a != c}

    # T8 identical reward across algorithms -----------------------------
    cfgj = json.loads((HERE / "train_config.json").read_text())
    res["T8_reward_shared"] = {
        "lambda_quality": cfgj["reward"]["lambda_quality"],
        "lambda_latency": cfgj["reward"]["lambda_latency"],
        "gamma": cfgj["gamma"],
        "pass": cfgj["reward"]["lambda_quality"] == 1.0
                and cfgj["reward"]["lambda_latency"] == 0.5
                and cfgj["gamma"] == 1.0}

    # T9 frozen environment parameters ----------------------------------
    env_cfg = json.loads((ROOT / "configs/default.json").read_text())
    want = {"d_max_ms": 400.0, "initial_energy_j": 360000.0,
            "task_energy_budget_j": 360000.0, "coverage_m": 500.0,
            "max_steps": 200, "feature_bits": 32, "v_forward_mps": 3.0,
            "uav_height_m": 50.0, "mec_antenna_height_m": 10.0}
    res["T9_env_frozen"] = {"values": {k: env_cfg.get(k) for k in want},
                            "pass": all(abs(float(env_cfg[k]) - v) < 1e-9
                                        for k, v in want.items())}

    out = ROOT / "smoke/checks_p1a.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    n = sum(1 for v in res.values() if v["pass"])
    print(f"PASSED {n}/{len(res)}")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
