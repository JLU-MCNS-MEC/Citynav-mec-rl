#!/usr/bin/env python3
"""Stage 17-C-R2 C1-C7: reward ordering, mixed-batch updates, temperature
direction, deterministic evaluation, per-algorithm gradient step, checkpoint
round-trip, config precedence.

Pure arithmetic / virtual tensors: no checkpoint, no environment, no episode.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nets17 import N_ACT, STATE17_DIM, ActorCritic17, QNet17, masked_softmax  # noqa: E402
from reward17c import DualObjectiveRewardR1, RewardConfigR1                  # noqa: E402
import train17c                                                              # noqa: E402

D_MAX, T_MAX = 400.0, 200


class StubEnv:
    def __init__(self, success):
        self.success = success
        self.energy_budget_j = 360000.0


class FakeReplay:
    def __init__(self, s, m, a, r, s2, m2, d):
        self.t = (s, m, a, r, s2, m2, d)

    def sample(self, bs, device):
        return self.t


def ep_reward(success, d_sum_ms):
    rw = DualObjectiveRewardR1(RewardConfigR1(1.0, 0.5), D_MAX, T_MAX)
    # one terminal step carrying the whole episode latency
    return rw.step(StubEnv(success), {"d_coll_ms": d_sum_ms, "e_total_step_j": 0.0},
                   True).total_training_reward


def main():
    res = {}

    # C1 reward ordering -------------------------------------------------
    ok_a = ep_reward(True, 5000.0) > ep_reward(False, 5000.0)
    ok_b = ep_reward(True, 3000.0) > ep_reward(True, 6000.0)
    ok_c = ep_reward(False, 3000.0) > ep_reward(False, 6000.0)
    ok_d = ep_reward(True, 0.0) > 0 > ep_reward(False, 80000.0)
    # step-count independence: same D_sum, different number of steps
    rw = DualObjectiveRewardR1(RewardConfigR1(1.0, 0.5), D_MAX, T_MAX)
    e = StubEnv(True)
    total_2 = sum(rw.step(e, {"d_coll_ms": 2000.0, "e_total_step_j": 0.0}, False).total_training_reward
                  for _ in range(2))
    total_2 += rw.step(e, {"d_coll_ms": 1000.0, "e_total_step_j": 0.0}, True).total_training_reward
    one = ep_reward(True, 5000.0)
    res["C1_reward"] = {"success_beats_failure": ok_a, "lower_latency_better_success": ok_b,
                        "lower_latency_better_failure": ok_c,
                        "sign_boundary": ok_d,
                        "step_count_independent": abs(total_2 - one) < 1e-9,
                        "pass": all([ok_a, ok_b, ok_c, ok_d])
                                and abs(total_2 - one) < 1e-9}

    # C2 TD3 mixed batch ------------------------------------------------
    torch.manual_seed(0)
    ag = train17c.make_agents("td3", "cpu", {"hidden": 32, "lr_actor": 1e-3,
                                             "lr_dqn": 1e-4})
    B = 6
    s = torch.randn(B, STATE17_DIM); s2 = torch.randn(B, STATE17_DIM)
    m = torch.ones(B, N_ACT); m2 = torch.ones(B, N_ACT)
    m2[4:] = 0.0                       # two empty next masks
    d = torch.zeros(B); d[2:4] = 1.0   # two terminal samples
    a = torch.randint(0, N_ACT, (B,)); r = torch.randn(B)
    replay = FakeReplay(s, m, a, r, s2, m2, d)
    out = train17c.update_offpolicy("td3", ag, replay, "cpu", 1.0, 200,
                                    {"batch": B, "tau": 0.005,
                                     "target_sync_steps": 200})
    res["C2_td3_mixed_batch"] = {"loss": out["loss_q"],
                                 "finite": math.isfinite(out["loss_q"]),
                                 "pass": math.isfinite(out["loss_q"])}

    # C3 SAC temperature direction --------------------------------------
    log_alpha = torch.zeros(1, requires_grad=True)
    eta = 0.5
    def alpha_grad(p):
        ent = float(-(p * torch.log(p + 1e-12)).sum())
        k = int((p > 0).sum())
        h_t = eta * math.log(k)
        loss = log_alpha * (ent - h_t)
        g, = torch.autograd.grad(loss, log_alpha)
        return ent, h_t, float(g)
    p_flat = torch.full((4,), 0.25)
    p_peak = torch.tensor([0.97, 0.01, 0.01, 0.01])
    ent1, ht1, g1 = alpha_grad(p_flat)
    ent2, ht2, g2 = alpha_grad(p_peak)
    res["C3_sac_temperature"] = {
        "entropy_uniform": ent1, "h_target": ht1, "grad_uniform": g1,
        "entropy_peak": ent2, "grad_peak": g2,
        "pass": ent1 > ht1 and g1 > 0 and ent2 < ht2 and g2 < 0}

    # C4 deterministic evaluation ---------------------------------------
    ag2 = train17c.make_agents("ppo", "cpu", {"hidden": 32, "lr_actor": 1e-3,
                                              "lr_dqn": 1e-4})
    st = np.random.default_rng(0).standard_normal(STATE17_DIM).astype(np.float32)
    mask = np.ones(N_ACT, np.float32); mask[[1, 7]] = 0.0
    a1, _, _ = train17c.act_policy("ppo", ag2, st, mask, "cpu", np.random.default_rng(1),
                                   explore=False)
    a2, _, _ = train17c.act_policy("ppo", ag2, st, mask, "cpu", np.random.default_rng(2),
                                   explore=False)
    empty = np.zeros(N_ACT, np.float32)
    a3, _, _ = train17c.act_policy("ppo", ag2, st, empty, "cpu", np.random.default_rng(3),
                                   explore=False)
    res["C4_deterministic_eval"] = {"a1": a1, "a2": a2, "empty": a3,
                                    "pass": a1 == a2 and a1 not in (1, 7) and a3 is None}

    # C5 one gradient step per algorithm --------------------------------
    steps = {}
    for algo in ("dqn", "ddpg", "td3", "sac"):
        torch.manual_seed(1)
        agx = train17c.make_agents(algo, "cpu", {"hidden": 32, "lr_actor": 1e-3,
                                                 "lr_dqn": 1e-4})
        m_all = torch.ones(8, N_ACT); m_all[:, 0] = 0.0
        rp = FakeReplay(torch.randn(8, STATE17_DIM), m_all,
                        torch.randint(1, N_ACT, (8,)), torch.randn(8),
                        torch.randn(8, STATE17_DIM), m_all, torch.zeros(8))
        o = train17c.update_offpolicy(algo, agx, rp, "cpu", 1.0, 200,
                                      {"batch": 8, "tau": 0.005,
                                       "target_sync_steps": 200, "sac_eta": 0.5})
        steps[algo] = math.isfinite(o["loss_q"])
    res["C5_gradient_steps"] = {"finite": steps, "pass": all(steps.values())}

    # C6 checkpoint round-trip ------------------------------------------
    tmp = Path(tempfile.mkdtemp())
    cfg = dict(train17c.DEFAULTS); cfg.update({"lambda_quality": 1.0,
                                               "lambda_latency": 0.5})
    class E:  # minimal environment-summary carrier
        d_max_ms = D_MAX; task_energy_budget_j = 360000.0
        coverage_m = 500.0; v_forward_mps = 3.0
    train17c.save_checkpoint(tmp / "best.pt", "ppo", ag2, cfg, E(), 25, 1234,
                             {"sr": 0.5, "d_sum_ms": 6000.0}, -0.2, True)
    sd = torch.load(tmp / "best.pt", map_location="cpu", weights_only=False)
    net = ActorCritic17(hidden=32); net.load_state_dict(sd["policy"]["ac"])
    res["C6_checkpoint"] = {"keys": sorted(sd),
                            "gamma": sd["gamma"], "score": sd["dev_score"],
                            "pass": sd["gamma"] == 1.0 and sd["best"] is True
                                    and abs(sd["dev_score"] + 0.2) < 1e-9
                                    and sd["environment"]["d_max_ms"] == D_MAX}

    # C7 config precedence ----------------------------------------------
    p = tmp / "c.json"
    p.write_text(json.dumps({"gamma": 1.0, "batch": 64,
                             "reward": {"lambda_quality": 1.0, "lambda_latency": 0.5},
                             "sac_eta": 0.5}))
    c = train17c.resolve_config({}, p)
    res["C7_config"] = {"gamma": c["gamma"], "batch": c["batch"],
                        "lambda_latency": c["lambda_latency"], "eta": c["sac_eta"],
                        "pass": c["gamma"] == 1.0 and c["batch"] == 64
                                and c["lambda_latency"] == 0.5 and c["sac_eta"] == 0.5}

    out = HERE.parent / "smoke/checks_r2.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    n = sum(1 for v in res.values() if v["pass"])
    print(f"PASSED {n}/{len(res)}")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
