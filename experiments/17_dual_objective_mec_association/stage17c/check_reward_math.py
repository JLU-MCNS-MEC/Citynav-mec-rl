#!/usr/bin/env python3
"""Stage 17-C: pure-arithmetic check of the dual-objective reward.

No environment, no checkpoints: a stub object supplies pos/target/success/
energy_budget_j so only the reward algebra is exercised.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reward17c import DualObjectiveReward, RewardConfig  # noqa: E402


class Stub:
    def __init__(self, pos, target, success=False, budget=360000.0):
        self.pos = np.asarray(pos, float)
        self.target = np.asarray(target, float)
        self.success = success
        self.energy_budget_j = budget


def main():
    cfg = RewardConfig()
    res = {}

    # 1) one forward toward the target: progress = 5/5 = 1
    r = DualObjectiveReward(cfg, forward_m=5.0)
    env = Stub([0.0, 0.0], [100.0, 0.0])
    r.reset(env)
    env.pos = np.asarray([5.0, 0.0])
    sig = r.step(env, {"d_coll_ms": 200.0, "e_total_step_j": 300.0}, done=False)
    expected = (cfg.lambda_quality * (cfg.progress_weight * 1.0)
                - cfg.lambda_latency * 200.0 / cfg.d_ref_ms)
    res["progress_step"] = {"total": sig.total_training_reward,
                            "expected": expected,
                            "pass": abs(sig.total_training_reward - expected) < 1e-9}

    # 2) terminal success adds +1 and is not discounted inside the term
    env.success = True
    sig = r.step(env, {"d_coll_ms": 100.0, "e_total_step_j": 0.0}, done=True)
    # pos unchanged -> progress 0
    expected = cfg.lambda_quality * (cfg.terminal_bonus - 0.0) \
        - 100.0 / cfg.d_ref_ms
    res["terminal_success"] = {"total": sig.total_training_reward,
                               "expected": expected,
                               "pass": abs(sig.total_training_reward - expected) < 1e-9}

    # 3) failure with failure_penalty = 0 gives no terminal term
    r2 = DualObjectiveReward(cfg, 5.0)
    env2 = Stub([0.0, 0.0], [10.0, 0.0], success=False)
    r2.reset(env2)
    sig = r2.step(env2, {"d_coll_ms": 0.0, "e_total_step_j": 0.0}, done=True)
    res["failure_zero_penalty"] = {"total": sig.total_training_reward,
                                   "pass": abs(sig.total_training_reward) < 1e-12}

    # 4) progress clip
    r3 = DualObjectiveReward(cfg, 5.0)
    env3 = Stub([0.0, 0.0], [100.0, 0.0])
    r3.reset(env3)
    env3.pos = np.asarray([50.0, 0.0])          # 50 m jump -> clipped to 1
    sig = r3.step(env3, {"d_coll_ms": 0.0, "e_total_step_j": 0.0}, done=False)
    res["progress_clip"] = {"progress": sig.navigation_progress_reward,
                            "pass": abs(sig.navigation_progress_reward - 1.0) < 1e-12}

    # 5) D_ref scaling: doubling lambda_D halves the latency term
    c2 = RewardConfig(lambda_latency=2.0)
    r4 = DualObjectiveReward(c2, 5.0)
    env4 = Stub([0.0, 0.0], [0.0, 0.0])
    r4.reset(env4)
    sig = r4.step(env4, {"d_coll_ms": cfg.d_ref_ms, "e_total_step_j": 0.0}, done=False)
    res["lambda_scaling"] = {"total": sig.total_training_reward,
                             "expected": -2.0,
                             "pass": abs(sig.total_training_reward + 2.0) < 1e-9}

    out = HERE.parent / "smoke/reward_math_check.json"
    import json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    n = sum(1 for v in res.values() if v["pass"])
    print(f"PASSED {n}/{len(res)}")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
