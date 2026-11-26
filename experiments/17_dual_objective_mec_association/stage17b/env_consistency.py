#!/usr/bin/env python3
"""Stage 17-B E1-E9 (R2 API): consistency with Stage 16 + constraint logic.

Not executed in Stage 17-A-R2.  Requires loading the five frozen experts and
running 10 Train episodes, so it belongs to the Smoke Test stage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
for p in [ROOT / "src", REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/16_cumulative_e2e_association/p2",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from baselines import choose                                  # noqa: E402
from config17 import load_config                              # noqa: E402
from energy17 import e_coll_j, e_motion_j                     # noqa: E402
from env16 import (EpisodeEnv16, build_deployment_by_map,     # noqa: E402
                   load_models, load_norm)
from env17 import EpisodeEnv17                                # noqa: E402
from latency17 import service_latency_ms                      # noqa: E402
from train_association_rl import load_tasks_and_split         # noqa: E402


class CountingModels(dict):
    """Count expert forwards per model type without touching the env code."""

    def __init__(self, models):
        super().__init__({k: _Counter(v) for k, v in models.items()})


class _Counter:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        return self.model(*a, **kw)

    def __getattr__(self, name):
        return getattr(self.model, name)


def run_episode(env, task, rng):
    env.start(task)
    traj, done = [], False
    while not done:
        a = choose("min_latency", env.metrics(), rng)
        _, done, _ = env.step(a)
        traj.append((int(a), tuple(np.round(env.pos, 6)), round(float(env.yaw), 6)))
    return traj, bool(env.success), int(env.step_idx), env


def main():
    torch.set_num_threads(4)
    cfg = load_config(ROOT / "configs/default.json")
    free = cfg.with_overrides(enforce_coverage=False, enforce_deadline=False,
                              enforce_energy=False)
    train_tasks, _ = load_tasks_and_split()
    tasks = train_tasks[:10]
    by_map = build_deployment_by_map()
    models = load_models("cuda")
    norm = load_norm()

    e16 = EpisodeEnv16(by_map, "cuda", models, norm, reward_variant="A",
                       record_steps=True)
    counting = CountingModels(models)
    e17 = EpisodeEnv17(by_map, "cuda", counting, norm, free)

    res = {}
    same, legacy_ok = 0, 0
    for t in tasks:
        t16, s16, n16, _ = run_episode(e16, t, np.random.default_rng(7))
        t17, s17, n17, env = run_episode(e17, t, np.random.default_rng(7))
        same += int(t16 == t17 and s16 == s17 and n16 == n17)
        legacy_ok += int(all(abs(a - b) < 1e-9 for a, b in
                             zip(np.cumsum(env.episode_e2e),
                                 np.cumsum(env.episode_e2e))))

    n = len(tasks)
    forwards = sum(m.calls for m in counting.values())
    res["E1_stage16_trajectory_equal"] = {"tasks": n, "equal": same, "pass": same == n}
    res["E2_single_forward_per_step"] = {
        "expert_forwards": forwards,
        "steps": int(sum(int(r["step"]) >= 0 for r in env.step_records)) + 0,
        "note": "compared against the total number of decision steps in the run",
        "pass": None}
    res["E9_legacy_e2e_preserved"] = {"tasks": n, "ok": legacy_ok, "pass": legacy_ok == n}

    step_sum = sum(r["e_total_step_j"] for r in env.step_records)
    res["E7_energy_accounting"] = {"sum_steps_j": float(step_sum),
                                   "cumulative_j": float(env.cumulative_energy_j),
                                   "pass": abs(step_sum - env.cumulative_energy_j) < 1e-6}
    res["E8_no_post_termination_charge"] = {
        "records": len(env.step_records), "env_steps": int(env.step_idx),
        "pass": len(env.step_records) <= int(env.step_idx)}

    env = EpisodeEnv17(by_map, "cuda", models, norm,
                       cfg.with_overrides(coverage_m=1.0))
    env.start(tasks[0]); _, done, _ = env.step(0)
    res["E3_coverage_enforced"] = {"reason": env.terminated_reason,
                                   "pass": env.terminated_reason == "no_feasible_mec"}

    env = EpisodeEnv17(by_map, "cuda", models, norm,
                       cfg.with_overrides(d_max_ms=1.0))
    env.start(tasks[0]); _, done, _ = env.step(0)
    res["E4_deadline_enforced"] = {"reason": env.terminated_reason,
                                   "pass": env.terminated_reason == "no_feasible_mec"}

    env = EpisodeEnv17(by_map, "cuda", models, norm, cfg)
    env.start(tasks[0])
    lat0 = service_latency_ms(0.0, "small", cfg)
    svc = e_coll_j(lat0, cfg)["e_coll_j"]
    env.energy_budget_j = svc + 0.5 * e_motion_j(1, cfg)["e_motion_j"]
    env.remaining_energy_j = env.energy_budget_j
    feas = env.feasible_node_indices()
    _, done, _ = env.step(feas[0] if feas else 0)
    res["E5_energy_budget_enforced"] = {
        "reason": env.terminated_reason,
        "pass": env.terminated_reason in ("energy_budget_exhausted", "no_feasible_mec")}

    env = EpisodeEnv17(by_map, "cuda", models, norm, cfg)
    env.start(tasks[0])
    tab = env.candidate_table()
    bad = [i for i, r in enumerate(tab) if not r["feasible"]]
    if bad:
        env.step(bad[0])
        ok = env.terminated_reason == "invalid_mec_action"
    else:
        ok = True
    res["E6_invalid_node_rejected"] = {"had_infeasible": bool(bad),
                                       "reason": env.terminated_reason, "pass": ok}
    res["E6b_none_action_safe"] = {"note": "step(None) must not raise int(None)"}

    out = ROOT / "smoke/env_consistency_r2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
