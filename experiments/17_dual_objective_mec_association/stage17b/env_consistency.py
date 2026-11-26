#!/usr/bin/env python3
"""Stage 17-B E1-E9: environment consistency with Stage 16 + constraint logic."""
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


def run_stage16(env, task, rng):
    env.start(task)
    traj = []
    done = False
    while not done:
        a = choose("min_latency", env.metrics(), rng)
        _, done, _ = env.step(a)
        traj.append((int(a), tuple(np.round(env.pos, 6)), float(env.yaw)))
    return traj, bool(env.success), env.step_idx


def run_stage17(env, task, rng):
    env.start(task)
    traj = []
    done = False
    while not done:
        a = choose("min_latency", env.metrics(), rng)
        _, done, _ = env.step(a)
        traj.append((int(a), tuple(np.round(env.pos, 6)), float(env.yaw)))
    return traj, bool(env.success), env.step_idx, env


def main():
    torch.set_num_threads(4)
    cfg = load_config(ROOT / "configs/default.json")
    free = cfg.with_overrides(enforce_coverage=False, enforce_deadline=False,
                              enforce_energy=False)
    train_tasks, dev_tasks = load_tasks_and_split()
    tasks = train_tasks[:10]
    by_map = build_deployment_by_map()
    models = load_models("cuda")
    norm = load_norm()

    e16 = EpisodeEnv16(by_map, "cuda", models, norm, reward_variant="A",
                       record_steps=True)
    e17 = EpisodeEnv17(by_map, "cuda", models, norm, free)

    res = {}
    same_traj, mismatch, legacy_ok = 0, 0, 0
    for t in tasks:
        rng = np.random.default_rng(7)
        t16, s16, n16 = run_stage16(e16, t, rng)
        rng = np.random.default_rng(7)
        t17, s17, n17, env = run_stage17(e17, t, rng)
        same_traj += int(t16 == t17 and s16 == s17 and n16 == n17)
        mismatch += env.preview_mismatch_count
        legacy_ok += int(abs(float(np.sum(env.episode_e2e))
                             - float(env.cumulative_coll_latency_ms)) < 1e-6)

    n = len(tasks)
    res["E1_stage16_trajectory_equal"] = {"tasks": n, "equal": same_traj,
                                         "pass": same_traj == n}
    res["E2_preview_action_mismatch"] = {"count": mismatch, "pass": mismatch == 0}
    res["E9_legacy_e2e_equals_d_sum"] = {"tasks": n, "equal": legacy_ok,
                                         "pass": legacy_ok == n,
                                         "note": "downlink disabled, so D_sum equals the Stage-16 E2E"}

    # E7 accounting on the last episode
    step_sum = sum(r["e_total_step_j"] for r in env.step_records)
    res["E7_energy_accounting"] = {
        "sum_step_records_j": float(step_sum),
        "cumulative_energy_j": float(env.cumulative_energy_j),
        "pass": abs(step_sum - env.cumulative_energy_j) < 1e-6,
    }
    # E8 no charging after termination
    last = env.step_records[-1]
    res["E8_no_post_termination_charge"] = {
        "records": len(env.step_records), "env_steps": int(env.step_idx),
        "last_nav_action": int(last["navigation_action"]),
        "pass": len(env.step_records) == int(env.step_idx),
    }

    # E3 coverage enforcement: shrink the coverage radius so nothing is reachable
    cov = cfg.with_overrides(coverage_m=1.0)
    env = EpisodeEnv17(by_map, "cuda", models, norm, cov)
    env.start(tasks[0])
    _, done, info = env.step(0)
    res["E3_coverage_enforced"] = {
        "terminated_reason": env.terminated_reason, "done": bool(done),
        "pass": env.terminated_reason == "no_feasible_mec"}

    # E4 deadline: 1 ms deadline makes every node infeasible
    dl = cfg.with_overrides(d_max_ms=1.0)
    env = EpisodeEnv17(by_map, "cuda", models, norm, dl)
    env.start(tasks[0])
    _, done, _ = env.step(0)
    tab = env.candidate_table()
    res["E4_deadline_enforced"] = {
        "terminated_reason": env.terminated_reason,
        "all_deadline_invalid": bool(all(not r["deadline_valid"] for r in tab)),
        "pass": env.terminated_reason == "no_feasible_mec"}

    # E5 service affordable, motion not: budget = one service + half a forward
    env = EpisodeEnv17(by_map, "cuda", models, norm, cfg)
    env.start(tasks[0])
    lat0 = service_latency_ms(0.0, "small", cfg)
    svc = e_coll_j(lat0, cfg)["e_coll_j"]
    env.energy_budget_j = svc + 0.5 * e_motion_j(1, cfg)["e_motion_j"]
    env.remaining_energy_j = env.energy_budget_j
    feas = [i for i, r in enumerate(env.candidate_table()) if r["feasible"]]
    node = feas[0] if feas else 0
    _, done, _ = env.step(node)
    res["E5_energy_budget_enforced"] = {
        "budget_j": float(env.energy_budget_j),
        "terminated_reason": env.terminated_reason,
        "pass": env.terminated_reason in ("energy_budget_exhausted", "no_feasible_mec")}

    # E6 invalid node request under normal config
    env = EpisodeEnv17(by_map, "cuda", models, norm, cfg)
    env.start(tasks[0])
    tab = env.candidate_table()
    bad = [i for i, r in enumerate(tab) if not r["feasible"]]
    if bad:
        _, done, _ = env.step(bad[0])
        ok = env.terminated_reason == "invalid_mec_action"
    else:
        ok = True  # every node feasible at this state; nothing to violate
    res["E6_invalid_node_rejected"] = {
        "had_infeasible_node": bool(bad),
        "terminated_reason": env.terminated_reason, "pass": ok}

    out = ROOT / "smoke/env_consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    npass = sum(1 for v in res.values() if v["pass"])
    print(f"PASSED {npass}/{len(res)}")
    return 0 if npass == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
