#!/usr/bin/env python3
"""Stage 17-B E1-E15 (R3): environment consistency and constraint assertions.

NOT executed in Stage 17-A-R3.  Requires the five frozen experts and real
episodes, so it is part of the Smoke Test stage.

Each check returns {"pass": bool, ...} - no check may silently return None.
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

from baselines import choose                                   # noqa: E402
from config17 import load_config                               # noqa: E402
from compute17 import uav_compute_ms                           # noqa: E402
from energy17 import e_coll_j, e_forward_j, e_motion_j, step_energy_j  # noqa: E402
from env16 import (EpisodeEnv16, build_deployment_by_map,      # noqa: E402
                   load_models, load_norm)
from env17 import EpisodeEnv17                                 # noqa: E402
from latency17 import legacy_stage16_e2e_ms, service_latency_ms  # noqa: E402
from multimodal17 import calculate_uplink_payload_bits         # noqa: E402
from train_association_rl import load_tasks_and_split          # noqa: E402


class _Counter:
    def __init__(self, model):
        self.model, self.calls = model, 0

    def __call__(self, *a, **kw):
        self.calls += 1
        return self.model(*a, **kw)

    def __getattr__(self, name):
        return getattr(self.model, name)


def run_stage16(env, task, rng):
    env.start(task)
    traj, e2e, done = [], [], False
    while not done:
        a = choose("min_latency", env.metrics(), rng)
        _, done, _ = env.step(a)
        traj.append((int(a), tuple(np.round(env.pos, 6)), round(float(env.yaw), 6)))
        e2e.append(float(env.episode_e2e[-1]))
    return traj, e2e, bool(env.success), env.failure, int(env.step_idx)


def run_stage17(env, task, rng):
    env.start(task)
    traj, e2e, done = [], [], False
    while not done:
        a = choose("min_latency", env.metrics(), rng)
        _, done, _ = env.step(a)
        traj.append((int(a), tuple(np.round(env.pos, 6)), round(float(env.yaw), 6)))
        e2e.append(float(env.episode_e2e[-1]))
    return traj, e2e, bool(env.success), env.terminated_reason, int(env.step_idx)


def episode_energy_sum(env):
    return float(sum(r["e_total_step_j"] for r in env.step_records))


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
    res = {}

    e16 = EpisodeEnv16(by_map, "cuda", models, norm, reward_variant="A",
                       record_steps=True)
    counters = {k: _Counter(v) for k, v in models.items()}
    e17 = EpisodeEnv17(by_map, "cuda", counters, norm, free)

    # ---- E1 / E9 / E2 -------------------------------------------------
    traj_ok = legacy_ok = 0
    inference_total = 0
    for t in tasks:
        t16, e16_list, s16, f16, n16 = run_stage16(e16, t, np.random.default_rng(7))
        t17, e17_list, s17, f17, n17 = run_stage17(e17, t, np.random.default_rng(7))
        traj_ok += int(t16 == t17 and s16 == s17 and n16 == n17 and f16 == f17)
        legacy_ok += int(len(e16_list) == len(e17_list)
                         and all(abs(a - b) < 1e-9
                                 for a, b in zip(e16_list, e17_list)))
        inference_total += int(e17.completed_inference_count)
    res["E1_trajectory_equal"] = {"tasks": len(tasks), "ok": traj_ok,
                                  "pass": traj_ok == len(tasks)}
    res["E9_legacy_e2e_independent"] = {"tasks": len(tasks), "ok": legacy_ok,
                                        "pass": legacy_ok == len(tasks),
                                        "note": "per-step legacy E2E of Stage 16 vs Stage 17, computed independently"}
    forwards = sum(c.calls for c in counters.values())
    res["E2_single_forward_per_step"] = {
        "expert_forwards": forwards, "completed_inferences": inference_total,
        "pass": forwards == inference_total}

    # ---- E10 payload / E11 power / E12 matching / E14 latency ---------
    payload = calculate_uplink_payload_bits(cfg)
    res["E10_payload"] = {"bits": payload["uplink_payload_bits"],
                          "expected": 4281664,
                          "relation_uploaded": any(m["name"] == "relation" and m["uploaded"]
                                                   for m in payload["modalities"]),
                          "pass": payload["uplink_payload_bits"] == 4281664
                                  and not any(m["name"] == "relation" and m["uploaded"]
                                              for m in payload["modalities"])}
    comp = uav_compute_ms(cfg)
    res["E12_matching"] = {"flops": comp["matching_flops"],
                           "ratio": comp["matching_ratio_of_visual"],
                           "negligible": comp["matching_negligible"],
                           "verified": comp["d_matching_verified"],
                           "pass": comp["d_matching_verified"]
                                   and comp["matching_ratio_of_visual"] < 0.01}
    lat = service_latency_ms(200.0, "ts", cfg, payload)
    res["E14_new_latency_objective"] = {
        "d_coll_ms": lat["d_coll_ms"],
        "recomputed": (comp["d_uav_total_ms"] + lat["d_up_ms"] + lat["d_inf_ms"]),
        "legacy": legacy_stage16_e2e_ms(200.0, "ts"),
        "pass": abs(lat["d_coll_ms"] - (comp["d_uav_total_ms"] + lat["d_up_ms"]
                                        + lat["d_inf_ms"])) < 1e-9
                and abs(lat["d_coll_ms"] - legacy_stage16_e2e_ms(200.0, "ts")) > 1.0}

    # ---- E3 coverage ---------------------------------------------------
    env = EpisodeEnv17(by_map, "cuda", models, norm, cfg.with_overrides(coverage_m=1.0))
    env.start(tasks[0]); _, d, _ = env.step(0)
    tab_ok = all(r["distance_m"] > 1.0 for r in env.candidate_table())
    res["E3_coverage_param"] = {"reason": env.terminated_reason,
                                "all_nodes_beyond_radius": tab_ok,
                                "cfg_radius": cfg.coverage_m,
                                "pass": env.terminated_reason == "no_feasible_mec"
                                        and tab_ok and cfg.coverage_m == 500.0}

    # ---- E4 deadline ---------------------------------------------------
    env = EpisodeEnv17(by_map, "cuda", models, norm, cfg.with_overrides(d_max_ms=1.0))
    env.start(tasks[0]); _, d, _ = env.step(0)
    tab = env.candidate_table()
    res["E4_deadline"] = {"reason": env.terminated_reason,
                          "all_invalid": all(not r["deadline_valid"] for r in tab),
                          "pass": env.terminated_reason == "no_feasible_mec"
                                  and all(not r["deadline_valid"] for r in tab)}

    # ---- E5 service paid, motion refused -------------------------------
    env = EpisodeEnv17(by_map, "cuda", models, norm, cfg)
    env.start(tasks[0])
    feas = env.feasible_node_indices()
    if feas:
        lat0 = service_latency_ms(0.0, "small", cfg, env.payload)
        svc = e_coll_j(lat0, cfg)["e_coll_j"]
        env.energy_budget_j = svc + 0.5 * e_motion_j(1, cfg)["e_motion_j"]
        env.remaining_energy_j = env.energy_budget_j
        _, _, _ = env.step(feas[0])
        rec = env.step_records[-1]
        res["E5_motion_refused"] = {
            "reason": env.terminated_reason,
            "inference_done": env.completed_inference_count == 1,
            "motion_executed": rec["motion_executed"],
            "service_charged": rec["e_coll_j"] > 0,
            "budget_respected": env.cumulative_energy_j <= env.energy_budget_j + 1e-9,
            "pass": (env.terminated_reason == "energy_budget_exhausted"
                     and env.completed_inference_count == 1
                     and not rec["motion_executed"]
                     and rec["e_coll_j"] > 0
                     and env.cumulative_energy_j <= env.energy_budget_j + 1e-9)}
    else:
        res["E5_motion_refused"] = {"pass": False, "error": "no feasible node"}

    # ---- E6 None action ------------------------------------------------
    env = EpisodeEnv17(by_map, "cuda", models, norm, cfg.with_overrides(coverage_m=1.0))
    env.start(tasks[0])
    try:
        env.step(None); ok = env.terminated_reason == "no_feasible_mec"
        err = ""
    except Exception as exc:  # noqa: BLE001
        ok, err = False, repr(exc)
    res["E6_none_action"] = {"ok": ok, "error": err, "pass": ok}

    # ---- E7 accounting across termination kinds ------------------------
    cases = {}
    for name, over in [("default", {}), ("tiny_budget", {"task_energy_budget_j": 1.0})]:
        env = EpisodeEnv17(by_map, "cuda", models, norm, cfg.with_overrides(**over))
        env.start(tasks[0])
        done = False
        while not done:
            feas = env.feasible_node_indices()
            node = feas[0] if feas else None
            _, done, _ = env.step(node)
        cases[name] = {"sum": episode_energy_sum(env),
                       "cumulative": env.cumulative_energy_j,
                       "reason": env.terminated_reason}
    res["E7_energy_accounting"] = {
        "cases": cases,
        "pass": all(abs(v["sum"] - v["cumulative"]) < 1e-6 for v in cases.values())}

    # ---- E8 no charging after termination ------------------------------
    env = EpisodeEnv17(by_map, "cuda", models, norm, cfg)
    env.start(tasks[0])
    done = False
    while not done:
        feas = env.feasible_node_indices()
        _, done, _ = env.step(feas[0] if feas else None)
    n_rec, e_after = len(env.step_records), env.cumulative_energy_j
    try:
        env.step(0)
        raised = False
    except RuntimeError:
        raised = True
    res["E8_no_post_termination_charge"] = {
        "records": n_rec, "steps": int(env.step_idx),
        "energy_unchanged": env.cumulative_energy_j == e_after,
        "step_raised": raised,
        "pass": raised and env.cumulative_energy_j == e_after
                and n_rec == int(env.step_idx)}

    # ---- E13 counters ---------------------------------------------------
    res["E13_counters"] = {
        "attempts": int(env.service_attempt_count),
        "inferences": int(env.completed_inference_count),
        "executed": int(env.executed_navigation_action_count),
        "records": n_rec,
        "pass": (int(env.service_attempt_count) == n_rec
                 and int(env.completed_inference_count)
                 == sum(1 for r in env.step_records if r["d_coll_ms"] > 0)
                 and int(env.executed_navigation_action_count)
                 == sum(1 for r in env.step_records if r["motion_executed"]))}

    # ---- E15 aborted Forward charged in full ---------------------------
    env = EpisodeEnv17(by_map, "cuda", models, norm, free)
    env.start(tasks[0])
    aborted_rec = None
    for _ in range(200):
        neigh = [i for i, m in enumerate(env.metrics())
                 if m.node["model_type"] == "ts"][0]
        _, done, _ = env.step(neigh)
        if env.step_records and env.step_records[-1]["motion_aborted"]:
            aborted_rec = env.step_records[-1]
            break
        if done:
            break
    if aborted_rec is None:
        res["E15_aborted_forward_charged"] = {"pass": None,
                                              "note": "no collision/OOB in this episode"}
    else:
        res["E15_aborted_forward_charged"] = {
            "nav_action": aborted_rec["navigation_action"],
            "e_forward_j": aborted_rec["e_forward_j"],
            "expected": e_forward_j(cfg) if aborted_rec["navigation_action"] == 1 else None,
            "pass": (aborted_rec["navigation_action"] != 1
                     or abs(aborted_rec["e_forward_j"] - e_forward_j(cfg)) < 1e-9)}

    out = ROOT / "smoke/env_consistency_r3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    n_pass = sum(1 for v in res.values() if v.get("pass") is True)
    n_fail = sum(1 for v in res.values() if v.get("pass") is not True)
    print(f"PASS {n_pass} / NOT-PASS {n_fail}")


if __name__ == "__main__":
    main()
