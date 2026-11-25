#!/usr/bin/env python3
"""Stage 15-Extension F5: 100-task UAV energy statistics on Train tasks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
REPO = EXT.parents[2]
for p in [EXT, REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/16_cumulative_e2e_association/p2",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from env16 import (EpisodeEnv16, build_deployment_by_map, load_models,  # noqa: E402
                   load_norm)
from train_association_rl import load_tasks_and_split  # noqa: E402
from baselines import choose  # noqa: E402
from energy_model import step_energy, BATTERY_J  # noqa: E402


def main():
    ap_n = 100
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=ap_n)
    ap.add_argument("--policy", default="min_latency")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "energy_stats_100.csv"))
    args = ap.parse_args()
    torch.set_num_threads(4)
    train_tasks, dev_tasks = load_tasks_and_split()
    tasks = train_tasks[: args.n]
    by_map = build_deployment_by_map()
    models = load_models(args.device)
    norm = load_norm()
    env = EpisodeEnv16(by_map, args.device, models, norm, reward_variant="A",
                       record_steps=True)
    rng = np.random.default_rng(42)
    rows = []
    for t in tasks:
        env.start(t)
        done = False
        steps_energy = []
        while not done:
            metrics = env.metrics()
            a = choose(args.policy, metrics, rng)
            _, done, info = env.step(a)
            e = step_energy(a, env.enc_ms / 1e3, info["comm_ms"] / 1e3,
                            info["inf_ms"] / 1e3)
            steps_energy.append(e)
        agg = {k: float(sum(s[k] for s in steps_energy))
               for k in ("e_move_j", "e_wait_j", "e_aux_j", "e_total_j")}
        agg.update({
            "episode_id": t.get("episode_id", ""), "map": t["map"],
            "success": bool(env.success), "failure": env.failure,
            "steps": env.step_idx, "cum_e2e_ms": float(np.sum(env.episode_e2e)),
            "e_wh": agg["e_total_j"] / 3600.0,
            "battery_frac": agg["e_total_j"] / BATTERY_J,
            "n_forward": env.n_forward,
        })
        rows.append(agg)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    succ = df[df.success]
    fail = df[~df.success]
    def q(s, p):
        return float(np.percentile(s, p)) if len(s) else None
    summary = {
        "n": int(len(df)), "policy": args.policy,
        "success_n": int(len(succ)),
        "energy_j_mean": float(df.e_total_j.mean()),
        "energy_j_p50": q(df.e_total_j, 50), "energy_j_p90": q(df.e_total_j, 90),
        "energy_j_p95": q(df.e_total_j, 95),
        "energy_wh_mean": float(df.e_wh.mean()),
        "success_energy_j_mean": float(succ.e_total_j.mean()) if len(succ) else None,
        "failure_energy_j_mean": float(fail.e_total_j.mean()) if len(fail) else None,
        "component_share": {
            "move": float(df.e_move_j.sum() / df.e_total_j.sum()),
            "wait": float(df.e_wait_j.sum() / df.e_total_j.sum()),
            "aux": float(df.e_aux_j.sum() / df.e_total_j.sum()),
        },
        "battery_wh": float(BATTERY_J / 3600.0),
        "battery_frac_mean": float(df.battery_frac.mean()),
        "battery_frac_p95": q(df.battery_frac, 95),
        "corr_energy_steps": float(df.e_total_j.corr(df.steps)),
        "corr_energy_e2e": float(df.e_total_j.corr(df.cum_e2e_ms)),
        "battery_capacity_assumed_wh": 100.0,
    }
    (HERE / "energy_stats_100_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
