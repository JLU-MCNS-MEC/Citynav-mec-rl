#!/usr/bin/env python3
"""Stage 15-Extension F2/F5: SR-vs-energy-budget feasibility surface.

Runs the three static association baselines in closed loop with per-step
energy accounting, then reports for a sweep of episode budgets B:
  * share of episodes whose energy fits B          (budget attainability)
  * success rate among those episodes              (SR under budget)
so a dual-constraint CMDP can later be sized against real numbers instead of
a guessed 100 Wh pack.

Uses Train + Dev task pools only; Val-Seen is never touched for sizing.
"""
from __future__ import annotations

import argparse
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

from baselines import choose                     # noqa: E402
from energy_model import BATTERY_J, step_energy  # noqa: E402
from env16 import (EpisodeEnv16, build_deployment_by_map, load_models,  # noqa: E402
                   load_norm)
from train_association_rl import load_tasks_and_split  # noqa: E402

BUDGETS_J = [1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]


def run_policy(env, tasks, policy, rng):
    rows = []
    for t in tasks:
        env.start(t)
        done = False
        energies = []
        while not done:
            a = choose(policy, env.metrics(), rng)
            _, done, info = env.step(a)
            energies.append(step_energy(a, env.enc_ms / 1e3,
                                        info["comm_ms"] / 1e3, info["inf_ms"] / 1e3))
        rows.append({
            "episode_id": t.get("episode_id", ""), "map": t["map"],
            "split": t.get("split", ""), "policy": policy,
            "success": bool(env.success), "failure": env.failure,
            "steps": env.step_idx, "cum_e2e_ms": float(np.sum(env.episode_e2e)),
            "energy_j": float(sum(e["e_total_j"] for e in energies)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--policies", default="min_latency,max_quality,random")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "feasibility_baselines.csv"))
    args = ap.parse_args()
    torch.set_num_threads(4)

    train_tasks, dev_tasks = load_tasks_and_split()
    rng_sel = np.random.default_rng(20261125)
    tr = [train_tasks[i] for i in rng_sel.permutation(len(train_tasks))[: args.n_train]]
    dv = [dev_tasks[i] for i in rng_sel.permutation(len(dev_tasks))[: args.n_dev]]
    for t in tr:
        t["split"] = "train"
    for t in dv:
        t["split"] = "dev"
    tasks = tr + dv

    env = EpisodeEnv16(build_deployment_by_map(), args.device, load_models(args.device),
                       load_norm(), reward_variant="A", record_steps=True)
    rng = np.random.default_rng(42)
    rows = []
    for pol in args.policies.split(","):
        rows += run_policy(env, tasks, pol, rng)
        part = pd.DataFrame(rows)
        print(f"[{pol}] n={len(part[part.policy == pol])} "
              f"SR={part[part.policy == pol].success.mean():.3f} "
              f"E50={part[part.policy == pol].energy_j.median():.0f}J", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    table = {}
    for (pol, split), g in df.groupby(["policy", "split"]):
        s = g.energy_j
        row = {"n": int(len(g)), "sr": float(g.success.mean()),
               "e_p50": float(s.median()), "e_p95": float(s.quantile(0.95)),
               "e_mean": float(s.mean()),
               "battery": {f"{b}": {"fit_frac": float((s <= b).mean()),
                                    "sr_fit": float(g[g.energy_j <= b].success.mean())
                                    if (g.energy_j <= b).any() else None}
                           for b in BUDGETS_J}}
        table[f"{pol}/{split}"] = row
    out = {"budgets_j": BUDGETS_J, "battery_j": BATTERY_J,
           "battery_wh": BATTERY_J / 3600.0, "table": table,
           "n_train": len(tr), "n_dev": len(dv),
           "note": "fit_frac here is a post-hoc filter on the unconstrained "
                   "policy, i.e. an upper bound on what a budget-constrained "
                   "policy could achieve without changing behaviour."}
    Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
