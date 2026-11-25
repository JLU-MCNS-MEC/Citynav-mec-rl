#!/usr/bin/env python3
"""Baselines on the same Dev pool / budget accounting as the CMDP agents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
REPO = EXT.parents[2]
for p in [HERE, EXT / "energy", REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/16_cumulative_e2e_association/p2",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from baselines import choose                       # noqa: E402
from env_constrained import ConstrainedEnv, load_budgets   # noqa: E402
from env16 import build_deployment_by_map, load_models, load_norm  # noqa: E402
from train_association_rl import load_tasks_and_split       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-file", default=str(HERE / "budget_reference_1000.json"))
    ap.add_argument("--slack", type=float, default=0.8)
    ap.add_argument("--policies", default="min_latency,max_quality,random")
    ap.add_argument("--out", default=str(HERE / "out/slack08/eval/baselines.json"))
    args = ap.parse_args()
    budgets = load_budgets(args.budget_file)
    _, dev_tasks = load_tasks_and_split()
    dv = [t for t in dev_tasks if int(t["index"]) in budgets]
    env = ConstrainedEnv(build_deployment_by_map(), "cuda", load_models("cuda"),
                         load_norm(), reward_variant="A", budgets=budgets,
                         slack=args.slack,
                         v_forward=float(json.loads(
                             Path(args.budget_file).read_text())["v_forward"]))
    res = {}
    for pol in args.policies.split(","):
        rng = np.random.default_rng(42)
        rows = []
        for t in dv:
            env.start(t)
            done = False
            while not done:
                a = choose(pol, env.metrics(), rng)
                _, done, _ = env.step(a)
            rows.append({"success": bool(env.success), "failure": env.failure,
                         "steps": env.step_idx, "energy_j": env.energy_j,
                         "ratio": env.energy_ratio(),
                         "cum_e2e_ms": float(np.sum(env.episode_e2e))})
        res[pol] = {"n": len(rows),
                    "sr": float(np.mean([r["success"] for r in rows])),
                    "mean_cum_e2e_ms": float(np.mean([r["cum_e2e_ms"] for r in rows])),
                    "mean_energy_j": float(np.mean([r["energy_j"] for r in rows])),
                    "mean_ratio": float(np.mean([r["ratio"] for r in rows])),
                    "viol_rate": float(np.mean([r["ratio"] > 1 for r in rows])),
                    "failure_counts": {f: int(sum(r["failure"] == f for r in rows))
                                       for f in sorted({r["failure"] for r in rows})}}
        print(pol, json.dumps(res[pol]), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
