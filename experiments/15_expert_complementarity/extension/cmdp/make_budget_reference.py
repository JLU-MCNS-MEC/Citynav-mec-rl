#!/usr/bin/env python3
"""Stage 15-Extension F5: per-task reference energy for the relative budget.

Reference policy: Min-Latency (lowest step E2E), the same baseline used in
Stage 16.  Kinematics v = 3 m/s, matching the CMDP environment, so the budget
stays relative: B_task = slack * E_ref(task).

Writes reference_j keyed by the task index in the Train/Dev pool.
"""
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
for p in [EXT / "energy", REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/16_cumulative_e2e_association/p2",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from baselines import choose                        # noqa: E402
from env16 import (EpisodeEnv16, build_deployment_by_map,  # noqa: E402
                   load_models, load_norm)
from train_association_rl import load_tasks_and_split  # noqa: E402

sys.path.insert(0, str(HERE))
from env_constrained import ConstrainedEnv          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--v-forward", type=float, default=3.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "budget_reference.json"))
    args = ap.parse_args()
    torch.set_num_threads(4)

    train_tasks, dev_tasks = load_tasks_and_split()
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train_tasks))[: args.n_train]
    tr = [train_tasks[i] for i in idx]
    tasks = tr + list(dev_tasks)

    env = ConstrainedEnv(build_deployment_by_map(), args.device,
                         load_models(args.device), load_norm(),
                         reward_variant="A", budgets={}, slack=1.0,
                         v_forward=args.v_forward)
    rng2 = np.random.default_rng(args.seed)
    ref, rows = {}, []
    for t in tasks:
        env.start(t)
        done = False
        while not done:
            node = choose("min_latency", env.metrics(), rng2)
            _, done, _ = env.step(node)
        ref[int(t["index"])] = float(env.energy_j)
        rows.append({"index": int(t["index"]), "map": t["map"],
                     "energy_j": float(env.energy_j),
                     "steps": env.step_idx, "success": bool(env.success)})

    out = {"v_forward": args.v_forward, "slack_default": 1.2,
           "n": len(rows), "reference_j": ref,
           "summary": {
               "train_n": len(tr), "dev_n": len(dev_tasks),
               "mean_j": float(np.mean([r["energy_j"] for r in rows])),
               "p50_j": float(np.median([r["energy_j"] for r in rows])),
               "p95_j": float(np.percentile([r["energy_j"] for r in rows], 95)),
               "sr": float(np.mean([r["success"] for r in rows])),
           }}
    Path(args.out).write_text(json.dumps(out, indent=2))
    import pandas as pd
    pd.DataFrame(rows).to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
