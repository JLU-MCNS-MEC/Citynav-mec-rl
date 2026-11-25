#!/usr/bin/env python3
"""Closed-loop evaluation of a trained CMDP policy on the Dev pool.

Reports SR, cumulative E2E, energy vs the relative budget, and the failure
breakdown.  Dev only: the Val-Seen pool stays untouched for tuning/selection.
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
for p in [HERE, EXT / "energy", REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/16_cumulative_e2e_association/p3",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from env_constrained import ConstrainedEnv, load_budgets   # noqa: E402
from env16 import build_deployment_by_map, load_models, load_norm  # noqa: E402
from train16 import (ActorCritic, QNet, masked_softmax,  # noqa: E402
                     state_features)
from train_association_rl import load_tasks_and_split       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--budget-file", required=True)
    ap.add_argument("--slack", type=float, default=0.8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    budgets = load_budgets(args.budget_file)
    _, dev_tasks = load_tasks_and_split()
    dv = [t for t in dev_tasks if int(t["index"]) in budgets]
    env = ConstrainedEnv(build_deployment_by_map(), args.device,
                         load_models(args.device), load_norm(),
                         reward_variant="A", budgets=budgets, slack=args.slack,
                         v_forward=float(json.loads(
                             Path(args.budget_file).read_text())["v_forward"]))
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    pol = sd["policy"]
    # ppo / sac -> "ac"; td3 / ddpg -> "pi"; dqn -> "q"
    if "ac" in pol:
        net = ActorCritic().to(args.device)
        net.load_state_dict(pol["ac"])
    elif "pi" in pol:
        net = ActorCritic().to(args.device)
        net.load_state_dict(pol["pi"])
    elif "q" in pol:
        net = QNet().to(args.device)
        net.load_state_dict(pol["q"])
    else:
        raise ValueError(f"unrecognised policy keys: {list(pol)}")
    net.eval()

    def act_greedy(s, mask):
        with torch.no_grad():
            x = torch.as_tensor(s[None], device=args.device)
            if isinstance(net, QNet):
                logits = net(x)[0]
            else:
                logits, _ = net(x)
            p = masked_softmax(logits.unsqueeze(0),
                               torch.as_tensor(mask[None], device=args.device))
        return int(p[0].argmax())

    rows = np.zeros(10, int)
    out = []
    for t in dv:
        env.start(t)
        s, m = state_features(env)
        done = False
        nodes = []
        while not done:
            a = act_greedy(s, m); nodes.append(a); rows[a] += 1
            _, done, _ = env.step(a)
            s, m = state_features(env)
        out.append({"index": int(t["index"]), "success": bool(env.success),
                    "failure": env.failure, "steps": env.step_idx,
                    "energy_j": env.energy_j, "budget_j": env.budget_j,
                    "ratio": env.energy_ratio(),
                    "cum_e2e_ms": float(np.sum(env.episode_e2e)),
                    "nodes": nodes})
    summ = {
        "ckpt": args.ckpt, "slack": args.slack, "n": len(out),
        "sr": float(np.mean([r["success"] for r in out])),
        "mean_cum_e2e_ms": float(np.mean([r["cum_e2e_ms"] for r in out])),
        "mean_energy_j": float(np.mean([r["energy_j"] for r in out])),
        "mean_ratio": float(np.mean([r["ratio"] for r in out])),
        "viol_rate": float(np.mean([r["ratio"] > 1 for r in out])),
        "failure_counts": {f: int(sum(r["failure"] == f for r in out))
                           for f in sorted({r["failure"] for r in out})},
        "node_usage": rows.tolist(),
        "lam_e": sd.get("lam_e"), "lam_sr": sd.get("lam_sr"),
    }
    print(json.dumps(summ, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({**summ, "episodes": out}, indent=2))


if __name__ == "__main__":
    main()
