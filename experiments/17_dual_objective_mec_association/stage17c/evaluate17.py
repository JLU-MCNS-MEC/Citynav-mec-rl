#!/usr/bin/env python3
"""Stage 17-C evaluation: DRL checkpoints and the three baselines.

Baselines use the Stage-17 service latency (not the Stage-16 one):
  random       uniform over feasible_mec_mask
  min_latency  argmin candidate_service_latency_ms among feasible nodes
  max_quality  a Transformer-S node (Stage-14 quality leader), cheapest among
               the feasible TS nodes

Dev split only for tuning/model selection; Val-Seen is reserved for the final
report and is never used here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
for p in [HERE, ROOT / "src", REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from config17 import load_config                                  # noqa: E402
from env16 import build_deployment_by_map, load_models, load_norm  # noqa: E402
from env17 import EpisodeEnv17                                    # noqa: E402
from state17 import build_state17                                 # noqa: E402
from train_association_rl import load_tasks_and_split             # noqa: E402
from nets17 import ActorCritic17, QNet17, masked_softmax          # noqa: E402


def load_policy(ckpt, algo, device):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    pol = sd["policy"]
    if algo == "dqn":
        net = QNet17().to(device); net.load_state_dict(pol["q"]); kind = "q"
    elif algo in ("ppo", "sac"):
        net = ActorCritic17().to(device); net.load_state_dict(pol["ac"]); kind = "ac"
    elif algo in ("td3", "ddpg"):
        net = ActorCritic17().to(device); net.load_state_dict(pol["pi"]); kind = "pi"
    else:
        raise ValueError(algo)
    net.eval()
    return net, kind


def greedy(net, kind, s, mask, device):
    """Deterministic argmax over feasible nodes; None when none is feasible."""
    if float(np.sum(mask)) < 0.5:
        return None
    with torch.no_grad():
        x = torch.as_tensor(s[None], device=device)
        logits = net(x)[0] if kind == "q" else net(x)[0]
        p = masked_softmax(logits.unsqueeze(0),
                           torch.as_tensor(mask[None], device=device))
    return int(p[0].argmax())


def baseline_choice(name, table, rng):
    feas = [r for r in table if r["feasible"]]
    if not feas:
        return None
    if name == "random":
        return int(rng.choice([r["node"] for r in feas]))
    if name == "min_latency":
        return int(min(feas, key=lambda r: r["latency"]["d_coll_ms"])["node"])
    if name == "max_quality":
        ts = [r for r in feas if r["model_type"] == "ts"]
        pool = ts if ts else feas
        return int(min(pool, key=lambda r: r["latency"]["d_coll_ms"])["node"])
    raise ValueError(name)


def run_episode(env, task, chooser):
    env.start(task)
    done = False
    while not done:
        a = chooser(env)
        _, done, _ = env.step(a)          # env.step(None) terminates cleanly
    return {"success": bool(env.success), "failure": env.failure,
            "d_sum_ms": float(env.cumulative_coll_latency_ms),
            "legacy_e2e_ms": float(np.sum(env.episode_e2e)),
            "energy_j": float(env.cumulative_energy_j),
            "energy_ratio": float(env.cumulative_energy_j)
            / max(env.energy_budget_j, 1e-9),
            "steps": int(env.step_idx),
            "attempts": int(env.service_attempt_count),
            "inferences": int(env.completed_inference_count),
            "executed": int(env.executed_navigation_action_count),
            "model_selection": env.model_selection.astype(int).tolist(),
            "osr": int(env.success or env.min_dist <= env.cfg.stop_radius_m)}


def summarize(rows):
    return {"n": len(rows), "sr": float(np.mean([r["success"] for r in rows])),
            "d_sum_ms": float(np.mean([r["d_sum_ms"] for r in rows])),
            "legacy_e2e_ms": float(np.mean([r["legacy_e2e_ms"] for r in rows])),
            "energy_j": float(np.mean([r["energy_j"] for r in rows])),
            "energy_ratio": float(np.mean([r["energy_ratio"] for r in rows])),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "osr": float(np.mean([r["osr"] for r in rows])),
            "failures": {f: int(sum(r["failure"] == f for r in rows))
                         for f in sorted({r["failure"] for r in rows})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--algorithm", default="")
    ap.add_argument("--baselines", default="random,min_latency,max_quality")
    ap.add_argument("--split", choices=["dev"], default="dev")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs/default.json")
    _, dev_tasks = load_tasks_and_split()
    tasks = dev_tasks[: args.n]
    env = EpisodeEnv17(build_deployment_by_map(), args.device,
                       load_models(args.device), load_norm(), cfg)
    res = {}

    if args.ckpt:
        net, kind = load_policy(args.ckpt, args.algorithm, args.device)

        def chooser(e):
            s, meta = build_state17(e, cfg)
            return greedy(net, kind, s, meta["feasible_mec_mask"], args.device)
        rows = [run_episode(env, t, chooser) for t in tasks]
        res[args.algorithm] = summarize(rows)

    for name in [b for b in args.baselines.split(",") if b]:
        rng = np.random.default_rng(42)
        rows = [run_episode(env, t,
                            lambda e, n=name: baseline_choice(n, e.candidate_table(), rng))
                for t in tasks]
        res[name] = summarize(rows)

    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
