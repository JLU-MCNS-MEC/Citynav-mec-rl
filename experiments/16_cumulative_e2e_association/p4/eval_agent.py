#!/usr/bin/env python3
"""Stage 16-P4: evaluate a trained association agent on the 2498 Val-Seen tasks."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP16 = HERE.parent
REPO = EXP16.parents[1]
for p in [EXP16 / "p1", EXP16 / "p2", EXP16 / "p3",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from env16 import (EpisodeEnv16, build_deployment_by_map, load_models,  # noqa: E402
                   load_norm, MAX_STEPS)
from baselines import load_val_tasks, metrics_from_episode, summarize  # noqa: E402
from train16 import QNet, ActorCritic, state_features, N_ACT, STATE_DIM  # noqa: E402


def load_policy(ckpt, algorithm, device):
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    st = payload["policy"]
    if algorithm == "dqn":
        q = QNet().to(device); q.load_state_dict(st["q"]); q.eval()
        return {"q": q}
    if algorithm == "ppo":
        ac = ActorCritic().to(device); ac.load_state_dict(st["ac"]); ac.eval()
        return {"ac": ac}
    if algorithm in ("td3", "ddpg"):
        pi = ActorCritic().to(device); pi.load_state_dict(st["pi"]); pi.eval()
        return {"pi": pi}
    ac = ActorCritic().to(device); ac.load_state_dict(st["ac"]); ac.eval()
    return {"ac": ac}


def act(policy, algorithm, s, mask, device):
    with torch.no_grad():
        x = torch.as_tensor(s[None], device=device)
        if algorithm == "dqn":
            qv = policy["q"](x)[0].cpu().numpy()
        elif algorithm == "ppo":
            logits, _ = policy["ac"](x); qv = logits[0].cpu().numpy()
        elif algorithm in ("td3", "ddpg"):
            logits, _ = policy["pi"](x); qv = logits[0].cpu().numpy()
        else:
            logits, _ = policy["ac"](x); qv = logits[0].cpu().numpy()
    qv[mask < 0.5] = -1e18
    return int(np.argmax(qv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--algorithm", required=True,
                    choices=["dqn", "ppo", "td3", "sac", "ddpg"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(args.ckpt).parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tasks = load_val_tasks()
    by_map = build_deployment_by_map()
    models = load_models(args.device)
    norm = load_norm()
    env = EpisodeEnv16(by_map, args.device, models, norm, reward_variant="A")
    policy = load_policy(args.ckpt, args.algorithm, args.device)
    rows = []
    for i, task in enumerate(tasks):
        env.start(task)
        env._node_counts, env._model_counts = {}, {}
        env._comm_sum = env._inf_sum = env._enc_sum = 0.0
        s, mask = state_features(env)
        done = False
        while not done:
            metrics = env.metrics()
            a = act(policy, args.algorithm, s, mask, args.device)
            m = metrics[a]
            env._comm_sum += m.comm_ms
            env._inf_sum += m.inf_ms
            env._enc_sum += env.enc_ms
            gid = m.node["gbs_id"]
            env._node_counts[gid] = env._node_counts.get(gid, 0) + 1
            env._model_counts[m.node["model_type"]] = \
                env._model_counts.get(m.node["model_type"], 0) + 1
            _, done, _ = env.step(a)
            s, mask = state_features(env)
        rows.append(metrics_from_episode(env, task))
        if (i + 1) % 200 == 0:
            print(f"[{args.algorithm}] {i+1}/{len(tasks)} "
                  f"SR={np.mean([r['success'] for r in rows]):.3f}", flush=True)
    with (out / "episode_results.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["episode_id", "map", "length_group", "success", "osr", "spl",
                    "ne_m", "steps", "cum_e2e_ms", "mean_e2e_ms",
                    "p95_step_e2e_ms", "failure"])
        for r in rows:
            w.writerow([r["episode_id"], r["map"], r["length_group"], r["success"],
                        r["osr"], f"{r['spl']:.6f}", f"{r['ne_m']:.3f}", r["steps"],
                        f"{r['cum_e2e_ms']:.3f}", f"{r['mean_e2e_ms']:.3f}",
                        f"{r['p95_step_e2e_ms']:.3f}", r["failure"]])
    s = summarize(rows, args.algorithm)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
