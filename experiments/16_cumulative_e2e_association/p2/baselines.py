#!/usr/bin/env python3
"""Stage 16-P2: non-RL association baselines on the 2498 Val-Seen tasks.

Policies
  random       : uniformly random over covered nodes
  min_latency  : node with the smallest predicted E2E (encode+comm+infer)
  max_quality  : node hosting the highest Dev Macro-F1 expert; ties -> min E2E

All three use the same coverage mask and the same cumulative-E2E environment.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP16 = HERE.parent
REPO = EXP16.parents[1]
sys.path.insert(0, str(EXP16 / "p1"))
sys.path.insert(0, str(REPO / "experiments/12_hetero_mec_association/scripts"))

from env16 import (EpisodeEnv16, build_deployment_by_map, load_models,  # noqa: E402
                   load_norm, QUALITY_DEV_F1, MODEL_ORDER, MAX_STEPS)
import protocol as P12  # noqa: E402


def load_val_tasks():
    manifest = json.loads(P12.MANIFEST.read_text(encoding="utf-8"))
    tasks = [t for t in manifest["tasks"] if t["split"] == "val_seen"]
    assert len(tasks) == 2498, len(tasks)
    return tasks


def choose(policy, metrics, rng):
    avail = [i for i, m in enumerate(metrics) if m.available]
    if not avail:
        return int(np.argmin([m.distance for m in metrics]))
    if policy == "random":
        return int(rng.choice(avail))
    if policy == "min_latency":
        return min(avail, key=lambda i: metrics[i].e2e_ms)
    if policy == "max_quality":
        best_q = max(QUALITY_DEV_F1[metrics[i].node["model_type"]] for i in avail)
        cand = [i for i in avail
                if QUALITY_DEV_F1[metrics[i].node["model_type"]] == best_q]
        return min(cand, key=lambda i: metrics[i].e2e_ms)
    raise ValueError(policy)


def metrics_from_episode(env, task):
    steps = env.step_idx
    e2e = np.asarray(env.episode_e2e, np.float64)
    final_dist = float(np.linalg.norm(env.pos - env.target))
    osr = bool(env.min_dist <= 20.0)
    path_len = 5.0 * getattr(env, "n_forward", 0)
    l_star = float(np.linalg.norm(
        np.asarray([task["start_x"], task["start_y"]]) - env.target))
    spl = float(env.success * min(1.0, l_star / path_len)) if path_len > 0 else float(env.success)
    return {
        "episode_id": task["episode_id"], "map": task["map"],
        "length_group": task["length_group"],
        "success": bool(env.success), "osr": osr, "spl": spl,
        "ne_m": final_dist, "min_dist_m": float(env.min_dist),
        "steps": steps, "path_len_m": path_len, "l_star_m": l_star,
        "cum_e2e_ms": float(e2e.sum()), "mean_e2e_ms": float(e2e.mean()) if len(e2e) else 0.0,
        "p50_step_e2e_ms": float(np.percentile(e2e, 50)) if len(e2e) else 0.0,
        "p95_step_e2e_ms": float(np.percentile(e2e, 95)) if len(e2e) else 0.0,
        "failure": env.failure,
        "collision": bool(env.collision), "out_of_bounds": bool(env.oob),
        "node_counts": getattr(env, "_node_counts", {}),
        "model_counts": getattr(env, "_model_counts", {}),
        "comm_ms_sum": getattr(env, "_comm_sum", 0.0),
        "inf_ms_sum": getattr(env, "_inf_sum", 0.0),
        "encode_ms_sum": getattr(env, "_enc_sum", 0.0),
    }


def run(policy, tasks, env, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    rows = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        env.start(task)
        env._node_counts, env._model_counts = {}, {}
        env._comm_sum = env._inf_sum = env._enc_sum = 0.0
        done = False
        while not done:
            metrics = env.metrics()
            a = choose(policy, metrics, rng)
            m = metrics[a]
            env._comm_sum += m.comm_ms
            env._inf_sum += m.inf_ms
            env._enc_sum += env.enc_ms
            gid = m.node["gbs_id"]
            env._node_counts[gid] = env._node_counts.get(gid, 0) + 1
            env._model_counts[m.node["model_type"]] = \
                env._model_counts.get(m.node["model_type"], 0) + 1
            _, done, _ = env.step(a)
        rows.append(metrics_from_episode(env, task))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            sr = np.mean([r["success"] for r in rows])
            print(f"[{policy}] {i+1}/{len(tasks)} elapsed={el:.0f}s "
                  f"eta={el/(i+1)*(len(tasks)-i-1):.0f}s SR={sr:.3f}", flush=True)
    return rows


def summarize(rows, policy):
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    fail = [r for r in rows if not r["success"]]
    cum = np.asarray([r["cum_e2e_ms"] for r in rows])
    mean_step = np.asarray([r["mean_e2e_ms"] for r in rows])
    node_counts = {}
    model_counts = {}
    for r in rows:
        for k, v in r["node_counts"].items():
            node_counts[k] = node_counts.get(k, 0) + v
        for k, v in r["model_counts"].items():
            model_counts[k] = model_counts.get(k, 0) + v
    tot_steps = sum(r["steps"] for r in rows)
    return {
        "policy": policy, "n": n,
        "sr": float(np.mean([r["success"] for r in rows])),
        "osr": float(np.mean([r["osr"] for r in rows])),
        "spl": float(np.mean([r["spl"] for r in rows])),
        "ne_m": float(np.mean([r["ne_m"] for r in rows])),
        "mean_e2e_per_step_ms": float(mean_step.mean()),
        "cum_e2e_mean_ms": float(cum.mean()),
        "cum_e2e_success_only_ms": float(np.mean([r["cum_e2e_ms"] for r in succ])) if succ else None,
        "cum_e2e_failure_only_ms": float(np.mean([r["cum_e2e_ms"] for r in fail])) if fail else None,
        "cum_e2e_p50_ms": float(np.percentile(cum, 50)),
        "cum_e2e_p95_ms": float(np.percentile(cum, 95)),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "success_n": len(succ), "failure_n": len(fail),
        "premature_stop_n": int(sum(1 for r in rows if r["failure"] == "premature_stop")),
        "collision_n": int(sum(1 for r in rows if r["collision"])),
        "oob_n": int(sum(1 for r in rows if r["out_of_bounds"])),
        "step_cap_n": int(sum(1 for r in rows if r["failure"] == "step_cap")),
        "failure_taxonomy": {k: int(sum(1 for r in rows if r["failure"] == k))
                             for k in sorted({r["failure"] for r in rows})},
        "node_selection_share": {k: v / max(tot_steps, 1) for k, v in node_counts.items()},
        "model_selection_share": {k: v / max(tot_steps, 1) for k, v in model_counts.items()},
        "comm_ms_mean": float(np.mean([r["comm_ms_sum"] / max(r["steps"], 1) for r in rows])),
        "inf_ms_mean": float(np.mean([r["inf_ms_sum"] / max(r["steps"], 1) for r in rows])),
        "encode_ms_mean": float(np.mean([r["encode_ms_sum"] / max(r["steps"], 1) for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="random,min_latency,max_quality")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tasks = load_val_tasks()
    if args.limit:
        tasks = tasks[: args.limit]
    by_map = build_deployment_by_map()
    models = load_models(args.device)
    norm = load_norm()
    env = EpisodeEnv16(by_map, args.device, models, norm, reward_variant="A")
    summaries = {}
    for policy in args.policies.split(","):
        rows = run(policy, tasks, env)
        (out / f"results_{policy}.csv").write_text(
            "episode_id,map,length_group,success,osr,spl,ne_m,steps,cum_e2e_ms,"
            "mean_e2e_ms,p95_step_e2e_ms,failure\n")
        import csv
        with (out / f"results_{policy}.csv").open("a", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            for r in rows:
                w.writerow([r["episode_id"], r["map"], r["length_group"],
                            r["success"], r["osr"], f"{r['spl']:.6f}",
                            f"{r['ne_m']:.3f}", r["steps"],
                            f"{r['cum_e2e_ms']:.3f}", f"{r['mean_e2e_ms']:.3f}",
                            f"{r['p95_step_e2e_ms']:.3f}", r["failure"]])
        s = summarize(rows, policy)
        summaries[policy] = s
        (out / f"summary_{policy}.json").write_text(json.dumps(s, indent=2))
        print(json.dumps(s, indent=2))
    (out / "summary_all.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
