#!/usr/bin/env python3
"""Stage 12: Val-Seen (2,498) greedy evaluation of one RL association run.

Same protocol as the baselines: real closed loop, step cap 200, STOP radius
20 m, per-step decomposition (encode / comm / inference / E2E) and expert share.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402
from train_association_rl import (ActorCritic, EpisodeEnv, N_ACT, QNet,  # noqa: E402
                                  STOP_RADIUS, load_models, masked_argmax)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step-cap", type=int, default=P.MAX_STEPS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run = Path(a.run_dir)
    if not run.is_absolute():
        run = P.RES / run
    out = Path(a.out_dir)
    if not out.is_absolute():
        out = P.RES / out
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(run / "best.pt", map_location=a.device, weights_only=False)
    group = ck["group"]
    prior = ck["prior"]
    types = sorted(set(P.GROUP_SPECS[group]["models"]))
    models = load_models(types, a.device)
    env = EpisodeEnv(P.deployment(group), prior, a.device, models, a.step_cap)

    if ck["algorithm"] == "dqn":
        net = QNet().to(a.device).eval()
        net.load_state_dict(ck["online"])

        def policy(s, mask):
            with torch.no_grad():
                q = net(torch.as_tensor(s[None], device=a.device))
            return masked_argmax(q[0], mask)
    else:
        net = ActorCritic().to(a.device).eval()
        net.load_state_dict(ck["ac"])

        def policy(s, mask):
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(s[None], device=a.device))
            lg = logits[0].cpu().numpy().astype(np.float64).copy()
            lg[mask < 0.5] = -1e18
            return int(np.argmax(lg))

    tasks = P.eval_tasks(a.limit)
    jpath = out / "journal.jsonl"
    done = set()
    if jpath.exists():
        done = {json.loads(line)["episode_id"]
                for line in jpath.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    pending = [t for t in tasks if t["episode_id"] not in done]
    print(f"[eval] run={run.name} group={group} algo={ck['algorithm']} "
          f"seed={ck['seed']} tasks={len(tasks)} pending={len(pending)}",
          flush=True)
    model_steps = Counter()
    node_steps = Counter()
    total_steps = 0
    with jpath.open("a", encoding="utf-8") as jf:
        for i, task in enumerate(pending, 1):
            env.start(task)
            s, mask = env.state()
            used = []
            done_ep = False
            while not done_ep:
                act = policy(s, mask)
                _, done_ep, info = env.step(act)
                used.append({"node": info["node"], "model": info["model"],
                             "e2e_ms": info["e2e_ms"],
                             "inf_ms": info["inf_ms"],
                             "comm_ms": info["comm_ms"],
                             "encode_ms": env.enc_ms})
                s, mask = env.state()
            dist = float(np.linalg.norm(env.pos - env.target))
            l_star = float(np.linalg.norm(
                np.asarray([task["start_x"], task["start_y"]]) - env.target))
            path_len = P.FORWARD_M * env.n_forward
            spl = (float(env.success) * min(1.0, l_star / path_len)
                   if path_len > 0 else float(env.success))
            e2e = np.asarray([n["e2e_ms"] for n in used])
            inf = np.asarray([n["inf_ms"] for n in used])
            comm = np.asarray([n["comm_ms"] for n in used])
            enc = np.asarray([n["encode_ms"] for n in used])
            for n in used:
                model_steps[n["model"]] += 1
                node_steps[n["node"]] += 1
            total_steps += len(used)
            rec = {"episode_id": task["episode_id"], "map": task["map"],
                   "index": int(task["index"]),
                   "length_group": task["length_group"], "group": group,
                   "algorithm": ck["algorithm"], "seed": ck["seed"],
                   "steps": env.step_idx, "success": bool(env.success),
                   "osr": bool(env.min_dist <= STOP_RADIUS),
                   "ne_m": dist, "spl": spl,
                   "steps_forward": env.n_forward,
                   "e2e_mean_ms": float(e2e.mean()),
                   "e2e_p95_ms": float(np.percentile(e2e, 95)),
                   "inf_mean_ms": float(inf.mean()),
                   "comm_mean_ms": float(comm.mean()),
                   "encode_mean_ms": float(enc.mean()),
                   "inf_share": float((inf / e2e).mean()),
                   "failure": env.failure, "nodes": used}
            jf.write(json.dumps(rec) + "\n")
            jf.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(pending)}", flush=True)

    rows = [json.loads(line) for line in
            jpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [r for r in rows if r.get("steps", 0) > 0]
    e2e = np.asarray([r["e2e_mean_ms"] for r in ok])
    inf = np.asarray([r["inf_mean_ms"] for r in ok])
    comm = np.asarray([r["comm_mean_ms"] for r in ok])
    enc = np.asarray([r["encode_mean_ms"] for r in ok])
    summary = {
        "run": run.name, "group": group, "algorithm": ck["algorithm"],
        "seed": ck["seed"], "best_episode": ck["best_ep"],
        "n": len(ok),
        "sr": float(np.mean([r["success"] for r in ok])),
        "osr": float(np.mean([r["osr"] for r in ok])),
        "ne_mean_m": float(np.mean([r["ne_m"] for r in ok])),
        "spl_mean": float(np.mean([r["spl"] for r in ok])),
        "steps_mean": float(np.mean([r["steps"] for r in ok])),
        "e2e_mean_ms": float(e2e.mean()),
        "e2e_p50_ms": float(np.percentile(e2e, 50)),
        "e2e_p95_ms": float(np.percentile(e2e, 95)),
        "e2e_p99_ms": float(np.percentile(e2e, 99)),
        "inf_mean_ms": float(inf.mean()),
        "comm_mean_ms": float(comm.mean()),
        "encode_mean_ms": float(enc.mean()),
        "inf_share_mean": float((inf / e2e).mean()),
        "model_step_share": {k: v / max(total_steps, 1)
                             for k, v in model_steps.items()},
        "node_step_share": {k: v / max(total_steps, 1)
                            for k, v in node_steps.items()},
        "failure_counts": {k: sum(1 for r in ok if r["failure"] == k)
                           for k in sorted({r["failure"] for r in ok})},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 ensure_ascii=False))
    print("[eval done] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
