#!/usr/bin/env python3
"""RL 关联策略的 Val-Seen 独立闭环评测（2,498 任务，与 Baseline 同协议）。

读取训练 run 的 best.pt（最低 Dev E2E 权重），贪心执行关联，逐任务记录
选择的节点/模型/推理时延/通信时延/E2E/成功，输出 journal + summary。
时延口径与训练一致：解析推理（FLOPs/0.2 TFLOPS）+ 通信 + 编码 1.68ms。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from train_association_rl import (STATE_DIM, ActorCritic, EpisodeEnv,  # noqa: E402
                                  N_ACT, QNet, STOP_RADIUS, FORWARD_M,
                                  build_models, masked_argmax)

RES = HERE.parent / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step-cap", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run = RES / a.run_dir if not Path(a.run_dir).is_absolute() else Path(a.run_dir)
    out = RES / a.out_dir if not Path(a.out_dir).is_absolute() else Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(run / "best.pt", map_location=a.device, weights_only=False)
    cfgname = {"A": "A_10xSmall", "B": "B_10xTransformerM",
               "C": "C_5Small_5TM"}[ck["config"]]
    deployment = json.loads((RES / "deployment_configs.json").read_text())[cfgname]
    prior = ck["prior"]
    models = build_models(sorted({"small", "transformer_m"}), a.device)
    env = EpisodeEnv(deployment, prior, a.device, models, a.step_cap)

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

    mani = json.loads((REPO / "experiments/06_closed_loop_success_rate/results"
                       "/seed42/manifest.json").read_text())
    tasks = [t for t in mani["tasks"] if t["split"] == "val_seen"]
    if a.limit:
        tasks = tasks[:a.limit]
    jpath = out / "journal.jsonl"
    done = set()
    if jpath.exists():
        done = {json.loads(l)["episode_id"] for l in jpath.read_text().splitlines()}
    pending = [t for t in tasks if t["episode_id"] not in done]
    print(f"[eval] run={run.name} config={ck['config']} algo={ck['algorithm']} "
          f"tasks={len(tasks)} pending={len(pending)}", flush=True)
    with jpath.open("a") as jf:
        for i, task in enumerate(pending, 1):
            env.start({"map": task["map"], "start_x": task["start_x"],
                       "start_y": task["start_y"], "start_yaw": task["start_yaw"],
                       "target_x": task["target_x"], "target_y": task["target_y"]})
            s, mask = env.state()
            nodes_used = []
            done_ep = False
            while not done_ep:
                act = policy(s, mask)
                _, done_ep, info = env.step(act)
                nodes_used.append({"node": info["node"], "model": info["model"],
                                   "e2e_ms": info["e2e_ms"],
                                   "inf_ms": info["inf_ms"],
                                   "comm_ms": info["comm_ms"]})
                s, mask = env.state()
            dist = float(np.linalg.norm(env.pos - env.target))
            l_star = float(np.linalg.norm(
                np.asarray([task["start_x"], task["start_y"]]) - env.target))
            path_len = FORWARD_M * env.n_forward
            spl = (float(env.success) * min(1.0, l_star / path_len)
                   if path_len > 0 else float(env.success))
            e2e = np.asarray([n["e2e_ms"] for n in nodes_used])
            inf = np.asarray([n["inf_ms"] for n in nodes_used])
            rec = {"episode_id": task["episode_id"], "map": task["map"],
                   "steps": env.step_idx, "success": bool(env.success),
                   "osr": bool(env.min_dist <= STOP_RADIUS),
                   "ne_m": dist, "spl": spl, "steps_forward": env.n_forward,
                   "e2e_mean_ms": float(e2e.mean()),
                   "e2e_p95_ms": float(np.percentile(e2e, 95)),
                   "inf_mean_ms": float(inf.mean()),
                   "inf_share": float((inf / e2e).mean()),
                   "comm_mean_ms": float(np.mean([n["comm_ms"] for n in nodes_used])),
                   "failure": env.failure,
                   "nodes": nodes_used}
            jf.write(json.dumps(rec) + "\n"); jf.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(pending)}", flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines()]
    ok = [r for r in rows if r.get("failure") != "exception"]
    e2e = np.asarray([r["e2e_mean_ms"] for r in ok])
    inf = np.asarray([r["inf_mean_ms"] for r in ok])
    summary = {"run": run.name, "config": ck["config"],
               "algorithm": ck["algorithm"], "seed": ck["seed"],
               "best_episode": ck["best_ep"], "n": len(ok),
               "sr": float(np.mean([r["success"] for r in ok])),
               "osr": float(np.mean([r["osr"] for r in ok])),
               "ne_mean_m": float(np.mean([r["ne_m"] for r in ok])),
               "spl_mean": float(np.mean([r["spl"] for r in ok])),
               "e2e_mean_ms": float(e2e.mean()),
               "e2e_p95_ms": float(np.percentile(e2e, 95)),
               "e2e_p99_ms": float(np.percentile(e2e, 99)),
               "inf_mean_ms": float(inf.mean()),
               "inf_share_mean": float((inf / e2e).mean()),
               "steps_mean": float(np.mean([r["steps"] for r in ok])),
               "failure_counts": {k: sum(1 for r in ok if r["failure"] == k)
                                  for k in sorted({r["failure"] for r in ok})}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[eval done] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
