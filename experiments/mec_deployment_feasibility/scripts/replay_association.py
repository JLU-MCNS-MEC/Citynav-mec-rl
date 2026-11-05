#!/usr/bin/env python3
"""阶段 7 补充：A/B 配置的策略级结果重放。

A(10×Small) / B(10×TM) 中所有节点托管同一模型：关联选择仅影响时延，不影响
动作轨迹（模型输入与所选 MEC 无关，与 v8.2 "model type / compute capacity
对 MEC 输入不可见" 语义一致）。因此用阶段 1 已完成的同协议闭环轨迹
（small / v5fix）逐 step 重放选路与时延，结果与"重新跑 5 遍闭环"完全等价。

输出 results/assoc_{A,B}_{strategy}.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_association_loop import (PAYLOAD_BITS, QUALITY,  # noqa: E402
                                  node_metrics, select_node)

REPO = HERE.parents[2]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
RES = HERE.parent / "results"
EVAL_DIRS = {"A": REPO / "experiments/v4_closed_loop_revalidation/results"
             "/small_val_seen_20261103/episodes",
             "B": REPO / "experiments/v4_closed_loop_revalidation/results"
             "/v5fix_val_seen_20261103/episodes"}
CONFIG_KEYS = {"A": "A_10xSmall", "B": "B_10xTransformerM"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["A", "B"], required=True)
    ap.add_argument("--strategy", required=True,
                    choices=["random", "nearest", "max_rate", "min_latency",
                             "best_quality"])
    args = ap.parse_args()

    cfg = json.loads((RES / "deployment_configs.json").read_text())
    by_map: dict[str, list[dict]] = {}
    for n in cfg[CONFIG_KEYS[args.config]]:
        by_map.setdefault(n["map"], []).append(n)

    rng = np.random.default_rng(42)
    recs = [json.loads(p.read_text())
            for p in sorted(EVAL_DIRS[args.config].glob("*.json"))]
    e2e_all, inf_all, comm_all = [], [], []
    per_task = []
    for r in recs:
        nodes = by_map[r["map"]]
        e2e, inf, comm, nsteps = [], [], [], 0
        for s in r.get("trajectory", []):
            pos = np.asarray([s["x"], s["y"]], np.float64)
            node, metrics = select_node(nodes, pos, args.strategy, rng)
            e2e.append(metrics["e2e_ms"])
            inf.append(metrics["inf_ms"])
            comm.append(metrics["comm_ms"])
            nsteps += 1
        if not e2e:
            continue
        e2e = np.asarray(e2e)
        inf = np.asarray(inf)
        comm = np.asarray(comm)
        e2e_all.extend(e2e.tolist())
        inf_all.extend(inf.tolist())
        comm_all.extend(comm.tolist())
        per_task.append({"episode_id": r["episode_id"], "success": r["success"],
                         "steps": nsteps,
                         "e2e_mean_ms": float(e2e.mean()),
                         "inf_share": float((inf / e2e).mean())})
    e2e_all = np.asarray(e2e_all)
    inf_all = np.asarray(inf_all)
    comm_all = np.asarray(comm_all)
    out = {"config": args.config, "strategy": args.strategy,
           "n_tasks": len(per_task),
           "n_steps": int(len(e2e_all)),
           "sr": float(np.mean([p["success"] for p in per_task])),
           "e2e_mean_ms": float(e2e_all.mean()),
           "e2e_p50_ms": float(np.percentile(e2e_all, 50)),
           "e2e_p95_ms": float(np.percentile(e2e_all, 95)),
           "e2e_p99_ms": float(np.percentile(e2e_all, 99)),
           "inf_mean_ms": float(inf_all.mean()),
           "comm_mean_ms": float(comm_all.mean()),
           "inf_share_mean": float((inf_all / e2e_all).mean()),
           "comm_share_mean": float((comm_all / e2e_all).mean()),
           "per_task": per_task}
    path = RES / f"assoc_{args.config}_{args.strategy}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{args.config}/{args.strategy}: SR {out['sr']:.4f} | "
          f"E2E mean {out['e2e_mean_ms']:.1f} P95 {out['e2e_p95_ms']:.1f} | "
          f"inf share {out['inf_share_mean']:.4f} -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
