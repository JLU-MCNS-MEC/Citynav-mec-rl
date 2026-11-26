#!/usr/bin/env python3
"""Task-level paired comparison: PPO / SAC vs the three baselines on Dev-60."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_last_dev(path: Path) -> dict:
    rows = [json.loads(x) for x in open(path)]
    return rows[-1]


def stats(a: dict, b: dict, name_a: str, name_b: str) -> dict:
    ta = {r["task_index"]: r for r in a["per_task"]}
    tb = {r["task_index"]: r for r in b["per_task"]}
    common = sorted(set(ta) & set(tb))
    sa = {k for k in common if ta[k]["success"]}
    sb = {k for k in common if tb[k]["success"]}
    both = sa & sb
    only_a = sa - sb
    only_b = sb - sa
    dsum_both = ([ta[k]["d_sum_ms"] for k in both],
                 [tb[k]["d_sum_ms"] for k in both])
    return {
        "n_common": len(common),
        f"sr_{name_a}": len(sa) / max(len(common), 1),
        f"sr_{name_b}": len(sb) / max(len(common), 1),
        "both_success": len(both), f"only_{name_a}": len(only_a),
        f"only_{name_b}": len(only_b), "neither": len(common) - len(sa | sb),
        "d_sum_on_both_success": {
            name_a: float(pd.Series(dsum_both[0]).mean()) if both else None,
            name_b: float(pd.Series(dsum_both[1]).mean()) if both else None,
        },
        "only_a_tasks": sorted(only_a), "only_b_tasks": sorted(only_b),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="stage17d/p1")
    ap.add_argument("--out", default="stage17d/p1/paired_analysis.json")
    args = ap.parse_args()
    root = Path(args.root)
    base = json.loads((root / "baselines_dev.json").read_text())
    res = {}
    for algo in ("ppo", "sac"):
        dev = load_last_dev(root / algo / "dev.jsonl")
        for b in ("random", "min_latency", "max_quality"):
            res[f"{algo}_vs_{b}"] = stats(dev, base[b], algo, b)
    Path(args.out).write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        a, b = k.split("_vs_")
        print("%s: SR %.3f vs %.3f | only-%s %d, only-%s %d, both %d" % (
            k, v["sr_" + a], v["sr_" + b], a, v["only_" + a], b,
            v["only_" + b], v["both_success"]))
    print(json.dumps(res, indent=2)[:1500])


if __name__ == "__main__":
    main()
