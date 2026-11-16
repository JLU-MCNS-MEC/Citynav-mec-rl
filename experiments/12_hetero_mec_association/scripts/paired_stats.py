#!/usr/bin/env python3
"""Paired per-task statistics for Stage 12 policies (SR and E2E).

For every group and every pair of policies with a per-episode journal, report:
  * McNemar exact p-value for SR differences (paired tasks);
  * paired bootstrap 95% CI for the E2E mean difference (10,000 resamples).

Outputs results/summary/paired_stats.{json,md}.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

GROUP_ORDER = ("two_homo", "five_homo", "two_hetero", "five_hetero")
STRATEGIES = ("random", "max_rate", "max_quality", "oracle")
ALGOS = ("ppo", "dqn")
SEEDS = (42, 43)
OUT = P.RES / "summary"


def read_baseline(group, strategy):
    rows = {}
    d = P.RES / "baselines" / group / strategy / f"seed{P.EVAL_SEED}"
    for p in d.glob("journal.part*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("execution_status") == "ok" and r.get("steps", 0) > 0:
                rows[r["episode_id"]] = (bool(r["success"]), r["e2e_mean_ms"])
    return rows


def read_rl(group, algo, seed):
    p = P.RES / "eval" / f"{group}_{algo}_seed{seed}" / "journal.jsonl"
    rows = {}
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("steps", 0) > 0:
            rows[r["episode_id"]] = (bool(r["success"]), r["e2e_mean_ms"])
    return rows


def policies(group):
    out = {s: read_baseline(group, s) for s in STRATEGIES}
    for algo in ALGOS:
        for seed in SEEDS:
            out[f"{algo}-s{seed}"] = read_rl(group, algo, seed)
    return {k: v for k, v in out.items() if v}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    # two-sided exact binomial with p=0.5, computed in log space
    from math import lgamma, log
    k = min(b, c)
    logs = []
    for i in range(0, k + 1):
        logs.append(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1))
    log_sum = float(np.logaddexp.reduce(np.asarray(logs)))
    log_p = log(2.0) + log_sum - n * log(2.0)
    return float(min(1.0, np.exp(log_p)))


def paired_e2e(ids, a, b, rng, reps=10000):
    va = np.asarray([a[i][1] for i in ids], np.float64)
    vb = np.asarray([b[i][1] for i in ids], np.float64)
    diff = va - vb
    means = np.empty(reps, np.float64)
    n = len(ids)
    for k in range(reps):
        idx = rng.integers(0, n, size=n)
        means[k] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261116)
    results = {}
    lines = ["# Paired per-task statistics", ""]
    for g in GROUP_ORDER:
        pol = policies(g)
        if len(pol) < 2:
            continue
        results[g] = {}
        lines.append(f"## {g}")
        lines.append("")
        lines.append("| pair | n | ΔSR (pp) | McNemar p | ΔE2E (ms) | 95% CI |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for a, b in combinations(pol, 2):
            ids = sorted(set(pol[a]) & set(pol[b]))
            if len(ids) < 10:
                continue
            sa = np.mean([pol[a][i][0] for i in ids])
            sb = np.mean([pol[b][i][0] for i in ids])
            b01 = sum(1 for i in ids if pol[a][i][0] and not pol[b][i][0])
            b10 = sum(1 for i in ids if not pol[a][i][0] and pol[b][i][0])
            p = mcnemar_exact(b01, b10)
            d, lo, hi = paired_e2e(ids, pol[a], pol[b], rng)
            results[g][f"{a} vs {b}"] = {
                "n": len(ids), "delta_sr_pp": 100 * (sa - sb),
                "mcnemar_p": p, "delta_e2e_ms": d,
                "delta_e2e_ci": [lo, hi]}
            lines.append(f"| {a} vs {b} | {len(ids)} | {100*(sa-sb):+.2f} | "
                         f"{p:.3g} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
        lines.append("")
    (OUT / "paired_stats.json").write_text(json.dumps(results, indent=1),
                                           encoding="utf-8")
    (OUT / "paired_stats.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT/'paired_stats.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
