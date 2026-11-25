#!/usr/bin/env python3
"""Stage 15-Extension F2/F5: SR vs energy-budget feasibility from E4 rollouts.

For each association policy we have per-episode (success, energy_j).  A hard
per-task energy budget B constrains an episode to energy_j <= B.  Two things
matter for sizing a dual-constraint CMDP:

  fit_frac(B)  how often an unconstrained policy already satisfies B
  sr_fit(B)    SR among the episodes that satisfy B

sr_fit is the *post-hoc filter* view: it is what a budget would buy if the
policy did not change its behaviour.  If it drops below the unconstrained SR,
the budget is discarding successful episodes, i.e. energy and SR are in
conflict -- which is exactly what the failure-energy numbers show.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUDGETS_J = [10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 80_000, 100_000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4", nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=str(HERE / "budget_sweep.json"))
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.e4], ignore_index=True)
    d = df[df.split == args.split].copy()
    out = {"split": args.split, "budgets_j": BUDGETS_J, "policies": {}}
    for pol, g in d.groupby("policy"):
        sr = float(g.success.mean())
        e = g.energy_j
        row = {"n": int(len(g)), "sr": sr,
               "e_kj_mean": float(e.mean() / 1e3),
               "e_kj_p50": float(e.median() / 1e3),
               "e_kj_p95": float(e.quantile(0.95) / 1e3),
               "sr_success_e_kj": float(g[g.success].energy_j.mean() / 1e3)
               if g.success.any() else None,
               "sr_failure_e_kj": float(g[~g.success].energy_j.mean() / 1e3)
               if (~g.success).any() else None,
               "budget": {}}
        for b in BUDGETS_J:
            m = e <= b
            row["budget"][str(b)] = {
                "fit_frac": float(m.mean()),
                "sr_fit": float(g.success[m].mean()) if m.any() else None,
                "sr_loss_vs_unconstrained": float(sr - g.success[m].mean())
                if m.any() else None,
                "frac_of_successes_lost": float(
                    (~m[g.success]).mean()) if g.success.any() else None,
            }
        out["policies"][pol] = row

    # Pareto front over policies in (energy, SR)
    pts = [(v["e_kj_mean"], v["sr"], k) for k, v in out["policies"].items()]
    front = []
    for e, s, k in pts:
        if not any((e2 <= e and s2 >= s) and (e2 < e or s2 > s) for e2, s2, _ in pts):
            front.append({"policy": k, "e_kj_mean": e, "sr": s})
    out["pareto_front"] = sorted(front, key=lambda r: r["e_kj_mean"])
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
