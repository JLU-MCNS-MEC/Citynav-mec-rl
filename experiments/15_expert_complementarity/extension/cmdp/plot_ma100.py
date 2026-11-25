#!/usr/bin/env python3
"""MA(100) curves for the constrained CMDP runs (reward and cumulative E2E)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load(run: Path) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(l) for l in open(run / "history.jsonl")])
    df = df.sort_values("episode").reset_index(drop=True)
    for col, out in [("reward", "reward_ma100"),
                     ("cum_e2e_ms", "e2e_ma100"),
                     ("energy_ratio", "ratio_ma100"),
                     ("base_success", "sr_ma100"),
                     ("lam_e", "lam_e_ma100"),
                     ("lam_sr", "lam_sr_ma100")]:
        df[out] = df[col].astype(float).rolling(100, min_periods=20).mean()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    labels = args.labels or [Path(r).name for r in args.runs]

    curves = []
    for run, lab in zip(args.runs, labels):
        df = load(Path(run))
        df["run"] = lab
        curves.append(df[["episode", "run", "reward_ma100", "e2e_ma100",
                          "ratio_ma100", "sr_ma100", "lam_e_ma100",
                          "lam_sr_ma100"]])
    allc = pd.concat(curves, ignore_index=True)
    allc.to_csv(out / "ma100_curves.csv", index=False)

    specs = [("reward_ma100", "Reward MA(100)", "reward"),
             ("e2e_ma100", "Cumulative E2E MA(100) [ms]", "cum_e2e"),
             ("sr_ma100", "Success rate MA(100)", "sr"),
             ("ratio_ma100", "Energy / budget MA(100)", "ratio"),
             ("lam_e_ma100", "lambda_E MA(100)", "lambda_e"),
             ("lam_sr_ma100", "lambda_SR MA(100)", "lambda_sr")]
    for col, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
        for lab in labels:
            g = allc[allc.run == lab]
            ax.plot(g.episode, g[col], label=lab, lw=1.6)
        ax.set_xlabel("episode"); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3); ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"{fname}_ma100.png")
        plt.close(fig)
    print(json.dumps({"runs": labels, "out": str(out),
                      "points": int(len(allc))}))


if __name__ == "__main__":
    main()
