#!/usr/bin/env python3
"""Stage 17-D: MA100 comparison of the five algorithms (reward / D_sum / SR)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = {"dqn": "#1f77b4", "ddpg": "#2ca02c", "td3": "#9467bd",
     "sac": "#d62728", "ppo": "#8c564b"}


def load(p: Path):
    return pd.DataFrame([json.loads(x) for x in open(p)]) if p.exists() else None


def ma(ax, df, col, label, color, win=100):
    if df is None or col not in df or df[col].isna().all():
        return
    ax.plot(df.episode, df[col].rolling(win, min_periods=20).mean(),
            label=label, color=color, lw=1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--algos", default="dqn,ddpg,td3,sac,ppo")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    algos = [a for a in args.algos.split(",") if a]
    h = {a: load(root / a / "history.jsonl") for a in algos}
    d = {a: load(root / a / "dev.jsonl") for a in algos}

    for col, ylab, fn in [("reward", "episode reward (MA100)", "reward_ma100"),
                          ("d_sum_ms", "cumulative D_coll ms (MA100)", "d_sum_ma100"),
                          ("success", "train success (MA100)", "train_sr_ma100"),
                          ("energy_j", "cumulative energy J (MA100)", "energy_ma100")]:
        fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
        for a in algos:
            ma(ax, h[a], col, a, C[a])
        ax.set_xlabel("episode"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
        ax.legend(frameon=False, ncol=5)
        fig.tight_layout(); fig.savefig(out / f"{fn}.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    for a in algos:
        if d[a] is not None:
            ax.plot(d[a].episode, d[a].sr, marker="o", ms=3, label=a, color=C[a])
    ax.axhline(0.6167, ls="--", color="gray", label="Max-Quality")
    ax.set_xlabel("episode"); ax.set_ylabel("Dev SR"); ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(out / "dev_sr.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    for a in algos:
        if d[a] is not None:
            ax.plot(d[a].episode, d[a].d_sum_ms, marker="o", ms=3, label=a, color=C[a])
    ax.axhline(9482.0, ls="--", color="gray", label="Max-Quality")
    ax.set_xlabel("episode"); ax.set_ylabel("Dev D_sum (ms)"); ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(out / "dev_d_sum.png"); plt.close(fig)

    rows = []
    for a in algos:
        x = h[a]
        if x is None:
            continue
        dd = d[a]
        rows.append({"algo": a, "episodes": int(x.episode.max()),
                     "steps": int(x.total_steps.iloc[-1]),
                     "updates": int(x.updates.iloc[-1]),
                     "reward_last200": float(x.reward.tail(200).mean()),
                     "sr_train_last200": float(x.success.tail(200).mean()),
                     "d_sum_last200": float(x.d_sum_ms.tail(200).mean()),
                     "energy_last200": float(x.energy_j.tail(200).mean()),
                     "premature": int((x.failure == "premature_stop").sum()),
                     "oob": int((x.failure == "out_of_bounds").sum()),
                     "step_cap": int((x.failure == "step_cap").sum()),
                     "dev_sr_last": float(dd.sr.iloc[-1]) if dd is not None and len(dd) else None,
                     "dev_sr_best": float(dd.sr.max()) if dd is not None and len(dd) else None,
                     "dev_d_sum_last": float(dd.d_sum_ms.iloc[-1]) if dd is not None and len(dd) else None,
                     "dev_score_last": float(dd.score.iloc[-1]) if dd is not None and len(dd) else None,
                     "nan_inf": int(x.reward.isna().sum())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
