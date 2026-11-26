#!/usr/bin/env python3
"""Stage 17-D-P0 figures: MA20 training trends, Dev curves, model usage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"dqn": "#1f77b4", "ddpg": "#2ca02c", "td3": "#9467bd",
          "sac": "#d62728", "ppo": "#8c564b"}


def load_hist(d: Path):
    p = d / "history.jsonl"
    return pd.DataFrame([json.loads(x) for x in open(p)]) if p.exists() else None


def load_dev(d: Path):
    p = d / "dev.jsonl"
    return pd.DataFrame([json.loads(x) for x in open(p)]) if p.exists() else None


def plot_series(series_map, ylabel, fname, out, ma=20):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    for name, (df, col) in series_map.items():
        if df is None or col not in df:
            continue
        ax.plot(df["episode"], df[col].rolling(ma, min_periods=1).mean(),
                label=name, color=COLORS.get(name), lw=1.6)
    ax.set_xlabel("episode"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / fname); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="stage17d/p0")
    ap.add_argument("--out", default="stage17d/p0/figures")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    algos = ["dqn", "ddpg", "td3", "sac", "ppo"]
    hist = {a: load_hist(root / a) for a in algos}
    dev = {a: load_dev(root / a) for a in algos}

    plot_series({a: (hist[a], "reward") for a in algos},
                "episode reward (MA20)", "reward_ma20.png", out)
    plot_series({a: (hist[a], "d_sum_ms") for a in algos},
                "cumulative D_coll (ms, MA20)", "d_sum_ma20.png", out)
    plot_series({a: (hist[a], "energy_j") for a in algos},
                "cumulative energy (J, MA20)", "energy_ma20.png", out)
    plot_series({a: (dev[a], "sr") for a in algos},
                "Dev SR", "dev_sr.png", out, ma=1)
    plot_series({a: (dev[a], "d_sum_ms") for a in algos},
                "Dev cumulative D_coll (ms)", "dev_d_sum.png", out, ma=1)
    plot_series({a: (hist[a], "loss_q") for a in algos},
                "critic loss (MA20)", "loss_critic_ma20.png", out)
    plot_series({a: (hist[a], "loss_pi") for a in algos},
                "actor loss (MA20)", "loss_actor_ma20.png", out)
    plot_series({a: (hist[a], "grad_pi") for a in algos},
                "actor grad norm (MA20)", "grad_pi_ma20.png", out)

    # SAC alpha / entropy
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    if hist["sac"] is not None and "alpha" in hist["sac"]:
        h = hist["sac"]
        ax.plot(h["episode"], h["alpha"].rolling(20, min_periods=1).mean(),
                label="alpha", color=COLORS["sac"])
        ax2 = ax.twinx()
        ax2.plot(h["episode"], h["entropy"].rolling(20, min_periods=1).mean(),
                 label="policy entropy", color="gray", ls="--")
        ax2.set_ylabel("entropy")
    ax.set_xlabel("episode"); ax.set_ylabel("alpha")
    ax.grid(alpha=0.3); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "sac_alpha_entropy.png"); plt.close(fig)

    # model selection frequency (last 20 episodes)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    width = 0.15
    for i, a in enumerate(algos):
        if hist[a] is None or "model_selection" not in hist[a]:
            continue
        freq = np.sum(np.vstack(hist[a]["model_selection"].tail(20).values), axis=0)
        freq = freq / max(freq.sum(), 1)
        ax.bar(np.arange(10) + i * width, freq, width, label=a, color=COLORS[a])
    ax.set_xticks(np.arange(10) + 2 * width)
    ax.set_xticklabels([f"G{i+1}" for i in range(10)])
    ax.set_ylabel("share of services (last 20 ep)"); ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=5)
    fig.tight_layout(); fig.savefig(out / "mec_selection.png"); plt.close(fig)

    # summary table
    rows = []
    for a in algos:
        h, d = hist[a], dev[a]
        if h is None:
            continue
        rows.append({
            "algo": a, "episodes": int(h.episode.max()),
            "steps": int(h.total_steps.iloc[-1]),
            "updates": int(h.updates.iloc[-1]) if "updates" in h else None,
            "reward_mean": float(h.reward.mean()),
            "success_rate": float(h.success.mean()),
            "premature": int((h.failure == "premature_stop").sum()),
            "oob": int((h.failure == "out_of_bounds").sum()),
            "collision": int((h.failure == "collision").sum()),
            "step_cap": int((h.failure == "step_cap").sum()),
            "d_sum_ms_mean": float(h.d_sum_ms.mean()),
            "energy_j_mean": float(h.energy_j.mean()),
            "dev_sr_last": float(d.sr.iloc[-1]) if d is not None and len(d) else None,
            "dev_d_sum_last": float(d.d_sum_ms.iloc[-1]) if d is not None and len(d) else None,
            "nan_or_inf": int(h[["reward"]].isna().sum().sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "p0_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
