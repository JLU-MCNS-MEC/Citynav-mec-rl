#!/usr/bin/env python3
"""Stage 17-D-P1 figures: PPO/SAC training, diagnostics, MEC usage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HOME = "stage17d/p1"
C = {"ppo": "#8c564b", "sac": "#d62728"}
MAXQ = {"sr": 0.6166666666666667, "d_sum_ms": 9482.015727013848}


def load(p: Path):
    return pd.DataFrame([json.loads(x) for x in open(p)]) if p.exists() else None


def line(ax, df, col, label, color, ma=20):
    if df is None or col not in df or df[col].isna().all():
        return
    ax.plot(df.episode, df[col].rolling(ma, min_periods=1).mean(),
            label=label, color=color, lw=1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HOME)
    ap.add_argument("--out", default=f"{HOME}/figures")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    h = {a: load(root / a / "history.jsonl") for a in ("ppo", "sac")}
    d = {a: load(root / a / "dev.jsonl") for a in ("ppo", "sac")}

    for col, ylab, fn in [("reward", "episode reward (MA20)", "train_reward"),
                          ("d_sum_ms", "train D_sum ms (MA20)", "train_d_sum"),
                          ("energy_j", "train energy J (MA20)", "train_energy"),
                          ("steps", "train steps (MA20)", "train_steps"),
                          ("loss_q", "critic loss (MA20)", "loss_critic"),
                          ("loss_pi", "actor loss (MA20)", "loss_actor"),
                          ("grad_pi", "actor grad norm (MA20)", "grad_actor")]:
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
        for a in h:
            line(ax, h[a], col, a, C[a])
        ax.set_xlabel("episode"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
        ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / f"{fn}.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    for a in d:
        if d[a] is not None:
            ax.plot(d[a].episode, d[a].sr, marker="o", label=a, color=C[a])
    ax.axhline(MAXQ["sr"], ls="--", color="gray", label="Max-Quality")
    ax.set_xlabel("episode"); ax.set_ylabel("Dev SR"); ax.grid(alpha=0.3)
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "dev_sr.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    for a in d:
        if d[a] is not None:
            ax.plot(d[a].episode, d[a].d_sum_ms, marker="o", label=a, color=C[a])
    ax.axhline(MAXQ["d_sum_ms"], ls="--", color="gray", label="Max-Quality")
    ax.set_xlabel("episode"); ax.set_ylabel("Dev D_sum (ms)"); ax.grid(alpha=0.3)
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "dev_d_sum.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    for a in d:
        if d[a] is not None and "score" in d[a]:
            ax.plot(d[a].episode, d[a].score, marker="o", label=a, color=C[a])
    ax.set_xlabel("episode"); ax.set_ylabel("Dev score J"); ax.grid(alpha=0.3)
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "dev_score.png"); plt.close(fig)

    # PPO diagnostics
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    line(ax, h["ppo"], "policy_entropy", "policy entropy", C["ppo"])
    line(ax, h["ppo"], "approx_kl", "approx KL", "black")
    ax.set_xlabel("episode"); ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "ppo_diagnostics.png"); plt.close(fig)

    # SAC diagnostics
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    line(ax, h["sac"], "alpha", "alpha", C["sac"])
    line(ax, h["sac"], "entropy", "policy entropy", "black")
    line(ax, h["sac"], "sac_target_entropy", "target entropy", "gray")
    ax.set_xlabel("episode"); ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "sac_alpha_entropy.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    line(ax, h["sac"], "mean_q_min", "mean Q_min", C["sac"])
    line(ax, h["sac"], "mean_bellman_target", "mean Bellman target", "black")
    ax.set_xlabel("episode"); ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "sac_q_target.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    line(ax, h["sac"], "mean_env_reward", "env reward", C["sac"])
    line(ax, h["sac"], "mean_entropy_bonus", "alpha*H bonus", "gray")
    ax.set_xlabel("episode"); ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "sac_entropy_bonus.png"); plt.close(fig)

    # MEC selection
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    for i, a in enumerate(("ppo", "sac")):
        if h[a] is None:
            continue
        freq = np.sum(np.vstack(h[a].model_selection.tail(50).values), axis=0)
        ax.bar(np.arange(10) + i * 0.4, freq / max(freq.sum(), 1), 0.4,
               label=a, color=C[a])
    ax.set_xticks(np.arange(10) + 0.2)
    ax.set_xticklabels([f"G{i+1}" for i in range(10)])
    ax.set_ylabel("share (last 50 ep)"); ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "mec_selection.png"); plt.close(fig)

    rows = []
    for a in ("ppo", "sac"):
        x = h[a]
        if x is None:
            continue
        dd = d[a]
        rows.append({"algo": a, "episodes": int(x.episode.max()),
                     "steps": int(x.total_steps.iloc[-1]),
                     "updates": int(x.updates.iloc[-1]),
                     "reward_mean": float(x.reward.mean()),
                     "sr_train": float(x.success.mean()),
                     "premature": int((x.failure == "premature_stop").sum()),
                     "oob": int((x.failure == "out_of_bounds").sum()),
                     "step_cap": int((x.failure == "step_cap").sum()),
                     "d_sum_mean": float(x.d_sum_ms.mean()),
                     "energy_mean": float(x.energy_j.mean()),
                     "dev_sr_last": float(dd.sr.iloc[-1]) if dd is not None else None,
                     "dev_d_sum_last": float(dd.d_sum_ms.iloc[-1]) if dd is not None else None,
                     "dev_score_last": float(dd.score.iloc[-1]) if dd is not None else None,
                     "nan_inf": int(x.reward.isna().sum())})
    pd.DataFrame(rows).to_csv(out / "p1_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
