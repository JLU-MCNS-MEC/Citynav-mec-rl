#!/usr/bin/env python3
"""Stage 16-P5: reward and cumulative-E2E MA100 curves for the DRL agents."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP16 = HERE.parent
RUNS = EXP16 / "p4_stage12" / "runs"
FIG = EXP16 / "figures"
FIG.mkdir(exist_ok=True)
ALGOS = ["ppo", "dqn", "td3", "sac", "ddpg"]
COLORS = {"ppo": "#4c78a8", "dqn": "#f58518", "td3": "#54a24b", "sac": "#b279a2", "ddpg": "#e45756"}
LABEL = {"ppo": "PPO", "dqn": "DQN", "td3": "TD3 (discrete)", "sac": "Discrete SAC", "ddpg": "DDPG (discrete)"}


def load(algo):
    p = RUNS / algo / "seed42" / "history.jsonl"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("episode")
    df["reward_ma100"] = df["reward"].rolling(100, min_periods=1).mean()
    df["cum_e2e_ma100"] = df["cumulative_e2e_ms"].rolling(100, min_periods=1).mean()
    df["steps_ma100"] = df["steps"].rolling(100, min_periods=1).mean()
    df["sr_ma100"] = df["success"].rolling(100, min_periods=1).mean()
    return df


def main():
    data = {}
    for a in ALGOS:
        d = load(a)
        if d is not None:
            data[a] = d
    if not data:
        print("no histories yet")
        return
    out_csv = HERE / "stage12_ma100_curves.csv"
    frames = []
    for a, d in data.items():
        dd = d[["episode", "reward_ma100", "cum_e2e_ma100", "steps_ma100", "sr_ma100"]].copy()
        dd["algorithm"] = a
        frames.append(dd)
    pd.concat(frames).to_csv(out_csv, index=False)

    for field, ylabel, fname, logy in [
            ("reward_ma100", "episode reward (MA100)", "stage12_drl_reward_ma100.png", False),
            ("cum_e2e_ma100", "cumulative E2E (ms, MA100)", "stage12_drl_cum_e2e_ma100.png", False),
            ("sr_ma100", "success rate (MA100)", "stage12_drl_sr_ma100.png", False),
            ("steps_ma100", "steps (MA100)", "stage12_drl_steps_ma100.png", False)]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for a, d in data.items():
            ax.plot(d["episode"], d[field], lw=1.2, color=COLORS[a], label=LABEL[a])
        ax.set_xlabel("episode"); ax.set_ylabel(ylabel)
        ax.set_title(f"Stage 16 · Stage-12 reward · {ylabel}")
        ax.grid(alpha=.3); ax.legend()
        if logy:
            ax.set_yscale("log")
        fig.tight_layout(); fig.savefig(FIG / fname, dpi=140); plt.close(fig)
    print(f"wrote {out_csv} and {len(data)} curves under {FIG}")


if __name__ == "__main__":
    main()
