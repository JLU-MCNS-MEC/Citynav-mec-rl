#!/usr/bin/env python3
"""Figures for the direct target-directed teacher (Stage 03B)."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE.parent / "04_visual_goal_global_map_pipeline/scripts"))

from obstacles import RGBD  # noqa: E402

DATA = Path("/home/20261024_1/Workspace/DATA/sbfnav-data/original_citynav")
RESULTS = HERE / "results"
FIG = HERE / "figures"


def read_rows(split: str) -> list[dict]:
    with (RESULTS / f"teacher_{split}.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fig_steps(rows: list[dict], groups: dict) -> None:
    steps = np.asarray([int(r["steps"]) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(steps, bins=np.arange(0, steps.max() + 5, 5), color="tab:blue", alpha=0.85)
    for name, x in (("short", groups["short_max"]), ("medium", groups["medium_max"])):
        ax.axvline(x, color="tab:red", linestyle="--", lw=1)
        ax.text(x + 1, ax.get_ylim()[1] * 0.9, f"{name} <= {x}", fontsize=8, color="tab:red")
    ax.set_xlabel("teacher actions per task")
    ax.set_ylabel("tasks")
    ax.set_title(f"Direct teacher action counts (Train-Seen, n={len(steps)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "steps_hist.png", dpi=150)
    plt.close(fig)


def fig_paths(rows: list[dict]) -> None:
    human = np.asarray([float(r["human_path_len_m"]) for r in rows])
    direct = np.asarray([float(r["path_len_m"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(human, direct, s=2, alpha=0.15, color="tab:purple")
    lim = max(human.max(), direct.max()) * 1.02
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("human trajectory length (m)")
    ax.set_ylabel("direct teacher path length (m)")
    ax.set_title("Human path vs direct target path")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "path_length_compare.png", dpi=150)
    plt.close(fig)


def fig_maps(rows: list[dict]) -> None:
    counts = Counter(r["map"] for r in rows)
    names = sorted(counts)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(names)), [counts[n] for n in names], color="tab:green")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=75, fontsize=7, ha="right")
    ax.set_ylabel("tasks")
    ax.set_title("Train-Seen task coverage by map")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "tasks_by_map.png", dpi=150)
    plt.close(fig)


def fig_actions_by_phase() -> None:
    z = np.load(RESULTS / "teacher_train_seen.npz")
    off = z["offsets"]
    acts = z["actions"]
    names = ["Stop", "Forward", "TurnLeft", "TurnRight"]
    bins = defaultdict(Counter)
    for i in range(0, len(off) - 1, 7):       # sample every 7th episode
        a = acts[off[i]:off[i + 1]]
        n = len(a)
        for k, act in enumerate(a):
            phase = 0 if k < n / 3 else (1 if k < 2 * n / 3 else 2)
            bins[phase][names[int(act)]] += 1
    phases = ["first third", "middle third", "last third"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(3)
    colors = {"Stop": "tab:red", "Forward": "tab:blue",
              "TurnLeft": "tab:orange", "TurnRight": "tab:green"}
    for name in names:
        vals = np.asarray([bins[p][name] for p in range(3)], float)
        vals = vals / vals.sum() if vals.sum() else vals
        ax.bar(phases, vals, bottom=bottom, label=name, color=colors[name])
        bottom += vals
    ax.set_ylabel("action share")
    ax.set_title("Action mix across task progress (Train-Seen sample)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "actions_by_phase.png", dpi=150)
    plt.close(fig)


def fig_actions_by_distance() -> None:
    """Action mix as a function of the remaining distance to the target."""
    z = np.load(RESULTS / "teacher_train_seen.npz")
    off, acts, states, targets = z["offsets"], z["actions"], z["states"], z["targets"]
    buckets = [(0, 20), (20, 30), (30, 50), (50, 100), (100, 200), (200, 10 ** 9)]
    labels = ["<=20", "20-30", "30-50", "50-100", "100-200", ">200"]
    counts = np.zeros((len(buckets), 4))
    for i in range(0, len(off) - 1, 3):
        a = acts[off[i]:off[i + 1]]
        s = states[off[i]:off[i + 1]].astype(np.float64)
        t = targets[i].astype(np.float64)
        d = np.linalg.norm(s[:, :2] - t, axis=1)
        for k, act in enumerate(a):
            for b, (lo, hi) in enumerate(buckets):
                if (d[k] <= hi) and (b == 0 or d[k] > lo):
                    counts[b, int(act)] += 1
                    break
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(buckets))
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:green"]
    for j, name in enumerate(["Stop", "Forward", "TurnLeft", "TurnRight"]):
        ax.bar(labels, share[:, j], bottom=bottom, label=name, color=colors[j])
        bottom += share[:, j]
    ax.set_xlabel("distance to target at the decision state (m)")
    ax.set_ylabel("action share")
    ax.set_title("Teacher action mix by remaining distance (Train-Seen)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "actions_by_distance.png", dpi=150)
    plt.close(fig)


def fig_chatter_compare() -> None:
    path = RESULTS / "analysis/chatter_compare.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    names = [r["variant"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, title in zip(
            axes,
            ("steps_mean", "turns_per_episode", "turn_share"),
            ("teacher actions per task", "turns per task", "turn share")):
        vals = [float(r[key]) for r in rows]
        ax.bar(range(len(rows)), vals, color=["tab:red"] + ["tab:blue"] * (len(rows) - 1))
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=20, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("15 deg rule (naive) vs deadband variants, Train-Seen", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "chatter_compare.png", dpi=150)
    plt.close(fig)


def fig_trajectories(rows: list[dict], records: list[dict]) -> None:
    z = np.load(RESULTS / "teacher_train_seen.npz")
    keys = list(z["keys"])
    off = z["offsets"]
    states = z["states"]
    by_index = {int(r["index"]): r for r in rows}
    picks = []
    counts = Counter(r["map"] for r in rows)
    for name, _ in counts.most_common():
        if any(r["map"] == name for r in rows) and name not in [p[0] for p in picks]:
            cand = [r for r in rows if r["map"] == name]
            # a mid-length task, to make both paths visible
            cand.sort(key=lambda r: abs(int(r["steps"]) - 55))
            picks.append((name, cand[0]))
        if len(picks) == 4:
            break
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, (name, row) in zip(axes.ravel(), picks):
        idx = int(row["index"])
        img = cv2.cvtColor(cv2.imread(str(RGBD / f"{name}.png")), cv2.COLOR_BGR2RGB)
        left, bottom, right, top = 0, 0, img.shape[1] * 0.1, img.shape[0] * 0.1
        # bounds are not stored in the RGB; recover from the elevation metadata
        meta = json.loads((HERE / f"cache/obstacles/{name}.json").read_text())
        left, bottom, right, top = meta["bounds"]
        ax.imshow(img, extent=(left, right, bottom, top), origin="upper")
        human = np.asarray(records[idx]["trajectory"], float)[:, :2]
        i = keys.index(str(idx))
        s = states[off[i]:off[i + 1]]
        ax.plot(human[:, 0], human[:, 1], color="tab:blue", lw=1.0, label="human path")
        ax.plot(s[:, 0], s[:, 1], color="tab:orange", lw=1.2, linestyle="--",
                label="direct teacher")
        ax.scatter(*s[0, :2], c="lime", edgecolor="black", s=35, label="start")
        ax.scatter(*z["targets"][i], c="red", marker="*", s=150, edgecolor="black",
                   label="target")
        ax.set_title(f"{name}  index={idx}  steps={row['steps']}", fontsize=8)
        ax.legend(fontsize=6, loc="lower left")
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "trajectory_examples.png", dpi=100)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(exist_ok=True)
    groups = json.loads((RESULTS / "summary.json").read_text())["length_groups"]
    rows = read_rows("train_seen")
    records = json.loads((DATA / "citynav_train_seen.json").read_text())
    fig_steps(rows, groups)
    fig_paths(rows)
    fig_maps(rows)
    fig_actions_by_phase()
    fig_actions_by_distance()
    fig_chatter_compare()
    fig_trajectories(rows, records)
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
