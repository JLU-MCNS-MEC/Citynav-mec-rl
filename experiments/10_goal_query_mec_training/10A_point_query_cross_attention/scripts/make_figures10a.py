#!/usr/bin/env python3
"""10A figures: error CDF, metric bars, distance/length breakdowns,
example localisations, attention maps, score calibration.

Reads results/eval_states.csv.gz + eval_metrics.csv; needs CITYNAV_DATA.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common10a  # noqa: E402
from common10a import FIG, METHODS, RES, require_data_root  # noqa: E402

from model10a import load_matcher  # noqa: E402

sys.path.insert(0, str(common10a.REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(common10a.REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
import matching  # noqa: E402
from continuous_belief import subpixel_match  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"c1": "#2c7fb8", "pq": "#d95f0e", "ca": "#31a354"}
NAMES = {"c1": "C1 (fixed centre, 09C-1)", "pq": "PQ (point query)",
         "ca": "CA (cross-attention)"}


def load_states() -> list[dict]:
    with gzip.open(RES / "eval_states.csv.gz", "rt", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # join pose/target fields from the 09A manifest (keyed by episode_id:step)
    with common10a.MANIFEST_09A.open(encoding="utf-8") as fh:
        manifest = {f"{r['episode_id']}:{r['step']}": r for r in csv.DictReader(fh)}
    for r in rows:
        for k in list(r):
            if k.startswith(("c1_", "pq_", "ca_")) or k in ("true_col", "true_row"):
                r[k] = float(r[k])
        r["target_in_view"] = int(r["target_in_view"])
        r["valid_fraction"] = float(r["valid_fraction"])
        r["true_dist_m"] = float(r["true_dist_m"])
        m = manifest.get(r["state_id"], {})
        r["x"] = float(m.get("x", 0.0))
        r["y"] = float(m.get("y", 0.0))
        r["yaw"] = float(m.get("yaw", 0.0))
        r["target_x"] = float(m.get("target_x", 0.0))
        r["target_y"] = float(m.get("target_y", 0.0))
    return rows


def fig_cdf(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, split in zip(axes, ("val_seen", "val_unseen")):
        sub = [r for r in rows if r["split"] == split]
        for m in METHODS:
            err = np.array([r[f"{m}_err_m"] for r in sub
                            if r["target_in_view"] and np.isfinite(r[f"{m}_err_m"])])
            xs = np.linspace(0, 50, 501)
            ys = [np.mean(err <= x) for x in xs]
            ax.plot(xs, ys, color=COLORS[m], label=NAMES[m], lw=2)
        ax.set_title(f"{split} (in-view)")
        ax.set_xlabel("localisation error (m)")
        ax.set_xlim(0, 50)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cumulative fraction")
    axes[0].legend(loc="lower right")
    fig.suptitle("10A: localisation error CDF (offline, in-view states)")
    fig.tight_layout()
    fig.savefig(FIG / "10a_error_cdf.png", dpi=150)
    plt.close(fig)


def fig_bars(rows):
    metrics = {"mae_m": "MAE (m)", "median_m": "Median (m)", "p90_m": "P90 (m)",
               "prec_5m": "Prec@5m", "prec_10m": "Prec@10m", "prec_20m": "Prec@20m"}
    thr = {"prec_5m": 5.0, "prec_10m": 10.0, "prec_20m": 20.0}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (key, label) in zip(axes.flat, metrics.items()):
        x = np.arange(3)
        for split, off, hatch in (("val_seen", -0.21, ""), ("val_unseen", 0.21, "//")):
            vals = []
            for m in METHODS:
                err = np.array([r[f"{m}_err_m"] for r in rows
                                if r["split"] == split and r["target_in_view"]
                                and np.isfinite(r[f"{m}_err_m"])])
                if key in thr:
                    vals.append(np.mean(err <= thr[key]))
                elif key == "mae_m":
                    vals.append(np.mean(err))
                elif key == "median_m":
                    vals.append(np.median(err))
                else:
                    vals.append(np.percentile(err, 90))
            ax.bar(x + off, vals, width=0.38, color=[COLORS[m] for m in METHODS],
                   label=split, hatch=hatch, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in METHODS])
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend()
    fig.suptitle("10A: offline localisation metrics (in-view states)")
    fig.tight_layout()
    fig.savefig(FIG / "10a_metrics_bars.png", dpi=150)
    plt.close(fig)


def fig_breakdown(rows):
    bins = ["0-20", "20-40", "40-60", "60-80", ">80"]
    groups = ["short", "medium", "long"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for m in METHODS:
        ys = []
        for b in bins:
            err = np.array([r[f"{m}_err_m"] for r in rows
                            if r["distance_bin"] == b and r["target_in_view"]
                            and np.isfinite(r[f"{m}_err_m"])])
            ys.append(np.mean(err) if len(err) else np.nan)
        axes[0].plot(range(len(bins)), ys, "o-", color=COLORS[m], label=NAMES[m])
    axes[0].set_xticks(range(len(bins)))
    axes[0].set_xticklabels(bins)
    axes[0].set_xlabel("true target distance bin (m)")
    axes[0].set_ylabel("MAE (m)")
    axes[0].set_title("MAE by distance bin (in-view)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    for m in METHODS:
        ys = []
        for g in groups:
            err = np.array([r[f"{m}_err_m"] for r in rows
                            if r["length_group"] == g and r["target_in_view"]
                            and np.isfinite(r[f"{m}_err_m"])])
            ys.append(np.mean(err) if len(err) else np.nan)
        axes[1].plot(range(len(groups)), ys, "o-", color=COLORS[m], label=NAMES[m])
    axes[1].set_xticks(range(len(groups)))
    axes[1].set_xticklabels(groups)
    axes[1].set_xlabel("trajectory length group")
    axes[1].set_ylabel("MAE (m)")
    axes[1].set_title("MAE by length group (in-view)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("10A: error breakdown by distance / length")
    fig.tight_layout()
    fig.savefig(FIG / "10a_breakdown.png", dpi=150)
    plt.close(fig)


def fig_score_calibration(rows, tau):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, m in zip(axes, METHODS):
        sub = [r for r in rows if r["target_in_view"]]
        err = np.array([r[f"{m}_err_m"] for r in sub])
        sc = np.array([r[f"{m}_score"] for r in sub])
        ok = np.isfinite(err)
        ax.scatter(sc[ok], err[ok], s=2, alpha=0.15, color=COLORS[m])
        ax.axhline(10, color="gray", ls="--", lw=1)
        ax.axvline(tau[m], color="red", ls="--", lw=1, label=f"tau={tau[m]:.2f}")
        ax.set_xlabel("confidence score")
        ax.set_ylabel("error (m)")
        ax.set_title(NAMES[m])
        ax.set_ylim(0, 60)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("10A: score vs error (in-view); gate tau")
    fig.tight_layout()
    fig.savefig(FIG / "10a_score_calibration.png", dpi=150)
    plt.close(fig)


def fig_examples(rows, rasters, encoder, models, device):
    rng = np.random.default_rng(42)
    cand = [r for r in rows if r["split"] == "val_unseen"
            and r["target_in_view"] and 0 < r["true_dist_m"] < 60]
    picks = rng.choice(len(cand), size=3, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12))
    goals: dict = {}
    for col, idx in enumerate(picks):
        r = cand[idx]
        obs = matching.prepare_observation(rasters.get(r["map"]), r["x"], r["y"],
                                           r["yaw"])
        k = (r["map"], round(float(r["target_x"]), 2), round(float(r["target_y"]), 2))
        if k not in goals:
            goals[k] = matching.prepare_goal(rasters.get(r["map"]),
                                             r["target_x"], r["target_y"])
        of = encoder.features(obs[None], ("layer2",))["layer2"]
        gf = encoder.features(goals[k][None], ("layer2",))["layer2"]
        of_t = torch.from_numpy(of).to(device)
        gf_t = torch.from_numpy(gf).to(device)
        preds = {}
        for m in ("pq", "ca"):
            rr = models[m](of_t, gf_t)
            preds[m] = (float(rr["col"][0]), float(rr["row"][0]),
                        float(rr["score"][0]))
        mm = subpixel_match(of[0], gf[0], 2)
        preds["c1"] = (mm["col"], mm["row"], mm["score"])
        for row, m in enumerate(METHODS):
            ax = axes[row, col]
            ax.imshow(obs)
            ax.plot(r["true_col"], r["true_row"], "gx", ms=12, mew=3,
                    label=f"true ({r['true_dist_m']:.0f} m)")
            ax.plot(preds[m][0], preds[m][1], "r+", ms=12, mew=3,
                    label=f"pred err={r[f'{m}_err_m']:.1f} m")
            ax.set_title(f"{m.upper()} | score {preds[m][2]:.2f}", fontsize=10)
            ax.set_axis_off()
        if col == 0:
            axes[0, 0].legend(loc="upper right", fontsize=7)
        axes[2, col].set_xlabel(r["episode_id"])
    fig.suptitle("10A: example localisations (val_unseen, in-view)")
    fig.tight_layout()
    fig.savefig(FIG / "10a_examples.png", dpi=150)
    plt.close(fig)
    return [(r["episode_id"], r["step"], r["map"]) for r in
            [cand[i] for i in picks]]


def fig_attention(rows, rasters, encoder, models, device):
    from model10a import sinusoidal_pos_2d
    rng = np.random.default_rng(7)
    cand = [r for r in rows if r["split"] == "val_seen" and r["target_in_view"]
            and 0 < r["true_dist_m"] < 40]
    idx = int(rng.choice(len(cand)))
    r = cand[idx]
    obs = matching.prepare_observation(rasters.get(r["map"]), r["x"], r["y"],
                                       r["yaw"])
    k = (r["map"], round(float(r["target_x"]), 2), round(float(r["target_y"]), 2))
    goal = matching.prepare_goal(rasters.get(r["map"]), r["target_x"], r["target_y"])
    of = encoder.features(obs[None], ("layer2",))["layer2"]
    gf = encoder.features(goal[None], ("layer2",))["layer2"]
    of_t = torch.from_numpy(of).to(device)
    gf_t = torch.from_numpy(gf).to(device)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(obs)
    axes[0].plot(r["true_col"], r["true_row"], "gx", ms=12, mew=3)
    axes[0].set_title(f"obs | true target")
    axes[0].set_axis_off()
    for ax, m in zip(axes[1:], ("pq", "ca")):
        rr = models[m](of_t, gf_t)
        attn = rr["attn"] if m == "pq" else rr["attn_cross"]
        # winning proposal
        out = rr["out"][0]
        scores = out[:, 2].sigmoid()
        w = int(scores.argmax())
        a = attn[0, w].detach().cpu().numpy().reshape(28, 28)
        hm = np.repeat(np.repeat(a, 8, axis=0), 8, axis=1)
        ax.imshow(obs)
        ax.imshow(hm, cmap="jet", alpha=0.45)
        ax.plot(float(rr["col"][0]), float(rr["row"][0]), "r+", ms=14, mew=4)
        if m == "pq":
            anc = rr["anchor"].detach().cpu().numpy()
            ax.plot(anc[:, 0] * 111.5 + 111.5, anc[:, 1] * 111.5 + 111.5,
                    "yo", ms=5, label="query anchors")
            ax.legend(fontsize=8)
        ax.set_title(f"{m.upper()} winning-proposal attention "
                     f"(err {r[f'{m}_err_m']:.1f} m)")
        ax.set_axis_off()
    fig.suptitle(f"10A: attention visualisation | {r['episode_id']} step {r['step']}")
    fig.tight_layout()
    fig.savefig(FIG / "10a_attention.png", dpi=150)
    plt.close(fig)
    return r["episode_id"]


def main() -> int:
    require_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from dataset_v2 import RasterStore

    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_states()
    tau = {}
    for m in METHODS:
        if m == "c1":
            tau[m] = json.loads((common10a.REPO
                                 / "experiments/09_online_visual_goal_matching"
                                 / "results/09a_calibration.json")
                                .read_text(encoding="utf-8"))["tau"]["layer2"]
        else:
            tau[m] = json.loads((RES / m / "calibration.json")
                                .read_text(encoding="utf-8"))["tau"]
    print(f"states {len(rows)} | tau {tau}", flush=True)
    fig_cdf(rows)
    fig_bars(rows)
    fig_breakdown(rows)
    fig_score_calibration(rows, tau)
    encoder = matching.Encoder(args.device)
    rasters = RasterStore()
    models = {m: load_matcher(m, str(RES / m / "checkpoints" / "best.pt"),
                              args.device) for m in ("pq", "ca")}
    picked = fig_examples(rows, rasters, encoder, models, args.device)
    att = fig_attention(rows, rasters, encoder, models, args.device)
    (FIG / "figure_cases.json").write_text(json.dumps(
        {"examples": picked, "attention_case": att}, indent=2) + "\n",
        encoding="utf-8")
    print("figures written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
