#!/usr/bin/env python3
"""09C figures: SR comparison, localisation error, gating and case panels."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "experiments/05_transformer_m_training/scripts"))
from dataset_v2 import CandidateStore, RasterStore  # noqa: E402

METHODS = ["transformer", "top1_static", "top5_static", "dynamic_24m",
           "continuous_c1", "continuous_c2"]
LABEL = {"transformer": "Transformer-M", "top1_static": "Static Top-1",
         "top5_static": "Static Top-5", "dynamic_24m": "Dynamic 24 m",
         "continuous_c1": "Continuous C1", "continuous_c2": "Continuous C2"}
COLOR = {"transformer": "#b23b3b", "top1_static": "#2f5d8a", "top5_static": "#4f9d69",
         "dynamic_24m": "#7a3fa0", "continuous_c1": "#c58a2e", "continuous_c2": "#1f7a7a"}


def main() -> int:
    comp = json.loads((RES / "09c_comparison.json").read_text(encoding="utf-8"))
    FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    ax = axes[0]
    m = comp["methods"]
    vals = [m[k]["sr"] for k in METHODS]
    los = [m[k]["sr"] - m[k]["sr_ci95"][0] for k in METHODS]
    his = [m[k]["sr_ci95"][1] - m[k]["sr"] for k in METHODS]
    ax.bar(range(len(METHODS)), vals, yerr=[los, his], capsize=3,
           color=[COLOR[k] for k in METHODS])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v * 100:.1f}%", ha="center", fontsize=7)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LABEL[k] for k in METHODS], rotation=25, fontsize=7)
    ax.set_ylim(0, 0.78); ax.set_ylabel("SR", fontsize=8)
    ax.set_title("SR on the same 5,324 tasks", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for mode in ("c1", "c2"):
        folder = RES / f"09c_{mode}" / "episodes"
        errs = []
        for p in folder.glob("*.json"):
            rec = json.loads(p.read_text(encoding="utf-8"))
            for s in rec["trajectory"]:
                if s.get("estimate_err_to_true_m") is not None:
                    d = s["true_dist_m"]
                    # visible states only (recompute roughly from stored fields)
                    if d <= 48:
                        errs.append(s["estimate_err_to_true_m"])
        errs = np.sort(np.array(errs))
        if len(errs):
            ax.plot(errs, np.arange(1, len(errs) + 1) / len(errs),
                    label=f"{LABEL['continuous_' + mode]} (n={len(errs)})",
                    color=COLOR["continuous_" + mode])
    ax.axvline(10, color="k", ls="--", lw=1)
    ax.set_xlim(0, 80); ax.set_xlabel("continuous estimate error (m)", fontsize=8)
    ax.set_ylabel("fraction", fontsize=8)
    ax.set_title("Estimate error CDF (visible states)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    g = comp["gating"]
    xs = np.arange(3)
    w = 0.35
    ax.bar(xs - w / 2, [g["c1"]["accept_precision_10m"], g["c1"]["wrong_accept_rate"],
                        g["c1"]["out_of_view_accept_rate"]], w,
           label="C1 (no observability gate)", color=COLOR["continuous_c1"])
    ax.bar(xs + w / 2, [g["c2"]["accept_precision_10m"], g["c2"]["wrong_accept_rate"],
                        g["c2"]["out_of_view_accept_rate"]], w,
           label="C2 (+ observability gate)", color=COLOR["continuous_c2"])
    ax.set_xticks(xs)
    ax.set_xticklabels(["accepted matches\n<=10 m", "accepted matches\n>20 m",
                        "out-of-view\naccepted"], fontsize=7)
    ax.set_ylim(0, 1); ax.set_title("Evidence gate quality", fontsize=9)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "09c_sr_localisation_gating.png", dpi=150)
    plt.close(fig)

    # case panels: 6 C2 episodes (3 success, 3 failure), trajectory + estimates
    folder = RES / "09c_c2" / "episodes"
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
    picks = [r for r in records if r["success"]][:3] + \
            [r for r in records if not r["success"]][:3]
    rasters = RasterStore()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, rec in zip(axes.ravel(), picks):
        raster = rasters.get(rec["map"])
        target = np.asarray(rec["target"])
        pts = np.asarray([[s["x"], s["y"]] for s in rec["trajectory"]])
        est = np.asarray([s["estimate"] for s in rec["trajectory"]][::max(1, len(pts) // 12)])
        allp = np.vstack([pts, est, target[None]])
        lo = allp.min(axis=0) - 25; hi = allp.max(axis=0) + 25
        c0, r0 = raster.pixel(lo[0], hi[1]); c1, r1 = raster.pixel(hi[0], lo[1])
        c0, r0 = max(int(c0), 0), max(int(r0), 0)
        c1, r1 = min(int(c1), raster.image.shape[1]), min(int(r1), raster.image.shape[0])
        ax.imshow(raster.image[r0:r1, c0:c1],
                  extent=[raster.left + c0 * raster.res, raster.left + c1 * raster.res,
                          raster.top - r1 * raster.res, raster.top - r0 * raster.res],
                  origin="upper")
        ax.plot(pts[:, 0], pts[:, 1], "-o", color="#1f4e79", ms=2, lw=1)
        ax.plot(est[:, 0], est[:, 1], "x", color="#c58a2e", ms=5)
        ax.plot(target[0], target[1], "*", color="#b23b3b", ms=13)
        ax.add_patch(plt.Circle(target, 20, fill=False, color="#b23b3b", ls="--", lw=1))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{rec['episode_id']} | {rec['failure']} | steps {rec['steps']}\n"
                     f"updates {rec['belief_updates']} | min {rec['min_dist']:.1f} m",
                     fontsize=7)
    fig.suptitle("Continuous C2: trajectory (blue), continuous estimates (orange x), "
                 "true target (red star)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "09c_cases.png", dpi=150)
    plt.close(fig)
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
