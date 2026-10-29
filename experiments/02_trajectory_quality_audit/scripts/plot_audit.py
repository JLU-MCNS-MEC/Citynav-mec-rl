#!/usr/bin/env python3
"""Plots for the Stage 02 trajectory audit (reads results/train_metrics.jsonl)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA = Path("/home/20261024_1/Workspace/DATA")
RGBD = DATA / "rgbd"
SPLIT_JSON = DATA / "sbfnav-data/original_citynav/citynav_train_seen.json"
BOUNDS: dict[str, tuple[float, float, float, float]] = {}


def bounds(name: str) -> tuple[float, float, float, float]:
    if name not in BOUNDS:
        import rasterio

        with rasterio.open(RGBD / f"{name}.tif") as ds:
            b = ds.bounds
        BOUNDS[name] = (b.left, b.bottom, b.right, b.top)
    return BOUNDS[name]


def load_rows(results: Path) -> list[dict]:
    """Merge the classified CSV with the per-episode journal (action counts)."""
    import csv

    journal = {
        int(json.loads(line)["index"]): json.loads(line)
        for line in (results / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = []
    for raw in csv.DictReader((results / "train_quality.csv").open(encoding="utf-8")):
        row = dict(raw)
        for key in (
            "index", "path_len_m", "efficiency", "backtrack_ratio", "teacher_actions",
            "duplicate_ratio", "final_target_m", "start_target_m", "min_target_m",
            "turn_total_deg", "max_jump_m", "outside_ratio", "nodata_ratio", "score",
        ):
            row[key] = float(row[key]) if row.get(key) not in (None, "") else 0.0
        row["index"] = int(row["index"])
        row["replay_success"] = row["replay_success"] == "True"
        row["entered_20m"] = row["entered_20m"] == "True"
        row["action_counts"] = journal.get(row["index"], {}).get("action_counts", {})
        row["degenerate"] = False
        rows.append(row)
    return rows


def crop_image(name: str, x0: float, y0: float, x1: float, y1: float):
    left, bottom, right, top = bounds(name)
    img = np.asarray(Image.open(RGBD / f"{name}.png").convert("RGB"))
    h, w = img.shape[:2]
    sx, sy = w / (right - left), h / (top - bottom)
    c0 = max(0, int((x0 - left) * sx)); c1 = min(w, int((x1 - left) * sx))
    r0 = max(0, int((top - y1) * sy)); r1 = min(h, int((top - y0) * sy))
    return img[r0:r1, c0:c1], (left + c0 / sx, left + c1 / sx, top - r1 / sy, top - r0 / sy)


def case_panel(ax, record: dict, row: dict, title: str) -> None:
    name = row["map"]
    xy = np.asarray(record["trajectory"], float)[:, :2]
    target = np.asarray(record["target_positions"][-1], float)[:2]
    marker = np.asarray(record["marker_positions"][-1], float)[:2]
    x0, x1 = xy[:, 0].min() - 40, xy[:, 0].max() + 40
    y0, y1 = xy[:, 1].min() - 40, xy[:, 1].max() + 40
    img, extent = crop_image(name, x0, y0, x1, y1)
    ax.imshow(img, extent=extent, origin="upper")
    ax.plot(xy[:, 0], xy[:, 1], "-", color="tab:blue", lw=1.6, label="human path")
    ax.scatter(*xy[0], c="lime", edgecolor="black", s=40, zorder=5, label="start")
    ax.scatter(*target, c="red", marker="*", s=150, edgecolor="black", zorder=6, label="target")
    ax.scatter(*marker, c="cyan", marker="x", s=60, zorder=6, label="marker")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.tick_params(labelsize=6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--figures", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results)
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    records = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))

    finite = [r for r in rows if not r.get("degenerate")]
    high = sorted([r for r in finite if r.get("quality") == "high"], key=lambda r: -r.get("score", 0))
    medium = [r for r in finite if r.get("quality") == "medium"]
    low = [r for r in finite if r.get("quality") == "low"]

    # 1) case panels
    for tag, cases in (
        ("high", high[:3]),
        ("medium", medium[len(medium) // 2: len(medium) // 2 + 3] if medium else []),
        ("low", low[:3]),
    ):
        if not cases:
            continue
        fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 5))
        axes = np.atleast_1d(axes)
        for ax, row in zip(axes, cases):
            case_panel(ax, records[row["index"]], row,
                       f"{tag}: {row['map']}\nNE={row['final_target_m']:.1f} m  replay={row['replay_success']}")
        fig.tight_layout()
        fig.savefig(args.figures / f"cases_{tag}.png", dpi=140)
        plt.close(fig)

    # 2) cleaning example: highest duplicate ratio
    dup = max(finite, key=lambda r: r.get("duplicate_ratio", 0.0))
    rec = records[dup["index"]]
    raw = np.asarray(rec["trajectory"], float)[:, :2]
    keep = [0]
    for i in range(1, len(raw)):
        if np.linalg.norm(raw[i] - raw[keep[-1]]) >= 0.05:
            keep.append(i)
    cleaned = raw[keep]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pts, title in ((axes[0], raw, f"raw ({len(raw)} pts)"),
                           (axes[1], cleaned, f"cleaned ({len(cleaned)} pts)")):
        x0, x1 = raw[:, 0].min() - 20, raw[:, 0].max() + 20
        y0, y1 = raw[:, 1].min() - 20, raw[:, 1].max() + 20
        img, extent = crop_image(dup["map"], x0, y0, x1, y1)
        ax.imshow(img, extent=extent, origin="upper")
        ax.plot(pts[:, 0], pts[:, 1], "-", color="tab:blue", lw=1.2)
        ax.scatter(*pts[0], c="lime", edgecolor="black", s=35)
        ax.set_title(f"{dup['map']} — {title}")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    fig.suptitle(f"duplicate ratio {dup['duplicate_ratio']:.1%}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.figures / "cleaning_example.png", dpi=140)
    plt.close(fig)

    # 3) metric distributions with thresholds
    thr = summary.get("thresholds", {})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    series = [
        ("path length (m)", [r["path_len_m"] for r in finite], None, True),
        ("efficiency (straight/path)", [r["efficiency"] for r in finite], thr.get("eff_p25"), False),
        ("backtrack ratio", [r["backtrack_ratio"] for r in finite], thr.get("backtrack_p90"), False),
        ("teacher steps", [r["teacher_actions"] for r in finite], None, True),
    ]
    for ax, (title, values, line, logx) in zip(axes.ravel(), series):
        values = np.asarray(values)
        bins = np.logspace(np.log10(max(values.min(), 1e-3)), np.log10(max(values.max(), 1e-2)), 60) if logx else 60
        ax.hist(values, bins=bins, color="tab:blue", alpha=0.8)
        if logx:
            ax.set_xscale("log")
        if line is not None:
            ax.axvline(line, color="red", ls="--", label=f"threshold {line:.3g}")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.figures / "dist_metrics.png", dpi=140)
    plt.close(fig)

    # 4) quality by length bin
    bins = summary.get("quality_by_length_bin", {})
    labels = list(bins)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(labels))
    for q, color in (("high", "tab:green"), ("medium", "tab:orange"), ("low", "tab:red")):
        vals = np.asarray([bins[b][q] for b in labels])
        ax.bar(labels, vals, bottom=bottom, label=q, color=color)
        bottom += vals
    ax.set_title("quality by teacher-step length bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures / "quality_by_bin.png", dpi=140)
    plt.close(fig)

    # 5) map coverage and high-quality fraction
    by_map = summary.get("by_map", {})
    maps = sorted(by_map)
    n = np.asarray([by_map[m]["n"] for m in maps])
    hi = np.asarray([by_map[m]["high"] for m in maps])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(maps))
    ax.bar(x, n, color="lightgray", label="episodes")
    ax.bar(x, hi, color="tab:green", label="high quality")
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_block_", "_") for m in maps], rotation=75, fontsize=7)
    ax.set_title("Train-Seen coverage and high-quality count per map")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures / "coverage_by_map.png", dpi=140)
    plt.close(fig)

    # 6) teacher action distribution: all vs high
    names = ["Stop", "Forward", "TurnLeft", "TurnRight"]
    def counts(rs):
        out = {k: 0 for k in names}
        for r in rs:
            for k in names:
                out[k] += r.get("action_counts", {}).get(k, 0)
        return out
    all_c, hi_c = counts(finite), counts(high)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [all_c[k] for k in names], 0.4, label=f"all (n={len(finite)})", color="lightgray")
    ax.bar(x + 0.2, [hi_c[k] for k in names], 0.4, label=f"high (n={len(high)})", color="tab:green")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_title("four-action teacher distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures / "action_distribution.png", dpi=140)
    plt.close(fig)

    print("figures written to", args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
