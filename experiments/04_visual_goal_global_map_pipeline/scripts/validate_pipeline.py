#!/usr/bin/env python3
"""Small-scale validation of the fixed-50 m visual-goal / global-map pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE.parent / "03_fixed50m_four_action_teacher/scripts"))

from four_action_teacher import convert_episode  # noqa: E402
from pipeline import (  # noqa: E402
    DATA, HISTORY_LEN, CandidateLibrary, FrozenResNet, MapRaster,
    belief_features, encode_episode, history_window, load_ground,
    target_embedding, view_area_corners,
)

SPLITS = DATA / "sbfnav-data/original_citynav"
CACHE = HERE / "cache/candidates"
RECALL_RADIUS = 48.0


def value_check() -> dict:
    """Coordinate-frame check: front-edge centre must land on the top edge."""
    half, out = 50.0, 224
    x, y, yaw = 200.0, 600.0, 0.7
    corners = np.asarray(view_area_corners(x, y, yaw, half), np.float64)
    sides = np.linalg.norm(np.diff(np.vstack([corners, corners[:1]]), axis=0), axis=1)
    center = corners.mean(axis=0)
    return {
        "corner_side_error_m": float(np.max(np.abs(sides - 2 * half))),
        "corner_center_error_m": float(np.linalg.norm(center - np.array([x, y]))),
    }


def pipeline_checks(raster: MapRaster, feats: np.ndarray, states: np.ndarray,
                    actions: np.ndarray, t: int) -> dict:
    hist = history_window(feats, states, actions, t)
    feats2, states2, actions2 = feats.copy(), states.copy(), actions.copy()
    if t + 1 < len(states):
        states2[t + 1:] += 123.0
        actions2[t + 1:] = (actions2[t + 1:] + 1) % 4
        feats2[t + 1:] = 0.0
    hist2 = history_window(feats2, states2, actions2, t)
    causal = all(np.array_equal(hist[k], hist2[k]) for k in hist)
    mask_ok = int(hist["history_mask"].sum()) == min(t, HISTORY_LEN)

    # coordinate frame: project the front-edge centre through the actual crop matrix
    out = 224
    corners = view_area_corners(float(states[t][0]), float(states[t][1]), float(states[t][2]), 50.0)
    cols, rows = raster.pixel([p[0] for p in corners], [p[1] for p in corners])
    src = np.stack([cols, rows], axis=1).astype(np.float32)
    dst = np.array([[0, 0], [out - 1, 0], [out - 1, out - 1], [0, out - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    fx = states[t][0] + 50 * math.cos(states[t][2])
    fy = states[t][1] + 50 * math.sin(states[t][2])
    fcol, frow = raster.pixel(fx, fy)
    front = np.array([[[float(fcol), float(frow)]]], np.float32)
    projected = cv2.perspectiveTransform(front, matrix)[0, 0]
    frame_err = float(np.linalg.norm(projected - np.array([(out - 1) / 2.0, 0.0])))

    return {
        "history_causal": bool(causal),
        "history_mask_ok": bool(mask_ok),
        "front_edge_pixel_error": frame_err,
        "history_shapes": {k: list(v.shape) for k, v in hist.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 20])
    args = parser.parse_args()

    records = json.loads((SPLITS / "citynav_val_seen.json").read_text(encoding="utf-8"))
    stride = max(1, len(records) // args.episodes)
    chosen = records[::stride][: args.episodes]
    ground = load_ground()
    encoder = FrozenResNet(device="cpu")

    recall = {"1": [], "5": [], "10": []}
    recall24 = {"1": [], "5": [], "10": []}
    top1_dist, entropies, wm_dist = [], [], []
    checks_all = []
    encode_times = []
    used = 0
    figure_done = False
    started = time.time()

    for k, record in enumerate(chosen, 1):
        name = f"{record['area']}_block_{record['block']}"
        result = convert_episode(record, ground.get(name))
        if not result.get("success"):
            continue
        states = np.asarray(result["states"], dtype=np.float64)
        actions = np.asarray(result["actions"], dtype=np.int64)
        if len(states) < 2:
            continue
        raster = MapRaster.load(name)
        lib = CandidateLibrary.load(name, CACHE / f"{name}.npz")
        target = np.asarray(record["target_positions"][-1], np.float64)[:2]
        t0 = time.time()
        target_emb = target_embedding(raster, target, encoder)
        encode_times.append(time.time() - t0)
        max_t = min(max(args.steps), len(states) - 1)
        feats = encode_episode(raster, states[: max_t + 1], encoder)
        used += 1
        for t in args.steps:
            t = min(t, len(states) - 1)
            if t < 1:
                continue
            belief, info = belief_features(lib, target_emb, states[t][:2], float(states[t][2]))
            d = np.linalg.norm(info["positions"] - target[None, :], axis=1)
            for kk in recall:
                recall[kk].append(bool((d[: int(kk)] <= RECALL_RADIUS).any()))
                recall24[kk].append(bool((d[: int(kk)] <= 24.0).any()))
            top1_dist.append(float(d[0]))
            entropies.append(float(belief[2]))
            wm = info["weights"] @ info["positions"]
            wm_dist.append(float(np.linalg.norm(wm - target)))
            checks_all.append(pipeline_checks(raster, feats, states, actions, t))
            if not figure_done:
                plot_case(raster, record, result, states, lib, info, target, t, belief)
                figure_done = True
        if k % 10 == 0:
            print(f"  {k}/{len(chosen)} processed, {used} usable, "
                  f"{time.time() - started:.0f}s", flush=True)

    checks = {
        "all_history_causal": all(c["history_causal"] for c in checks_all),
        "all_history_mask_ok": all(c["history_mask_ok"] for c in checks_all),
        "max_front_edge_pixel_error": max((c["front_edge_pixel_error"] for c in checks_all), default=None),
    }
    summary = {
        "split": "val_seen",
        "test_unseen_read": False,
        "episodes_requested": len(chosen),
        "episodes_used": used,
        "steps_per_episode": args.steps,
        "recall_48m": {k: float(np.mean(v)) for k, v in recall.items()},
        "recall_24m": {k: float(np.mean(v)) for k, v in recall24.items()},
        "top1_distance_m": {"mean": float(np.mean(top1_dist)), "median": float(np.median(top1_dist))},
        "belief_entropy_mean": float(np.mean(entropies)),
        "top5_weighted_distance_m": {"mean": float(np.mean(wm_dist)),
                                     "median": float(np.median(wm_dist)),
                                     "within_24m": float(np.mean(np.asarray(wm_dist) <= 24.0))},
        "observation": {"size": 224, "half_extent_m": 50.0,
                        "spatial_feature": [2048, 7, 7]},
        "target_reference": {"half_extent_m": 48.0, "embedding_dim": 2048},
        "candidates": {"stride_m": 48.0, "patch_m": 96.0},
        "history": {"length": HISTORY_LEN,
                    "visual_dim": 2048, "pose_dim": 3, "action_dim": 4},
        "checks": checks,
        "timing": {"candidate_build_seconds": None,
                   "target_encode_seconds_mean": float(np.mean(encode_times)),
                   "validation_seconds": round(time.time() - started, 1)},
        "geometry": value_check(),
    }
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results/validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def plot_case(raster, record, result, states, lib, info, target, t, belief) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    obs = raster.heading_crop(float(states[t][0]), float(states[t][1]), float(states[t][2]))
    axes[0].imshow(obs)
    axes[0].set_title(f"current heading crop  t={t}")
    ref = raster.north_crop(float(target[0]), float(target[1]), 48.0)
    axes[1].imshow(ref)
    axes[1].set_title("target reference (north-up)")

    left, bottom, right, top = raster.left, raster.bottom, raster.right, raster.top
    axes[2].imshow(raster.image, extent=(left, right, bottom, top), origin="upper")
    xy = np.asarray(record["trajectory"], float)[:, :2]
    axes[2].plot(xy[:, 0], xy[:, 1], "-", color="tab:blue", lw=1.2, label="human path")
    axes[2].scatter(*states[0][:2], c="lime", edgecolor="black", s=30, label="start")
    axes[2].scatter(*target, c="red", marker="*", s=160, edgecolor="black", label="target (eval only)")
    cand = lib.positions
    axes[2].scatter(cand[:, 0], cand[:, 1], c="white", s=6, alpha=0.6, label="candidates")
    top_pos = info["positions"]
    axes[2].scatter(top_pos[:, 0], top_pos[:, 1], c="yellow", edgecolor="black", s=45, label="top-5 match")
    axes[2].scatter(*states[t][:2], c="cyan", marker="o", s=40, label="current")
    axes[2].legend(fontsize=7, loc="lower left")
    axes[2].set_title(f"belief: top1 sim={belief[0]:.3f}")
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "figures/pipeline_case.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
