#!/usr/bin/env python3
"""Global-goal localisation validation for the fixed-50 m pipeline.

For every convertible Val-Seen / Val-Unseen task, at t = 5, t = 20 and the
near-goal state (the teacher's STOP state), measure how well the visual
reference image retrieves the true target position from the fixed candidate
grid:

* Recall@1 / @5 / @10 at 20 m and 24 m (computed on exactly K candidates -
  the old version reused a 5-element array for Recall@10);
* top-1 candidate distance;
* top-5 weighted position error;
* normalised belief entropy;
* observation / candidate elevation-validity statistics.

`test_unseen` is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import OrderedDict, defaultdict
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
CACHE = HERE / "cache_v2/candidates48"
RADII = (20.0, 24.0)
KS = (1, 5, 10)


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


def pipeline_checks(raster: MapRaster, feats: dict, states: np.ndarray,
                    actions: np.ndarray, t: int) -> dict:
    hist = history_window(feats["pooled"], states, actions, t)
    feats2 = {k: v.copy() for k, v in feats.items()}
    states2, actions2 = states.copy(), actions.copy()
    if t + 1 < len(states):
        states2[t + 1:] += 123.0
        actions2[t + 1:] = (actions2[t + 1:] + 1) % 4
        feats2["pooled"][t + 1:] = 0.0
        feats2["spatial"][t + 1:] = 0.0
    hist2 = history_window(feats2["pooled"], states2, actions2, t)
    causal = all(np.array_equal(hist[k], hist2[k]) for k in hist)
    mask_ok = int(hist["history_mask"].sum()) == min(t, HISTORY_LEN)
    return {"history_causal": bool(causal), "history_mask_ok": bool(mask_ok),
            "history_shapes": {k: list(v.shape) for k, v in hist.items()},
            "spatial_shape": list(feats["spatial"].shape),
            "pooled_shape": list(feats["pooled"].shape)}


class RasterCache:
    """Small LRU of map rasters (each ~48 MB RGB + 16 MB validity)."""

    def __init__(self, size: int = 3):
        self.size = size
        self.items: OrderedDict[str, MapRaster] = OrderedDict()

    def get(self, name: str) -> MapRaster:
        if name in self.items:
            self.items.move_to_end(name)
            return self.items[name]
        raster = MapRaster.load(name)
        self.items[name] = raster
        if len(self.items) > self.size:
            self.items.popitem(last=False)
        return raster


def grouped_records(split: str) -> dict[str, list[tuple[int, dict]]]:
    records = json.loads((SPLITS / f"citynav_{split}.json").read_text(encoding="utf-8"))
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[f"{record['area']}_block_{record['block']}"].append((idx, record))
    return dict(sorted(groups.items()))


def sample_rows(states: np.ndarray, steps: list[int]) -> list[tuple[str, int]]:
    out = []
    n = len(states)
    for s in steps:
        if s == -1:
            if n >= 2:
                out.append(("near", n - 1))
        elif s < n:
            out.append((f"t{s}", s))
    return out


def metric_block(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for radius in RADII:
        tag = f"{int(radius)}m"
        for k in KS:
            out[f"recall@{k}_{tag}"] = float(np.mean([r[f"recall@{k}_{tag}"] for r in rows]))
    out["top1_distance_m"] = {
        "mean": float(np.mean([r["top1_distance"] for r in rows])),
        "median": float(np.median([r["top1_distance"] for r in rows])),
    }
    out["top5_weighted_distance_m"] = {
        "mean": float(np.mean([r["top5_weighted"] for r in rows])),
        "median": float(np.median([r["top5_weighted"] for r in rows])),
        "within_20m": float(np.mean([r["top5_weighted"] <= 20.0 for r in rows])),
        "within_24m": float(np.mean([r["top5_weighted"] <= 24.0 for r in rows])),
    }
    out["belief_entropy_mean"] = float(np.mean([r["entropy"] for r in rows]))
    out["observation_valid_fraction"] = {
        "mean": float(np.mean([r["obs_valid"] for r in rows])),
        "min": float(np.min([r["obs_valid"] for r in rows])),
    }
    return out


def evaluate_split(split: str, encoder: FrozenResNet, steps: list[int], limit: int,
                   max_maps: int, figure: bool, cache: Path = CACHE) -> dict:
    groups = grouped_records(split)
    rasters = RasterCache()
    libraries: dict[str, CandidateLibrary] = {}
    rows: list[dict] = []
    per_map: dict[str, int] = {}
    failures: dict[str, int] = defaultdict(int)
    used = 0
    checks: list[dict] = []
    started = time.time()
    done_maps = 0
    for name, items in groups.items():
        if max_maps and done_maps >= max_maps:
            break
        done_maps += 1
        cache_path = cache / f"{name}.npz"
        if not cache_path.exists():
            failures["missing_candidates"] += len(items)
            print(f"  {name}: no candidate cache, skipped", flush=True)
            continue
        raster = rasters.get(name)
        lib = libraries.get(name) or CandidateLibrary.load(name, cache_path)
        libraries[name] = lib
        per_map[name] = len(lib.positions)
        for idx, record in items:
            if limit and used >= limit:
                break
            result = convert_episode(record, load_ground_cached().get(name))
            if not result.get("success"):
                failures[result.get("reason", "failed")] += 1
                continue
            states = np.asarray(result["states"], np.float64)
            actions = np.asarray(result["actions"], np.int64)
            chosen = sample_rows(states, steps)
            if not chosen:
                failures["short"] += 1
                continue
            target = np.asarray(record["target_positions"][-1], np.float64)[:2]
            if not raster.height_valid(float(target[0]), float(target[1])):
                failures["target_invalid"] += 1
                continue
            phase_of = {t: label for label, t in chosen}
            t_idx = sorted(phase_of)
            images = [raster.heading_crop(float(states[t][0]), float(states[t][1]),
                                          float(states[t][2]), return_valid=True)
                      for t in t_idx]
            target_emb = target_embedding(raster, target, encoder)
            for t, (image, obs_valid) in zip(t_idx, images):
                belief, info = belief_features(lib, target_emb, states[t][:2],
                                               float(states[t][2]), topk=max(KS))
                # Localisation error: distance from each ranked *candidate* to
                # the true target.  (info["rank_distances"] is the distance from
                # the UAV pose to the candidate and must not be used here.)
                rank_d = np.linalg.norm(lib.positions[info["order"]] - target[None, :], axis=1)
                wm = info["top5_weights"] @ info["top5_positions"]
                row = {"split": split, "map": name, "index": idx,
                       "phase": phase_of[t], "t": t,
                       "top1_distance": float(rank_d[0]),
                       "top5_weighted": float(np.linalg.norm(wm - target)),
                       "entropy": float(belief[2]), "obs_valid": float(obs_valid)}
                for radius in RADII:
                    for k in KS:
                        row[f"recall@{k}_{int(radius)}m"] = float(
                            bool((rank_d[:k] <= radius).any()))
                rows.append(row)
                if not checks and figure:
                    feats = encode_episode(raster, states[: t + 1], encoder)
                    checks.append(pipeline_checks(raster, feats, states, actions, t))
                    plot_case(raster, record, result, states, lib, info, target, t, belief)
            used += 1
        print(f"  {name}: {len(items)} tasks, {len(lib.positions)} candidates "
              f"({time.time() - started:.0f}s)", flush=True)
    by_phase = defaultdict(list)
    for row in rows:
        by_phase[row["phase"]].append(row)
    summary = {
        "split": split,
        "test_unseen_read": False,
        "candidates": str(cache),
        "candidate_counts": {"maps": len(per_map), "total": int(sum(per_map.values())),
                             "min": int(min(per_map.values())) if per_map else 0,
                             "max": int(max(per_map.values())) if per_map else 0},
        "episodes_used": used,
        "failures": dict(failures),
        "steps": steps,
        "all": metric_block(rows),
        "by_phase": {phase: metric_block(v) for phase, v in by_phase.items()},
        "checks": {
            "all_history_causal": all(c["history_causal"] for c in checks),
            "all_history_mask_ok": all(c["history_mask_ok"] for c in checks),
            "spatial_shape": checks[0]["spatial_shape"] if checks else None,
            "pooled_shape": checks[0]["pooled_shape"] if checks else None,
        },
        "geometry": value_check(),
        "seconds": round(time.time() - started, 1),
    }
    return {"summary": summary, "rows": rows}


_GROUND = None


def load_ground_cached() -> dict[str, float]:
    global _GROUND
    if _GROUND is None:
        _GROUND = load_ground()
    return _GROUND


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
    axes[2].plot(np.asarray(states)[:, 0], np.asarray(states)[:, 1], "--",
                 color="tab:orange", lw=1.2, label="four-action teacher")
    axes[2].scatter(*states[0][:2], c="lime", edgecolor="black", s=30, label="start")
    axes[2].scatter(*target, c="red", marker="*", s=160, edgecolor="black",
                    label="target (eval only)")
    cand = lib.positions
    axes[2].scatter(cand[:, 0], cand[:, 1], c="white", s=6, alpha=0.6, label="candidates")
    top_pos = info["positions"]
    axes[2].scatter(top_pos[:, 0], top_pos[:, 1], c="yellow", edgecolor="black", s=45,
                    label="top-5 match")
    axes[2].scatter(*states[t][:2], c="cyan", marker="o", s=40, label="current")
    axes[2].legend(fontsize=7, loc="lower left")
    axes[2].set_title(f"belief: top1 sim={belief[0]:.3f}")
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/pipeline_case_v2.png", dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", nargs="+", default=["val_seen", "val_unseen"])
    parser.add_argument("--episodes", type=int, default=0,
                        help="per split cap; 0 = all convertible tasks")
    parser.add_argument("--maps", type=int, default=0, help="cap number of maps (debug)")
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 20, -1],
                        help="-1 = near-goal (teacher STOP state)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", type=Path, default=CACHE)
    parser.add_argument("--out", default="validation_v2")
    args = parser.parse_args()

    encoder = FrozenResNet(device=args.device)
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    combined = {}
    for split in args.split:
        print(f"=== {split} ===", flush=True)
        result = evaluate_split(split, encoder, args.steps, args.episodes,
                                args.maps, figure=(split == "val_seen"),
                                cache=args.candidates)
        summary, rows = result["summary"], result["rows"]
        (HERE / f"results/{args.out}_{split}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if rows:
            keys = list(rows[0].keys())
            with (HERE / f"results/{args.out}_{split}.csv").open("w", newline="",
                                                                 encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        combined[split] = summary
    (HERE / f"results/{args.out}_combined.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
