#!/usr/bin/env python3
"""Select 256 high-quality episodes and cache per-episode model features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE.parent / "03_fixed50m_four_action_teacher/scripts"))
sys.path.insert(0, str(HERE.parent / "04_visual_goal_global_map_pipeline/scripts"))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))

from four_action_teacher import convert_episode  # noqa: E402
from pipeline import (  # noqa: E402
    DATA, CandidateLibrary, FrozenResNet, MapRaster, belief_features,
    load_ground, target_embedding,
)

SPLIT_JSON = DATA / "sbfnav-data/original_citynav/citynav_train_seen.json"
STAGE2 = HERE.parent / "02_trajectory_quality_audit/results/selection_high.json"
STAGE3 = HERE.parent / "03_fixed50m_four_action_teacher/results/teacher_episodes.csv"
CANDIDATES = HERE.parent / "04_visual_goal_global_map_pipeline/cache/candidates"
DEFAULT_OUT = HERE / "features"


def select_episodes(per_map: int = 11) -> dict:
    high = set(json.loads(STAGE2.read_text(encoding="utf-8"))["indices"])
    by_map: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with STAGE3.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["success"] != "True":
                continue
            idx = int(row["key"].split("|")[0])
            if idx not in high:
                continue
            by_map[row["map"]].append((idx, int(row["steps"])))
    train, val = [], []
    for name in sorted(by_map):
        rows = sorted(by_map[name], key=lambda r: r[1])
        if len(rows) <= per_map:
            chosen = rows
        else:
            picks = np.linspace(0, len(rows) - 1, per_map).round().astype(int)
            chosen = [rows[i] for i in sorted(set(picks))]
        n_val = max(1, round(0.2 * len(chosen)))
        val.extend(idx for idx, _ in chosen[:n_val])
        train.extend(idx for idx, _ in chosen[n_val:])
    return {"train": sorted(train), "val": sorted(val),
            "per_map": per_map, "maps": sorted(by_map)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-map", type=int, default=11)
    parser.add_argument("--selection", type=Path, default=HERE / "results/selection.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selection_path = args.selection
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        selection = select_episodes(args.per_map)
        (HERE / "results").mkdir(exist_ok=True)
        selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    indices = selection["train"] + selection["val"]
    if args.limit:
        indices = indices[: args.limit]
    print(f"episodes: {len(indices)} (train {len(selection['train'])}, val {len(selection['val'])})", flush=True)

    records = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    ground = load_ground()
    encoder = FrozenResNet(device=args.device)
    started = time.time()
    done = 0
    for idx in indices:
        path = args.out / f"{idx:05d}.npz"
        if path.exists():
            done += 1
            continue
        record = records[idx]
        name = f"{record['area']}_block_{record['block']}"
        result = convert_episode(record, ground.get(name))
        if not result.get("success"):
            print(f"skip {idx}: teacher failure", flush=True)
            continue
        states = np.asarray(result["states"], np.float32)
        actions = np.asarray(result["actions"], np.int8)
        raster = MapRaster.load(name)
        library = CandidateLibrary.load(name, CANDIDATES / f"{name}.npz")
        target = np.asarray(record["target_positions"][-1], np.float64)[:2]
        target_emb = target_embedding(raster, target, encoder)

        images = [raster.heading_crop(float(x), float(y), float(yaw)) for x, y, yaw in states]
        spatial = []
        for i in range(0, len(images), 16):
            spatial.append(encoder.spatial(images[i:i + 16]))
        spatial = np.concatenate(spatial).transpose(0, 2, 3, 1).astype(np.float16)  # (T,7,7,2048)
        pooled = spatial.astype(np.float32).mean(axis=(1, 2))
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)

        belief = np.zeros((len(states), 12), np.float32)
        cand_idx = np.zeros((len(states), 5), np.int32)
        cand_sim = np.zeros((len(states), 5), np.float32)
        cand_rel = np.zeros((len(states), 5, 3), np.float32)
        for t, (x, y, yaw) in enumerate(states):
            feat, info = belief_features(library, target_emb, np.asarray([x, y]), float(yaw))
            belief[t] = feat
            cand_idx[t] = info["top_indices"]
            cand_sim[t] = info["sims"]
            delta = info["positions"] - np.asarray([x, y])[None, :]
            cos, sin = math.cos(yaw), math.sin(yaw)
            cand_rel[t, :, 0] = delta[:, 0] * cos + delta[:, 1] * sin
            cand_rel[t, :, 1] = -delta[:, 0] * sin + delta[:, 1] * cos
            cand_rel[t, :, 2] = np.linalg.norm(delta, axis=1)

        np.savez(
            path,
            map=np.asarray(name),
            states=states, actions=actions,
            spatial=spatial, pooled=pooled.astype(np.float16),
            target_emb=target_emb.astype(np.float16),
            belief=belief, cand_idx=cand_idx, cand_sim=cand_sim, cand_rel=cand_rel,
        )
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(indices)}  {time.time() - started:.0f}s", flush=True)
        del raster
    print(f"features ready: {done} episodes in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
