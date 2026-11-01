#!/usr/bin/env python3
"""11B materialisation: 11B feature cache -> memmap arrays.

Reads the 11B cache states.csv (train_seen: ALL decision states of the 11A
final pool, uncapped; val_seen: fixed 400x4 validation subset) and streams the
per-episode npz from build_features_11b.py into memmap .npy files for chunked
DataLoader training (doc 4.3: no whole-pool RAM loading).

Per-state arrays (index order == state_id order in states.csv):
  hist_vis   (N,16,2048) f16  causal history pooled layer4 (dataset_v2)
  hist_pose  (N,16,3)    f32  relative pose, current heading frame
  hist_act   (N,16)      u8   past four-action indices
  hist_mask  (N,16)      f32  1 for real past steps
  belief     (N,10)      f32  09C continuous belief (10C contract)
  match      (N,5)       f32  C1 measurement (dx,dy,score,margin,sharpness)
  target_emb (N,512)     f32  C1 goal descriptor (central 2x2 layer2, L2)
  states     (N,3)       f32  x,y,yaw (current-image crop)
  labels     (N,)        u8   Stage 03 four-action teacher labels
  maps       (N,)        u8   map codes (meta.json)

Derivations replicate materialize_10d.py exactly (history_block from
dataset_v2, target_emb central block [13:15,13:15] L2).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
CACHE = HERE.parent / "results" / "cache"
OUT_ROOT = HERE.parent / "results" / "data"
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
from dataset_v2 import history_block  # noqa: E402

HIST = 16


def open_mem(path: Path, shape: tuple[int, ...], dtype: str) -> np.memmap:
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=dtype, shape=shape, fortran_order=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train_seen", "val_seen"])
    args = parser.parse_args()
    split = args.split

    with (CACHE / split / "states.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    print(f"{split}: {n} states", flush=True)
    maps_sorted = sorted({r["map"] for r in rows})
    map_code = {m: i for i, m in enumerate(maps_sorted)}

    out = OUT_ROOT / f"{split}_memmap"
    out.mkdir(parents=True, exist_ok=True)
    hist_vis = open_mem(out / "hist_vis.npy", (n, HIST, 2048), "float16")
    hist_pose = open_mem(out / "hist_pose.npy", (n, HIST, 3), "float32")
    hist_act = open_mem(out / "hist_act.npy", (n, HIST), "uint8")
    hist_mask = open_mem(out / "hist_mask.npy", (n, HIST), "float32")
    belief = open_mem(out / "belief.npy", (n, 10), "float32")
    match = open_mem(out / "match.npy", (n, 5), "float32")
    target_emb = open_mem(out / "target_emb.npy", (n, 512), "float32")
    states = open_mem(out / "states.npy", (n, 3), "float32")
    labels = open_mem(out / "labels.npy", (n,), "uint8")
    maps_arr = open_mem(out / "maps.npy", (n,), "uint8")

    t0 = time.time()
    ep_cache: dict[str, tuple] = {}

    def episode_arrays(file: str) -> tuple:
        if file not in ep_cache:
            z = np.load(CACHE / split / file)
            g = np.asarray(z["target_emb"], np.float32)
            desc = g[:, 13:15, 13:15].mean(axis=(1, 2))
            desc = desc / np.maximum(np.linalg.norm(desc, keepdims=True), 1e-8)
            ep_cache[file] = (z["pooled"], z["states"], z["actions"],
                              z["belief"], z["match"], desc)
        return ep_cache[file]

    for pos, r in enumerate(rows):
        pooled, st, act, bel, mch, desc = episode_arrays(r["file"])
        t_i = int(r["t"])
        h = history_block(pooled, st, act, t_i)
        hist_vis[pos] = h["history_vis"].astype(np.float16)
        hist_pose[pos] = h["history_pose"]
        hist_act[pos] = h["history_action"].argmax(-1).astype(np.uint8)
        hist_mask[pos] = h["history_mask"]
        belief[pos] = bel[t_i]
        match[pos] = mch[t_i]
        target_emb[pos] = desc
        states[pos] = st[t_i]
        labels[pos] = int(r["label"])
        maps_arr[pos] = map_code[r["map"]]
        if (pos + 1) % 200000 == 0:
            el = time.time() - t0
            print(f"  {pos + 1}/{n} states, {el:.0f}s, {(pos + 1) / el:.0f} st/s",
                  flush=True)

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": split,
        "n_states": n,
        "source_cache": str(CACHE / split),
        "map_codes": map_code,
        "arrays": {
            "hist_vis": ["float16", (n, HIST, 2048)],
            "hist_pose": ["float32", (n, HIST, 3)],
            "hist_act": ["uint8", (n, HIST)],
            "hist_mask": ["float32", (n, HIST)],
            "belief": ["float32", (n, 10)],
            "match": ["float32", (n, 5)],
            "target_emb": ["float32", (n, 512)],
            "states": ["float32", (n, 3)],
            "labels": ["uint8", (n,)],
            "maps": ["uint8", (n,)],
        },
        "history_len": HIST,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                   encoding="utf-8")
    print(f"{split}: materialized {n} states in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
