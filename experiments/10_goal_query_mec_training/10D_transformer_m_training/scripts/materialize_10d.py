#!/usr/bin/env python3
"""10D materialisation: states.csv + episode npz -> per-state RAM tensors.

Training loads one combined array file into RAM once (no per-sample disk IO):
  train_combined.npz  teacher(200k) + recovery(50k) states
  val.npz             1,564 fixed Val-Seen states

Arrays (index order == states.csv order):
  hist_vis  (N,16,2048) float16   causal history pooled layer4
  hist_pose (N,16,3)    float32
  hist_act  (N,16)      uint8     (one-hot built at sample time)
  hist_mask (N,16)      float32
  belief    (N,10)      float32   09C continuous state
  match     (N,5)       float32   C1 measurement
  target_emb(N,512)     float32   C1 goal query descriptor (derived from the
                                  stored layer2 map, central 2x2 block, L2)
  states    (N,3)       float32   x,y,yaw (current-image crop)
  labels    (N,)        uint8
  maps      (N,)        object    map name strings
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
CACHE = (REPO / "experiments/10_goal_query_mec_training"
         "/10C_mec_input_dataset/results/cache")
OUT = HERE.parent / "results"
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))
from dataset_v2 import history_block  # noqa: E402


def load_rows(split: str) -> list[dict]:
    with (CACHE / split / "states.csv").open(encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    for r in rows:
        r["t"] = int(r["t"])
        r["file"] = f"{split}/{r['file']}"
    return rows


def materialise(rows: list[dict], out_path: Path) -> int:
    t0 = time.time()
    n = len(rows)
    hist_vis = np.zeros((n, 16, 2048), np.float16)
    hist_pose = np.zeros((n, 16, 3), np.float32)
    hist_act = np.zeros((n, 16), np.uint8)
    hist_mask = np.zeros((n, 16), np.float32)
    belief = np.zeros((n, 10), np.float32)
    match = np.zeros((n, 5), np.float32)
    target_emb = np.zeros((n, 512), np.float32)
    states = np.zeros((n, 3), np.float32)
    labels = np.zeros(n, np.uint8)
    maps = np.empty(n, dtype=object)
    cache: dict[str, dict] = {}
    last_file = None
    for i, r in enumerate(rows):
        if r["file"] != last_file:
            with np.load(CACHE / r["file"]) as z:
                cache = {k: z[k] for k in ("pooled", "belief", "match",
                                           "states", "actions", "target_emb")}
            last_file = r["file"]
        t = r["t"]
        data = cache
        states[i] = data["states"][t]
        labels[i] = data["actions"][t]
        belief[i] = data["belief"][t]
        match[i] = data["match"][t]
        g = np.asarray(data["target_emb"], np.float32)
        desc = g[:, 13:15, 13:15].mean(axis=(1, 2))
        target_emb[i] = desc / np.maximum(np.linalg.norm(desc, keepdims=True), 1e-8)
        maps[i] = r["map"]
        hist = history_block(np.asarray(data["pooled"], np.float32),
                             data["states"], data["actions"], t)
        hist_vis[i] = hist["history_vis"].astype(np.float16)
        hist_pose[i] = hist["history_pose"]
        hist_act[i] = hist["history_action"].argmax(-1).astype(np.uint8)
        hist_mask[i] = hist["history_mask"]
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{n} | {time.time() - t0:.0f}s", flush=True)
    # np.savez's zip container corrupts entries after a >4GB member (zip64
    # central-directory quirk seen at load time in forked workers), so each
    # array is written as its own .npy in a directory instead.
    import os
    out_dir = out_path.with_suffix("")
    tmp_dir = out_path.with_name(f".{out_path.stem}.tmpdir")
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"hist_vis": hist_vis, "hist_pose": hist_pose, "hist_act": hist_act,
              "hist_mask": hist_mask, "belief": belief, "match": match,
              "target_emb": target_emb, "states": states, "labels": labels,
              "maps": maps}
    for name, arr in arrays.items():
        np.save(tmp_dir / f"{name}.npy", arr, allow_pickle=(name == "maps"))
    (tmp_dir / "meta.json").write_text(json.dumps(
        {"states": n, "rows": len(rows),
         "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), encoding="utf-8")
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    os.replace(tmp_dir, out_dir)
    if out_path.exists():
        os.remove(out_path)
    print(f"wrote {out_dir} ({n} states, {time.time() - t0:.0f}s)", flush=True)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.split == "train":
        rows = load_rows("train_seen") + load_rows("recovery")
        out = OUT / "train_combined.npz"
    else:
        rows = load_rows("val_seen")
        out = OUT / "val.npz"
    materialise(rows, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
