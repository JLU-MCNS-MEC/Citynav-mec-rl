#!/usr/bin/env python3
"""11B current-vis materialisation（效率优化，简报第 5 节）。

对 memmap 中每个状态的 heading_crop 用冻结 ResNet（FrozenEncoder）编码一次，
物化为 current.npy (N,49,2048) f16——此后训练 DataLoader 直接读特征
（dataset11b.has_current 快路径），不再每 Epoch 重复裁剪与视觉编码。

- 输入：results/data/<split>_memmap/{meta.json, maps.npy, states.npy} + RasterStore；
- 输出：current.npy + meta.json 增补 current 条目 + current_journal.jsonl（断点续跑，
  已写状态数；重跑自动跳过）；
- 大小：train_seen 约 257 GB（f16）；val_seen 约 320 MB。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
OUT_ROOT = HERE.parent / "results" / "data"
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
from dataset_v2 import FrozenEncoder, RasterStore  # noqa: E402

N_TOKENS = 49
SPATIAL_CH = 2048


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train_seen", "val_seen"])
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not __import__("os").environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")

    root = OUT_ROOT / f"{args.split}_memmap"
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    map_names = {v: k for k, v in meta["map_codes"].items()}
    states = np.lib.format.open_memmap(root / "states.npy", mode="r")
    maps_arr = np.lib.format.open_memmap(root / "maps.npy", mode="r")
    n = len(states)
    print(f"{args.split}: {n} states, chunk {args.chunk}", flush=True)

    journal = root / "current_journal.jsonl"
    done = 0
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            done = max(done, int(json.loads(line)["done_states"]))
    print(f"resume from state {done}", flush=True)

    out_path = root / "current.npy"
    if done == 0:
        cur = np.lib.format.open_memmap(out_path, mode="w+", dtype="float16",
                                        shape=(n, N_TOKENS, SPATIAL_CH))
    else:
        cur = np.lib.format.open_memmap(out_path, mode="r+", dtype="float16",
                                        shape=(n, N_TOKENS, SPATIAL_CH))

    enc = FrozenEncoder(args.device)
    rasters = RasterStore()
    t0 = time.time()
    pos = done
    while pos < n:
        end = min(pos + args.chunk, n)
        crops = []
        for k in range(pos, end):
            x, y, yaw = (float(v) for v in states[k])
            crops.append(rasters.get(map_names[int(maps_arr[k])])
                         .heading_crop(x, y, yaw))
        with torch.no_grad():
            feats = enc(np.stack(crops))
        cur[pos:end] = feats.cpu().numpy().astype(np.float16)
        with journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"done_states": end,
                                 "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                     + "\n")
        el = time.time() - t0
        rate = (end - done) / max(el, 1e-9)
        eta = (n - end) / max(rate, 1e-9)
        print(f"  {end}/{n} states, {el:.0f}s, {rate:.0f} st/s, "
              f"eta {eta / 60:.0f} min", flush=True)
        pos = end

    meta["arrays"]["current"] = ["float16", [n, N_TOKENS, SPATIAL_CH]]
    meta["current_materialized"] = True
    (root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                    encoding="utf-8")
    print(f"{args.split}: current.npy done in {time.time() - t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
