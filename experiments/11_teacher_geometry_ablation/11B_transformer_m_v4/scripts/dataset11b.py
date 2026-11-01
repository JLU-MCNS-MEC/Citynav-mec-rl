#!/usr/bin/env python3
"""11B datasets: memmap-backed, chunk-friendly (doc 4.3: no whole-pool RAM).

ChunkedMemmapDataset serves one decision state per __getitem__ from the
materialised memmap arrays (results/data/<split>_memmap/).  Only the current
image crop is produced per sample (same as 10D: RasterStore + heading_crop).

ValMemmapDataset loads the small fixed validation memmap into RAM once
(1,600 states) for fast per-epoch evaluation (mirrors 10D's RamDataset).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DATA = HERE.parent / "results" / "data"
import sys  # noqa: E402
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
from dataset_v2 import RasterStore  # noqa: E402


def open_memmap(dir_path: Path, names: tuple[str, ...]) -> dict[str, np.memmap]:
    out = {}
    for name in names:
        out[name] = np.lib.format.open_memmap(dir_path / f"{name}.npy", mode="r")
    return out


class ChunkedMemmapDataset(Dataset):
    """One sample = one decision state, read from memmap (OS page cache)."""

    def __init__(self, split: str, idx: np.ndarray):
        root = DATA / f"{split}_memmap"
        self.z = open_memmap(root, ("hist_vis", "hist_pose", "hist_act",
                                    "hist_mask", "belief", "match", "target_emb",
                                    "states", "labels", "maps"))
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.map_names = {v: k for k, v in meta["map_codes"].items()}
        self.idx = idx
        self.rasters: RasterStore | None = None

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict:
        if self.rasters is None:      # one raster cache per worker process
            self.rasters = RasterStore()
        k = self.idx[i]
        x, y, yaw = (float(v) for v in self.z["states"][k])
        image = self.rasters.get(self.map_names[int(self.z["maps"][k])]) \
            .heading_crop(x, y, yaw)
        return {"image": image,
                "target_emb": np.asarray(self.z["target_emb"][k], np.float32),
                "belief": np.asarray(self.z["belief"][k], np.float32),
                "match": np.asarray(self.z["match"][k], np.float32),
                "history_vis": self.z["hist_vis"][k].astype(np.float32),
                "history_pose": np.asarray(self.z["hist_pose"][k], np.float32),
                "history_action": np.eye(4, dtype=np.float32)[self.z["hist_act"][k]],
                "history_mask": np.asarray(self.z["hist_mask"][k], np.float32),
                "label": int(self.z["labels"][k])}


class ValMemmapDataset(ChunkedMemmapDataset):
    """Small fixed validation set loaded into RAM (like 10D)."""

    def __init__(self, split: str):
        root = DATA / f"{split}_memmap"
        self.z = {name: np.load(root / f"{name}.npy")
                  for name in ("hist_vis", "hist_pose", "hist_act", "hist_mask",
                               "belief", "match", "target_emb", "states",
                               "labels", "maps")}
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.map_names = {v: k for k, v in meta["map_codes"].items()}
        self.idx = np.arange(len(self.z["labels"]))
        self.rasters = None
