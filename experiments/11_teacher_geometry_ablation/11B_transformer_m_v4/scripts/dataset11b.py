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
from v5fix_contract import EMPTY_BELIEF, EMPTY_LANDMARKS, EMPTY_MATCH  # noqa: E402


def apply_blank(sample: dict, blank: bool) -> dict:
    """V5 修正 5a：blank=True 时用 v5fix_contract 空候选常量替换
    belief/match/landmarks（与推理空候选分支同源），其余字段透传。"""
    if not blank:
        return sample
    sample["belief"] = EMPTY_BELIEF.copy()
    sample["match"] = EMPTY_MATCH.copy()
    sample["landmarks"] = EMPTY_LANDMARKS.copy()
    return sample


def open_memmap(dir_path: Path, names: tuple[str, ...]) -> dict[str, np.memmap]:
    out = {}
    for name in names:
        out[name] = np.lib.format.open_memmap(dir_path / f"{name}.npy", mode="r")
    return out


class ChunkedMemmapDataset(Dataset):
    """One sample = one decision state, read from memmap (OS page cache).

    效率优化（简报第 5 节）：若 memmap 目录存在 materialize_current_11b.py
    物化的 current.npy（冻结 ResNet 输出 f16），则 __getitem__ 直接读特征，
    不再逐 state 生成裁剪 + 每 Epoch 重复视觉编码；否则回退到
    RasterStore crop + 训练循环内编码的旧路径。
    """

    def __init__(self, split: str, idx: np.ndarray,
                 blank_mask: np.ndarray | None = None):
        root = DATA / f"{split}_memmap"
        names = ["hist_vis", "hist_pose", "hist_act", "hist_mask", "belief",
                 "match", "target_emb", "states", "labels", "maps", "landmarks"]
        self.has_current = (root / "current.npy").exists()
        if self.has_current:
            names.append("current")
        self.z = open_memmap(root, tuple(names))
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.map_names = {v: k for k, v in meta["map_codes"].items()}
        self.idx = idx
        if blank_mask is not None and len(blank_mask) != len(idx):
            raise ValueError(f"blank_mask 长度 {len(blank_mask)} != idx {len(idx)}")
        self.blank_mask = blank_mask
        self.rasters: RasterStore | None = None

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict:
        k = self.idx[i]
        sample = {"target_emb": np.asarray(self.z["target_emb"][k], np.float32),
                  "belief": np.asarray(self.z["belief"][k], np.float32),
                  "match": np.asarray(self.z["match"][k], np.float32),
                  "history_vis": self.z["hist_vis"][k].astype(np.float32),
                  "history_pose": np.asarray(self.z["hist_pose"][k], np.float32),
                  "history_action": np.eye(4, dtype=np.float32)[self.z["hist_act"][k]],
                  "history_mask": np.asarray(self.z["hist_mask"][k], np.float32),
                  "states": np.asarray(self.z["states"][k], np.float32),
                  "landmarks": np.asarray(self.z["landmarks"][k], np.float32),
                  "label": int(self.z["labels"][k])}
        if self.has_current:
            sample["current"] = np.asarray(self.z["current"][k], np.float32)
            return apply_blank(sample,
                               self.blank_mask is not None
                               and bool(self.blank_mask[i]))
        if self.rasters is None:      # one raster cache per worker process
            self.rasters = RasterStore()
        x, y, yaw = (float(v) for v in self.z["states"][k])
        sample["image"] = self.rasters.get(self.map_names[int(self.z["maps"][k])]) \
            .heading_crop(x, y, yaw)
        return apply_blank(sample,
                           self.blank_mask is not None and bool(self.blank_mask[i]))


class ValMemmapDataset(ChunkedMemmapDataset):
    """Small fixed validation set loaded into RAM (like 10D)."""

    def __init__(self, split: str):
        root = DATA / f"{split}_memmap"
        names = ["hist_vis", "hist_pose", "hist_act", "hist_mask", "belief",
                 "match", "target_emb", "states", "labels", "maps", "landmarks"]
        self.has_current = (root / "current.npy").exists()
        if self.has_current:
            names.append("current")
        self.z = {name: np.load(root / f"{name}.npy") for name in names}
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.map_names = {v: k for k, v in meta["map_codes"].items()}
        self.idx = np.arange(len(self.z["labels"]))
        self.blank_mask = None
        self.rasters = None
