#!/usr/bin/env python3
"""Episode feature store and four-action sample builder."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

N_HISTORY = 16
SPATIAL_HW = 7
SPATIAL_CH = 2048


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class EpisodeStore:
    def __init__(self, feature_dir: Path, library_dir: Path, cache_size: int = 24):
        self.feature_dir = Path(feature_dir)
        self.library_dir = Path(library_dir)
        self.cache_size = cache_size
        self._cache: "OrderedDict[int, dict]" = OrderedDict()
        self._libraries: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def feature_path(self, index: int) -> Path:
        return self.feature_dir / f"{index:05d}.npz"

    def length(self, index: int) -> int:
        return len(self.get(index)["actions"])

    def get(self, index: int) -> dict:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        z = np.load(self.feature_path(index), allow_pickle=False)
        episode = {key: z[key] for key in z.files}
        episode["map"] = str(episode["map"])
        self._cache[index] = episode
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode

    def library(self, map_name: str) -> tuple[np.ndarray, np.ndarray]:
        if map_name not in self._libraries:
            z = np.load(self.library_dir / f"{map_name}.npz")
            self._libraries[map_name] = (
                z["positions"].astype(np.float32),
                z["embeddings"].astype(np.float32),
            )
        return self._libraries[map_name]

    def sample(self, index: int, t: int) -> dict:
        ep = self.get(index)
        states = ep["states"]
        actions = ep["actions"]
        pooled = ep["pooled"]
        belief = ep["belief"]
        _, embeddings = self.library(ep["map"])

        hist_vis = np.zeros((N_HISTORY, SPATIAL_CH), np.float32)
        hist_pose = np.zeros((N_HISTORY, 3), np.float32)
        hist_action = np.zeros((N_HISTORY,), np.int64)
        hist_mask = np.zeros((N_HISTORY,), np.float32)
        x, y, yaw = states[t]
        cos, sin = math.cos(yaw), math.sin(yaw)
        for k in range(1, N_HISTORY + 1):
            j = t - k
            if j < 0:
                break
            px, py, pyaw = states[j]
            dx, dy = px - x, py - y
            hist_pose[k - 1] = [dx * cos + dy * sin, -dx * sin + dy * cos, wrap(pyaw - yaw)]
            hist_action[k - 1] = int(actions[j])
            hist_vis[k - 1] = pooled[j]
            hist_mask[k - 1] = 1.0

        cand_idx = ep["cand_idx"][t]
        cand_emb = embeddings[cand_idx]
        cand_meta = np.concatenate([ep["cand_rel"][t], ep["cand_sim"][t][:, None]], axis=1)
        return {
            "current": ep["spatial"][t].astype(np.float32),
            "target_emb": ep["target_emb"].astype(np.float32),
            "history_vis": hist_vis,
            "history_pose": hist_pose,
            "history_action": hist_action,
            "history_mask": hist_mask,
            "candidate_emb": cand_emb,
            "candidate_meta": cand_meta,
            "belief": belief[t].astype(np.float32),
            "label": int(actions[t]),
        }


def collate(samples: list[dict]) -> dict:
    import torch

    keys = ("current", "target_emb", "history_vis", "history_pose", "history_action",
            "history_mask", "candidate_emb", "candidate_meta", "belief")
    out = {key: torch.from_numpy(np.stack([s[key] for s in samples])) for key in keys}
    out["label"] = torch.tensor([s["label"] for s in samples], dtype=torch.long)
    return out


def episode_batches(indices: list[int], store: EpisodeStore, batch_size: int,
                    shuffle: bool, seed: int = 0):
    """Yield batches of consecutive steps within episodes (disk friendly)."""
    rng = np.random.default_rng(seed)
    order = list(indices)
    if shuffle:
        rng.shuffle(order)
    for index in order:
        n = store.length(index)
        starts = list(range(0, n, batch_size))
        if shuffle:
            rng.shuffle(starts)
        for start in starts:
            end = min(n, start + batch_size)
            yield [(index, t) for t in range(start, end)]
