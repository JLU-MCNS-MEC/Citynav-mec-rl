#!/usr/bin/env python3
"""Stage-05 episode store: 03B teacher labels + frozen-ResNet visual features.

One sample = one decision state of one teacher trajectory:

* current visual          : 7x7x2048 spatial map (computed on the fly)
* target reference        : 2048 global embedding (per episode, cached)
* history visual          : 16x2048 pooled embeddings of states t-16..t-1 (cached)
* history pose / action / mask: 16x3 / 16x4 / 16
* global goal belief      : 12 (cached; visual match + heading-frame geometry)
* label                   : the teacher's four-action label at t

Nothing here reads the true target coordinates into the model: the belief is a
function of the target *image* embedding and the candidate library, exactly as
in stage 04.  ``target`` is kept only to build the reference image.

Cache layout (episode level, resumable):

    features_v2/<split>/episodes/<key>.npz     one decision episode per file
    features_v2/<split>/journal.jsonl          one line per finished episode
    features_v2/<split>/index.json             episode order + cache identity

Every episode file is written to a temporary name first and renamed with
``os.replace``.  The journal records the SHA-256 of each finished file, so a
restart re-reads and re-hashes what is on disk instead of trusting that a file
exists.  ``index.json`` pins the encoder, the candidate library and the cache
version: loading a cache built with a different candidate grid raises instead
of silently mixing 24 m and 48 m beliefs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
PIPE = ROOT / "04_visual_goal_global_map_pipeline/scripts"
sys.path.insert(0, str(PIPE))

from pipeline import (  # noqa: E402
    ACTION_VEC, HISTORY_LEN, OBS_HALF_M, OBS_SIZE, TARGET_HALF_M,
    CandidateLibrary, MapRaster, belief_features,
)

TEACHER = HERE.parent / "03b_target_directed_teacher/results"
CANDS = ROOT / "04_visual_goal_global_map_pipeline/cache_v2/candidates24"
CACHE = HERE / "features_v2"
SPLITS = ("train_seen", "val_seen", "val_unseen")
DIMS = {"spatial": (7, 7, 2048), "target": 2048, "hist_vis": 2048,
        "hist_pose": 3, "hist_action": 4, "belief": 12, "actions": 4}

CACHE_VERSION = "episode-v1"
ENCODER_ID = "frozen torchvision resnet50 IMAGENET1K_V2 (layer4, fp16)"
CANDIDATE_VERSION = CANDS.name                     # "candidates24"
# Same batch size as training: fp16 cuDNN picks kernels by batch size, so a
# cache pass at a different batch would produce slightly different features for
# the same image (measured: up to 0.09 absolute at batch 32 vs 128).
ENCODE_CHUNK = 128

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
class FrozenEncoder:
    """Frozen ImageNet ResNet-50 layer4 in fp16 - used by BOTH the cache pass
    and the training loop, so the cached history features and the on-the-fly
    current observation come from the exact same network and precision.

    Images stay uint8 until they reach the GPU and normalisation happens there
    for BOTH the cache pass and the training loop.  An earlier version
    normalised on the CPU during the cache pass; the two paths differed by
    ~7e-7 (a few float32 ULP), which the fp16 cast then amplified to ~0.1 in
    the layer4 features, so only one implementation is kept.
    ``normalise_cpu`` remains as the reference for ``verify_normalization.py``.
    """

    def __init__(self, device: str = "cuda"):
        import torch
        import torchvision

        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = torch.nn.Sequential(*list(model.children())[:-2]).to(device).eval().half()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.device = device
        self.torch = torch
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # -- normalisation ------------------------------------------------------
    def normalise_cpu(self, images: list[np.ndarray]) -> np.ndarray:
        """(B,3,H,W) float32 - normalise in float32 on the CPU."""
        arr = np.stack(images).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))

    def normalise_gpu(self, images: np.ndarray) -> "object":
        """(B,3,H,W) float32 on the GPU - uint8 over the bus, then normalise."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        return x.sub_(self._mean).div_(self._std)

    # -- forward ------------------------------------------------------------
    def forward_nchw(self, x: "object") -> "object":
        """(B,2048,7,7) float32 given (B,3,H,W) float32."""
        with self.torch.no_grad():
            return self.backbone(x.half()).float()

    def _features(self, images: list[np.ndarray] | np.ndarray):
        return self.forward_nchw(self.normalise_gpu(np.stack(images)))

    def spatial_maps(self, images: list[np.ndarray]) -> np.ndarray:
        """(B,2048,7,7) float32 - the current-visual input (cache path)."""
        return self._features(images).cpu().numpy()

    def __call__(self, images: np.ndarray):
        """(B,49,2048) float32 tokens on the GPU (training path)."""
        feats = self.forward_nchw(self.normalise_gpu(images))
        b = feats.shape[0]
        return feats.permute(0, 2, 3, 1).reshape(b, 49, 2048)

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """(B,2048) L2-normalised global embeddings (target reference, pooled)."""
        feats = self.spatial_maps(images)
        vec = feats.mean(axis=(2, 3))
        return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)


# --------------------------------------------------------------------------- #
# teacher data
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    key: int
    map: str
    states: np.ndarray      # (T,3) float32: x, y, yaw
    actions: np.ndarray     # (T,) int64
    target: np.ndarray      # (2,) float32 - offline only (reference image)


def load_split(split: str) -> list[Episode]:
    """Read the 03B direct-teacher trajectories for one split."""
    z = np.load(TEACHER / f"teacher_{split}.npz")
    keys = [int(k) for k in z["keys"]]
    off = z["offsets"]
    # read the big arrays once: indexing NpzFile inside the loop re-decompresses
    # the whole array on every access
    states_all = z["states"]
    actions_all = z["actions"]
    targets_all = z["targets"]
    with (TEACHER / f"teacher_{split}.csv").open(encoding="utf-8") as fh:
        maps = {int(r["index"]): r["map"] for r in csv.DictReader(fh)}
    episodes = []
    for i, key in enumerate(keys):
        episodes.append(Episode(key=key, map=maps[key],
                                states=states_all[off[i]:off[i + 1]].astype(np.float32),
                                actions=actions_all[off[i]:off[i + 1]].astype(np.int64),
                                target=targets_all[i].astype(np.float32)))
    return episodes


class RasterStore:
    """Lazy cache of the 28 map rasters (48 MB each)."""

    def __init__(self) -> None:
        self._rasters: dict[str, MapRaster] = {}

    def get(self, name: str) -> MapRaster:
        if name not in self._rasters:
            self._rasters[name] = MapRaster.load(name)
        return self._rasters[name]


class CandidateStore:
    def __init__(self, root: Path = CANDS) -> None:
        self.root = root
        self._libs: dict[str, CandidateLibrary] = {}

    def get(self, name: str) -> CandidateLibrary:
        if name not in self._libs:
            self._libs[name] = CandidateLibrary.load(name, self.root / f"{name}.npz")
        return self._libs[name]


# --------------------------------------------------------------------------- #
# history
# --------------------------------------------------------------------------- #
def history_block(pooled: np.ndarray, states: np.ndarray, actions: np.ndarray,
                  t: int, length: int = HISTORY_LEN) -> dict:
    """Strictly causal 16-step history at state ``t`` (indices < t only)."""
    vis = np.zeros((length, pooled.shape[1]), np.float32)
    pose = np.zeros((length, 3), np.float32)
    act = np.zeros((length, 4), np.float32)
    mask = np.zeros(length, np.float32)
    x, y, yaw = states[t]
    cos, sin = math.cos(yaw), math.sin(yaw)
    for k in range(1, length + 1):
        j = t - k
        if j < 0:
            continue
        px, py, pyaw = states[j]
        dx, dy = float(px - x), float(py - y)
        pose[k - 1] = [dx * cos + dy * sin, -dx * sin + dy * cos,
                       float((pyaw - yaw + math.pi) % (2 * math.pi) - math.pi)]
        act[k - 1, ACTION_VEC[int(actions[j])]] = 1.0
        vis[k - 1] = pooled[j]
        mask[k - 1] = 1.0
    return {"history_vis": vis, "history_pose": pose,
            "history_action": act, "history_mask": mask}


# --------------------------------------------------------------------------- #
# episode file / journal
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def episode_path(split: str, key: int, root: Path = CACHE) -> Path:
    return root / split / "episodes" / f"{key}.npz"


def prepare_arrays(ep: Episode, raster: MapRaster, lib: CandidateLibrary,
                   encoder: FrozenEncoder) -> dict:
    """Frozen-ResNet pass over one episode (pooled / belief / validity)."""
    t_len = len(ep.actions)
    tvec = encoder.embed([raster.north_crop(float(ep.target[0]), float(ep.target[1]),
                                            TARGET_HALF_M)])[0].astype(np.float32)
    images, fracs = [], []
    for x, y, yaw in ep.states:
        img, frac = raster.heading_crop(float(x), float(y), float(yaw), return_valid=True)
        images.append(img)
        fracs.append(frac)
    pooled = []
    for i in range(0, len(images), ENCODE_CHUNK):
        maps = encoder.spatial_maps(images[i:i + ENCODE_CHUNK]).transpose(0, 2, 3, 1)
        vec = maps.mean(axis=(1, 2))
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        pooled.append(vec)
    pooled = np.concatenate(pooled) if pooled else np.zeros((0, DIMS["target"]), np.float32)

    # the candidate scores of one episode depend only on (library, target_emb)
    sims = lib.embeddings @ tvec
    belief = np.zeros((t_len, DIMS["belief"]), np.float32)
    for t in range(t_len):
        belief[t], _ = belief_features(lib, tvec, ep.states[t, :2], float(ep.states[t, 2]),
                                       sims=sims)
    return {"pooled": pooled.astype(np.float16),
            "belief": belief,
            "valid": np.asarray(fracs, np.float32).astype(np.float16),
            "target_emb": tvec.astype(np.float16),
            "states": ep.states.astype(np.float32),
            "actions": ep.actions.astype(np.int8)}


def episode_meta(ep: Episode, split: str) -> dict:
    return {"key": int(ep.key), "map": ep.map, "split": split,
            "length": int(len(ep.actions)), "encoder": ENCODER_ID,
            "candidate_library": str(CANDS), "cache_version": CACHE_VERSION}


def write_episode(ep: Episode, arrays: dict, split: str, root: Path = CACHE) -> dict:
    """Atomically write one episode file and return its journal entry."""
    path = episode_path(split, ep.key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    meta = episode_meta(ep, split)
    np.savez(tmp, meta=json.dumps(meta), **arrays)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"split": split, "key": int(ep.key), "status": "done",
            "file": str(path.relative_to(root)),
            "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "states": int(len(ep.actions)), "map": ep.map,
            "encoder": ENCODER_ID, "candidate_library": str(CANDS),
            "cache_version": CACHE_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def read_journal(split: str, root: Path = CACHE) -> dict[int, dict]:
    """Latest journal entry per episode id (part journals are merged first)."""
    entries: dict[int, dict] = {}
    base = root / split
    for path in sorted(base.glob("journal*.jsonl")):
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:      # torn last line after a kill
                        continue
                    entries[int(e["key"])] = e
        except FileNotFoundError:
            continue
    return entries


def verify_journal_entry(entry: dict, root: Path = CACHE) -> tuple[bool, str]:
    """A finished episode is only reused when its bytes still match the journal."""
    if entry.get("status") != "done":
        return False, "status"
    if entry.get("candidate_library") != str(CANDS) or entry.get("cache_version") != CACHE_VERSION:
        return False, "identity"
    path = root / entry["file"]
    if not path.exists():
        return False, "missing"
    if path.stat().st_size != entry["bytes"]:
        return False, "size"
    if sha256_file(path) != entry["sha256"]:
        return False, "sha256"
    return True, "ok"


def build_index(split: str, root: Path = CACHE) -> dict:
    """Order the finished episode files exactly like the teacher array."""
    episodes = load_split(split)
    entries = read_journal(split, root)
    missing = [ep.key for ep in episodes if ep.key not in entries]
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} episodes missing from the journal "
                           f"(first: {missing[:5]})")
    items, states = [], 0
    for ep in episodes:
        e = entries[ep.key]
        items.append({"key": int(ep.key), "map": ep.map, "length": int(len(ep.actions)),
                      "file": e["file"], "states": int(e["states"])})
        states += int(e["states"])
    index = {"split": split, "cache_version": CACHE_VERSION, "encoder": ENCODER_ID,
             "candidate_library": str(CANDS), "candidate_version": CANDIDATE_VERSION,
             "dims": DIMS, "history_len": HISTORY_LEN, "obs_half_m": OBS_HALF_M,
             "target_half_m": TARGET_HALF_M, "states": states,
             "episodes_total": len(items), "episodes": items}
    (root / split / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


# --------------------------------------------------------------------------- #
# training-side store
# --------------------------------------------------------------------------- #
class EpisodeStore:
    """Read-only view of the episode cache, loaded lazily per episode."""

    def __init__(self, split: str, root: Path = CACHE, cache_size: int = 512) -> None:
        index_path = root / split / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"{split}: no cache index at {index_path} - run prepare_v2.py")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("cache_version") != CACHE_VERSION:
            raise RuntimeError(f"{split}: cache version {index.get('cache_version')} != {CACHE_VERSION}")
        if index.get("candidate_library") != str(CANDS):
            raise RuntimeError(f"{split}: cache built with {index.get('candidate_library')}, "
                               f"expected {CANDS} - refusing to mix candidate grids")
        if index.get("encoder") != ENCODER_ID:
            raise RuntimeError(f"{split}: cache encoder {index.get('encoder')} != {ENCODER_ID}")
        self.split = split
        self.root = root
        self.index = index
        self.episodes = index["episodes"]
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cache_size = cache_size

    def __len__(self) -> int:
        return len(self.episodes)

    def length(self, e: int) -> int:
        return int(self.episodes[e]["length"])

    def array(self, e: int) -> dict:
        item = self.episodes[e]
        hit = self._cache.get(e)
        if hit is not None:
            self._cache.move_to_end(e)
            return hit
        path = self.root / item["file"]          # journal paths are cache-root relative
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            if meta["key"] != item["key"] or meta["candidate_library"] != str(CANDS):
                raise RuntimeError(f"{path}: episode identity mismatch {meta}")
            data = {k: z[k] for k in ("pooled", "belief", "valid", "target_emb",
                                      "states", "actions", "meta")}
        data["meta"] = meta
        self._cache[e] = data
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data

    def sample(self, e: int, t: int, rasters: RasterStore) -> dict:
        data = self.array(e)
        states, actions = data["states"], data["actions"]
        hist = history_block(data["pooled"], states, actions, t)
        raster = rasters.get(self.episodes[e]["map"])
        image = raster.heading_crop(float(states[t, 0]), float(states[t, 1]),
                                    float(states[t, 2]))
        return {"image": image,
                "target_emb": np.asarray(data["target_emb"], np.float32),
                "belief": np.asarray(data["belief"][t], np.float32),
                "label": int(actions[t]), **hist}

    def reference(self, e: int, t: int) -> np.ndarray:
        """(12,) belief without touching the raster (tests / diagnostics)."""
        return np.asarray(self.array(e)["belief"][t], np.float32)
