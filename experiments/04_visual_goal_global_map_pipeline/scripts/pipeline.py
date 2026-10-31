#!/usr/bin/env python3
"""Fixed-50 m visual-goal / global-map input pipeline.

Inputs built here (no language, no BERT, no network state):

* current top-down local RGB, rotated to heading-forward, fed to a frozen
  ImageNet ResNet-50 to keep a 7x7x2048 spatial feature map;
* target reference image: a north-up crop of the target area, encoded to a
  global embedding (image content only - no coordinates/ID/distance are passed);
* global map candidates on a fixed 48 m grid, each encoded to a global
  embedding; the target belief is the visual match between the reference and
  the candidates, expressed in the current heading frame;
* causal history: up to 16 past visual embeddings, past four actions and
  relative poses, built only from steps strictly before the current one.

The pipeline is for validation and for stage-05 online training; it does not
build a fused token cache.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rasterio

# New-server portability: CITYNAV_DATA overrides the old-server data root.
DATA = Path(os.environ.get("CITYNAV_DATA", "/home/20261024_1/Workspace/DATA"))
RGBD = DATA / "rgbd"
GROUND_SRC = Path("/tmp/citynav-official/gsamllavanav/mapdata.py")

OBS_SIZE = 224
OBS_HALF_M = 50.0          # 100 m x 100 m local view
TARGET_HALF_M = 48.0       # match the candidate patch scale (96 m x 96 m)
CAND_HALF_M = 48.0         # 96 m x 96 m candidate patch
CAND_STRIDE_M = 48.0
HISTORY_LEN = 16
STOP, FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3
ACTION_VEC = {STOP: 0, FORWARD: 1, TURN_LEFT: 2, TURN_RIGHT: 3}


def load_ground() -> dict[str, float]:
    if not GROUND_SRC.exists():
        return {}
    import importlib.util

    spec = importlib.util.spec_from_file_location("official_mapdata", GROUND_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: float(v) for k, v in module.GROUND_LEVEL.items()}


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def view_area_corners(x: float, y: float, yaw: float, half: float) -> list[tuple[float, float]]:
    """Official `space.view_area_corners` with altitude_from_ground = half."""
    cos, sin = math.cos(yaw), math.sin(yaw)
    front = np.array([cos, sin])
    left = np.array([-sin, cos])
    center = np.array([x, y])
    return [
        tuple(center + half * (front + left)),
        tuple(center + half * (front - left)),
        tuple(center + half * (-front - left)),
        tuple(center + half * (-front + left)),
    ]


@dataclass
class MapRaster:
    name: str
    image: np.ndarray
    left: float
    bottom: float
    right: float
    top: float
    res: float
    validity: np.ndarray     # bool at 1 m resolution (decimated by `valid_scale`)
    valid_scale: int = 10
    nodata: float | None = None
    min_valid_z: float = 0.0
    invalid_ratio: float = 0.0

    @classmethod
    def load(cls, name: str, root: Path = RGBD) -> "MapRaster":
        """Load the RGB map and a *correct* elevation-validity mask.

        The official ``rasterize.py`` writes ``nodata = -9999`` and, depending on
        the version, may afterwards replace every nodata pixel with ``min_z - 1``
        (``mapdata.py``/``rasterize.py`` note).  Both conventions are handled:

        1. the TIF's ``nodata`` tag is honoured verbatim;
        2. a large plateau at the global minimum is treated as fill, because a
           real 0.1 m surface almost never has >2% of pixels at one exact value.

        The old check ``height > -1000`` accepted ``min_z - 1`` fill as valid
        ground and is removed.
        """
        img = cv2.cvtColor(cv2.imread(str(root / f"{name}.png")), cv2.COLOR_BGR2RGB)
        with rasterio.open(root / f"{name}.tif") as ds:
            b = ds.bounds
            z = ds.read(1).astype(np.float32)
            nodata = ds.nodata
        valid = np.isfinite(z)
        if nodata is not None:
            valid &= z != np.float32(nodata)
        if not valid.any():
            raise ValueError(f"{name}: no valid elevation pixels")
        vmin = float(z[valid].min())
        at_min = int((z == np.float32(vmin)).sum())
        if at_min > 0.02 * z.size:            # official min_z-1 fill plateau
            valid &= z != np.float32(vmin)
            vmin = float(z[valid].min())
        validity = valid[::cls.valid_scale, ::cls.valid_scale]
        return cls(name, img, b.left, b.bottom, b.right, b.top, 0.1, validity,
                   cls.valid_scale, None if nodata is None else float(nodata),
                   vmin, float(1.0 - valid.mean()))

    def pixel(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        col = (np.asarray(x) - self.left) / self.res
        row = (self.top - np.asarray(y)) / self.res
        return col, row

    def valid_cell(self, x: float, y: float) -> tuple[int, int]:
        """Row/col of the 1 m validity cell containing (x, y)."""
        return (int((self.top - y) / (self.res * self.valid_scale)),
                int((x - self.left) / (self.res * self.valid_scale)))

    def height_valid(self, x: float, y: float) -> bool:
        h, w = self.validity.shape
        row, col = self.valid_cell(x, y)
        if not (0 <= row < h and 0 <= col < w):
            return False
        return bool(self.validity[row, col])

    def _patch_window(self, x: float, y: float, half: float):
        step = self.res * self.valid_scale
        h, w = self.validity.shape
        r0 = int(math.floor((self.top - (y + half)) / step))
        r1 = int(math.ceil((self.top - (y - half)) / step))
        c0 = int(math.floor((x - half - self.left) / step))
        c1 = int(math.ceil((x + half - self.left) / step))
        sr0, sr1 = max(r0, 0), min(r1, h)
        sc0, sc1 = max(c0, 0), min(c1, w)
        total = max(r1 - r0, 1) * max(c1 - c0, 1)
        inside = max(sr1 - sr0, 0) * max(sc1 - sc0, 0)
        count = int(self.validity[sr0:sr1, sc0:sc1].sum()) if inside else 0
        return total, inside, count

    def valid_fraction(self, x: float, y: float, half: float) -> float:
        """Valid fraction of the patch, **within the raster**; 0 if fully outside."""
        _, inside, count = self._patch_window(x, y, half)
        return float(count / inside) if inside else 0.0

    def coverage(self, x: float, y: float, half: float) -> float:
        """Fraction of the patch that lies inside the raster."""
        total, inside, _ = self._patch_window(x, y, half)
        return float(inside / total)

    def heading_crop(self, x: float, y: float, yaw: float, half: float = OBS_HALF_M,
                     out: int = OBS_SIZE, return_valid: bool = False):
        """Heading-forward local observation; invalid elevations are zeroed."""
        corners = view_area_corners(x, y, yaw, half)
        cols, rows = self.pixel([p[0] for p in corners], [p[1] for p in corners])
        src = np.stack([cols, rows], axis=1).astype(np.float32)
        dst = np.array([[0, 0], [out - 1, 0], [out - 1, out - 1], [0, out - 1]], np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(self.image, matrix, (out, out), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = self._mask_valid(self.validity, matrix, (out, out))
        image = image * mask[:, :, None]
        if return_valid:
            return image, float(mask.mean())
        return image

    def north_crop(self, x: float, y: float, half: float, out: int = OBS_SIZE):
        """North-up axis-aligned crop; zero-padded outside the raster."""
        c0, r0 = self.pixel(x - half, y + half)
        c1, r1 = self.pixel(x + half, y - half)
        c0, r0, c1, r1 = int(round(c0)), int(round(r0)), int(round(c1)), int(round(r1))
        h, w = self.image.shape[:2]
        ph, pw = max(r1 - r0, 1), max(c1 - c0, 1)
        canvas = np.zeros((ph, pw, 3), np.uint8)
        mask = np.zeros((ph, pw), np.uint8)
        sc0, sr0 = max(c0, 0), max(r0, 0)
        sc1, sr1 = min(c1, w), min(r1, h)
        if sc1 > sc0 and sr1 > sr0:
            canvas[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = self.image[sr0:sr1, sc0:sc1]
            s = self.valid_scale
            sub = self.validity[sr0 // s:(sr1 + s - 1) // s, sc0 // s:(sc1 + s - 1) // s]
            if sub.size:
                sub = cv2.resize(sub.astype(np.uint8) * 255, (sc1 - sc0, sr1 - sr0),
                                 interpolation=cv2.INTER_NEAREST)
                mask[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = sub
        canvas = cv2.resize(canvas, (out, out), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (out, out), interpolation=cv2.INTER_NEAREST)
        return canvas * (mask > 0)[:, :, None]

    def _mask_valid(self, validity: np.ndarray, matrix: np.ndarray, out: tuple[int, int]) -> np.ndarray:
        """Warp the 1 m validity grid with the same transform as the RGB crop.

        ``matrix`` maps full-resolution (0.1 m) pixel coordinates, so it is
        composed with a scale matrix to apply to the decimated grid.
        """
        s = float(self.valid_scale)
        scale = np.array([[s, 0, 0.5 * s], [0, s, 0.5 * s], [0, 0, 1.0]], np.float64)
        canvas = (validity.astype(np.uint8) * 255)
        warped = cv2.warpPerspective(canvas, matrix @ scale, out, flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (warped > 0).astype(np.uint8)


def to_tensor(images: list[np.ndarray]):
    """RGB uint8 list -> normalized float tensor (B,3,224,224)."""
    import torch

    arr = np.stack(images).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(arr.transpose(0, 3, 1, 2)).contiguous()


class FrozenResNet:
    def __init__(self, device: str = "cpu"):
        import torch
        import torchvision

        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = torch.nn.Sequential(*list(model.children())[:-2]).to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.device = device
        self.torch = torch

    def spatial(self, images: list[np.ndarray]) -> np.ndarray:
        """(B,2048,7,7) float32 feature maps."""
        with self.torch.no_grad():
            out = self.backbone(to_tensor(images).to(self.device))
        return out.cpu().numpy().astype(np.float32)

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """(B,2048) normalized global embeddings."""
        feats = self.spatial(images)
        vec = feats.mean(axis=(2, 3))
        return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)


@dataclass
class CandidateLibrary:
    name: str
    positions: np.ndarray          # (N,2) world xy of fixed-grid candidates
    embeddings: np.ndarray         # (N,2048) normalized
    valid_fraction: np.ndarray | None = None   # (N,) valid elevation ratio (in-raster)
    coverage: np.ndarray | None = None         # (N,) in-raster ratio of the patch

    def save(self, path: Path) -> None:
        kwargs = {}
        if self.valid_fraction is not None:
            kwargs["valid_fraction"] = self.valid_fraction.astype(np.float16)
        if self.coverage is not None:
            kwargs["coverage"] = self.coverage.astype(np.float16)
        np.savez_compressed(path, positions=self.positions,
                            embeddings=self.embeddings.astype(np.float16), **kwargs)

    @classmethod
    def load(cls, name: str, path: Path) -> "CandidateLibrary":
        z = np.load(path)
        vf = z["valid_fraction"].astype(np.float32) if "valid_fraction" in z.files else None
        cov = z["coverage"].astype(np.float32) if "coverage" in z.files else None
        return cls(name, z["positions"].astype(np.float32),
                   z["embeddings"].astype(np.float32), vf, cov)


def build_candidates(raster: MapRaster, encoder: FrozenResNet, stride: float = CAND_STRIDE_M,
                     half: float = CAND_HALF_M,
                     min_valid: float = 0.7,
                     min_coverage: float = 0.4) -> CandidateLibrary:
    """Fixed-grid candidates that are mostly valid ground.

    Conditions (all must hold):

    * patch centre is a valid elevation pixel;
    * at least ``min_coverage`` of the 96 m patch lies inside the raster
      (narrow blocks would otherwise have no candidate at all);
    * at least ``min_valid`` of the in-raster part is valid elevation.

    This replaces the old ``height > -1000`` check, which accepted the official
    ``min_z - 1`` nodata fill as real ground.
    """
    margin = 8.0
    xs = np.arange(raster.left + margin, raster.right - margin + 1e-6, stride)
    ys = np.arange(raster.bottom + margin, raster.top - margin + 1e-6, stride)
    positions, fractions, coverages = [], [], []
    for x in xs:
        for y in ys:
            if not raster.height_valid(x, y):
                continue
            cov = raster.coverage(x, y, half)
            if cov < min_coverage:
                continue
            frac = raster.valid_fraction(x, y, half)
            if frac < min_valid:
                continue
            positions.append((x, y))
            fractions.append(frac)
            coverages.append(cov)
    if not positions:
        return CandidateLibrary(raster.name, np.zeros((0, 2), np.float32),
                                np.zeros((0, 2048), np.float32), np.zeros(0, np.float32),
                                np.zeros(0, np.float32))
    images = [raster.north_crop(float(x), float(y), half) for x, y in positions]
    embeddings = []
    for i in range(0, len(images), 16):
        embeddings.append(encoder.embed(images[i:i + 16]))
    return CandidateLibrary(raster.name, np.asarray(positions, np.float32),
                            np.concatenate(embeddings), np.asarray(fractions, np.float32),
                            np.asarray(coverages, np.float32))


def target_embedding(raster: MapRaster, target_xy: np.ndarray, encoder: FrozenResNet) -> np.ndarray:
    """Encode the target reference image (image content only)."""
    return encoder.embed([raster.north_crop(float(target_xy[0]), float(target_xy[1]), TARGET_HALF_M)])[0]


def belief_features(library: CandidateLibrary, target_emb: np.ndarray, pose_xy: np.ndarray,
                    yaw: float, topk: int = 5,
                    sims: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Visual match + relative geometry; never uses the true target coordinates.

    ``sims`` optionally supplies the already computed ``library.embeddings @
    target_emb`` for this episode; it must be the same array, and passing it only
    avoids recomputing a per-episode constant once per decision state.
    """
    if len(library.positions) == 0:
        return np.zeros(12, np.float32), {
            "top_indices": np.zeros(0, int), "sims": np.zeros(0),
            "distances": np.zeros(0), "positions": np.zeros((0, 2)),
            "order": np.zeros(0, int), "rank_distances": np.zeros(0),
            "top5_positions": np.zeros((0, 2)), "top5_weights": np.zeros(0)}
    if sims is None:
        sims = library.embeddings @ target_emb
    order = np.argsort(-sims)
    # the 12-d belief is always defined on the top-5; `topk` only widens the
    # ranking information returned for Recall@K, it must not change the feature
    k = min(5, len(order))
    idx = order[:k]
    weights = np.exp(10.0 * (sims[idx] - sims[idx].max()))
    weights = weights / max(weights.sum(), 1e-12)
    cand = library.positions[idx]
    delta = cand - pose_xy[None, :]
    cos, sin = math.cos(yaw), math.sin(yaw)
    dx_f = delta[:, 0] * cos + delta[:, 1] * sin
    dy_l = -delta[:, 0] * sin + delta[:, 1] * cos
    dist = np.linalg.norm(delta, axis=1)
    wm = weights @ np.stack([dx_f, dy_l], axis=1)
    wm_dist = float(weights @ dist)
    spread = float(np.sqrt(max(weights @ ((dist - wm_dist) ** 2), 0.0)))
    probs = np.exp(10.0 * (sims - sims.max()))
    probs = probs / probs.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(sims))) if len(sims) > 1 else 0.0
    top_delta = delta[0]
    top_dx = float(top_delta[0] * cos + top_delta[1] * sin)
    top_dy = float(-top_delta[0] * sin + top_delta[1] * cos)
    feature = np.asarray([
        float(sims[idx[0]]), float(sims[idx].mean()), entropy,
        top_dx, top_dy, float(dist[0]),
        float(wm[0]), float(wm[1]), wm_dist, spread,
        min(len(library.positions) / 100.0, 10.0), 1.0,
    ], np.float32)
    # Full rank order is kept so Recall@K is computed on exactly K candidates
    # (the old validation reused a 5-element array for Recall@10).
    rank_dist = np.linalg.norm(library.positions[order] - pose_xy[None, :], axis=1)
    k_info = min(max(topk, 5), len(order))
    idx_info = order[:k_info]
    info_delta = library.positions[idx_info] - pose_xy[None, :]
    return feature, {
        "top_indices": idx_info, "sims": sims[idx_info],
        "distances": np.linalg.norm(info_delta, axis=1),
        "positions": library.positions[idx_info],
        "order": order, "rank_distances": rank_dist,
        "top5_positions": cand, "top5_weights": weights,
    }


def encode_episode(raster: MapRaster, states: np.ndarray, encoder: FrozenResNet,
                   return_valid: bool = False):
    """Encode every state once with the frozen ResNet.

    Returns the **7x7x2048 spatial map** per state (the model's current-visual
    input) and the L2-normalised 2048-d pooled vector used for the causal
    history.  One forward pass produces both, so they cannot drift apart.
    """
    images, fracs = [], []
    for x, y, yaw in states:
        img, frac = raster.heading_crop(float(x), float(y), float(yaw), return_valid=True)
        images.append(img)
        fracs.append(frac)
    spatial, pooled = [], []
    for i in range(0, len(images), 16):
        maps = encoder.spatial(images[i:i + 16]).transpose(0, 2, 3, 1)   # (B,7,7,2048)
        vec = maps.mean(axis=(1, 2))
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        spatial.append(maps)
        pooled.append(vec)
    if not spatial:
        empty = np.zeros((0, 7, 7, 2048), np.float32)
        out = {"spatial": empty, "pooled": np.zeros((0, 2048), np.float32)}
        return (out, np.zeros(0, np.float32)) if return_valid else out
    out = {"spatial": np.concatenate(spatial).astype(np.float32),
           "pooled": np.concatenate(pooled).astype(np.float32)}
    return (out, np.asarray(fracs, np.float32)) if return_valid else out


def history_window(feats: np.ndarray, states: np.ndarray, actions: np.ndarray, t: int,
                   length: int = HISTORY_LEN) -> dict:
    """Causal 16-step history at step t (only indices < t are used)."""
    vis = np.zeros((length, feats.shape[1]), np.float32)
    pose = np.zeros((length, 3), np.float32)   # dx_forward, dy_left, dyaw
    act = np.zeros((length, 4), np.float32)
    mask = np.zeros(length, np.float32)
    x, y, yaw = states[t]
    for k in range(1, length + 1):
        j = t - k
        if j < 0:
            continue
        px, py, pyaw = states[j]
        dx, dy = px - x, py - y
        cos, sin = math.cos(yaw), math.sin(yaw)
        pose[k - 1] = [dx * cos + dy * sin, -dx * sin + dy * cos, wrap(pyaw - yaw)]
        act[k - 1, ACTION_VEC[int(actions[j])]] = 1.0
        vis[k - 1] = feats[j]
        mask[k - 1] = 1.0
    return {"history_vis": vis, "history_pose": pose, "history_action": act, "history_mask": mask}
