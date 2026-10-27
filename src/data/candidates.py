"""Annotation-independent orthophoto candidate library and retrieval features.

Candidates are fixed grid centers and multiple fixed side lengths over the
published orthophoto.  The builder never reads CityRefer object IDs,
annotations, target boxes, or target positions.  Target positions are used
only by :func:`positive_mask_for_target`, which is a training-label helper and
is never part of the inference API.

A candidate is represented by:
  * its map center ``(x, y)`` and side length in metres;
  * a 96x96 orthophoto crop (candidate image);
  * its frozen visual-encoder mean token (384-D).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.config import DataConfig, ObservationConfig, PROJECT_ROOT
from src.data.observation_adapter import TopDownObservationAdapter
from src.data.paths import ensure_output_dir
from src.models.token_utils import images_to_tensor


CANDIDATE_OUTPUT_SIZE = 96
CANDIDATE_STRIDE_M = 48.0
CANDIDATE_SCALES_M = (32.0, 64.0, 128.0)
CANDIDATE_POSITIVE_RADIUS_M = 40.0
MAP_POSITION_SCALE_M = 500.0
MAP_TOP_K = 5
GOAL_TOP_K = 3


@dataclass
class CandidateLibrary:
    map_name: str
    centers: np.ndarray       # [N,2] float32
    scales: np.ndarray        # [N] float32
    images: np.ndarray        # [N,96,96,3] uint8
    embeddings: np.ndarray | None = None  # [N,384] float32

    @property
    def count(self) -> int:
        return int(len(self.centers))

    def with_embeddings(
        self,
        encoder: torch.nn.Module,
        device: torch.device | str = "cpu",
        *,
        requires_grad: bool = False,
    ) -> "CandidateLibrary":
        self.embeddings = encode_candidate_images(
            self.images, encoder, device, requires_grad=requires_grad
        ).detach().cpu().numpy()
        return self


def _library_key(
    map_name: str,
    stride_m: float,
    scales: Sequence[float],
    output_size: int,
) -> str:
    payload = json.dumps(
        {
            "map": map_name,
            "stride": float(stride_m),
            "scales": [float(v) for v in scales],
            "output": int(output_size),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _cache_paths(map_name: str, key: str) -> tuple[Path, Path]:
    root = PROJECT_ROOT / "data_cache" / "candidates"
    return root / f"{map_name}_{key}.npz", root / f"{map_name}_{key}.json"


def _grid_centers(bounds, stride_m: float) -> np.ndarray:
    left, bottom, right, top = (
        float(bounds.left), float(bounds.bottom),
        float(bounds.right), float(bounds.top),
    )
    xs = np.arange(left + stride_m / 2.0, right, stride_m, dtype=np.float64)
    ys = np.arange(bottom + stride_m / 2.0, top, stride_m, dtype=np.float64)
    # Include both boundary-centered positions so small blocks are not
    # represented by a single center far from one edge.
    xs = np.unique(
        np.concatenate([xs, [left + stride_m / 2.0, right - stride_m / 2.0]])
    )
    ys = np.unique(
        np.concatenate([ys, [bottom + stride_m / 2.0, top - stride_m / 2.0]])
    )
    xs = xs[(xs >= left) & (xs <= right)]
    ys = ys[(ys >= bottom) & (ys <= top)]
    if len(xs) == 0:
        xs = np.asarray([(left + right) / 2.0])
    if len(ys) == 0:
        ys = np.asarray([(bottom + top) / 2.0])
    return np.asarray(
        [[float(x), float(y)] for x in xs for y in ys], dtype=np.float32
    )


def build_candidate_library(
    data_config: DataConfig,
    map_name: str,
    observation_config: ObservationConfig | None = None,
    *,
    encoder: torch.nn.Module | None = None,
    device: torch.device | str = "cpu",
    stride_m: float = CANDIDATE_STRIDE_M,
    scales_m: Sequence[float] = CANDIDATE_SCALES_M,
    output_size: int = CANDIDATE_OUTPUT_SIZE,
    rebuild: bool = False,
) -> CandidateLibrary:
    """Build or load a fixed grid/multi-scale orthophoto candidate library."""
    config = observation_config or ObservationConfig()
    key = _library_key(map_name, stride_m, scales_m, output_size)
    npz_path, meta_path = _cache_paths(map_name, key)
    if npz_path.exists() and meta_path.exists() and not rebuild:
        payload = np.load(npz_path)
        library = CandidateLibrary(
            map_name=map_name,
            centers=payload["centers"].astype(np.float32),
            scales=payload["scales"].astype(np.float32),
            images=payload["images"].astype(np.uint8),
        )
        if "embeddings" in payload and len(payload["embeddings"]):
            library.embeddings = payload["embeddings"].astype(np.float32)
    else:
        adapter = TopDownObservationAdapter(data_config, config)
        try:
            data = adapter.store.get(map_name)
            centers = _grid_centers(data.dataset.bounds, float(stride_m))
            images = []
            scales = []
            for center in centers:
                for scale in scales_m:
                    images.append(
                        adapter.render_center_crop(
                            map_name,
                            center,
                            float(scale),
                            output_size=int(output_size),
                        )
                    )
                    scales.append(float(scale))
            # Repeat each center for its scales in the same order.
            centers = np.repeat(centers, len(scales_m), axis=0)
            library = CandidateLibrary(
                map_name=map_name,
                centers=centers.astype(np.float32),
                scales=np.asarray(scales, dtype=np.float32),
                images=np.stack(images).astype(np.uint8),
            )
        finally:
            adapter.close()
        ensure_output_dir(npz_path.parent, data_config=data_config)
        np.savez_compressed(
            npz_path,
            centers=library.centers,
            scales=library.scales,
            images=library.images,
        )
        meta_path.write_text(
            json.dumps(
                {
                    "map_name": map_name,
                    "key": key,
                    "stride_m": float(stride_m),
                    "scales_m": [float(v) for v in scales_m],
                    "output_size": int(output_size),
                    "count": library.count,
                    "source": "published orthophoto; no CityRefer target labels",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if encoder is not None:
        library.embeddings = encode_candidate_images(
            library.images, encoder, device
        ).detach().cpu().numpy()
        np.savez_compressed(
            npz_path,
            centers=library.centers,
            scales=library.scales,
            images=library.images,
            embeddings=library.embeddings,
        )
    return library


def encode_candidate_images(
    images: np.ndarray,
    encoder: torch.nn.Module,
    device: torch.device | str = "cpu",
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Mean-pooled 384-D candidate visual token.

    The candidate crop is a real map patch, not an object-on-background goal
    crop, so foreground-weighted goal pooling is deliberately not used.
    """
    array = np.asarray(images, dtype=np.uint8)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"images must be [N,H,W,3], got {array.shape}")
    x = images_to_tensor(array, device)
    image_size = int(getattr(encoder, "image_size", 336))
    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(
            x, size=(image_size, image_size), mode="bilinear",
            align_corners=False,
        )
    if requires_grad:
        tokens = encoder(x)
    else:
        with torch.no_grad():
            tokens = encoder(x)
    return tokens.mean(dim=1)


def positive_mask_for_target(
    centers: np.ndarray,
    target_xy: Sequence[float],
    radius_m: float = CANDIDATE_POSITIVE_RADIUS_M,
) -> np.ndarray:
    """Training-only candidate labels from a target position.

    This function is not used by the inference/retrieval API.
    """
    points = np.asarray(centers, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centers must have shape [N,2]")
    if target.shape != (2,):
        raise ValueError("target_xy must have shape [2]")
    return np.linalg.norm(points - target[None, :], axis=1) <= float(radius_m)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if k <= 0:
        raise ValueError("k must be positive")
    order = np.argsort(-scores)
    return order[: min(int(k), len(order))]


def build_goal_tokens(
    candidate_embeddings: np.ndarray,
    top_indices: Sequence[int],
    *,
    k: int = GOAL_TOP_K,
    dim: int = 384,
) -> tuple[np.ndarray, int]:
    """Return zero-padded [k,dim] goal tokens and their valid count.

    This is the explicit retrieval-failure policy: missing candidates become
    zero tokens, and the caller reports ``valid_count``; no target fallback is
    ever used.
    """
    embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != int(dim):
        raise ValueError(
            f"candidate_embeddings must be [N,{dim}], got {embeddings.shape}"
        )
    tokens = np.zeros((int(k), int(dim)), dtype=np.float32)
    indices = [int(index) for index in top_indices[: int(k)]]
    for slot, index in enumerate(indices):
        if 0 <= index < len(embeddings):
            tokens[slot] = embeddings[index]
    return tokens, len(indices)


def map_features_from_scores(
    pose,
    centers: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int = MAP_TOP_K,
    scale_m: float = MAP_POSITION_SCALE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [22] relative retrieval features from predicted top-k candidates.

    Only candidate positions selected by retrieval scores are used.  No
    absolute target coordinate and no object ID is read.
    """
    points = np.asarray(centers, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(points) != len(scores):
        raise ValueError("centers and scores must have the same length")
    order = topk_indices(scores, int(top_k))
    features = np.zeros((4 * int(top_k) + 2,), dtype=np.float32)
    for slot, index in enumerate(order):
        nx, ny = float(points[index, 0]), float(points[index, 1])
        dx = nx - float(pose.x)
        dy = ny - float(pose.y)
        forward = (dx * np.cos(pose.yaw) + dy * np.sin(pose.yaw)) / float(scale_m)
        right = (-dx * np.sin(pose.yaw) + dy * np.cos(pose.yaw)) / float(scale_m)
        distance = float(np.hypot(dx, dy) / float(scale_m))
        features[4 * slot:4 * slot + 4] = [
            float(scores[index]), forward, right, distance
        ]
    if len(order):
        features[-2] = float(scores[order[0]])
        features[-1] = (
            float(scores[order[0]] - scores[order[1]])
            if len(order) > 1 else 0.0
        )
    return features, order


def recall_at_k(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
) -> dict:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive_mask, dtype=bool).reshape(-1)
    if len(scores) != len(positive):
        raise ValueError("scores and positive_mask must have the same length")
    if not positive.any():
        return {f"recall@{int(k)}": None for k in ks} | {"positive": 0}
    output = {"positive": int(positive.sum())}
    for k in ks:
        top = topk_indices(scores, int(k))
        output[f"recall@{int(k)}"] = bool(positive[top].any())
    return output
