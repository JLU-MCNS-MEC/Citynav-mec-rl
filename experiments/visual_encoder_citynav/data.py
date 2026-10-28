"""Data and rendering utilities for CityNav visual-encoder pretraining.

Three supervision sources are kept separated:

* action labels from the official six-action teacher;
* cross-view matching near the *current* UAV pose;
* language-to-map retrieval labels near the *target* position.

Only the last two need map candidates.  The current-pose positive mask and
the target-position positive mask have different names and are never mixed.

Raw CityNav/CityRefer/RGB/D PLY data are read-only.  Derived point-cloud
downsamples, candidate libraries, manifests and language features live under
``data_cache/visual_encoder_citynav/`` and are never committed.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.config import DataConfig, ObservationConfig, PROJECT_ROOT, TextConfig
from src.data.actions import ACTION_NAMES, ACTION_ORDER
from src.data.candidates import (
    CANDIDATE_POSITIVE_RADIUS_M,
    CANDIDATE_SCALES_M,
    CANDIDATE_STRIDE_M,
    CandidateLibrary,
    build_candidate_library,
)
from src.data.citynav_dataset import CityNavDataset
from src.data.fpv_renderer import (
    camera_intrinsics,
    camera_pitch,
    render_point_cloud,
)
from src.data.language_adapter import build_language_adapter
from src.data.observation_adapter import open_ply_cloud
from src.data.paths import ensure_output_dir
from src.data.trajectory_adapter import Pose5D, TrajectoryAdapter


CURRENT_SIZE = 336
CANDIDATE_SIZE = 96
CANDIDATE_K = 16
MAX_POINTS_DEFAULT = 4_000_000
CURRENT_MATCH_RADIUS_M = 30.0
TARGET_MATCH_RADIUS_M = 40.0
HARD_NEGATIVE_MAX_M = 120.0
MAX_CURRENT_POSITIVES = 4
MAX_TARGET_POSITIVES = 4
MAX_HARD_NEGATIVES = 8
RENDER_NEAR_M = 0.5
RENDER_FAR_M = 250.0


def _cloud_path(map_name: str, max_points: int) -> Path:
    root = PROJECT_ROOT / "data_cache" / "visual_encoder_citynav" / "clouds"
    return root / f"{map_name}_{int(max_points)}.npz"


def build_point_cloud_cache(
    data_config: DataConfig,
    map_name: str,
    *,
    max_points: int = MAX_POINTS_DEFAULT,
    rebuild: bool = False,
) -> dict:
    """Create a small deterministic XYZRGB downsample of one raw PLY."""
    path = _cloud_path(map_name, int(max_points))
    if path.exists() and not rebuild:
        payload = np.load(path)
        return {
            "xyz": payload["xyz"].astype(np.float32),
            "rgb": payload["rgb"].astype(np.uint8),
        }
    cloud = open_ply_cloud(data_config.ply_path(map_name))
    total = int(cloud.header.vertex_count)
    stride = max(1, int(math.ceil(total / max(1, int(max_points)))))
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    chunk = 4_000_000
    for start in range(0, total, chunk):
        block = np.asarray(cloud.data[start:start + chunk])
        offset = (-start) % stride
        sub = block[offset::stride]
        if len(sub) == 0:
            continue
        xyz_parts.append(
            np.stack(
                [sub["x"], sub["y"], sub["z"]], axis=1
            ).astype(np.float32)
        )
        rgb_parts.append(
            np.stack(
                [sub["red"], sub["green"], sub["blue"]], axis=1
            ).astype(np.uint8)
        )
    xyz = np.concatenate(xyz_parts, axis=0) if xyz_parts else np.empty((0, 3), np.float32)
    rgb = np.concatenate(rgb_parts, axis=0) if rgb_parts else np.empty((0, 3), np.uint8)
    if len(xyz) > int(max_points):
        keep = np.linspace(0, len(xyz) - 1, int(max_points), dtype=int)
        xyz, rgb = xyz[keep], rgb[keep]
    ensure_output_dir(path.parent, data_config=data_config)
    np.savez_compressed(path, xyz=xyz, rgb=rgb)
    return {"xyz": xyz, "rgb": rgb}


class PointCloudStore:
    def __init__(
        self,
        data_config: DataConfig,
        max_points: int = MAX_POINTS_DEFAULT,
        *,
        rebuild: bool = False,
    ):
        self.data_config = data_config
        self.max_points = int(max_points)
        self.rebuild = bool(rebuild)
        self._cache: dict[str, dict] = {}

    def get(self, map_name: str) -> dict:
        if map_name not in self._cache:
            self._cache[map_name] = build_point_cloud_cache(
                self.data_config,
                map_name,
                max_points=self.max_points,
                rebuild=self.rebuild,
            )
        return self._cache[map_name]


class CandidateStore:
    def __init__(
        self,
        data_config: DataConfig,
        observation_config: ObservationConfig,
        *,
        stride_m: float = CANDIDATE_STRIDE_M,
        scales_m: Sequence[float] = CANDIDATE_SCALES_M,
        rebuild: bool = False,
    ):
        self.data_config = data_config
        self.observation_config = observation_config
        self.stride_m = float(stride_m)
        self.scales_m = tuple(float(v) for v in scales_m)
        self.rebuild = bool(rebuild)
        self._cache: dict[str, CandidateLibrary] = {}

    def get(self, map_name: str) -> CandidateLibrary:
        if map_name not in self._cache:
            self._cache[map_name] = build_candidate_library(
                self.data_config,
                map_name,
                self.observation_config,
                encoder=None,
                stride_m=self.stride_m,
                scales_m=self.scales_m,
                rebuild=self.rebuild,
            )
        return self._cache[map_name]


def render_current_rgb(
    points: dict,
    pose: Pose5D,
    observation_config: ObservationConfig,
) -> tuple[np.ndarray, float]:
    camera = camera_intrinsics(observation_config)
    result = render_point_cloud(
        points["xyz"],
        points["rgb"],
        (pose.x, pose.y, pose.z),
        pose.yaw,
        camera_pitch(pose, observation_config),
        camera,
        near=RENDER_NEAR_M,
        far=RENDER_FAR_M,
        # Adaptive point size: 0 = one pixel per point (baseline), >=1 fills a
        # (2r+1)^2 neighbourhood.  This only widens real points; it never
        # synthesises geometry for empty regions.
        splat_radius=int(getattr(observation_config, "fpv_splat_radius", 0)),
    )
    return result.rgb, float(result.valid_pixel_ratio)


def select_candidate_subset(
    centers: np.ndarray,
    current_xy: Sequence[float],
    target_xy: Sequence[float],
    *,
    rng: np.random.Generator,
    current_radius: float = CURRENT_MATCH_RADIUS_M,
    target_radius: float = TARGET_MATCH_RADIUS_M,
    k: int = CANDIDATE_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate indices and strictly separate task masks.

    ``current_mask``/``current_valid`` are built only from the current UAV
    position; ``target_mask``/``target_valid`` only from the target label.
    The two task denominators are therefore disjoint even when one image is
    shared in the union.  This is training data preparation only; neither
    position is passed to the model.
    """
    points = np.asarray(centers, dtype=np.float64)
    current = np.asarray(current_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    d_current = np.linalg.norm(points - current[None, :], axis=1)
    d_target = np.linalg.norm(points - target[None, :], axis=1)

    current_pos = np.flatnonzero(d_current <= float(current_radius))
    target_pos = np.flatnonzero(d_target <= float(target_radius))
    current_pos = current_pos[np.argsort(d_current[current_pos])]
    target_pos = target_pos[np.argsort(d_target[target_pos])]
    current_pos = current_pos[:MAX_CURRENT_POSITIVES]
    target_pos = target_pos[:MAX_TARGET_POSITIVES]

    current_hard = [
        int(i) for i in range(len(points))
        if int(i) not in set(int(v) for v in current_pos)
        and float(current_radius) < d_current[i] <= HARD_NEGATIVE_MAX_M
    ]
    target_hard = [
        int(i) for i in range(len(points))
        if int(i) not in set(int(v) for v in target_pos)
        and float(target_radius) < d_target[i] <= HARD_NEGATIVE_MAX_M
    ]
    current_hard.sort(key=lambda i: d_current[i])
    target_hard.sort(key=lambda i: d_target[i])
    current_hard = current_hard[:MAX_HARD_NEGATIVES // 2]
    target_hard = target_hard[:MAX_HARD_NEGATIVES // 2]
    chosen = list(dict.fromkeys(
        [int(v) for v in current_pos]
        + current_hard
        + [int(v) for v in target_pos]
        + target_hard
    ))
    if len(chosen) < int(k):
        pool = [i for i in range(len(points)) if i not in set(chosen)]
        if pool:
            extra = rng.choice(
                np.asarray(pool),
                size=min(int(k) - len(chosen), len(pool)),
                replace=False,
            )
            chosen.extend(int(v) for v in extra)
    chosen = chosen[:int(k)]

    current_pos_set = set(int(v) for v in current_pos)
    target_pos_set = set(int(v) for v in target_pos)
    current_valid_set = current_pos_set | set(int(v) for v in current_hard)
    target_valid_set = target_pos_set | set(int(v) for v in target_hard)
    current_mask = np.zeros((int(k),), dtype=np.float32)
    target_mask = np.zeros((int(k),), dtype=np.float32)
    current_valid = np.zeros((int(k),), dtype=np.float32)
    target_valid = np.zeros((int(k),), dtype=np.float32)
    for slot, index in enumerate(chosen):
        current_mask[slot] = 1.0 if index in current_pos_set else 0.0
        target_mask[slot] = 1.0 if index in target_pos_set else 0.0
        current_valid[slot] = 1.0 if index in current_valid_set else 0.0
        target_valid[slot] = 1.0 if index in target_valid_set else 0.0
    return (
        np.asarray(chosen, dtype=np.int64),
        current_mask,
        target_mask,
        current_valid,
        target_valid,
    )


def _select_step(
    actions: Sequence[int],
    desired_action: int,
    index: int,
) -> int:
    matches = [i for i, action in enumerate(actions) if int(action) == int(desired_action)]
    if matches:
        return int(matches[len(matches) // 2])
    # Keep the actual label; fall back to a deterministic trajectory step.
    return int(min(max(1, len(actions) // 3), len(actions) - 1))


def _select_steps(
    actions: Sequence[int],
    desired_action: int,
    index: int,
    count: int,
    seed: int,
) -> list[int]:
    """Choose several teacher steps per trajectory (E3/E4 only).

    The first step is the baseline selection, so E3/E4 stay comparable to E1.
    Additional steps are drawn deterministically from:

    * action-transition positions (``a[i] != a[i-1]``), which cover the moment
      a turn/vertical command starts;
    * per-class positions spread across early / middle / late thirds of the
      trajectory, which raises directional and vertical coverage without
      touching labels.

    Nothing is synthesised: every step exists in the official teacher rollout.
    """
    if int(count) <= 1 or len(actions) <= 1:
        return [_select_step(actions, desired_action, index)]

    base = _select_step(actions, desired_action, index)
    chosen: list[int] = [int(base)]
    rng = np.random.default_rng(int(seed) + int(index))

    transitions = [
        i for i in range(1, len(actions)) if int(actions[i]) != int(actions[i - 1])
    ]
    thirds = [
        list(range(0, max(1, len(actions) // 3))),
        list(range(max(1, len(actions) // 3), max(2, 2 * len(actions) // 3))),
        list(range(max(2, 2 * len(actions) // 3), len(actions))),
    ]
    per_class: dict[int, list[int]] = {}
    for i, action in enumerate(actions):
        per_class.setdefault(int(action), []).append(i)

    def _spread(indices: list[int], k: int) -> list[int]:
        if not indices:
            return []
        picks = np.linspace(0, len(indices) - 1, min(k, len(indices)), dtype=int)
        return [int(indices[p]) for p in picks]

    candidates: list[int] = []
    # transitions first (turn/vertical onset), then class coverage per third
    if transitions:
        candidates.extend(_spread(transitions, max(1, count)))
    for action in sorted(per_class, key=lambda a: (len(per_class[a]), a)):
        candidates.extend(_spread(per_class[action], max(1, count // 2)))
    for third in thirds:
        candidates.extend(_spread(third, 1))
    rng.shuffle(candidates)

    for candidate in candidates:
        if len(chosen) >= int(count):
            break
        if candidate not in chosen:
            chosen.append(int(candidate))
    # deterministic fill if the trajectory is short
    filler = list(range(len(actions)))
    rng.shuffle(filler)
    for candidate in filler:
        if len(chosen) >= int(count):
            break
        if candidate not in chosen:
            chosen.append(int(candidate))
    return sorted(chosen[: int(count)])


def build_manifest(
    data_config: DataConfig,
    split: str,
    *,
    max_records: int | None = None,
    seed: int = 0,
    language_backend: str = "bert",
    output_dir: Path | None = None,
    rebuild: bool = False,
    steps_per_trajectory: int = 1,
) -> dict:
    """Build a teacher-step manifest.

    ``steps_per_trajectory=1`` reproduces the E1 baseline exactly.  Values > 1
    are only used by the E3/E4 ablations; validation manifests always stay at 1.
    """
    if split == "test_unseen":
        raise PermissionError("test_unseen is sealed")
    output_dir = output_dir or (
        PROJECT_ROOT / "data_cache" / "visual_encoder_citynav" / "manifests"
    )
    output_dir = ensure_output_dir(output_dir, data_config=data_config)
    steps_per_trajectory = max(1, int(steps_per_trajectory))
    tag = f"{split}_{'all' if max_records is None else int(max_records)}_{seed}"
    if steps_per_trajectory > 1:
        tag += f"_s{steps_per_trajectory}"
    npz_path = output_dir / f"{tag}.npz"
    json_path = output_dir / f"{tag}.json"
    if npz_path.exists() and json_path.exists() and not rebuild:
        payload = np.load(npz_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        return {
            **{key: payload[key] for key in payload.files},
            **meta,
        }

    dataset = CityNavDataset(
        data_config, split, include_processed_descriptions=False
    )
    records = dataset.records
    if max_records is not None and int(max_records) < len(records):
        indices = np.linspace(0, len(records) - 1, int(max_records), dtype=int)
        records = [records[int(i)] for i in indices]
    adapter = TrajectoryAdapter(dataset.teacher_config)
    rows = []
    for index, record in enumerate(records):
        result = adapter.rollout(
            record.poses, record.marker_xyz, record.target_xyz
        )
        actions = [int(v) for v in result.actions]
        if not actions:
            continue
        desired = int(ACTION_ORDER[index % len(ACTION_ORDER)])
        if steps_per_trajectory > 1:
            steps = _select_steps(
                actions, desired, index, steps_per_trajectory, seed
            )
        else:
            steps = [_select_step(actions, desired, index)]
        for step in steps:
            pose = result.poses[step]
            rows.append(
                {
                    "map_name": record.map_name,
                    "pose": pose,
                    "target_xy": tuple(float(v) for v in record.target_xy),
                    "action": int(actions[step]),
                    "description": record.description,
                    "step": int(step),
                    "teacher_steps": int(len(actions)),
                    "desired_action": int(desired),
                }
            )
    map_names = sorted({row["map_name"] for row in rows})
    map_to_id = {name: index for index, name in enumerate(map_names)}
    language_adapter, text_dim, backend, warning = build_language_adapter(
        TextConfig(backend=language_backend, local_files_only=True),
        device="cpu",
    )
    features = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            batch = [row["description"] for row in rows[start:start + 64]]
            features.append(
                language_adapter.encode_tensor(batch, device="cpu").cpu().numpy()
            )
    language = (
        np.concatenate(features, axis=0).astype(np.float32)
        if features else np.zeros((0, int(text_dim)), dtype=np.float32)
    )
    language_adapter.close()
    arrays = {
        "map_id": np.asarray(
            [map_to_id[row["map_name"]] for row in rows], dtype=np.int64
        ),
        "pose": np.asarray(
            [row["pose"].as_tuple() for row in rows], dtype=np.float32
        ),
        "target_xy": np.asarray(
            [row["target_xy"] for row in rows], dtype=np.float32
        ),
        "action": np.asarray([row["action"] for row in rows], dtype=np.int64),
        "step": np.asarray([row["step"] for row in rows], dtype=np.int64),
        "teacher_steps": np.asarray(
            [row["teacher_steps"] for row in rows], dtype=np.int64
        ),
        "desired_action": np.asarray(
            [row["desired_action"] for row in rows], dtype=np.int64
        ),
        "language_features": language,
    }
    np.savez_compressed(npz_path, **arrays)
    action_counts = Counter(int(v) for v in arrays["action"])
    metadata = {
        "split": split,
        "records": len(rows),
        "available_records": len(dataset.records),
        "maps": len(map_names),
        "map_names": map_names,
        "action_counts": {
            ACTION_NAMES[action]: int(action_counts.get(int(action), 0))
            for action in ACTION_ORDER
        },
        "desired_action_counts": {
            ACTION_NAMES[action]: int(np.sum(arrays["desired_action"] == int(action)))
            for action in ACTION_ORDER
        },
        "language_backend": backend,
        "language_dim": int(text_dim),
        "language_warning": warning,
        "manifest_file": str(npz_path.relative_to(PROJECT_ROOT)),
        "test_unseen_read": False,
        "steps_per_trajectory": int(steps_per_trajectory),
        "selection": (
            "one teacher step per trajectory; desired six-action cycle; actual label kept"
            if steps_per_trajectory == 1
            else "multiple teacher steps per trajectory: baseline step + action "
            "transitions + per-class early/middle/late spread; actual labels kept"
        ),
    }
    json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {**arrays, **metadata}


def load_manifest(path: Path) -> dict:
    path = Path(path)
    npz = np.load(path.with_suffix(".npz"))
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return {**{key: npz[key] for key in npz.files}, **meta}


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest: dict,
        data_config: DataConfig,
        observation_config: ObservationConfig,
        candidate_store: CandidateStore,
        cloud_store: PointCloudStore,
        *,
        seed: int = 0,
    ):
        self.manifest = manifest
        self.data_config = data_config
        self.observation_config = observation_config
        self.candidate_store = candidate_store
        self.cloud_store = cloud_store
        self.seed = int(seed)
        self.map_names = list(manifest["map_names"])
        self.pose = np.asarray(manifest["pose"], dtype=np.float64)
        self.target_xy = np.asarray(manifest["target_xy"], dtype=np.float64)
        self.map_id = np.asarray(manifest["map_id"], dtype=np.int64)
        self.action = np.asarray(manifest["action"], dtype=np.int64)
        self.language = np.asarray(manifest["language_features"], dtype=np.float32)
        self.step = np.asarray(manifest["step"], dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.action))

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        map_name = self.map_names[int(self.map_id[index])]
        pose = Pose5D(*[float(v) for v in self.pose[index]])
        points = self.cloud_store.get(map_name)
        render_start = time.perf_counter()
        current_rgb, valid_ratio = render_current_rgb(
            points, pose, self.observation_config
        )
        render_seconds = time.perf_counter() - render_start
        library = self.candidate_store.get(map_name)
        rng = np.random.default_rng(self.seed + index)
        (
            indices, current_mask, target_mask,
            current_valid, target_valid,
        ) = select_candidate_subset(
            library.centers,
            pose.xy,
            self.target_xy[index],
            rng=rng,
        )
        images = np.zeros(
            (CANDIDATE_K, CANDIDATE_SIZE, CANDIDATE_SIZE, 3), dtype=np.uint8
        )
        for slot, candidate_index in enumerate(indices):
            if 0 <= int(candidate_index) < library.count:
                images[slot] = library.images[int(candidate_index)]
        return {
            "current_rgb": current_rgb.astype(np.uint8),
            "candidate_rgbs": images,
            "current_mask": current_mask.astype(np.float32),
            "target_mask": target_mask.astype(np.float32),
            "current_valid": current_valid.astype(np.float32),
            "target_valid": target_valid.astype(np.float32),
            "action": int(self.action[index]),
            "language": self.language[index].astype(np.float32),
            "map_id": int(self.map_id[index]),
            "sample_index": index,
            "valid_pixel_ratio": float(valid_ratio),
            "render_seconds": float(render_seconds),
        }
