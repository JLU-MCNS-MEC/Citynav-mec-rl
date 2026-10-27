"""Episode-level CityNav dataset with strict split and asset checks.

The reader is intentionally independent from ``processed_citynav``: it reads the
official ``original_citynav`` JSON files only.  Test-Unseen is blocked unless the
caller passes ``allow_test_unseen=True``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.config import DataConfig, ObservationConfig, TeacherConfig
from src.data.mapdata import GROUND_LEVEL
from src.data.trajectory_adapter import (
    Pose5D,
    TeacherResult,
    TrajectoryAdapter,
    load_trajectory,
)


CITYNAV_SPLITS = ("train_seen", "val_seen", "val_unseen", "test_unseen")


@lru_cache(maxsize=4)
def _load_json_cached(path: str) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path | str) -> object:
    return _load_json_cached(str(Path(path).resolve()))


@lru_cache(maxsize=4)
def load_cityrefer_objects(path: str) -> dict:
    return load_json(path)


@lru_cache(maxsize=4)
def load_processed_descriptions(path: str) -> dict:
    return load_json(path)


@dataclass
class EpisodeRecord:
    episode_id: str
    split: str
    area: str
    block: int
    map_name: str
    object_ids: tuple[int, ...]
    ann_ids: tuple[int, ...]
    descriptions: tuple[str, ...]
    description: str
    description_match: bool
    description_match_normalized: bool
    trajectory_raw: tuple[tuple[float, ...], ...]
    marker_positions: tuple[tuple[float, ...], ...]
    target_positions: tuple[tuple[float, ...], ...]
    total_score: float
    dist_marker_to_target: float
    poses: tuple[Pose5D, ...]
    start_pose: Pose5D
    target_xyz: tuple[float, float, float]
    target_xy: tuple[float, float]
    marker_xyz: tuple[float, float, float]
    ground_level_m: float
    object_info: dict | None
    processed_descriptions: tuple[dict, ...]
    target_object_distance_m: float | None
    _teacher_cache: TeacherResult | None = field(default=None, repr=False)

    def teacher(self, config: TeacherConfig | None = None) -> TeacherResult:
        if self._teacher_cache is None:
            self._teacher_cache = TrajectoryAdapter(config).rollout(
                self.poses, self.target_xy
            )
        return self._teacher_cache

    def supervision_steps(
        self, config: TeacherConfig | None = None
    ) -> list[tuple[Pose5D, int]]:
        return self.teacher(config).supervision_steps()

    def as_summary(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "split": self.split,
            "map_name": self.map_name,
            "object_id": int(self.object_ids[0]),
            "ann_id": int(self.ann_ids[0]),
            "instruction": self.description,
            "start_pose": [round(v, 3) for v in self.start_pose.as_tuple()],
            "target_xy": [round(v, 3) for v in self.target_xy],
            "trajectory_points": len(self.trajectory_raw),
            "description_match": bool(self.description_match),
            "description_match_normalized": bool(self.description_match_normalized),
            "target_object_distance_m": (
                None
                if self.target_object_distance_m is None
                else round(float(self.target_object_distance_m), 3)
            ),
        }


class CityReferIndex:
    def __init__(self, objects_path: Path, descriptions_path: Path | None = None):
        self.objects_path = Path(objects_path)
        self.objects: dict = load_cityrefer_objects(str(self.objects_path.resolve()))
        self.descriptions: dict = {}
        if descriptions_path is not None and Path(descriptions_path).exists():
            self.descriptions = load_processed_descriptions(
                str(Path(descriptions_path).resolve())
            )

    @property
    def map_names(self) -> list[str]:
        return sorted(self.objects)

    @property
    def object_count(self) -> int:
        return sum(len(values) for values in self.objects.values())

    def get(self, map_name: str, object_id: int) -> dict | None:
        return self.objects.get(map_name, {}).get(str(int(object_id)))

    def processed(self, map_name: str, object_id: int) -> list[dict]:
        return self.descriptions.get(map_name, {}).get(str(int(object_id)), [])


def _stable_episode_id(
    map_name: str,
    object_id: int,
    ann_id: int,
    start_pose: Pose5D,
    description: str,
) -> str:
    payload = (
        f"{map_name}|{int(object_id)}|{int(ann_id)}|"
        f"{start_pose.x:.3f}|{start_pose.y:.3f}|{start_pose.z:.3f}|"
        f"{start_pose.yaw:.5f}|{description}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def _point3(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) < 3:
        raise ValueError(f"expected 3 coordinates, got {value}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _normalize_description(text: str) -> str:
    """Normalize whitespace and ASCII double quotes for annotation checks.

    CityNav split files and CityRefer ``objects.json`` occasionally differ only
    by doubled spaces or by quotes around a single letter/phrase (for example
    ``"L"`` versus ``L``, or ``"L",`` versus ``L,``).  Quotes are removed rather
    than replaced with a space so punctuation-adjacent cases still match.  The
    raw text is kept as-is for language input; this normalized form is only used
    to audit the annotation index.
    """
    return " ".join(str(text).replace('"', "").split())


def build_episode_record(
    raw: dict,
    *,
    split: str,
    cityrefer: CityReferIndex,
    observation: ObservationConfig,
    index: int | None = None,
) -> EpisodeRecord:
    for key in (
        "area", "block", "object_ids", "ann_ids", "descriptions",
        "trajectory", "marker_positions", "target_positions", "split",
    ):
        if key not in raw:
            raise ValueError(f"episode {index}: missing field {key}")

    if str(raw["split"]) != split:
        raise ValueError(
            f"episode {index}: split field {raw['split']!r} does not match {split!r}"
        )

    object_ids = tuple(int(v) for v in raw["object_ids"])
    ann_ids = tuple(int(v) for v in raw["ann_ids"])
    descriptions = tuple(str(v) for v in raw["descriptions"])
    if not object_ids or not ann_ids or not descriptions:
        raise ValueError(f"episode {index}: empty object/ann/description list")
    if not (len(object_ids) == len(ann_ids) == len(descriptions)):
        raise ValueError(
            f"episode {index}: object_ids/ann_ids/descriptions length mismatch"
        )

    area = str(raw["area"])
    block = int(raw["block"])
    map_name = f"{area}_block_{block}"
    if map_name not in GROUND_LEVEL:
        raise ValueError(f"episode {index}: no GROUND_LEVEL entry for {map_name}")

    trajectory_raw = tuple(tuple(float(v) for v in point) for point in raw["trajectory"])
    marker_positions = tuple(
        tuple(float(v) for v in point) for point in raw["marker_positions"]
    )
    target_positions = tuple(
        tuple(float(v) for v in point) for point in raw["target_positions"]
    )
    if not trajectory_raw or not target_positions:
        raise ValueError(f"episode {index}: empty trajectory or target_positions")

    ground_level = float(GROUND_LEVEL[map_name])
    poses = load_trajectory(
        trajectory_raw,
        fix_altitude_m=(
            float(observation.altitude_m) if observation.fix_altitude else None
        ),
        ground_level_m=ground_level if observation.fix_altitude else None,
    )
    start_pose = poses[0]
    target_xyz = _point3(target_positions[-1])
    marker_xyz = _point3(marker_positions[-1]) if marker_positions else target_xyz

    object_info = cityrefer.get(map_name, object_ids[0])
    description_match = False
    description_match_normalized = False
    if object_info is not None:
        object_descriptions = object_info.get("descriptions", []) or []
        ann_id = int(ann_ids[0])
        if 0 <= ann_id < len(object_descriptions):
            annotation_text = str(object_descriptions[ann_id])
            description_match = annotation_text == descriptions[0]
            description_match_normalized = _normalize_description(
                annotation_text
            ) == _normalize_description(descriptions[0])
    target_object_distance = None
    if object_info is not None and object_info.get("position"):
        target_object_distance = _xy_distance(
            target_xyz, object_info["position"]
        )

    processed = (
        tuple(cityrefer.processed(map_name, object_ids[0]))
        if cityrefer.descriptions
        else tuple()
    )

    return EpisodeRecord(
        episode_id=_stable_episode_id(
            map_name, object_ids[0], ann_ids[0], start_pose, descriptions[0]
        ),
        split=split,
        area=area,
        block=block,
        map_name=map_name,
        object_ids=object_ids,
        ann_ids=ann_ids,
        descriptions=descriptions,
        description=descriptions[0],
        description_match=bool(description_match),
        description_match_normalized=bool(description_match_normalized),
        trajectory_raw=trajectory_raw,
        marker_positions=marker_positions,
        target_positions=target_positions,
        total_score=float(raw.get("total_score", 0.0)),
        dist_marker_to_target=float(raw.get("dist_marker_to_target", 0.0)),
        poses=tuple(poses),
        start_pose=start_pose,
        target_xyz=target_xyz,
        target_xy=(target_xyz[0], target_xyz[1]),
        marker_xyz=marker_xyz,
        ground_level_m=ground_level,
        object_info=object_info,
        processed_descriptions=processed,
        target_object_distance_m=target_object_distance,
    )


class CityNavDataset:
    def __init__(
        self,
        data_config: DataConfig,
        split: str,
        *,
        observation_config: ObservationConfig | None = None,
        teacher_config: TeacherConfig | None = None,
        allow_test_unseen: bool = False,
        max_episodes: int | None = None,
        strict_objects: bool = True,
        include_processed_descriptions: bool = True,
    ):
        if split not in CITYNAV_SPLITS:
            raise ValueError(f"unknown split {split!r}")
        if split == "test_unseen" and not allow_test_unseen:
            raise PermissionError(
                "test_unseen is reserved for the final frozen evaluation; "
                "pass allow_test_unseen=True only for the final audit"
            )
        self.data_config = data_config
        self.split = split
        self.observation_config = observation_config or ObservationConfig()
        self.teacher_config = teacher_config or TeacherConfig()
        self.allow_test_unseen = allow_test_unseen

        self.cityrefer = CityReferIndex(
            data_config.cityrefer_dir / "objects.json",
            (
                data_config.cityrefer_dir / "processed_descriptions.json"
                if include_processed_descriptions
                else None
            ),
        )
        self.split_path = data_config.split_path(split)
        if not self.split_path.exists():
            raise FileNotFoundError(self.split_path)
        raw_records = load_json(self.split_path)
        if not isinstance(raw_records, list):
            raise ValueError(f"{self.split_path.name} is not a JSON list")

        self.records: list[EpisodeRecord] = []
        self.skipped: list[dict] = []
        for index, raw in enumerate(raw_records):
            if max_episodes is not None and len(self.records) >= int(max_episodes):
                break
            try:
                record = build_episode_record(
                    raw,
                    split=split,
                    cityrefer=self.cityrefer,
                    observation=self.observation_config,
                    index=index,
                )
                if strict_objects and record.object_info is None:
                    raise ValueError(
                        f"episode {index}: CityRefer object "
                        f"{record.object_ids[0]} not found in {record.map_name}"
                    )
                self.records.append(record)
            except Exception as exc:  # noqa: BLE001 - audit needs the reason
                if strict_objects:
                    raise
                self.skipped.append({"index": index, "error": str(exc)})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EpisodeRecord:
        return self.records[index]

    def summary(self) -> dict:
        maps = Counter(record.map_name for record in self.records)
        lengths = [len(record.trajectory_raw) for record in self.records]
        target_distances = [
            record.target_object_distance_m
            for record in self.records
            if record.target_object_distance_m is not None
        ]
        return {
            "split": self.split,
            "records": len(self.records),
            "skipped": len(self.skipped),
            "maps": len(maps),
            "map_counts": dict(sorted(maps.items())),
            "trajectory_points": {
                "min": int(min(lengths)) if lengths else 0,
                "median": float(np.median(lengths)) if lengths else 0.0,
                "max": int(max(lengths)) if lengths else 0,
            },
            "description_match": sum(
                int(record.description_match) for record in self.records
            ),
            "description_match_normalized": sum(
                int(record.description_match_normalized)
                for record in self.records
            ),
            "missing_objects": sum(
                int(record.object_info is None) for record in self.records
            ),
            "target_object_distance_m": {
                "min": round(float(min(target_distances)), 3) if target_distances else None,
                "median": (
                    round(float(np.median(target_distances)), 3)
                    if target_distances
                    else None
                ),
                "max": round(float(max(target_distances)), 3) if target_distances else None,
            },
        }

    def teacher_success_rate(
        self,
        *,
        limit: int | None = None,
        seed: int = 0,
    ) -> dict:
        records = list(self.records)
        if limit is not None and limit < len(records):
            rng = np.random.default_rng(seed)
            indices = np.sort(
                rng.choice(len(records), size=int(limit), replace=False)
            )
            records = [records[int(i)] for i in indices]
        results = []
        for record in records:
            result = record.teacher(self.teacher_config)
            results.append(result)
        if not results:
            return {"sampled": 0}
        steps = [result.steps for result in results]
        cross_track = [result.cross_track_median_m for result in results]
        final = [result.final_distance_m for result in results]
        return {
            "sampled": len(results),
            "success_rate": round(
                sum(int(result.success) for result in results) / len(results), 4
            ),
            "steps_median": int(np.median(steps)),
            "steps_p90": int(np.percentile(steps, 90)),
            "final_distance_m_median": round(float(np.median(final)), 3),
            "final_distance_m_p90": round(float(np.percentile(final, 90)), 3),
            "cross_track_median_m": round(float(np.median(cross_track)), 3),
            "wrong_stop_reasons": dict(
                Counter(
                    result.stop_reason
                    for result in results
                    if not result.success
                )
            ),
        }


def count_split_records(
    data_config: DataConfig,
    splits: Iterable[str] = CITYNAV_SPLITS,
) -> dict[str, int]:
    counts = {}
    for split in splits:
        path = data_config.split_path(split)
        if not path.exists():
            counts[split] = -1
            continue
        payload = load_json(path)
        counts[split] = len(payload) if isinstance(payload, list) else -1
    return counts


def map_names_for_splits(
    data_config: DataConfig,
    splits: Iterable[str] = CITYNAV_SPLITS,
) -> list[str]:
    names: set[str] = set()
    for split in splits:
        path = data_config.split_path(split)
        if not path.exists():
            continue
        for raw in load_json(path):
            names.add(f"{raw['area']}_block_{raw['block']}")
    return sorted(names)


def missing_map_assets(data_config: DataConfig, map_names: Iterable[str]) -> dict:
    missing: dict[str, list[str]] = {}
    for map_name in sorted(set(map_names)):
        absent = []
        if not data_config.rgb_path(map_name).exists():
            absent.append("png")
        if not data_config.height_path(map_name).exists():
            absent.append("tif")
        if not data_config.ply_path(map_name).exists():
            absent.append("ply")
        if absent:
            missing[map_name] = absent
    return missing
