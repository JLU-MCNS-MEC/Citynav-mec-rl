"""Dataset, split and CityRefer annotation contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import DataConfig, ObservationConfig
from src.data.citynav_dataset import (
    CityNavDataset,
    build_episode_record,
    count_split_records,
    map_names_for_splits,
)


class FakeCityRefer:
    def __init__(self, entry: dict | None):
        self.entry = entry
        self.descriptions: dict = {}

    def get(self, map_name: str, object_id: int) -> dict | None:
        return self.entry

    def processed(self, map_name: str, object_id: int) -> list[dict]:
        return []


def _raw(split: str = "val_seen") -> dict:
    return {
        "area": "birmingham",
        "block": 0,
        "object_ids": [1],
        "ann_ids": [0],
        "descriptions": ["Find the target building."],
        "trajectory": [
            [0.0, 0.0, 20.0, 1.0, 0.0, 0.0],
            [10.0, 0.0, 20.0, 1.0, 0.0, 0.0],
        ],
        "marker_positions": [[10.0, 0.0, 20.0]],
        "target_positions": [[10.0, 0.0, 20.0]],
        "split": split,
        "total_score": 1.0,
        "dist_marker_to_target": 0.0,
    }


def _entry(description: str = "Find the target building.") -> dict:
    return {
        "descriptions": [description],
        "position": [10.0, 0.0, 20.0],
        "dimension": [10.0, 10.0, 5.0],
        "name": "",
        "object_type": "Building",
        "contour": [],
        "id": 1,
        "map_name": "birmingham_block_0",
    }


def test_build_episode_record_maps_fields_and_description() -> None:
    record = build_episode_record(
        _raw(),
        split="val_seen",
        cityrefer=FakeCityRefer(_entry()),
        observation=ObservationConfig(altitude_m=50.0, fix_altitude=True),
        index=0,
    )
    assert record.map_name == "birmingham_block_0"
    assert record.description == "Find the target building."
    assert record.description_match is True
    assert record.description_match_normalized is True
    assert record.start_pose.z == pytest.approx(record.ground_level_m + 50.0)
    assert record.target_xy == pytest.approx((10.0, 0.0))
    assert record.target_object_distance_m == pytest.approx(0.0)
    assert record.teacher().success is True


def test_description_mismatch_is_reported_not_hidden() -> None:
    record = build_episode_record(
        _raw(),
        split="val_seen",
        cityrefer=FakeCityRefer(_entry("A different description.")),
        observation=ObservationConfig(),
        index=0,
    )
    assert record.description_match is False
    assert record.description_match_normalized is False


def test_description_whitespace_difference_is_visible_but_normalized() -> None:
    record = build_episode_record(
        _raw(),
        split="val_seen",
        cityrefer=FakeCityRefer(_entry("Find  the target  building.")),
        observation=ObservationConfig(),
        index=0,
    )
    assert record.description_match is False
    assert record.description_match_normalized is True


def test_description_quote_difference_is_visible_but_normalized() -> None:
    record = build_episode_record(
        _raw(),
        split="val_seen",
        cityrefer=FakeCityRefer(_entry('Find the "target" building.')),
        observation=ObservationConfig(),
        index=0,
    )
    assert record.description_match is False
    assert record.description_match_normalized is True


def test_missing_object_is_tolerated_by_the_record_builder() -> None:
    record = build_episode_record(
        _raw(),
        split="val_seen",
        cityrefer=FakeCityRefer(None),
        observation=ObservationConfig(),
        index=0,
    )
    assert record.object_info is None
    assert record.description_match is False
    assert record.description_match_normalized is False
    assert record.target_object_distance_m is None


def test_split_field_must_match_requested_split() -> None:
    with pytest.raises(ValueError, match="split field"):
        build_episode_record(
            _raw("val_seen"),
            split="val_unseen",
            cityrefer=FakeCityRefer(_entry()),
            observation=ObservationConfig(),
            index=0,
        )


def test_annotation_lists_must_have_equal_lengths() -> None:
    raw = _raw()
    raw["ann_ids"] = [0, 1]
    with pytest.raises(ValueError, match="length mismatch"):
        build_episode_record(
            raw,
            split="val_seen",
            cityrefer=FakeCityRefer(_entry()),
            observation=ObservationConfig(),
            index=0,
        )


def test_test_unseen_is_blocked_without_explicit_flag(tmp_path: Path) -> None:
    config = DataConfig(data_root=tmp_path / "data", ply_root=tmp_path / "ply")
    with pytest.raises(PermissionError):
        CityNavDataset(config, "test_unseen")


def test_dataset_reads_a_minimal_cityrefer_tree(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    citynav_dir = data_root / "original_citynav"
    cityrefer_dir = data_root / "cityrefer"
    citynav_dir.mkdir(parents=True)
    cityrefer_dir.mkdir(parents=True)

    objects = {"birmingham_block_0": {"1": _entry()}}
    (cityrefer_dir / "objects.json").write_text(
        json.dumps(objects), encoding="utf-8"
    )
    (citynav_dir / "citynav_val_seen.json").write_text(
        json.dumps([_raw()]), encoding="utf-8"
    )

    config = DataConfig(data_root=data_root, ply_root=tmp_path / "ply")
    dataset = CityNavDataset(config, "val_seen")

    assert len(dataset) == 1
    summary = dataset.summary()
    assert summary["records"] == 1
    assert summary["maps"] == 1
    assert summary["description_match"] == 1
    assert summary["description_match_normalized"] == 1
    assert summary["missing_objects"] == 0
    assert count_split_records(config, ["val_seen"]) == {"val_seen": 1}
    assert map_names_for_splits(config, ["val_seen"]) == ["birmingham_block_0"]


def test_strict_objects_rejects_unknown_cityrefer_objects(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "original_citynav").mkdir(parents=True)
    (data_root / "cityrefer").mkdir(parents=True)
    (data_root / "cityrefer" / "objects.json").write_text("{}", encoding="utf-8")
    (data_root / "original_citynav" / "citynav_val_seen.json").write_text(
        json.dumps([_raw()]), encoding="utf-8"
    )
    config = DataConfig(data_root=data_root, ply_root=tmp_path / "ply")
    with pytest.raises(ValueError, match="CityRefer object"):
        CityNavDataset(config, "val_seen", strict_objects=True)
    relaxed = CityNavDataset(config, "val_seen", strict_objects=False)
    assert len(relaxed) == 1
    assert relaxed.summary()["missing_objects"] == 1
