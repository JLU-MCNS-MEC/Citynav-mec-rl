"""Configuration expansion and output-path isolation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import DataConfig, load_yaml
from src.data.paths import display_path, ensure_output_dir, is_within


def test_load_yaml_expands_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CITYNAV_TEST_ROOT", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text("root: ${CITYNAV_TEST_ROOT}\n", encoding="utf-8")
    assert load_yaml(path)["root"] == str(tmp_path)


def test_load_yaml_rejects_unresolved_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CITYNAV_TEST_MISSING", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text("root: ${CITYNAV_TEST_MISSING}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unresolved environment"):
        load_yaml(path)


def test_data_config_requires_both_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data_root and ply_root"):
        DataConfig.from_mapping({"data_root": str(tmp_path)})


def test_display_path_redacts_raw_roots(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    raw_file = data_root / "original_citynav" / "citynav_val_seen.json"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("[]", encoding="utf-8")
    config = DataConfig(data_root=data_root, ply_root=tmp_path / "ply")

    assert display_path(raw_file, config) == (
        "<CITYNAV_DATA_ROOT>/original_citynav/citynav_val_seen.json"
    )
    assert display_path(tmp_path / "elsewhere.json", config) == "<outside-project>"


def test_output_directory_must_stay_inside_project(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside project"):
        ensure_output_dir(tmp_path / "outside")


def test_is_within_checks_resolved_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    child = root / "child"
    child.mkdir()
    assert is_within(child, root)
    assert not is_within(tmp_path / "other", root)
