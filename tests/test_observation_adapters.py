"""Observation geometry and PLY/raster audit contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import DataConfig
from src.data.observation_adapter import (
    audit_ply_assets,
    read_first_ply_vertices,
    read_ply_header,
)
from src.data.trajectory_adapter import Pose5D
from src.models.token_utils import images_to_tensor


def test_view_area_corners_match_official_citynav_geometry() -> None:
    from src.data.observation_adapter import TopDownObservationAdapter

    corners = TopDownObservationAdapter._view_area_corners(
        Pose5D(0.0, 0.0, 60.0, 0.0, 0.0), ground_level=10.0
    )
    assert corners == pytest.approx(
        [(50.0, 50.0), (50.0, -50.0), (-50.0, -50.0), (-50.0, 50.0)]
    )


def test_view_area_rejects_pose_at_or_below_ground() -> None:
    from src.data.observation_adapter import TopDownObservationAdapter

    with pytest.raises(ValueError):
        TopDownObservationAdapter._view_area_corners(
            Pose5D(0.0, 0.0, 5.0, 0.0, 0.0), ground_level=10.0
        )


def _write_ply(path: Path, *, fmt: str = "binary_little_endian") -> None:
    header = (
        "ply\n"
        f"format {fmt} 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    vertices = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4"
    ).tobytes()
    path.write_bytes(header + vertices)


def test_ply_header_and_first_vertices(tmp_path: Path) -> None:
    path = tmp_path / "birmingham_block_0.ply"
    _write_ply(path)

    header = read_ply_header(path)
    assert header.format == "binary_little_endian"
    assert header.vertex_count == 2
    assert header.vertex_properties == [("x", "float"), ("y", "float"), ("z", "float")]

    vertices = read_first_ply_vertices(path, count=2)
    assert vertices.dtype.names == ("x", "y", "z")
    assert np.allclose(vertices["x"], [1.0, 4.0])
    assert np.allclose(vertices["z"], [3.0, 6.0])


def test_ascii_ply_is_rejected_for_vertex_sampling(tmp_path: Path) -> None:
    path = tmp_path / "birmingham_block_0.ply"
    _write_ply(path, fmt="ascii")
    with pytest.raises(ValueError, match="unsupported PLY format"):
        read_first_ply_vertices(path, count=2)


def test_audit_ply_assets_reports_xyz_and_size(tmp_path: Path) -> None:
    ply_root = tmp_path / "ply"
    ply_root.mkdir()
    _write_ply(ply_root / "birmingham_block_0.ply")
    config = DataConfig(data_root=tmp_path / "data", ply_root=ply_root)

    audit = audit_ply_assets(config, ["birmingham_block_0"])
    entry = audit["birmingham_block_0"]
    assert entry["exists"] is True
    assert entry["has_xyz"] is True
    assert entry["vertex_count"] == 2
    assert entry["sample_count"] == 2
    assert entry["sample_finite"] is True


def test_images_to_tensor_contract() -> None:
    tensor = images_to_tensor(np.zeros((2, 4, 5, 3), dtype=np.uint8))
    assert tuple(tensor.shape) == (2, 3, 4, 5)
    assert tensor.dtype == torch.float32
    assert float(tensor.max()) == 0.0
