"""Synthetic tests for the point-cloud pinhole renderer and PLY loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.fpv_renderer import (
    CameraIntrinsics,
    camera_basis,
    project_world,
    render_point_cloud,
    render_quality,
)
from src.data.observation_adapter import crop_ply_cloud, open_ply_cloud


def _camera() -> CameraIntrinsics:
    return CameraIntrinsics(width=64, height=64, hfov_deg=90.0)


def test_camera_axes_match_image_convention() -> None:
    camera = _camera()
    camera_xyz = (0.0, 0.0, 10.0)

    center = project_world(
        (10.0, 0.0, 10.0), camera_xyz, 0.0, 0.0, camera,
        near=0.5, far=100.0,
    )
    left = project_world(
        (10.0, 5.0, 10.0), camera_xyz, 0.0, 0.0, camera,
        near=0.5, far=100.0,
    )
    right = project_world(
        (10.0, -5.0, 10.0), camera_xyz, 0.0, 0.0, camera,
        near=0.5, far=100.0,
    )
    above = project_world(
        (10.0, 0.0, 15.0), camera_xyz, 0.0, 0.0, camera,
        near=0.5, far=100.0,
    )

    assert center["in_image"] is True
    assert center["u"] == pytest.approx(31.5)
    assert center["v"] == pytest.approx(31.5)
    assert center["forward_depth_m"] == pytest.approx(10.0)
    assert left["u"] < right["u"]
    assert above["v"] < center["v"]


def test_renderer_depth_buffer_keeps_nearest_point_and_forward_depth() -> None:
    points = np.asarray(
        [
            [10.0, 0.0, 10.0],
            [5.0, 0.0, 10.0],
            [20.0, 0.0, 10.0],
        ],
        dtype=np.float32,
    )
    colors = np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    result = render_point_cloud(
        points,
        colors,
        (0.0, 0.0, 10.0),
        0.0,
        0.0,
        _camera(),
        near=0.5,
        far=100.0,
        splat_radius=0,
    )
    center = result.depth[32, 32]
    assert center == pytest.approx(5.0)
    assert tuple(result.rgb[32, 32]) == (0, 255, 0)
    quality = render_quality(result)
    assert quality["valid_pixel_ratio"] > 0.0
    # Only one surface is visible in this synthetic frame.
    assert quality["depth_non_degenerate"] is False


def test_renderer_rejects_points_behind_camera() -> None:
    result = render_point_cloud(
        np.asarray([[-10.0, 0.0, 10.0]], dtype=np.float32),
        np.asarray([[255, 255, 255]], dtype=np.uint8),
        (0.0, 0.0, 10.0),
        0.0,
        0.0,
        _camera(),
        near=0.5,
        far=100.0,
    )
    assert result.valid_mask.sum() == 0
    assert result.empty is True


def _write_tiny_ply(path: Path) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar class\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("class", "u1"),
        ]
    )
    points = np.asarray(
        [(0.0, 0.0, 1.0, 10, 20, 30, 0),
         (5.0, 0.0, 1.0, 40, 50, 60, 1),
         (20.0, 0.0, 1.0, 70, 80, 90, 2)],
        dtype=dtype,
    )
    path.write_bytes(header + points.tobytes())


def test_ply_memmap_and_crop(tmp_path: Path) -> None:
    path = tmp_path / "birmingham_block_0.ply"
    _write_tiny_ply(path)
    cloud = open_ply_cloud(path)
    assert cloud.header.vertex_count == 3
    crop = crop_ply_cloud(cloud, center_xy=(2.0, 0.0), radius_m=5.0)
    assert crop.point_count == 2
    assert np.allclose(crop.xyz[:, 0], [0.0, 5.0])
    assert np.array_equal(crop.rgb, [[10, 20, 30], [40, 50, 60]])


def test_camera_basis_is_orthonormal() -> None:
    right, forward, up = camera_basis(0.7, -0.4)
    assert np.allclose(np.linalg.norm([right, forward, up], axis=1), 1.0)
    assert abs(float(right @ forward)) < 1e-9
    assert abs(float(right @ up)) < 1e-9
    assert abs(float(forward @ up)) < 1e-9
