"""CPU point-cloud first-person renderer for SensatUrban / CityNav.

This is a purpose-built perspective renderer, **not** AirSim.  It projects the
official SensatUrban XYZRGB point cloud into a pinhole camera at a CityNav pose
and resolves visibility with a per-pixel metric z-buffer.  There is no surface
reconstruction, lighting, or texture interpolation: holes are expected where
the point cloud is sparse, and occluded points are resolved only at point
granularity.

Coordinate and camera conventions
---------------------------------
* CityNav / SensatUrban local frame: ``x`` and ``y`` are horizontal metres,
  ``z`` is up.  The transform from CityNav to SensatUrban is the identity on
  the inspected blocks (verified against the official orthophoto crop).
* Yaw is ``atan2(dy, dx)``; positive yaw turns toward ``+y`` (left).
* Pitch is ``atan2(dz, hypot(dx, dy))``; negative pitch looks downward.
* Camera basis at yaw=0, pitch=0: forward=+x, right=-y, up=+z.  A point at
  +y appears on the left of the image, matching the official top-down crop.
* Intrinsics: square image, horizontal FOV 90 degrees, ``fx=fy=W/(2*tan(45))``,
  principal point at ``((W-1)/2, (H-1)/2)``.
* Depth is camera-forward depth ``dot(point-camera, forward)`` in metres, not
  Euclidean range and not the top-down height-above-ground image.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from src.config import ObservationConfig
from src.data.trajectory_adapter import Pose5D


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    hfov_deg: float = 90.0

    @property
    def fx(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    @property
    def vfov_deg(self) -> float:
        return math.degrees(
            2.0 * math.atan(
                math.tan(math.radians(self.hfov_deg) / 2.0)
                * self.height / self.width
            )
        )

    def as_dict(self) -> dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "hfov_deg": float(self.hfov_deg),
            "vfov_deg": float(self.vfov_deg),
            "fx_px": float(self.fx),
            "fy_px": float(self.fy),
            "cx_px": float(self.cx),
            "cy_px": float(self.cy),
        }


@dataclass
class RenderResult:
    rgb: np.ndarray
    depth: np.ndarray
    valid_mask: np.ndarray
    density: np.ndarray
    camera: CameraIntrinsics

    @property
    def empty(self) -> bool:
        return not bool(self.valid_mask.any())

    @property
    def valid_pixel_ratio(self) -> float:
        return float(self.valid_mask.mean())


def camera_intrinsics(config: ObservationConfig) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(config.fpv_width),
        height=int(config.fpv_height),
        hfov_deg=float(config.fpv_hfov_deg),
    )


def camera_basis(yaw_rad: float, pitch_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right, forward, up unit vectors in the CityNav world frame."""
    cy, sy = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    cp, sp = math.cos(float(pitch_rad)), math.sin(float(pitch_rad))
    right = np.asarray([sy, -cy, 0.0], dtype=np.float64)
    forward = np.asarray([cp * cy, cp * sy, sp], dtype=np.float64)
    up = np.asarray([-sp * cy, -sp * sy, cp], dtype=np.float64)
    return right, forward, up


def camera_pitch(pose: Pose5D, config: ObservationConfig) -> float:
    mode = str(config.fpv_pitch_mode).lower()
    if mode == "trajectory":
        return float(pose.pitch)
    if mode == "fixed":
        return math.radians(float(config.fpv_fixed_pitch_deg))
    raise ValueError(f"unknown fpv_pitch_mode: {config.fpv_pitch_mode}")


def project_world(
    point: Sequence[float],
    camera_xyz: Sequence[float],
    yaw_rad: float,
    pitch_rad: float,
    camera: CameraIntrinsics,
    *,
    near: float = 0.5,
    far: float = 300.0,
) -> dict:
    """Project one world point; returns pixel, forward depth and FOV flag."""
    right, forward, up = camera_basis(yaw_rad, pitch_rad)
    delta = np.asarray(point, dtype=np.float64) - np.asarray(
        camera_xyz, dtype=np.float64
    )
    xc = float(delta @ right)
    yc = float(delta @ up)
    zc = float(delta @ forward)
    if zc <= near or zc > far:
        return {
            "u": None,
            "v": None,
            "forward_depth_m": zc,
            "in_image": False,
        }
    u = camera.fx * xc / zc + camera.cx
    v = camera.cy - camera.fy * yc / zc
    in_image = bool(0 <= u < camera.width and 0 <= v < camera.height)
    return {
        "u": float(u) if in_image else None,
        "v": float(v) if in_image else None,
        "forward_depth_m": zc,
        "in_image": in_image,
    }


def render_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    camera_xyz: Sequence[float],
    yaw_rad: float,
    pitch_rad: float,
    camera: CameraIntrinsics,
    *,
    near: float = 0.5,
    far: float = 300.0,
    splat_radius: int = 0,
) -> RenderResult:
    """Render XYZRGB points with a pinhole camera and metric z-buffer."""
    points = np.asarray(xyz, dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if colors.shape != points.shape:
        raise ValueError("rgb must have shape [N,3]")
    if camera.width <= 0 or camera.height <= 0:
        raise ValueError("invalid image dimensions")
    if not (0 < camera.hfov_deg < 180):
        raise ValueError("invalid horizontal FOV")
    if far <= near:
        raise ValueError("far must be greater than near")

    width, height = camera.width, camera.height
    empty = np.zeros((height, width, 3), dtype=np.uint8)
    empty_depth = np.full((height, width), np.inf, dtype=np.float32)
    empty_mask = np.zeros((height, width), dtype=bool)
    empty_density = np.zeros((height, width), dtype=np.uint32)
    if len(points) == 0:
        return RenderResult(empty, empty_depth, empty_mask, empty_density, camera)

    right, forward, up = camera_basis(yaw_rad, pitch_rad)
    delta = points.astype(np.float64) - np.asarray(camera_xyz, dtype=np.float64)
    xc = delta @ right
    yc = delta @ up
    zc = delta @ forward
    valid = (
        (zc >= near)
        & (zc <= far)
    )
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return RenderResult(empty, empty_depth, empty_mask, empty_density, camera)

    u = camera.fx * xc[indices] / np.maximum(zc[indices], 1e-12) + camera.cx
    v = camera.cy - camera.fy * yc[indices] / np.maximum(zc[indices], 1e-12)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    indices = indices[inside]
    if len(indices) == 0:
        return RenderResult(empty, empty_depth, empty_mask, empty_density, camera)
    ui = np.rint(u[inside]).astype(np.int64)
    vi = np.rint(v[inside]).astype(np.int64)
    # Rounding can push a pixel just inside the continuous bounds to W/H.
    inside_round = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    indices = indices[inside_round]
    ui = ui[inside_round]
    vi = vi[inside_round]
    if len(indices) == 0:
        return RenderResult(empty, empty_depth, empty_mask, empty_density, camera)

    radius = max(0, int(splat_radius))
    pixel_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > (radius + 0.25) ** 2:
                continue
            uu = ui + dx
            vv = vi + dy
            keep = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if not bool(keep.any()):
                continue
            pixel_parts.append(vv[keep] * width + uu[keep])
            depth_parts.append(zc[indices[keep]].astype(np.float32))
            color_parts.append(colors[indices[keep]])

    if not pixel_parts:
        return RenderResult(empty, empty_depth, empty_mask, empty_density, camera)
    pixel = np.concatenate(pixel_parts)
    depths = np.concatenate(depth_parts)
    colors_flat = np.concatenate(color_parts)

    # Linear-time z-buffer.  np.minimum.at finds the closest depth per pixel;
    # the colors of all points at that depth are written, so ties are resolved
    # by the last point without an O(N log N) sort.
    depth_image = np.full(width * height, np.inf, dtype=np.float32)
    rgb_image = np.zeros((width * height, 3), dtype=np.uint8)
    np.minimum.at(depth_image, pixel, depths)
    winners = depths <= depth_image[pixel] + 1e-5
    rgb_image[pixel[winners]] = colors_flat[winners]
    density = np.bincount(
        pixel, minlength=width * height
    ).astype(np.uint32)
    mask = np.isfinite(depth_image)
    return RenderResult(
        rgb=rgb_image.reshape(height, width, 3),
        depth=depth_image.reshape(height, width),
        valid_mask=mask.reshape(height, width),
        density=density.reshape(height, width),
        camera=camera,
    )


def render_quality(result: RenderResult) -> dict:
    depth = result.depth[result.valid_mask]
    if len(depth) == 0:
        return {
            "valid_pixel_ratio": 0.0,
            "hole_ratio": 1.0,
            "depth_min_m": None,
            "depth_median_m": None,
            "depth_max_m": None,
            "depth_std_m": None,
            "depth_non_degenerate": False,
            "empty_render": True,
            "density_mean": 0.0,
        }
    return {
        "valid_pixel_ratio": float(result.valid_mask.mean()),
        "hole_ratio": float(1.0 - result.valid_mask.mean()),
        "depth_min_m": float(depth.min()),
        "depth_median_m": float(np.median(depth)),
        "depth_max_m": float(depth.max()),
        "depth_std_m": float(depth.std()),
        "depth_non_degenerate": bool(
            depth.std() > 1e-3 and depth.max() - depth.min() > 1e-2
        ),
        "empty_render": False,
        "density_mean": float(result.density.mean()),
    }
