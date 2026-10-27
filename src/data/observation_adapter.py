"""Visual observation adapters for CityNav.

Two schemes are intentionally separated:

1. ``TopDownObservationAdapter`` reproduces the official CityNav baseline
   observation: a yaw-rotated square crop of the published orthophoto and DSM.
   The crop's half-extent equals the flight altitude above ground (the side
   length is twice that altitude), matching ``vlnce/space.py:view_area_corners``
   in the official CityNav release.  This is the default for the smoke test.
   The "depth" channel is the official ``pose.z - height`` height-above-ground
   proxy, not a line-of-sight depth image.

2. ``render_fpv_heightfield`` is a 2.5-D perspective proxy derived from the same
   RGB/height rasters.  It preserves yaw and pitch but cannot reconstruct
   facades or occlusions.  It is **not** a true first-person renderer and is not
   used for model training in this stage.

A true PLY-based first-person RGB/depth renderer is not implemented.  The PLY
audit below only reads the header and a small number of vertices; loading an
80-million-point SensatUrban block for offline rendering is a separate decision.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import rasterio
from PIL import Image

from src.config import DataConfig, ObservationConfig
from src.data.mapdata import GROUND_LEVEL
from src.data.trajectory_adapter import Pose5D


@dataclass
class _MapData:
    rgb: np.ndarray
    height: np.ndarray
    dataset: rasterio.DatasetReader


class OrthoStore:
    """Small LRU cache for orthophoto + DSM pairs.

    A 4000x4000 RGB image is 48 MB and its float32 DSM is 64 MB.  Caching all 40
    blocks would consume several GB, so the default limit is two maps.
    """

    def __init__(self, rgbd_dir: Path | str, max_maps: int = 2):
        self.rgbd_dir = Path(rgbd_dir)
        self.max_maps = max(1, int(max_maps))
        self._cache: "OrderedDict[str, _MapData]" = OrderedDict()

    def get(self, map_name: str) -> _MapData:
        if map_name in self._cache:
            self._cache.move_to_end(map_name)
            return self._cache[map_name]
        rgb_path = self.rgbd_dir / f"{map_name}.png"
        height_path = self.rgbd_dir / f"{map_name}.tif"
        if not rgb_path.exists():
            raise FileNotFoundError(rgb_path)
        if not height_path.exists():
            raise FileNotFoundError(height_path)
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"failed to decode RGB image: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dataset = rasterio.open(height_path)
        height = dataset.read(1).astype(np.float32)
        data = _MapData(rgb=rgb, height=height, dataset=dataset)
        self._cache[map_name] = data
        while len(self._cache) > self.max_maps:
            _old_name, old = self._cache.popitem(last=False)
            old.dataset.close()
        return data

    def close(self) -> None:
        for data in self._cache.values():
            data.dataset.close()
        self._cache.clear()


class TopDownObservationAdapter:
    """Official CityNav top-down crop + goal-reference crops."""

    source = "citynav_orthophoto_topdown"

    def __init__(
        self,
        data_config: DataConfig,
        observation_config: ObservationConfig | None = None,
        store: OrthoStore | None = None,
    ):
        self.data_config = data_config
        self.config = observation_config or ObservationConfig()
        self.store = store or OrthoStore(
            data_config.rgbd_dir, max_maps=self.config.map_cache_size
        )

    @staticmethod
    def _view_area_corners(pose: Pose5D, ground_level: float) -> list[tuple[float, float]]:
        altitude = float(pose.z) - float(ground_level)
        if altitude <= 0:
            raise ValueError(
                f"pose z={pose.z:.3f} is not above ground={ground_level:.3f}"
            )
        cos_yaw, sin_yaw = math.cos(pose.yaw), math.sin(pose.yaw)
        front = np.asarray([cos_yaw, sin_yaw], dtype=np.float64)
        left = np.asarray([-sin_yaw, cos_yaw], dtype=np.float64)
        center = np.asarray([pose.x, pose.y], dtype=np.float64)
        corners = [
            center + altitude * (front + left),
            center + altitude * (front - left),
            center + altitude * (-front - left),
            center + altitude * (-front + left),
        ]
        return [(float(x), float(y)) for x, y in corners]

    def _warp_map(
        self,
        map_name: str,
        pose: Pose5D,
        size: int,
        *,
        kind: str,
    ) -> np.ndarray:
        if map_name not in GROUND_LEVEL:
            raise ValueError(f"no GROUND_LEVEL entry for {map_name}")
        data = self.store.get(map_name)
        corners = self._view_area_corners(pose, GROUND_LEVEL[map_name])
        rowcol = np.asarray(
            [data.dataset.index(x, y) for x, y in corners], dtype=np.float32
        )
        colrow = np.flip(rowcol, axis=-1)
        dst = np.asarray(
            [(0, 0), (size - 1, 0), (size - 1, size - 1), (0, size - 1)],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(colrow, dst)
        if kind == "rgb":
            return cv2.warpPerspective(
                data.rgb,
                transform,
                (int(size), int(size)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        if kind == "height":
            height = cv2.warpPerspective(
                data.height,
                transform,
                (int(size), int(size)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=float(pose.z),
            )
            return np.maximum(float(pose.z) - height, 0.0).astype(np.float32)
        raise ValueError(kind)

    def render_rgb(self, map_name: str, pose: Pose5D, size: int | None = None) -> np.ndarray:
        return self._warp_map(
            map_name, pose, int(size or self.config.image_size), kind="rgb"
        )

    def render_depth(
        self, map_name: str, pose: Pose5D, size: int | None = None
    ) -> np.ndarray:
        depth = self._warp_map(
            map_name, pose, int(size or self.config.depth_size), kind="height"
        )
        return depth[..., None]

    def render_goal_crop(
        self,
        map_name: str,
        object_info: dict,
        scale_px: int,
        output_size: int | None = None,
    ) -> np.ndarray:
        if object_info is None or not object_info.get("position"):
            raise ValueError("CityRefer object_info with a position is required")
        output_size = int(output_size or self.config.goal_crop_size)
        data = self.store.get(map_name)
        x, y = (float(v) for v in object_info["position"][:2])
        dimension = object_info.get("dimension") or [20.0, 20.0, 10.0]
        object_size_m = max(float(dimension[0]), float(dimension[1]), 1.0)
        side_m = object_size_m * float(output_size) / float(scale_px)
        pixel_size_m = abs(float(data.dataset.transform.a))
        side_px = max(2, int(round(side_m / max(pixel_size_m, 1e-9))))
        row, col = data.dataset.index(x, y)
        patch = cv2.getRectSubPix(
            data.rgb, (side_px, side_px), (float(col), float(row))
        )
        interpolation = (
            cv2.INTER_AREA if side_px > output_size else cv2.INTER_LINEAR
        )
        patch = cv2.resize(
            patch, (output_size, output_size), interpolation=interpolation
        )
        return patch

    def render_goal_crops(
        self,
        map_name: str,
        object_info: dict,
        scales: Sequence[int] | None = None,
        output_size: int | None = None,
    ) -> np.ndarray:
        scales = tuple(scales or self.config.goal_scales)
        return np.stack(
            [
                self.render_goal_crop(
                    map_name, object_info, int(scale), output_size=output_size
                )
                for scale in scales
            ],
            axis=0,
        )

    def render_step(
        self,
        map_name: str,
        pose: Pose5D,
        object_info: dict,
    ) -> dict:
        return {
            "map_name": map_name,
            "pose": np.asarray(pose.as_tuple(), dtype=np.float32),
            "current_rgb": self.render_rgb(map_name, pose),
            "depth": self.render_depth(map_name, pose),
            "goal_rgbs": self.render_goal_crops(map_name, object_info),
            "goal_scales": tuple(int(v) for v in self.config.goal_scales),
            "observation_source": self.source,
        }

    def render_fpv_heightfield(
        self,
        map_name: str,
        pose: Pose5D,
        size: int | None = None,
        horizontal_fov_deg: float = 90.0,
    ) -> np.ndarray:
        """2.5-D perspective proxy; explicitly not a true first-person frame."""
        size = int(size or self.config.image_size)
        data = self.store.get(map_name)
        x, y, z, yaw, pitch = pose.as_tuple()
        downward = float(np.clip(-pitch / (math.pi / 2), 0.05, 1.0))
        far_m = 25.0 + (1.0 - downward) * 110.0
        near_m = max(2.0, z * 0.04)
        half_fov = math.radians(horizontal_fov_deg) / 2.0
        half_near = near_m * math.tan(half_fov)
        half_far = far_m * math.tan(half_fov)
        forward = np.asarray([math.cos(yaw), math.sin(yaw)])
        right = np.asarray([-math.sin(yaw), math.cos(yaw)])
        origin = np.asarray([x, y])
        corners = np.stack(
            [
                origin + forward * near_m - right * half_near,
                origin + forward * near_m + right * half_near,
                origin + forward * far_m + right * half_far,
                origin + forward * far_m - right * half_far,
            ]
        )
        src = np.asarray(
            [
                [data.dataset.index(px, py)[1], data.dataset.index(px, py)[0]]
                for px, py in corners
            ],
            dtype=np.float32,
        )
        dst = np.asarray(
            [[0, size - 1], [size - 1, size - 1], [size - 1, 0], [0, 0]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            data.rgb,
            transform,
            (size, size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(127, 127, 127),
        )

    def close(self) -> None:
        self.store.close()


# ---------------------------------------------------------------------------
# PLY audit (header + a few vertices only)
# ---------------------------------------------------------------------------


_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class PlyHeader:
    format: str
    vertex_count: int
    vertex_properties: list[tuple[str, str]]
    header_bytes: int
    file_bytes: int
    element_count: int


def read_ply_header(path: Path | str) -> PlyHeader:
    path = Path(path)
    lines: list[str] = []
    with path.open("rb") as handle:
        for _ in range(10000):
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PLY header not terminated: {path.name}")
            line = raw.decode("ascii", errors="strict").strip()
            lines.append(line)
            if line == "end_header":
                header_bytes = handle.tell()
                break
        else:
            raise ValueError(f"PLY header too long: {path.name}")

    ply_format = ""
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element = None
    element_count = 0
    for line in lines:
        if line.startswith("format "):
            ply_format = line.split()[1]
        elif line.startswith("element "):
            parts = line.split()
            element_count += 1
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif line.startswith("property ") and current_element == "vertex":
            parts = line.split()
            if len(parts) == 3:
                vertex_properties.append((parts[2], parts[1]))
            elif len(parts) == 5 and parts[1] == "list":
                # List properties are not expected on vertex elements here.
                vertex_properties.append((parts[4], "list"))
    if not ply_format:
        raise ValueError(f"missing PLY format line: {path.name}")
    return PlyHeader(
        format=ply_format,
        vertex_count=int(vertex_count),
        vertex_properties=vertex_properties,
        header_bytes=int(header_bytes),
        file_bytes=int(path.stat().st_size),
        element_count=int(element_count),
    )


@dataclass
class PlyCloud:
    """Read-only memmap of an official SensatUrban binary PLY."""

    path: Path
    header: PlyHeader
    data: np.memmap


@dataclass
class PointCloudCrop:
    """XYZRGB points selected from a PLY crop."""

    xyz: np.ndarray
    rgb: np.ndarray
    source_path: Path
    center_xy: tuple[float, float]
    radius_m: float

    @property
    def point_count(self) -> int:
        return int(len(self.xyz))


def open_ply_cloud(path: Path | str) -> PlyCloud:
    """Open a binary SensatUrban PLY without loading all points."""
    path = Path(path)
    header = read_ply_header(path)
    if header.format != "binary_little_endian":
        raise ValueError(f"unsupported PLY format: {header.format}")
    fields = []
    for name, type_name in header.vertex_properties:
        if type_name == "list":
            raise ValueError("list vertex properties are not supported")
        if type_name not in _PLY_TYPES:
            raise ValueError(f"unsupported PLY scalar type: {type_name}")
        fields.append((name, _PLY_TYPES[type_name]))
    missing = [name for name in ("x", "y", "z", "red", "green", "blue")
               if name not in {field[0] for field in fields}]
    if missing:
        raise ValueError(f"PLY is missing color/XYZ field(s): {missing}")
    dtype = np.dtype(fields)
    data = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=int(header.header_bytes),
        shape=(int(header.vertex_count),),
    )
    return PlyCloud(path=path, header=header, data=data)


def crop_ply_cloud(
    cloud: PlyCloud,
    center_xy: Sequence[float],
    radius_m: float,
    *,
    chunk_points: int = 4_000_000,
) -> PointCloudCrop:
    """Select points within ``radius_m`` of ``center_xy`` in chunks."""
    center = np.asarray(center_xy, dtype=np.float64)
    if center.shape != (2,):
        raise ValueError("center_xy must have shape [2]")
    radius = float(radius_m)
    if radius <= 0:
        raise ValueError("radius_m must be positive")
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    total = int(cloud.header.vertex_count)
    chunk = max(1, int(chunk_points))
    radius_sq = radius * radius
    for start in range(0, total, chunk):
        block = np.asarray(cloud.data[start:start + chunk])
        dx = block["x"].astype(np.float64) - center[0]
        dy = block["y"].astype(np.float64) - center[1]
        mask = (dx * dx + dy * dy) <= radius_sq
        if not bool(mask.any()):
            continue
        xyz_parts.append(
            np.stack(
                [block["x"][mask], block["y"][mask], block["z"][mask]], axis=1
            ).astype(np.float32)
        )
        rgb_parts.append(
            np.stack(
                [block["red"][mask], block["green"][mask], block["blue"][mask]],
                axis=1,
            ).astype(np.uint8)
        )
    if xyz_parts:
        xyz = np.concatenate(xyz_parts, axis=0)
        rgb = np.concatenate(rgb_parts, axis=0)
    else:
        xyz = np.empty((0, 3), dtype=np.float32)
        rgb = np.empty((0, 3), dtype=np.uint8)
    return PointCloudCrop(
        xyz=xyz,
        rgb=rgb,
        source_path=cloud.path,
        center_xy=(float(center[0]), float(center[1])),
        radius_m=radius,
    )


def read_first_ply_vertices(
    path: Path | str, count: int = 16
) -> np.ndarray:
    header = read_ply_header(path)
    if header.format != "binary_little_endian":
        raise ValueError(f"unsupported PLY format: {header.format}")
    if not header.vertex_properties:
        raise ValueError("PLY has no vertex properties")
    dtype_fields = []
    for name, type_name in header.vertex_properties:
        if type_name == "list":
            raise ValueError("list vertex properties are not supported")
        if type_name not in _PLY_TYPES:
            raise ValueError(f"unsupported PLY scalar type: {type_name}")
        dtype_fields.append((name, _PLY_TYPES[type_name]))
    dtype = np.dtype(dtype_fields)
    expected = min(int(count), int(header.vertex_count))
    with Path(path).open("rb") as handle:
        handle.seek(header.header_bytes)
        vertices = np.fromfile(handle, dtype=dtype, count=expected)
    if len(vertices) != expected:
        raise ValueError(
            f"expected {expected} vertices, read {len(vertices)} "
            f"from {Path(path).name}"
        )
    return vertices


def audit_ply_assets(
    data_config: DataConfig,
    map_names: Iterable[str],
    *,
    sample_vertices: int = 16,
) -> dict:
    summary = {}
    for map_name in sorted(set(map_names)):
        path = data_config.ply_path(map_name)
        if not path.exists():
            summary[map_name] = {"exists": False}
            continue
        header = read_ply_header(path)
        vertices = read_first_ply_vertices(path, count=sample_vertices)
        names = set(vertices.dtype.names or ())
        xyz = np.stack(
            [vertices["x"], vertices["y"], vertices["z"]], axis=-1
        ).astype(np.float64)
        summary[map_name] = {
            "exists": True,
            "format": header.format,
            "vertex_count": int(header.vertex_count),
            "file_bytes": int(header.file_bytes),
            "properties": [name for name, _ in header.vertex_properties],
            "has_xyz": {"x", "y", "z"}.issubset(names),
            "sample_count": int(len(vertices)),
            "sample_finite": bool(np.isfinite(xyz).all()),
        }
    return summary


# ---------------------------------------------------------------------------
# Raster audit
# ---------------------------------------------------------------------------


def audit_raster_assets(
    data_config: DataConfig,
    map_names: Iterable[str],
) -> dict:
    summary = {}
    for map_name in sorted(set(map_names)):
        png_path = data_config.rgb_path(map_name)
        tif_path = data_config.height_path(map_name)
        entry: dict = {
            "png_exists": png_path.exists(),
            "tif_exists": tif_path.exists(),
        }
        if png_path.exists():
            with Image.open(png_path) as image:
                entry["png_size"] = list(image.size)
                entry["png_mode"] = image.mode
        if tif_path.exists():
            with rasterio.open(tif_path) as dataset:
                entry.update(
                    {
                        "tif_size": [int(dataset.width), int(dataset.height)],
                        "tif_dtype": str(dataset.dtypes[0]),
                        "tif_transform": [float(v) for v in dataset.transform][:6],
                        "tif_bounds": [
                            float(dataset.bounds.left),
                            float(dataset.bounds.bottom),
                            float(dataset.bounds.right),
                            float(dataset.bounds.top),
                        ],
                        "tif_crs": None if dataset.crs is None else str(dataset.crs),
                    }
                )
        summary[map_name] = entry
    return summary


def pose_inside_raster(pose: Pose5D, dataset: rasterio.DatasetReader) -> bool:
    bounds = dataset.bounds
    return bool(
        bounds.left <= pose.x <= bounds.right
        and bounds.bottom <= pose.y <= bounds.top
    )
