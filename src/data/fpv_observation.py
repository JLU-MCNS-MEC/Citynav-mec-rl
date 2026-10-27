"""First-person RGB/Depth observation adapter built on the SensatUrban PLY."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np

from src.config import DataConfig, ObservationConfig
from src.data.fpv_renderer import (
    CameraIntrinsics,
    RenderResult,
    camera_intrinsics,
    camera_pitch,
    render_point_cloud,
    render_quality,
)
from src.data.observation_adapter import (
    PointCloudCrop,
    TopDownObservationAdapter,
    crop_ply_cloud,
    open_ply_cloud,
)
from src.data.trajectory_adapter import Pose5D


class PlyCropStore:
    """Small LRU cache of in-memory PLY crops.

    This is deliberately not a full-dataset cache.  It keeps only the map
    regions needed by the current few frames.
    """

    def __init__(
        self,
        data_config: DataConfig,
        *,
        max_maps: int = 2,
        chunk_points: int = 4_000_000,
    ):
        self.data_config = data_config
        self.max_maps = max(1, int(max_maps))
        self.chunk_points = int(chunk_points)
        self._cache: "OrderedDict[tuple, PointCloudCrop]" = OrderedDict()

    def get(
        self,
        map_name: str,
        center_xy: Sequence[float],
        radius_m: float,
    ) -> PointCloudCrop:
        center = np.asarray(center_xy, dtype=np.float64)
        if center.shape != (2,):
            raise ValueError("center_xy must have shape [2]")
        key = (
            str(map_name),
            round(float(center[0]), 1),
            round(float(center[1]), 1),
            round(float(radius_m), 1),
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self.data_config.ply_path(map_name)
        if not path.exists():
            raise FileNotFoundError(path)
        cloud = open_ply_cloud(path)
        crop = crop_ply_cloud(
            cloud,
            center,
            float(radius_m),
            chunk_points=self.chunk_points,
        )
        self._cache[key] = crop
        while len(self._cache) > self.max_maps:
            self._cache.popitem(last=False)
        return crop

    def get_bbox(
        self,
        map_name: str,
        min_xy: Sequence[float],
        max_xy: Sequence[float],
        *,
        margin_m: float = 30.0,
    ) -> PointCloudCrop:
        low = np.asarray(min_xy, dtype=np.float64)
        high = np.asarray(max_xy, dtype=np.float64)
        if low.shape != (2,) or high.shape != (2,):
            raise ValueError("min_xy and max_xy must have shape [2]")
        center = (low + high) / 2.0
        radius = float(np.linalg.norm(high - low) / 2.0 + float(margin_m))
        return self.get(map_name, center, radius)

    def clear(self) -> None:
        self._cache.clear()


class FPVObservationAdapter:
    """Render current RGB and metric forward depth from the raw PLY."""

    source = "sensaturban_ply_pinhole_fpv"

    def __init__(
        self,
        data_config: DataConfig,
        observation_config: ObservationConfig | None = None,
        *,
        store: PlyCropStore | None = None,
        goal_adapter: TopDownObservationAdapter | None = None,
    ):
        self.data_config = data_config
        self.config = observation_config or ObservationConfig()
        self.camera: CameraIntrinsics = camera_intrinsics(self.config)
        self.store = store or PlyCropStore(
            data_config, max_maps=self.config.map_cache_size
        )
        # Goal references are a separate, target-centred appearance modality.
        # They are not the current egocentric observation.
        self.goal_adapter = goal_adapter or TopDownObservationAdapter(
            data_config, self.config
        )

    def render_parts(
        self,
        map_name: str,
        pose: Pose5D,
        *,
        crop_radius_m: float | None = None,
    ) -> tuple[RenderResult, PointCloudCrop]:
        radius = (
            float(self.config.fpv_crop_radius_m)
            if crop_radius_m is None
            else float(crop_radius_m)
        )
        crop = self.store.get(map_name, pose.xy, radius)
        pitch = camera_pitch(pose, self.config)
        result = render_point_cloud(
            crop.xyz,
            crop.rgb,
            (pose.x, pose.y, pose.z),
            pose.yaw,
            pitch,
            self.camera,
            near=float(self.config.fpv_near_m),
            far=float(self.config.fpv_far_m),
            splat_radius=int(self.config.fpv_splat_radius),
        )
        return result, crop

    def render(
        self,
        map_name: str,
        pose: Pose5D,
        object_info: dict | None,
        *,
        crop_radius_m: float | None = None,
    ) -> dict:
        result, crop = self.render_parts(
            map_name, pose, crop_radius_m=crop_radius_m
        )
        goal_rgbs = (
            None
            if object_info is None
            else self.goal_adapter.render_goal_crops(map_name, object_info)
        )
        return {
            "map_name": map_name,
            "pose": np.asarray(pose.as_tuple(), dtype=np.float32),
            "current_rgb": result.rgb,
            "depth": result.depth[..., None],
            "depth_valid_mask": result.valid_mask,
            "depth_density": result.density,
            "depth_quality": render_quality(result),
            "goal_rgbs": goal_rgbs,
            "goal_scales": tuple(int(v) for v in self.config.goal_scales),
            "observation_source": self.source,
            "camera": {
                **self.camera.as_dict(),
                "pitch_rad": float(camera_pitch(pose, self.config)),
                "pitch_mode": str(self.config.fpv_pitch_mode),
                "near_m": float(self.config.fpv_near_m),
                "far_m": float(self.config.fpv_far_m),
                "depth_definition": "camera_forward_depth_m",
                "coordinate_frame": "citynav_sensaturban_local_xyz_z_up",
            },
            "point_count_in_crop": int(crop.point_count),
        }

    def render_step(self, map_name: str, pose: Pose5D, object_info: dict | None) -> dict:
        return self.render(map_name, pose, object_info)

    def close(self) -> None:
        self.store.clear()
        self.goal_adapter.close()
