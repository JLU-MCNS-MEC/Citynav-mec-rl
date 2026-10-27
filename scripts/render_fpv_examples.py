#!/usr/bin/env python3
"""Render a small set of real first-person RGB/Depth examples.

The script is a validation tool, not a dataset builder.  It reads a few
CityNav episodes from val_seen / val_unseen, renders real SensatUrban PLY
frames with the custom pinhole renderer, overlays the evaluation target and
teacher marker, and performs one point-cloud-vs-orthophoto orientation check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    PROJECT_ROOT,
    ObservationConfig,
    load_data_config,
)
from src.data.citynav_dataset import CityNavDataset  # noqa: E402
from src.data.fpv_observation import FPVObservationAdapter  # noqa: E402
from src.data.fpv_renderer import (  # noqa: E402
    camera_intrinsics,
    camera_pitch,
    project_world,
    render_point_cloud,
)
from src.data.paths import ensure_output_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["val_seen", "val_unseen"],
                        choices=("val_seen", "val_unseen"))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--scan-records", type=int, default=300)
    parser.add_argument(
        "--maps",
        nargs="*",
        default=None,
        help="optional map whitelist for examples",
    )
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--crop-radius", type=float, default=200.0)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--image-dir", type=Path, default=PROJECT_ROOT / "figures" / "fpv_examples"
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=PROJECT_ROOT / "reports" / "fpv_examples"
    )
    return parser.parse_args()


def _depth_visual(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    values = depth[valid]
    if values.size == 0:
        return out
    low, high = np.percentile(values, [2, 98])
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    normalized = (normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    out[valid] = colored[valid]
    return out


def _draw_projection(
    image: np.ndarray,
    projection: dict,
    label: str,
    color: tuple[int, int, int],
) -> None:
    if not projection.get("in_image") or projection.get("u") is None:
        return
    u = int(round(float(projection["u"])))
    v = int(round(float(projection["v"])))
    if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
        return
    cv2.drawMarker(image, (u, v), color, cv2.MARKER_CROSS, 18, 2)
    cv2.putText(
        image,
        f"{label} d={float(projection['forward_depth_m']):.1f}m",
        (min(u + 8, image.shape[1] - 180), max(v - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _candidate_rows(
    dataset: CityNavDataset,
    observation_config: ObservationConfig,
    limit: int,
    map_names: set[str] | None = None,
) -> list[dict]:
    camera = camera_intrinsics(observation_config)
    rows: list[dict] = []
    records = list(dataset.records)
    if limit and len(records) > limit:
        indices = np.linspace(0, len(records) - 1, int(limit), dtype=int)
        records = [records[int(index)] for index in indices]
    for record in records:
        if map_names and record.map_name not in map_names:
            continue
        result = record.teacher(dataset.teacher_config)
        for fraction in (0.25, 0.5, 0.75):
            index = min(
                len(result.poses) - 1,
                max(0, int(round(fraction * (len(result.poses) - 1)))),
            )
            pose = result.poses[index]
            pitch = camera_pitch(pose, observation_config)
            target = project_world(
                record.target_xyz,
                (pose.x, pose.y, pose.z),
                pose.yaw,
                pitch,
                camera,
                near=float(observation_config.fpv_near_m),
                far=float(observation_config.fpv_far_m),
            )
            marker = project_world(
                record.marker_xyz,
                (pose.x, pose.y, pose.z),
                pose.yaw,
                pitch,
                camera,
                near=float(observation_config.fpv_near_m),
                far=float(observation_config.fpv_far_m),
            )
            rows.append(
                {
                    "record": record,
                    "teacher": result,
                    "index": index,
                    "pose": pose,
                    "action": int(
                        result.actions[min(index, len(result.actions) - 1)]
                    ),
                    "target": target,
                    "marker": marker,
                    "in_fov": bool(
                        target["in_image"] or marker["in_image"]
                    ),
                    "altitude_m": float(pose.z - record.ground_level_m),
                }
            )
    return rows


def _select_candidates(rows: list[dict], count: int) -> list[dict]:
    """Pick distinct maps, preferring target/marker in FOV and lower altitude."""
    by_map: dict[str, list[dict]] = {}
    for row in rows:
        by_map.setdefault(row["record"].map_name, []).append(row)
    ranked = []
    for map_name, candidates in by_map.items():
        candidates.sort(
            key=lambda row: (
                not row["in_fov"],
                row["altitude_m"],
                row["record"].episode_id,
                row["index"],
            )
        )
        ranked.append((map_name, candidates[0]))
    selected: list[dict] = []
    used_maps: set[str] = set()
    for _map_name, row in ranked:
        selected.append(row)
        used_maps.add(row["record"].map_name)
        if len(selected) >= count:
            return selected
    for row in rows:
        if row["record"].map_name in used_maps:
            continue
        selected.append(row)
        used_maps.add(row["record"].map_name)
        if len(selected) >= count:
            return selected
    for row in rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def _topdown_check(
    adapter: FPVObservationAdapter,
    record,
    pose,
    output_dir: Path,
    data_config,
) -> dict:
    """Compare a pitch=-90 PLY render with the official orthophoto crop."""
    camera = camera_intrinsics(adapter.config)
    crop = adapter.store.get(
        record.map_name, pose.xy, float(adapter.config.fpv_crop_radius_m)
    )
    reference = adapter.goal_adapter.render_rgb(record.map_name, pose, camera.width)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rows = []
    best = None
    for offset_deg in (0, 90, 180, 270):
        result = render_point_cloud(
            crop.xyz,
            crop.rgb,
            (pose.x, pose.y, pose.z),
            pose.yaw + math.radians(offset_deg),
            math.radians(-90.0),
            camera,
            near=float(adapter.config.fpv_near_m),
            far=float(adapter.config.fpv_far_m),
            splat_radius=1,
        )
        gray = cv2.cvtColor(result.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mask = result.valid_mask
        correlation = (
            float(np.corrcoef(gray[mask], gray_reference[mask])[0, 1])
            if int(mask.sum()) > 100
            else float("nan")
        )
        rows.append(
            {
                "yaw_offset_deg": int(offset_deg),
                "correlation": None if math.isnan(correlation) else round(correlation, 4),
                "valid_pixel_ratio": round(float(result.valid_pixel_ratio), 4),
            }
        )
        if best is None or (
            not math.isnan(correlation)
            and correlation > float(best[1])
        ):
            best = (offset_deg, correlation, result)
    assert best is not None
    comparison = np.hstack([reference, best[2].rgb])
    path = ensure_output_file(
        output_dir / "topdown_vs_ply.jpg", data_config=data_config
    )
    cv2.imwrite(
        str(path),
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    return {
        "map_name": record.map_name,
        "episode_id": record.episode_id,
        "best_yaw_offset_deg": int(best[0]),
        "best_correlation": None if math.isnan(best[1]) else round(float(best[1]), 4),
        "yaw_offsets": rows,
        "comparison_file": str(path.relative_to(PROJECT_ROOT)),
        "note": "pitch=-90 point-cloud render vs official top-down orthophoto crop",
    }


def main() -> int:
    args = _parse_args()
    data_config = load_data_config()
    data_config.validate()
    observation_config = ObservationConfig(
        fpv_width=int(args.render_size),
        fpv_height=int(args.render_size),
        fpv_crop_radius_m=float(args.crop_radius),
        fpv_splat_radius=int(args.splat_radius),
    )
    image_dir = Path(args.image_dir)
    metadata_dir = Path(args.metadata_dir)
    adapter = FPVObservationAdapter(data_config, observation_config)
    all_rows: list[dict] = []
    try:
        for split in args.splits:
            dataset = CityNavDataset(data_config, split)
            all_rows.extend(
                _candidate_rows(
                    dataset,
                    observation_config,
                    int(args.scan_records),
                    map_names=set(args.maps) if args.maps else None,
                )
            )
        rows = _select_candidates(all_rows, int(args.samples))
        if not rows:
            raise SystemExit("no candidate pose has target/marker in the camera FOV")

        samples = []
        topdown_check = None
        for number, row in enumerate(rows):
            record = row["record"]
            pose = row["pose"]
            start = time.perf_counter()
            observation = adapter.render(
                record.map_name,
                pose,
                record.object_info,
                crop_radius_m=float(args.crop_radius),
            )
            elapsed = time.perf_counter() - start
            result_rgb = observation["current_rgb"]
            depth = observation["depth"][..., 0]
            valid = observation["depth_valid_mask"]
            overlay = result_rgb.copy()
            _draw_projection(overlay, row["target"], "target", (255, 0, 0))
            _draw_projection(overlay, row["marker"], "marker", (0, 0, 255))
            stem = f"{number:02d}_{record.split}_{record.map_name}_step{row['index']}"
            rgb_path = ensure_output_file(
                image_dir / f"{stem}_rgb.jpg", data_config=data_config
            )
            depth_path = ensure_output_file(
                image_dir / f"{stem}_depth.png", data_config=data_config
            )
            overlay_path = ensure_output_file(
                image_dir / f"{stem}_overlay.jpg", data_config=data_config
            )
            cv2.imwrite(
                str(rgb_path),
                cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            cv2.imwrite(str(depth_path), _depth_visual(depth, valid))
            cv2.imwrite(
                str(overlay_path),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            sample = {
                "sample": number,
                "split": record.split,
                "map_name": record.map_name,
                "episode_id": record.episode_id,
                "teacher_step": int(row["index"]),
                "teacher_action": int(row["action"]),
                "description_match": bool(record.description_match),
                "pose": [round(float(v), 4) for v in pose.as_tuple()],
                "camera_pitch_deg": round(
                    math.degrees(camera_pitch(pose, observation_config)), 3
                ),
                "target_projection": {
                    key: (round(float(value), 4) if isinstance(value, float) else value)
                    for key, value in row["target"].items()
                },
                "marker_projection": {
                    key: (round(float(value), 4) if isinstance(value, float) else value)
                    for key, value in row["marker"].items()
                },
                "depth_quality": observation["depth_quality"],
                "point_count_in_crop": observation["point_count_in_crop"],
                "render_seconds": round(float(elapsed), 4),
                "files": {
                    "rgb": str(rgb_path.relative_to(PROJECT_ROOT)),
                    "depth": str(depth_path.relative_to(PROJECT_ROOT)),
                    "overlay": str(overlay_path.relative_to(PROJECT_ROOT)),
                },
            }
            samples.append(sample)
            if topdown_check is None and observation["depth_quality"][
                "valid_pixel_ratio"
            ] > 0.2:
                topdown_check = _topdown_check(
                    adapter, record, pose, image_dir, data_config
                )

        # Simple contact sheet of RGB examples.
        panels = []
        for sample in samples:
            image = cv2.imread(str(PROJECT_ROOT / sample["files"]["rgb"]))
            image = cv2.resize(image, (384, 384))
            cv2.putText(
                image,
                f"{sample['split']} {sample['map_name']} step {sample['teacher_step']}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(image)
        if panels:
            while len(panels) % 2:
                panels.append(np.zeros_like(panels[0]))
            sheet = np.vstack(
                [np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)]
            )
            sheet_path = ensure_output_file(
                image_dir / "contact_sheet.jpg", data_config=data_config
            )
            cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "camera": {
                **camera_intrinsics(observation_config).as_dict(),
                "pitch_mode": observation_config.fpv_pitch_mode,
                "near_m": observation_config.fpv_near_m,
                "far_m": observation_config.fpv_far_m,
                "depth_definition": "camera_forward_depth_m",
                "coordinate_frame": "citynav_sensaturban_local_xyz_z_up",
            },
            "renderer": "custom CPU pinhole point-cloud renderer (not AirSim)",
            "samples": samples,
            "topdown_check": topdown_check,
        }
        json_path = ensure_output_file(
            metadata_dir / "metadata.json", data_config=data_config
        )
        json_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("FPV examples rendered")
        for sample in samples:
            print(
                f"  {sample['split']:11s} {sample['map_name']:20s} "
                f"step={sample['teacher_step']:3d} action={sample['teacher_action']} "
                f"valid={sample['depth_quality']['valid_pixel_ratio']:.3f} "
                f"points={sample['point_count_in_crop']} "
                f"time={sample['render_seconds']:.2f}s"
            )
        if topdown_check:
            print(
                "  top-down check: best yaw offset "
                f"{topdown_check['best_yaw_offset_deg']} deg, "
                f"correlation={topdown_check['best_correlation']}"
            )
        print(f"  metadata: {json_path.relative_to(PROJECT_ROOT)}")
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
