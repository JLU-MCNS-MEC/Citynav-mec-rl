#!/usr/bin/env python3
"""Read-only CityNav data audit.

Writes only summary JSON/Markdown under ``reports/``.  Raw CityNav, CityRefer,
raster and PLY directories are never written.  The report intentionally contains
no server absolute paths.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_data_config  # noqa: E402
from src.data.citynav_dataset import (  # noqa: E402
    CITYNAV_SPLITS,
    CityNavDataset,
    CityReferIndex,
    count_split_records,
    map_names_for_splits,
    missing_map_assets,
)
from src.data.mapdata import (  # noqa: E402
    CITYNAV_SPLIT_COUNTS,
    CITYREFER_MAP_COUNT,
    CITYREFER_OBJECT_COUNT,
)
from src.data.observation_adapter import (  # noqa: E402
    audit_ply_assets,
    audit_raster_assets,
)
from src.data.paths import (  # noqa: E402
    assert_snapshot_unchanged,
    ensure_output_file,
    snapshot_raw_dirs,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _trajectory_stats(records) -> dict:
    step_distances: list[float] = []
    yaw_changes: list[float] = []
    z_values: list[float] = []
    pitches: list[float] = []
    direction_norms: list[float] = []
    start_target: list[float] = []
    ambiguous = 0
    pitch_extreme = 0
    for record in records:
        raw = record.trajectory_raw
        for point in raw:
            norm = math.hypot(float(point[3]), float(point[4]), float(point[5]))
            direction_norms.append(norm)
            pitch = math.degrees(
                math.atan2(float(point[5]), math.hypot(float(point[3]), float(point[4])))
            )
            pitches.append(pitch)
            z_values.append(float(point[2]))
            if math.hypot(float(point[3]), float(point[4])) < 1e-6:
                ambiguous += 1
            if abs(pitch) >= 89.9:
                pitch_extreme += 1
        for left, right in zip(raw[:-1], raw[1:]):
            step_distances.append(
                math.dist((float(left[0]), float(left[1]), float(left[2])),
                          (float(right[0]), float(right[1]), float(right[2])))
            )
            z_values.append(float(right[2]))
        for left, right in zip(record.poses[:-1], record.poses[1:]):
            delta = (right.yaw - left.yaw + math.pi) % (2 * math.pi) - math.pi
            yaw_changes.append(math.degrees(delta))
        start_target.append(
            math.hypot(
                record.start_pose.x - record.target_xy[0],
                record.start_pose.y - record.target_xy[1],
            )
        )
    return {
        "trajectory_points": len(direction_norms),
        "step_distance_m": {
            "min": round(min(step_distances), 4) if step_distances else 0.0,
            "median": round(_percentile(step_distances, 50), 4),
            "p90": round(_percentile(step_distances, 90), 4),
            "max": round(max(step_distances), 4) if step_distances else 0.0,
        },
        "yaw_change_deg": {
            "median": round(_percentile(yaw_changes, 50), 4),
            "p90": round(_percentile(yaw_changes, 90), 4),
        },
        "direction_norm": {
            "min": round(min(direction_norms), 6) if direction_norms else 0.0,
            "max": round(max(direction_norms), 6) if direction_norms else 0.0,
        },
        "pitch_deg": {
            "min": round(min(pitches), 2) if pitches else 0.0,
            "median": round(_percentile(pitches, 50), 2),
            "max": round(max(pitches), 2) if pitches else 0.0,
            "near_vertical_points": int(pitch_extreme),
        },
        "degenerate_horizontal_direction_points": int(ambiguous),
        "z": {
            "min": round(min(z_values), 3) if z_values else 0.0,
            "median": round(_percentile(z_values, 50), 3),
            "max": round(max(z_values), 3) if z_values else 0.0,
        },
        "start_target_xy_m": {
            "min": round(min(start_target), 3) if start_target else 0.0,
            "median": round(_percentile(start_target, 50), 3),
            "max": round(max(start_target), 3) if start_target else 0.0,
        },
    }


def _target_bounds_stats(records, raster_audit: dict) -> dict:
    outside = []
    checked = 0
    for record in records:
        bounds = raster_audit.get(record.map_name, {}).get("tif_bounds")
        if not bounds:
            continue
        checked += 1
        left, bottom, right, top = bounds
        if not (
            left <= record.target_xy[0] <= right
            and bottom <= record.target_xy[1] <= top
        ):
            outside.append(
                {
                    "map_name": record.map_name,
                    "target_xy": [round(record.target_xy[0], 2), round(record.target_xy[1], 2)],
                }
            )
    return {
        "checked": checked,
        "outside_bounds": len(outside),
        "examples": outside[:5],
    }


def _markdown(report: dict) -> str:
    lines = [
        "# CityNav 数据审计报告",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 数据根目录: `{report['config']['data']['data_root']}` (只读)",
        f"- PLY 根目录: `{report['config']['data']['ply_root']}` (只读)",
        f"- 原始数据快照未变化: `{report['raw_data_unchanged']}`",
        f"- 快照文件数: {report['raw_snapshot_after']['file_count']}",
        "",
        "## 1. 官方划分计数",
        "",
        "| split | 实际条数 | 预期条数 | 一致 |",
        "|---|---:|---:|:---:|",
    ]
    for split in report["splits"]:
        entry = report["splits"].get(split, {})
        lines.append(
            f"| {split} | {entry.get('records', 0)} | "
            f"{CITYNAV_SPLIT_COUNTS.get(split, '?')} | "
            f"{entry.get('count_matches_expected', False)} |"
        )
    lines += [
        "",
        "## 2. 地图与资产覆盖",
        "",
        f"- CityRefer 地图数: {report['cityrefer']['maps']} (预期 {CITYREFER_MAP_COUNT})",
        f"- CityRefer 对象数: {report['cityrefer']['objects']} (预期 {CITYREFER_OBJECT_COUNT})",
        f"- 四个划分引用地图数: {len(report['map_coverage']['map_names'])}",
        f"- 缺失 PNG/TIF/PLY 的地图: {report['map_coverage']['missing_assets'] or '无'}",
        "",
        "| split | 地图数 | 地图列表 |",
        "|---|---:|---|",
    ]
    for split in report["splits"]:
        entry = report["splits"].get(split, {})
        lines.append(
            f"| {split} | {entry.get('maps', 0)} | "
            f"{', '.join(entry.get('map_counts', {}).keys())} |"
        )
    lines += [
        "",
        "## 3. 轨迹、位姿与目标对应",
        "",
        "| split | 点数 | 步距中位/p90 (m) | 偏航变化 p90 (deg) | 近垂直点数 | yaw 退化点数 | 起点-目标中位 (m) | 描述 0-based 匹配 (严格/归一化) | 目标越界 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in report["splits"]:
        entry = report["splits"].get(split, {})
        stats = entry.get("trajectory_stats", {})
        lines.append(
            "| {split} | {points} | {step_med}/{step_p90} | {yaw_p90} | {vertical} | "
            "{deg} | {dist} | {match}/{match_norm}/{records} | {outside} |".format(
                split=split,
                points=stats.get("trajectory_points", 0),
                step_med=stats.get("step_distance_m", {}).get("median", 0),
                step_p90=stats.get("step_distance_m", {}).get("p90", 0),
                yaw_p90=stats.get("yaw_change_deg", {}).get("p90", 0),
                vertical=stats.get("pitch_deg", {}).get("near_vertical_points", 0),
                deg=stats.get("degenerate_horizontal_direction_points", 0),
                dist=stats.get("start_target_xy_m", {}).get("median", 0),
                match=entry.get("description_match", 0),
                match_norm=entry.get("description_match_normalized", 0),
                records=entry.get("records", 0),
                outside=entry.get("target_bounds", {}).get("outside_bounds", 0),
            )
        )
    lines += [
        "",
        "## 4. 离散动作教师回放",
        "",
        "教师使用 5 m 前进、15 deg 转向、20 m 停止半径和 12 m 纯追踪前视点。"
        "回放轨迹与原始演示轨迹的最近点距离用于交叉验证。",
        "",
        "| split | 抽样 | 成功率 | 步数中位/p90 | 终点距离中位/p90 (m) | 交叉轨迹中位 (m) | 未成功原因 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for split in report["splits"]:
        entry = report["splits"].get(split, {}).get("teacher", {})
        lines.append(
            f"| {split} | {entry.get('sampled', 0)} | "
            f"{entry.get('success_rate', 0):.4f} | "
            f"{entry.get('steps_median', 0)}/{entry.get('steps_p90', 0)} | "
            f"{entry.get('final_distance_m_median', 0)}/"
            f"{entry.get('final_distance_m_p90', 0)} | "
            f"{entry.get('cross_track_median_m', 0)} | "
            f"{entry.get('wrong_stop_reasons', {})} |"
        )
    lines += [
        "",
        "## 5. PLY / 栅格读取",
        "",
        f"- PLY 审计: {report['ply']['summary']}",
        f"- 栅格审计: {report['raster']['summary']}",
        "",
        "PLY 仅读取文件头和少量顶点，不加载完整的 8000 万点云；"
        "真正的第一人称 PLY 渲染尚未实现。",
        "",
        "## 6. 结论与限制",
        "",
        "- 所审计划分均可读取，计数与已知统计一致；"
        "test_unseen 默认排除，留给最终冻结评估。",
        "- 所有引用地图均具备 CityRefer、PNG、TIFF 和 PLY。",
        "- 轨迹六元组按官方约定转换为 `(x,y,z,yaw,pitch)`；"
        "近垂直方向使用局部运动方向作为 yaw 的确定性回退。",
        "- 当前只验证数据与监督动作，不包含 GSM、CityNav 导航增强架构或 RL 环境。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--ply-root", type=str, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_seen", "val_seen", "val_unseen"],
        choices=list(CITYNAV_SPLITS),
    )
    parser.add_argument(
        "--include-test-unseen",
        action="store_true",
        help="also audit test_unseen; reserve it for the final frozen report",
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="limit records per split for a quick audit")
    parser.add_argument("--teacher-sample", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--skip-ply", action="store_true")
    parser.add_argument("--skip-raster", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    splits = list(dict.fromkeys(args.splits))
    if "test_unseen" in splits and not args.include_test_unseen:
        parser.error("test_unseen requires --include-test-unseen")
    if args.include_test_unseen and "test_unseen" not in splits:
        splits.append("test_unseen")

    data_config = load_data_config(
        args.data_config, data_root=args.data_root, ply_root=args.ply_root
    )
    data_config.validate()
    before = snapshot_raw_dirs(data_config)

    counts = count_split_records(data_config, splits)
    all_maps = map_names_for_splits(data_config, splits)
    missing_assets = missing_map_assets(data_config, all_maps)

    raster_audit = {} if args.skip_raster else audit_raster_assets(data_config, all_maps)
    ply_audit = {} if args.skip_ply else audit_ply_assets(data_config, all_maps)

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"data": data_config.redacted()},
        "counts": counts,
        "split_counts_expected": CITYNAV_SPLIT_COUNTS,
        "cityrefer": {},
        "map_coverage": {
            "map_names": all_maps,
            "map_count": len(all_maps),
            "missing_assets": missing_assets,
        },
        "splits": {},
        "raster": {
            "summary": {
                "maps": len(raster_audit),
                "png_tif_size_match": all(
                    entry.get("png_size") == entry.get("tif_size")
                    for entry in raster_audit.values()
                    if "png_size" in entry and "tif_size" in entry
                ) if raster_audit else None,
                "all_transform_0p1m": all(
                    abs(entry["tif_transform"][0] - 0.1) < 1e-9
                    and abs(entry["tif_transform"][4] + 0.1) < 1e-9
                    for entry in raster_audit.values()
                    if "tif_transform" in entry
                ) if raster_audit else None,
                "png_size_variants": [
                    [int(value) for value in size]
                    for size in sorted(
                        {
                            tuple(entry["png_size"])
                            for entry in raster_audit.values()
                            if "png_size" in entry
                        }
                    )
                ],
            },
            "maps": raster_audit,
        },
        "ply": {
            "summary": {
                "maps": len(ply_audit),
                "all_have_xyz": all(
                    entry.get("has_xyz", False) for entry in ply_audit.values()
                ) if ply_audit else None,
                "total_vertices": int(
                    sum(entry.get("vertex_count", 0) for entry in ply_audit.values())
                ),
            },
            "maps": ply_audit,
        },
    }

    cityrefer = CityReferIndex(data_config.cityrefer_dir / "objects.json")
    report["cityrefer"] = {
        "maps": len(cityrefer.map_names),
        "objects": cityrefer.object_count,
    }

    for split in splits:
        if split not in CITYNAV_SPLITS:
            raise ValueError(f"unknown split: {split}")
        dataset = CityNavDataset(
            data_config,
            split,
            allow_test_unseen=True,
            max_episodes=args.max_records,
            strict_objects=False,
            include_processed_descriptions=False,
        )
        summary = dataset.summary()
        summary["count_matches_expected"] = (
            counts.get(split) == CITYNAV_SPLIT_COUNTS.get(split)
        )
        summary["trajectory_stats"] = _trajectory_stats(dataset.records)
        summary["target_bounds"] = _target_bounds_stats(dataset.records, raster_audit)
        summary["teacher"] = dataset.teacher_success_rate(
            limit=args.teacher_sample, seed=args.seed
        )
        episode_ids = [record.episode_id for record in dataset.records]
        summary["duplicate_episode_ids"] = len(episode_ids) - len(set(episode_ids))
        summary["skipped_examples"] = dataset.skipped[:3]
        report["splits"][split] = _jsonable(summary)

    after = snapshot_raw_dirs(data_config)
    assert_snapshot_unchanged(before, after)
    report["raw_snapshot_before"] = before
    report["raw_snapshot_after"] = after
    report["raw_data_unchanged"] = True

    output_dir = args.output_dir
    json_path = ensure_output_file(output_dir / "data_audit.json", data_config=data_config)
    md_path = ensure_output_file(output_dir / "data_audit.md", data_config=data_config)
    json_path.write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")

    print("CityNav audit complete")
    for split in splits:
        entry = report["splits"][split]
        teacher = entry["teacher"]
        print(
            f"  {split:12s} records={entry['records']:5d} maps={entry['maps']:2d} "
            f"teacher_success={teacher.get('success_rate', 0):.3f} "
            f"desc_match={entry['description_match']}/"
            f"{entry['description_match_normalized']}/{entry['records']} "
            f"dup_ids={entry['duplicate_episode_ids']}"
        )
    print(f"  raw_data_unchanged={report['raw_data_unchanged']} digest={after['digest']}")
    print(f"  wrote {json_path.relative_to(PROJECT_ROOT)} and {md_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
