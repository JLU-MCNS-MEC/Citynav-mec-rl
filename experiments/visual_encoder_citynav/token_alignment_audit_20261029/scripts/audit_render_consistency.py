#!/usr/bin/env python3
"""A. Render consistency audit: E2 pretraining vs token build vs closed loop.

Renders the same map/pose under the three protocols (plus controlled variants),
then compares valid-pixel ratio, point counts, RGB/depth differences and frozen
E2-FOV-Pose encoder token differences.

Everything is written under the audit directory; no shared cache or weight is
modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[2]
sys.path.insert(0, str(ROOT))

from src.config import ObservationConfig, load_data_config  # noqa: E402
from src.data.citynav_dataset import CityNavDataset  # noqa: E402
from src.data.fpv_observation import FPVObservationAdapter, PlyCropStore  # noqa: E402
from src.data.fpv_renderer import (  # noqa: E402
    camera_intrinsics,
    camera_pitch,
    render_point_cloud,
)
from src.models import VisualTokenEncoder, load_encoder_checkpoint  # noqa: E402
from src.models.token_utils import encode_current_tokens  # noqa: E402
from experiments.visual_encoder_citynav.data import PointCloudStore  # noqa: E402

ENCODER = ROOT / "runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/best_encoder.pt"


def _cfg(*, splat: int, crop: float) -> ObservationConfig:
    return ObservationConfig(
        image_size=336,
        depth_size=336,
        goal_crop_size=96,
        goal_scales=(14, 28, 56),
        map_cache_size=2,
        fpv_width=336,
        fpv_height=336,
        fpv_crop_radius_m=float(crop),
        fpv_splat_radius=int(splat),
    )


PRETRAIN = _cfg(splat=1, crop=300)   # e2_fov_pose.yaml: splat_radius=1, default crop
TOKEN = _cfg(splat=0, crop=120)      # t5_smoke.py / closed_loop_eval.py


def _render_points(points_xyz, points_rgb, pose, cfg):
    camera = camera_intrinsics(cfg)
    start = time.perf_counter()
    result = render_point_cloud(
        points_xyz,
        points_rgb,
        (pose.x, pose.y, pose.z),
        pose.yaw,
        camera_pitch(pose, cfg),
        camera,
        near=float(cfg.fpv_near_m),
        far=float(cfg.fpv_far_m),
        splat_radius=int(cfg.fpv_splat_radius),
    )
    seconds = time.perf_counter() - start
    return {
        "rgb": result.rgb,
        "depth": result.depth,
        "valid_mask": result.valid_mask,
        "seconds": seconds,
    }


def _render_ply(adapter, map_name, pose, crop):
    start = time.perf_counter()
    obs = adapter.render_current(map_name, pose, crop_radius_m=float(crop))
    seconds = time.perf_counter() - start
    return {
        "rgb": obs["current_rgb"],
        "depth": obs["depth"][..., 0],
        "valid_mask": obs["depth_valid_mask"],
        "seconds": seconds,
        "point_count": int(obs["point_count_in_crop"]),
    }


def _crop_cloud(xyz, rgb, pose, radius):
    dx = xyz[:, 0].astype(np.float64) - float(pose.x)
    dy = xyz[:, 1].astype(np.float64) - float(pose.y)
    keep = (dx * dx + dy * dy) <= float(radius) ** 2
    return xyz[keep], rgb[keep]


def _image_metrics(ref, other):
    ref_rgb = ref["rgb"].astype(np.float32)
    rgb = other["rgb"].astype(np.float32)
    diff = np.abs(ref_rgb - rgb).mean(axis=2)
    common = ref["valid_mask"] & other["valid_mask"]
    depth_diff = (
        float(np.abs(ref["depth"][common] - other["depth"][common]).mean())
        if common.any()
        else float("nan")
    )
    return {
        "mean_abs_rgb": float(diff.mean()),
        "p95_abs_rgb": float(np.percentile(diff, 95)),
        "fraction_gt10": float((diff > 10.0).mean()),
        "depth_mean_abs_m": depth_diff,
        "valid_overlap": float(common.mean()),
    }


def _token_metrics(encoder, device, ref_rgb, rgb):
    with torch.no_grad():
        ref = encode_current_tokens(encoder, ref_rgb[None], device)[0].float().cpu()
        cur = encode_current_tokens(encoder, rgb[None], device)[0].float().cpu()
    a = ref.reshape(-1)
    b = cur.reshape(-1)
    denom = float(a.norm() * b.norm()) + 1e-9
    return {
        "token_l2": float((ref - cur).norm() / (ref.numel() ** 0.5)),
        "token_cos": float(torch.dot(a, b) / denom),
        "token_max_abs": float((ref - cur).abs().max()),
    }


def _save_figure(path, pose_idx, panels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="cambridge_block_25")
    ap.add_argument("--split", default="val_seen")
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--poses", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=AUDIT / "results")
    ap.add_argument("--include-ply300", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = AUDIT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_config = load_data_config()
    dataset = CityNavDataset(data_config, args.split, include_processed_descriptions=False)
    records = [r for r in dataset.records if r.map_name == args.map]
    if not records:
        raise SystemExit(f"no {args.split} record on {args.map}")
    record = records[args.record_index % len(records)]
    teacher = record.teacher(dataset.teacher_config)
    total = len(teacher.poses)
    if total == 0:
        raise SystemExit("empty teacher rollout")
    pose_ids = sorted(set(int(v) for v in np.linspace(0, total - 1, args.poses)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VisualTokenEncoder()
    load_encoder_checkpoint(encoder, ENCODER, map_location="cpu")
    encoder.to(device).eval()

    ply_store = PlyCropStore(data_config, max_maps=1)
    ply_token_adapter = FPVObservationAdapter(data_config, TOKEN, store=ply_store)
    ply_closed_adapter = FPVObservationAdapter(data_config, TOKEN, store=ply_store)
    ply_pretrain_adapter = FPVObservationAdapter(data_config, PRETRAIN, store=ply_store)
    cloud_store = PointCloudStore(data_config, max_points=4_000_000)

    cloud = cloud_store.get(args.map)
    cloud_xyz, cloud_rgb = cloud["xyz"], cloud["rgb"]

    rows = []
    summary = {
        "map": args.map,
        "split": args.split,
        "record_index": args.record_index,
        "episode_id": record.episode_id,
        "teacher_poses": total,
        "pose_ids": pose_ids,
        "pretrain_config": PRETRAIN.__dict__,
        "token_config": TOKEN.__dict__,
        "cloud_points": int(len(cloud_xyz)),
        "poses": [],
    }

    for pid in pose_ids:
        pose = teacher.poses[pid]
        panels = []

        # 1. Pretraining protocol: 4M-point map cache, full cloud, 300 m, splat 1.
        pretrain = _render_points(cloud_xyz, cloud_rgb, pose, PRETRAIN)
        pretrain["point_count"] = int(len(cloud_xyz))

        # 2/3. Token and closed-loop protocols: raw PLY, 120 m, splat 0.
        token = _render_ply(ply_token_adapter, args.map, pose, 120.0)
        closed = _render_ply(ply_closed_adapter, args.map, pose, 120.0)

        # 5. Same 120 m/splat 0 on the 4M cloud: isolates point-density effects.
        cxyz, crgb = _crop_cloud(cloud_xyz, cloud_rgb, pose, 120.0)
        token_on_cloud = _render_points(cxyz, crgb, pose, TOKEN)
        token_on_cloud["point_count"] = int(len(cxyz))

        variants = {
            "pretrain": pretrain,
            "token": token,
            "closed": closed,
            "token_on_cloud": token_on_cloud,
        }
        if args.include_ply300 and pid == pose_ids[0]:
            variants["pretrain_on_ply300"] = _render_ply(
                ply_pretrain_adapter, args.map, pose, 300.0
            )

        ref = token
        for name, obs in variants.items():
            if name == "token":
                m = {"mean_abs_rgb": 0.0, "p95_abs_rgb": 0.0, "fraction_gt10": 0.0,
                     "depth_mean_abs_m": 0.0, "valid_overlap": 1.0}
                t = {"token_l2": 0.0, "token_cos": 1.0, "token_max_abs": 0.0}
            else:
                m = _image_metrics(ref, obs)
                t = _token_metrics(encoder, device, ref["rgb"], obs["rgb"])
            rows.append({
                "pose_id": pid,
                "config": name,
                "valid_ratio": float(np.mean(obs["valid_mask"])),
                "point_count": int(obs.get("point_count", -1)),
                "render_seconds": float(obs["seconds"]),
                **m,
                **t,
            })

        summary["poses"].append({
            "pose_id": pid,
            "xyz_yaw": [float(pose.x), float(pose.y), float(pose.z), float(pose.yaw)],
            "valid_ratio": {k: float(np.mean(v["valid_mask"])) for k, v in variants.items()},
            "point_count": {k: int(v.get("point_count", -1)) for k, v in variants.items()},
        })

        panels = [
            ("pretrain 4M/300m/s1", pretrain["rgb"], None),
            ("token raw/120m/s0", token["rgb"], None),
            ("|token - pretrain|", np.abs(token["rgb"].astype(np.int16) - pretrain["rgb"].astype(np.int16)).mean(2), "magma"),
            ("depth pretrain", pretrain["depth"], "viridis"),
            ("depth token", token["depth"], "viridis"),
        ]
        _save_figure(fig_dir / f"render_pose{pid}.png", pid, panels)

    csv_path = args.out_dir / f"render_consistency_{args.map}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate means across poses (excluding the one-pose optional variant).
    agg = {}
    for name in ("pretrain", "token", "closed", "token_on_cloud"):
        sel = [r for r in rows if r["config"] == name]
        if not sel:
            continue
        agg[name] = {
            key: float(np.mean([r[key] for r in sel]))
            for key in ("valid_ratio", "point_count", "render_seconds",
                        "mean_abs_rgb", "fraction_gt10", "token_l2", "token_cos")
        }
    summary["aggregate"] = agg
    (args.out_dir / f"render_consistency_{args.map}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(agg, indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
