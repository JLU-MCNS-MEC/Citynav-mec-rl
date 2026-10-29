#!/usr/bin/env python3
"""A. Re-tokenize an existing audit subset with the E2 pretraining renderer.

Same supervised states, labels, poses, actions, goal/map/language inputs; only
the current RGB tokens, visual descriptors, heatmaps and GAB are recomputed with
the pretraining protocol (4M-point map cache, full cloud, 300 m far plane,
splat radius 1).  This isolates the "encoder saw a different image
distribution" effect.

One shard-file partition per process; merge_retokenized.py assembles the final
npz in the original row order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[2]
sys.path.insert(0, str(ROOT))

from src.config import ObservationConfig, load_data_config  # noqa: E402
from src.data.fpv_renderer import (  # noqa: E402
    camera_intrinsics,
    camera_pitch,
    render_point_cloud,
)
from src.data.gab import GABConfig, build_gab_sequence  # noqa: E402
from src.data.t5_features import history_arrays, visual_descriptor  # noqa: E402
from src.data.trajectory_adapter import Pose5D  # noqa: E402
from src.models import VisualTokenEncoder, load_encoder_checkpoint  # noqa: E402
from src.models.mec_common import build_spatial_heatmap  # noqa: E402
from src.models.token_utils import encode_current_tokens  # noqa: E402
from experiments.visual_encoder_citynav.data import PointCloudStore  # noqa: E402

ENCODER = ROOT / "runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/best_encoder.pt"

PRETRAIN = ObservationConfig(
    image_size=336,
    depth_size=336,
    goal_crop_size=96,
    goal_scales=(14, 28, 56),
    map_cache_size=2,
    fpv_width=336,
    fpv_height=336,
    fpv_crop_radius_m=300.0,
    fpv_splat_radius=1,
)


def _load_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        episode = {key: data[key] for key in (
            "current_tokens", "visuals", "heatmaps", "goal_tokens",
            "map_features", "language_features",
        )}
        episode.update(
            map_name=str(data["map_name"]),
            episode_id=str(data["episode_id"]),
            split=str(data["split"]),
            poses=[Pose5D(*(float(v) for v in row)) for row in data["poses"]],
            actions=[int(v) for v in data["actions"]],
            start_step=int(data["start_step"]),
            history_context=(int(data["history_context"])
                             if "history_context" in data.files else 0),
        )
    return episode


def _render_pretrain(cloud, pose: Pose5D, camera):
    result = render_point_cloud(
        cloud["xyz"],
        cloud["rgb"],
        (pose.x, pose.y, pose.z),
        pose.yaw,
        camera_pitch(pose, PRETRAIN),
        camera,
        near=float(PRETRAIN.fpv_near_m),
        far=float(PRETRAIN.fpv_far_m),
        splat_radius=int(PRETRAIN.fpv_splat_radius),
    )
    return result.rgb, float(np.mean(result.valid_mask))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--token-meta", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.token_meta.read_text(encoding="utf-8"))
    gab_config = GABConfig(
        reliability_threshold=float(meta["gab_reliability_threshold"]),
        **meta["gab_config"],
    )
    data_config = load_data_config()
    device = torch.device(args.device)
    encoder = VisualTokenEncoder()
    load_encoder_checkpoint(encoder, ENCODER, map_location="cpu")
    encoder.to(device).eval()
    cloud_store = PointCloudStore(data_config, max_points=4_000_000)
    camera = camera_intrinsics(PRETRAIN)

    shards = sorted(
        p for p in args.shard_dir.glob("*.npz")
        if not p.name.endswith(".tmp.npz")
    )
    shards = [p for i, p in enumerate(shards) if i % args.parts == args.part]

    keys, tokens, histories, gabs = [], [], [], []
    valid_by_key = {}

    for path in shards:
        episode = _load_shard(path)
        cloud = cloud_store.get(episode["map_name"])
        poses = episode["poses"]
        actions = episode["actions"]
        hist_ctx = int(episode["history_context"])
        start = int(episode["start_step"])
        lo = max(0, start - hist_ctx)
        sup_start = start - lo
        sup_indices = list(range(sup_start, len(actions)))

        new_tokens, new_visuals, new_heatmaps, valid_ratios = [], [], [], []
        for pose in poses:
            rgb, valid = _render_pretrain(cloud, pose, camera)
            tok = encode_current_tokens(encoder, rgb[None], device)[0].float().cpu()
            new_tokens.append(tok.to(torch.float16).numpy())
            new_visuals.append(
                visual_descriptor(tok[None].to(device))[0].cpu().numpy()
            )
            new_heatmaps.append(
                build_spatial_heatmap(
                    tok[None].to(device),
                    torch.from_numpy(episode["goal_tokens"]).float().to(device)[None],
                )[0].cpu().numpy()
            )
            valid_ratios.append(valid)
        new_visuals = np.stack(new_visuals).astype(np.float32)
        ep_gab = build_gab_sequence(
            [new_heatmaps[i] for i in sup_indices],
            [actions[i] for i in sup_indices],
            gab_config,
        )
        for step_offset, index in enumerate(sup_indices):
            h_vis, _, _, _ = history_arrays(poses, actions, index, new_visuals)
            key = (f"{episode['split']}|{episode['map_name']}|"
                   f"{episode['episode_id']}|{start + step_offset}")
            keys.append(key)
            tokens.append(new_tokens[index])
            histories.append(h_vis.astype(np.float16))
            gabs.append(ep_gab[step_offset])
            valid_by_key[key] = valid_ratios[index]

    np.savez_compressed(
        args.out_dir / f"part_{args.part:02d}.npz",
        keys=np.asarray(keys),
        tokens=np.stack(tokens) if tokens else np.zeros((0, 576, 384), np.float16),
        history_vis=np.stack(histories) if histories else np.zeros((0, 16, 768), np.float16),
        gab=np.stack(gabs) if gabs else np.zeros((0, 20), np.float32),
        valid_ratio=np.asarray([valid_by_key[k] for k in keys], dtype=np.float32),
    )
    print(f"part {args.part}: {len(keys)} frames from {len(shards)} shard(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
