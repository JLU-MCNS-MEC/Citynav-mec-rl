#!/usr/bin/env python3
"""Small overfit validation for the FPV + six-action training path.

It uses 128-256 train_seen trajectories, renders a small bounded FPV cache,
uses frozen BERT and a trainable MEC visual encoder, and runs a short overfit
loop for Small and Transformer-M.  This is a gradient/label/pipeline check,
not a training result and not a validation benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    PROJECT_ROOT,
    ObservationConfig,
    TextConfig,
    TrainingConfig,
    load_data_config,
)
from src.data.actions import (  # noqa: E402
    ACTION_NAMES,
    GO_DOWN,
    GO_UP,
    MOVE_FORWARD,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
)
from src.data.citynav_dataset import CityNavDataset  # noqa: E402
from src.data.fpv_renderer import (  # noqa: E402
    camera_intrinsics,
    camera_pitch,
    render_point_cloud,
)
from src.data.language_adapter import build_language_adapter  # noqa: E402
from src.data.observation_adapter import (  # noqa: E402
    TopDownObservationAdapter,
    crop_ply_cloud,
    open_ply_cloud,
)
from src.data.paths import (  # noqa: E402
    assert_snapshot_unchanged,
    ensure_output_dir,
    ensure_output_file,
    snapshot_raw_dirs,
)
from src.models import SmallMEC, TransformerMMEC  # noqa: E402
from src.models.mec_common import placeholder_network_state  # noqa: E402
from src.models.token_utils import encode_scene  # noqa: E402
from src.models.vision_encoder import (  # noqa: E402
    VisualTokenEncoder,
    load_encoder_checkpoint,
)
from src.training.supervised import (  # noqa: E402
    factorized_loss,
    train_step,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--frames-per-trajectory", type=int, default=2)
    parser.add_argument("--max-map-ply-mb", type=float, default=500.0)
    parser.add_argument("--n-maps", type=int, default=8)
    parser.add_argument("--maps", nargs="*", default=None)
    parser.add_argument("--render-size", type=int, default=336)
    parser.add_argument("--crop-radius", type=float, default=120.0)
    parser.add_argument("--max-points", type=int, default=4_000_000)
    parser.add_argument("--splat-radius", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["small", "transformer_m"],
        choices=("small", "transformer_m"),
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "data_cache" / "weights" / "shared_encoder.pt",
    )
    parser.add_argument("--train-vision-encoder", action="store_true", default=True)
    parser.add_argument("--freeze-vision-encoder", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--check-raw", action="store_true", default=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "data_cache" / "fpv_overfit"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "data_cache" / "overfit_checkpoints",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "overfit"
    )
    return parser.parse_args()


def _sanitize(text: object) -> object:
    if isinstance(text, str):
        value = text.replace(str(PROJECT_ROOT), "<PROJECT_ROOT>")
        return value.replace(str(Path.home()), "<HOME>")
    if isinstance(text, dict):
        return {key: _sanitize(value) for key, value in text.items()}
    if isinstance(text, list):
        return [_sanitize(value) for value in text]
    return text


def _cache_key(args: argparse.Namespace) -> str:
    payload = {
        "trajectories": args.trajectories,
        "frames_per": args.frames_per_trajectory,
        "max_map_mb": args.max_map_ply_mb,
        "n_maps": args.n_maps,
        "maps": sorted(args.maps) if args.maps else None,
        "render_size": args.render_size,
        "crop_radius": args.crop_radius,
        "max_points": args.max_points,
        "splat": args.splat_radius,
        "seed": args.seed,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def _select_records(
    dataset: CityNavDataset,
    data_config,
    args: argparse.Namespace,
) -> list[object]:
    groups: dict[str, list] = defaultdict(list)
    for record in dataset.records:
        groups[record.map_name].append(record)
    candidates = []
    for map_name, records in groups.items():
        path = data_config.ply_path(map_name)
        if not path.exists():
            continue
        size_mb = path.stat().st_size / 1e6
        if size_mb > float(args.max_map_ply_mb):
            continue
        if args.maps and map_name not in set(args.maps):
            continue
        if len(records) < 4:
            continue
        candidates.append((size_mb, map_name, records))
    if not candidates:
        raise SystemExit("no eligible train_seen map with a PLY under the size limit")
    candidates.sort(key=lambda item: (item[0], item[1]))
    if args.maps:
        chosen = [item for item in candidates if item[1] in set(args.maps)]
    else:
        chosen = candidates[: max(1, int(args.n_maps))]
    per_map = max(1, int(math.ceil(args.trajectories / len(chosen))))
    selected: list[object] = []
    selected_ids: set[str] = set()
    for offset in range(per_map):
        for _size, map_name, records in chosen:
            index = int(round(offset * (len(records) - 1) / max(per_map - 1, 1)))
            record = records[index]
            if record.episode_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(record.episode_id)
            if len(selected) >= args.trajectories:
                return selected
    if len(selected) < args.trajectories:
        for _size, _map_name, records in candidates:
            for record in records:
                if record.episode_id in selected_ids:
                    continue
                selected.append(record)
                selected_ids.add(record.episode_id)
                if len(selected) >= args.trajectories:
                    return selected
    return selected[: args.trajectories]


def _frames_for_records(
    records: Sequence[object],
    frames_per_trajectory: int,
) -> list[dict]:
    """Select frames with explicit class coverage, not just the first steps.

    Every trajectory contributes one vertical action when possible (Up is
    preferred over Down).  The remaining slots rotate through
    Stop / Forward / Right / Left so the overfit test can measure all six
    classes and Stop precision/recall.
    """
    frames = []
    non_vertical_cycle = [STOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT]
    single_cycle = [GO_UP, GO_DOWN, STOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT]
    for record_index, record in enumerate(records):
        result = record.teacher()
        steps = result.supervision_steps()
        if not steps:
            continue
        actions = [int(action) for _pose, action in steps]
        by_action: dict[int, list[int]] = defaultdict(list)
        for index, action in enumerate(actions):
            by_action[action].append(index)
        chosen: list[int] = []
        if frames_per_trajectory >= 2:
            if by_action.get(int(GO_UP)):
                chosen.append(by_action[int(GO_UP)][len(by_action[int(GO_UP)]) // 2])
            elif by_action.get(int(GO_DOWN)):
                chosen.append(
                    by_action[int(GO_DOWN)][len(by_action[int(GO_DOWN)]) // 2]
                )
            target = non_vertical_cycle[record_index % len(non_vertical_cycle)]
        else:
            target = single_cycle[record_index % len(single_cycle)]
        target_indices = by_action.get(int(target), [])
        if target_indices:
            index = target_indices[len(target_indices) // 2]
            if index not in chosen:
                chosen.append(index)
        for candidate in (
            len(steps) // 2,
            len(steps) // 3,
            2 * len(steps) // 3,
            0,
            len(steps) - 1,
        ):
            if len(chosen) >= frames_per_trajectory:
                break
            if candidate not in chosen:
                chosen.append(candidate)
        for index in chosen[:frames_per_trajectory]:
            pose, action = steps[index]
            frames.append(
                {
                    "record": record,
                    "description": record.description,
                    "map_name": record.map_name,
                    "episode_id": record.episode_id,
                    "object_id": int(record.object_ids[0]),
                    "pose": pose,
                    "action": int(action),
                    "teacher_index": int(index),
                    "teacher_steps": int(result.steps),
                    "teacher_success": bool(result.success),
                    "teacher_reason": result.stop_reason,
                }
            )
    return frames


def _render_cache(
    frames: Sequence[dict],
    data_config,
    observation_config: ObservationConfig,
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    cache_dir = ensure_output_dir(args.cache_dir, data_config=data_config)
    key = _cache_key(args)
    cache_path = cache_dir / f"frames_{key}.npz"
    meta_path = cache_dir / f"frames_{key}.json"
    if cache_path.exists() and meta_path.exists() and not args.rebuild_cache:
        arrays = dict(np.load(cache_path))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if len(arrays["actions"]) == len(metadata["frames"]):
            return arrays, metadata

    camera = camera_intrinsics(observation_config)
    goal_adapter = TopDownObservationAdapter(data_config, observation_config)
    by_map: dict[str, list[int]] = defaultdict(list)
    for index, frame in enumerate(frames):
        by_map[frame["record"].map_name].append(index)
    current_rgbs = [None] * len(frames)
    goal_rgbs = [None] * len(frames)
    depths = [None] * len(frames)
    metadata_frames = [None] * len(frames)
    goal_cache: dict[tuple, np.ndarray] = {}
    render_seconds = 0.0
    crop_points = {}
    try:
        for map_name, indices in by_map.items():
            poses = np.asarray(
                [[frames[i]["pose"].x, frames[i]["pose"].y] for i in indices]
            )
            low, high = poses.min(axis=0), poses.max(axis=0)
            center = (low + high) / 2.0
            radius = float(np.linalg.norm(high - low) / 2.0 + 50.0)
            radius = min(max(radius, float(args.crop_radius)), 450.0)
            cloud = open_ply_cloud(data_config.ply_path(map_name))
            crop = crop_ply_cloud(cloud, center, radius)
            xyz, rgb = crop.xyz, crop.rgb
            if args.max_points and len(xyz) > args.max_points:
                rng = np.random.default_rng(
                    args.seed + int(hashlib.sha1(map_name.encode()).hexdigest()[:8], 16)
                )
                keep = rng.choice(len(xyz), int(args.max_points), replace=False)
                xyz, rgb = xyz[keep], rgb[keep]
            crop_points[map_name] = int(len(xyz))
            for index in indices:
                frame = frames[index]
                record = frame["record"]
                pose = frame["pose"]
                start = time.perf_counter()
                result = render_point_cloud(
                    xyz,
                    rgb,
                    (pose.x, pose.y, pose.z),
                    pose.yaw,
                    camera_pitch(pose, observation_config),
                    camera,
                    near=float(observation_config.fpv_near_m),
                    far=float(observation_config.fpv_far_m),
                    splat_radius=int(observation_config.fpv_splat_radius),
                )
                key_goal = (record.map_name, int(record.object_ids[0]))
                if key_goal not in goal_cache:
                    goal_cache[key_goal] = goal_adapter.render_goal_crops(
                        record.map_name, record.object_info
                    )
                current_rgbs[index] = result.rgb
                goal_rgbs[index] = goal_cache[key_goal]
                depths[index] = result.depth.astype(np.float16)
                render_seconds += time.perf_counter() - start
                metadata_frames[index] = {
                    "split": record.split,
                    "map_name": record.map_name,
                    "episode_id": record.episode_id,
                    "object_id": int(record.object_ids[0]),
                    "description": record.description,
                    "teacher_index": frame["teacher_index"],
                    "teacher_steps": frame["teacher_steps"],
                    "teacher_success": frame["teacher_success"],
                    "teacher_reason": frame["teacher_reason"],
                    "action": frame["action"],
                    "pose": [round(float(v), 5) for v in pose.as_tuple()],
                    "target_xyz": [round(float(v), 4) for v in record.target_xyz],
                    "marker_xyz": [round(float(v), 4) for v in record.marker_xyz],
                    "render_quality": {
                        "valid_pixel_ratio": round(float(result.valid_pixel_ratio), 4),
                        "depth_min_m": (
                            None if not result.valid_mask.any()
                            else round(float(result.depth[result.valid_mask].min()), 3)
                        ),
                        "depth_median_m": (
                            None if not result.valid_mask.any()
                            else round(float(np.median(result.depth[result.valid_mask])), 3)
                        ),
                    },
                }
            del crop, cloud, xyz, rgb
    finally:
        goal_adapter.close()

    arrays = {
        "current_rgbs": np.stack(current_rgbs),
        "goal_rgbs": np.stack(goal_rgbs),
        "depths": np.stack(depths),
        "actions": np.asarray([frame["action"] for frame in frames], dtype=np.int64),
        "poses": np.asarray(
            [[frame["pose"].x, frame["pose"].y, frame["pose"].z,
              frame["pose"].yaw, frame["pose"].pitch] for frame in frames],
            dtype=np.float32,
        ),
    }
    np.savez_compressed(cache_path, **arrays)
    metadata = {
        "cache_key": key,
        "cache_file": str(cache_path.relative_to(PROJECT_ROOT)),
        "cache_bytes": int(cache_path.stat().st_size),
        "render_seconds": round(float(render_seconds), 3),
        "camera": {
            **camera.as_dict(),
            "pitch_mode": observation_config.fpv_pitch_mode,
            "near_m": observation_config.fpv_near_m,
            "far_m": observation_config.fpv_far_m,
            "depth_definition": "camera_forward_depth_m",
        },
        "crop_points": crop_points,
        "frames": metadata_frames,
        "action_counts": dict(Counter(int(frame["action"]) for frame in frames)),
        "test_unseen_read": False,
    }
    meta_path.write_text(
        json.dumps(_sanitize(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return arrays, metadata


def _load_bert(data_config):
    adapter, text_dim, backend, warning = build_language_adapter(
        TextConfig(backend="bert", local_files_only=True),
        device="cpu",
    )
    if backend != "bert":
        raise SystemExit(f"formal training requires BERT, got {backend}")
    parameters = list(adapter.model.parameters())
    fingerprint = float(
        sum(parameter.detach().double().sum().item() for parameter in parameters)
    )
    return adapter, text_dim, parameters, fingerprint, warning


def _language_features(
    adapter,
    frames: Sequence[dict],
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    texts = [frame["description"] for frame in frames]
    unique = sorted(set(texts))
    with torch.no_grad():
        encoded = adapter.encode_tensor(unique, device="cpu")
    mapping = {text: index for index, text in enumerate(unique)}
    features = torch.stack([encoded[mapping[text]] for text in texts]).to(device)
    return features, {
        "backend": "bert",
        "model": "bert-base-uncased",
        "unique_descriptions": len(unique),
        "feature_dim": int(encoded.size(1)),
    }


def _parameter_vector(module: torch.nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(
        [parameter.detach().float().cpu() for parameter in module.parameters()]
    )


def _grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().double().pow(2).sum().item())
    return math.sqrt(total)


def _make_batch(
    arrays: dict,
    language_features: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = torch.from_numpy(
        arrays["actions"][indices].astype(np.int64)
    ).to(device)
    language = language_features[indices].to(device)
    network = torch.from_numpy(
        np.stack(
            [placeholder_network_state(seed + int(index)) for index in indices]
        )
    ).float().to(device)
    return (
        arrays["current_rgbs"][indices],
        arrays["goal_rgbs"][indices],
        actions,
        language,
        network,
    )


def _forward_inputs(
    encoder,
    current_rgbs: np.ndarray,
    goal_rgbs: np.ndarray,
    language: torch.Tensor,
    network: torch.Tensor,
    device: torch.device,
    *,
    requires_grad: bool,
) -> dict:
    scene = encode_scene(
        encoder,
        current_rgbs,
        goal_rgbs,
        device=device,
        requires_grad=requires_grad,
    )
    return {
        "current_tokens": scene["current_tokens"],
        "goal_tokens": scene["goal_tokens"],
        "heatmap": scene["heatmap"],
        "network_state": network,
        "language_features": language,
    }


@torch.no_grad()
def _evaluate(
    model,
    encoder,
    arrays: dict,
    language_features: torch.Tensor,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> dict:
    model.eval()
    encoder.eval()
    total = len(arrays["actions"])
    loss_sum = 0.0
    accuracy_correct = 0
    move_correct = 0
    move_total = 0
    counts = Counter()
    correct = Counter()
    confusion = np.zeros((6, 6), dtype=np.int64)
    stop_tp = stop_fp = stop_fn = 0
    start = time.perf_counter()
    for begin in range(0, total, batch_size):
        indices = np.arange(begin, min(begin + batch_size, total))
        current, goal, actions, language, network = _make_batch(
            arrays, language_features, indices, device, seed
        )
        inputs = _forward_inputs(
            encoder, current, goal, language, network, device,
            requires_grad=False,
        )
        components = model.forward_components(**inputs)
        losses = factorized_loss(components, actions)
        batch = len(indices)
        loss_sum += float(losses["loss"].item()) * batch
        accuracy_correct += int(
            round(float(losses["accuracy"]) * batch)
        )
        predictions = losses["predictions"].detach().cpu().numpy()
        true = actions.detach().cpu().numpy()
        np.add.at(confusion, (true, predictions), 1)
        for _action, action_name in ACTION_NAMES.items():
            counts[action_name] += int(
                losses["per_class_counts"][action_name]
            )
            correct[action_name] += int(
                losses["per_class_correct"][action_name]
            )
        move_mask = true != int(STOP)
        move_total += int(move_mask.sum())
        move_correct += int((predictions[move_mask] == true[move_mask]).sum())
        stop_true = true == int(STOP)
        stop_pred = predictions == int(STOP)
        stop_tp += int((stop_true & stop_pred).sum())
        stop_fp += int((~stop_true & stop_pred).sum())
        stop_fn += int((stop_true & ~stop_pred).sum())
    metrics = {
        "frames": int(total),
        "loss": round(loss_sum / max(total, 1), 6),
        "accuracy": round(accuracy_correct / max(total, 1), 6),
        "move_accuracy": round(move_correct / max(move_total, 1), 6),
        "stop_precision": round(stop_tp / max(stop_tp + stop_fp, 1), 6),
        "stop_recall": round(stop_tp / max(stop_tp + stop_fn, 1), 6),
        "per_class_counts": dict(counts),
        "per_class_accuracy": {
            name: round(correct[name] / counts[name], 6) if counts[name] else 0.0
            for name in counts
        },
        "confusion": confusion.tolist(),
        "eval_seconds": round(float(time.perf_counter() - start), 3),
    }
    return metrics


def _train_model(
    name: str,
    model,
    encoder,
    arrays: dict,
    language_features: torch.Tensor,
    bert_parameters: Sequence[torch.nn.Parameter],
    bert_fingerprint_before: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model = model.to(device)
    encoder = encoder.to(device)
    trainable_encoder = bool(args.train_vision_encoder)
    if args.freeze_vision_encoder:
        trainable_encoder = False
    encoder.requires_grad_(trainable_encoder)
    if trainable_encoder:
        encoder.train()
    else:
        encoder.eval()
    model.train()

    groups = []
    if trainable_encoder:
        groups.append(
            {"params": list(encoder.parameters()), "lr": float(args.encoder_lr)}
        )
    groups.append({"params": list(model.parameters()), "lr": float(args.lr)})
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(args.weight_decay)
    )
    clip_parameters = (
        list(encoder.parameters()) + list(model.parameters())
        if trainable_encoder
        else None
    )
    training = TrainingConfig(
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        encoder_lr=float(args.encoder_lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        train_vision_encoder=trainable_encoder,
        freeze_text_encoder=True,
        seed=int(args.seed),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_eval = _evaluate(
        model, encoder, arrays, language_features, device, args.seed,
        args.batch_size,
    )
    encoder_before = _parameter_vector(encoder)
    model_before = _parameter_vector(model)
    trace = []
    grad_checks = {}
    total = len(arrays["actions"])
    rng = np.random.default_rng(args.seed + (0 if name == "Small" else 1))
    start = time.perf_counter()
    for step in range(int(args.steps)):
        indices = rng.choice(total, size=min(int(args.batch_size), total), replace=False)
        current, goal, actions, language, network = _make_batch(
            arrays, language_features, indices, device, args.seed
        )
        if trainable_encoder:
            encoder.train()
        inputs = _forward_inputs(
            encoder, current, goal, language, network, device,
            requires_grad=trainable_encoder,
        )
        result = train_step(
            model,
            optimizer,
            {**inputs, "actions": actions},
            training,
            clip_parameters=clip_parameters,
        )
        trace.append(
            {
                "step": int(step),
                "loss": round(float(result["loss"]), 6),
                "accuracy": round(float(result["accuracy"]), 6),
                "move_accuracy": round(float(result["move_accuracy"]), 6),
                "stop_precision": round(float(result["stop_precision"]), 6),
                "stop_recall": round(float(result["stop_recall"]), 6),
                "grad_norm": round(float(result["grad_norm"]), 6),
            }
        )
        if step == 0:
            grad_checks = {
                "encoder_grad_norm": _grad_norm(encoder)
                if trainable_encoder else 0.0,
                "mec_grad_norm": _grad_norm(model),
                "encoder_gradients_finite": all(
                    torch.isfinite(parameter.grad).all().item()
                    for parameter in encoder.parameters()
                    if parameter.grad is not None
                ),
                "mec_gradients_finite": all(
                    torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ),
                "bert_grad_parameters": sum(
                    1 for parameter in bert_parameters if parameter.grad is not None
                ),
            }
    train_seconds = time.perf_counter() - start
    final_eval = _evaluate(
        model, encoder, arrays, language_features, device, args.seed,
        args.batch_size,
    )
    encoder_after = _parameter_vector(encoder)
    model_after = _parameter_vector(model)
    bert_fingerprint_after = float(
        sum(parameter.detach().double().sum().item() for parameter in bert_parameters)
    )
    encoder_delta = float((encoder_after - encoder_before).norm())
    model_delta = float((model_after - model_before).norm())

    peak_memory = None
    if device.type == "cuda":
        peak_memory = {
            "max_allocated_mb": round(
                torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2
            ),
            "max_reserved_mb": round(
                torch.cuda.max_memory_reserved(device) / 1024 ** 2, 2
            ),
        }

    checkpoint_dir = ensure_output_dir(args.checkpoint_dir / name.lower())
    torch.save(encoder.state_dict(), checkpoint_dir / "encoder.pt")
    torch.save(model.state_dict(), checkpoint_dir / "mec.pt")

    return {
        "model": name,
        "parameters": {
            "encoder": int(sum(p.numel() for p in encoder.parameters())),
            "mec": int(sum(p.numel() for p in model.parameters())),
            "trainable_encoder": bool(trainable_encoder),
        },
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "training_trace": trace,
        "gradient_checks": {
            **grad_checks,
            "encoder_parameter_delta_l2": round(encoder_delta, 8),
            "mec_parameter_delta_l2": round(model_delta, 8),
            "bert_fingerprint_before": round(bert_fingerprint_before, 6),
            "bert_fingerprint_after": round(bert_fingerprint_after, 6),
            "bert_parameter_delta": round(
                abs(bert_fingerprint_after - bert_fingerprint_before), 8
            ),
            "bert_requires_grad_any": any(
                parameter.requires_grad for parameter in bert_parameters
            ),
            "encoder_updated_as_expected": (
                encoder_delta > 0.0 if trainable_encoder else encoder_delta == 0.0
            ),
            "mec_updated": bool(model_delta > 0.0),
            "bert_unchanged": bool(
                abs(bert_fingerprint_after - bert_fingerprint_before) < 1e-9
                and not any(parameter.requires_grad for parameter in bert_parameters)
            ),
        },
        "timing": {
            "train_seconds": round(float(train_seconds), 3),
            "steps_per_second": round(
                float(args.steps) / max(train_seconds, 1e-9), 3
            ),
            "frames_per_second": round(
                float(args.steps * args.batch_size) / max(train_seconds, 1e-9), 3
            ),
            "eval_seconds": final_eval["eval_seconds"],
        },
        "gpu_memory": peak_memory,
        "checkpoints": {
            "encoder": str((checkpoint_dir / "encoder.pt").relative_to(PROJECT_ROOT)),
            "mec": str((checkpoint_dir / "mec.pt").relative_to(PROJECT_ROOT)),
        },
    }


def _markdown(report: dict) -> str:
    lines = [
        "# FPV 六动作过拟合验证",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 设备: `{report['device']}`",
        f"- 轨迹数: {report['dataset']['trajectories']}",
        f"- FPV 帧数: {report['dataset']['frames']}",
        f"- BERT: {report['bert']['model']}（冻结，{report['bert']['feature_dim']} 维）",
        f"- 动作计数: {report['dataset']['action_counts']}",
        f"- 原始数据快照未变化: `{report['raw_data_unchanged']}`",
        "",
        "## 缓存与渲染",
        "",
        f"- 缓存: `{report['cache']['cache_file']}`",
        f"- 缓存大小: {report['cache']['cache_bytes'] / 1e6:.1f} MB",
        f"- 渲染耗时: {report['cache']['render_seconds']} s",
        f"- 每图点数: {report['cache']['crop_points']}",
        "",
        "## 过拟合结果",
        "",
        "| model | trainable encoder | initial acc | final acc | final move acc | stop P/R | train s | peak GPU MB |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["models"]:
        peak = (
            "n/a"
            if result["gpu_memory"] is None
            else str(result["gpu_memory"]["max_allocated_mb"])
        )
        lines.append(
            f"| {result['model']} | {result['parameters']['trainable_encoder']} | "
            f"{result['initial_eval']['accuracy']:.4f} | "
            f"{result['final_eval']['accuracy']:.4f} | "
            f"{result['final_eval']['move_accuracy']:.4f} | "
            f"{result['final_eval']['stop_precision']:.3f}/"
            f"{result['final_eval']['stop_recall']:.3f} | "
            f"{result['timing']['train_seconds']} | {peak} |"
        )
    lines += [
        "",
        "## 六类动作（最终评估）",
        "",
        "| model | counts | per-class accuracy |",
        "|---|---|---|",
    ]
    for result in report["models"]:
        lines.append(
            f"| {result['model']} | {result['final_eval']['per_class_counts']} | "
            f"{result['final_eval']['per_class_accuracy']} |"
        )
    lines += [
        "",
        "## 梯度与冻结检查",
        "",
        "| model | encoder grad norm | MEC grad norm | encoder delta | MEC delta | BERT grad params | BERT changed |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in report["models"]:
        checks = result["gradient_checks"]
        lines.append(
            f"| {result['model']} | {checks['encoder_grad_norm']:.4f} | "
            f"{checks['mec_grad_norm']:.4f} | "
            f"{checks['encoder_parameter_delta_l2']:.6f} | "
            f"{checks['mec_parameter_delta_l2']:.6f} | "
            f"{checks['bert_grad_parameters']} | "
            f"{not checks['bert_unchanged']} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        "这是小规模过拟合验证，不是正式训练结果，也不代表 val 泛化；"
        "仅使用 train_seen，未读取 test_unseen。网络状态仍为占位向量。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    if args.trajectories < 128 or args.trajectories > 256:
        raise SystemExit("--trajectories must be between 128 and 256")
    if args.encoder_checkpoint and not Path(args.encoder_checkpoint).exists():
        raise SystemExit(f"encoder checkpoint not found: {args.encoder_checkpoint}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    data_config = load_data_config()
    data_config.validate()
    before = snapshot_raw_dirs(data_config) if args.check_raw else None

    dataset = CityNavDataset(
        data_config,
        "train_seen",
        include_processed_descriptions=False,
    )
    records = _select_records(dataset, data_config, args)
    frames = _frames_for_records(records, int(args.frames_per_trajectory))
    if not frames:
        raise SystemExit("no teacher supervision frame could be selected")
    observation_config = ObservationConfig(
        image_size=336,
        depth_size=int(args.render_size),
        goal_crop_size=96,
        goal_scales=(14, 28, 56),
        map_cache_size=2,
        fpv_width=int(args.render_size),
        fpv_height=int(args.render_size),
        fpv_crop_radius_m=float(args.crop_radius),
        fpv_splat_radius=int(args.splat_radius),
    )
    arrays, cache_meta = _render_cache(
        frames, data_config, observation_config, args
    )
    bert_adapter, text_dim, bert_parameters, bert_fingerprint, bert_warning = (
        _load_bert(data_config)
    )
    language_features, bert_meta = _language_features(
        bert_adapter, frames, device
    )
    model_map = {
        "small": SmallMEC,
        "transformer_m": TransformerMMEC,
    }
    results = []
    for model_name in args.models:
        encoder = VisualTokenEncoder()
        load_encoder_checkpoint(
            encoder,
            str(args.encoder_checkpoint),
            map_location="cpu",
        )
        model = model_map[model_name](text_dim=text_dim)
        results.append(
            _train_model(
                "Small" if model_name == "small" else "Transformer-M",
                model,
                encoder,
                arrays,
                language_features,
                bert_parameters,
                bert_fingerprint,
                args,
                device,
            )
        )
    bert_adapter.close()

    after = snapshot_raw_dirs(data_config) if args.check_raw else None
    if before is not None and after is not None:
        assert_snapshot_unchanged(before, after)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "dataset": {
            "split": "train_seen",
            "trajectories": len(records),
            "frames": len(frames),
            "action_counts": {
                ACTION_NAMES[action]: int(count)
                for action, count in sorted(
                    Counter(int(frame["action"]) for frame in frames).items()
                )
            },
            "teacher_stop_distance_m": float(dataset.teacher_config.stop_distance_m),
            "success_radius_m": float(dataset.teacher_config.success_radius_m),
            "test_unseen_read": False,
        },
        "cache": cache_meta,
        "bert": {
            **bert_meta,
            "parameters": int(sum(p.numel() for p in bert_parameters)),
            "requires_grad_any": any(p.requires_grad for p in bert_parameters),
            "warning": _sanitize(bert_warning),
        },
        "models": results,
        "raw_data_unchanged": True if before is not None else None,
        "raw_snapshot": (
            None
            if after is None
            else {
                "file_count": int(after["file_count"]),
                "total_bytes": int(after["total_bytes"]),
                "digest": str(after["digest"]),
            }
        ),
        "note": (
            "Small overfit validation only; no val_unseen or test_unseen labels "
            "were used for training or selection."
        ),
    }
    output_dir = Path(args.output_dir)
    json_path = ensure_output_file(
        output_dir / "overfit_report.json", data_config=data_config
    )
    md_path = ensure_output_file(
        output_dir / "overfit_report.md", data_config=data_config
    )
    json_path.write_text(
        json.dumps(_sanitize(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    print("FPV six-action overfit smoke completed")
    print(
        f"  trajectories={len(records)} frames={len(frames)} "
        f"cache={cache_meta['cache_file']} device={device}"
    )
    for result in results:
        print(
            f"  {result['model']:14s} initial_acc={result['initial_eval']['accuracy']:.4f} "
            f"final_acc={result['final_eval']['accuracy']:.4f} "
            f"move_acc={result['final_eval']['move_accuracy']:.4f} "
            f"stop_P/R={result['final_eval']['stop_precision']:.3f}/"
            f"{result['final_eval']['stop_recall']:.3f} "
            f"encoder_delta={result['gradient_checks']['encoder_parameter_delta_l2']:.6f} "
            f"mec_delta={result['gradient_checks']['mec_parameter_delta_l2']:.6f} "
            f"bert_grad={result['gradient_checks']['bert_grad_parameters']}"
        )
    print(f"  wrote {json_path.relative_to(PROJECT_ROOT)} and {md_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
