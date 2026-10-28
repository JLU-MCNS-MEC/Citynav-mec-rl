#!/usr/bin/env python3
"""CityNav shared visual-encoder pretraining.

Only the encoder is saved.  Action/cross-view/retrieval heads are temporary
auxiliary heads and are never official MEC checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    ObservationConfig,
    PROJECT_ROOT,
    load_data_config,
    load_yaml,
)
from src.data.actions import ACTION_NAMES, ACTION_ORDER  # noqa: E402
from src.data.paths import (  # noqa: E402
    assert_snapshot_unchanged,
    ensure_output_dir,
    snapshot_raw_dirs,
)
from experiments.visual_encoder_citynav.data import (  # noqa: E402
    CANDIDATE_SIZE,
    CURRENT_SIZE,
    MAX_POINTS_DEFAULT,
    CandidateStore,
    PointCloudStore,
    PretrainDataset,
    build_manifest,
)
from experiments.visual_encoder_citynav.losses import (  # noqa: E402
    multi_positive_infonce,
    weighted_action_loss,
)
from experiments.visual_encoder_citynav.model import (  # noqa: E402
    PretrainModel,
    encoder_sha256,
    load_pretrained_encoder,
    save_encoder_only,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "pretrain.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--rebuild-manifests", action="store_true")
    parser.add_argument("--rebuild-clouds", action="store_true")
    parser.add_argument("--rebuild-candidates", action="store_true")
    return parser.parse_args()


def _config(args) -> dict:
    cfg = load_yaml(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.val_samples is not None:
        cfg["val_samples"] = int(args.val_samples)
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)
    return cfg


def _macro_f1(confusion: np.ndarray) -> float:
    f1s = []
    for index in range(6):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return float(np.mean(f1s))


def _class_weights(actions: np.ndarray) -> tuple[np.ndarray, dict]:
    counts = np.bincount(
        np.asarray(actions, dtype=np.int64), minlength=6
    ).astype(np.float64)
    total = counts.sum()
    weights = np.asarray(
        [total / (6.0 * max(count, 1.0)) for count in counts],
        dtype=np.float32,
    )
    return weights, {
        ACTION_NAMES[int(index)]: int(counts[index])
        for index in ACTION_ORDER
    }


def _to_device_batch(batch: dict, device: torch.device) -> dict:
    current = (
        batch["current_rgb"].float().permute(0, 3, 1, 2).to(device) / 255.0
    )
    candidates = (
        batch["candidate_rgbs"].float().permute(0, 1, 4, 2, 3).to(device)
        / 255.0
    )
    return {
        "current": current,
        "candidates": candidates,
        "language": batch["language"].float().to(device),
        "actions": batch["action"].long().to(device),
        "current_mask": batch["current_mask"].float().to(device),
        "target_mask": batch["target_mask"].float().to(device),
        "current_valid": batch["current_valid"].float().to(device),
        "target_valid": batch["target_valid"].float().to(device),
        "valid_pixel_ratio": batch["valid_pixel_ratio"].float().to(device),
        "render_seconds": batch["render_seconds"].float().to(device),
        "map_id": batch["map_id"].long().to(device),
        "sample_index": batch["sample_index"].long().to(device),
    }


def _compute_losses(
    model: PretrainModel,
    batch: dict,
    class_weights: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict]:
    outputs = model(
        batch["current"], batch["candidates"], batch["language"]
    )
    action = weighted_action_loss(
        outputs["action_logits"], batch["actions"], class_weights
    )
    xview, xview_metrics = multi_positive_infonce(
        outputs["xview_query"],
        outputs["xview_candidate"],
        batch["current_mask"],
        batch["current_valid"],
        temperature=float(cfg["temperature"]),
        logit_scale=outputs["xview_logit_scale"],
    )
    retrieval, retrieval_metrics = multi_positive_infonce(
        outputs["retrieval_query"],
        outputs["retrieval_candidate"],
        batch["target_mask"],
        batch["target_valid"],
        temperature=float(cfg["temperature"]),
        logit_scale=outputs["retrieval_logit_scale"],
    )
    total = (
        action
        + float(cfg["lambda_xview"]) * xview
        + float(cfg["lambda_retrieval"]) * retrieval
    )
    return total, {
        "action_loss": action,
        "xview_loss": xview,
        "retrieval_loss": retrieval,
        "xview_metrics": xview_metrics,
        "retrieval_metrics": retrieval_metrics,
        "action_logits": outputs["action_logits"],
    }


def _bias_zero_logit_scale(module: PretrainModel) -> None:
    with torch.no_grad():
        module.xview_logit_scale.zero_()
        module.retrieval_logit_scale.zero_()


@torch.no_grad()
def _candidate_embeddings(
    model: PretrainModel,
    candidate_store: CandidateStore,
    map_names: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output = {}
    for map_name in map_names:
        library = candidate_store.get(map_name)
        images = (
            torch.from_numpy(library.images)
            .float()
            .permute(0, 3, 1, 2)
            / 255.0
        ).to(device)
        output[map_name] = model.encode_candidate_images(images)
    return output


def _evaluate(
    model: PretrainModel,
    dataset: PretrainDataset,
    manifest: dict,
    candidate_store: CandidateStore,
    device: torch.device,
    cfg: dict,
) -> dict:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("val_batch_size", 32)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    actions = []
    logits = []
    languages = []
    globals_ = []
    map_ids = []
    sample_indices = []
    valid_ratios = []
    render_times = []
    for batch in loader:
        current = batch["current_rgb"].float().permute(0, 3, 1, 2).to(device) / 255.0
        language = batch["language"].float().to(device)
        tokens, current_global = model.encode_current(current)
        action_logits = model.action_head(
            torch.cat([current_global, language], dim=-1)
        )
        globals_.append(current_global.detach().cpu())
        languages.append(language.detach().cpu())
        logits.append(action_logits.detach().cpu())
        actions.append(batch["action"].long())
        map_ids.append(batch["map_id"].long())
        sample_indices.append(batch["sample_index"].long())
        valid_ratios.append(batch["valid_pixel_ratio"].float())
        render_times.append(batch["render_seconds"].float())
    if not actions:
        return {"samples": 0}
    actions = torch.cat(actions).numpy()
    logits = torch.cat(logits).numpy()
    languages = torch.cat(languages).numpy()
    globals_ = torch.cat(globals_)
    map_ids = torch.cat(map_ids).numpy()
    sample_indices = torch.cat(sample_indices).numpy()
    valid_ratios = torch.cat(valid_ratios).numpy()
    render_times = torch.cat(render_times).numpy()
    candidate_embs = _candidate_embeddings(
        model, candidate_store, list(manifest["map_names"]), device
    )
    map_names = list(manifest["map_names"])
    target_xy = np.asarray(manifest["target_xy"], dtype=np.float64)
    pose = np.asarray(manifest["pose"], dtype=np.float64)
    confusion = np.zeros((6, 6), dtype=np.int64)
    per_map = defaultdict(
        lambda: {
            "frames": 0,
            "action_correct": 0,
            "current_positive": 0,
            "current_top1": 0,
            "current_top5": 0,
            "target_positive": 0,
            "retrieval_top1": 0,
            "retrieval_top5": 0,
            "retrieval_top10": 0,
            "valid_ratio_sum": 0.0,
            "candidate_count": 0,
        }
    )
    xview_losses = []
    retrieval_losses = []
    action_losses = []
    xview_top1 = []
    xview_top5 = []
    retrieval_r1 = []
    retrieval_r5 = []
    retrieval_r10 = []
    for row in range(len(actions)):
        map_name = map_names[int(map_ids[row])]
        map_group = per_map[map_name]
        map_group["frames"] += 1
        map_group["valid_ratio_sum"] += float(valid_ratios[row])
        library = candidate_store.get(map_name)
        centers = library.centers
        emb = candidate_embs[map_name]
        current_xy = pose[int(sample_indices[row]), :2]
        target = target_xy[int(sample_indices[row])]
        d_current = np.linalg.norm(
            centers.astype(np.float64) - current_xy[None, :], axis=1
        )
        d_target = np.linalg.norm(
            centers.astype(np.float64) - target[None, :], axis=1
        )
        current_mask = d_current <= float(cfg["current_match_radius_m"])
        target_mask = d_target <= float(cfg["target_match_radius_m"])
        map_group["candidate_count"] = int(library.count)
        confusion[int(actions[row]), int(logits[row].argmax())] += 1
        if int(actions[row]) == int(logits[row].argmax()):
            map_group["action_correct"] += 1
        language = torch.from_numpy(languages[row]).float().to(device)[None]
        current_global = globals_[row].to(device)[None]
        xq = model.xview_query(current_global)
        xc = model.xview_candidate(emb)
        x_scores = F.normalize(xq, dim=-1) @ F.normalize(xc, dim=-1).T
        rq = model.retrieval_query(language)
        rc = model.retrieval_candidate(emb)
        r_scores = F.normalize(rq, dim=-1) @ F.normalize(rc, dim=-1).T
        if bool(current_mask.any()):
            map_group["current_positive"] += 1
            top1 = int(x_scores[0].argmax().item())
            top5 = torch.topk(x_scores[0], k=min(5, x_scores.size(1))).indices
            hit1 = bool(current_mask[top1])
            hit5 = bool(current_mask[top5.cpu().numpy()].any())
            map_group["current_top1"] += int(hit1)
            map_group["current_top5"] += int(hit5)
            xview_top1.append(hit1)
            xview_top5.append(hit5)
            mask = torch.from_numpy(current_mask.astype(np.float32)).to(device)[None]
            valid = torch.ones((1, len(centers)), dtype=torch.float32, device=device)
            x_loss, _ = multi_positive_infonce(
                xq, xc[None], mask, valid,
                temperature=float(cfg["temperature"]),
                logit_scale=model.xview_logit_scale,
            )
            xview_losses.append(float(x_loss.item()))
        if bool(target_mask.any()):
            map_group["target_positive"] += 1
            top1 = int(r_scores[0].argmax().item())
            top5 = torch.topk(r_scores[0], k=min(5, r_scores.size(1))).indices
            top10 = torch.topk(r_scores[0], k=min(10, r_scores.size(1))).indices
            hit1 = bool(target_mask[top1])
            hit5 = bool(target_mask[top5.cpu().numpy()].any())
            hit10 = bool(target_mask[top10.cpu().numpy()].any())
            map_group["retrieval_top1"] += int(hit1)
            map_group["retrieval_top5"] += int(hit5)
            map_group["retrieval_top10"] += int(hit10)
            retrieval_r1.append(hit1)
            retrieval_r5.append(hit5)
            retrieval_r10.append(hit10)
            mask = torch.from_numpy(target_mask.astype(np.float32)).to(device)[None]
            valid = torch.ones((1, len(centers)), dtype=torch.float32, device=device)
            r_loss, _ = multi_positive_infonce(
                rq, rc[None], mask, valid,
                temperature=float(cfg["temperature"]),
                logit_scale=model.retrieval_logit_scale,
            )
            retrieval_losses.append(float(r_loss.item()))
    logits_tensor = torch.from_numpy(logits)
    actions_tensor = torch.from_numpy(actions)
    action_loss = float(
        F.cross_entropy(logits_tensor, actions_tensor).item()
    )
    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    per_class = {
        ACTION_NAMES[index]: (
            float(confusion[index, index] / max(confusion[index].sum(), 1))
        )
        for index in ACTION_ORDER
    }
    per_map_output = {}
    for map_name, group in per_map.items():
        frames = group["frames"]
        per_map_output[map_name] = {
            "frames": frames,
            "action_accuracy": group["action_correct"] / max(frames, 1),
            "current_positive_samples": group["current_positive"],
            "current_top1": (
                group["current_top1"] / group["current_positive"]
                if group["current_positive"] else None
            ),
            "current_top5": (
                group["current_top5"] / group["current_positive"]
                if group["current_positive"] else None
            ),
            "target_positive_samples": group["target_positive"],
            "retrieval_recall@1": (
                group["retrieval_top1"] / group["target_positive"]
                if group["target_positive"] else None
            ),
            "retrieval_recall@5": (
                group["retrieval_top5"] / group["target_positive"]
                if group["target_positive"] else None
            ),
            "retrieval_recall@10": (
                group["retrieval_top10"] / group["target_positive"]
                if group["target_positive"] else None
            ),
            "valid_pixel_ratio_mean": group["valid_ratio_sum"] / max(frames, 1),
            "candidate_count": group["candidate_count"],
        }
    xview_top1_mean = float(np.mean(xview_top1)) if xview_top1 else None
    xview_top5_mean = float(np.mean(xview_top5)) if xview_top5 else None
    retrieval_r1_mean = float(np.mean(retrieval_r1)) if retrieval_r1 else None
    retrieval_r5_mean = float(np.mean(retrieval_r5)) if retrieval_r5 else None
    retrieval_r10_mean = float(np.mean(retrieval_r10)) if retrieval_r10 else None
    macro_f1 = _macro_f1(confusion)
    selection_score = (
        0.4 * macro_f1
        + 0.3 * (xview_top1_mean or 0.0)
        + 0.3 * (retrieval_r5_mean or 0.0)
    )
    return {
        "samples": int(len(actions)),
        "action": {
            "loss": action_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class_accuracy": per_class,
            "confusion": confusion.tolist(),
            "counts": {
                ACTION_NAMES[index]: int(confusion[index].sum())
                for index in ACTION_ORDER
            },
        },
        "xview": {
            "samples_with_current_positive": len(xview_top1),
            "samples_without_current_positive": int(len(actions) - len(xview_top1)),
            "top1": xview_top1_mean,
            "top5": xview_top5_mean,
            "loss": float(np.mean(xview_losses)) if xview_losses else None,
        },
        "retrieval": {
            "samples_with_target_positive": len(retrieval_r1),
            "samples_without_target_positive": int(len(actions) - len(retrieval_r1)),
            "recall@1": retrieval_r1_mean,
            "recall@5": retrieval_r5_mean,
            "recall@10": retrieval_r10_mean,
            "loss": float(np.mean(retrieval_losses)) if retrieval_losses else None,
        },
        "valid_pixel_ratio_mean": float(valid_ratios.mean()),
        "render_seconds_mean": float(render_times.mean()),
        "selection_score": float(selection_score),
        "per_map": per_map_output,
    }


def _checkpoint_steps(total_steps: int, fractions: Sequence[float]) -> set[int]:
    values = {
        min(total_steps, max(1, int(round(total_steps * float(fraction)))))
        for fraction in fractions
    }
    return values


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _relative_payload(value):
    """Make committed summary JSON free of server absolute paths."""
    prefix = str(PROJECT_ROOT) + "/"
    if isinstance(value, str):
        return value[len(prefix):] if value.startswith(prefix) else value
    if isinstance(value, dict):
        return {key: _relative_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relative_payload(item) for item in value]
    return value


def main() -> int:
    args = _parse_args()
    cfg = _config(args)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    data_config = load_data_config()
    data_config.validate()
    before = snapshot_raw_dirs(data_config)

    run_name = args.run_name or (
        f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if args.smoke else f"full_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = ensure_output_dir(PROJECT_ROOT / "runs" / "visual_encoder_citynav" / run_name)
    manifest_dir = PROJECT_ROOT / "data_cache" / "visual_encoder_citynav" / "manifests"
    if args.smoke:
        train_max = int(cfg["smoke"]["train_samples"])
        val_max = int(cfg["smoke"]["val_samples"])
    else:
        train_max = args.max_train_records
        val_max = int(cfg["val_samples"])
    train_manifest = build_manifest(
        data_config,
        "train_seen",
        max_records=train_max,
        seed=seed,
        output_dir=manifest_dir,
        rebuild=args.rebuild_manifests,
        steps_per_trajectory=int(cfg.get("steps_per_trajectory", 1)),
    )
    val_seen_manifest = build_manifest(
        data_config,
        "val_seen",
        max_records=val_max,
        seed=seed,
        output_dir=manifest_dir,
        rebuild=args.rebuild_manifests,
    )
    val_unseen_manifest = build_manifest(
        data_config,
        "val_unseen",
        max_records=val_max,
        seed=seed,
        output_dir=manifest_dir,
        rebuild=args.rebuild_manifests,
    )
    all_maps = sorted(
        set(train_manifest["map_names"])
        | set(val_seen_manifest["map_names"])
        | set(val_unseen_manifest["map_names"])
    )
    candidate_store = CandidateStore(
        data_config,
        ObservationConfig(
            image_size=CURRENT_SIZE,
            goal_crop_size=CANDIDATE_SIZE,
            fpv_width=CURRENT_SIZE,
            fpv_height=CURRENT_SIZE,
            fpv_splat_radius=0,
        ),
        stride_m=float(cfg["candidate_stride_m"]),
        scales_m=tuple(float(v) for v in cfg["candidate_scales_m"]),
        rebuild=args.rebuild_candidates,
    )
    cloud_store = PointCloudStore(
        data_config,
        max_points=int(cfg.get("max_points", MAX_POINTS_DEFAULT)),
        rebuild=args.rebuild_clouds,
    )
    observation_config = ObservationConfig(
        image_size=CURRENT_SIZE,
        goal_crop_size=CANDIDATE_SIZE,
        fpv_width=CURRENT_SIZE,
        fpv_height=CURRENT_SIZE,
        # E2/E4 set this > 0 to widen real points; baseline stays 0.
        fpv_splat_radius=int(cfg.get("splat_radius", 0)),
    )
    train_dataset = PretrainDataset(
        train_manifest,
        data_config,
        observation_config,
        candidate_store,
        cloud_store,
        seed=seed,
    )
    val_seen_dataset = PretrainDataset(
        val_seen_manifest,
        data_config,
        observation_config,
        candidate_store,
        cloud_store,
        seed=seed + 100,
    )
    val_unseen_dataset = PretrainDataset(
        val_unseen_manifest,
        data_config,
        observation_config,
        candidate_store,
        cloud_store,
        seed=seed + 200,
    )
    batch_size = int(cfg["batch_size"])
    if args.smoke:
        batch_size = int(cfg["smoke"].get("batch_size", batch_size))
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        persistent_workers=bool(int(cfg.get("num_workers", 0)) > 0),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    steps_per_epoch = int(math.ceil(len(train_dataset) / batch_size))
    if args.smoke:
        total_steps = int(cfg["smoke"]["steps"])
        epochs = int(math.ceil(total_steps / max(steps_per_epoch, 1)))
    else:
        epochs = int(cfg.get("epochs", 1))
        total_steps = epochs * steps_per_epoch
        # E3/E4 add samples per trajectory; a fixed update budget keeps the
        # four ablations comparable (default 2751 for 22,002 / batch 8).
        if cfg.get("total_steps") is not None:
            total_steps = int(cfg["total_steps"])
    checkpoint_fractions = (
        cfg.get("smoke", {}).get("checkpoint_fractions", [1.0])
        if args.smoke
        else cfg.get("checkpoint_fractions", [0.5, 1.0])
    )
    checkpoint_steps = _checkpoint_steps(
        total_steps, checkpoint_fractions
    )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    encoder_checkpoint = Path(cfg["encoder_checkpoint"])
    if not encoder_checkpoint.is_absolute():
        encoder_checkpoint = PROJECT_ROOT / encoder_checkpoint
    encoder, init_report = load_pretrained_encoder(encoder_checkpoint, device)
    model = PretrainModel().to(device)
    model.encoder.load_state_dict(encoder.state_dict(), strict=True)
    model.train()
    class_weights_np, action_counts = _class_weights(train_manifest["action"])
    class_weights = torch.from_numpy(class_weights_np).to(device)
    encoder_params = list(model.encoder.parameters())
    head_params = [
        parameter for parameter in model.parameters()
        if not any(parameter is item for item in encoder_params)
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(cfg["lr_encoder"])},
            {"params": head_params, "lr": float(cfg["lr_heads"])},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps, 1),
        eta_min=float(cfg["lr_encoder"]) * 0.1,
    )
    initial_encoder = torch.nn.utils.parameters_to_vector(
        [p.detach().float().cpu() for p in encoder_params]
    )
    metrics_path = run_dir / "train_metrics.csv"
    fieldnames = [
        "step", "epoch", "total_loss", "action_loss", "xview_loss",
        "retrieval_loss", "action_accuracy", "xview_top1", "xview_top5",
        "retrieval_top1", "retrieval_top5", "encoder_grad_norm",
        "head_grad_norm", "lr_encoder", "lr_heads", "encoder_param_delta",
        "render_seconds_mean", "step_seconds", "data_seconds",
        "samples_per_second", "gpu_allocated_mb", "valid_pixel_ratio_mean",
    ]
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir / "tensorboard")
    except Exception:
        writer = None
    gradient_checks = None
    best_score = -1.0
    best_step = None
    best_encoder_path = None
    best_metrics = None
    val_history = []
    step = 0
    epoch = 0
    start_time = time.perf_counter()
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        csv_writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        while step < total_steps:
            epoch += 1
            for batch in loader:
                if step >= total_steps:
                    break
                step_start = time.perf_counter()
                data_start = time.perf_counter()
                batch_device = _to_device_batch(batch, device)
                data_seconds = time.perf_counter() - data_start
                optimizer.zero_grad(set_to_none=True)
                total_loss, loss_parts = _compute_losses(
                    model, batch_device, class_weights, cfg
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"non-finite total loss at step {step}"
                    )
                if step == 0:
                    per_loss = {}
                    for name in ("action_loss", "xview_loss", "retrieval_loss"):
                        value = loss_parts[name]
                        grads = torch.autograd.grad(
                            value, encoder_params,
                            retain_graph=True, allow_unused=True,
                        )
                        norm = math.sqrt(
                            sum(
                                float(g.detach().double().pow(2).sum().item())
                                for g in grads if g is not None
                            )
                        )
                        per_loss[name] = norm
                    gradient_checks = {
                        "step": 0,
                        "encoder_grad_norm_by_loss": per_loss,
                        "loss_values": {
                            key: float(loss_parts[key].detach().item())
                            for key in ("action_loss", "xview_loss", "retrieval_loss")
                        },
                    }
                total_loss.backward()
                encoder_grad_norm = math.sqrt(
                    sum(
                        float(p.grad.detach().double().pow(2).sum().item())
                        for p in encoder_params if p.grad is not None
                    )
                )
                head_grad_norm = math.sqrt(
                    sum(
                        float(p.grad.detach().double().pow(2).sum().item())
                        for p in head_params if p.grad is not None
                    )
                )
                if not math.isfinite(encoder_grad_norm) or not math.isfinite(head_grad_norm):
                    raise FloatingPointError(f"non-finite gradient at step {step}")
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()), float(cfg["grad_clip"])
                )
                optimizer.step()
                scheduler.step()
                step += 1
                if step % max(1, int(cfg.get("log_every", 10))) == 0 or step == 1:
                    current_encoder = torch.nn.utils.parameters_to_vector(
                        [p.detach().float().cpu() for p in encoder_params]
                    )
                    param_delta = float((current_encoder - initial_encoder).norm())
                    prediction = loss_parts["action_logits"].argmax(dim=-1)
                    action_accuracy = float(
                        (prediction == batch_device["actions"]).float().mean().item()
                    )
                    step_seconds = time.perf_counter() - step_start
                    gpu_allocated = (
                        torch.cuda.memory_allocated(device) / 1024 ** 2
                        if device.type == "cuda" else 0.0
                    )
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "total_loss": float(total_loss.detach().item()),
                        "action_loss": float(loss_parts["action_loss"].detach().item()),
                        "xview_loss": float(loss_parts["xview_loss"].detach().item()),
                        "retrieval_loss": float(loss_parts["retrieval_loss"].detach().item()),
                        "action_accuracy": action_accuracy,
                        "xview_top1": loss_parts["xview_metrics"]["top1"],
                        "xview_top5": loss_parts["xview_metrics"]["top5"],
                        "retrieval_top1": loss_parts["retrieval_metrics"]["top1"],
                        "retrieval_top5": loss_parts["retrieval_metrics"]["top5"],
                        "encoder_grad_norm": encoder_grad_norm,
                        "head_grad_norm": head_grad_norm,
                        "lr_encoder": optimizer.param_groups[0]["lr"],
                        "lr_heads": optimizer.param_groups[1]["lr"],
                        "encoder_param_delta": param_delta,
                        "render_seconds_mean": float(
                            batch_device["render_seconds"].mean().item()
                        ),
                        "step_seconds": step_seconds,
                        "data_seconds": data_seconds,
                        "samples_per_second": batch_size / max(step_seconds, 1e-9),
                        "gpu_allocated_mb": gpu_allocated,
                        "valid_pixel_ratio_mean": float(
                            batch_device["valid_pixel_ratio"].mean().item()
                        ),
                    }
                    csv_writer.writerow(row)
                    metrics_file.flush()
                    if writer is not None:
                        for key in (
                            "total_loss", "action_loss", "xview_loss",
                            "retrieval_loss", "action_accuracy", "xview_top1",
                            "xview_top5", "retrieval_top1", "retrieval_top5",
                            "encoder_grad_norm", "head_grad_norm",
                            "encoder_param_delta", "lr_encoder", "lr_heads",
                            "render_seconds_mean", "samples_per_second",
                            "gpu_allocated_mb",
                        ):
                            writer.add_scalar(f"train/{key}", row[key], step)
                if step in checkpoint_steps:
                    val_seen = _evaluate(
                        model, val_seen_dataset, val_seen_manifest,
                        candidate_store, device, cfg,
                    )
                    val_unseen = _evaluate(
                        model, val_unseen_dataset, val_unseen_manifest,
                        candidate_store, device, cfg,
                    )
                    record = {
                        "step": step,
                        "val_seen": val_seen,
                        "val_unseen": val_unseen,
                    }
                    val_history.append(record)
                    _write_json(run_dir / "val_history.json", val_history)
                    checkpoint_path = (
                        run_dir / "checkpoints" / f"encoder_step_{step:06d}.pt"
                    )
                    save_encoder_only(model.encoder, checkpoint_path)
                    score = float(val_seen.get("selection_score", -1.0))
                    if score > best_score:
                        best_score = score
                        best_step = step
                        best_metrics = record
                        best_encoder_path = save_encoder_only(
                            model.encoder, run_dir / "best_encoder.pt"
                        )
                    if writer is not None:
                        writer.add_scalar("val_seen/selection_score", score, step)
                        writer.add_scalar(
                            "val_seen/action_macro_f1",
                            val_seen["action"]["macro_f1"], step,
                        )
                        writer.add_scalar(
                            "val_seen/xview_top1",
                            val_seen["xview"]["top1"] or 0.0, step,
                        )
                        writer.add_scalar(
                            "val_seen/retrieval_recall@5",
                            val_seen["retrieval"]["recall@5"] or 0.0, step,
                        )
                    print(
                        f"checkpoint step={step} val_seen score={score:.4f} "
                        f"action_f1={val_seen['action']['macro_f1']:.4f} "
                        f"xview_top1={val_seen['xview']['top1']} "
                        f"retrieval_r5={val_seen['retrieval']['recall@5']}",
                        flush=True,
                    )
                    model.train()
                if step >= total_steps:
                    break
            if step >= total_steps:
                break
    if writer is not None:
        writer.close()
    if step not in checkpoint_steps:
        val_seen = _evaluate(
            model, val_seen_dataset, val_seen_manifest,
            candidate_store, device, cfg,
        )
        val_unseen = _evaluate(
            model, val_unseen_dataset, val_unseen_manifest,
            candidate_store, device, cfg,
        )
        record = {"step": step, "val_seen": val_seen, "val_unseen": val_unseen}
        val_history.append(record)
        _write_json(run_dir / "val_history.json", val_history)
        score = float(val_seen.get("selection_score", -1.0))
        if score > best_score:
            best_score = score
            best_step = step
            best_metrics = record
            best_encoder_path = save_encoder_only(
                model.encoder, run_dir / "best_encoder.pt"
            )
    final_encoder_path = save_encoder_only(
        model.encoder, run_dir / "encoder_final.pt"
    )
    if best_encoder_path is None:
        best_encoder_path = final_encoder_path
    elapsed = time.perf_counter() - start_time
    after = snapshot_raw_dirs(data_config)
    assert_snapshot_unchanged(before, after)
    results_dir = EXPERIMENT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_path, results_dir / f"{run_name}_train_metrics.csv")
    _write_json(results_dir / f"{run_name}_val_history.json", val_history)
    if gradient_checks is not None:
        _write_json(results_dir / f"{run_name}_gradient_checks.json", gradient_checks)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        "smoke": bool(args.smoke),
        "device": str(device),
        "seed": seed,
        "config": cfg,
        "train_manifest": {
            key: train_manifest[key]
            for key in (
                "split", "records", "maps", "action_counts",
                "desired_action_counts", "manifest_file",
            )
        },
        "val_seen_manifest": {
            key: val_seen_manifest[key]
            for key in ("split", "records", "maps", "action_counts", "manifest_file")
        },
        "val_unseen_manifest": {
            key: val_unseen_manifest[key]
            for key in ("split", "records", "maps", "action_counts", "manifest_file")
        },
        "class_weights": {
            ACTION_NAMES[index]: float(class_weights_np[index])
            for index in ACTION_ORDER
        },
        "action_counts": action_counts,
        "initialization": init_report,
        "gradient_checks": gradient_checks,
        "steps": step,
        "total_steps_planned": total_steps,
        "best_step": best_step,
        "best_score_val_seen": best_score,
        "best_metrics": best_metrics,
        "best_encoder": best_encoder_path,
        "final_encoder": final_encoder_path,
        "encoder_final_sha256": final_encoder_path["sha256"],
        "best_encoder_sha256": best_encoder_path["sha256"],
        "elapsed_seconds": round(float(elapsed), 3),
        "gpu_max_allocated_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2)
            if device.type == "cuda" else 0.0
        ),
        "raw_data_unchanged": True,
        "raw_snapshot": {
            "file_count": int(after["file_count"]),
            "total_bytes": int(after["total_bytes"]),
            "digest": str(after["digest"]),
        },
        "test_unseen_read": False,
    }
    _write_json(run_dir / "run_summary.json", summary)
    _write_json(
        results_dir / f"{run_name}_summary.json",
        _relative_payload(summary),
    )
    if best_metrics is not None:
        _write_json(results_dir / f"{run_name}_best_metrics.json", best_metrics)
    print("visual encoder pretraining finished")
    print(
        f"  run={run_name} steps={step} best_step={best_step} "
        f"best_score={best_score:.4f}"
    )
    print(f"  best_encoder={best_encoder_path}")
    print(f"  final_encoder_sha256={final_encoder_path['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
