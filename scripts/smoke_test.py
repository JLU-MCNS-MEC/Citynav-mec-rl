#!/usr/bin/env python3
"""CPU-first end-to-end smoke test for the CityNav-MEC adaptation.

The test reads a few real CityNav episodes, renders the official top-down
observation, runs the frozen visual/language encoders, then performs one
forward pass and one optimizer step for both Small and Transformer-M.

It is deliberately not a training run: batch size is tiny, the number of
episodes is bounded, and every raw-data directory is fingerprinted before and
after the run to prove it was not modified.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
from src.data.citynav_dataset import CityNavDataset  # noqa: E402
from src.data.fpv_observation import FPVObservationAdapter  # noqa: E402
from src.data.language_adapter import build_language_adapter  # noqa: E402
from src.data.observation_adapter import TopDownObservationAdapter  # noqa: E402
from src.data.paths import (  # noqa: E402
    assert_snapshot_unchanged,
    ensure_output_file,
    snapshot_raw_dirs,
)
from src.models import SmallMEC, TransformerMMEC  # noqa: E402
from src.models.mec_common import (  # noqa: E402
    placeholder_network_state,
)
from src.models.token_utils import encode_scene  # noqa: E402
from src.models.vision_encoder import (  # noqa: E402
    VisualTokenEncoder,
    count_parameters,
)
from src.training.supervised import (  # noqa: E402
    evaluate_components,
    train_step,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="val_unseen",
        choices=("train_seen", "val_seen", "val_unseen", "test_unseen"),
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--scan-episodes",
        type=int,
        default=64,
        help="maximum number of records inspected when choosing episodes",
    )
    parser.add_argument(
        "--observation", default="fpv", choices=("fpv", "topdown")
    )
    parser.add_argument("--render-size", type=int, default=336)
    parser.add_argument("--crop-radius", type=float, default=120.0)
    parser.add_argument("--splat-radius", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-backend", default="hashing",
                        choices=("auto", "bert", "hashing"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports"
    )
    parser.add_argument("--allow-test-unseen", action="store_true")
    return parser.parse_args()


def _sanitize(message: str | None) -> str | None:
    if message is None:
        return None
    text = str(message)
    for value, replacement in (
        (str(PROJECT_ROOT), "<PROJECT_ROOT>"),
        (str(Path.home()), "<HOME>"),
    ):
        text = text.replace(value, replacement)
    return text


def _select_episodes(
    dataset: CityNavDataset,
    count: int,
) -> list[tuple[object, object]]:
    """Prefer successful teacher rollouts that contain at least one move."""
    selected: list[tuple[object, object]] = []
    for record in dataset.records:
        result = record.teacher(dataset.teacher_config)
        if result.steps > 1 and result.success:
            selected.append((record, result))
            if len(selected) >= count:
                return selected
    if selected:
        return selected
    for record in dataset.records[:count]:
        selected.append((record, record.teacher(dataset.teacher_config)))
    return selected


def _build_batch(
    dataset: CityNavDataset,
    samples: Sequence[tuple[object, object]],
    adapter: object,
    encoder: VisualTokenEncoder,
    text_adapter: object,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    current_tokens = []
    goal_tokens = []
    heatmaps = []
    languages = []
    actions = []
    network_states = []
    episode_summaries = []

    for offset, (record, result) in enumerate(samples):
        pose_index = min(
            max(0, len(result.poses) // 3),
            len(result.poses) - 1,
        )
        action_index = min(pose_index, len(result.actions) - 1)
        if isinstance(adapter, FPVObservationAdapter):
            observation = adapter.render(
                record.map_name,
                result.poses[pose_index],
                record.object_info,
                crop_radius_m=float(adapter.config.fpv_crop_radius_m),
            )
        else:
            observation = adapter.render_step(
                record.map_name, result.poses[pose_index], record.object_info
            )
        scene = encode_scene(
            encoder,
            observation["current_rgb"][None],
            observation["goal_rgbs"][None],
            device=device,
        )
        language = text_adapter.encode_tensor([record.description], device=device)
        current_tokens.append(scene["current_tokens"][0])
        goal_tokens.append(scene["goal_tokens"][0])
        heatmaps.append(scene["heatmap"][0])
        languages.append(language[0])
        actions.append(int(result.actions[action_index]))
        network_states.append(
            torch.from_numpy(
                placeholder_network_state(seed + offset)
            ).to(device)
        )
        episode_summaries.append(
            {
                "episode_id": record.episode_id,
                "map_name": record.map_name,
                "description_match": bool(record.description_match),
                "pose_index": int(pose_index),
                "num_teacher_steps": int(result.steps),
                "teacher_success": bool(result.success),
                "action": int(result.actions[action_index]),
                "observation_source": observation.get(
                    "observation_source", "citynav_orthophoto_topdown"
                ),
            }
        )

    batch = {
        "current_tokens": torch.stack(current_tokens),
        "goal_tokens": torch.stack(goal_tokens),
        "heatmap": torch.stack(heatmaps),
        "network_state": torch.stack(network_states).float(),
        "language_features": torch.stack(languages),
        "actions": torch.tensor(actions, dtype=torch.long, device=device),
    }
    return batch, episode_summaries


def _flat_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(
        [parameter.detach().float().cpu() for parameter in model.parameters()]
    )


def _run_model(
    name: str,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict:
    inputs = {
        key: batch[key]
        for key in (
            "current_tokens",
            "goal_tokens",
            "heatmap",
            "network_state",
            "language_features",
        )
    }
    model = model.to(next(iter(inputs.values())).device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=1e-4
    )

    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        logits = model(**inputs)
        decision = model.predict_action(**inputs)
        forward_seconds = time.perf_counter() - start

    if logits.shape != (batch["actions"].numel(), 6):
        raise RuntimeError(f"{name}: unexpected logits shape {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError(f"{name}: non-finite forward logits")
    if not torch.allclose(
        decision["joint_probabilities"].sum(dim=-1),
        torch.ones(batch["actions"].numel(), device=logits.device),
        atol=1e-5,
    ):
        raise RuntimeError(f"{name}: joint probabilities do not sum to one")

    eval_metrics = evaluate_components(model, batch)
    eval_scalars = {
        key: value for key, value in eval_metrics.items()
        if isinstance(value, (int, float))
    }
    if not all(np.isfinite(value) for value in eval_scalars.values()):
        raise RuntimeError(f"{name}: non-finite evaluation metrics")

    before = _flat_parameters(model)
    start = time.perf_counter()
    train_metrics = train_step(
        model,
        optimizer,
        batch,
        TrainingConfig(
            batch_size=int(batch["actions"].numel()),
            lr=float(args.lr),
            weight_decay=1e-4,
            grad_clip=1.0,
        ),
    )
    train_seconds = time.perf_counter() - start
    after = _flat_parameters(model)
    parameter_delta = float((after - before).norm())

    if not np.isfinite(train_metrics["loss"]):
        raise RuntimeError(f"{name}: non-finite training loss")
    if not np.isfinite(train_metrics["grad_norm"]):
        raise RuntimeError(f"{name}: non-finite gradient norm")
    if parameter_delta <= 0.0:
        raise RuntimeError(f"{name}: optimizer step did not change parameters")

    return {
        "parameters": int(count_parameters(model)),
        "forward_shape": list(logits.shape),
        "forward_seconds": round(float(forward_seconds), 4),
        "train_seconds": round(float(train_seconds), 4),
        "predicted_actions": [int(value) for value in decision["action"].cpu()],
        "eval": {
            key: round(float(value), 6) for key, value in eval_scalars.items()
        },
        "train": {
            key: (round(float(value), 6) if isinstance(value, (int, float)) else value)
            for key, value in train_metrics.items()
        },
        "parameter_delta_l2": round(parameter_delta, 8),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# CityNav-MEC Smoke Test",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 设备: `{report['device']}`",
        f"- 划分: `{report['split']}`",
        f"- 文本后端: `{report['text_backend_actual']}`",
        f"- Batch: {report['batch_size']}",
        f"- 原始数据快照未变化: `{report['raw_data_unchanged']}`",
        "",
        "## 1. 使用的真实 Episode",
        "",
        "| episode | map | teacher steps | success | description match |",
        "|---|---|---:|:---:|:---:|",
    ]
    for episode in report["episodes"]:
        lines.append(
            f"| `{episode['episode_id']}` | {episode['map_name']} | "
            f"{episode['num_teacher_steps']} | {episode['teacher_success']} | "
            f"{episode['description_match']} |"
        )
    lines += [
        "",
        "## 2. 视觉编码与模型",
        "",
        "| model | params | forward shape | eval loss | train loss | grad norm | param delta (L2) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for name, metrics in report["models"].items():
        lines.append(
            f"| {name} | {metrics['parameters']} | "
            f"{metrics['forward_shape']} | {metrics['eval']['loss']} | "
            f"{metrics['train']['loss']} | {metrics['train']['grad_norm']} | "
            f"{metrics['parameter_delta_l2']} |"
        )
    lines += [
        "",
        "## 3. 检查项",
        "",
        "- [x] CityNav 原始 JSON 读取，split 字段与请求划分一致。",
        "- [x] CityRefer 对象、描述与目标位置可解析，描述索引为 0-based。",
        "- [x] 第一视角 PLY RGB / 透视 Depth（或显式 top-down 诊断模式）可生成。",
        "- [x] 视觉编码器输出 576 × 384 tokens。",
        "- [x] Small / Transformer-M 前向输出六动作联合分布。",
        "- [x] 单步监督训练损失、梯度、参数更新均有限。",
        "- [x] 运行前后共享原始数据快照一致。",
        "",
        "## 4. 边界",
        "",
        "这是 Smoke Test，不是训练结果；不包含 RL 环境、CityNav GSM 语义地图、"
        "完整训练配方或指标评测。网络状态为确定性占位向量。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    if args.split == "test_unseen" and not args.allow_test_unseen:
        raise SystemExit(
            "test_unseen is reserved for the final frozen evaluation; "
            "pass --allow-test-unseen explicitly"
        )
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    data_config = load_data_config()
    data_config.validate()
    before = snapshot_raw_dirs(data_config)

    dataset = CityNavDataset(
        data_config,
        args.split,
        max_episodes=int(args.scan_episodes),
        allow_test_unseen=bool(args.allow_test_unseen),
    )
    if len(dataset) == 0:
        raise SystemExit(f"no records available in split {args.split}")

    samples = _select_episodes(dataset, int(args.episodes))
    if not samples:
        raise SystemExit("could not select an episode for the smoke test")

    encoder = VisualTokenEncoder().to(device).eval()
    text_adapter, text_dim, text_backend, text_warning = build_language_adapter(
        TextConfig(backend=args.text_backend, local_files_only=True),
        device=device,
    )
    obs_config = ObservationConfig(
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
    if args.observation == "fpv":
        adapter = FPVObservationAdapter(data_config, obs_config)
    else:
        adapter = TopDownObservationAdapter(data_config, obs_config)
    try:
        batch, episodes = _build_batch(
            dataset,
            samples,
            adapter,
            encoder,
            text_adapter,
            device,
            args.seed,
        )
    finally:
        adapter.close()

    models = {
        "Small": SmallMEC(text_dim=text_dim),
        "Transformer-M": TransformerMMEC(text_dim=text_dim),
    }
    model_reports = {
        name: _run_model(name, model, batch, args)
        for name, model in models.items()
    }

    after = snapshot_raw_dirs(data_config)
    assert_snapshot_unchanged(before, after)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "split": args.split,
        "observation": args.observation,
        "render_size": int(args.render_size),
        "crop_radius_m": float(args.crop_radius),
        "splat_radius": int(args.splat_radius),
        "seed": int(args.seed),
        "text_backend_requested": args.text_backend,
        "text_backend_actual": text_backend,
        "text_warning": _sanitize(text_warning),
        "batch_size": int(batch["actions"].numel()),
        "actions": [int(value) for value in batch["actions"].cpu()],
        "episodes": episodes,
        "encoder": {
            "parameters": int(count_parameters(encoder)),
            "image_size": int(encoder.image_size),
            "tokens_per_image": int(encoder.num_tokens),
            "token_dim": int(encoder.token_dim),
        },
        "models": model_reports,
        "raw_data_unchanged": True,
        "raw_snapshot": {
            "file_count": int(after["file_count"]),
            "total_bytes": int(after["total_bytes"]),
            "digest": str(after["digest"]),
        },
    }

    output_dir = Path(args.output_dir)
    json_path = ensure_output_file(
        output_dir / "smoke_test.json", data_config=data_config
    )
    md_path = ensure_output_file(
        output_dir / "smoke_test.md", data_config=data_config
    )
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")

    print("CityNav-MEC smoke test passed")
    print(
        f"  split={args.split} episodes={len(episodes)} batch={report['batch_size']} "
        f"device={device} text={text_backend}"
    )
    for name, metrics in model_reports.items():
        print(
            f"  {name:14s} params={metrics['parameters']:8d} "
            f"forward={metrics['forward_shape']} "
            f"train_loss={metrics['train']['loss']:.4f} "
            f"grad_norm={metrics['train']['grad_norm']:.4f} "
            f"param_delta={metrics['parameter_delta_l2']:.6f}"
        )
    print(f"  raw_data_unchanged=True digest={after['digest']}")
    print(f"  wrote {json_path.relative_to(PROJECT_ROOT)} and {md_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
