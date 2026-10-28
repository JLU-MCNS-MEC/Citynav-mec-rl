#!/usr/bin/env python3
"""Small T5 + language retrieval / token / MEC smoke pipeline.

Stages:
  1. fixed-grid, multi-scale orthophoto candidate libraries (no target labels);
  2. language-to-map retriever training (target positions only as training
     labels) and Recall@1/5/10 on val_seen / val_unseen;
  3. freeze the best encoder/retriever and build a small T5 token cache;
  4. train Small / Transformer-M full T5+Language and a map-free baseline.

This is a bounded Smoke Test, not a full training run.  test_unseen is never
opened.
"""

from __future__ import annotations

import argparse
import json
import os
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
from src.data.actions import ACTION_NAMES, STOP  # noqa: E402
from src.data.candidates import (  # noqa: E402
    CANDIDATE_SCALES_M,
    CANDIDATE_STRIDE_M,
    CandidateLibrary,
    build_candidate_library,
    build_goal_tokens,
    encode_candidate_images,
    map_features_from_scores,
    topk_indices,
)
from src.data.citynav_dataset import CityNavDataset  # noqa: E402
from src.data.fpv_observation import FPVObservationAdapter  # noqa: E402
from src.data.gab import (  # noqa: E402
    GABConfig,
    build_gab_sequence,
    fit_reliability_threshold,
)
from src.data.language_adapter import build_language_adapter  # noqa: E402
from src.data.paths import (  # noqa: E402
    assert_snapshot_unchanged,
    ensure_output_dir,
    ensure_output_file,
    snapshot_raw_dirs,
)
from src.data.t5_features import (  # noqa: E402
    LANGUAGE_DIM,
    T5_FEATURE_DIMS,
    history_arrays,
    visual_descriptor,
)
from src.data.trajectory_adapter import Pose5D  # noqa: E402
from src.models import VisualTokenEncoder, load_encoder_checkpoint  # noqa: E402
from src.models.mec_common import build_spatial_heatmap  # noqa: E402
from src.models.t5_mec import build_t5_model  # noqa: E402
from src.models.token_utils import encode_current_tokens  # noqa: E402
from src.training.retrieval import (  # noqa: E402
    LanguageMapRetriever,
    RetrievalSample,
    train_retriever,
)
from src.training.supervised import factorized_loss, train_step  # noqa: E402


DEFAULT_TRAIN_MAPS = (
    "birmingham_block_13",
    "birmingham_block_11",
    "birmingham_block_12",
    "birmingham_block_10",
    "cambridge_block_25",
    "cambridge_block_17",
    "cambridge_block_23",
    "cambridge_block_4",
)
DEFAULT_VAL_SEEN_MAPS = (
    "birmingham_block_13",
    "birmingham_block_11",
    "cambridge_block_25",
)
DEFAULT_VAL_UNSEEN_MAPS = (
    "birmingham_block_5",
    "cambridge_block_10",
    "cambridge_block_2",
    "cambridge_block_3",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrieval-steps", type=int, default=80)
    parser.add_argument("--retrieval-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--freeze-visual-encoder", action="store_true")
    parser.add_argument("--retrieval-train-per-map", type=int, default=16)
    parser.add_argument("--retrieval-val-per-map", type=int, default=12)
    parser.add_argument("--train-trajectories", type=int, default=24)
    parser.add_argument("--val-seen-trajectories", type=int, default=6)
    parser.add_argument("--val-unseen-trajectories", type=int, default=8)
    parser.add_argument("--steps-per-trajectory", type=int, default=6)
    parser.add_argument("--render-size", type=int, default=336)
    parser.add_argument("--crop-radius", type=float, default=120.0)
    parser.add_argument("--candidate-stride", type=float, default=CANDIDATE_STRIDE_M)
    parser.add_argument(
        "--candidate-scales",
        nargs="+",
        type=float,
        default=list(CANDIDATE_SCALES_M),
    )
    parser.add_argument("--mec-steps", type=int, default=150)
    parser.add_argument("--mec-batch-size", type=int, default=8)
    parser.add_argument("--mec-lr", type=float, default=1e-4)
    parser.add_argument("--mec-weight-decay", type=float, default=1e-4)
    parser.add_argument("--rebuild-candidates", action="store_true")
    parser.add_argument("--rebuild-tokens", action="store_true")
    parser.add_argument(
        "--token-shards",
        action="store_true",
        help=(
            "persist each collected token episode to cache_dir/tokens/shards so an "
            "interrupted token build resumes instead of restarting. Off by default: "
            "the resulting token arrays are identical either way."
        ),
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "only collect episode shards and exit before GAB threshold fitting, "
            "finalization and MEC training. Requires --token-shards."
        ),
    )
    parser.add_argument(
        "--prep-only",
        action="store_true",
        help=(
            "build the disk language-feature cache and exit, so parallel "
            "collection processes share one BERT pass."
        ),
    )
    parser.add_argument(
        "--train-range",
        default=None,
        help="START:END slice of selected train records for parallel collection",
    )
    parser.add_argument(
        "--val-seen-range",
        default=None,
        help="START:END slice of selected val_seen records",
    )
    parser.add_argument(
        "--val-unseen-range",
        default=None,
        help="START:END slice of selected val_unseen records",
    )
    parser.add_argument(
        "--group-by-map",
        action="store_true",
        help=(
            "collect the selected records grouped by map to avoid PLY cache "
            "thrashing. Episodes are independent and keyed by identity, so the "
            "merged token order is unchanged."
        ),
    )
    parser.add_argument(
        "--reuse-retrieval",
        action="store_true",
        help=(
            "load existing cache_dir/retrieval/{encoder,retriever}.pt instead of "
            "retraining the retriever (the encoder stays frozen)."
        ),
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "data_cache" / "weights" / "shared_encoder.pt",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "data_cache" / "t5_smoke"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "t5"
    )
    parser.add_argument("--train-maps", nargs="+", default=list(DEFAULT_TRAIN_MAPS))
    parser.add_argument("--val-seen-maps", nargs="+", default=list(DEFAULT_VAL_SEEN_MAPS))
    parser.add_argument("--val-unseen-maps", nargs="+", default=list(DEFAULT_VAL_UNSEEN_MAPS))
    return parser.parse_args()


def _load_encoder(args, device: torch.device) -> VisualTokenEncoder:
    encoder = VisualTokenEncoder()
    if args.encoder_checkpoint and Path(args.encoder_checkpoint).exists():
        load_encoder_checkpoint(
            encoder, str(args.encoder_checkpoint), map_location="cpu"
        )
    encoder.to(device)
    encoder.eval()
    return encoder


def _load_bert() -> tuple[object, dict]:
    adapter, dim, backend, warning = build_language_adapter(
        TextConfig(backend="bert", local_files_only=True),
        device="cpu",
    )
    if backend != "bert":
        raise SystemExit(f"formal pipeline requires BERT, got {backend}")
    params = list(adapter.model.parameters())
    if any(parameter.requires_grad for parameter in params):
        raise RuntimeError("BERT must be frozen")
    return adapter, {
        "backend": backend,
        "feature_dim": int(dim),
        "parameters": int(sum(p.numel() for p in params)),
        "requires_grad_any": False,
        "warning": warning,
    }


def _language_map(
    descriptions: Sequence[str],
    adapter,
    cache_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    unique = sorted(set(str(text) for text in descriptions))
    cached: dict[str, np.ndarray] = {}
    if cache_path is not None and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                texts = [str(value) for value in payload["texts"]]
                features = payload["features"]
            cached = {
                text: features[index].astype(np.float32)
                for index, text in enumerate(texts)
            }
        except Exception as exc:  # corrupted cache: recompute from scratch
            print(f"  language cache unreadable, rebuilding ({exc})", flush=True)
            cached = {}
    missing = [text for text in unique if text not in cached]
    if missing:
        with torch.no_grad():
            encoded = adapter.encode_tensor(missing, device="cpu").numpy()
        for text, vector in zip(missing, encoded):
            cached[text] = vector.astype(np.float32)
    if cache_path is not None and (missing or not cache_path.exists()):
        merged = sorted(cached)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name: parallel collectors must not race on one temp file.
        tmp_path = cache_path.with_name(
            f"{cache_path.name}.{os.getpid()}.tmp.npz"
        )
        np.savez_compressed(
            tmp_path,
            texts=np.asarray(merged),
            features=np.stack([cached[text] for text in merged]),
        )
        tmp_path.replace(cache_path)
    mapping = {text: cached[text] for text in unique}
    encoded = np.stack([mapping[text] for text in unique])
    return mapping, {
        "unique_descriptions": len(unique),
        "feature_dim": int(encoded.shape[1]),
    }


def _records_by_map(dataset: CityNavDataset, maps: Sequence[str]) -> dict[str, list]:
    allowed = set(maps)
    groups: dict[str, list] = defaultdict(list)
    for record in dataset.records:
        if record.map_name in allowed:
            groups[record.map_name].append(record)
    return groups


def _select_records(
    dataset: CityNavDataset,
    maps: Sequence[str],
    per_map: int,
) -> list:
    groups = _records_by_map(dataset, maps)
    selected = []
    for map_name in maps:
        records = groups.get(map_name, [])
        if not records:
            continue
        count = min(int(per_map), len(records))
        indices = np.linspace(0, len(records) - 1, count, dtype=int)
        selected.extend(records[int(index)] for index in dict.fromkeys(indices.tolist()))
    return selected


def _select_teacher_records(
    dataset: CityNavDataset,
    maps: Sequence[str],
    count: int,
    steps: int,
) -> list:
    groups = _records_by_map(dataset, maps)
    selected = []
    cursors = {map_name: 0 for map_name in maps}
    while len(selected) < int(count):
        progressed = False
        for map_name in maps:
            records = groups.get(map_name, [])
            while cursors.get(map_name, 0) < len(records):
                index = cursors[map_name]
                cursors[map_name] = index + 1
                record = records[index]
                result = record.teacher(dataset.teacher_config)
                if len(result.actions) >= int(steps):
                    selected.append(record)
                    progressed = True
                    break
            if len(selected) >= int(count):
                break
        if not progressed:
            break
    return selected


def _build_libraries(
    data_config,
    observation_config: ObservationConfig,
    maps: Sequence[str],
    encoder,
    device: torch.device,
    args,
) -> dict[str, CandidateLibrary]:
    libraries = {}
    for map_name in sorted(set(maps)):
        libraries[map_name] = build_candidate_library(
            data_config,
            map_name,
            observation_config,
            encoder=encoder,
            device=device,
            stride_m=float(args.candidate_stride),
            scales_m=tuple(float(v) for v in args.candidate_scales),
            rebuild=bool(args.rebuild_candidates),
        )
    return libraries


def _retrieval_samples(
    dataset,
    libraries: dict[str, CandidateLibrary],
    language_map: dict[str, np.ndarray],
    maps: Sequence[str],
    per_map: int,
    split: str,
) -> list[RetrievalSample]:
    records = _select_records(dataset, maps, per_map)
    samples = []
    for record in records:
        library = libraries.get(record.map_name)
        if library is None:
            continue
        samples.append(
            RetrievalSample(
                split=split,
                episode_id=record.episode_id,
                map_name=record.map_name,
                language_features=np.asarray(
                    language_map[record.description], dtype=np.float32
                ),
                library=library,
                target_xy=tuple(float(v) for v in record.target_xy),
            )
        )
    return samples


def _collect_episode_frames(
    record,
    result,
    *,
    observation_config: ObservationConfig,
    adapter: FPVObservationAdapter,
    encoder,
    retriever,
    library: CandidateLibrary,
    language_features: np.ndarray,
    start: int,
    steps: int,
    device: torch.device,
    crop_radius: float,
) -> dict | None:
    poses = list(result.poses[start:start + int(steps)])
    actions = [int(v) for v in result.actions[start:start + int(steps)]]
    if len(poses) != int(steps) or len(actions) != int(steps):
        return None
    with torch.no_grad():
        language = torch.from_numpy(language_features).float().to(device).reshape(1, -1)
        candidates = torch.from_numpy(
            np.asarray(library.embeddings, dtype=np.float32)
        ).to(device)
        scores = retriever(language, candidates).detach().cpu().numpy()
    top5 = topk_indices(scores, 5)
    top5_padded = np.full((5,), -1, dtype=np.int64)
    top5_padded[: len(top5)] = top5
    goal_tokens, goal_valid = build_goal_tokens(
        np.asarray(library.embeddings), top5[:3], k=3
    )
    current_tokens = []
    visuals = []
    heatmaps = []
    map_features = []
    render_seconds = 0.0
    quality = []
    for pose in poses:
        start_time = time.perf_counter()
        observation = adapter.render_current(
            record.map_name, pose, crop_radius_m=float(crop_radius)
        )
        render_seconds += time.perf_counter() - start_time
        tokens = encode_current_tokens(
            encoder,
            observation["current_rgb"][None],
            device,
            requires_grad=False,
        )[0].detach().cpu()
        current_tokens.append(tokens.to(torch.float16).numpy())
        descriptor = visual_descriptor(tokens[None].to(device))[0].detach().cpu().numpy()
        visuals.append(descriptor.astype(np.float32))
        map_feature, _ = map_features_from_scores(
            pose, library.centers, scores, top_k=5
        )
        map_features.append(map_feature)
        heatmap = build_spatial_heatmap(
            tokens[None].to(device),
            torch.from_numpy(goal_tokens).float().to(device)[None],
        )[0].detach().cpu().numpy()
        heatmaps.append(heatmap.astype(np.float32))
        quality.append(observation["depth_quality"]["valid_pixel_ratio"])
    return {
        "map_name": record.map_name,
        "episode_id": record.episode_id,
        "description": record.description,
        "split": record.split,
        "poses": poses,
        "actions": actions,
        "start_step": int(start),
        "current_tokens": np.stack(current_tokens),
        "visuals": np.stack(visuals),
        "heatmaps": np.stack(heatmaps),
        "goal_tokens": goal_tokens.astype(np.float32),
        "goal_valid": int(goal_valid),
        "top5_indices": [int(v) for v in top5_padded],
        "map_features": np.stack(map_features),
        "language_features": np.asarray(language_features, dtype=np.float32),
        "render_seconds": float(render_seconds),
        "valid_pixel_ratio_mean": float(np.mean(quality)),
    }


def _finalize_episode(
    episode: dict,
    gab_config: GABConfig,
    split_id: int,
) -> tuple[dict, list[dict]]:
    poses = episode["poses"]
    actions = episode["actions"]
    visuals = episode["visuals"]
    gab = build_gab_sequence(episode["heatmaps"], actions, gab_config)
    current = []
    goals = []
    history_vis = []
    history_pose = []
    history_action = []
    history_mask = []
    gab_rows = []
    map_rows = []
    language = []
    labels = []
    step_ids = []
    goal_valid = []
    top5_rows = []
    for index, action in enumerate(actions):
        h_vis, h_pose, h_action, h_mask = history_arrays(
            poses, actions, index, visuals
        )
        current.append(episode["current_tokens"][index])
        goals.append(episode["goal_tokens"])
        history_vis.append(h_vis.astype(np.float16))
        history_pose.append(h_pose)
        history_action.append(h_action)
        history_mask.append(h_mask)
        gab_rows.append(gab[index])
        map_rows.append(episode["map_features"][index])
        language.append(episode["language_features"])
        labels.append(int(action))
        step_ids.append(episode["start_step"] + index)
        goal_valid.append(episode["goal_valid"])
        top5_rows.append(episode["top5_indices"])
    arrays = {
        "current_tokens": np.stack(current).astype(np.float16),
        "goal_tokens": np.stack(goals).astype(np.float32),
        "history_vis": np.stack(history_vis).astype(np.float16),
        "history_pose": np.stack(history_pose).astype(np.float32),
        "history_action": np.stack(history_action).astype(np.float32),
        "history_mask": np.stack(history_mask).astype(np.float32),
        "gab": np.stack(gab_rows).astype(np.float32),
        "map_feat": np.stack(map_rows).astype(np.float32),
        "language_features": np.stack(language).astype(np.float32),
        "actions": np.asarray(labels, dtype=np.int64),
        "split_id": np.full(len(labels), int(split_id), dtype=np.int64),
        "step_id": np.asarray(step_ids, dtype=np.int64),
        "goal_valid": np.asarray(goal_valid, dtype=np.int64),
        "top5_indices": np.asarray(top5_rows, dtype=np.int64),
    }
    metadata = [
        {
            "split": episode["split"],
            "map_name": episode["map_name"],
            "episode_id": episode["episode_id"],
            "teacher_step": int(episode["start_step"] + index),
            "action": int(action),
            "goal_valid": int(episode["goal_valid"]),
            "top5_indices": [int(v) for v in episode["top5_indices"]],
            "render_seconds": episode["render_seconds"],
            "valid_pixel_ratio_mean": episode["valid_pixel_ratio_mean"],
        }
        for index, action in enumerate(actions)
    ]
    return arrays, metadata


def _concat_parts(parts: Sequence[dict]) -> dict:
    keys = parts[0].keys()
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in keys
    }


_EPISODE_ARRAY_KEYS = (
    "current_tokens",
    "visuals",
    "heatmaps",
    "goal_tokens",
    "map_features",
    "language_features",
)


def _parse_range(spec: str | None, total: int) -> tuple[int, int]:
    """Parse ``START:END`` into a clamped slice; ``None`` means the full list."""
    total = int(total)
    if spec is None or str(spec).strip() == "":
        return 0, total
    start, _, end = str(spec).partition(":")
    low = int(start) if start.strip() else 0
    high = int(end) if end.strip() else total
    low = max(0, min(low, total))
    high = max(low, min(high, total))
    return low, high


def _episode_key(split: str, map_name: str, episode_id, start: int) -> tuple:
    return (str(split), str(map_name), str(episode_id), int(start))


def _episode_shard_path(
    shard_dir: Path, split: str, map_name: str, episode_id, start: int
) -> Path:
    safe = "_".join(
        part.replace("/", "-").replace(" ", "_")
        for part in (str(split), str(map_name), str(episode_id))
    )
    return shard_dir / f"{safe}_{int(start):04d}.npz"


def _save_episode_shard(path: Path, episode: dict) -> None:
    payload: dict = {
        "map_name": np.asarray(str(episode["map_name"])),
        "episode_id": np.asarray(str(episode["episode_id"])),
        "split": np.asarray(str(episode["split"])),
        "poses": np.asarray(
            [pose.as_tuple() for pose in episode["poses"]], dtype=np.float64
        ),
        "actions": np.asarray(episode["actions"], dtype=np.int64),
        "start_step": np.asarray(int(episode["start_step"]), dtype=np.int64),
        "goal_valid": np.asarray(int(episode["goal_valid"]), dtype=np.int64),
        "top5_indices": np.asarray(episode["top5_indices"], dtype=np.int64),
        "render_seconds": np.asarray(float(episode["render_seconds"])),
        "valid_pixel_ratio_mean": np.asarray(
            float(episode["valid_pixel_ratio_mean"])
        ),
    }
    for key in _EPISODE_ARRAY_KEYS:
        payload[key] = np.asarray(episode[key])
    # PID-unique temp: two collectors may render the same episode when ranges
    # overlap, and a shared temp path produced a CRC-corrupted shard once.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(path)


def _load_episode_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        episode = {key: data[key] for key in _EPISODE_ARRAY_KEYS}
        episode.update(
            map_name=str(data["map_name"]),
            episode_id=str(data["episode_id"]),
            split=str(data["split"]),
            poses=[Pose5D(*(float(v) for v in row)) for row in data["poses"]],
            actions=[int(v) for v in data["actions"]],
            start_step=int(data["start_step"]),
            goal_valid=int(data["goal_valid"]),
            top5_indices=[int(v) for v in data["top5_indices"]],
            render_seconds=float(data["render_seconds"]),
            valid_pixel_ratio_mean=float(data["valid_pixel_ratio_mean"]),
        )
    return episode


def _load_episode_shards(shard_dir: Path) -> dict:
    episodes: dict = {}
    if not shard_dir.exists():
        return episodes
    for path in sorted(shard_dir.glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        try:
            episode = _load_episode_shard(path)
        except Exception as exc:  # corrupted/partial shard: recompute that episode
            print(f"  token shard unreadable, will recompute: {path.name} ({exc})")
            continue
        episodes[
            _episode_key(
                episode["split"],
                episode["map_name"],
                episode["episode_id"],
                episode["start_step"],
            )
        ] = episode
    return episodes


def _build_token_cache(
    data_config,
    observation_config: ObservationConfig,
    encoder,
    retriever,
    libraries: dict[str, CandidateLibrary],
    language_map: dict[str, np.ndarray],
    args,
    device: torch.device,
    cache_dir: Path,
) -> tuple[dict, dict]:
    cache_dir = ensure_output_dir(cache_dir, data_config=data_config)
    token_path = cache_dir / "t5_tokens.npz"
    meta_path = cache_dir / "t5_tokens.json"
    if token_path.exists() and meta_path.exists() and not args.rebuild_tokens:
        arrays = dict(np.load(token_path))
        return arrays, json.loads(meta_path.read_text(encoding="utf-8"))

    adapter = FPVObservationAdapter(data_config, observation_config)
    split_specs = [
        (
            "train_seen",
            args.train_maps,
            int(args.train_trajectories),
            0,
            args.train_range,
        ),
        (
            "val_seen",
            args.val_seen_maps,
            int(args.val_seen_trajectories),
            1,
            args.val_seen_range,
        ),
        (
            "val_unseen",
            args.val_unseen_maps,
            int(args.val_unseen_trajectories),
            2,
            args.val_unseen_range,
        ),
    ]
    use_shards = bool(getattr(args, "token_shards", False))
    shard_dir = cache_dir / "shards"
    if use_shards:
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_cache = _load_episode_shards(shard_dir)
        print(
            f"  token shards: {len(shard_cache)} episode(s) reused from {shard_dir}",
            flush=True,
        )
    else:
        shard_cache = {}
    train_episodes = []
    val_specs = []
    try:
        for split, maps, count, split_id, range_spec in split_specs[:1]:
            dataset = CityNavDataset(
                data_config, split, include_processed_descriptions=False
            )
            records = _select_teacher_records(
                dataset, maps, count, int(args.steps_per_trajectory)
            )
            low, high = _parse_range(range_spec, len(records))
            records = records[low:high]
            if getattr(args, "group_by_map", False):
                records = sorted(records, key=lambda item: item.map_name)
            if range_spec is not None:
                print(
                    f"  {split} record range {low}:{high} of {count} selected",
                    flush=True,
                )
            for record in records:
                result = record.teacher(dataset.teacher_config)
                total = len(result.actions)
                if total < int(args.steps_per_trajectory):
                    continue
                starts = {
                    min(
                        max(1, total // 3),
                        total - int(args.steps_per_trajectory),
                    ),
                    total - int(args.steps_per_trajectory),
                }
                for start in sorted(starts):
                    if start < 0:
                        continue
                    key = _episode_key(
                        split, record.map_name, record.episode_id, start
                    )
                    episode = shard_cache.get(key)
                    if episode is None:
                        episode = _collect_episode_frames(
                            record,
                            result,
                            observation_config=observation_config,
                            adapter=adapter,
                            encoder=encoder,
                            retriever=retriever,
                            library=libraries[record.map_name],
                            language_features=language_map[record.description],
                            start=start,
                            steps=int(args.steps_per_trajectory),
                            device=device,
                            crop_radius=float(args.crop_radius),
                        )
                        if episode is not None and use_shards:
                            _save_episode_shard(
                                _episode_shard_path(
                                    shard_dir,
                                    split,
                                    record.map_name,
                                    record.episode_id,
                                    start,
                                ),
                                episode,
                            )
                    if episode is not None:
                        train_episodes.append(episode)
                        if len(train_episodes) % 100 == 0:
                            print(
                                f"  train episodes={len(train_episodes)} "
                                f"frames={sum(len(e['actions']) for e in train_episodes)}",
                                flush=True,
                            )
        for split, maps, count, split_id, range_spec in split_specs[1:]:
            dataset = CityNavDataset(
                data_config, split, include_processed_descriptions=False
            )
            records = _select_teacher_records(
                dataset, maps, count, int(args.steps_per_trajectory)
            )
            low, high = _parse_range(range_spec, len(records))
            records = records[low:high]
            if getattr(args, "group_by_map", False):
                records = sorted(records, key=lambda item: item.map_name)
            if range_spec is not None:
                print(
                    f"  {split} record range {low}:{high} of {count} selected",
                    flush=True,
                )
            for record in records:
                result = record.teacher(dataset.teacher_config)
                total = len(result.actions)
                if total < int(args.steps_per_trajectory):
                    continue
                starts = {
                    min(
                        max(1, total // 3),
                        total - int(args.steps_per_trajectory),
                    ),
                    total - int(args.steps_per_trajectory),
                }
                for start in sorted(starts):
                    if start < 0:
                        continue
                    key = _episode_key(
                        split, record.map_name, record.episode_id, start
                    )
                    episode = shard_cache.get(key)
                    if episode is None:
                        episode = _collect_episode_frames(
                            record,
                            result,
                            observation_config=observation_config,
                            adapter=adapter,
                            encoder=encoder,
                            retriever=retriever,
                            library=libraries[record.map_name],
                            language_features=language_map[record.description],
                            start=start,
                            steps=int(args.steps_per_trajectory),
                            device=device,
                            crop_radius=float(args.crop_radius),
                        )
                        if episode is not None and use_shards:
                            _save_episode_shard(
                                _episode_shard_path(
                                    shard_dir,
                                    split,
                                    record.map_name,
                                    record.episode_id,
                                    start,
                                ),
                                episode,
                            )
                    if episode is not None:
                        val_specs.append((split, split_id, episode))
        if getattr(args, "collect_only", False):
            if not use_shards:
                raise SystemExit("--collect-only requires --token-shards")
            print(
                f"  collect-only: {len(train_episodes)} train + "
                f"{len(val_specs)} val episode(s) ready on disk; "
                "exit before threshold/finalization",
                flush=True,
            )
            return None, {
                "frames": 0,
                "trajectories": {
                    "train_seen": len(train_episodes),
                    "val_seen": sum(
                        1 for item in val_specs if item[0] == "val_seen"
                    ),
                    "val_unseen": sum(
                        1 for item in val_specs if item[0] == "val_unseen"
                    ),
                },
                "action_counts": {},
                "gab_reliability_threshold": None,
                "token_file": None,
                "token_bytes": 0,
                "test_unseen_read": False,
            }

        train_heatmaps = [
            heat for episode in train_episodes for heat in episode["heatmaps"]
        ]
        threshold = fit_reliability_threshold(train_heatmaps, percentile=60.0)
        gab_config = GABConfig(reliability_threshold=threshold)
        train_parts = [
            _finalize_episode(episode, gab_config, 0)
            for episode in train_episodes
        ]
        all_parts = [part[0] for part in train_parts]
        all_meta = [item for part in train_parts for item in part[1]]

        for split, split_id, episode in val_specs:
            arrays, metadata = _finalize_episode(
                episode, gab_config, split_id
            )
            all_parts.append(arrays)
            all_meta.extend(metadata)
    finally:
        adapter.close()

    merged = _concat_parts(all_parts)
    np.savez_compressed(token_path, **merged)
    action_counts = Counter()
    for split_id in (0, 1, 2):
        counts = Counter(
            int(action)
            for action, sid in zip(merged["actions"], merged["split_id"])
            if int(sid) == int(split_id)
        )
        action_counts[str(split_id)] = {
            ACTION_NAMES[int(action)]: int(count)
            for action, count in sorted(counts.items())
        }
    metadata = {
        "token_file": str(token_path.relative_to(PROJECT_ROOT)),
        "token_bytes": int(token_path.stat().st_size),
        "frames": int(len(merged["actions"])),
        "trajectories": {
            "train_seen": len(train_episodes),
            "val_seen": sum(1 for item in val_specs if item[0] == "val_seen"),
            "val_unseen": sum(1 for item in val_specs if item[0] == "val_unseen"),
        },
        "action_counts": action_counts,
        "gab_reliability_threshold": float(gab_config.reliability_threshold),
        "gab_config": {
            "alpha": gab_config.alpha,
            "beta": gab_config.beta,
            "forward_beta": gab_config.forward_beta,
            "max_age": gab_config.max_age,
            "bins": gab_config.bins,
            "fov_deg": gab_config.fov_deg,
            "turn_deg": gab_config.turn_deg,
        },
        "frames_metadata": all_meta,
        "test_unseen_read": False,
    }
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return merged, metadata


def _t5_batch(
    arrays: dict,
    indices: np.ndarray,
    device: torch.device,
    *,
    use_retrieval: bool,
) -> dict:
    actions = torch.from_numpy(
        arrays["actions"][indices].astype(np.int64)
    ).to(device)
    batch = {
        "current_tokens": torch.from_numpy(
            arrays["current_tokens"][indices].astype(np.float32)
        ).to(device),
        "history_vis": torch.from_numpy(
            arrays["history_vis"][indices].astype(np.float32)
        ).to(device),
        "history_pose": torch.from_numpy(
            arrays["history_pose"][indices].astype(np.float32)
        ).to(device),
        "history_action": torch.from_numpy(
            arrays["history_action"][indices].astype(np.float32)
        ).to(device),
        "history_mask": torch.from_numpy(
            arrays["history_mask"][indices].astype(np.float32)
        ).to(device),
        "language_features": torch.from_numpy(
            arrays["language_features"][indices].astype(np.float32)
        ).to(device),
        "actions": actions,
    }
    if use_retrieval:
        batch["goal_tokens"] = torch.from_numpy(
            arrays["goal_tokens"][indices].astype(np.float32)
        ).to(device)
        batch["gab"] = torch.from_numpy(
            arrays["gab"][indices].astype(np.float32)
        ).to(device)
        batch["map_feat"] = torch.from_numpy(
            arrays["map_feat"][indices].astype(np.float32)
        ).to(device)
    return batch


@torch.no_grad()
def _evaluate_t5(
    model,
    arrays: dict,
    indices: np.ndarray,
    device: torch.device,
    *,
    use_retrieval: bool,
    batch_size: int = 32,
) -> dict:
    model.eval()
    total = len(indices)
    loss_sum = 0.0
    accuracy_correct = 0
    move_correct = 0
    move_total = 0
    counts = Counter()
    correct = Counter()
    stop_tp = stop_fp = stop_fn = 0
    start = time.perf_counter()
    for begin in range(0, total, int(batch_size)):
        chunk = indices[begin:begin + int(batch_size)]
        batch = _t5_batch(arrays, chunk, device, use_retrieval=use_retrieval)
        inputs = {
            key: value for key, value in batch.items() if key != "actions"
        }
        components = model.forward_components(**inputs)
        losses = factorized_loss(components, batch["actions"])
        size = len(chunk)
        loss_sum += float(losses["loss"].item()) * size
        accuracy_correct += int(round(float(losses["accuracy"]) * size))
        predictions = losses["predictions"].detach().cpu().numpy()
        truth = batch["actions"].detach().cpu().numpy()
        for name in ACTION_NAMES.values():
            counts[name] += int(losses["per_class_counts"][name])
            correct[name] += int(losses["per_class_correct"][name])
        move_mask = truth != int(STOP)
        move_total += int(move_mask.sum())
        move_correct += int((predictions[move_mask] == truth[move_mask]).sum())
        stop_true = truth == int(STOP)
        stop_pred = predictions == int(STOP)
        stop_tp += int((stop_true & stop_pred).sum())
        stop_fp += int((~stop_true & stop_pred).sum())
        stop_fn += int((stop_true & ~stop_pred).sum())
    return {
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
        "eval_seconds": round(float(time.perf_counter() - start), 3),
    }


def _train_t5_model(
    name: str,
    arrays: dict,
    train_indices: np.ndarray,
    eval_indices: dict[str, np.ndarray],
    device: torch.device,
    args,
) -> dict:
    use_retrieval = name.endswith(":full")
    model_name = name.split(":")[0]
    model = build_t5_model(
        model_name,
        use_retrieval=use_retrieval,
        use_map=use_retrieval,
        text_dim=LANGUAGE_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.mec_lr),
        weight_decay=float(args.mec_weight_decay),
    )
    training = TrainingConfig(
        batch_size=int(args.mec_batch_size),
        lr=float(args.mec_lr),
        weight_decay=float(args.mec_weight_decay),
        grad_clip=1.0,
        seed=int(args.seed),
    )
    initial = {
        split: _evaluate_t5(
            model, arrays, indices, device,
            use_retrieval=use_retrieval,
            batch_size=int(args.mec_batch_size),
        )
        for split, indices in eval_indices.items()
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(args.seed)
    trace = []
    start = time.perf_counter()
    for step in range(int(args.mec_steps)):
        indices = rng.choice(
            train_indices,
            size=min(int(args.mec_batch_size), len(train_indices)),
            replace=False,
        )
        batch = _t5_batch(arrays, indices, device, use_retrieval=use_retrieval)
        result = train_step(model, optimizer, batch, training)
        if step % 25 == 0 or step == int(args.mec_steps) - 1:
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
    train_seconds = time.perf_counter() - start
    final = {
        split: _evaluate_t5(
            model, arrays, indices, device,
            use_retrieval=use_retrieval,
            batch_size=int(args.mec_batch_size),
        )
        for split, indices in eval_indices.items()
    }
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
    checkpoint_dir = ensure_output_dir(
        args.cache_dir / "mec" / name.replace(":", "_").lower()
    )
    torch.save(model.state_dict(), checkpoint_dir / "mec.pt")
    return {
        "name": name,
        "model": model_name,
        "use_retrieval": bool(use_retrieval),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "initial": initial,
        "final": final,
        "trace": trace,
        "train_seconds": round(float(train_seconds), 3),
        "frames_per_second": round(
            float(args.mec_steps * min(int(args.mec_batch_size), len(train_indices)))
            / max(train_seconds, 1e-9), 2
        ),
        "gpu_memory": peak_memory,
        "checkpoint": str(
            (checkpoint_dir / "mec.pt").relative_to(PROJECT_ROOT)
        ),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# T5 + 冻结 BERT 地图检索 Smoke 报告",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 设备: `{report['device']}`",
        f"- 数据: train_seen / val_seen / val_unseen；test_unseen 未读取",
        f"- 原始数据快照未变化: `{report['raw_data_unchanged']}`",
        "",
        "## 1. 输入模态",
        "",
        "| 模态 | 维度 |",
        "|---|---|",
        "| Current RGB tokens | 576 × 384 |",
        "| Goal RGB tokens（Top-3 检索候选） | 3 × 384 |",
        "| History visual | 16 × 768 |",
        "| Pose history | 16 × 12（6 连续 + 6 动作 one-hot） |",
        "| Action history | 16 × 6 |",
        "| History mask | 16 |",
        "| GAB | 20 |",
        "| Top-5 map retrieval | 22 |",
        "| Frozen BERT | 768 |",
        "",
        "## 2. 候选库与检索",
        "",
        f"- 网格 stride: {report['retrieval']['candidate_stride_m']} m",
        f"- 尺度: {report['retrieval']['candidate_scales_m']} m",
        f"- 候选地图数: {report['retrieval']['map_count']}",
        f"- 候选总数: {report['retrieval']['candidate_count']}",
        f"- 训练视觉编码器: {report['retrieval']['train_visual_encoder']}",
        "",
        "| split | samples | with positive | Recall@1 | Recall@5 | Recall@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, row in report["retrieval"]["val"].items():
        lines.append(
            f"| {split} | {row['samples']} | {row['with_positive']} | "
            f"{row['recall@1']} | {row['recall@5']} | {row['recall@10']} |"
        )
    lines += [
        "",
        "Top-3 Goal 不足时零填充并记录 `goal_valid`；Top-5 不足时 map 特征零填充。"
        "推理端不接收 target 坐标/ID/裁剪/教师动作/成功标签。",
        "",
        "## 3. T5 Token 数据",
        "",
        f"- 帧数: {report['tokens']['frames']}",
        f"- 轨迹数: {report['tokens']['trajectories']}",
        f"- GAB 阈值（仅 train 拟合）: {report['tokens']['gab_reliability_threshold']}",
        f"- 动作计数: {report['tokens']['action_counts']}",
        "",
        "## 4. 小规模 MEC 训练",
        "",
        "| model | retrieval | initial val_seen acc | final val_seen acc | final val_unseen acc | move acc (val_unseen) | Stop P/R (val_unseen) |",
        "|---|:---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["models"]:
        vs = result["final"].get("val_seen", {})
        vu = result["final"].get("val_unseen", {})
        iv = result["initial"].get("val_seen", {})
        lines.append(
            f"| {result['model']} | {result['use_retrieval']} | "
            f"{iv.get('accuracy', 0):.4f} | {vs.get('accuracy', 0):.4f} | "
            f"{vu.get('accuracy', 0):.4f} | {vu.get('move_accuracy', 0):.4f} | "
            f"{vu.get('stop_precision', 0):.3f}/{vu.get('stop_recall', 0):.3f} |"
        )
    lines += [
        "",
        "## 5. 结论与限制",
        "",
        "- 这是小规模检索/六模态/训练 Smoke，不是正式训练或泛化结论；",
        "- Goal RGB 来自语言检索的固定网格候选，不使用 Ground Truth 目标裁剪；",
        "- 候选检索训练仅用 train_seen 的 target_positions 生成正负标签；"
        "推理/验证排序不使用 target 坐标、ID 或裁剪；",
        "- map-free baseline 去掉 Goal/GAB 地图证据和 Top-5 map 分支，"
        "使用 current/history/pose/language；",
        "- 点云 FPV 有空洞，部分地图配准仍需更严格检查；",
        "- 未启动完整训练、RL、DQN/PPO 或 GSM。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    data_config = load_data_config()
    data_config.validate()
    before = snapshot_raw_dirs(data_config)
    observation_config = ObservationConfig(
        image_size=336,
        depth_size=int(args.render_size),
        goal_crop_size=96,
        goal_scales=(14, 28, 56),
        map_cache_size=2,
        fpv_width=int(args.render_size),
        fpv_height=int(args.render_size),
        fpv_crop_radius_m=float(args.crop_radius),
        fpv_splat_radius=0,
    )
    all_maps = sorted(
        set(args.train_maps) | set(args.val_seen_maps) | set(args.val_unseen_maps)
    )
    encoder = _load_encoder(args, device)
    bert, bert_meta = _load_bert()

    dataset_train = CityNavDataset(
        data_config, "train_seen", include_processed_descriptions=False
    )
    dataset_val_seen = CityNavDataset(
        data_config, "val_seen", include_processed_descriptions=False
    )
    dataset_val_unseen = CityNavDataset(
        data_config, "val_unseen", include_processed_descriptions=False
    )
    descriptions = []
    for dataset in (dataset_train, dataset_val_seen, dataset_val_unseen):
        descriptions.extend(record.description for record in dataset.records)
    language_map, language_meta = _language_map(
        descriptions, bert, args.cache_dir / "language_features.npz"
    )
    if getattr(args, "prep_only", False):
        after = snapshot_raw_dirs(data_config)
        assert_snapshot_unchanged(before, after)
        print(
            f"language cache ready: {language_meta['unique_descriptions']} "
            f"description(s) -> {args.cache_dir / 'language_features.npz'}",
            flush=True,
        )
        return 0

    libraries = _build_libraries(
        data_config, observation_config, all_maps, encoder, device, args
    )
    train_samples = _retrieval_samples(
        dataset_train, libraries, language_map, args.train_maps,
        args.retrieval_train_per_map, "train_seen",
    )
    val_seen_samples = _retrieval_samples(
        dataset_val_seen, libraries, language_map, args.val_seen_maps,
        args.retrieval_val_per_map, "val_seen",
    )
    val_unseen_samples = _retrieval_samples(
        dataset_val_unseen, libraries, language_map, args.val_unseen_maps,
        args.retrieval_val_per_map, "val_unseen",
    )
    train_encoder = not bool(args.freeze_visual_encoder)
    reuse_retrieval = bool(getattr(args, "reuse_retrieval", False)) and (
        args.cache_dir / "retrieval" / "retriever.pt"
    ).exists()
    if reuse_retrieval:
        retriever_path = args.cache_dir / "retrieval" / "retriever.pt"
        encoder_path = args.cache_dir / "retrieval" / "encoder.pt"
        retriever = LanguageMapRetriever().to(device)
        retriever.load_state_dict(
            torch.load(retriever_path, map_location="cpu", weights_only=True)
        )
        retriever.eval()
        if encoder_path.exists():
            stored = torch.load(encoder_path, map_location="cpu", weights_only=True)
            current = encoder.state_dict()
            diff = max(
                float((current[key].detach().cpu() - stored[key]).abs().max())
                for key in current
            )
            if diff > 0.0:
                raise SystemExit(
                    "reuse-retrieval: stored encoder differs from the frozen "
                    f"checkpoint (max abs diff {diff:.3e}); refusing"
                )
        retrieval_val = {}
        retrieval_info = {
            "best_retriever_state": None,
            "best_encoder_state": None,
            "trace": [],
            "best_recall@1": None,
        }
        print(f"reuse-retrieval: loaded {retriever_path}", flush=True)
    else:
        retriever, retrieval_val, retrieval_info = train_retriever(
            train_samples,
            val_seen_samples + val_unseen_samples,
            encoder=encoder,
            device=device,
            steps=int(args.retrieval_steps),
            lr=float(args.retrieval_lr),
            encoder_lr=float(args.encoder_lr),
            train_encoder=train_encoder,
            seed=int(args.seed),
            log_every=20,
            eval_every=max(20, int(args.retrieval_steps) // 2),
        )
        if retrieval_info["best_retriever_state"] is not None:
            retriever.load_state_dict(retrieval_info["best_retriever_state"])
        if train_encoder and retrieval_info["best_encoder_state"] is not None:
            encoder.load_state_dict(retrieval_info["best_encoder_state"])
    encoder.eval()
    encoder.requires_grad_(False)
    retriever.eval()
    for library in libraries.values():
        library.embeddings = encode_candidate_images(
            library.images, encoder, device
        ).detach().cpu().numpy()
    retrieval_dir = ensure_output_dir(
        args.cache_dir / "retrieval", data_config=data_config
    )
    torch.save(encoder.state_dict(), retrieval_dir / "encoder.pt")
    torch.save(retriever.state_dict(), retrieval_dir / "retriever.pt")
    retrieval_meta = {
        "candidate_stride_m": float(args.candidate_stride),
        "candidate_scales_m": [float(v) for v in args.candidate_scales],
        "map_count": len(libraries),
        "candidate_count": int(sum(lib.count for lib in libraries.values())),
        "train_visual_encoder": bool(train_encoder),
        "train_samples": len(train_samples),
        "val": retrieval_val,
        "trace": retrieval_info["trace"],
        "best_recall@1": retrieval_info["best_recall@1"],
        "encoder_checkpoint": (
            str(args.encoder_checkpoint)
            if args.encoder_checkpoint and Path(args.encoder_checkpoint).exists()
            else None
        ),
        "retriever_checkpoint": str(
            (retrieval_dir / "retriever.pt").relative_to(PROJECT_ROOT)
        ),
        "encoder_state": str(
            (retrieval_dir / "encoder.pt").relative_to(PROJECT_ROOT)
        ),
        "test_unseen_read": False,
    }

    token_arrays, token_meta = _build_token_cache(
        data_config,
        observation_config,
        encoder,
        retriever,
        libraries,
        language_map,
        args,
        device,
        args.cache_dir / "tokens",
    )
    if getattr(args, "collect_only", False):
        after = snapshot_raw_dirs(data_config)
        assert_snapshot_unchanged(before, after)
        print(
            "token collection finished (collect-only); "
            f"train={token_meta['trajectories']['train_seen']} "
            f"val_seen={token_meta['trajectories']['val_seen']} "
            f"val_unseen={token_meta['trajectories']['val_unseen']}",
            flush=True,
        )
        print(f"  raw data unchanged: {after['digest']}", flush=True)
        return 0
    train_indices = np.flatnonzero(token_arrays["split_id"] == 0)
    val_indices = {
        "val_seen": np.flatnonzero(token_arrays["split_id"] == 1),
        "val_unseen": np.flatnonzero(token_arrays["split_id"] == 2),
    }
    model_results = []
    for model_name in ("small", "transformer_m"):
        for full_name in (f"{model_name}:full", f"{model_name}:map_free"):
            model_results.append(
                _train_t5_model(
                    full_name,
                    token_arrays,
                    train_indices,
                    val_indices,
                    device,
                    args,
                )
            )
    after = snapshot_raw_dirs(data_config)
    assert_snapshot_unchanged(before, after)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "seed": int(args.seed),
        "input_dims": T5_FEATURE_DIMS,
        "bert": {**bert_meta, **language_meta},
        "retrieval": retrieval_meta,
        "tokens": {
            key: token_meta[key]
            for key in (
                "frames", "trajectories", "action_counts",
                "gab_reliability_threshold", "token_file", "token_bytes",
                "test_unseen_read",
            )
        },
        "models": model_results,
        "raw_data_unchanged": True,
        "raw_snapshot": {
            "file_count": int(after["file_count"]),
            "total_bytes": int(after["total_bytes"]),
            "digest": str(after["digest"]),
        },
        "note": (
            "Small-scale retrieval + T5/language smoke; not a full training "
            "result. test_unseen was never read."
        ),
    }
    output_dir = Path(args.output_dir)
    json_path = ensure_output_file(
        output_dir / "t5_report.json", data_config=data_config
    )
    md_path = ensure_output_file(
        output_dir / "t5_report.md", data_config=data_config
    )
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    print("T5 retrieval + language smoke completed")
    print(
        f"  maps={len(libraries)} candidates={retrieval_meta['candidate_count']} "
        f"train_encoder={train_encoder} tokens={token_meta['frames']}"
    )
    for split, row in retrieval_val.items():
        print(
            f"  retrieval {split:11s} R@1={row['recall@1']} "
            f"R@5={row['recall@5']} R@10={row['recall@10']} "
            f"samples={row['samples']} with_pos={row['with_positive']}"
        )
    for result in model_results:
        vu = result["final"].get("val_unseen", {})
        print(
            f"  {result['name']:24s} val_unseen acc={vu.get('accuracy', 0):.4f} "
            f"move={vu.get('move_accuracy', 0):.4f} "
            f"stop_P/R={vu.get('stop_precision', 0):.3f}/"
            f"{vu.get('stop_recall', 0):.3f}"
        )
    print(f"  wrote {json_path.relative_to(PROJECT_ROOT)} and {md_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
