"""Closed-loop CityNav evaluation for a frozen encoder + trained MEC.

At every step the agent renders the current first-person view, encodes it with
the frozen encoder, updates causal history/GAB, asks the MEC for one of the six
official actions, executes it, and re-renders.  The model never receives future
trajectory points, future teacher actions, target coordinates or success labels;
the target is used only by the evaluator.

Usage (single process):
  python closed_loop_eval.py --split val_seen --episodes 20 \
    --mec-checkpoint data_cache/t5_frozen_1000/mec/small_full/mec.pt \
    --encoder-checkpoint runs/.../best_encoder.pt \
    --retriever-checkpoint data_cache/t5_frozen_1000/retrieval/retriever.pt \
    --model small --use-retrieval --out results/closed_loop_small.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Eight shards run at once; torch's default thread pool oversubscribes the
# machine and pushed closed-loop latency from ~2 s to ~10 s per step.
torch.set_num_threads(int(os.environ.get("CITYNAV_TORCH_THREADS", "1")))

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ObservationConfig, TeacherConfig, TextConfig, load_data_config  # noqa: E402
from src.data.actions import ACTION_NAMES, STOP  # noqa: E402
from src.data.candidates import (  # noqa: E402
    build_candidate_library,
    build_goal_tokens,
    map_features_from_scores,
    topk_indices,
)
from src.data.citynav_dataset import CityNavDataset  # noqa: E402
from src.data.fpv_observation import FPVObservationAdapter  # noqa: E402
from src.data.gab import GABConfig, GABState  # noqa: E402
from src.data.language_adapter import build_language_adapter  # noqa: E402
from src.data.t5_features import history_arrays, visual_descriptor  # noqa: E402
from src.data.trajectory_adapter import Pose5D, transition_pose  # noqa: E402
from src.models.mec_common import build_spatial_heatmap  # noqa: E402
from src.models.t5_mec import build_t5_model  # noqa: E402
from src.models.token_utils import encode_current_tokens  # noqa: E402
from src.models.vision_encoder import VisualTokenEncoder  # noqa: E402
from src.models import load_encoder_checkpoint  # noqa: E402
from src.training.retrieval import LanguageMapRetriever  # noqa: E402

SUCCESS_RADIUS_M = 20.0
NDTW_SIGMA_M = 12.0


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _ndtw(pred: np.ndarray, ref: np.ndarray, sigma: float = NDTW_SIGMA_M) -> float:
    """Normalised DTW with exponential distance (lower distance = higher score)."""
    if len(pred) == 0 or len(ref) == 0:
        return 0.0
    d = np.linalg.norm(pred[:, None, :] - ref[None, :, :], axis=-1)
    cost = np.exp(-d / sigma)
    n, m = cost.shape
    dp = np.zeros((n + 1, m + 1), dtype=np.float64)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = cost[i - 1, j - 1] + max(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n, m))


def _load_encoder(path: Path, device: torch.device) -> VisualTokenEncoder:
    encoder = VisualTokenEncoder()
    load_encoder_checkpoint(encoder, str(path), map_location="cpu")
    return encoder.to(device).eval()


def run_episode(
    record,
    *,
    encoder,
    model,
    retriever,
    library,
    language_features: np.ndarray,
    adapter: FPVObservationAdapter,
    teacher: TeacherConfig,
    device: torch.device,
    observation_config: ObservationConfig,
    use_retrieval: bool,
    max_steps: int,
    crop_radius: float,
    gab_config: GABConfig,
) -> dict:
    scores = None
    goal_tokens = None
    if use_retrieval:
        with torch.no_grad():
            language = torch.from_numpy(language_features).float().to(device).reshape(1, -1)
            scores = retriever(language, torch.from_numpy(
                np.asarray(library.embeddings, dtype=np.float32)
            ).to(device)).cpu().numpy().reshape(-1)
            top5 = topk_indices(scores, 5)
            goal_tokens, goal_valid = build_goal_tokens(
                np.asarray(library.embeddings), top5[:3], k=3
            )

    pose = record.start_pose
    poses = [pose]
    actions: list[int] = []
    visuals: list[np.ndarray] = []
    positions = [np.asarray([pose.x, pose.y], dtype=np.float64)]
    target_xyz = np.asarray(record.target_xyz, dtype=np.float64)
    step_distances = [float(np.linalg.norm(pose.xyz - target_xyz))]
    step_xy_distances = [
        float(np.linalg.norm(np.asarray([pose.x, pose.y]) - target_xyz[:2]))
    ]
    gab_state = GABState(gab_config)
    previous_action = -1
    latencies: list[float] = []
    stopped = False

    for _ in range(int(max_steps)):
        t0 = time.perf_counter()
        observation = adapter.render_current(
            record.map_name, pose, crop_radius_m=float(crop_radius)
        )
        tokens = encode_current_tokens(
            encoder, observation["current_rgb"][None], device, requires_grad=False
        )
        descriptor = visual_descriptor(tokens)[0].detach().cpu().numpy()
        visuals.append(descriptor.astype(np.float32))

        history_pose = list(poses)
        history_actions = actions + [0]
        h_vis, h_pose, h_action, h_mask = history_arrays(
            history_pose, history_actions, len(poses) - 1, np.asarray(visuals)
        )
        inputs = {
            "current_tokens": tokens.float(),
            "history_vis": torch.from_numpy(h_vis)[None].to(device),
            "history_pose": torch.from_numpy(h_pose)[None].to(device),
            "history_action": torch.from_numpy(h_action)[None].to(device),
            "history_mask": torch.from_numpy(h_mask)[None].to(device),
            "language_features": torch.from_numpy(language_features)[None].to(device),
        }
        if use_retrieval:
            heatmap = build_spatial_heatmap(
                tokens.float(),
                torch.from_numpy(goal_tokens).float()[None].to(device),
            )[0].detach().cpu().numpy()
            gab_state.update(heatmap, previous_action)
            map_feature, _ = map_features_from_scores(
                pose, library.centers, scores, top_k=5
            )
            inputs["goal_tokens"] = torch.from_numpy(goal_tokens)[None].float().to(device)
            inputs["gab"] = torch.from_numpy(gab_state.features())[None].float().to(device)
            inputs["map_feat"] = torch.from_numpy(map_feature)[None].float().to(device)

        with torch.no_grad():
            logits = model(**inputs)
        action = int(logits.argmax(dim=-1).item())
        latencies.append(time.perf_counter() - t0)
        actions.append(action)
        previous_action = action
        if action == int(STOP):
            stopped = True
            step_distances.append(step_distances[-1])
            step_xy_distances.append(step_xy_distances[-1])
            break
        pose = transition_pose(pose, action, teacher)
        poses.append(pose)
        positions.append(np.asarray([pose.x, pose.y], dtype=np.float64))
        step_distances.append(float(np.linalg.norm(pose.xyz - target_xyz)))
        step_xy_distances.append(
            float(np.linalg.norm(np.asarray([pose.x, pose.y]) - target_xyz[:2]))
        )

    target = np.asarray(record.target_xyz, dtype=np.float64)
    final = np.asarray([poses[-1].x, poses[-1].y, poses[-1].z], dtype=np.float64)
    ne = float(np.linalg.norm(final - target))
    path = np.asarray(positions)
    dists = np.linalg.norm(path - target[None, :2], axis=1)
    min_dist = float(dists.min())
    reference = np.asarray([[p.x, p.y] for p in record.poses], dtype=np.float64)
    ref_len = _polyline_length(reference)
    exec_len = _polyline_length(path)
    success = ne <= SUCCESS_RADIUS_M
    spl = 0.0
    if success and ref_len > 0:
        spl = ref_len / max(exec_len, ref_len)
    step_distances_array = np.asarray(step_distances, dtype=np.float64)
    entered_20m = bool((step_distances_array <= SUCCESS_RADIUS_M).any())
    first_entry = (
        int(np.argmax(step_distances_array <= SUCCESS_RADIUS_M))
        if entered_20m else None
    )
    left_after_entry = bool(
        entered_20m and (step_distances_array[first_entry:] > SUCCESS_RADIUS_M).any()
    )
    stopped_within_20m = bool(stopped and step_distances_array[-1] <= SUCCESS_RADIUS_M)
    max_steps_reached = bool(not stopped and len(actions) >= int(max_steps))
    return {
        "episode_id": record.episode_id,
        "map_name": record.map_name,
        "steps": len(actions),
        "max_steps_reached": max_steps_reached,
        "stopped": bool(stopped),
        "stopped_within_20m": stopped_within_20m,
        "success": bool(success),
        "oracle_success": bool(min_dist <= SUCCESS_RADIUS_M),
        "oracle_success_3d": bool(step_distances_array.min() <= SUCCESS_RADIUS_M),
        "ne_m": ne,
        "min_dist_m": min_dist,
        "min_dist_3d_m": float(step_distances_array.min()),
        "entered_20m": entered_20m,
        "first_entry_step": first_entry,
        "left_after_entering_20m": left_after_entry,
        "spl": float(spl),
        "ndtw": _ndtw(path, reference),
        "action_counts": {
            ACTION_NAMES[int(a)]: int(c) for a, c in Counter(actions).items()
        },
        "action_sequence": [int(a) for a in actions],
        "step_distances_m": [float(v) for v in step_distances],
        "step_xy_distances_m": [float(v) for v in step_xy_distances],
        "mean_step_latency_s": float(np.mean(latencies)) if latencies else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ply-root", required=True)
    ap.add_argument("--split", default="val_seen")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument(
        "--episode-indices",
        type=Path,
        default=None,
        help="JSON list of record indices; overrides --episodes selection",
    )
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--crop-radius", type=float, default=120.0)
    ap.add_argument("--encoder-checkpoint", type=Path, required=True)
    ap.add_argument("--mec-checkpoint", type=Path, required=True)
    ap.add_argument("--retriever-checkpoint", type=Path)
    ap.add_argument("--model", default="small")
    ap.add_argument("--use-retrieval", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-maps", type=int, default=4)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument(
        "--token-meta",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data_cache/t5_frozen_1000/tokens/t5_tokens.json"
        ),
        help="token metadata used to reuse the training-time GAB threshold",
    )
    ap.add_argument(
        "--gab-threshold",
        type=float,
        default=None,
        help="override the GAB reliability threshold (default: token metadata)",
    )
    ap.add_argument(
        "--journal",
        type=Path,
        default=None,
        help="append one JSON line per finished episode (default: <out>.jsonl)",
    )
    args = ap.parse_args()

    if args.split == "test_unseen":
        raise PermissionError("test_unseen stays sealed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = load_data_config(data_root=args.data_root, ply_root=args.ply_root)
    # Must match the token-build observation config exactly: 336 px FPV,
    # 120 m crop, no point splatting.  The default 300 m crop would change the
    # rendered distribution and is several times slower.
    observation_config = ObservationConfig(
        image_size=336,
        depth_size=336,
        goal_crop_size=96,
        goal_scales=(14, 28, 56),
        map_cache_size=2,
        fpv_width=336,
        fpv_height=336,
        fpv_crop_radius_m=float(args.crop_radius),
        fpv_splat_radius=0,
    )
    dataset = CityNavDataset(data_config, args.split, include_processed_descriptions=False)
    records = dataset.records
    episode_indices = None
    if args.episode_indices is not None:
        episode_indices = [int(v) for v in json.loads(
            args.episode_indices.read_text(encoding="utf-8")
        )]
        if len(set(episode_indices)) != len(episode_indices):
            raise SystemExit("--episode-indices contains duplicates")
        if any(i < 0 or i >= len(records) for i in episode_indices):
            raise SystemExit("--episode-indices contains out-of-range index")
        records = [records[i] for i in episode_indices]
    elif args.episodes < len(records):
        idx = np.linspace(0, len(records) - 1, args.episodes, dtype=int)
        records = [records[int(i)] for i in idx]
    # Group by map before sharding: episodes are independent and metrics are
    # aggregated, and this avoids reloading a PLY on almost every episode
    # (map_cache_size=2).  Shards take contiguous blocks of the map-sorted
    # list, so each process covers only a few maps instead of all of them.
    records = sorted(records, key=lambda record: record.map_name)
    if args.shards > 1:
        total = len(records)
        low = int(args.shard_index) * total // int(args.shards)
        high = (int(args.shard_index) + 1) * total // int(args.shards)
        records = records[low:high]

    encoder = _load_encoder(args.encoder_checkpoint, device)
    model = build_t5_model(
        args.model, use_retrieval=args.use_retrieval,
        use_map=args.use_retrieval, text_dim=768,
    ).to(device)
    state = torch.load(args.mec_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    retriever = None
    if args.use_retrieval:
        if not args.retriever_checkpoint:
            raise ValueError("--retriever-checkpoint required with --use-retrieval")
        retriever = LanguageMapRetriever().to(device)
        retriever.load_state_dict(
            torch.load(args.retriever_checkpoint, map_location="cpu", weights_only=True)
        )
        retriever.eval()

    language_adapter, _, _, _ = build_language_adapter(
        TextConfig(backend="bert", local_files_only=True), device="cpu"
    )
    adapter = FPVObservationAdapter(data_config, observation_config)
    teacher = TeacherConfig()
    threshold = args.gab_threshold
    if threshold is None and args.token_meta.exists():
        meta = json.loads(args.token_meta.read_text(encoding="utf-8"))
        threshold = float(meta["gab_reliability_threshold"])
    if threshold is None:
        threshold = 0.35
        print(
            "warning: token metadata not found; falling back to GAB threshold 0.35",
            flush=True,
        )
    print(f"GAB reliability threshold = {threshold:.6f}", flush=True)
    gab_config = GABConfig(reliability_threshold=float(threshold))

    libraries = {}
    for name in sorted({r.map_name for r in records}):
        libraries[name] = build_candidate_library(
            data_config, name, observation_config,
            encoder=encoder, device=device,
            stride_m=48.0, scales_m=(32.0, 64.0, 128.0),
        )

    journal_path = args.journal or args.out.with_name(args.out.stem + ".jsonl")
    journal_meta = {
        "split": args.split,
        "model": args.model,
        "use_retrieval": bool(args.use_retrieval),
        "episodes": int(args.episodes),
        "shard_index": int(args.shard_index),
        "shards": int(args.shards),
        "max_steps": int(args.max_steps),
        "episode_indices": episode_indices,
        "crop_radius": float(args.crop_radius),
        "gab_threshold": float(threshold),
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "mec_checkpoint": str(args.mec_checkpoint),
    }
    finished: dict = {}
    if journal_path.exists():
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if "meta" in item:
                if item["meta"] != journal_meta:
                    raise SystemExit(
                        f"journal {journal_path} belongs to a different run; "
                        "use a new --out/--journal"
                    )
            else:
                finished[(item["map_name"], item["episode_id"])] = item
        print(f"  journal: {len(finished)} episode(s) already finished", flush=True)
    else:
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(
            json.dumps({"meta": journal_meta}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    language_cache = {}
    results = []
    start = time.perf_counter()
    for i, record in enumerate(records, 1):
        if record.description not in language_cache:
            language_cache[record.description] = (
                language_adapter.encode_tensor([record.description], device="cpu")
                .cpu().numpy()[0]
            )
        key = (record.map_name, record.episode_id)
        if key in finished:
            results.append(finished[key])
        else:
            result = run_episode(
                record,
                encoder=encoder,
                model=model,
                retriever=retriever,
                library=libraries[record.map_name],
                language_features=language_cache[record.description],
                adapter=adapter,
                teacher=teacher,
                device=device,
                observation_config=observation_config,
                use_retrieval=bool(args.use_retrieval),
                max_steps=int(args.max_steps),
                crop_radius=float(args.crop_radius),
                gab_config=gab_config,
            )
            results.append(result)
            with journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        if i % 5 == 0:
            print(
                f"  {i}/{len(records)} episodes | "
                f"SR={np.mean([r['success'] for r in results]):.3f} "
                f"NE={np.mean([r['ne_m'] for r in results]):.1f}",
                flush=True,
            )

    summary = {
        "split": args.split,
        "episodes": len(results),
        "model": args.model,
        "use_retrieval": bool(args.use_retrieval),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "oracle_success_rate": float(np.mean([r["oracle_success"] for r in results])),
        "spl": float(np.mean([r["spl"] for r in results])),
        "ne_m": float(np.mean([r["ne_m"] for r in results])),
        "ndtw": float(np.mean([r["ndtw"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "stop_rate": float(np.mean([r["stopped"] for r in results])),
        "mean_step_latency_s": float(np.mean([
            r["mean_step_latency_s"] for r in results if r["mean_step_latency_s"]
        ])),
        "shard_index": int(args.shard_index),
        "shards": int(args.shards),
        "gab_threshold": float(threshold),
        "journal": str(journal_path),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "test_unseen_read": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "episodes": results},
                                   indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    adapter.close()
    language_adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
