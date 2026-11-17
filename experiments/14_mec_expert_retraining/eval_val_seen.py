#!/usr/bin/env python3
"""Stage 14 closed-loop evaluator for the five R2 preset MEC models.

Adapted from experiments/v4_closed_loop_revalidation/scripts/run_closed_loop.py
(same corrected navigation protocol, unchanged); only the model registry is
swapped to the Stage 14 presets in models.py.  Historical evaluator and its
results are untouched.

Usage:
  python run_closed_loop_14.py --model small|medium|ts|tm|large \
      --ckpt <best.pt> --manifest <m.json> --out-dir <dir> \
      [--task-file ids.txt] [--limit N] [--step-cap 200]

Original protocol notes (retained for reference):

Fixes vs the Stage-10E harness (which Stage 11D had copied verbatim):

1. HISTORY ORDER -- training (dataset_v2.history_block) fills slot 0 with the
   MOST RECENT step (t-1); the 10E live loop fed the OLDEST first.  Fixed to
   match training exactly: vis/pose/act slot k <- state t-(k+1).
2. POSE ENCODING -- training feeds relative pose in the CURRENT heading frame
   (dx*cos+dy*sin, -dx*sin+dy*cos, wrapped yaw diff); 10E fed ABSOLUTE x,y,yaw.
   Fixed to the training formula.
3. EMPTY CANDIDATES -- tasks whose map has an empty candidate library are NOT
   skipped; the full loop runs with the empty representation
   belief=[0,0,0,0,0, unc=50, ent=0, since=99, 0, 0], match=zeros (documented
   in INPUT notes); the model decides on its own (no ground-truth fallback).
4. PER-TASK RECORDS -- episode_id, instruction_id, block_id, candidate_status,
   candidate_count, initial/target/final position, final_distance, step_count,
   stop_reason, path_length, success, oracle_success, execution_status,
   exception, plus NE (final dist) and SPL (success-weighted path optimality,
   L* = start-target euclidean).
5. UNIFIED METRICS -- SR (stop within 20 m), OSR (ever within 20 m), NE
   (mean final distance), SPL; fixed step cap (default 200, the benchmark
   protocol); STOP is always the model's own prediction.

Usage:
  python run_closed_loop.py --model v4|v5|v5fix|v3 --ckpt <best.pt> \
      --manifest <m.json> --out-dir <dir> [--task-file ids.txt] \
      [--limit N] [--step-cap 200]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
sys.path.insert(0, str(REPO / "experiments/10_goal_query_mec_training"
                       "/10E_final_closed_loop/scripts"))
sys.path.insert(0, str(REPO / "experiments/10_goal_query_mec_training"
                       "/10D_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(REPO / "experiments/06_closed_loop_success_rate/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))

import matching  # noqa: E402
from closed_loop import (CLASSES, FORWARD_M, STOP_RADIUS, TURN_RAD, classify,  # noqa: E402
                         obstacle_for, segment_status)
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from pipeline import TARGET_HALF_M  # noqa: E402
from model11b import SPATIAL_CH, SPATIAL_HW  # noqa: E402
from models import build_model as build_preset_model  # noqa: E402
from v5fix_contract import (EMPTY_BELIEF, EMPTY_LANDMARKS,  # noqa: E402
                            EMPTY_MATCH)

CAL = json.loads((REPO / "experiments/09_online_visual_goal_matching"
                  "/09C_continuous_localization/results/09c_calibration.json")
                 .read_text(encoding="utf-8"))
BEL = json.loads((REPO / "experiments/09_online_visual_goal_matching"
                  "/09C_continuous_localization/results/09c_belief_calibration.json")
                 .read_text(encoding="utf-8"))
TAU = 0.875
AGREE_M = 10.0
HISTORY_LEN = 16
N_LM = 8


STAGE14_CONTROLLERS = {
    "small": "small_preset_mlp",
    "medium": "medium_preset_gca",
    "ts": "ts_preset_transformer_s",
    "tm": "tm_preset_transformer_m",
    "large": "large_preset_transformer_l",
}
NAMES = {"small": "Small", "medium": "Medium", "ts": "Transformer-S",
         "tm": "Transformer-M", "large": "Large"}
# 五种 Stage 14 预设全部使用 landmarks/states 几何输入
GEOM_MODELS = ("small", "medium", "ts", "tm", "large")


def build_model(name: str, device: str):
    """按 Stage 14 预设名构造 eval 模式模型；未知名拒绝。"""
    if name not in STAGE14_CONTROLLERS:
        raise SystemExit(f"unknown --model {name} (choices: "
                         f"{sorted(STAGE14_CONTROLLERS)})")
    model = build_preset_model(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_journal(out: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    for p in sorted(out.glob("journal*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[rec["episode_id"]] = rec
    return done


def history_inputs(hist_vis, hist_pose, hist_act, pos, yaw) -> tuple[np.ndarray, ...]:
    """Build the 16-slot history block EXACTLY like dataset_v2.history_block:
    slot 0 = most recent step, pose relative to the current heading frame."""
    vis = np.zeros((1, HISTORY_LEN, 2048), np.float32)
    pose = np.zeros((1, HISTORY_LEN, 3), np.float32)
    act = np.zeros((1, HISTORY_LEN, 4), np.float32)
    hmask = np.zeros((1, HISTORY_LEN), np.float32)
    cos, sin = math.cos(yaw), math.sin(yaw)
    for k in range(len(hist_vis)):
        j = len(hist_vis) - 1 - k            # k=0 -> most recent
        vis[0, k] = hist_vis[j]
        px, py, pyaw = hist_pose[j]
        dx, dy = float(px - pos[0]), float(py - pos[1])
        pose[0, k] = [dx * cos + dy * sin, -dx * sin + dy * cos,
                      float((pyaw - yaw + math.pi) % (2 * math.pi) - math.pi)]
        act[0, k, hist_act[j]] = 1.0
        hmask[0, k] = 1.0
    return vis, pose, act, hmask


def pool_history_from_current(cur: torch.Tensor) -> np.ndarray:
    """V5 修正 5b：单次前向的当前帧 token → history 池化特征。

    cur: (B, 49, 2048)（FrozenEncoder.__call__ 输出）；返回 (B, 2048)：
    与旧 `enc4.spatial_maps(...).transpose(0,2,3,1)` + mean + L2 归一路径
    数值一致（同一前向输出，同一池化公式，fp32 numpy 计算）。
    """
    maps = cur.cpu().numpy().reshape(cur.shape[0], SPATIAL_HW, SPATIAL_HW,
                                     SPATIAL_CH)
    pooled = maps.mean(axis=(1, 2))
    return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True),
                               1e-9)


def belief_features(est: np.ndarray, pos: np.ndarray, yaw: float, score: float,
                    unc: float, ent: float, since: float, dx: float,
                    dy: float) -> np.ndarray:
    """按训练公式（build_features_11b.causal_episode）构造 belief 10 维特征。

    与训练缓存逐值一致（契约测试 test_train_inference_contract.py 对拍 11B cache）。
    """
    vec = est - pos
    fwd = float(vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
    lat = float(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw))
    dist_est = float(np.linalg.norm(vec))
    bearing = math.atan2(lat, fwd)
    return np.asarray([fwd, lat, dist_est, bearing, score, unc, ent, since,
                       dx, dy], np.float32)


def landmark_inputs(belief: ContinuousBelief | None, empty: bool) -> np.ndarray:
    """构造 landmarks (1,8,3)=[mean_x, mean_y, weight]（与训练缓存同源构造）。"""
    lm = np.zeros((1, 8, 3), np.float32)
    if not empty:
        n_k = int(belief.means.shape[0])
        lm[0, :n_k, :2] = belief.means.astype(np.float32)
        lm[0, :n_k, 2] = belief.weights.astype(np.float32)
    return lm


TEACHER_NPZ = (REPO / "experiments/11_teacher_geometry_ablation"
               "/11B_transformer_m_v4/results/cache/val_teacher/"
               "teacher_val_seen.npz")
NDTW_SIGMA_M = 12.0


def load_teacher_refs() -> dict[int, np.ndarray]:
    """manifest index -> teacher trajectory xy (Val-Seen reference path)."""
    z = np.load(TEACHER_NPZ, allow_pickle=True)
    offs, states, keys = z["offsets"], z["states"], z["keys"]
    return {int(keys[i]): states[offs[i]:offs[i + 1], :2].astype(np.float64)
            for i in range(len(keys))}


def executed_path(steps: list[dict], start, final_pos) -> np.ndarray:
    """Positions after each executed action, starting at the task start."""
    p = np.asarray(start[:2], np.float64).copy()
    yaw = float(start[2])
    path = [p.copy()]
    for s in steps:
        a = int(s["action"])
        if a == 1:
            p = p + FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)])
        elif a == 2:
            yaw = (yaw + TURN_RAD + math.pi) % (2 * math.pi) - math.pi
        elif a == 3:
            yaw = (yaw - TURN_RAD + math.pi) % (2 * math.pi) - math.pi
        path.append(p.copy())
    if len(path) > 1:
        path[-1] = np.asarray(final_pos[:2], np.float64)
    return np.asarray(path)


def _ndtw(pred: np.ndarray, ref: np.ndarray, sigma: float = NDTW_SIGMA_M) -> float:
    """Normalised DTW with exponential distance (repo protocol, sigma=12 m)."""
    if len(pred) == 0 or len(ref) == 0:
        return 0.0
    d = np.linalg.norm(pred[:, None, :] - ref[None, :, :], axis=-1)
    cost = np.exp(-d / sigma)
    n, m = cost.shape
    dp = np.zeros((n + 1, m + 1), dtype=np.float64)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = cost[i - 1, j - 1] + max(
                dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n, m))


def straight_ref(start, target, spacing_m: float = 2.0) -> np.ndarray:
    a = np.asarray(start[:2], np.float64)
    b = np.asarray(target[:2], np.float64)
    n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / spacing_m)) + 1)
    return np.linspace(a, b, n)


def reference_for(task: dict, refs: dict[int, np.ndarray]) -> tuple[np.ndarray, bool]:
    ref = refs.get(int(task["index"]))
    if ref is not None and len(ref) >= 2:
        return ref, True
    return straight_ref([task["start_x"], task["start_y"], task["start_yaw"]],
                        [task["target_x"], task["target_y"]]), False


def write_summaries(out: Path, merged: dict[str, dict], n_expected: int,
                    refs: dict[int, np.ndarray], tasks: list[dict],
                    ckpt: str, manifest_sha: str, ckpt_sha: str,
                    step_cap: int) -> dict:
    """Write val_seen_episode_results.csv / val_seen_summary.json /
    failure_analysis.json from the merged journal."""
    by_id = {t["episode_id"]: t for t in tasks}
    recs = [merged[k] for k in sorted(merged)]
    for r in recs:                       # backfill nDTW for resumed records
        if r.get("ndtw") is None:
            task = by_id.get(r["episode_id"])
            ep = out / "episodes" / f"{r['episode_id'].replace(':', '_')}.json"
            traj = []
            if ep.exists():
                try:
                    traj = json.loads(ep.read_text(encoding="utf-8")).get(
                        "trajectory", [])
                except json.JSONDecodeError:
                    traj = []
            if task is None or not traj:
                continue
            ref, _ = reference_for(task, refs)
            path = executed_path(traj, r["start"], r.get("final_pos") or r["start"])
            r["ndtw"] = _ndtw(path, ref)
    n = len(recs)
    succ = [r for r in recs if r.get("success")]
    stopped = [r for r in recs if r.get("stopped")]
    stop_ok = [r for r in recs if r.get("stopped")
               and (r.get("stop_dist") is not None
                    and r["stop_dist"] <= STOP_RADIUS)]
    nes = [float(r["ne_m"]) for r in recs if r.get("ne_m") is not None]
    ndtws = [float(r["ndtw"]) for r in recs if r.get("ndtw") is not None]
    failures = Counter(r.get("failure", "unknown") for r in recs)
    summary = {
        "model": recs[0].get("controller") if recs else None,
        "ckpt": ckpt, "ckpt_sha": ckpt_sha, "manifest_sha": manifest_sha,
        "step_cap": step_cap, "success_radius_m": STOP_RADIUS,
        "completed_episodes": n, "expected_episodes": n_expected,
        "complete": n == n_expected,
        "sr": float(np.mean([bool(r.get("success")) for r in recs])) if recs else None,
        "osr": float(np.mean([bool(r.get("osr")) for r in recs])) if recs else None,
        "spl": float(np.mean([float(r.get("spl") or 0.0) for r in recs])) if recs else None,
        "ne_m": float(np.mean(nes)) if nes else None,
        "ndtw": float(np.mean(ndtws)) if ndtws else None,
        "ndtw_coverage": len(ndtws),
        "mean_steps": float(np.mean([int(r.get("steps", 0)) for r in recs])) if recs else None,
        "stop_rate": len(stopped) / n if n else None,
        "stop_success_rate": len(stop_ok) / n if n else None,
        "stop_precision": len(stop_ok) / len(stopped) if stopped else None,
        "success_tasks": len(succ), "failure_tasks": n - len(succ),
        "max_step_tasks": sum(1 for r in recs if r.get("stop_reason") == "step_cap"),
        "early_stop_tasks": sum(1 for r in recs if r.get("stopped")
                                and r.get("stop_dist") is not None
                                and r["stop_dist"] > STOP_RADIUS),
        "collisions": sum(1 for r in recs if r.get("collision")),
        "out_of_bounds": sum(1 for r in recs if r.get("out_of_bounds")),
        "unreached_tasks": sum(1 for r in recs if not r.get("osr")),
        "exceptions": sum(1 for r in recs
                          if r.get("execution_status") == "exception"),
        "failure_taxonomy": dict(failures),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "val_seen_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    examples = {k: [r["episode_id"] for r in recs if r.get("failure") == k][:20]
                for k in failures}
    (out / "failure_analysis.json").write_text(json.dumps({
        "counts": dict(failures), "examples": examples,
        "collisions": [r["episode_id"] for r in recs if r.get("collision")][:20],
        "out_of_bounds": [r["episode_id"] for r in recs if r.get("out_of_bounds")][:20],
        "max_steps": [r["episode_id"] for r in recs
                      if r.get("stop_reason") == "step_cap"][:20],
        "exceptions": [(r["episode_id"], r.get("exception")) for r in recs
                       if r.get("execution_status") == "exception"][:20],
    }, indent=2) + "\n", encoding="utf-8")
    fields = ["episode_id", "split", "map", "length_group", "cohort",
              "success", "osr", "spl", "ne_m", "ndtw", "steps", "stopped",
              "stop_step", "stop_dist", "stop_reason", "collision",
              "out_of_bounds", "failure", "path_len_m", "l_star_m",
              "candidate_status", "candidate_count", "execution_status",
              "exception"]
    with (out / "val_seen_episode_results.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in fields})
    print(json.dumps({k: summary[k] for k in
                      ["completed_episodes", "sr", "osr", "spl", "ne_m",
                       "ndtw", "mean_steps", "stop_success_rate",
                       "failure_taxonomy"]}, indent=1), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", required=True,
                        choices=["small", "medium", "ts", "tm", "large"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--task-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--step-cap", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha_path = manifest_path.with_name(manifest_path.name + ".sha256")
    if not sha_path.exists():
        sha_path = manifest_path.with_name(manifest_path.stem + ".sha256")
    manifest_sha = sha_path.read_text(encoding="utf-8").strip()
    ckpt_sha = sha256(Path(args.ckpt))

    out = Path(args.out_dir) if args.out_dir else (
        HERE / "results" / NAMES[args.model] / "closed_loop")
    out.mkdir(parents=True, exist_ok=True)
    tasks = manifest["tasks"]
    if args.task_file:
        wanted = {x.strip() for x in Path(args.task_file).read_text(
            encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
        tasks = [t for t in tasks if t["episode_id"] in wanted]
    refs = load_teacher_refs()
    done = load_journal(out)
    pending = [t for t in tasks if t["episode_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"tasks {len(tasks)} | done {len(done)} | pending {len(pending)}", flush=True)
    if not pending:
        write_summaries(out, load_journal(out), len(tasks), refs, tasks,
                        args.ckpt, manifest_sha, ckpt_sha, args.step_cap)
        return 0
    atomic_json(out / "eval_config.json", {
        "ckpt": str(Path(args.ckpt).resolve()), "ckpt_sha": ckpt_sha,
        "manifest_sha": manifest_sha, "step_cap": args.step_cap,
        "model": args.model, "protocol": "corrected-history-pose-emptycand",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    device = args.device
    model = build_model(args.model, device)
    controller = STAGE14_CONTROLLERS[args.model]
    st = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(st["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    enc4 = FrozenEncoder(device)
    matcher_enc = matching.Encoder(device)
    rasters, cands = RasterStore(), CandidateStore()

    episodes = out / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    journal = out / "journal.part00.jsonl"
    started = time.time()
    n = 0
    with journal.open("a", encoding="utf-8") as jf:
        for i, task in enumerate(pending, 1):
            name = task["map"]
            eid = task["episode_id"]
            rec_base = {"episode_id": eid, "split": task["split"],
                        "instruction_id": task.get("instruction_id", ""),
                        "block_id": name, "index": int(task["index"]),
                        "map": name, "length_group": task["length_group"],
                        "cohort": task["cohort"], "controller": controller,
                        "start": [float(task["start_x"]), float(task["start_y"]),
                                  float(task["start_yaw"])],
                        "target": [float(task["target_x"]), float(task["target_y"])],
                        "step_cap": int(args.step_cap),
                        "manifest_sha": manifest_sha, "ckpt_sha": ckpt_sha}
            try:
                raster = rasters.get(name)
                lib = cands.get(name)
                n_cand = int(len(lib.positions))
                rec_base["candidate_count"] = n_cand
                rec_base["candidate_status"] = "ok" if n_cand else "empty"
                info = obstacle_for(name)
                pos = np.asarray([rec_base["start"][0], rec_base["start"][1]],
                                 np.float64)
                yaw = float(rec_base["start"][2])
                target = np.asarray(rec_base["target"], np.float64)
                goal_img = matching.prepare_goal(raster, float(target[0]),
                                                 float(target[1]))
                goal_l2 = matcher_enc.features(goal_img[None], ("layer2",))["layer2"][0]
                gdesc = goal_l2[:, 13:15, 13:15].mean(axis=(1, 2))
                gdesc = gdesc / np.maximum(np.linalg.norm(gdesc, keepdims=True), 1e-8)
                empty_cands = n_cand == 0
                belief = None
                if not empty_cands:
                    tvec = enc4.embed([raster.north_crop(float(target[0]),
                                                         float(target[1]),
                                                         TARGET_HALF_M)])[0] \
                        .astype(np.float32)
                    sims = lib.embeddings @ tvec
                    belief = ContinuousBelief(lib.positions.astype(np.float64),
                                              sims.astype(np.float64), 0.1, k=8,
                                              sigma_prior_m=12.0)
                hist_vis: deque = deque(maxlen=HISTORY_LEN)
                hist_pose: deque = deque(maxlen=HISTORY_LEN)
                hist_act: deque = deque(maxlen=HISTORY_LEN)

                steps = []
                stopped = collision = out_of_bounds = False
                stop_reason = "step_cap"
                stop_dist = None
                min_dist = float(np.linalg.norm(pos - target))
                min_step = 0
                for step in range(args.step_cap):
                    dist = float(np.linalg.norm(pos - target))
                    if dist < min_dist:
                        min_dist, min_step = dist, step
                    if empty_cands:
                        belief_feat = EMPTY_BELIEF.copy()
                        match_feat = EMPTY_MATCH.copy()
                        est = np.asarray([pos[0], pos[1]], np.float64)
                        dist_est = 0.0
                        score = 0.0
                        accept = False
                    else:
                        obs = matching.prepare_observation(raster, pos[0], pos[1], yaw)
                        of = matcher_enc.features(obs[None], ("layer2",))["layer2"][0]
                        m = subpixel_match(of, goal_l2, 2)
                        dx, dy = matching.pixel_to_offset(m["col"], m["row"])
                        z = np.array([pos[0] + dx, pos[1] + dy])
                        score, margin = m["score"], m["margin"]
                        agree = (belief.last_accept is not None
                                 and float(np.linalg.norm(z - belief.last_accept))
                                 <= AGREE_M)
                        conf_ok = score >= TAU
                        accept = conf_ok and agree
                        if conf_ok:
                            belief.last_accept = z.copy()
                        if accept:
                            belief.update(z, CAL["sigma_vis_m"], BEL["best"]["eta"],
                                          BEL["best"]["rho"], step)
                        est = belief.estimate()
                        unc = float(math.sqrt(np.mean(belief.vars)))
                        since = float(step - belief.last_update)
                        ent = float(belief.entropy())
                        if not math.isfinite(ent):
                            ent = 0.0
                        belief_feat = belief_features(est, pos, yaw, score, unc,
                                                      ent, since, dx, dy)
                        dist_est = float(np.linalg.norm(est - pos))
                        match_feat = np.asarray([dx, dy, score, margin,
                                                 m["sharpness"]], np.float32)
                    # corrected history: slot 0 = most recent, relative pose
                    vis, pose_h, act_h, hmask = history_inputs(
                        hist_vis, hist_pose, hist_act, pos, yaw)
                    cur_img = raster.heading_crop(float(pos[0]), float(pos[1]),
                                                  float(yaw))
                    batch = {
                        "current": None,
                        "target_emb": torch.from_numpy(
                            gdesc[None].copy()).to(device),
                        "history_vis": torch.from_numpy(vis).to(device),
                        "history_pose": torch.from_numpy(pose_h).to(device),
                        "history_action": torch.from_numpy(act_h).to(device),
                        "history_mask": torch.from_numpy(hmask).to(device),
                        "belief": torch.from_numpy(
                            belief_feat[None].copy()).to(device),
                        "match": torch.from_numpy(
                            match_feat[None].copy()).to(device),
                    }
                    if args.model in GEOM_MODELS:
                        batch["landmarks"] = torch.from_numpy(
                            landmark_inputs(belief, empty_cands)).to(device)
                        batch["states"] = torch.from_numpy(
                            np.asarray([[pos[0], pos[1], yaw]],
                                       np.float32)).to(device)
                    # V5 修正 5b：当前帧单次前向编码（旧实现每步编码两次）
                    cur_img_np = np.asarray(cur_img, np.uint8)
                    with torch.no_grad():
                        cur = enc4(np.asarray([cur_img_np]))
                        batch["current"] = cur
                        logits = model(batch)
                    action = int(logits.argmax(-1).item())
                    hist_vis.append(
                        pool_history_from_current(cur)[0].astype(np.float32))
                    hist_pose.append((float(pos[0]), float(pos[1]), float(yaw)))
                    hist_act.append(action)
                    steps.append({"step": step, "x": float(pos[0]),
                                  "y": float(pos[1]), "yaw": float(yaw),
                                  "action": int(action),
                                  "estimate": [float(est[0]), float(est[1])],
                                  "est_distance_m": dist_est,
                                  "visual_score": float(score),
                                  "accept": bool(accept),
                                  "true_dist_m": dist})
                    if action == 0:
                        stopped = True
                        stop_dist = dist
                        stop_reason = "model_stop"
                        break
                    if action == 1:
                        nxt = pos + FORWARD_M * np.asarray(
                            [math.cos(yaw), math.sin(yaw)])
                        status = segment_status(info, float(pos[0]), float(pos[1]),
                                                float(nxt[0]), float(nxt[1]))
                        if status == "collision":
                            collision = True
                            stop_reason = "collision"
                            break
                        if status == "out_of_bounds":
                            out_of_bounds = True
                            stop_reason = "out_of_bounds"
                            break
                        pos = nxt
                    elif action == 2:
                        yaw = (yaw + TURN_RAD + math.pi) % (2 * math.pi) - math.pi
                    elif action == 3:
                        yaw = (yaw - TURN_RAD + math.pi) % (2 * math.pi) - math.pi
                final_dist = float(np.linalg.norm(pos - target))
                osr = bool(min_dist <= STOP_RADIUS)
                success = bool(stopped and stop_dist is not None
                               and stop_dist <= STOP_RADIUS
                               and final_dist <= STOP_RADIUS)
                path_len = FORWARD_M * sum(1 for s in steps if s["action"] == 1)
                l_star = float(np.linalg.norm(np.asarray(rec_base["start"][:2])
                                              - target))
                spl = float(success * min(1.0, l_star / path_len)) if path_len > 0 \
                    else float(success)
                ref_path, used_teacher_ref = reference_for(task, refs)
                ndtw = _ndtw(executed_path(
                    steps, rec_base["start"],
                    [float(pos[0]), float(pos[1]), float(yaw)]), ref_path)
                rec = {**rec_base,
                       "steps": len(steps),
                       "stopped": stopped,
                       "stop_step": (len(steps) - 1) if stopped else None,
                       "stop_reason": stop_reason,
                       "stop_dist": stop_dist,
                       "final_pos": [float(pos[0]), float(pos[1]), float(yaw)],
                       "final_dist": final_dist, "min_dist": min_dist,
                       "min_dist_step": int(min_step), "osr": osr,
                       "success": success, "collision": collision,
                       "out_of_bounds": out_of_bounds,
                       "path_len_m": path_len, "l_star_m": l_star,
                       "spl": spl, "ne_m": final_dist, "ndtw": ndtw,
                       "ndtw_teacher_ref": used_teacher_ref,
                       "oracle_success": osr,
                       "action_counts": {CLASSES[a]: int(sum(
                           1 for s in steps if s["action"] == a))
                           for a in range(4)},
                       "trajectory": steps,
                       "execution_status": "ok", "exception": "",
                       "seconds": round(time.time() - started, 3)}
                rec["failure"] = classify(rec)
            except Exception as e:  # noqa: BLE001 -- record, never abort a run
                rec = {**rec_base, "success": False, "osr": False,
                       "final_dist": None, "min_dist": None, "steps": 0,
                       "stopped": False, "stop_reason": "exception",
                       "stop_dist": None, "path_len_m": 0.0, "spl": 0.0,
                       "ne_m": None, "oracle_success": False,
                       "failure": "exception",
                       "execution_status": "exception",
                       "exception": f"{type(e).__name__}: {e}"[:300],
                       "trajectory": [],
                       "seconds": round(time.time() - started, 3)}
            atomic_json(episodes / f"{rec['episode_id'].replace(':', '_')}.json", rec)
            jf.write(json.dumps({k: v for k, v in rec.items()
                                 if k != "trajectory"}) + "\n")
            jf.flush()
            n += 1
            if i % 25 == 0 or i == len(pending):
                rate = i / max(time.time() - started, 1e-9)
                print(f"{i}/{len(pending)} | {rate:.2f} ep/s | elapsed "
                      f"{time.time() - started:.0f}s | "
                      f"eta {(len(pending) - i) / rate:.0f}s", flush=True)
    merged = load_journal(out)
    tmp = out / "journal.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        for eid in sorted(merged):
            fh.write(json.dumps(merged[eid]) + "\n")
    os.replace(tmp, out / "journal.jsonl")
    (out / "journal.part00.jsonl").unlink(missing_ok=True)
    write_summaries(out, merged, len(tasks), refs, tasks, args.ckpt,
                    manifest_sha, ckpt_sha, args.step_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
