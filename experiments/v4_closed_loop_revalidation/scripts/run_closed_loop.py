#!/usr/bin/env python3
"""Corrected closed-loop evaluator for M-v4 / M-v5 (unified, fair protocol).

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
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
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
from model11b import SPATIAL_CH, SPATIAL_HW, TransformerM11  # noqa: E402
from model11c import TransformerM12  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402
from model10c import TransformerM  # noqa: E402
from model11_small import SmallMEC4  # noqa: E402
from model11_medium import MediumMEC4  # noqa: E402
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


MODEL_FACTORY = {
    "v4": (TransformerM11, "v4_teacher_only"),
    "v5": (TransformerM12, "v5_teacher_only_landmarks"),
    "v5fix": (TransformerM12Fix, "v5fix_teacher_only_landmarks"),
    "v3": (TransformerM, "v3_teacher_only"),
    "small": (SmallMEC4, "small_meanpool_mlp"),
    "medium": (MediumMEC4, "medium_goal_cross_attention"),
}
# 需要 landmarks/states 几何输入（belief 混合地标）的模型
GEOM_MODELS = ("v5", "v5fix", "small", "medium")


def build_model(name: str, device: str):
    """按模型名构造 eval 模式模型；未知名拒绝。"""
    if name not in MODEL_FACTORY:
        raise SystemExit(f"unknown --model {name} (choices: "
                         f"{sorted(MODEL_FACTORY)})")
    cls, _ = MODEL_FACTORY[name]
    model = cls().to(device).eval()
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", required=True,
                        choices=["v4", "v5", "v5fix", "v3", "small", "medium"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
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

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = manifest["tasks"]
    if args.task_file:
        wanted = {x.strip() for x in Path(args.task_file).read_text(
            encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
        tasks = [t for t in tasks if t["episode_id"] in wanted]
    done = load_journal(out)
    pending = [t for t in tasks if t["episode_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"tasks {len(tasks)} | done {len(done)} | pending {len(pending)}", flush=True)
    if not pending:
        return 0
    atomic_json(out / "eval_config.json", {
        "ckpt": str(Path(args.ckpt).resolve()), "ckpt_sha": ckpt_sha,
        "manifest_sha": manifest_sha, "step_cap": args.step_cap,
        "model": args.model, "protocol": "corrected-history-pose-emptycand",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    device = args.device
    model = build_model(args.model, device)
    controller = MODEL_FACTORY[args.model][1]
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
                       "spl": spl, "ne_m": final_dist,
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
    succ = [r for r in merged.values() if r["success"]]
    print(f"finished {n} episodes | SR={len(succ)}/{len(merged)} = "
          f"{len(succ)/max(len(merged),1):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
