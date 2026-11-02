#!/usr/bin/env python3
"""11D: closed-loop evaluation of Transformer-M v4 / v5.

Identical harness to run_10e.py (same C1/09C live loop, same geometry /
obstacles / step-cap / success rules, same manifests), with the 11B models:

* --model v4: TransformerM11 (12-d deduplicated state, history time emb)
* --model v5: TransformerM12 (v4 + multi-landmark geometry tokens; the live
  loop additionally feeds the current top-K belief mixture as landmarks)

Success = the model stops by itself AND the stop position is within 20 m of
the true target (evaluator-only knowledge).  Resumable journal keyed by
manifest SHA + model checkpoint SHA.
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
REPO = HERE.parents[3]
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
from model11b import TransformerM11  # noqa: E402
from model11c import TransformerM12  # noqa: E402

STAGE06 = REPO / "experiments/06_closed_loop_success_rate/results/seed42"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="11B checkpoint (best.pt)")
    parser.add_argument("--model", required=True, choices=["v4", "v5"])
    parser.add_argument("--manifest", default=str(STAGE06 / "manifest.json"))
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
        HERE.parent / "results" / f"11d_loop_{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    tasks = manifest["tasks"]
    if args.task_file:
        wanted = {x.strip() for x in Path(args.task_file).read_text(
            encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
        tasks = [t for t in tasks if t["episode_id"] in wanted]
    done = {}
    for eid, rec in load_journal(out).items():
        if (out / "episodes" / f"{eid.replace(':', '_')}.json").exists():
            done[eid] = rec
    pending = [t for t in tasks if t["episode_id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"tasks {len(tasks)} | done {len(done)} | pending {len(pending)}", flush=True)
    if not pending:
        return 0
    atomic_json(out / "eval_config.json", {
        "ckpt": str(Path(args.ckpt).resolve()), "ckpt_sha": ckpt_sha,
        "manifest_sha": manifest_sha, "step_cap": args.step_cap,
        "model": args.model,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    device = args.device
    if args.model == "v4":
        model = TransformerM11().to(device).eval()
        controller = "11d_transformer_m_v4"
    else:
        model = TransformerM12().to(device).eval()
        controller = "11d_transformer_m_v5"
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
            raster = rasters.get(name)
            lib = cands.get(name)
            if len(lib.positions) == 0:
                # frozen C1 module has no candidate library for this map
                # (e.g. cambridge_block_32): unevaluable, recorded as such
                rec = {"episode_id": task["episode_id"], "split": task["split"],
                       "index": int(task["index"]), "map": name,
                       "length_group": task["length_group"],
                       "cohort": task["cohort"], "controller": controller,
                       "success": False, "failure": "c1_no_candidates",
                       "steps": 0, "stopped": False, "stop_dist": None,
                       "min_dist": None, "final_dist": None,
                       "manifest_sha": manifest_sha, "ckpt_sha": ckpt_sha,
                       "seconds": round(time.time() - started, 3)}
                atomic_json(episodes / f"{rec['episode_id'].replace(':', '_')}.json", rec)
                jf.write(json.dumps(rec) + "\n")
                jf.flush()
                n += 1
                continue
            info = obstacle_for(name)
            pos = np.asarray([float(task["start_x"]), float(task["start_y"])],
                             np.float64)
            yaw = float(task["start_yaw"])
            target = np.asarray([float(task["target_x"]), float(task["target_y"])],
                                np.float64)
            goal_img = matching.prepare_goal(raster, float(target[0]), float(target[1]))
            goal_l2 = matcher_enc.features(goal_img[None], ("layer2",))["layer2"][0]
            gdesc = goal_l2[:, 13:15, 13:15].mean(axis=(1, 2))
            gdesc = gdesc / np.maximum(np.linalg.norm(gdesc, keepdims=True), 1e-8)
            tvec = enc4.embed([raster.north_crop(float(target[0]), float(target[1]),
                                                 TARGET_HALF_M)])[0].astype(np.float32)
            sims = lib.embeddings @ tvec
            belief = ContinuousBelief(lib.positions.astype(np.float64),
                                      sims.astype(np.float64), 0.1, k=8,
                                      sigma_prior_m=12.0)
            hist_vis: deque = deque(maxlen=HISTORY_LEN)
            hist_pose: deque = deque(maxlen=HISTORY_LEN)
            hist_act: deque = deque(maxlen=HISTORY_LEN)

            steps = []
            stopped = collision = out_of_bounds = False
            stop_dist = None
            min_dist = float(np.linalg.norm(pos - target))
            min_step = 0
            for step in range(args.step_cap):
                dist = float(np.linalg.norm(pos - target))
                if dist < min_dist:
                    min_dist, min_step = dist, step
                obs = matching.prepare_observation(raster, pos[0], pos[1], yaw)
                of = matcher_enc.features(obs[None], ("layer2",))["layer2"][0]
                m = subpixel_match(of, goal_l2, 2)
                dx, dy = matching.pixel_to_offset(m["col"], m["row"])
                z = np.array([pos[0] + dx, pos[1] + dy])
                score, margin = m["score"], m["margin"]
                agree = (belief.last_accept is not None
                         and float(np.linalg.norm(z - belief.last_accept)) <= AGREE_M)
                conf_ok = score >= TAU
                accept = conf_ok and agree
                if conf_ok:
                    belief.last_accept = z.copy()
                if accept:
                    belief.update(z, CAL["sigma_vis_m"], BEL["best"]["eta"],
                                  BEL["best"]["rho"], step)
                est = belief.estimate()
                vec = est - pos
                fwd = float(vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
                lat = float(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw))
                dist_est = float(np.linalg.norm(vec))
                bearing = math.atan2(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw),
                                     vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
                unc = float(math.sqrt(np.mean(belief.vars)))
                since = float(step - belief.last_update)
                ent = float(belief.entropy())
                if not math.isfinite(ent):
                    ent = 0.0
                belief_feat = np.asarray(
                    [fwd, lat, dist_est, bearing, score, unc, ent,
                     since, dx, dy], np.float32)
                match_feat = np.asarray([dx, dy, score, margin, m["sharpness"]],
                                        np.float32)
                vis = np.zeros((1, HISTORY_LEN, 2048), np.float32)
                pose = np.zeros((1, HISTORY_LEN, 3), np.float32)
                act = np.zeros((1, HISTORY_LEN, 4), np.float32)
                hmask = np.zeros((1, HISTORY_LEN), np.float32)
                for k in range(len(hist_vis)):
                    p = hist_pose[k]
                    vis[0, k] = hist_vis[k]
                    pose[0, k] = p
                    act[0, k, hist_act[k]] = 1.0
                    hmask[0, k] = 1.0
                cur_img = raster.heading_crop(float(pos[0]), float(pos[1]), float(yaw))
                batch = {
                    "current": None,
                    "target_emb": torch.from_numpy(gdesc[None].copy()).to(device),
                    "history_vis": torch.from_numpy(vis).to(device),
                    "history_pose": torch.from_numpy(pose).to(device),
                    "history_action": torch.from_numpy(act).to(device),
                    "history_mask": torch.from_numpy(hmask).to(device),
                    "belief": torch.from_numpy(belief_feat[None].copy()).to(device),
                    "match": torch.from_numpy(match_feat[None].copy()).to(device),
                }
                if args.model == "v5":
                    lm = np.zeros((1, N_LM, 3), np.float32)
                    n_k = int(belief.means.shape[0])
                    lm[0, :n_k, :2] = belief.means.astype(np.float32)
                    lm[0, :n_k, 2] = belief.weights.astype(np.float32)
                    batch["landmarks"] = torch.from_numpy(lm).to(device)
                    batch["states"] = torch.from_numpy(
                        np.asarray([[pos[0], pos[1], yaw]], np.float32)).to(device)
                with torch.no_grad():
                    cur = enc4(np.asarray([cur_img], np.uint8))
                    batch["current"] = cur
                    logits = model(batch)
                action = int(logits.argmax(-1).item())
                cur_img_np = np.asarray(cur_img, np.uint8)
                with torch.no_grad():
                    maps = enc4.spatial_maps([cur_img_np]).transpose(0, 2, 3, 1)
                    pooled = maps.mean(axis=(1, 2))
                    pooled = pooled / np.maximum(
                        np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
                hist_vis.append(pooled[0].astype(np.float32))
                hist_pose.append((float(pos[0]), float(pos[1]), float(yaw)))
                hist_act.append(action)
                steps.append({"step": step, "x": float(pos[0]), "y": float(pos[1]),
                              "yaw": float(yaw), "action": int(action),
                              "estimate": [float(est[0]), float(est[1])],
                              "est_distance_m": dist_est,
                              "visual_score": float(score),
                              "accept": bool(accept),
                              "true_dist_m": dist})
                if action == 0:
                    stopped = True
                    stop_dist = dist
                    break
                if action == 1:
                    nxt = pos + FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)])
                    status = segment_status(info, float(pos[0]), float(pos[1]),
                                            float(nxt[0]), float(nxt[1]))
                    if status == "collision":
                        collision = True
                        break
                    if status == "out_of_bounds":
                        out_of_bounds = True
                        break
                    pos = nxt
                elif action == 2:
                    yaw = (yaw + TURN_RAD + math.pi) % (2 * math.pi) - math.pi
                elif action == 3:
                    yaw = (yaw - TURN_RAD + math.pi) % (2 * math.pi) - math.pi
            final_dist = float(np.linalg.norm(pos - target))
            osr = bool(min_dist <= STOP_RADIUS)
            success = bool(stopped and stop_dist is not None
                           and stop_dist <= STOP_RADIUS and final_dist <= STOP_RADIUS)
            rec = {
                "episode_id": task["episode_id"], "split": task["split"],
                "index": int(task["index"]), "map": name,
                "length_group": task["length_group"], "cohort": task["cohort"],
                "controller": controller,
                "start": [float(task["start_x"]), float(task["start_y"]),
                          float(task["start_yaw"])],
                "target": [float(task["target_x"]), float(task["target_y"])],
                "step_cap": int(args.step_cap), "steps": len(steps),
                "stopped": stopped, "stop_step": (len(steps) - 1) if stopped else None,
                "stop_dist": stop_dist,
                "final_pos": [float(pos[0]), float(pos[1]), float(yaw)],
                "final_dist": final_dist, "min_dist": min_dist,
                "min_dist_step": int(min_step), "osr": osr, "success": success,
                "collision": collision, "out_of_bounds": out_of_bounds,
                "path_len_m": FORWARD_M * sum(1 for s in steps if s["action"] == 1),
                "action_counts": {CLASSES[a]: int(sum(1 for s in steps
                                                      if s["action"] == a))
                                  for a in range(4)},
                "trajectory": steps,
                "seconds": round(time.time() - started, 3),
                "manifest_sha": manifest_sha, "ckpt_sha": ckpt_sha,
            }
            rec["failure"] = classify(rec)
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
