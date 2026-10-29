#!/usr/bin/env python3
"""Closed-loop SR evaluation for the fixed-50 m Transformer-M.

Success requires the model to emit STOP, stop within 20 m of the target, and end
within 20 m.  The evaluator uses the true target for metrics only; the model
never receives it.  The step cap is 500 (99.9% of successful teacher rollouts).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE.parent / "03_fixed50m_four_action_teacher/scripts"))
sys.path.insert(0, str(HERE.parent / "04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(HERE.parent / "05_transformer_m_training/scripts"))

from four_action_teacher import convert_episode, move  # noqa: E402
from pipeline import (  # noqa: E402
    DATA, CandidateLibrary, FrozenResNet, MapRaster, belief_features,
    load_ground, target_embedding,
)
from model import TransformerM  # noqa: E402

SPLITS = DATA / "sbfnav-data/original_citynav"
CANDIDATES = HERE.parent / "04_visual_goal_global_map_pipeline/cache/candidates"
RUNS = HERE.parent / "05_transformer_m_training/runs"
STEP_CAP = 500
BOUNDS = (44, 77)  # training teacher-step terciles
ACTION_NAMES = {0: "Stop", 1: "Forward", 2: "TurnLeft", 3: "TurnRight"}


def bin_name(steps: int) -> str:
    if steps <= BOUNDS[0]:
        return "short"
    if steps <= BOUNDS[1]:
        return "medium"
    return "long"


def select_episodes(split: str, per_bin: int, seed: int = 0) -> list[dict]:
    path = HERE / "results" / f"selection_{split}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["episodes"]
    records = json.loads((SPLITS / f"citynav_{split}.json").read_text(encoding="utf-8"))
    ground = load_ground()
    by_bin: dict[str, list[dict]] = {"short": [], "medium": [], "long": []}
    for idx, record in enumerate(records):
        name = f"{record['area']}_block_{record['block']}"
        result = convert_episode(record, ground.get(name))
        if not result.get("success"):
            continue
        steps = int(result["teacher_steps"])
        by_bin[bin_name(steps)].append({
            "key": f"{idx}|{name}", "index": idx, "map": name,
            "object_ids": record["object_ids"], "ann_ids": record["ann_ids"],
            "teacher_steps": steps, "bin": bin_name(steps),
            "target_xy": result["target_xy"], "start_xy": result["states"][0][:2],
            "start_yaw": result["start_yaw"],
        })
    rng = np.random.default_rng(seed)
    chosen = []
    for b in ("short", "medium", "long"):
        rows = by_bin[b]
        rng.shuffle(rows)
        chosen.extend(rows[:per_bin])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"split": split, "bounds": BOUNDS, "episodes": chosen},
                               indent=2) + "\n", encoding="utf-8")
    return chosen


def load_model(seed: int, device: str):
    run = RUNS / f"seed{seed}"
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    ckpt = run / "checkpoint.pt" if (run / "checkpoint.pt").exists() else run / "best.pt"
    model = TransformerM(config["d_model"], config["layers"], config["heads"])
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), ckpt.name


def build_history(pooled, states, actions, pos, yaw):
    vis = np.zeros((16, 2048), np.float32)
    pose = np.zeros((16, 3), np.float32)
    act = np.zeros((16,), np.int64)
    mask = np.zeros((16,), np.float32)
    n = len(pooled)
    cos, sin = math.cos(yaw), math.sin(yaw)
    for k in range(1, 17):
        j = n - k
        if j < 0:
            break
        px, py, pyaw = states[j]
        dx, dy = px - pos[0], py - pos[1]
        pose[k - 1] = [dx * cos + dy * sin, -dx * sin + dy * cos, (pyaw - yaw + math.pi) % (2 * math.pi) - math.pi]
        act[k - 1] = actions[j]
        vis[k - 1] = pooled[j]
        mask[k - 1] = 1.0
    return vis, pose, act, mask


def run_episode(model, encoder, raster, library, target_emb, episode, device: str):
    pos = np.asarray(episode["start_xy"], np.float64)
    yaw = float(episode["start_yaw"])
    target = np.asarray(episode["target_xy"], np.float64)
    pooled_hist, state_hist, action_hist = [], [], []
    actions, dists = [], []
    stopped = False
    started = time.time()
    with torch.no_grad():
        for step in range(STEP_CAP):
            dist = float(np.linalg.norm(pos - target))
            dists.append(dist)
            crop = raster.heading_crop(float(pos[0]), float(pos[1]), yaw)
            spatial = encoder.spatial([crop])  # (1,2048,7,7)
            spatial16 = spatial.transpose(0, 2, 3, 1).astype(np.float16)
            pooled = spatial16.astype(np.float32).mean(axis=(1, 2))
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
            vis, pose, act, mask = build_history(pooled_hist, state_hist, action_hist, pos, yaw)
            belief, info = belief_features(library, target_emb, np.asarray([pos[0], pos[1]]), yaw)
            cand_emb = library.embeddings[info["top_indices"]]
            delta = info["positions"] - pos[None, :]
            cos, sin = math.cos(yaw), math.sin(yaw)
            rel = np.stack([delta[:, 0] * cos + delta[:, 1] * sin,
                            -delta[:, 0] * sin + delta[:, 1] * cos,
                            np.linalg.norm(delta, axis=1)], axis=1)
            cand_meta = np.concatenate([rel, info["sims"][:, None]], axis=1)
            batch = {
                "current": torch.from_numpy(spatial16.astype(np.float32)).to(device),
                "target_emb": torch.from_numpy(target_emb.astype(np.float32))[None].to(device),
                "history_vis": torch.from_numpy(vis)[None].to(device),
                "history_pose": torch.from_numpy(pose)[None].to(device),
                "history_action": torch.from_numpy(act)[None].to(device),
                "history_mask": torch.from_numpy(mask)[None].to(device),
                "candidate_emb": torch.from_numpy(cand_emb.astype(np.float32))[None].to(device),
                "candidate_meta": torch.from_numpy(cand_meta.astype(np.float32))[None].to(device),
                "belief": torch.from_numpy(belief.astype(np.float32))[None].to(device),
            }
            action = int(model(batch).argmax(-1).item())
            actions.append(action)
            if action == 0:
                stopped = True
                break
            pooled_hist.append(pooled[0])
            state_hist.append((float(pos[0]), float(pos[1]), float(yaw)))
            action_hist.append(action)
            pos, yaw = move(pos, yaw, action)
    final_dist = float(np.linalg.norm(pos - target))
    min_dist = float(min(dists + [final_dist]))
    stop_dist = float(dists[-1]) if stopped else None
    counts = Counter(ACTION_NAMES[a] for a in actions)
    return {
        "key": f"{episode['index']}|{episode['map']}",
        "map": episode["map"], "bin": episode["bin"], "teacher_steps": episode["teacher_steps"],
        "steps": len(actions), "stopped": stopped,
        "success": bool(stopped and stop_dist is not None and stop_dist <= 20.0 and final_dist <= 20.0),
        "osr": bool(min_dist <= 20.0),
        "early_stop": bool(stopped and stop_dist is not None and stop_dist > 20.0),
        "cap_fail": bool(not stopped and len(actions) >= STEP_CAP),
        "stop_dist": stop_dist, "final_dist": final_dist, "min_dist": min_dist,
        "action_counts": dict(counts), "seconds": round(time.time() - started, 2),
    }


def summarize(rows: list[dict]) -> dict:
    def agg(rs):
        n = len(rs)
        if not n:
            return {"n": 0}
        return {
            "n": n,
            "sr": float(np.mean([r["success"] for r in rs])),
            "osr": float(np.mean([r["osr"] for r in rs])),
            "stop_rate": float(np.mean([r["stopped"] for r in rs])),
            "early_stop_rate": float(np.mean([r["early_stop"] for r in rs])),
            "non_stop_rate": float(np.mean([not r["stopped"] for r in rs])),
            "cap_fail_rate": float(np.mean([r["cap_fail"] for r in rs])),
            "mean_steps": float(np.mean([r["steps"] for r in rs])),
            "mean_stop_dist": float(np.mean([r["stop_dist"] for r in rs if r["stop_dist"] is not None])) if any(r["stop_dist"] is not None for r in rs) else None,
            "mean_final_dist": float(np.mean([r["final_dist"] for r in rs])),
            "mean_min_dist": float(np.mean([r["min_dist"] for r in rs])),
            "actions": dict(sum((Counter(r["action_counts"]) for r in rs), Counter())),
        }
    out = {"all": agg(rows)}
    for b in ("short", "medium", "long"):
        out[b] = agg([r for r in rows if r["bin"] == b])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val_seen", "val_unseen"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per-bin", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    episodes = select_episodes(args.split, args.per_bin)
    if args.limit:
        episodes = episodes[: args.limit]
    model, ckpt = load_model(args.seed, args.device)
    encoder = FrozenResNet(args.device)
    journal = HERE / "results" / f"closed_loop_{args.split}_seed{args.seed}.jsonl"
    done = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    print(f"{args.split} seed{args.seed}: {len(episodes)} episodes, ckpt={ckpt}, "
          f"cap={STEP_CAP}", flush=True)
    started = time.time()
    rasters, libraries, targets = {}, {}, {}
    with journal.open("a", encoding="utf-8") as fh:
        for i, episode in enumerate(episodes, 1):
            if episode["key"] in done:
                continue
            name = episode["map"]
            if name not in rasters:
                rasters[name] = MapRaster.load(name)
                libraries[name] = CandidateLibrary.load(name, CANDIDATES / f"{name}.npz")
            raster, library = rasters[name], libraries[name]
            if name not in targets:
                targets[name] = {}
            key = episode["key"]
            if key not in targets[name]:
                targets[name][key] = target_embedding(raster, np.asarray(episode["target_xy"]), encoder)
            row = run_episode(model, encoder, raster, library, targets[name][key], episode, args.device)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0 or args.timing:
                elapsed = time.time() - started
                print(f"  {i}/{len(episodes)}  {elapsed:.0f}s  last={row['steps']} steps "
                      f"{row['seconds']}s", flush=True)
            if args.timing and i >= 3:
                break
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {"split": args.split, "seed": args.seed, "step_cap": STEP_CAP,
               "bounds": BOUNDS, "checkpoint": ckpt, "episodes": len(rows),
               "metrics": summarize(rows)}
    out = HERE / "results" / f"summary_{args.split}_seed{args.seed}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"]["all"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
