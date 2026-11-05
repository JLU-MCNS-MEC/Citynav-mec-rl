#!/usr/bin/env python3
"""阶段 7：三种 10-MEC 配置 × Baseline 关联策略的闭环评测。

策略集 = mec-v8.2 `association_strategies.py`（commit d988a9a）实际实现的：
  random / nearest / max_rate / min_latency / best_quality
（v8.2 另有 oracle：需真实最优动作，属特权上界，不可部署，本轮不跑，报告注明）

关键点（公平性）：
- A(10×Small) 与 B(10×TM) 中所有节点托管同一模型 → 关联选择不改变动作
  轨迹（选择只影响时延）。因此 A/B 的策略级结果 = 复用阶段 1 已完成的
  small/v5fix 闭环轨迹（同一代码路径、同一协议）逐任务重放 + 逐步选路与时延
  重算。这不是近似，而是完全等价（模型输入与所选 MEC 无关，与 v8.2 语义一致：
  model type / compute capacity 对 MEC 输入不可见）。
- C(5×Small+5×TM) 中选路会改变该步执行的模型 → 必须真实逐步切换模型闭环。
- 模型输入契约与 run_closed_loop.py 完全一致（直接复用其 helper）。

用法:
  python run_association_loop.py --config C --strategy min_latency \
      --out-dir results/assoc_C_min_latency
  python run_association_loop.py --replay A --strategy nearest \
      --out-dir results/assoc_A_nearest   # A/B 重放（无需 GPU 模型）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
assert (REPO / "experiments").is_dir(), REPO
sys.path.insert(0, str(REPO / "experiments/v4_closed_loop_revalidation/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
sys.path.insert(0, str(REPO / "experiments/06_closed_loop_success_rate/scripts"))

import matching  # noqa: E402
from closed_loop import (CLASSES, FORWARD_M, STOP_RADIUS, TURN_RAD, classify,  # noqa: E402
                         obstacle_for, segment_status)
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from model11_small import SmallMEC4  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402
from run_closed_loop import (AGREE_M, BEL, CAL, HISTORY_LEN, TARGET_HALF_M,  # noqa: E402
                             TAU, belief_features, history_inputs,
                             landmark_inputs, load_journal, pool_history_from_current)
from v5fix_contract import EMPTY_BELIEF, EMPTY_MATCH  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"

# ---- 阶段 3/4 实测值（模型推理解析模型）----
FLOP = {"small": 4.5e6, "medium": 55.1e6, "transformer_m": 556.0e6}
CAP = 0.2e12                      # 人为建模假设
ENCODE_MS = 1.680
PAYLOAD_BITS = 49 * 2048 * 16
QUALITY = {"small": 0.3467, "transformer_m": 0.5360, "medium": 0.2398}

# 通信模型常量（mec-v8.2 config.py）
GBS_H, UAV_H, CARRIER, BW, TXP = 10.0, 80.0, 2.0e9, 5.0e6, 0.1
NOISE_PSD, NF, LOS_A, LOS_B, EXC_LOS, EXC_NLOS = -174.0, 9.0, 9.61, 0.16, 1.0, 20.0
COVERAGE_R = 500.0
C_LIGHT = 299_792_458.0


def dbm_to_watt(dbm):
    return 10.0 ** ((dbm - 30.0) / 10.0)


def uplink_rate_bps(h_m):
    h = max(float(h_m), 0.0)
    v = abs(UAV_H - GBS_H)
    d3 = max(math.sqrt(h * h + v * v), 1.0)
    elev = math.degrees(math.atan2(v, max(h, 1e-15)))
    p_los = 1.0 / (1.0 + LOS_A * math.exp(-LOS_B * (elev - LOS_A)))
    fsl = 20.0 * math.log10(4.0 * math.pi * CARRIER * d3 / C_LIGHT)
    pl = fsl + p_los * EXC_LOS + (1 - p_los) * EXC_NLOS
    noise = dbm_to_watt(NOISE_PSD + 10.0 * math.log10(BW) + NF)
    snr = TXP * 10.0 ** (-pl / 10.0) / max(noise, 1e-15)
    return max(BW * math.log2(1.0 + snr), 1e-15)


def node_metrics(node: dict, pos: np.ndarray) -> dict:
    d = float(np.linalg.norm(np.asarray([node["x"], node["y"]]) - pos))
    comm_ms = PAYLOAD_BITS / uplink_rate_bps(d) * 1e3
    inf_ms = FLOP[node["model_type"]] / node["compute_capacity_flops_per_s"] * 1e3
    return {"distance_m": d, "comm_ms": comm_ms, "inf_ms": inf_ms,
            "e2e_ms": ENCODE_MS + comm_ms + inf_ms}


def select_node(nodes: list[dict], pos: np.ndarray, strategy: str,
                rng: np.random.Generator) -> tuple[dict, dict]:
    avail = [n for n in nodes if np.linalg.norm(
        np.asarray([n["x"], n["y"]]) - pos) <= n["coverage_radius_m"]]
    if not avail:                       # 兜底：最近节点（阶段 2 实测最小 8 个可用）
        avail = [min(nodes, key=lambda n: np.linalg.norm(
            np.asarray([n["x"], n["y"]]) - pos))]
    if strategy == "random":
        node = avail[int(rng.integers(0, len(avail)))]
        return node, node_metrics(node, pos)
    if strategy == "nearest":
        node = min(avail, key=lambda n: np.linalg.norm(
            np.asarray([n["x"], n["y"]]) - pos))
        return node, node_metrics(node, pos)
    if strategy == "max_rate":
        node = max(avail, key=lambda n: uplink_rate_bps(
            np.linalg.norm(np.asarray([n["x"], n["y"]]) - pos)))
        return node, node_metrics(node, pos)
    if strategy == "min_latency":
        node = min(avail, key=lambda n: node_metrics(n, pos)["e2e_ms"])
        return node, node_metrics(node, pos)
    if strategy == "best_quality":
        node = max(avail, key=lambda n: (QUALITY[n["model_type"]],
                                         -node_metrics(n, pos)["e2e_ms"]))
        return node, node_metrics(node, pos)
    raise ValueError(f"unknown strategy {strategy}")


def load_config(config: str) -> dict:
    cfg = json.loads((RES / "deployment_configs.json").read_text())
    key = {"A": "A_10xSmall", "B": "B_10xTransformerM",
           "C": "C_5Small_5TM"}[config]
    nodes = cfg[key]
    by_map: dict[str, list[dict]] = {}
    for n in nodes:
        by_map.setdefault(n["map"], []).append(n)
    return by_map


def make_batch(pos, yaw, belief_feat, match_feat, gdesc, vis, pose_h, act_h,
               hmask, belief, empty_cands, device):
    batch = {
        "target_emb": torch.from_numpy(gdesc[None].copy()).to(device),
        "history_vis": torch.from_numpy(vis).to(device),
        "history_pose": torch.from_numpy(pose_h).to(device),
        "history_action": torch.from_numpy(act_h).to(device),
        "history_mask": torch.from_numpy(hmask).to(device),
        "belief": torch.from_numpy(belief_feat[None].copy()).to(device),
        "match": torch.from_numpy(match_feat[None].copy()).to(device),
        "landmarks": torch.from_numpy(
            landmark_inputs(belief, empty_cands)).to(device),
        "states": torch.from_numpy(np.asarray([[pos[0], pos[1], yaw]],
                                              np.float32)).to(device),
    }
    return batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["A", "B", "C"], required=True)
    ap.add_argument("--strategy", required=True,
                    choices=["random", "nearest", "max_rate", "min_latency",
                             "best_quality"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step-cap", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = RES / out
    out.mkdir(parents=True, exist_ok=True)
    episodes = out / "episodes"
    episodes.mkdir(exist_ok=True)

    mani = json.loads((REPO / "experiments/06_closed_loop_success_rate/results"
                       "/seed42/manifest.json").read_text())
    tasks = [t for t in mani["tasks"] if t["split"] == "val_seen"]
    if args.limit:
        tasks = tasks[: args.limit]
    done = load_journal(out) if (out / "journal.part00.jsonl").exists() else {}
    pending = [t for t in tasks if t["episode_id"] not in done]
    print(f"config={args.config} strategy={args.strategy} | tasks {len(tasks)} "
          f"| done {len(done)} | pending {len(pending)}", flush=True)
    if not pending:
        return 0

    by_map = load_config(args.config)
    rng = np.random.default_rng(42)
    device = args.device
    models: dict[str, torch.nn.Module] = {}
    ckpts = {"small": "runs_small/seed42/best.pt",
             "transformer_m": "runs_11c_v5fix/seed42/best.pt"}
    classes = {"small": SmallMEC4, "transformer_m": TransformerM12Fix}
    enc4 = FrozenEncoder(device)
    matcher_enc = matching.Encoder(device)
    rasters, cands = RasterStore(), CandidateStore()
    journal = out / "journal.part00.jsonl"
    started = time.time()
    n = 0
    with journal.open("a", encoding="utf-8") as jf:
        for i, task in enumerate(pending, 1):
            name = task["map"]
            nodes = by_map[name]
            rec_base = {"episode_id": task["episode_id"], "split": task["split"],
                        "block_id": name, "index": int(task["index"]),
                        "map": name, "length_group": task["length_group"],
                        "cohort": task["cohort"],
                        "controller": f"assoc_{args.config}_{args.strategy}",
                        "start": [float(task["start_x"]), float(task["start_y"]),
                                  float(task["start_yaw"])],
                        "target": [float(task["target_x"]),
                                   float(task["target_y"])],
                        "step_cap": args.step_cap, "config": args.config,
                        "strategy": args.strategy}
            try:
                raster = rasters.get(name)
                lib = cands.get(name)
                n_cand = int(len(lib.positions))
                info = obstacle_for(name)
                pos = np.asarray(rec_base["start"][:2], np.float64)
                yaw = float(rec_base["start"][2])
                target = np.asarray(rec_base["target"], np.float64)
                goal_img = matching.prepare_goal(raster, float(target[0]),
                                                 float(target[1]))
                goal_l2 = matcher_enc.features(goal_img[None],
                                               ("layer2",))["layer2"][0]
                gdesc = goal_l2[:, 13:15, 13:15].mean(axis=(1, 2))
                gdesc = gdesc / np.maximum(np.linalg.norm(gdesc, keepdims=True),
                                           1e-8)
                empty_cands = n_cand == 0
                belief = None
                if not empty_cands:
                    tvec = enc4.embed([raster.north_crop(
                        float(target[0]), float(target[1]), TARGET_HALF_M)]
                    )[0].astype(np.float32)
                    sims = lib.embeddings @ tvec
                    belief = ContinuousBelief(lib.positions.astype(np.float64),
                                              sims.astype(np.float64), 0.1,
                                              k=8, sigma_prior_m=12.0)
                hist_vis, hist_pose, hist_act = (deque(maxlen=HISTORY_LEN),
                                                 deque(maxlen=HISTORY_LEN),
                                                 deque(maxlen=HISTORY_LEN))
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
                        score = 0.0
                        accept = False
                    else:
                        obs = matching.prepare_observation(raster, pos[0],
                                                           pos[1], yaw)
                        of = matcher_enc.features(obs[None],
                                                  ("layer2",))["layer2"][0]
                        m = subpixel_match(of, goal_l2, 2)
                        dx, dy = matching.pixel_to_offset(m["col"], m["row"])
                        z = np.array([pos[0] + dx, pos[1] + dy])
                        score, margin = m["score"], m["margin"]
                        agree = (belief.last_accept is not None
                                 and float(np.linalg.norm(
                                     z - belief.last_accept)) <= AGREE_M)
                        conf_ok = score >= TAU
                        accept = conf_ok and agree
                        if conf_ok:
                            belief.last_accept = z.copy()
                        if accept:
                            belief.update(z, CAL["sigma_vis_m"],
                                          BEL["best"]["eta"],
                                          BEL["best"]["rho"], step)
                        est = belief.estimate()
                        unc = float(math.sqrt(np.mean(belief.vars)))
                        since = float(step - belief.last_update)
                        ent = float(belief.entropy())
                        if not math.isfinite(ent):
                            ent = 0.0
                        belief_feat = belief_features(est, pos, yaw, score, unc,
                                                      ent, since, dx, dy)
                        match_feat = np.asarray([dx, dy, score, margin,
                                                 m["sharpness"]], np.float32)
                    node, metrics = select_node(nodes, pos, args.strategy, rng)
                    vis, pose_h, act_h, hmask = history_inputs(
                        hist_vis, hist_pose, hist_act, pos, yaw)
                    cur_img = raster.heading_crop(float(pos[0]), float(pos[1]),
                                                  float(yaw))
                    batch = make_batch(pos, yaw, belief_feat, match_feat, gdesc,
                                       vis, pose_h, act_h, hmask, belief,
                                       empty_cands, device)
                    mtype = node["model_type"]
                    if mtype not in models:
                        model = classes[mtype]().to(device).eval()
                        st = torch.load(
                            REPO / "experiments/11_teacher_geometry_ablation"
                            "/11B_transformer_m_v4/results" / ckpts[mtype],
                            map_location=device, weights_only=False)
                        model.load_state_dict(st["model"])
                        for p in model.parameters():
                            p.requires_grad_(False)
                        models[mtype] = model
                    with torch.no_grad():
                        cur = enc4(np.asarray([cur_img], np.uint8))
                        batch["current"] = cur
                        logits = models[mtype](batch)
                    action = int(logits.argmax(-1).item())
                    hist_vis.append(
                        pool_history_from_current(cur)[0].astype(np.float32))
                    hist_pose.append((float(pos[0]), float(pos[1]), float(yaw)))
                    hist_act.append(action)
                    steps.append({"step": step, "x": float(pos[0]),
                                  "y": float(pos[1]), "yaw": float(yaw),
                                  "action": action,
                                  "gbs_id": node["gbs_id"],
                                  "model_type": mtype,
                                  "distance_m": metrics["distance_m"],
                                  "comm_ms": metrics["comm_ms"],
                                  "inf_ms": metrics["inf_ms"],
                                  "e2e_ms": metrics["e2e_ms"],
                                  "true_dist_m": dist})
                    if action == 0:
                        stopped = True
                        stop_dist = dist
                        stop_reason = "model_stop"
                        break
                    if action == 1:
                        nxt = pos + FORWARD_M * np.asarray(
                            [math.cos(yaw), math.sin(yaw)])
                        status = segment_status(info, float(pos[0]),
                                                float(pos[1]), float(nxt[0]),
                                                float(nxt[1]))
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
                success = bool(stopped and stop_dist is not None
                               and stop_dist <= STOP_RADIUS
                               and final_dist <= STOP_RADIUS)
                osr = bool(min_dist <= STOP_RADIUS)
                e2e = np.asarray([s["e2e_ms"] for s in steps], np.float64)
                inf = np.asarray([s["inf_ms"] for s in steps], np.float64)
                comm = np.asarray([s["comm_ms"] for s in steps], np.float64)
                rec = {**rec_base, "steps": len(steps), "stopped": stopped,
                       "stop_reason": stop_reason, "stop_dist": stop_dist,
                       "final_dist": final_dist, "min_dist": min_dist,
                       "success": success, "osr": osr,
                       "collision": collision, "out_of_bounds": out_of_bounds,
                       "e2e_mean_ms": float(e2e.mean()) if len(e2e) else None,
                       "e2e_p95_ms": float(np.percentile(e2e, 95)) if len(e2e) else None,
                       "inf_mean_ms": float(inf.mean()) if len(inf) else None,
                       "comm_mean_ms": float(comm.mean()) if len(comm) else None,
                       "inf_share": float((inf / e2e).mean()) if len(e2e) else None,
                       "action_counts": {CLASSES[a]: int(sum(
                           1 for s in steps if s["action"] == a)) for a in range(4)},
                       "trajectory": steps,
                       "execution_status": "ok", "exception": "",
                       "seconds": round(time.time() - started, 3)}
                rec["failure"] = classify(rec)
            except Exception as e:  # noqa: BLE001
                rec = {**rec_base, "success": False, "osr": False, "steps": 0,
                       "failure": "exception", "execution_status": "exception",
                       "exception": f"{type(e).__name__}: {e}"[:300],
                       "trajectory": []}
            (episodes / f"{rec['episode_id'].replace(':', '_')}.json").write_text(
                json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            jf.write(json.dumps({k: v for k, v in rec.items()
                                 if k != "trajectory"}) + "\n")
            jf.flush()
            n += 1
            if i % 25 == 0 or i == len(pending):
                rate = i / max(time.time() - started, 1e-9)
                print(f"{i}/{len(pending)} | {rate:.2f} ep/s | "
                      f"eta {(len(pending)-i)/rate:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
