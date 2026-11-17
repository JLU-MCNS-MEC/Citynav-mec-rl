#!/usr/bin/env python3
"""Stage 12: real closed-loop association baselines (4 strategies x 4 groups).

Every strategy executes a *fresh* closed loop: the selected node's hosted expert
produces the navigation action, and the action changes the trajectory.  There is
no replay of Stage-11 trajectories.

Strategies (user-frozen): random / max_rate / max_quality / oracle.
Oracle is privileged and non-deployable; it is the only path importing
``oracle_teacher`` and it records the teacher agitation for auditing.

Usage:
  python run_association_closed_loop.py --group two_homo --strategy random \
      --seed 42 --out-dir results/baselines/two_homo/random/seed42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

# parallel jobs: keep each process from spawning a full CPU thread pool
torch.set_num_threads(int(os.environ.get("STAGE12_TORCH_THREADS", "4")))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

sys.path.insert(0, str(P.REPO / "experiments/v4_closed_loop_revalidation/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/06_closed_loop_success_rate/scripts"))

import matching  # noqa: E402
from closed_loop import (CLASSES, FORWARD_M, STOP_RADIUS, TURN_RAD,  # noqa: E402
                         classify, obstacle_for, segment_status)
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from run_closed_loop import (AGREE_M, BEL, CAL, HISTORY_LEN, TARGET_HALF_M,  # noqa: E402
                             TAU, belief_features, build_model,
                             history_inputs, landmark_inputs,
                             pool_history_from_current)
from v5fix_contract import EMPTY_BELIEF, EMPTY_MATCH  # noqa: E402
from oracle_teacher import TeacherOracle  # noqa: E402
from oracle_selection import (Candidate,  # noqa: E402
                              select_oracle_action_agreement)

ORACLE_V1 = "oracle"                       # historical E2E tie-break version
ORACLE_V2 = "oracle_action_agreement_v2"   # action-agreement-only version
ORACLE_STRATEGIES = (ORACLE_V1, ORACLE_V2)


def load_models(group: str, device: str) -> dict[str, torch.nn.Module]:
    types = sorted({m for m in P.GROUP_SPECS[group]["models"]})
    models: dict[str, torch.nn.Module] = {}
    for mt in types:
        model = build_model(P.RUNNER_MODEL[mt], device)
        ck = torch.load(P.ckpt_path(mt), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        models[mt] = model
    return models


def make_batch(pos, yaw, belief_feat, match_feat, gdesc, vis, pose_h, act_h,
               hmask, belief, empty_cands, device):
    return {
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


def choose_node(nodes, pos, strategy, rng, enc_ms, oracle, info, yaw, target,
                models, batch):
    """Return (node, metrics, action, extra) for one decision step."""
    avail = P.available_nodes(nodes, pos)
    metrics = {id(n): P.node_metrics(n, pos, enc_ms) for n in avail}

    if strategy not in ORACLE_STRATEGIES:
        node = P.select_node(nodes, pos, strategy, rng, enc_ms=enc_ms)
        m = P.node_metrics(node, pos, enc_ms)
        with torch.no_grad():
            logits = models[node["model_type"]](batch)[0]
        return node, m, int(logits.argmax().item()), {}

    if strategy == ORACLE_V2:
        # ---- Oracle v2: action agreement only, no latency/distance/quality ----
        a_star = oracle.action(info, pos, yaw, target)
        probs_by_type = {}
        for mt in {n["model_type"] for n in avail}:
            with torch.no_grad():
                logits = models[mt](batch)[0].float()
            probs_by_type[mt] = torch.softmax(logits, dim=-1).cpu().numpy()
        cands = [Candidate(n, probs_by_type[n["model_type"]])
                 for n in avail]
        sel = select_oracle_action_agreement(cands, a_star)
        node = sel["node"]
        m = metrics[id(node)]
        extra = {
            "teacher_action": a_star,
            "selected_node": node["gbs_id"],
            "selected_model": node["model_type"],
            "selected_action": sel["selected_action"],
            "selected_teacher_action_probability":
                sel["selected_teacher_action_probability"],
            "selected_agreement": sel["selected_agreement"],
            "available_candidate_count": sel["available_candidate_count"],
            "exact_agreement_candidate_count":
                sel["exact_agreement_candidate_count"],
            "selected_node_compute_tflops": node["compute_tflops"],
            "selected_node_comm_ms": m["comm_ms"],
            "selected_node_inf_ms": m["inf_ms"],
            "selected_node_e2e_ms": m["e2e_ms"],
            "tie_break_used": sel["tie_break_used"],
        }
        return node, m, sel["selected_action"], extra

    # ---- privileged Candidate Oracle v1 (E2E tie-break; historical) ----
    a_star = oracle.action(info, pos, yaw, target)
    logits_by_type = {}
    for mt in {n["model_type"] for n in avail}:
        with torch.no_grad():
            logits_by_type[mt] = models[mt](batch)[0].float()
    correct = [n for n in avail
               if int(logits_by_type[n["model_type"]].argmax().item()) == a_star]
    if correct:
        node = min(correct, key=lambda n: metrics[id(n)]["e2e_ms"])
        agree = True
        action = a_star
    else:
        def score(n):
            p = torch.softmax(logits_by_type[n["model_type"]], dim=-1)
            return (float(p[a_star].item()), -metrics[id(n)]["e2e_ms"])
        node = max(avail, key=score)
        agree = False
        action = int(logits_by_type[node["model_type"]].argmax().item())
    return node, metrics[id(node)], action, {"oracle_action": a_star,
                                             "oracle_agree": agree}


def episode(task, nodes, group, strategy, models, enc4, matcher_enc, rasters,
            cands, enc_ms, rng, oracle, step_cap, device):
    name = task["map"]
    if oracle is not None:
        oracle.reset()
    raster = rasters.get(name)
    lib = cands.get(name)
    info = obstacle_for(name)
    pos = np.asarray([task["start_x"], task["start_y"]], np.float64)
    yaw = float(task["start_yaw"])
    target = np.asarray([task["target_x"], task["target_y"]], np.float64)
    n_cand = int(len(lib.positions))
    empty_cands = n_cand == 0

    goal_img = matching.prepare_goal(raster, float(target[0]), float(target[1]))
    goal_l2 = matcher_enc.features(goal_img[None], ("layer2",))["layer2"][0]
    gdesc = goal_l2[:, 13:15, 13:15].mean(axis=(1, 2))
    gdesc = gdesc / np.maximum(np.linalg.norm(gdesc, keepdims=True), 1e-8)

    belief = None
    if not empty_cands:
        tvec = enc4.embed([raster.north_crop(
            float(target[0]), float(target[1]), TARGET_HALF_M)]
        )[0].astype(np.float32)
        sims = lib.embeddings @ tvec
        belief = ContinuousBelief(lib.positions.astype(np.float64),
                                  sims.astype(np.float64), 0.1,
                                  k=8, sigma_prior_m=12.0)

    hist_vis = deque(maxlen=HISTORY_LEN)
    hist_pose = deque(maxlen=HISTORY_LEN)
    hist_act = deque(maxlen=HISTORY_LEN)
    steps = []
    stopped = collision = out_of_bounds = False
    stop_reason, stop_dist = "step_cap", None
    min_dist = float(np.linalg.norm(pos - target))
    min_step = 0
    n_forward = 0

    for step in range(step_cap):
        dist = float(np.linalg.norm(pos - target))
        if dist < min_dist:
            min_dist, min_step = dist, step
        if empty_cands:
            belief_feat = EMPTY_BELIEF.copy()
            match_feat = EMPTY_MATCH.copy()
        else:
            obs = matching.prepare_observation(raster, pos[0], pos[1], yaw)
            of = matcher_enc.features(obs[None], ("layer2",))["layer2"][0]
            mm = subpixel_match(of, goal_l2, 2)
            dx, dy = matching.pixel_to_offset(mm["col"], mm["row"])
            z = np.array([pos[0] + dx, pos[1] + dy])
            score, margin = mm["score"], mm["margin"]
            agree = (belief.last_accept is not None
                     and float(np.linalg.norm(z - belief.last_accept)) <= AGREE_M)
            if score >= TAU:
                belief.last_accept = z.copy()
            if score >= TAU and agree:
                belief.update(z, CAL["sigma_vis_m"], BEL["best"]["eta"],
                              BEL["best"]["rho"], step)
            est = belief.estimate()
            unc = float(math.sqrt(np.mean(belief.vars)))
            since = float(step - belief.last_update)
            ent = float(belief.entropy())
            if not math.isfinite(ent):
                ent = 0.0
            belief_feat = belief_features(est, pos, yaw, score, unc, ent,
                                          since, dx, dy)
            match_feat = np.asarray([dx, dy, score, margin, mm["sharpness"]],
                                    np.float32)

        vis, pose_h, act_h, hmask = history_inputs(
            hist_vis, hist_pose, hist_act, pos, yaw)
        cur_img = raster.heading_crop(float(pos[0]), float(pos[1]), float(yaw))
        batch = make_batch(pos, yaw, belief_feat, match_feat, gdesc, vis,
                           pose_h, act_h, hmask, belief, empty_cands, device)
        with torch.no_grad():
            cur = enc4(np.asarray([cur_img], np.uint8))
            batch["current"] = cur
            node, m, action, extra = choose_node(
                nodes, pos, strategy, rng, enc_ms, oracle, info, yaw, target,
                models, batch)

        hist_vis.append(pool_history_from_current(cur)[0].astype(np.float32))
        hist_pose.append((float(pos[0]), float(pos[1]), float(yaw)))
        hist_act.append(action)
        rec_step = {"step": step, "x": float(pos[0]), "y": float(pos[1]),
                    "yaw": float(yaw), "action": action,
                    "gbs_id": node["gbs_id"],
                    "model_type": node["model_type"],
                    "compute_tflops": node["compute_tflops"],
                    "distance_m": m["distance_m"],
                    "encode_ms": m["encode_ms"],
                    "comm_ms": m["comm_ms"],
                    "inf_ms": m["inf_ms"],
                    "e2e_ms": m["e2e_ms"],
                    "true_dist_m": dist}
        rec_step.update(extra)
        steps.append(rec_step)

        if action == 0:
            stopped, stop_dist, stop_reason = True, dist, "model_stop"
            break
        if action == 1:
            nxt = pos + FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)])
            status = segment_status(info, float(pos[0]), float(pos[1]),
                                    float(nxt[0]), float(nxt[1]))
            if status == "collision":
                collision, stop_reason = True, "collision"
                break
            if status == "out_of_bounds":
                out_of_bounds, stop_reason = True, "out_of_bounds"
                break
            pos = nxt
            n_forward += 1
        elif action == 2:
            yaw = (yaw + TURN_RAD + math.pi) % (2 * math.pi) - math.pi
        elif action == 3:
            yaw = (yaw - TURN_RAD + math.pi) % (2 * math.pi) - math.pi

    final_dist = float(np.linalg.norm(pos - target))
    success = bool(stopped and stop_dist is not None
                   and stop_dist <= STOP_RADIUS and final_dist <= STOP_RADIUS)
    osr = bool(min_dist <= STOP_RADIUS)
    path_len = FORWARD_M * n_forward
    l_star = float(np.linalg.norm(
        np.asarray([task["start_x"], task["start_y"]]) - target))
    spl = (float(success) * min(1.0, l_star / path_len)
           if path_len > 0 else float(success))
    e2e = np.asarray([s["e2e_ms"] for s in steps], np.float64)
    inf = np.asarray([s["inf_ms"] for s in steps], np.float64)
    comm = np.asarray([s["comm_ms"] for s in steps], np.float64)
    enc = np.asarray([s["encode_ms"] for s in steps], np.float64)
    model_share = Counter(s["model_type"] for s in steps)
    node_share = Counter(s["gbs_id"] for s in steps)
    rec = {
        "episode_id": task["episode_id"], "split": task["split"],
        "block_id": name, "index": int(task["index"]), "map": name,
        "length_group": task["length_group"], "cohort": task.get("cohort", ""),
        "group": group, "strategy": strategy,
        "start": [float(task["start_x"]), float(task["start_y"]),
                  float(task["start_yaw"])],
        "target": [float(task["target_x"]), float(task["target_y"])],
        "step_cap": step_cap, "steps": len(steps), "stopped": stopped,
        "stop_reason": stop_reason, "stop_dist": stop_dist,
        "final_dist": final_dist, "min_dist": min_dist,
        "min_dist_step": min_step, "success": success, "osr": osr,
        "ne_m": final_dist, "path_len_m": path_len, "l_star_m": l_star,
        "spl": spl, "steps_forward": n_forward,
        "collision": collision, "out_of_bounds": out_of_bounds,
        "e2e_mean_ms": float(e2e.mean()) if len(e2e) else None,
        "e2e_p50_ms": float(np.percentile(e2e, 50)) if len(e2e) else None,
        "e2e_p95_ms": float(np.percentile(e2e, 95)) if len(e2e) else None,
        "e2e_p99_ms": float(np.percentile(e2e, 99)) if len(e2e) else None,
        "inf_mean_ms": float(inf.mean()) if len(inf) else None,
        "inf_p50_ms": float(np.percentile(inf, 50)) if len(inf) else None,
        "inf_p95_ms": float(np.percentile(inf, 95)) if len(inf) else None,
        "comm_mean_ms": float(comm.mean()) if len(comm) else None,
        "comm_p50_ms": float(np.percentile(comm, 50)) if len(comm) else None,
        "comm_p95_ms": float(np.percentile(comm, 95)) if len(comm) else None,
        "encode_mean_ms": float(enc.mean()) if len(enc) else None,
        "inf_share": float((inf / e2e).mean()) if len(e2e) else None,
        "model_step_share": {k: v / len(steps) for k, v in model_share.items()}
        if steps else {},
        "node_step_share": {k: v / len(steps) for k, v in node_share.items()}
        if steps else {},
        "action_counts": {CLASSES[a]: int(sum(1 for s in steps
                                              if s["action"] == a))
                          for a in range(4)},
        "oracle_agree_rate": (float(np.mean([bool(s.get("oracle_agree"))
                                             for s in steps]))
                              if strategy == ORACLE_V1 and steps else
                              (float(np.mean([bool(s.get("selected_agreement"))
                                              for s in steps]))
                               if strategy == ORACLE_V2 and steps else None)),
        "exact_agreement_rate": (float(np.mean(
            [bool(s.get("selected_agreement")) for s in steps]))
            if strategy == ORACLE_V2 and steps else None),
        "fallback_rate": (float(np.mean(
            [s.get("exact_agreement_candidate_count", 0) == 0
             for s in steps])) if strategy == ORACLE_V2 and steps else None),
        "tie_break_rate": (float(np.mean(
            [bool(s.get("tie_break_used")) for s in steps]))
            if strategy == ORACLE_V2 and steps else None),
        "mean_exact_agreement_candidates": (float(np.mean(
            [s.get("exact_agreement_candidate_count", 0) for s in steps]))
            if strategy == ORACLE_V2 and steps else None),
        "mean_available_candidates": (float(np.mean(
            [s.get("available_candidate_count", 0) for s in steps]))
            if strategy == ORACLE_V2 and steps else None),
        "trajectory": steps,
        "execution_status": "ok", "exception": "",
    }
    rec["failure"] = classify(rec)
    return rec


def summarize(out: Path, group: str, strategy: str, seed: int, n_tasks: int):
    rows = []
    for p in sorted(out.glob("journal.part*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    ok = [r for r in rows if r.get("execution_status") == "ok"
          and r.get("steps", 0) > 0]
    exc = [r for r in rows if r.get("execution_status") != "ok"]
    if not ok:
        summary = {"n_tasks": n_tasks, "n_ok": 0, "n_exception": len(exc)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    e2e = np.asarray([r["e2e_mean_ms"] for r in ok], np.float64)
    inf = np.asarray([r["inf_mean_ms"] for r in ok], np.float64)
    comm = np.asarray([r["comm_mean_ms"] for r in ok], np.float64)
    enc = np.asarray([r["encode_mean_ms"] for r in ok], np.float64)
    ne = np.asarray([r["final_dist"] for r in ok], np.float64)
    model_steps = Counter()
    node_steps = Counter()
    total_steps = 0
    for r in ok:
        total_steps += r["steps"]
        for k, v in r["model_step_share"].items():
            model_steps[k] += v * r["steps"]
        for k, v in r["node_step_share"].items():
            node_steps[k] += v * r["steps"]
    summary = {
        "group": group, "strategy": strategy, "seed": seed,
        "n_tasks": n_tasks, "n_ok": len(ok), "n_exception": len(exc),
        "sr": float(np.mean([r["success"] for r in ok])),
        "osr": float(np.mean([r["osr"] for r in ok])),
        "ne_mean_m": float(np.mean(ne)),
        "ne_p50_m": float(np.percentile(ne, 50)),
        "ne_p95_m": float(np.percentile(ne, 95)),
        "spl_mean": float(np.mean([r.get("spl", 0.0) for r in ok])),
        "steps_mean": float(np.mean([r["steps"] for r in ok])),
        "e2e_mean_ms": float(e2e.mean()),
        "e2e_p50_ms": float(np.percentile(e2e, 50)),
        "e2e_p95_ms": float(np.percentile(e2e, 95)),
        "e2e_p99_ms": float(np.percentile(e2e, 99)),
        "inf_mean_ms": float(inf.mean()),
        "inf_p50_ms": float(np.percentile(inf, 50)),
        "inf_p95_ms": float(np.percentile(inf, 95)),
        "comm_mean_ms": float(comm.mean()),
        "comm_p50_ms": float(np.percentile(comm, 50)),
        "comm_p95_ms": float(np.percentile(comm, 95)),
        "encode_mean_ms": float(enc.mean()),
        "inf_share_mean": float((inf / e2e).mean()),
        "model_step_share": {k: v / max(total_steps, 1)
                             for k, v in model_steps.items()},
        "node_step_share": {k: v / max(total_steps, 1)
                            for k, v in node_steps.items()},
        "failure_counts": {k: sum(1 for r in ok if r["failure"] == k)
                           for k in sorted({r["failure"] for r in ok})},
        "oracle_version": ("v2_action_agreement" if strategy == ORACLE_V2
                           else "v1_e2e_tiebreak"
                           if strategy == ORACLE_V1 else None),
        "oracle_agree_rate": (float(
            sum((r["oracle_agree_rate"] or 0.0) * r["steps"] for r in ok)
            / max(sum(r["steps"] for r in ok
                      if r.get("oracle_agree_rate") is not None), 1))
            if strategy in ORACLE_STRATEGIES else None),
    }
    if strategy == ORACLE_V2:
        def wmean(key):
            num = sum((r.get(key) or 0.0) * r["steps"] for r in ok)
            den = max(sum(r["steps"] for r in ok
                          if r.get(key) is not None), 1)
            return float(num / den)
        summary.update({
            "exact_agreement_rate": wmean("exact_agreement_rate"),
            "fallback_rate": wmean("fallback_rate"),
            "tie_break_rate": wmean("tie_break_rate"),
            "mean_exact_agreement_candidates":
                wmean("mean_exact_agreement_candidates"),
            "mean_available_candidates": wmean("mean_available_candidates"),
        })
    (out / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 ensure_ascii=False))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=P.GROUPS, required=True)
    ap.add_argument("--strategy",
                    choices=tuple(P.STRATEGIES) + (ORACLE_V2,),
                    required=True)
    ap.add_argument("--seed", type=int, default=P.EVAL_SEED)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step-cap", type=int, default=P.MAX_STEPS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = Path(a.out_dir)
    if not out.is_absolute():
        out = P.RES / out
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    shard_i, shard_n = (int(v) for v in a.shard.split("/"))
    part = out / f"journal.part{shard_i:02d}.jsonl"
    steps_part = out / f"steps.part{shard_i:02d}.jsonl"
    write_steps = a.strategy == ORACLE_V2

    tasks = P.eval_tasks(a.limit)
    tasks = [t for i, t in enumerate(tasks) if i % shard_n == shard_i]
    done = set()
    if part.exists():
        done = {json.loads(line)["episode_id"]
                for line in part.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    pending = [t for t in tasks if t["episode_id"] not in done]
    print(f"[start] group={a.group} strategy={a.strategy} seed={a.seed} "
          f"shard={a.shard} tasks={len(tasks)} done={len(done)} "
          f"pending={len(pending)}", flush=True)
    if not pending:
        summarize(out, a.group, a.strategy, a.seed, len(tasks))
        return 0

    device = a.device
    enc_ms = P.encode_ms()
    models = load_models(a.group, device)
    enc4 = FrozenEncoder(device)
    matcher_enc = matching.Encoder(device)
    rasters, cands = RasterStore(), CandidateStore()
    dep = P.deployment(a.group)
    oracle = TeacherOracle() if a.strategy in ORACLE_STRATEGIES else None
    started = time.time()

    step_fh = steps_part.open("a", encoding="utf-8") if write_steps else None
    with part.open("a", encoding="utf-8") as jf:
        for i, task in enumerate(pending, 1):
            # per-episode RNG: sharding / resume cannot change random choices
            rng = np.random.default_rng([a.seed, int(task["index"])])
            try:
                rec = episode(task, dep[task["map"]], a.group, a.strategy,
                              models, enc4, matcher_enc, rasters, cands,
                              enc_ms, rng, oracle, a.step_cap, device)
            except Exception as e:  # noqa: BLE001
                rec = {"episode_id": task["episode_id"], "split": task["split"],
                       "index": int(task["index"]), "map": task["map"],
                       "group": a.group, "strategy": a.strategy,
                       "success": False, "osr": False, "steps": 0,
                       "failure": "exception", "execution_status": "exception",
                       "exception": f"{type(e).__name__}: {e}"[:300],
                       "trajectory": []}
            rec["seconds"] = round(time.time() - started, 3)
            (out / "episodes" /
             f"{rec['episode_id'].replace(':', '_')}.json").write_text(
                json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            jf.write(json.dumps({k: v for k, v in rec.items()
                                 if k != "trajectory"}) + "\n")
            jf.flush()
            if step_fh is not None:
                for s in rec.get("trajectory", []):
                    step_fh.write(json.dumps(
                        {"episode_id": rec["episode_id"], **s}) + "\n")
                step_fh.flush()
            if i % 25 == 0 or i == len(pending):
                rate = i / max(time.time() - started, 1e-9)
                print(f"  {i}/{len(pending)} | {rate:.2f} ep/s | "
                      f"eta {(len(pending)-i)/rate:.0f}s", flush=True)

    if step_fh is not None:
        step_fh.close()
    summary = summarize(out, a.group, a.strategy, a.seed, len(tasks))
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
