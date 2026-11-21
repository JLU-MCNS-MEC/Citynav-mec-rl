#!/usr/bin/env python3
"""Stage 15 Phase C: strict counterfactual rollouts from saved snapshots.

For each snapshot (a full closed-loop state), run five branches:
  step s: expert X acts            (X in small/medium/ts/tm/large)
  step s+1..: fixed TS acts        (documented subsequent policy)
until success / failure / collision / out-of-bounds / step cap.

Also provides a consistency mode: resume a snapshot and continue with the
original expert, then compare the resulting positions with the snapshots that
were recorded later on the uninterrupted trajectory.
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
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXP14 = REPO / "experiments/14_mec_expert_retraining"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP14))
for p in [REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/scripts",
          REPO / "experiments/10_goal_query_mec_training/10D_transformer_m_training/scripts",
          REPO / "experiments/05_transformer_m_training/scripts",
          REPO / "experiments/04_visual_goal_global_map_pipeline/scripts",
          REPO / "experiments/06_closed_loop_success_rate/scripts",
          REPO / "experiments/09_online_visual_goal_matching/scripts",
          REPO / "experiments/09_online_visual_goal_matching/09C_continuous_localization/scripts"]:
    sys.path.insert(0, str(p))

import eval_val_seen as ev                     # noqa: E402
from snapshot_io import Snap, load_snapshot, restore_belief  # noqa: E402

KEYS = ["small", "medium", "ts", "tm", "large"]
CKPTS = {
    "small": EXP14 / "checkpoints/Small/best.pt",
    "medium": EXP14 / "checkpoints/Medium/best.pt",
    "ts": EXP14 / "stability_revision/runs/ts/seed42/checkpoints/best.pt",
    "tm": EXP14 / "stability_revision/runs/tm/seed42/checkpoints/best.pt",
    "large": EXP14 / "stability_revision/runs/large/seed42/checkpoints/best.pt",
}
NORMALIZED = {"ts", "tm", "large"}
NORM_FILE = EXP14 / "stability_revision/normalization_statistics.json"


def load_norm():
    d = json.loads(NORM_FILE.read_text())["standardized"]
    return {k: (np.asarray(d[k]["mean"], np.float32),
                np.asarray(d[k]["std"], np.float32))
            for k in ("belief", "match", "history_pose")}


def apply_norm(b, norm):
    for k, (mean, std) in norm.items():
        if k in b:
            b[k] = ((b[k] - mean) / std).astype(np.float32)
    return b


class Ctx:
    """Shared environment resources."""

    def __init__(self, device="cuda"):
        self.device = device
        self.raster_store = ev.RasterStore()
        self.cand_store = ev.CandidateStore()
        self.enc4 = ev.FrozenEncoder(device)
        self.matcher = ev.matching.Encoder(device)
        self.norm = load_norm()
        self.models = {}
        self._rasters = {}
        self._libs = {}
        self._goals = {}

    def model(self, key):
        if key not in self.models:
            m = ev.build_model(key, self.device)
            st = torch.load(CKPTS[key], map_location=self.device, weights_only=False)
            m.load_state_dict(st["model"])
            m.eval()
            self.models[key] = m
        return self.models[key]

    def env(self, name, target):
        if name not in self._rasters:
            self._rasters[name] = self.raster_store.get(name)
            self._libs[name] = self.cand_store.get(name)
        raster = self._rasters[name]
        key = (name, float(target[0]), float(target[1]))
        if key not in self._goals:
            goal_img = ev.matching.prepare_goal(raster, float(target[0]), float(target[1]))
            goal_l2 = self.matcher.features(goal_img[None], ("layer2",))["layer2"][0]
            gdesc = goal_l2[:, 13:15, 13:15].mean(axis=(1, 2))
            gdesc = gdesc / np.maximum(np.linalg.norm(gdesc, keepdims=True), 1e-8)
            self._goals[key] = (raster, self._libs[name], goal_l2, gdesc)
        return self._goals[key]


def decide(ctx, model_key, raster, goal_l2, gdesc, belief, hist_vis, hist_pose,
           hist_act, pos, yaw, step, empty):
    """One decision identical to the evaluator's preprocessing."""
    if empty:
        belief_feat = ev.EMPTY_BELIEF.copy()
        match_feat = ev.EMPTY_MATCH.copy()
        est = np.asarray([pos[0], pos[1]], np.float64)
        score, margin, accept = 0.0, 0.0, False
    else:
        obs = ev.matching.prepare_observation(raster, pos[0], pos[1], yaw)
        of = ctx.matcher.features(obs[None], ("layer2",))["layer2"][0]
        m = ev.subpixel_match(of, goal_l2, 2)
        dx, dy = ev.matching.pixel_to_offset(m["col"], m["row"])
        z = np.array([pos[0] + dx, pos[1] + dy])
        score, margin = m["score"], m["margin"]
        agree = (belief.last_accept is not None and
                 float(np.linalg.norm(z - belief.last_accept)) <= ev.AGREE_M)
        accept = bool(score >= ev.TAU)
        if accept:
            belief.last_accept = z.copy()
            if agree:
                belief.update(z, ev.CAL["sigma_vis_m"], ev.BEL["best"]["eta"],
                              ev.BEL["best"]["rho"], step)
        est = belief.estimate()
        unc = float(math.sqrt(np.mean(belief.vars)))
        since = float(step - belief.last_update)
        ent = float(belief.entropy())
        if not math.isfinite(ent):
            ent = 0.0
        belief_feat = ev.belief_features(est, pos, yaw, score, unc, ent, since, dx, dy)
        match_feat = np.asarray([dx, dy, score, margin, m["sharpness"]], np.float32)

    vis, pose_h, act_h, hmask = ev.history_inputs(hist_vis, hist_pose, hist_act,
                                                  pos, yaw)
    cur_img = raster.heading_crop(float(pos[0]), float(pos[1]), float(yaw))
    batch = {
        "current": None,
        "target_emb": torch.from_numpy(gdesc[None].copy()).to(ctx.device),
        "history_vis": torch.from_numpy(vis).to(ctx.device),
        "history_pose": torch.from_numpy(pose_h).to(ctx.device),
        "history_action": torch.from_numpy(act_h).to(ctx.device),
        "history_mask": torch.from_numpy(hmask).to(ctx.device),
        "belief": torch.from_numpy(belief_feat[None].copy()).to(ctx.device),
        "match": torch.from_numpy(match_feat[None].copy()).to(ctx.device),
        "landmarks": torch.from_numpy(ev.landmark_inputs(belief, empty)).to(ctx.device),
        "states": torch.from_numpy(np.asarray([[pos[0], pos[1], yaw]], np.float32)).to(ctx.device),
    }
    nn_input = {"belief": belief_feat[None].copy(), "match": match_feat[None].copy(),
                "history_pose": pose_h}
    if model_key in NORMALIZED:
        nn_input = apply_norm(nn_input, ctx.norm)
        batch["belief"] = torch.from_numpy(nn_input["belief"]).to(ctx.device)
        batch["match"] = torch.from_numpy(nn_input["match"]).to(ctx.device)
        batch["history_pose"] = torch.from_numpy(nn_input["history_pose"]).to(ctx.device)
    cur_img_np = np.asarray(cur_img, np.uint8)
    with torch.no_grad():
        cur = ctx.enc4(np.asarray([cur_img_np]))
        batch["current"] = cur
        logits = ctx.model(model_key)(batch)
    logits = logits.float()[0]
    probs = torch.softmax(logits, -1)
    return int(probs.argmax()), float(probs.max()), cur, logits.detach().cpu().numpy()


def rollout(ctx, s: Snap, first_key: str, subsequent_key: str = "ts",
            max_steps: int = 200):
    raster, lib, goal_l2, gdesc = ctx.env(s.map, s.target)
    target = np.asarray(s.target, np.float64)
    info = ev.obstacle_for(s.map)
    empty = bool(s.empty_cands)
    belief = restore_belief(s)
    hist_vis = deque([np.asarray(v, np.float32) for v in s.hist_vis], maxlen=ev.HISTORY_LEN)
    hist_pose = deque([tuple(map(float, p)) for p in s.hist_pose], maxlen=ev.HISTORY_LEN)
    hist_act = deque([int(a) for a in s.hist_act], maxlen=ev.HISTORY_LEN)

    pos = np.asarray(s.pos, np.float64)
    yaw = float(s.yaw)
    start_pos = pos.copy()
    min_dist = float(np.linalg.norm(pos - target))
    steps, actions, path_len = 0, [], 0.0
    stopped = collision = oob = False
    stop_reason = "step_cap"
    for step in range(int(s.step), max_steps):
        key = first_key if step == int(s.step) else subsequent_key
        action, conf, cur, logits_np = decide(ctx, key, raster, goal_l2, gdesc, belief,
                                   hist_vis, hist_pose, hist_act, pos, yaw,
                                   step, empty)
        actions.append({"step": step, "expert": key, "action": action,
                        "confidence": conf,
                        "logits": [float(x) for x in logits_np],
                        "x": float(pos[0]), "y": float(pos[1]), "yaw": float(yaw)})
        hist_vis.append(ev.pool_history_from_current(cur)[0].astype(np.float32))
        hist_pose.append((float(pos[0]), float(pos[1]), float(yaw)))
        hist_act.append(action)
        steps += 1
        if action == 0:
            stopped = True; stop_reason = "model_stop"; break
        if action == 1:
            nxt = pos + ev.FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)])
            status = ev.segment_status(info, float(pos[0]), float(pos[1]),
                                       float(nxt[0]), float(nxt[1]))
            if status == "collision":
                collision = True; stop_reason = "collision"; break
            if status == "out_of_bounds":
                oob = True; stop_reason = "out_of_bounds"; break
            path_len += ev.FORWARD_M
            pos = nxt
        elif action == 2:
            yaw = (yaw + ev.TURN_RAD + math.pi) % (2 * math.pi) - math.pi
        elif action == 3:
            yaw = (yaw - ev.TURN_RAD + math.pi) % (2 * math.pi) - math.pi
        min_dist = min(min_dist, float(np.linalg.norm(pos - target)))
    final_dist = float(np.linalg.norm(pos - target))
    l_star = float(np.linalg.norm(start_pos - target))
    success = bool(stopped and final_dist <= ev.STOP_RADIUS)
    spl = float(success * min(1.0, l_star / path_len)) if path_len > 0 else float(success)
    return {
        "episode_key": s.episode_key, "episode_id": s.episode_id, "map": s.map,
        "start_step": int(s.step), "first_expert": first_key,
        "subsequent_expert": subsequent_key,
        "success": success, "steps_used": steps, "final_dist_m": final_dist,
        "min_dist_m": min_dist, "path_len_m": path_len, "spl": spl,
        "stop_reason": stop_reason, "collision": collision,
        "out_of_bounds": oob, "start_x": float(s.pos[0]),
        "start_y": float(s.pos[1]), "start_yaw": float(s.yaw),
        "target_x": float(target[0]), "target_y": float(target[1]),
        "actions": actions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["consistency", "branch"], required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--selection", default="", help="CSV of snapshots to use")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(args.device)

    if args.mode == "consistency":
        idx = pd.read_csv(Path(args.snapshots) / "snapshot_index.csv")
        # pairs: step k and k+10 on the same episode
        tests = []
        for _, row in idx.iterrows():
            nxt = idx[(idx.episode_id == row.episode_id) &
                      (idx.step == row.step + 10)]
            if len(nxt):
                tests.append((row.snapshot, nxt.iloc[0].snapshot))
        tests = tests[: args.limit or 10]
        rows = []
        for a, b in tests:
            sa = load_snapshot(Path(a))
            sb = load_snapshot(Path(b))
            meta = json.loads(Path(a).with_suffix(".json").read_text())
            expected = meta.get("next_action")
            r = rollout(ctx, sa, "ts", "ts", max_steps=int(sb.step) + 1)
            first = r["actions"][0]["action"] if r["actions"] else None
            last = r["actions"][-1] if r["actions"] else None
            err = (None if last is None else
                   float(np.hypot(last["x"] - sb.pos[0], last["y"] - sb.pos[1])))
            rows.append({"episode_id": sa.episode_id, "start_step": sa.step,
                         "target_step": sb.step,
                         "expected_action": expected, "restored_action": first,
                         "action_match": bool(expected is not None and first == expected),
                         "expected_logits": str(meta.get("next_logits")),
                         "restored_logits": str(r["actions"][0].get("logits") if r["actions"] else None),
                         "reached_steps": r["steps_used"],
                         "final_err_m": err,
                         "ok": bool(err is not None and err < 1e-4),
                         "stop_reason": r["stop_reason"]})
        pd.DataFrame(rows).to_csv(out / "consistency_test.csv", index=False)
        print(json.dumps({"tests": len(rows),
                          "action_match": int(sum(r["action_match"] for r in rows)),
                          "passed": int(sum(r["ok"] for r in rows)),
                          "max_err_m": max([r["final_err_m"] or 0 for r in rows] or [0])},
                         indent=2))
        return

    sel = pd.read_csv(args.selection)
    snaps = sel["snapshot"].tolist()
    if args.limit:
        snaps = snaps[: args.limit]
    rows, traj = [], []
    t0 = time.time()
    for i, sp in enumerate(snaps):
        s = load_snapshot(Path(sp))
        for key in KEYS:
            r = rollout(ctx, s, key, "ts")
            traj.append({"episode_id": r["episode_id"], "start_step": r["start_step"],
                         "first_expert": key, "actions": r["actions"]})
            r = {k: v for k, v in r.items() if k != "actions"}
            rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(snaps)}] elapsed={time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out / "branch_rollout_results.csv", index=False)
    (out / "branch_action_traces.json").write_text(json.dumps(traj))
    print(json.dumps({"snapshots": len(snaps), "rollouts": len(rows),
                      "elapsed_sec": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
