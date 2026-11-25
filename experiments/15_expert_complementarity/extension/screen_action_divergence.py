#!/usr/bin/env python3
"""Stage 15-Extension D2b: cheap action-divergence screen for candidate states.

Stage 15 showed that outcome sensitivity across the five experts is entirely
driven by first-step action disagreement.  A first-step action costs one
perception pass + five model forwards (milliseconds), while a full five-branch
counterfactual rollout costs ~5 s.  So we screen every candidate state cheaply
and spend the expensive rollouts only on informative states.

Output: one row per snapshot with the five expert actions/confidences and a
`divergent` flag.  Selection for rollout is done separately so the screening
criterion never leaks into the evaluation split.
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
sys.path.insert(0, str(HERE.parent / "counterfactual"))
sys.path.insert(0, str(REPO / "experiments/14_mec_expert_retraining"))
for p in [REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/scripts",
          REPO / "experiments/10_goal_query_mec_training/10D_transformer_m_training/scripts",
          REPO / "experiments/05_transformer_m_training/scripts",
          REPO / "experiments/04_visual_goal_global_map_pipeline/scripts",
          REPO / "experiments/06_closed_loop_success_rate/scripts",
          REPO / "experiments/09_online_visual_goal_matching/scripts",
          REPO / "experiments/09_online_visual_goal_matching/09C_continuous_localization/scripts"]:
    sys.path.insert(0, str(p))

import eval_val_seen as ev                                    # noqa: E402
from branch_rollout import CKPTS, KEYS, NORMALIZED, Ctx, apply_norm  # noqa: E402
from snapshot_io import load_snapshot, restore_belief         # noqa: E402

NAMES = {0: "Stop", 1: "Forward", 2: "TurnLeft", 3: "TurnRight"}


def perceive(ctx: Ctx, s):
    """Replicate branch_rollout.decide's preprocessing exactly, once.

    Returns the un-normalized feature batch plus the tensors that are shared
    across all five expert models.
    """
    raster, lib, goal_l2, gdesc = ctx.env(s.map, s.target)
    empty = bool(s.empty_cands)
    belief = restore_belief(s)
    pos = np.asarray(s.pos, np.float64)
    yaw = float(s.yaw)
    hist_vis = deque([np.asarray(v, np.float32) for v in s.hist_vis], maxlen=ev.HISTORY_LEN)
    hist_pose = deque([tuple(map(float, p)) for p in s.hist_pose], maxlen=ev.HISTORY_LEN)
    hist_act = deque([int(a) for a in s.hist_act], maxlen=ev.HISTORY_LEN)
    step = int(s.step)

    if empty:
        belief_feat = ev.EMPTY_BELIEF.copy()
        match_feat = ev.EMPTY_MATCH.copy()
    else:
        obs = ev.matching.prepare_observation(raster, pos[0], pos[1], yaw)
        of = ctx.matcher.features(obs[None], ("layer2",))["layer2"][0]
        m = ev.subpixel_match(of, goal_l2, 2)
        dx, dy = ev.matching.pixel_to_offset(m["col"], m["row"])
        z = np.array([pos[0] + dx, pos[1] + dy])
        score, margin = m["score"], m["margin"]
        agree = (belief.last_accept is not None and
                 float(np.linalg.norm(z - belief.last_accept)) <= ev.AGREE_M)
        if score >= ev.TAU:
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

    vis, pose_h, act_h, hmask = ev.history_inputs(hist_vis, hist_pose, hist_act, pos, yaw)
    cur_img = raster.heading_crop(float(pos[0]), float(pos[1]), float(yaw))
    with torch.no_grad():
        cur = ctx.enc4(np.asarray([np.asarray(cur_img, np.uint8)]))
    batch = {
        "current": cur,
        "target_emb": torch.from_numpy(gdesc[None].copy()).to(ctx.device),
        "history_vis": torch.from_numpy(vis).to(ctx.device),
        "history_action": torch.from_numpy(act_h).to(ctx.device),
        "history_mask": torch.from_numpy(hmask).to(ctx.device),
        "landmarks": torch.from_numpy(ev.landmark_inputs(belief, empty)).to(ctx.device),
        "states": torch.from_numpy(np.asarray([[pos[0], pos[1], yaw]], np.float32)).to(ctx.device),
    }
    raw = {"belief": belief_feat[None].copy(), "match": match_feat[None].copy(),
           "history_pose": pose_h}
    return batch, raw, empty


def action_of(ctx, key, batch, raw):
    b = dict(batch)
    if key in NORMALIZED:
        n = apply_norm({k: v.copy() for k, v in raw.items()}, ctx.norm)
        b["belief"] = torch.from_numpy(n["belief"]).to(ctx.device)
        b["match"] = torch.from_numpy(n["match"]).to(ctx.device)
        b["history_pose"] = torch.from_numpy(n["history_pose"]).to(ctx.device)
    else:
        b["belief"] = torch.from_numpy(raw["belief"]).to(ctx.device)
        b["match"] = torch.from_numpy(raw["match"]).to(ctx.device)
        b["history_pose"] = torch.from_numpy(raw["history_pose"]).to(ctx.device)
    with torch.no_grad():
        logits = ctx.model(key)(b).float()[0]
    probs = torch.softmax(logits, -1)
    return int(probs.argmax()), float(probs.max()), float(
        -(probs * torch.log(probs + 1e-12)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    snap_dir = Path(args.snapshots)
    out = Path(args.out) if args.out else snap_dir.parent / "screen_actions.csv"

    idx = pd.read_csv(snap_dir / "snapshot_index.csv")
    if args.limit:
        idx = idx.head(args.limit)
    ctx = Ctx(args.device)
    for k in KEYS:                       # preload all five experts
        ctx.model(k)

    rows, t0 = [], time.time()
    def resolve(p):
        """Index files were written with either repo-relative or
        extension-relative paths depending on the launch directory."""
        p = Path(p)
        if p.is_absolute() and p.exists():
            return p
        for base in (REPO, HERE, Path.cwd()):
            cand = base / p
            if cand.exists():
                return cand
        return p if p.is_absolute() else REPO / p

    for i, r in idx.iterrows():
        s = load_snapshot(resolve(r["snapshot"]))
        batch, raw, _ = perceive(ctx, s)
        row = {"snapshot": r["snapshot"], "episode_id": s.episode_id,
               "episode_key": int(s.episode_key), "map": s.map, "step": int(s.step),
               "dist_to_target": float(r["dist_to_target"]),
               "empty_cands": bool(s.empty_cands)}
        acts = []
        for k in KEYS:
            a, c, h = action_of(ctx, k, batch, raw)
            row[f"a_{k}"] = a
            row[f"conf_{k}"] = c
            row[f"ent_{k}"] = h
            acts.append(a)
        row["n_distinct_actions"] = len(set(acts))
        row["divergent"] = bool(len(set(acts)) > 1)
        row["majority"] = max(set(acts), key=acts.count) if acts else -1
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(idx)}] elapsed={time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(json.dumps({
        "states": int(len(df)),
        "divergent": int(df.divergent.sum()),
        "divergent_pct": float(df.divergent.mean()),
        "distinct_hist": {int(k): int(v) for k, v in
                          df.n_distinct_actions.value_counts().sort_index().items()},
        "elapsed_sec": round(time.time() - t0, 1),
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
