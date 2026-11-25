#!/usr/bin/env python3
"""Stage 15 Phase C: collect strict full-state snapshots along TS closed-loop
rollouts on TRAIN-side navigation episodes.

Snapshots are taken at the start of selected decision steps, before the
expert acts, so a resumed controller can execute any expert's action from the
exact same state.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXP14 = REPO / "experiments/14_mec_expert_retraining"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "counterfactual"))
sys.path.insert(0, str(REPO / "experiments/16_cumulative_e2e_association/p1"))
sys.path.insert(0, str(REPO / "experiments/12_hetero_mec_association/scripts"))
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
from dataset_v2 import load_split              # noqa: E402
from snapshot_io import Snap, save_snapshot    # noqa: E402

NORM_FILE = EXP14 / "stability_revision/normalization_statistics.json"
_ND = json.loads(NORM_FILE.read_text())["standardized"]
NORM = {k: (np.asarray(_ND[k]["mean"], np.float32),
            np.asarray(_ND[k]["std"], np.float32))
        for k in ("belief", "match", "history_pose")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["train", "val", "stage16_train"], default="train")
    ap.add_argument("--manifest", default=str(REPO / "experiments/06_closed_loop_success_rate/results/seed42/manifest.json"))
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--max-snaps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "snapshots"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.source == "stage16_train":
        from train_association_rl import load_tasks_and_split
        train_tasks, dev_tasks = load_tasks_and_split()
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(train_tasks))
        sel = [train_tasks[i] for i in order[: args.episodes]]
        picked = [dict(key=int(order[j]), map=t["map"],
                       target=np.asarray([t["target_x"], t["target_y"]], np.float32),
                       start=np.asarray([t["start_x"], t["start_y"], t["start_yaw"]], np.float32),
                       episode_id=f"train_{int(order[j]):05d}",
                       length_group="train")
                  for j, t in enumerate(sel)]
    elif args.source == "val":
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        tasks = [t for t in manifest["tasks"] if t["split"] == "val_seen"]
        # stratified by length_group, then shuffled within group
        by_group = {}
        for t in tasks:
            by_group.setdefault(t["length_group"], []).append(t)
        picked = []
        per = max(1, args.episodes // max(len(by_group), 1))
        for g, items in sorted(by_group.items()):
            order = rng.permutation(len(items))
            picked.extend(items[i] for i in order[:per])
        picked = picked[: args.episodes]
        picked = [dict(key=int(t["index"]), map=t["map"],
                       target=np.asarray([t["target_x"], t["target_y"]], np.float32),
                       start=np.asarray([t["start_x"], t["start_y"], t["start_yaw"]], np.float32),
                       episode_id=t["episode_id"], length_group=t["length_group"])
                  for t in picked]
    else:
        episodes = load_split("train_seen")
        order = rng.permutation(len(episodes))
        picked = [dict(key=int(ep.key), map=ep.map,
                       target=np.asarray(ep.target, np.float32),
                       start=np.asarray(ep.states[0], np.float32),
                       episode_id=f"train_seen_{ep.key:05d}",
                       length_group="unknown")
                  for ep in [episodes[i] for i in order[: args.episodes]]]

    model = ev.build_model("ts", args.device)
    _ckpt = EXP14 / "stability_revision/runs/ts/seed42/checkpoints/best.pt"
    _st = torch.load(_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(_st["model"])
    model.eval()
    print(f"[collect] loaded TS checkpoint {_ckpt}", flush=True)
    enc4 = ev.FrozenEncoder(args.device)
    matcher_enc = ev.matching.Encoder(args.device)
    rasters, cands = ev.RasterStore(), ev.CandidateStore()

    index = []
    t0 = time.time()
    total_snaps = 0
    for ei, ep in enumerate(picked):
        name = ep["map"]
        raster = rasters.get(name)
        lib = cands.get(name)
        n_cand = int(len(lib.positions))
        empty = n_cand == 0
        target = np.asarray(ep["target"], np.float64)
        goal_img = ev.matching.prepare_goal(raster, float(target[0]), float(target[1]))
        goal_l2 = matcher_enc.features(goal_img[None], ("layer2",))["layer2"][0]
        gdesc = goal_l2[:, 13:15, 13:15].mean(axis=(1, 2))
        gdesc = gdesc / np.maximum(np.linalg.norm(gdesc, keepdims=True), 1e-8)
        info = ev.obstacle_for(name)

        if not empty:
            tvec = enc4.embed([raster.north_crop(float(target[0]), float(target[1]),
                                                 ev.TARGET_HALF_M)])[0].astype(np.float32)
            sims = lib.embeddings @ tvec
            belief = ev.ContinuousBelief(lib.positions.astype(np.float64),
                                         sims.astype(np.float64), 0.1, k=8,
                                         sigma_prior_m=12.0)
        else:
            belief = None
        hist_vis = deque(maxlen=ev.HISTORY_LEN)
        hist_pose = deque(maxlen=ev.HISTORY_LEN)
        hist_act = deque(maxlen=ev.HISTORY_LEN)

        pos = np.asarray(ep["start"][:2], np.float64)
        yaw = float(ep["start"][2])
        target_xy = target
        stopped = False
        for step in range(200):
            saved_json = None
            if step % args.snap_every == 0 and total_snaps < args.max_snaps:
                vis, pose_h, act_h, hmask = ev.history_inputs(
                    hist_vis, hist_pose, hist_act, pos, yaw)
                s = Snap(
                    episode_key=int(ep["key"]), episode_id=ep["episode_id"],
                    split=args.source, map=name, target=[float(x) for x in target_xy],
                    step=step, pos=[float(pos[0]), float(pos[1])], yaw=float(yaw),
                    empty_cands=empty,
                    belief_means=(belief.means.copy() if belief is not None else None),
                    belief_vars=(belief.vars.copy() if belief is not None else None),
                    belief_weights=(belief.weights.copy() if belief is not None else None),
                    belief_last_update=(belief.last_update if belief is not None else -99),
                    belief_last_accept=(None if belief is None or belief.last_accept is None
                                        else belief.last_accept.copy()),
                    hist_vis=np.asarray(list(hist_vis), np.float32),
                    hist_pose=np.asarray(list(hist_pose), np.float32),
                    hist_act=np.asarray(list(hist_act), np.int64),
                    controller="ts",
                )
                rel = out / f"{s.episode_id}_s{step:03d}"
                save_snapshot(rel.with_suffix(".npz"), s)
                index.append({"snapshot": str(rel.with_suffix(".npz")),
                              "episode_key": int(ep["key"]), "episode_id": s.episode_id,
                              "length_group": ep["length_group"],
                              "map": name, "step": step, "empty_cands": empty,
                              "pos_x": s.pos[0], "pos_y": s.pos[1], "yaw": s.yaw,
                              "dist_to_target": float(np.linalg.norm(pos - target_xy))})
                total_snaps += 1
                saved_json = rel.with_suffix(".json")

            dist = float(np.linalg.norm(pos - target_xy))
            if dist < ev.STOP_RADIUS:
                break
            # --- TS decision at this state ---
            if empty:
                belief_feat = ev.EMPTY_BELIEF.copy()
                match_feat = ev.EMPTY_MATCH.copy()
                est = np.asarray([pos[0], pos[1]], np.float64)
            else:
                obs = ev.matching.prepare_observation(raster, pos[0], pos[1], yaw)
                of = matcher_enc.features(obs[None], ("layer2",))["layer2"][0]
                m = ev.subpixel_match(of, goal_l2, 2)
                dx, dy = ev.matching.pixel_to_offset(m["col"], m["row"])
                z = np.array([pos[0] + dx, pos[1] + dy])
                score, margin = m["score"], m["margin"]
                agree = (belief.last_accept is not None and
                         float(np.linalg.norm(z - belief.last_accept)) <= ev.AGREE_M)
                if score >= ev.TAU:
                    belief.last_accept = z.copy()
                if (score >= ev.TAU) and agree:
                    belief.update(z, ev.CAL["sigma_vis_m"], ev.BEL["best"]["eta"],
                                  ev.BEL["best"]["rho"], step)
                est = belief.estimate()
                unc = float(math.sqrt(np.mean(belief.vars)))
                since = float(step - belief.last_update)
                ent = float(belief.entropy())
                if not math.isfinite(ent):
                    ent = 0.0
                belief_feat = ev.belief_features(est, pos, yaw, score, unc, ent,
                                                 since, dx, dy)
                match_feat = np.asarray([dx, dy, score, margin, m["sharpness"]],
                                        np.float32)
            vis, pose_h, act_h, hmask = ev.history_inputs(hist_vis, hist_pose,
                                                          hist_act, pos, yaw)
            if not empty or True:   # TS uses Stage 14-R2 normalization
                belief_feat = ((belief_feat - NORM["belief"][0]) /
                               NORM["belief"][1]).astype(np.float32)
                match_feat = ((match_feat - NORM["match"][0]) /
                              NORM["match"][1]).astype(np.float32)
                pose_h = ((pose_h - NORM["history_pose"][0]) /
                          NORM["history_pose"][1]).astype(np.float32)
            cur_img = raster.heading_crop(float(pos[0]), float(pos[1]), float(yaw))
            batch = {
                "current": None,
                "target_emb": torch.from_numpy(gdesc[None].copy()).to(args.device),
                "history_vis": torch.from_numpy(vis).to(args.device),
                "history_pose": torch.from_numpy(pose_h).to(args.device),
                "history_action": torch.from_numpy(act_h).to(args.device),
                "history_mask": torch.from_numpy(hmask).to(args.device),
                "belief": torch.from_numpy(belief_feat[None].copy()).to(args.device),
                "match": torch.from_numpy(match_feat[None].copy()).to(args.device),
            }
            batch["landmarks"] = torch.from_numpy(
                ev.landmark_inputs(belief, empty)).to(args.device)
            batch["states"] = torch.from_numpy(
                np.asarray([[pos[0], pos[1], yaw]], np.float32)).to(args.device)
            cur_img_np = np.asarray(cur_img, np.uint8)
            with torch.no_grad():
                cur = enc4(np.asarray([cur_img_np]))
                batch["current"] = cur
                logits = model(batch)
            action = int(logits.argmax(-1).item())
            if saved_json is not None:
                _m = json.loads(saved_json.read_text())
                _m["next_action"] = int(action)
                _m["next_action_name"] = ev.CLASSES[int(action)]
                _m["next_logits"] = [float(x) for x in logits.float()[0].tolist()]
                saved_json.write_text(json.dumps(_m, indent=2))
                saved_json = None
            hist_vis.append(ev.pool_history_from_current(cur)[0].astype(np.float32))
            hist_pose.append((float(pos[0]), float(pos[1]), float(yaw)))
            hist_act.append(action)

            if action == 0:
                stopped = True
                break
            if action == 1:
                nxt = pos + ev.FORWARD_M * np.asarray([math.cos(yaw), math.sin(yaw)])
                status = ev.segment_status(info, float(pos[0]), float(pos[1]),
                                           float(nxt[0]), float(nxt[1]))
                if status == "collision":
                    break
                pos = nxt
            else:
                yaw = yaw + (ev.TURN_RAD if action == 2 else -ev.TURN_RAD)
                yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        if (ei + 1) % 10 == 0:
            print(f"[{ei+1}/{len(picked)}] snaps={total_snaps} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    import pandas as pd
    pd.DataFrame(index).to_csv(out / "snapshot_index.csv", index=False)
    print(json.dumps({"episodes": len(picked), "snapshots": total_snaps,
                      "elapsed_sec": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
