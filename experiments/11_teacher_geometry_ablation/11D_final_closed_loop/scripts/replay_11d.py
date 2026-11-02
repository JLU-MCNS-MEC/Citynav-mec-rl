#!/usr/bin/env python3
"""11D offline replay: recompute the REAL M-v4 model softmax per stored step.

Same mechanism as replay_10e.py -- reconstruct the exact v4 model inputs at
every saved decision state (C1 subpixel match + 09C belief causal updates,
16-step causal history, current heading-forward visual), then run the 11B
best.pt forward pass to get the true four-class softmax.

Output: 11D videos/selected_cases/replay/{eid}.json with per-step probs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(REPO / "experiments/06_closed_loop_success_rate/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))

import matching  # noqa: E402
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from pipeline import TARGET_HALF_M  # noqa: E402
from model11b import TransformerM11  # noqa: E402

RES = HERE.parent / "results"
CKPT = (REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4"
        "/results/runs_11b/seed42/best.pt")
CAL = json.loads((REPO / "experiments/09_online_visual_goal_matching"
                  "/09C_continuous_localization/results/09c_calibration.json")
                 .read_text(encoding="utf-8"))
BEL = json.loads((REPO / "experiments/09_online_visual_goal_matching"
                  "/09C_continuous_localization/results/09c_belief_calibration.json")
                 .read_text(encoding="utf-8"))
TAU = 0.875
AGREE_M = 10.0
HISTORY_LEN = 16
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_episode(eid: str) -> dict:
    fn = f"{eid.replace(':', '_')}.json"
    for d in sorted(RES.glob("11d_loop_v4_*")):
        p = d / "episodes" / fn
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            rec["_source"] = str(p)
            return rec
    raise SystemExit(f"episode {eid} not found in 11D result dirs")


def replay(eid: str, device: str = "cuda") -> dict:
    if not CKPT.exists():
        raise SystemExit(f"checkpoint missing: {CKPT}")
    model = TransformerM11().to(device).eval()
    st = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(st["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    enc4 = FrozenEncoder(device)
    matcher_enc = matching.Encoder(device)
    rasters, cands = RasterStore(), CandidateStore()

    rec = load_episode(eid)
    traj = rec["trajectory"]
    raster = rasters.get(rec["map"])
    lib = cands.get(rec["map"])
    if len(lib.positions) == 0:
        raise SystemExit(f"{eid}: no candidates (unevaluable)")
    target = np.asarray(rec["target"], np.float64)

    goal_img = matching.prepare_goal(raster, float(target[0]), float(target[1]))
    goal_l2 = matcher_enc.features(goal_img[None], ("layer2",))["layer2"][0]
    gdesc = goal_l2[:, 13:15, 13:15].mean(axis=(1, 2))
    gdesc = gdesc / np.maximum(np.linalg.norm(gdesc, keepdims=True), 1e-8)
    tvec = enc4.embed([raster.north_crop(float(target[0]), float(target[1]),
                                         TARGET_HALF_M)])[0].astype(np.float32)
    sims = lib.embeddings @ tvec
    belief = ContinuousBelief(lib.positions.astype(np.float64), sims.astype(np.float64),
                              0.1, k=8, sigma_prior_m=12.0)
    hist_vis: deque = deque(maxlen=HISTORY_LEN)
    hist_pose: deque = deque(maxlen=HISTORY_LEN)
    hist_act: deque = deque(maxlen=HISTORY_LEN)

    steps_out = []
    mismatches = 0
    for s in traj:
        pos = np.asarray([s["x"], s["y"]], np.float64)
        yaw = float(s["yaw"])
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
                          BEL["best"]["rho"], s["step"])
        est = belief.estimate()
        vec = est - pos
        fwd = float(vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
        lat = float(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw))
        dist_est = float(np.linalg.norm(vec))
        bearing = math.atan2(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw),
                             vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
        unc = float(math.sqrt(np.mean(belief.vars)))
        since = float(s["step"] - belief.last_update)
        ent = float(belief.entropy())
        if not math.isfinite(ent):
            ent = 0.0
        belief_feat = np.asarray([fwd, lat, dist_est, bearing, score, unc, ent,
                                  since, dx, dy], np.float32)
        match_feat = np.asarray([dx, dy, score, margin, m["sharpness"]], np.float32)
        vis = np.zeros((1, HISTORY_LEN, 2048), np.float32)
        pose = np.zeros((1, HISTORY_LEN, 3), np.float32)
        act = np.zeros((1, HISTORY_LEN, 4), np.float32)
        hmask = np.zeros((1, HISTORY_LEN), np.float32)
        for k in range(len(hist_vis)):
            vis[0, k] = hist_vis[k]
            pose[0, k] = hist_pose[k]
            act[0, k, hist_act[k]] = 1.0
            hmask[0, k] = 1.0
        cur_img = raster.heading_crop(float(pos[0]), float(pos[1]), float(yaw))
        with torch.no_grad():
            cur = enc4(np.asarray([cur_img], np.uint8))
            logits = model({"current": cur,
                            "target_emb": torch.from_numpy(gdesc[None].copy()).to(device),
                            "history_vis": torch.from_numpy(vis).to(device),
                            "history_pose": torch.from_numpy(pose).to(device),
                            "history_action": torch.from_numpy(act).to(device),
                            "history_mask": torch.from_numpy(hmask).to(device),
                            "belief": torch.from_numpy(belief_feat[None].copy()).to(device),
                            "match": torch.from_numpy(match_feat[None].copy()).to(device)})
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float64)
        argmax = int(np.argmax(probs))
        ok = argmax == int(s["action"])
        if not ok:
            mismatches += 1
        with torch.no_grad():
            maps = enc4.spatial_maps([cur_img]).transpose(0, 2, 3, 1)
            pooled = maps.mean(axis=(1, 2))
            pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True),
                                         1e-9)
        hist_vis.append(pooled[0].astype(np.float32))
        hist_pose.append((float(pos[0]), float(pos[1]), float(yaw)))
        hist_act.append(int(s["action"]))
        steps_out.append({"step": int(s["step"]), "action": int(s["action"]),
                          "probs": [float(p) for p in probs],
                          "prob_sum": float(probs.sum()),
                          "argmax_matches_saved_action": ok,
                          "true_dist_m": float(s["true_dist_m"])})
    return {"episode_id": eid, "split": rec["split"], "map": rec["map"],
            "length_group": rec["length_group"], "success": rec["success"],
            "failure": rec["failure"], "stop_dist": rec.get("stop_dist"),
            "min_dist": rec.get("min_dist"), "steps": rec["steps"],
            "checkpoint": str(CKPT), "checkpoint_sha256": sha256(CKPT),
            "source_episode": rec["_source"],
            "source_sha256": sha256(Path(rec["_source"])),
            "argmax_mismatches": mismatches,
            "argmax_fully_consistent": bool(mismatches == 0),
            "prob_sums_all_one": bool(all(abs(s["prob_sum"] - 1.0) < 1e-6
                                          for s in steps_out)),
            "discrete_action_replay": True,
            "trajectory": steps_out}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[],
                        help="episode id (repeatable)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")
    out_dir = HERE.parent / "videos" / "selected_cases" / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = args.only
    if not ids:
        cases = json.loads((HERE.parent / "videos" / "selected_cases" / "MANIFEST.json")
                           .read_text(encoding="utf-8"))["cases"]
        ids = [c["episode_id"] for c in cases]
    for eid in ids:
        out = out_dir / f"{eid.replace(':', '_')}.json"
        if out.exists():
            print(f"skip {eid} (replay exists)", flush=True)
            continue
        r = replay(eid, args.device)
        out.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
        print(f"{eid}: steps={r['steps']} mismatches={r['argmax_mismatches']} "
              f"probs_sum_ok={r['prob_sums_all_one']} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
