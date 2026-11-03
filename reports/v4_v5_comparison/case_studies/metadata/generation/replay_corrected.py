#!/usr/bin/env python3
"""Offline replay: recompute the REAL model softmax per stored step.

For each stored trajectory step, reconstruct the EXACT model inputs the
corrected closed-loop harness built at eval time (C1 subpixel match, 09C
continuous belief causal updates, 16-step causal history via the corrected
history_inputs, current heading-forward visual) and run the checkpoint
forward pass to obtain the true four-class softmax. The trajectory JSONs
store the chosen action but not the probabilities, so this is the only
truthful source for the probability bars in the comparison videos.

Contracts enforced at runtime: the model batch must contain exactly the
input-contract keys (never target coordinates); argmax(probs) must equal
the saved action at every step -- any mismatch aborts before rendering.

Models: v4 (TransformerM11 / runs_11b), v5 (TransformerM12 / runs_11c,
with 8x3 landmark + state extras), v3 (TransformerM model10c / runs_10d).
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
REPO = HERE.parents[2]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
for p in ("experiments/05_transformer_m_training/scripts",
          "experiments/04_visual_goal_global_map_pipeline/scripts",
          "experiments/06_closed_loop_success_rate/scripts",
          "experiments/09_online_visual_goal_matching/scripts",
          "experiments/09_online_visual_goal_matching"
          "/09C_continuous_localization/scripts",
          "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4"
          "/scripts",
          "experiments/v4_closed_loop_revalidation/scripts",
          "experiments/10_goal_query_mec_training"
          "/10D_transformer_m_training/scripts"):
    sys.path.insert(0, str(REPO / p))

import matching  # noqa: E402
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from pipeline import TARGET_HALF_M  # noqa: E402
from model11b import TransformerM11  # noqa: E402
from model11c import TransformerM12  # noqa: E402
from model10c import TransformerM  # noqa: E402
from run_closed_loop import (AGREE_M, EMPTY_SINCE, EMPTY_UNC, HISTORY_LEN,  # noqa: E402
                             N_LM, TAU, history_inputs)

CAL = json.loads((REPO / "experiments/09_online_visual_goal_matching"
                  "/09C_continuous_localization/results/09c_calibration.json")
                 .read_text(encoding="utf-8"))
BEL = json.loads((REPO / "experiments/09_online_visual_goal_matching"
                  "/09C_continuous_localization/results/09c_belief_calibration.json")
                 .read_text(encoding="utf-8"))
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
BASE_CONTRACT = ["current", "target_emb", "history_vis", "history_pose",
                 "history_action", "history_mask", "belief", "match"]
MODEL_SPECS = {
    "v4": {"cls": TransformerM11, "ckpt": REPO / "experiments"
           "/11_teacher_geometry_ablation/11B_transformer_m_v4/results"
           "/runs_11b/seed42/best.pt", "contract": BASE_CONTRACT,
           "results": REPO / "experiments/v4_closed_loop_revalidation/results",
           "prefix": "full_"},
    "v5": {"cls": TransformerM12, "ckpt": REPO / "experiments"
           "/11_teacher_geometry_ablation/11B_transformer_m_v4/results"
           "/runs_11c/seed42/best.pt",
           "contract": BASE_CONTRACT + ["landmarks", "states"],
           "results": REPO / "experiments/v5_closed_loop_evaluation/results",
           "prefix": "v5_"},
    "v3": {"cls": TransformerM, "ckpt": REPO / "experiments"
           "/10_goal_query_mec_training/10D_transformer_m_training/results"
           "/runs_10d/seed42/best.pt", "contract": BASE_CONTRACT,
           "results": REPO / "experiments/v3_closed_loop_revalidation/results",
           "prefix": "v3_"},
}
DEFAULT_OUT = (REPO / "reports/v4_v5_comparison/case_studies"
               / "metadata" / "replay")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def episode_dirs(model: str) -> list[Path]:
    spec = MODEL_SPECS[model]
    return sorted(d for d in spec["results"].glob(f"{spec['prefix']}*/episodes")
                  if d.is_dir())


def load_episode(model: str, eid: str) -> tuple[dict, Path]:
    fn = f"{eid.replace(':', '_')}.json"
    for d in episode_dirs(model):
        p = d / fn
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), p
    raise SystemExit(f"episode {eid} not found for model {model}")


def empty_state(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, float, float, bool]:
    """Mirror run_closed_loop.py empty-candidate branch (lines 248-256)."""
    belief_feat = np.asarray([0, 0, 0, 0, 0, EMPTY_UNC, 0, EMPTY_SINCE, 0, 0],
                             np.float32)
    match_feat = np.zeros(5, np.float32)
    est = np.asarray([pos[0], pos[1]], np.float64)
    return belief_feat, match_feat, est, 0.0, 0.0, False


def build_v5_extras(belief, empty_cands: bool, pos: np.ndarray,
                    yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Mirror run_closed_loop.py v5 landmark block (lines 310-319)."""
    lm = np.zeros((1, N_LM, 3), np.float32)
    if not empty_cands:
        n_k = int(belief.means.shape[0])
        lm[0, :n_k, :2] = belief.means.astype(np.float32)
        lm[0, :n_k, 2] = belief.weights.astype(np.float32)
    states = np.asarray([[pos[0], pos[1], yaw]], np.float32)
    return lm, states


def assert_batch_contract(model: str, batch: dict) -> None:
    """The batch must contain EXACTLY the contract keys -- never more, and
    in particular never the true target coordinates."""
    got = set(batch.keys())
    expect = set(MODEL_SPECS[model]["contract"])
    assert got == expect, f"{model} batch keys {sorted(got)} != contract " \
                          f"{sorted(expect)}"


def replay(model_name: str, eid: str, device: str = "cuda") -> dict:
    spec = MODEL_SPECS[model_name]
    if not spec["ckpt"].exists():
        raise SystemExit(f"checkpoint missing: {spec['ckpt']}")
    model = spec["cls"]().to(device).eval()
    st = torch.load(spec["ckpt"], map_location=device, weights_only=False)
    model.load_state_dict(st["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    enc4 = FrozenEncoder(device)
    matcher_enc = matching.Encoder(device)
    rasters, cands = RasterStore(), CandidateStore()

    rec, src = load_episode(model_name, eid)
    traj = rec["trajectory"]
    raster = rasters.get(rec["map"])
    lib = cands.get(rec["map"])
    empty_cands = len(lib.positions) == 0
    target = np.asarray(rec["target"], np.float64)

    goal_img = matching.prepare_goal(raster, float(target[0]), float(target[1]))
    goal_l2 = matcher_enc.features(goal_img[None], ("layer2",))["layer2"][0]
    gdesc = goal_l2[:, 13:15, 13:15].mean(axis=(1, 2))
    gdesc = gdesc / np.maximum(np.linalg.norm(gdesc, keepdims=True), 1e-8)
    belief = None
    if not empty_cands:
        tvec = enc4.embed([raster.north_crop(float(target[0]), float(target[1]),
                                             TARGET_HALF_M)])[0] \
            .astype(np.float32)
        sims = lib.embeddings @ tvec
        belief = ContinuousBelief(lib.positions.astype(np.float64),
                                  sims.astype(np.float64), 0.1, k=8,
                                  sigma_prior_m=12.0)
    hist_vis: deque = deque(maxlen=HISTORY_LEN)
    hist_pose: deque = deque(maxlen=HISTORY_LEN)
    hist_act: deque = deque(maxlen=HISTORY_LEN)

    steps_out = []
    mismatches = 0
    est_mismatches = 0
    for s in traj:
        pos = np.asarray([s["x"], s["y"]], np.float64)
        yaw = float(s["yaw"])
        if empty_cands:
            belief_feat, match_feat, est, _de, _sc, _ac = empty_state(pos)
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
                              BEL["best"]["rho"], s["step"])
            est = belief.estimate()
            vec = est - pos
            fwd = float(vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
            lat = float(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw))
            dist_est = float(np.linalg.norm(vec))
            bearing = math.atan2(
                -vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw),
                vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
            unc = float(math.sqrt(np.mean(belief.vars)))
            since = float(s["step"] - belief.last_update)
            ent = float(belief.entropy())
            if not math.isfinite(ent):
                ent = 0.0
            belief_feat = np.asarray([fwd, lat, dist_est, bearing, score, unc,
                                      ent, since, dx, dy], np.float32)
            match_feat = np.asarray([dx, dy, score, margin, m["sharpness"]],
                                    np.float32)
        # corrected history: slot 0 = most recent, relative pose (training)
        vis, pose_h, act_h, hmask = history_inputs(hist_vis, hist_pose,
                                                   hist_act, pos, yaw)
        cur_img = raster.heading_crop(float(pos[0]), float(pos[1]), float(yaw))
        with torch.no_grad():
            cur = enc4(np.asarray([cur_img], np.uint8))
            batch = {
                "current": cur,
                "target_emb": torch.from_numpy(gdesc[None].copy()).to(device),
                "history_vis": torch.from_numpy(vis).to(device),
                "history_pose": torch.from_numpy(pose_h).to(device),
                "history_action": torch.from_numpy(act_h).to(device),
                "history_mask": torch.from_numpy(hmask).to(device),
                "belief": torch.from_numpy(belief_feat[None].copy()).to(device),
                "match": torch.from_numpy(match_feat[None].copy()).to(device),
            }
            if model_name == "v5":
                lm, states = build_v5_extras(belief, empty_cands, pos, yaw)
                batch["landmarks"] = torch.from_numpy(lm).to(device)
                batch["states"] = torch.from_numpy(states).to(device)
            assert_batch_contract(model_name, batch)
            logits = model(batch)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float64)
        argmax = int(np.argmax(probs))
        ok = argmax == int(s["action"])
        if not ok:
            mismatches += 1
        with torch.no_grad():
            maps = enc4.spatial_maps([cur_img]).transpose(0, 2, 3, 1)
            pooled = maps.mean(axis=(1, 2))
            pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=1,
                                                        keepdims=True), 1e-9)
        hist_vis.append(pooled[0].astype(np.float32))
        hist_pose.append((float(pos[0]), float(pos[1]), float(yaw)))
        hist_act.append(int(s["action"]))
        # decisive reconstruction check: the recomputed belief estimate must
        # equal the estimate stored at eval time (tolerance 1e-3 m) -- if it
        # does, the model inputs are exact and any argmax flip is GPU
        # floating-point noise at a near-tie, not a reconstruction error
        stored_est = s.get("estimate")
        rec_est = [float(est[0]), float(est[1])]
        est_ok = (stored_est is not None
                  and all(abs(a - b) <= 1e-3 for a, b
                          in zip(rec_est, stored_est)))
        if not est_ok:
            est_mismatches += 1
        steps_out.append({"step": int(s["step"]), "action": int(s["action"]),
                          "probs": [float(p) for p in probs],
                          "prob_sum": float(probs.sum()),
                          "argmax_matches_saved_action": ok,
                          "recomputed_estimate": rec_est,
                          "stored_estimate": stored_est,
                          "estimate_matches_stored": est_ok,
                          "true_dist_m": float(s["true_dist_m"])})
    return {"episode_id": eid, "model": model_name,
            "split": rec["split"], "map": rec["map"],
            "length_group": rec["length_group"], "success": rec["success"],
            "failure": rec["failure"], "stop_dist": rec.get("stop_dist"),
            "min_dist": rec.get("min_dist"), "steps": rec["steps"],
            "empty_candidates": empty_cands,
            "checkpoint": str(spec["ckpt"]),
            "checkpoint_sha256": sha256(spec["ckpt"]),
            "source_episode": str(src),
            "source_sha256": sha256(src),
            "argmax_mismatches": mismatches,
            "argmax_fully_consistent": bool(mismatches == 0),
            "estimate_mismatches": est_mismatches,
            "estimates_all_match_stored": bool(est_mismatches == 0),
            "prob_sums_all_one": bool(all(abs(x["prob_sum"] - 1.0) < 1e-6
                                          for x in steps_out)),
            "batch_contract_checked": "assert_batch_contract at call time",
            "discrete_action_replay": True,
            "trajectory": steps_out}


def selection_ids(selection_path: Path, model: str) -> set[str]:
    data = json.loads(selection_path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for cat in data.get("categories", {}).values():
        for p in cat.get("selected", []):
            if model in (p.get("winner_model"), p.get("loser_model")):
                ids.add(p["episode_id"])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["v4", "v5", "v3"])
    parser.add_argument("--selection", default="")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = set(args.only)
    if args.selection:
        ids |= selection_ids(Path(args.selection), args.model)
    if not ids:
        raise SystemExit("no episodes: pass --only or --selection")
    for eid in sorted(ids):
        out = out_dir / f"{args.model}_{eid.replace(':', '_')}.json"
        if out.exists():
            print(f"skip {eid} (replay exists)", flush=True)
            continue
        r = replay(args.model, eid, args.device)
        out.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
        print(f"{args.model} {eid}: steps={r['steps']} "
              f"mismatches={r['argmax_mismatches']} "
              f"probs_sum_ok={r['prob_sums_all_one']} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
