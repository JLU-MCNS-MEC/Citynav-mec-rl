#!/usr/bin/env python3
"""10B closed-loop: C1 / PQ / CA continuous target localisation.

Only the visual matching module differs between modes:

* c1 - 09C-1 ``subpixel_match`` (dense cosine + sub-pixel centroid);
       run_episode is a faithful copy of ``continuous_env.run_episode``
       (c1 branch), so results must reproduce 09C-1 on identical inputs;
* pq - 10A PointQueryMatcher (top-1 proposal, tau from 10A calibration);
* ca - 10A CrossAttnMatcher (top-1 proposal, tau from 10A calibration).

Everything else is frozen and identical across modes: prior from the 24 m
candidate library, ContinuousBelief Kalman updates (09C calibrated
sigma_vis_m / eta / rho), the agree_m=10 m two-frame gate and the geometric
controller (STOP 20 m / align 25 deg / forward 5 m / turn 30 deg, cap 200).

Resumable journal keyed by manifest SHA + mode (same scheme as run_09c.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/06_closed_loop_success_rate/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
sys.path.insert(0, str(REPO / "experiments/10_goal_query_mec_training"
                       "/10A_point_query_cross_attention/scripts"))

import matching  # noqa: E402
from closed_loop import (CLASSES, FORWARD_M, STOP_RADIUS, TURN_RAD, classify,  # noqa: E402
                         obstacle_for, segment_status)
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from continuous_env import record_summary  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from pipeline import TARGET_HALF_M  # noqa: E402

from model10a import load_matcher  # noqa: E402

STAGE06 = REPO / "experiments/06_closed_loop_success_rate/results/seed42"
RES09C = (REPO / "experiments/09_online_visual_goal_matching"
          "/09C_continuous_localization/results")
RES = HERE.parent / "results"
RES10A = (REPO / "experiments/10_goal_query_mec_training"
          "/10A_point_query_cross_attention/results")
ALIGN_RAD = math.radians(25.0)
DEFAULT_STEP_CAP = 200


class C1Matcher:
    """09C-1 dense cosine + sub-pixel centroid (CPU, as in production)."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def match(self, obs_feat: np.ndarray, goal_feat: np.ndarray) -> dict:
        m = subpixel_match(obs_feat, goal_feat, 2)
        return {"col": float(m["col"]), "row": float(m["row"]),
                "score": float(m["score"]), "margin": float(m["margin"])}


class LearnedMatcher:
    """10A PQ / CA heads (GPU)."""

    def __init__(self, method: str, ckpt: str, device: str = "cuda"):
        import torch
        self.torch = torch
        self.model = load_matcher(method, ckpt, device)
        self.device = device

    def match(self, obs_feat: np.ndarray, goal_feat: np.ndarray) -> dict:
        torch = self.torch
        of = torch.from_numpy(obs_feat[None]).to(self.device)
        gf = torch.from_numpy(goal_feat[None]).to(self.device)
        with torch.no_grad():
            r = self.model(of, gf)
        return {"col": float(r["col"][0]), "row": float(r["row"][0]),
                "score": float(r["score"][0]), "margin": float(r["margin"][0])}


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


def run_episode(task: dict, matcher, encoder, fe: FrozenEncoder,
                rasters: RasterStore, cands: CandidateStore, goal_cache: dict,
                goal_feat_cache: dict, params: dict, mode: str,
                step_cap: int = DEFAULT_STEP_CAP) -> dict:
    """Faithful copy of continuous_env.run_episode (c1 branch) with the
    matching call replaced by ``matcher.match``.  ``encoder`` is the frozen
    matching.Encoder (layer2 features); ``fe`` is the dataset_v2
    FrozenEncoder (candidate prior embedding)."""
    started = time.time()
    name = task["map"]
    raster = rasters.get(name)
    lib = cands.get(name)
    info = obstacle_for(name)
    pos = np.asarray([float(task["start_x"]), float(task["start_y"])], np.float64)
    yaw = float(task["start_yaw"])
    target = np.asarray([float(task["target_x"]), float(task["target_y"])], np.float64)

    tvec = fe.embed([raster.north_crop(float(target[0]), float(target[1]),
                                       TARGET_HALF_M)])[0].astype(np.float32)
    sims = lib.embeddings @ tvec
    belief = ContinuousBelief(lib.positions.astype(np.float64), sims.astype(np.float64),
                              params["temperature"], k=params.get("k", 8),
                              sigma_prior_m=params.get("sigma_prior_m", 12.0))

    gkey = (name, round(float(target[0]), 2), round(float(target[1]), 2))
    if gkey not in goal_cache:
        goal_cache[gkey] = matching.prepare_goal(raster, target[0], target[1])
    if gkey not in goal_feat_cache:
        goal_feat_cache[gkey] = encoder.features(goal_cache[gkey][None],
                                                 ("layer2",))["layer2"][0]

    steps, updates = [], 0
    stopped = collision = out_of_bounds = False
    stop_dist = None
    min_dist = float(np.linalg.norm(pos - target))
    min_step = 0
    entropies = []
    gate = params["gate"]

    for step in range(step_cap):
        dist = float(np.linalg.norm(pos - target))
        if dist < min_dist:
            min_dist, min_step = dist, step

        obs = matching.prepare_observation(raster, pos[0], pos[1], yaw)
        of = encoder.features(obs[None], ("layer2",))["layer2"][0]
        m = matcher.match(of, goal_feat_cache[gkey])
        dx, dy = matching.pixel_to_offset(m["col"], m["row"])
        z = np.array([pos[0] + dx, pos[1] + dy])

        score, margin = m["score"], m["margin"]
        agree_ref = belief.last_accept
        agree = (agree_ref is not None
                 and float(np.linalg.norm(z - agree_ref)) <= gate["agree_m"])
        validity = float(raster.valid_fraction(float(z[0]), float(z[1]), 8.0))
        maha = belief.best_mahalanobis(z)
        conf_ok = score >= gate["tau"]
        accept = conf_ok and agree
        if conf_ok:                     # remember even when the pair is rejected
            belief.last_accept = z.copy()

        updated = False
        if accept:
            belief.update(z, params["sigma_vis_m"], params["eta"], params["rho"], step)
            updates += 1
            updated = True
        est = belief.estimate()
        ent = belief.entropy()
        entropies.append(ent)

        vec = est - pos
        distance = float(np.linalg.norm(vec))
        bearing = math.atan2(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw),
                             vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
        if distance <= STOP_RADIUS:
            action = 0
        elif bearing > ALIGN_RAD:
            action = 2
        elif bearing < -ALIGN_RAD:
            action = 3
        else:
            action = 1

        steps.append({"step": step, "x": float(pos[0]), "y": float(pos[1]),
                      "yaw": float(yaw), "action": int(action),
                      "estimate": [float(est[0]), float(est[1])],
                      "est_distance_m": distance,
                      "est_bearing_deg": math.degrees(bearing),
                      "visual_score": float(score), "visual_margin": float(margin),
                      "visual_z": [float(z[0]), float(z[1])],
                      "visual_err_to_true_m": float(np.linalg.norm(z - target)),
                      "estimate_err_to_true_m": float(np.linalg.norm(est - target)),
                      "accept": bool(accept), "updated": bool(updated),
                      "agree": bool(agree), "validity": validity,
                      "mahalanobis": maha, "n_modes": int(len(belief.weights)),
                      "entropy": ent, "true_dist_m": dist})
        if params.get("record_belief"):   # 10C recovery-state feature building
            steps[-1]["belief_state"] = {
                "means": belief.means.tolist(), "vars": belief.vars.tolist(),
                "weights": belief.weights.tolist(),
                "last_accept": None if belief.last_accept is None
                else belief.last_accept.tolist(),
                "last_update": int(belief.last_update)}

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
                steps[-1]["blocked"] = "collision"
                break
            if status == "out_of_bounds":
                out_of_bounds = True
                steps[-1]["blocked"] = "out_of_bounds"
                break
            pos = nxt
        elif action == 2:
            yaw = (yaw + TURN_RAD + math.pi) % (2 * math.pi) - math.pi
        elif action == 3:
            yaw = (yaw - TURN_RAD + math.pi) % (2 * math.pi) - math.pi

    final_dist = float(np.linalg.norm(pos - target))
    osr = bool(min_dist <= STOP_RADIUS)
    success = bool(stopped and stop_dist is not None and stop_dist <= STOP_RADIUS
                   and final_dist <= STOP_RADIUS)
    rec = {
        "episode_id": task["episode_id"], "split": task["split"], "index": int(task["index"]),
        "map": name, "length_group": task["length_group"], "cohort": task["cohort"],
        "controller": f"10b_{mode}", "mode": mode,
        "start": [float(task["start_x"]), float(task["start_y"]), float(task["start_yaw"])],
        "target": [float(task["target_x"]), float(task["target_y"])],
        "step_cap": int(step_cap), "steps": len(steps),
        "stopped": stopped, "stop_step": (len(steps) - 1) if stopped else None,
        "stop_dist": stop_dist, "final_pos": [float(pos[0]), float(pos[1]), float(yaw)],
        "final_dist": final_dist, "min_dist": min_dist, "min_dist_step": int(min_step),
        "osr": osr, "success": success, "collision": collision,
        "out_of_bounds": out_of_bounds,
        "path_len_m": FORWARD_M * sum(1 for s in steps if s["action"] == 1),
        "teacher_path_len_m": float(task.get("teacher_path_len_m") or 0.0),
        "action_counts": {CLASSES[i]: int(sum(1 for s in steps if s["action"] == i))
                          for i in range(4)},
        "goal_image_invalid": bool(task.get("goal_image_invalid", False)),
        "target_valid_fraction": float(task.get("target_valid_fraction", 1.0)),
        "belief_updates": updates,
        "mean_entropy": float(np.mean(entropies)) if entropies else None,
        "final_entropy": float(entropies[-1]) if entropies else None,
        "trajectory": steps,
        "seconds": round(time.time() - started, 3),
        "manifest_sha": task.get("manifest_sha"),
    }
    rec["failure"] = classify(rec)
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["c1", "pq", "ca"], required=True)
    parser.add_argument("--manifest", default=str(STAGE06 / "manifest.json"))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-file", default="")
    parser.add_argument("--step-cap", type=int, default=DEFAULT_STEP_CAP)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record-belief", action="store_true",
                        help="store belief internals per step (10C recovery data)")
    args = parser.parse_args()

    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set "
                         "(new server: /mnt/hdd8t/Citynav-MEC/DATA)")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha_path = manifest_path.with_name(manifest_path.name + ".sha256")
    manifest_sha = sha_path.read_text(encoding="utf-8").strip()

    # belief update parameters: frozen from 09C calibration (identical for all
    # modes -- only the matcher differs)
    cal_9c = json.loads((RES09C / "09c_calibration.json").read_text(encoding="utf-8"))
    cal_b = json.loads((RES09C / "09c_belief_calibration.json").read_text(
        encoding="utf-8"))
    best = cal_b["best"]
    if args.mode == "c1":
        tau = 0.875          # frozen historical 09A layer2 calibration
        cal_sha = sha256(RES09C / "09c_calibration.json") + sha256(
            RES09C / "09c_belief_calibration.json")
    else:
        cal10 = json.loads((RES10A / args.mode / "calibration.json").read_text(
            encoding="utf-8"))
        tau = cal10["tau"]
        cal_sha = sha256(RES10A / args.mode / "calibration.json") + sha256(
            RES09C / "09c_belief_calibration.json")
    gate = {"tau": tau, "agree_m": 10.0}
    params = {"temperature": 0.1, "sigma_prior_m": 12.0, "k": 8,
              "sigma_vis_m": cal_9c["sigma_vis_m"], "eta": best["eta"],
              "rho": best["rho"], "gate": gate,
              "record_belief": bool(args.record_belief)}
    print("mode", args.mode, "tau", tau, "params", json.dumps(params), flush=True)

    out = Path(args.out_dir) if args.out_dir else RES / f"10b_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    tasks = manifest["tasks"]
    if args.task_file:
        wanted = {x.strip() for x in Path(args.task_file).read_text(
            encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
        tasks = [t for t in tasks if t["episode_id"] in wanted]
    done = {}
    for eid, rec in load_journal(out).items():
        if rec.get("mode") != args.mode:
            raise RuntimeError(f"journal entry {eid} belongs to another mode")
        if (out / "episodes" / f"{eid.replace(':', '_')}.json").exists():
            done[eid] = rec
    pending = [t for t in tasks if t["episode_id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"tasks {len(tasks)} | done {len(done)} | pending {len(pending)}", flush=True)
    if not pending:
        return 0
    atomic_json(out / "eval_config.json", {
        "mode": args.mode, "manifest_sha": manifest_sha, "calibration_sha": cal_sha,
        "tau": tau, "params": params, "step_cap": args.step_cap,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    fe = FrozenEncoder(args.device)
    encoder = matching.Encoder(args.device)
    rasters, cands = RasterStore(), CandidateStore()
    if args.mode == "c1":
        matcher = C1Matcher(args.device)
    else:
        matcher = LearnedMatcher(args.mode,
                                 str(RES10A / args.mode / "checkpoints" / "best.pt"),
                                 args.device)
    goal_cache: dict = {}
    goal_feat_cache: dict = {}
    episodes = out / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    journal = out / "journal.part00.jsonl"
    started = time.time()
    n = 0
    with journal.open("a", encoding="utf-8") as jf:
        for i, task in enumerate(pending, 1):
            task = dict(task, manifest_sha=manifest_sha)
            rec = run_episode(task, matcher, encoder, fe, rasters, cands,
                              goal_cache, goal_feat_cache, params, args.mode,
                              args.step_cap)
            atomic_json(episodes / f"{rec['episode_id'].replace(':', '_')}.json", rec)
            jf.write(json.dumps(record_summary(rec)) + "\n")
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
    print(f"finished {n} episodes in {(time.time() - started) / 60:.2f} min "
          f"| journal {len(merged)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
