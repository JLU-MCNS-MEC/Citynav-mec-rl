#!/usr/bin/env python3
"""10C feature builder: causal C1 belief + history features for MEC training.

For every teacher episode the causal loop reproduces 09C-1 exactly:

* C1 subpixel match (frozen layer2, fp32, batch-1 per step like 09C);
* the 09C ContinuousBelief (candidate prior, Kalman updates, score>=0.875 +
  two-frame agreement<=10 m gate -- no C2);
* per-step outputs: 10-d belief state, 5-d match measurement (dx,dy,score,
  margin,sharpness), 2048-d layer4 pooled visual (fp16, batched at 128 like
  Stage 05 so cache and training kernels match).

Sampling (phase 2, deterministic, seed 20261031): unit = decision state;
per-episode cap 12 with 5-distance-bin stratification (Stage-07 fix: short
episodes are no longer overrepresented per state, far bins get a quota);
then a global uniform subsample to --states states.

Parallel: shard episodes across workers with --worker-id/--n-workers; each
worker appends journal.part{id}.jsonl; assemble (--assemble) verifies SHAs,
builds index.json + states.csv.
"""

from __future__ import annotations

import argparse
import csv
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
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
RES = HERE.parent / "results"
CACHE = RES / "cache"
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))

import matching  # noqa: E402
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import (CANDS, ENCODE_CHUNK, CandidateStore, FrozenEncoder,  # noqa: E402
                        RasterStore, history_block, load_split)
from pipeline import TARGET_HALF_M  # noqa: E402

TEACHER = REPO / "experiments/03b_target_directed_teacher/results"
CAL = json.loads((REPO / "experiments/09_online_visual_goal_matching"
                  "/09C_continuous_localization/results/09c_calibration.json")
                 .read_text(encoding="utf-8"))
BEL = json.loads((REPO / "experiments/09_online_visual_goal_matching"
                  "/09C_continuous_localization/results/09c_belief_calibration.json")
                 .read_text(encoding="utf-8"))
BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, float("inf"))]
TAU = 0.875
AGREE_M = 10.0
CAP_PER_EPISODE = 12
SEED_SAMPLE = 20261031

DIMS = {"belief": 10, "match": 5, "target_emb": 512, "hist_vis": 2048}
CACHE_VERSION = "10c-c1belief-v1"
ENCODER_ID = "frozen resnet50 IMAGENET1K_V2 (layer4 fp16 history + layer2 fp32 C1 match)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bin_of(d: float) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= d < hi:
            return i
    return len(BINS) - 1


def pick_states(dists: np.ndarray, cap: int) -> list[int]:
    """Round-robin over 5 distance bins, evenly spaced in each; <= cap."""
    by_bin = {b: [] for b in range(5)}
    for t, d in enumerate(dists):
        by_bin[bin_of(float(d))].append(t)
    quota = max(1, cap // max(len([b for b in by_bin if by_bin[b]]), 1))
    chosen: list[int] = []
    for b in range(5):
        entries = by_bin[b]
        if not entries:
            continue
        if len(entries) <= quota:
            chosen.extend(entries)
        else:
            idx = sorted({round(i * (len(entries) - 1) / (quota - 1)) if quota > 1 else 0
                          for i in range(quota)})
            chosen.extend(entries[i] for i in idx)
    return sorted(chosen)[:cap]


def causal_episode(ep, raster, lib, matcher_enc, enc4) -> dict:
    """Run the 09C-1 causal loop over one teacher episode; returns per-step
    arrays (belief 10-d, match 5-d, pooled layer4 fp16) for ALL states."""
    states, actions, target = ep.states, ep.actions, ep.target
    t_len = len(actions)
    goal_img = matching.prepare_goal(raster, float(target[0]), float(target[1]))
    goal_l2 = matcher_enc.features(goal_img[None], ("layer2",))["layer2"][0]
    tvec = enc4.embed([raster.north_crop(float(target[0]), float(target[1]),
                                         TARGET_HALF_M)])[0].astype(np.float32)
    sims = lib.embeddings @ tvec
    belief = ContinuousBelief(lib.positions.astype(np.float64), sims.astype(np.float64),
                              0.1, k=8, sigma_prior_m=12.0)
    imgs = []
    for x, y, yaw in states:
        imgs.append(raster.heading_crop(float(x), float(y), float(yaw)))
    # layer4 pooled in 128-chunks (cache == training kernel, Stage-05 rule)
    pooled = []
    for i in range(0, len(imgs), ENCODE_CHUNK):
        maps = enc4.spatial_maps(imgs[i:i + ENCODE_CHUNK]).transpose(0, 2, 3, 1)
        vec = maps.mean(axis=(1, 2))
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        pooled.append(vec)
    pooled = (np.concatenate(pooled) if pooled else np.zeros((0, 2048), np.float32)
              ).astype(np.float16)

    belief_feat = np.zeros((t_len, 10), np.float32)
    match_feat = np.zeros((t_len, 5), np.float32)
    for t in range(t_len):
        x, y, yaw = float(states[t, 0]), float(states[t, 1]), float(states[t, 2])
        obs = matching.prepare_observation(raster, x, y, yaw)
        of = matcher_enc.features(obs[None], ("layer2",))["layer2"][0]
        m = subpixel_match(of, goal_l2, 2)
        dx, dy = matching.pixel_to_offset(m["col"], m["row"])
        z = np.array([x + dx, y + dy])
        score, margin = m["score"], m["margin"]
        agree = (belief.last_accept is not None
                 and float(np.linalg.norm(z - belief.last_accept)) <= AGREE_M)
        conf_ok = score >= TAU
        accept = conf_ok and agree
        if conf_ok:
            belief.last_accept = z.copy()
        if accept:
            belief.update(z, CAL["sigma_vis_m"], BEL["best"]["eta"],
                          BEL["best"]["rho"], t)
        est = belief.estimate()
        vec = est - np.asarray([x, y])
        fwd = float(vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
        lat = float(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw))
        dist = float(np.linalg.norm(vec))
        bearing = math.atan2(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw),
                             vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
        unc = float(math.sqrt(np.mean(belief.vars)))
        since = float(t - belief.last_update)
        ent = float(belief.entropy())
        if not math.isfinite(ent):   # single-component mixture: log(1)=0
            ent = 0.0                # -> maximum certainty, entropy 0
        belief_feat[t] = [fwd, lat, dist, bearing, score, unc,
                          ent, since, dx, dy]
        match_feat[t] = [dx, dy, score, margin, m["sharpness"]]
    return {"pooled": pooled, "belief": belief_feat, "match": match_feat,
            "states": states.astype(np.float32), "actions": actions.astype(np.int8),
            "target_emb": goal_l2.astype(np.float32)}


def run_worker(split: str, worker: int, n_workers: int) -> int:
    episodes = load_split(split)
    mine = [i for i in range(len(episodes)) if i % n_workers == worker]
    root = CACHE / split
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    jpath = root / f"journal.part{worker:02d}.jsonl"
    done = set()
    if jpath.exists():
        with jpath.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["key"])
                except json.JSONDecodeError:
                    continue
    rasters, cands = RasterStore(), CandidateStore()
    matcher_enc = matching.Encoder("cuda")
    enc4 = FrozenEncoder("cuda")
    t0 = time.time()
    n = 0
    with jpath.open("a", encoding="utf-8") as jf:
        for idx in mine:
            ep = episodes[idx]
            if ep.key in done:
                continue
            try:
                arr = causal_episode(ep, rasters.get(ep.map), cands.get(ep.map),
                                     matcher_enc, enc4)
            except Exception as e:  # record and continue; assemble will retry
                jf.write(json.dumps({"split": split, "key": int(ep.key),
                                     "status": "error", "error": str(e)[:200]}) + "\n")
                jf.flush()
                continue
            path = root / "episodes" / f"{ep.key}.npz"
            tmp = path.with_name(f".{ep.key}.{os.getpid()}.tmp.npz")
            np.savez(tmp, meta=json.dumps({"key": int(ep.key), "map": ep.map,
                                           "cache_version": CACHE_VERSION,
                                           "encoder": ENCODER_ID}),
                     **arr)
            os.replace(tmp, path)
            jf.write(json.dumps({"split": split, "key": int(ep.key),
                                 "status": "done", "file": f"episodes/{ep.key}.npz",
                                 "bytes": path.stat().st_size,
                                 "sha256": sha256_file(path),
                                 "states": int(len(ep.actions)), "map": ep.map,
                                 "cache_version": CACHE_VERSION,
                                 "encoder": ENCODER_ID,
                                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                     + "\n")
            jf.flush()
            n += 1
            if n % 50 == 0:
                print(f"w{worker} {n}/{len(mine)} | {time.time() - t0:.0f}s", flush=True)
    print(f"worker {worker} finished: {n} episodes in {time.time() - t0:.0f}s", flush=True)
    return 0


def assemble(split: str, val_episodes: int, val_states_per: int,
             global_states: int = 200000) -> int:
    episodes = load_split(split)
    root = CACHE / split
    entries = {}
    for p in sorted((root).glob("journal.part*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                e = json.loads(line)
                if e.get("status") == "done":
                    entries[int(e["key"])] = e
    missing = [ep.key for ep in episodes if ep.key not in entries]
    if missing:
        raise SystemExit(f"{split}: {len(missing)} episodes missing (first {missing[:5]})")
    bad = [k for k, e in entries.items()
           if not (root / e["file"]).exists()
           or (root / e["file"]).stat().st_size != e["bytes"]
           or sha256_file(root / e["file"]) != e["sha256"]]
    if bad:
        raise SystemExit(f"{split}: {len(bad)} entries fail verification (first {bad[:5]})")
    index = {"split": split, "cache_version": CACHE_VERSION, "encoder": ENCODER_ID,
             "dims": DIMS, "cap_per_episode": CAP_PER_EPISODE,
             "episodes": [{"key": int(ep.key), "map": ep.map,
                           "length": int(len(ep.actions)),
                           "file": entries[int(ep.key)]["file"]}
                          for ep in episodes]}
    (root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    # ---- sampling (decision-state units, per-state uniform) ----------------
    # Per-state uniform inclusion (the unit IS the state, not the episode):
    # episode i gets k_i = max(1, round(T_i * p)) evenly spaced states with
    # p = global_target / total_states, then a global uniform subsample pins
    # the exact target size.  Short episodes are NOT overrepresented per
    # state (Stage-07 fix); distance bins follow their natural frequency.
    with (TEACHER / f"teacher_{split}.csv").open(encoding="utf-8") as fh:
        meta_csv = {int(r["index"]): r for r in csv.DictReader(fh)}
    rows = []
    rng = np.random.default_rng(SEED_SAMPLE)
    if split == "val_seen":
        pick_ep = sorted(rng.choice(len(episodes), size=min(val_episodes, len(episodes)),
                                    replace=False))
        n_states_per = val_states_per
        p = None
    else:
        pick_ep = list(range(len(episodes)))
        n_states_per = None
        total_states = int(sum(len(ep.actions) for ep in episodes))
        p = global_states / max(total_states, 1)
    for idx in pick_ep:
        ep = episodes[idx]
        t_len = len(ep.actions)
        if n_states_per is not None:
            k = min(t_len, n_states_per)
        else:
            k = max(1, min(t_len, int(round(t_len * p))))
        # random-phase stride: time-uniform in expectation, no forced endpoints
        u = rng.random()
        chosen = sorted({min(t_len - 1, int((i + u) * t_len / k))
                         for i in range(k)})
        with np.load(root / "episodes" / f"{ep.key}.npz") as z:
            states, actions = z["states"], z["actions"]
            dists = np.linalg.norm(states[:, :2] - ep.target, axis=1)
        for t in chosen:
            rows.append({"split": split, "ep": int(idx), "key": int(ep.key),
                         "t": int(t), "map": ep.map,
                         "length_group": meta_csv[ep.key]["length_group"],
                         "dist_m": round(float(dists[t]), 2),
                         "dist_bin": bin_of(float(dists[t])),
                         "label": int(actions[t]),
                         "file": f"episodes/{ep.key}.npz"})
    print(f"{split}: sampled {len(rows)} states before global cap", flush=True)
    if split == "train_seen" and global_states and len(rows) > global_states:
        take = sorted(rng.choice(len(rows), size=global_states, replace=False))
        rows = [rows[i] for i in take]
        print(f"{split}: global subsample -> {len(rows)}", flush=True)
    rows.sort(key=lambda r: (r["ep"], r["t"]))
    for i, r in enumerate(rows):
        r["state_id"] = f"{split}:{i:07d}"
    out = root / "states.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # ---- audit stats -------------------------------------------------------
    from collections import Counter
    acts = Counter(r["label"] for r in rows)
    bins_c = Counter(r["dist_bin"] for r in rows)
    lgs = Counter(r["length_group"] for r in rows)
    (root / "sample_stats.json").write_text(json.dumps({
        "states": len(rows), "episodes": len({r["ep"] for r in rows}),
        "per_bin": {str(k): v for k, v in sorted(bins_c.items())},
        "per_length_group": dict(lgs),
        "action_dist": {str(k): v for k, v in sorted(acts.items())},
        "seed": SEED_SAMPLE}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json.loads((root / "sample_stats.json").read_text()),
                     indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train_seen", "val_seen"], required=True)
    parser.add_argument("--worker-id", type=int, default=-1)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--val-episodes", type=int, default=400)
    parser.add_argument("--val-states-per", type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")
    if args.assemble:
        return assemble(args.split, args.val_episodes, args.val_states_per)
    if args.worker_id < 0:
        raise SystemExit("pass --worker-id 0..n-1 (or --assemble)")
    return run_worker(args.split, args.worker_id, args.n_workers)


if __name__ == "__main__":
    raise SystemExit(main())
