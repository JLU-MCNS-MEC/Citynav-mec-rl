#!/usr/bin/env python3
"""10C recovery-state features: C1 train-seen rollouts -> training cache.

Reads the 10C C1 rollouts (run_10b --mode c1 --record-belief on the 22,002
Train-Seen tasks), and rebuilds the SAME per-episode cache format as
build_features_10c.py (teacher): pooled layer4 fp16, belief 10-d, match 5-d,
states/actions, target_emb (C1 goal desc).  The belief/match arrays are
reconstructed from the recorded per-step belief internals (deterministic:
the same causal loop the rollout ran), so recovery states carry the exact
09C semantics.  Sampling: <=12 states per episode, 5-bin stratified, global
subsample to --states (50k).  Labels = the C1 action (DAgger-style recovery
component; no class weighting).
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
RES = HERE.parent / "results"
CACHE = RES / "cache"
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))

import matching  # noqa: E402
from dataset_v2 import ENCODE_CHUNK, FrozenEncoder, RasterStore  # noqa: E402

ROLLOUTS = RES / "c1_rollouts"
CACHE_VERSION = "10c-c1belief-v1"
ENCODER_ID = "frozen resnet50 IMAGENET1K_V2 (layer4 fp16 history + layer2 fp32 C1 match)"
SEED_SAMPLE = 20261031
BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, float("inf"))]


def bin_of(d: float) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= d < hi:
            return i
    return len(BINS) - 1


def pick_states(dists: list[float], cap: int) -> list[int]:
    by_bin = {b: [] for b in range(5)}
    for t, d in enumerate(dists):
        by_bin[bin_of(d)].append(t)
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


def belief_from_step(step: dict, bs: dict) -> np.ndarray:
    """Reconstruct the 10-d belief state from a recorded rollout step."""
    x, y, yaw = step["x"], step["y"], step["yaw"]
    est = np.asarray(step["estimate"])
    vec = est - np.asarray([x, y])
    fwd = float(vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
    lat = float(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw))
    dist = float(np.linalg.norm(vec))
    bearing = math.atan2(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw),
                         vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
    means = np.asarray(bs["means"])
    vars_ = np.asarray(bs["vars"])
    weights = np.asarray(bs["weights"])
    unc = float(math.sqrt(np.mean(vars_)))
    w = weights / max(weights.sum(), 1e-12)
    ent = float(-(w * np.log(w + 1e-12)).sum() / math.log(max(len(w), 2)))
    since = float(step["step"] - int(bs["last_update"]))
    z = np.asarray(step["visual_z"])
    dx, dy = float(z[0] - x), float(z[1] - y)
    return np.asarray([fwd, lat, dist, bearing, step["visual_score"],
                       unc, ent, since, dx, dy], np.float32)


def assemble(states_cap: int) -> int:
    root = CACHE / "recovery"
    entries = {}
    for p in sorted(root.glob("journal.part*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                e = json.loads(line)
                if e.get("status") == "done":
                    entries[e["key"]] = e
    # full sample pass over the rollout episodes (need true_dist per step)
    ep_dists: dict[str, list[float]] = {}
    for c in range(4):
        for f in sorted((ROLLOUTS / f"chunk{c}" / "episodes").glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            ep_dists[f.stem] = [float(s["true_dist_m"]) for s in rec["trajectory"]]
    missing = [k for k in ep_dists if k not in entries]
    if missing:
        raise SystemExit(f"recovery: {len(missing)} episodes missing (first {missing[:5]})")
    rows = []
    rng = np.random.default_rng(SEED_SAMPLE)
    total_states = int(sum(len(v) for v in ep_dists.values()))
    p = states_cap / max(total_states, 1)
    for key, dists in sorted(ep_dists.items()):
        t_len = len(dists)
        k = max(1, min(t_len, int(round(t_len * p))))
        # random-phase stride: time-uniform in expectation, no forced endpoints
        u = rng.random()
        chosen = sorted({min(t_len - 1, int((i + u) * t_len / k))
                         for i in range(k)})
        for t in chosen:
            rows.append({"key": key, "t": t, "dist_m": round(dists[t], 2),
                         "dist_bin": bin_of(dists[t]), "file": entries[key]["file"],
                         "map": entries[key]["map"]})
    if states_cap and len(rows) > states_cap:
        take = sorted(rng.choice(len(rows), size=states_cap, replace=False))
        rows = [rows[i] for i in take]
    rows.sort(key=lambda r: (r["key"], r["t"]))
    for i, r in enumerate(rows):
        r["state_id"] = f"recovery:{i:07d}"
    # labels (C1 actions) + x/y/yaw for on-the-fly crops come from the npz
    out = root / "states.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    labels, bins_c = Counter(), Counter()
    for r in rows:
        with np.load(root / "episodes" / f"{r['key']}.npz") as z:
            labels[int(z["actions"][r["t"]])] += 1
        bins_c[r["dist_bin"]] += 1
    (root / "sample_stats.json").write_text(json.dumps({
        "states": len(rows), "episodes": len({r["key"] for r in rows}),
        "per_bin": {str(k): v for k, v in sorted(bins_c.items())},
        "action_dist": {str(k): v for k, v in sorted(labels.items())},
        "seed": SEED_SAMPLE, "source": "c1 train-seen rollouts (DAgger-style)"},
        indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json.loads((root / "sample_stats.json").read_text()), indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, default=-1)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--states", type=int, default=50000,
                        help="global recovery-state cap")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")
    if args.assemble:
        return assemble(args.states)
    if args.worker_id < 0:
        raise SystemExit("pass --worker-id 0..n-1 (or --assemble)")
    manifest = {t["episode_id"]: t for t in
                json.loads((RES / "train_manifest.json").read_text(
                    encoding="utf-8"))["tasks"]}
    ep_files = sorted(p for c in range(4)
                      for p in (ROLLOUTS / f"chunk{c}" / "episodes").glob("*.json"))
    mine = ep_files[args.worker_id::args.n_workers]
    out_root = CACHE / "recovery" / "episodes"
    out_root.mkdir(parents=True, exist_ok=True)
    jpath = CACHE / "recovery" / f"journal.part{args.worker_id:02d}.jsonl"
    done = set()
    if jpath.exists():
        with jpath.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["key"])
                except json.JSONDecodeError:
                    continue
    rasters = RasterStore()
    enc4 = FrozenEncoder(args.device)
    matcher_enc = matching.Encoder(args.device)
    t0 = time.time()
    n = 0
    with jpath.open("a", encoding="utf-8") as jf:
        for f in mine:
            key = f.stem
            if key in done:
                continue
            rec = json.loads(f.read_text(encoding="utf-8"))
            task = manifest[rec["episode_id"]]
            traj = rec["trajectory"]
            if not traj:
                continue
            steps = np.asarray([[s["x"], s["y"], s["yaw"]] for s in traj], np.float32)
            acts = np.asarray([s["action"] for s in traj], np.int8)
            goal_img = matching.prepare_goal(rasters.get(task["map"]),
                                             float(task["target_x"]),
                                             float(task["target_y"]))
            goal_l2 = matcher_enc.features(goal_img[None], ("layer2",))["layer2"][0]
            imgs = [rasters.get(task["map"]).heading_crop(float(x), float(y), float(yaw))
                    for x, y, yaw in steps]
            pooled = []
            for i in range(0, len(imgs), ENCODE_CHUNK):
                maps = enc4.spatial_maps(imgs[i:i + ENCODE_CHUNK]).transpose(0, 2, 3, 1)
                vec = maps.mean(axis=(1, 2))
                vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
                pooled.append(vec)
            pooled = (np.concatenate(pooled) if pooled
                      else np.zeros((0, 2048), np.float32)).astype(np.float16)
            belief = np.stack([belief_from_step(s, s["belief_state"]) for s in traj])
            match = np.stack([[float(s["visual_z"][0] - s["x"]),
                               float(s["visual_z"][1] - s["y"]),
                               float(s["visual_score"]), float(s["visual_margin"]),
                               0.0] for s in traj], axis=0).astype(np.float32)
            path = out_root / f"{key}.npz"
            tmp = path.with_name(f".{key}.{os.getpid()}.tmp.npz")
            np.savez(tmp, meta=json.dumps({"key": key, "map": task["map"],
                                           "cache_version": CACHE_VERSION,
                                           "encoder": ENCODER_ID,
                                           "source": "c1_train_rollout"}),
                     pooled=pooled, belief=belief, match=match,
                     states=steps, actions=acts, target_emb=goal_l2)
            os.replace(tmp, path)
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            jf.write(json.dumps({"key": key, "status": "done",
                                 "file": f"episodes/{key}.npz",
                                 "bytes": path.stat().st_size, "sha256": h,
                                 "states": len(traj), "map": task["map"],
                                 "cache_version": CACHE_VERSION}) + "\n")
            jf.flush()
            n += 1
            if n % 100 == 0:
                print(f"w{args.worker_id} {n}/{len(mine)} | {time.time() - t0:.0f}s",
                      flush=True)
    print(f"worker {args.worker_id} finished {n} episodes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
