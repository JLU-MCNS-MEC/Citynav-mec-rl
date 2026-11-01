#!/usr/bin/env python3
"""11B feature builder: causal C1 replay over the Stage 03 four-action teacher.

Same C1 / 09C computation as build_features_10c (causal_episode is imported
verbatim: subpixel layer2 match, 09C ContinuousBelief, layer4 pooled history),
but the replay trajectories are the Stage 03 fixed-50 m four-action teacher:

  train_seen: four_action_labels.npz, restricted to the 11A final pool
              (18,521 episodes / 1,279,882 states, NO cap, NO subsample);
  val_seen:   Stage-03-style four-action teacher built by
              build_val_teacher_11b.py (VALIDATION ONLY), fixed subset of
              400 episodes x 4 evenly spaced states.

No 03B trajectories, no C1 closed-loop states, no controller labels anywhere
in this pipeline (doc 2.1).

Parallel: shard episodes across workers with --worker-id/--n-workers; each
worker appends journal.part{id}.jsonl; assemble (--assemble) verifies SHAs,
builds states.csv + index.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
RES = HERE.parent / "results"
CACHE = RES / "cache"
S03R = REPO / "experiments/03_fixed50m_four_action_teacher/results"
MANIFEST_EP = (REPO / "experiments/11_teacher_geometry_ablation/11A_teacher_dataset_audit"
               "/results/teacher_episode_manifest.csv")
VAL_TEACHER = CACHE / "val_teacher"
sys.path.insert(0, str(REPO / "experiments/10_goal_query_mec_training/10C_mec_input_dataset/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))

from dataset_v2 import (CANDS, ENCODE_CHUNK, CandidateStore,  # noqa: E402
                        Episode, FrozenEncoder, RasterStore)
from build_features_10c import BEL, CAL, TAU, AGREE_M  # noqa: E402
import matching  # noqa: E402
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from pipeline import TARGET_HALF_M  # noqa: E402

import math  # noqa: E402


def causal_episode(ep, raster, lib, matcher_enc, enc4) -> dict:
    """09C-1 causal loop over one teacher episode (verbatim from
    build_features_10c.causal_episode) plus one v5 extension: per-step top-K
    landmark state ``landmarks`` (T,K,3) = [mean_x, mean_y, weight] of the
    09C belief mixture.  v4 training ignores this key; v5 uses it for the
    multi-landmark geometry tokens.  All v4 arrays are computed exactly as
    in 10C."""
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
    landmark_feat = np.zeros((t_len, 8, 3), np.float32)   # v5 extension
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
        n_k = int(belief.means.shape[0])   # mixture prunes low-weight components
        landmark_feat[t, :n_k, :2] = belief.means.astype(np.float32)
        landmark_feat[t, :n_k, 2] = belief.weights.astype(np.float32)
    return {"pooled": pooled, "belief": belief_feat, "match": match_feat,
            "states": states.astype(np.float32), "actions": actions.astype(np.int8),
            "target_emb": goal_l2.astype(np.float32),
            "landmarks": landmark_feat}

CACHE_VERSION = "11b-fouraction-v1"
SEED_SAMPLE = 20261101


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def train_episodes(worker_id: int, n_workers: int) -> list[Episode]:
    """Stage 03 four-action teacher, restricted to the 11A final pool.

    Big arrays are read ONCE into locals (indexing an NpzFile re-decompresses
    the whole member on every access -- see dataset_v2.load_split); each
    worker only constructs the Episode objects of its own shard.
    """
    z = np.load(S03R / "four_action_labels.npz")
    with MANIFEST_EP.open(encoding="utf-8") as fh:
        final = sorted(int(r["index"]) for r in csv.DictReader(fh))
    states_all = z["states"]
    actions_all = z["actions"]
    targets_all = z["target_xy"]
    maps_all = z["maps"]
    so = z["state_offsets"]
    mine = [i for i in final if i % n_workers == worker_id]
    episodes = []
    for i in mine:
        episodes.append(Episode(
            key=i, map=str(maps_all[i]),
            states=states_all[so[i]:so[i + 1]].astype(np.float32),
            actions=actions_all[so[i]:so[i + 1]].astype(np.int64),
            target=targets_all[i].astype(np.float32)))
    return episodes


def val_episodes(val_episodes_n: int) -> list[Episode]:
    z = np.load(VAL_TEACHER / "teacher_val_seen.npz")
    keys = [int(k) for k in z["keys"]]
    off = z["offsets"]
    states_all, actions_all, targets_all = z["states"], z["actions"], z["targets"]
    with (VAL_TEACHER / "teacher_val_seen.csv").open(encoding="utf-8") as fh:
        maps = {int(r["index"]): r["map"] for r in csv.DictReader(fh)}
    episodes = [Episode(key=k, map=maps[k],
                        states=states_all[off[i]:off[i + 1]].astype(np.float32),
                        actions=actions_all[off[i]:off[i + 1]].astype(np.int64),
                        target=targets_all[i].astype(np.float32))
                for i, k in enumerate(keys)]
    rng = np.random.default_rng(SEED_SAMPLE)
    pick = sorted(rng.choice(len(episodes), size=min(val_episodes_n, len(episodes)),
                             replace=False))
    return [episodes[i] for i in pick]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train_seen", "val_seen"])
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--val-episodes", type=int, default=400)
    parser.add_argument("--val-states-per", type=int, default=4)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = CACHE / args.split
    if args.assemble:
        return assemble(args.split)

    if args.split == "train_seen":
        episodes = train_episodes(args.worker_id, args.n_workers)
        mine = list(range(len(episodes)))   # already sharded
    else:
        episodes = val_episodes(args.val_episodes)
        mine = [i for i in range(len(episodes)) if i % args.n_workers == args.worker_id]
    root.mkdir(parents=True, exist_ok=True)
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    journal = root / f"journal.part{args.worker_id:02d}.jsonl"
    done = set()
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["key"])

    rasters = RasterStore()
    cands = CandidateStore()
    matcher_enc = matching.Encoder(args.device)
    enc4 = FrozenEncoder(args.device)
    t0 = time.time()
    n = 0
    for idx in mine:
        ep = episodes[idx]
        if ep.key in done:
            continue
        arr = causal_episode(ep, rasters.get(ep.map), cands.get(ep.map),
                             matcher_enc, enc4)
        path = root / "episodes" / f"{ep.key}.npz"
        np.savez_compressed(path, **arr)
        journal.open("a", encoding="utf-8").write(json.dumps(
            {"key": ep.key, "file": f"episodes/{ep.key}.npz",
             "sha256": sha256_file(path), "steps": int(len(arr["states"])),
             "status": "done"}) + "\n")
        n += 1
        if n % 20 == 0:
            print(f"worker {args.worker_id}: {n} eps in {time.time() - t0:.0f}s",
                  flush=True)
    print(f"worker {args.worker_id} finished: {n} episodes in {time.time() - t0:.0f}s",
          flush=True)
    return 0


def assemble(split: str) -> int:
    root = CACHE / split
    entries: dict[int, dict] = {}
    for jf in sorted(root.glob("journal.part*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            entries[int(e["key"])] = e
    episodes = train_episodes(0, 1) if split == "train_seen" else val_episodes(400)
    missing = [ep.key for ep in episodes if ep.key not in entries]
    if missing:
        raise SystemExit(f"{split}: {len(missing)} episodes missing "
                         f"(first {missing[:5]})")
    if split == "train_seen":
        # ALL decision states, no cap / no subsample (doc 4.3)
        rows = []
        z = np.load(S03R / "four_action_labels.npz")
        actions_all = z["actions"]          # read once (NpzFile re-decompresses)
        so = z["state_offsets"]
        state_id = 0
        for ep in episodes:
            n_steps = int(so[ep.key + 1] - so[ep.key])
            for t in range(n_steps):
                rows.append({"split": split, "ep": ep.key, "key": ep.key, "t": t,
                             "map": ep.map, "label": int(actions_all[so[ep.key] + t]),
                             "file": entries[ep.key]["file"], "state_id": state_id})
                state_id += 1
    else:
        rows = []
        state_id = 0
        for ep in episodes:
            n_steps = int(len(ep.states))
            idx = sorted({round(i * (n_steps - 1) / (4 - 1)) for i in range(4)})
            for t in idx:
                rows.append({"split": split, "ep": ep.key, "key": ep.key, "t": t,
                             "map": ep.map, "label": int(ep.actions[t]),
                             "file": entries[ep.key]["file"], "state_id": state_id})
                state_id += 1
    with (root / "states.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "ep", "key", "t", "map",
                                           "label", "file", "state_id"])
        w.writeheader()
        w.writerows(rows)
    (root / "index.json").write_text(json.dumps({
        "cache_version": CACHE_VERSION,
        "split": split,
        "n_episodes": len(episodes),
        "n_states": len(rows),
        "seed_sample": SEED_SAMPLE,
        "source": ("four_action_labels.npz (11A final pool)" if split == "train_seen"
                   else "teacher_val_seen.npz (validation only)"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"{split}: assembled {len(rows)} states from {len(episodes)} episodes",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
