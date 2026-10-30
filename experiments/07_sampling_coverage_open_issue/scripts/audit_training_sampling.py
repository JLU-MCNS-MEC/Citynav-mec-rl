#!/usr/bin/env python3
"""Reproduce Stage-05's exact sample draws and audit what the model saw.

Stage 05 trained with ``k=8`` decision states per episode per epoch, drawn
without replacement inside each episode, with ``rng = default_rng(seed*1000 +
epoch)`` and a final ``rng.shuffle`` of the pair list.  This script replays the
same RNG calls (no model, no training) and reports:

* supervisor pool vs per-epoch draws vs optimizer steps (three different numbers);
* per-action / per-map / per-length-group / per-distance coverage;
* how often a single STOP state is drawn over 30 epochs;
* repetition: draws per unique state.

Outputs results/training_sampling.json, results/sampling_tables.csv and
figures/sampling_coverage.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))

from dataset_v2 import CACHE  # noqa: E402

OUT = HERE.parent / "results"
FIG = HERE.parent / "figures"
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
GROUPS = ("short", "medium", "long")
DIST_BUCKETS = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 10 ** 9)]
DIST_LABELS = ["0-20", "20-50", "50-100", "100-200", ">200"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train_seen")
    args = parser.parse_args()

    index = json.loads((CACHE / args.split / "index.json").read_text(encoding="utf-8"))
    episodes = index["episodes"]
    lengths = np.asarray([e["length"] for e in episodes], np.int64)
    maps = [e["map"] for e in episodes]
    states_total = int(lengths.sum())
    assert states_total == index["states"]

    teacher = np.load(ROOT / f"03b_target_directed_teacher/results/teacher_{args.split}.npz")
    offsets = teacher["offsets"]
    actions = teacher["actions"].astype(np.int64)
    states = teacher["states"].astype(np.float32)
    targets = teacher["targets"].astype(np.float32)
    assert len(actions) == states_total

    # per-state descriptors
    ep_of_state = np.repeat(np.arange(len(lengths)), lengths)
    state_target = targets[ep_of_state]
    dists = np.linalg.norm(states[:, :2] - state_target, axis=1)
    dist_bucket = np.digitize(dists, [b[1] for b in DIST_BUCKETS[:-1]], right=True)
    act_counts_pool = np.bincount(actions, minlength=4)

    draws = np.zeros(states_total, np.int64)
    per_epoch = []
    steps_total = 0
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        pairs = []
        for e in range(len(lengths)):
            t_len = int(lengths[e])
            kk = min(args.k, t_len)
            for t in rng.choice(t_len, size=kk, replace=False):
                pairs.append((e, int(t)))
        rng.shuffle(pairs)
        batch_index = np.asarray([offsets[e] + t for e, t in pairs], np.int64)
        np.add.at(draws, batch_index, 1)
        n_draws = len(pairs)
        batches = (n_draws + args.batch - 1) // args.batch
        steps_total += batches
        per_epoch.append({"epoch": epoch, "draws": n_draws, "batches": batches,
                          "unique_states": int((np.bincount(batch_index, minlength=states_total) > 0).sum())})

    total_draws = int(draws.sum())
    covered = int((draws > 0).sum())
    covered_once = draws[draws > 0]

    def bucket_sum(mask):
        return {"pool_states": int(mask.sum()), "draws": int(draws[mask].sum()),
                "covered_states": int((draws[mask] > 0).sum()),
                "coverage": float((draws[mask] > 0).sum() / max(mask.sum(), 1)),
                "draw_share": float(draws[mask].sum() / max(total_draws, 1))}

    by_action = {CLASSES[c]: bucket_sum(actions == c) for c in range(4)}
    # length group is not in the cache index; recover it from the teacher CSV
    group_of_ep = {}
    with (ROOT / f"03b_target_directed_teacher/results/teacher_{args.split}.csv").open(
            encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            group_of_ep[int(row["index"])] = row["length_group"]
    # cache order != csv index order; the cache stores the teacher key
    ep_group = np.asarray([group_of_ep[e["key"]] for e in episodes])
    state_group = np.repeat(ep_group, lengths)
    by_group = {g: bucket_sum(state_group == g) for g in GROUPS}

    map_names = sorted(set(maps))
    ep_map = np.asarray(maps)
    state_map = np.repeat(ep_map, lengths)
    by_map = {m: bucket_sum(state_map == m) for m in map_names}
    by_distance = {DIST_LABELS[i]: bucket_sum(dist_bucket == i) for i in range(len(DIST_LABELS))}

    stop_mask = actions == 0
    per_epoch_p_stop = np.asarray([min(args.k, int(t)) / int(t) for t in lengths])
    # per-episode probability that its STOP state is drawn at least once
    p_at_least_once = 1.0 - (1.0 - per_epoch_p_stop) ** args.epochs

    result = {
        "split": args.split, "epochs": args.epochs, "k_per_episode": args.k,
        "batch": args.batch, "seed": args.seed,
        "supervisor_pool": {"episodes": int(len(lengths)), "states": states_total,
                            "actions": {CLASSES[c]: int(act_counts_pool[c]) for c in range(4)}},
        "draws": {"total": total_draws, "per_epoch": total_draws // args.epochs,
                  "optimizer_steps": steps_total, "steps_per_epoch": steps_total // args.epochs,
                  "unique_states_covered": covered,
                  "coverage": covered / states_total,
                  "mean_draws_per_covered_state": float(covered_once.mean()),
                  "max_draws_per_state": int(draws.max()),
                  "draws_per_pool_state": total_draws / states_total},
        "stop": {"pool_states": int(stop_mask.sum()),
                 "draws": int(draws[stop_mask].sum()),
                 "unique_covered": int((draws[stop_mask] > 0).sum()),
                 "coverage": float((draws[stop_mask] > 0).sum() / max(stop_mask.sum(), 1)),
                 "p_sampled_per_epoch_mean": float(np.mean(per_epoch_p_stop)),
                 "p_sampled_at_least_once_30ep_mean": float(np.mean(p_at_least_once)),
                 "p_sampled_at_least_once_30ep_min": float(np.min(p_at_least_once)),
                 "p_sampled_at_least_once_30ep_p10": float(np.percentile(p_at_least_once, 10))},
        "by_action": by_action, "by_group": by_group, "by_map": by_map,
        "by_distance": by_distance, "per_epoch": per_epoch,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training_sampling.json").write_text(json.dumps(result, indent=2) + "\n",
                                                encoding="utf-8")
    with (OUT / "sampling_tables.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scope", "key", "pool_states", "draws", "covered_states",
                         "coverage", "draw_share"])
        for scope, table in (("action", by_action), ("length_group", by_group),
                             ("map", by_map), ("distance", by_distance)):
            for key, v in table.items():
                writer.writerow([scope, key, v["pool_states"], v["draws"],
                                 v["covered_states"], f"{v['coverage']:.6f}",
                                 f"{v['draw_share']:.6f}"])

    # figure: pool share vs draw share, plus coverage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = CLASSES
    pool = [by_action[c]["pool_states"] for c in labels]
    draw = [by_action[c]["draws"] for c in labels]
    axes[0].bar(np.arange(4) - 0.2, np.asarray(pool) / sum(pool), 0.4, label="pool share")
    axes[0].bar(np.arange(4) + 0.2, np.asarray(draw) / sum(draw), 0.4, label="draw share")
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(labels, rotation=20, fontsize=8)
    axes[0].set_title("Action share: supervisor pool vs sampled draws", fontsize=9)
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)
    cov = [by_distance[d]["coverage"] for d in DIST_LABELS]
    axes[1].bar(DIST_LABELS, cov, color="#3b6ea5")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("State coverage after 30 epochs by target distance", fontsize=9)
    axes[1].set_xlabel("distance to target (m)", fontsize=8)
    axes[1].set_ylabel("unique states drawn / pool", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "sampling_coverage.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: result[k] for k in ("supervisor_pool", "draws", "stop")},
                     ensure_ascii=False, indent=1))
    for scope, table in (("action", by_action), ("group", by_group),
                         ("distance", by_distance)):
        print(f"-- {scope} --")
        for key, v in table.items():
            print(f"  {key:10s} pool={v['pool_states']:7d} draws={v['draws']:8d} "
                  f"cover={v['coverage']:.4f} share={v['draw_share']:.4f}")
    print(f"wrote {OUT / 'training_sampling.json'} and figures/sampling_coverage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
