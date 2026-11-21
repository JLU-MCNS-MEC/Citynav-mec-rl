#!/usr/bin/env python3
"""Select 100 representative snapshots for the counterfactual branch run.

Stratifies by the original TS action at the snapshot step and by length_group,
caps two states per episode for diversity, and (optionally) prioritises states
where the five experts disagree.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default=str(HERE / "snapshots_val"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=str(HERE / "selected_100.csv"))
    args = ap.parse_args()
    snap_dir = Path(args.snapshots)
    idx = pd.read_csv(snap_dir / "snapshot_index.csv")

    rows = []
    for _, r in idx.iterrows():
        meta = json.loads(Path(r["snapshot"]).with_suffix(".json").read_text())
        rows.append({
            "snapshot": r["snapshot"], "episode_id": r["episode_id"],
            "length_group": r["length_group"], "step": int(r["step"]),
            "next_action": int(meta.get("next_action", -1)),
            "next_action_name": meta.get("next_action_name", ""),
            "dist_to_target": float(r["dist_to_target"]),
            "map": r["map"],
        })
    df = pd.DataFrame(rows)
    df = df[df.next_action >= 0]

    # stratification targets: emphasise turn/stop states where disagreement lives
    target = {0: 20, 1: 30, 2: 25, 3: 25}   # 100
    picked, per_ep = [], defaultdict(int)
    for act, want in target.items():
        pool = df[df.next_action == act].sample(frac=1.0, random_state=42)
        # prefer length-group coverage within each action
        by_len = {g: s for g, s in pool.groupby("length_group")}
        quota = max(1, want // max(len(by_len), 1))
        for g, sub in sorted(by_len.items()):
            n = 0
            for _, r in sub.iterrows():
                if n >= quota or len(picked) >= args.n:
                    break
                if per_ep[r.episode_id] >= 2:
                    continue
                picked.append(r.to_dict())
                per_ep[r.episode_id] += 1
                n += 1
        # fill remaining from this action
        for _, r in pool.iterrows():
            if len(picked) >= args.n:
                break
            if sum(1 for x in picked if x["next_action"] == act) >= want:
                break
            if per_ep[r.episode_id] >= 2:
                continue
            if any(x["snapshot"] == r["snapshot"] for x in picked):
                continue
            picked.append(r.to_dict())
            per_ep[r.episode_id] += 1

    out = pd.DataFrame(picked[: args.n])
    out.to_csv(args.out, index=False)
    print(json.dumps({
        "selected": len(out),
        "by_action": out.next_action_name.value_counts().to_dict(),
        "by_length": out.length_group.value_counts().to_dict(),
        "episodes": out.episode_id.nunique(),
    }, indent=2))


if __name__ == "__main__":
    main()
