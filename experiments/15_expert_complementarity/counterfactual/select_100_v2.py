#!/usr/bin/env python3
"""Select 100 branch states from expert scores: 60 disagreement-first states
plus 40 agreement states, stratified by length_group and near-target distance,
max 2 per episode.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


PER_EP = 2
def pick(pool: pd.DataFrame, n: int, per_ep: dict, picked: list) -> None:
    for _, r in pool.iterrows():
        if len(picked) >= n:
            return
        if per_ep[r.episode_id] >= PER_EP:
            continue
        if any(x["snapshot"] == r["snapshot"] for x in picked):
            continue
        picked.append(r.to_dict())
        per_ep[r.episode_id] += 1
        if len(picked) >= n:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(HERE / "snapshot_expert_scores.csv"))
    ap.add_argument("--out", default=str(HERE / "selected_100.csv"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per-ep", type=int, default=2)
    args = ap.parse_args()
    global PER_EP
    PER_EP = args.per_ep
    N = args.n
    df = pd.read_csv(args.scores)
    df = df[df.act_ts >= 0].copy()
    per_ep = defaultdict(int)
    picked = []

    # 1) disagreement-first, stratified by length_group and near-target
    dis = df[~df.all_agree].copy()
    dis["near"] = dis.dist_to_target < 80
    strata = list(dis.groupby(["length_group", "near"]))
    dis_target = min(len(dis), max(1, int(N * 0.4)))
    for (lg, near), sub in strata:
        quota = max(1, dis_target // max(len(strata), 1))
        sub = sub.sample(frac=1.0, random_state=42)
        pick(sub, min(len(picked) + quota, dis_target), per_ep, picked)

    # 2) fill disagreement quota
    if len(picked) < dis_target:
        pick(dis.sample(frac=1.0, random_state=7), dis_target, per_ep, picked)

    # 3) agreement states to reach 100, stratified by action and length
    agr = df[df.all_agree].copy()
    for (lg, act), sub in agr.groupby(["length_group", "act_ts"]):
        if len(picked) >= N:
            break
        quota = max(1, (N - dis_target) // max(len(agr.groupby(["length_group", "act_ts"])), 1))
        sub = sub.sample(frac=1.0, random_state=11)
        pick(sub, min(len(picked) + quota, N), per_ep, picked)
    if len(picked) < N:
        pick(agr.sample(frac=1.0, random_state=13), N, per_ep, picked)

    out = pd.DataFrame(picked[:N])
    out.to_csv(args.out, index=False)
    print(json.dumps({
        "selected": len(out),
        "disagreement": int((~out.all_agree).sum()) if "all_agree" in out else None,
        "by_ts_action": out.act_ts.value_counts().to_dict(),
        "by_length": out.length_group.value_counts().to_dict(),
        "near_target_lt80": int((out.dist_to_target < 80).sum()),
        "episodes": out.episode_id.nunique(),
    }, indent=2))


if __name__ == "__main__":
    main()
