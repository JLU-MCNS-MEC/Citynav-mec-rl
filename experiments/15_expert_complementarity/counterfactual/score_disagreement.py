#!/usr/bin/env python3
"""Score every snapshot with all five experts (one step, no state mutation)
to measure action agreement, then select 100 representative states.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from branch_rollout import Ctx, KEYS, decide  # noqa: E402
from snapshot_io import load_snapshot, restore_belief  # noqa: E402
import eval_val_seen as ev  # noqa: E402
from collections import deque  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default=str(HERE / "snapshots_val"))
    ap.add_argument("--out", default=str(HERE / "snapshot_expert_scores.csv"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    snap_dir = Path(args.snapshots)
    idx = pd.read_csv(snap_dir / "snapshot_index.csv")
    ctx = Ctx(args.device)
    rows = []
    for i, r in idx.iterrows():
        s = load_snapshot(Path(r["snapshot"]))
        raster, lib, goal_l2, gdesc = ctx.env(s.map, s.target)
        base_belief = restore_belief(s)
        base_hv = deque([np.asarray(v, np.float32) for v in s.hist_vis], maxlen=ev.HISTORY_LEN)
        base_hp = deque([tuple(map(float, p)) for p in s.hist_pose], maxlen=ev.HISTORY_LEN)
        base_ha = deque([int(a) for a in s.hist_act], maxlen=ev.HISTORY_LEN)
        acts, confs = {}, {}
        for key in KEYS:
            belief = copy.deepcopy(base_belief) if base_belief is not None else None
            hv, hp, ha = (deque(base_hv, maxlen=ev.HISTORY_LEN),
                          deque(base_hp, maxlen=ev.HISTORY_LEN),
                          deque(base_ha, maxlen=ev.HISTORY_LEN))
            a, c, _, _ = decide(ctx, key, raster, goal_l2, gdesc, belief, hv, hp, ha,
                                np.asarray(s.pos, np.float64), float(s.yaw),
                                int(s.step), bool(s.empty_cands))
            acts[key] = a
            confs[key] = c
        rows.append({
            "snapshot": r["snapshot"], "episode_id": r["episode_id"],
            "length_group": r["length_group"], "step": int(r["step"]),
            "dist_to_target": float(r["dist_to_target"]),
            **{f"act_{k}": v for k, v in acts.items()},
            **{f"conf_{k}": v for k, v in confs.items()},
            "n_distinct_actions": len(set(acts.values())),
            "all_agree": len(set(acts.values())) == 1,
        })
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(idx)}]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(json.dumps({
        "snapshots": len(df),
        "all_agree": int(df.all_agree.sum()),
        "disagree": int((~df.all_agree).sum()),
        "distinct_action_counts": df.n_distinct_actions.value_counts().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
