# Experiment 02 live results (500-step budget)

- updated: 2026-10-29 15:10:12
- completed trajectories: **1**
- expected per run: 24; runs: 1 / 4 started

| run | done | SR | NE mean (m) | stop rate | entered 20m | left after entry | mean steps | actions (top 3) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| val_seen_small_roundA | 1/24 | 0.0% | 2337.9 | 0.0% | 0.0% | 0.0% | 500.0 | MoveForward 97%, GoDown 3%, Stop 0% |

## Per-trajectory results (completion order)

| # | time | run | map | episode | steps | stopped | success | NE (m) | min dist (m) | entered 20m | left after entry | actions top 3 |
|---:|---|---|---|---|---:|---|---|---:|---:|---|---|---|
| 1 | 15:10:12 | val_seen_small_roundA | cambridge_block_4 | 64427fa3b743 | 500 | False | False | 2337.9 | 112.2 | False | False | MoveForward 484, GoDown 16 |

Full per-step traces are in `runs/<run>/*.jsonl`. Test-Unseen is not used.
