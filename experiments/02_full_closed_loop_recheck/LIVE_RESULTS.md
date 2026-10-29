# Experiment 02 live results (500-step budget)

- updated: 2026-10-29 15:30:46
- completed trajectories: **5**
- expected per run: 24; runs: 2 / 4 started

| run | done | SR | NE mean (m) | stop rate | entered 20m | left after entry | mean steps | actions (top 3) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| val_seen_small_roundA | 3/24 | 0.0% | 2200.3 | 0.0% | 33.3% | 33.3% | 500.0 | MoveForward 94%, GoDown 6%, Stop 0% |
| val_seen_transformer_m_roundA | 2/24 | 0.0% | 2285.7 | 0.0% | 50.0% | 50.0% | 500.0 | MoveForward 95%, GoDown 5%, Stop 0% |

## Per-trajectory results (completion order)

| # | time | run | map | episode | steps | stopped | success | NE (m) | min dist (m) | entered 20m | left after entry | actions top 3 |
|---:|---|---|---|---|---:|---|---|---:|---:|---|---|---|
| 1 | 15:10:12 | val_seen_small_roundA | cambridge_block_4 | 64427fa3b743 | 500 | False | False | 2337.9 | 112.2 | False | False | MoveForward 484, GoDown 16 |
| 2 | 15:10:36 | val_seen_transformer_m_roundA | cambridge_block_4 | 64427fa3b743 | 500 | False | False | 2348.0 | 113.9 | False | False | MoveForward 486, GoDown 14 |
| 3 | 15:28:01 | val_seen_transformer_m_roundA | cambridge_block_18 | c6c3220f65c8 | 500 | False | False | 2223.4 | 12.5 | True | True | MoveForward 467, GoDown 33 |
| 4 | 15:28:25 | val_seen_small_roundA | cambridge_block_18 | c6c3220f65c8 | 500 | False | False | 2223.4 | 12.5 | True | True | MoveForward 467, GoDown 33 |
| 5 | 15:30:46 | val_seen_small_roundA | cambridge_block_33 | 314a19429de1 | 500 | False | False | 2039.7 | 38.0 | False | False | MoveForward 465, GoDown 35 |

Full per-step traces are in `runs/<run>/*.jsonl`. Test-Unseen is not used.
