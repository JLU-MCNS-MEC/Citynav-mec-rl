# Experiment 02 live results (500-step budget)

- updated: 2026-10-29 15:55:22
- completed trajectories: **8**
- expected per run: 24; runs: 2 / 4 started

| run | done | SR | NE mean (m) | stop rate | entered 20m | left after entry | mean steps | actions (top 3) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| val_seen_small_roundA | 4/24 | 0.0% | 2210.5 | 0.0% | 25.0% | 25.0% | 500.0 | MoveForward 94%, GoDown 6%, Stop 0% |
| val_seen_transformer_m_roundA | 4/24 | 0.0% | 2208.0 | 0.0% | 25.0% | 25.0% | 500.0 | MoveForward 93%, GoDown 7%, Stop 0% |

## Per-trajectory results (completion order)

| # | time | run | map | episode | steps | stopped | success | NE (m) | min dist (m) | entered 20m | left after entry | actions top 3 |
|---:|---|---|---|---|---:|---|---|---:|---:|---|---|---|
| 1 | 15:10:12 | val_seen_small_roundA | cambridge_block_4 | 64427fa3b743 | 500 | False | False | 2337.9 | 112.2 | False | False | MoveForward 484, GoDown 16 |
| 2 | 15:10:36 | val_seen_transformer_m_roundA | cambridge_block_4 | 64427fa3b743 | 500 | False | False | 2348.0 | 113.9 | False | False | MoveForward 486, GoDown 14 |
| 3 | 15:28:01 | val_seen_transformer_m_roundA | cambridge_block_18 | c6c3220f65c8 | 500 | False | False | 2223.4 | 12.5 | True | True | MoveForward 467, GoDown 33 |
| 4 | 15:28:25 | val_seen_small_roundA | cambridge_block_18 | c6c3220f65c8 | 500 | False | False | 2223.4 | 12.5 | True | True | MoveForward 467, GoDown 33 |
| 5 | 15:30:46 | val_seen_small_roundA | cambridge_block_33 | 314a19429de1 | 500 | False | False | 2039.7 | 38.0 | False | False | MoveForward 465, GoDown 35 |
| 6 | 15:31:10 | val_seen_transformer_m_roundA | cambridge_block_33 | 314a19429de1 | 500 | False | False | 2034.6 | 36.3 | False | False | MoveForward 464, GoDown 36 |
| 7 | 15:54:18 | val_seen_small_roundA | birmingham_block_1 | 68894f04f021 | 500 | False | False | 2241.0 | 195.7 | False | False | MoveForward 455, GoDown 45 |
| 8 | 15:55:22 | val_seen_transformer_m_roundA | birmingham_block_1 | 68894f04f021 | 500 | False | False | 2226.0 | 195.0 | False | False | MoveForward 452, GoDown 48 |

Full per-step traces are in `runs/<run>/*.jsonl`. Test-Unseen is not used.
