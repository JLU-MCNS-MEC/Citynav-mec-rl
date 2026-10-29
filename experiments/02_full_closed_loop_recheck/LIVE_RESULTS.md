# Experiment 02 live results (500-step budget)

- updated: 2026-10-29 16:30:45
- completed trajectories: **24**
- expected per run: 24; runs: 2 / 4 started

| run | done | SR | NE mean (m) | stop rate | entered 20m | left after entry | mean steps | actions (top 3) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| val_seen_small_roundA | 12/24 | 0.0% | 2141.2 | 0.0% | 16.7% | 16.7% | 500.0 | MoveForward 92%, GoDown 8%, Stop 0% |
| val_seen_transformer_m_roundA | 12/24 | 0.0% | 2133.3 | 0.0% | 16.7% | 16.7% | 500.0 | MoveForward 92%, GoDown 8%, Stop 0% |

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
| 9 | 16:01:08 | val_seen_transformer_m_roundA | cambridge_block_18 | cced5c0de6ac | 500 | False | False | 2156.6 | 84.5 | False | False | MoveForward 447, GoDown 53 |
| 10 | 16:01:33 | val_seen_small_roundA | cambridge_block_18 | cced5c0de6ac | 500 | False | False | 2166.6 | 84.7 | False | False | MoveForward 449, GoDown 51 |
| 11 | 16:01:53 | val_seen_small_roundA | cambridge_block_33 | 290f7cad7aed | 500 | False | False | 1972.2 | 63.3 | False | False | MoveForward 466, GoDown 34 |
| 12 | 16:02:33 | val_seen_transformer_m_roundA | cambridge_block_33 | 290f7cad7aed | 500 | False | False | 1957.2 | 62.9 | False | False | MoveForward 463, GoDown 37 |
| 13 | 16:07:38 | val_seen_transformer_m_roundA | birmingham_block_3 | 9ebfc3a95d6a | 500 | False | False | 2178.6 | 84.2 | False | False | MoveForward 448, GoDown 52 |
| 14 | 16:09:10 | val_seen_small_roundA | birmingham_block_3 | 9ebfc3a95d6a | 500 | False | False | 2183.7 | 84.4 | False | False | MoveForward 449, GoDown 51 |
| 15 | 16:09:53 | val_seen_small_roundA | birmingham_block_10 | d33b71d3064b | 500 | False | False | 1881.6 | 23.6 | False | False | MoveForward 436, GoDown 64 |
| 16 | 16:11:34 | val_seen_transformer_m_roundA | birmingham_block_10 | d33b71d3064b | 500 | False | False | 1871.6 | 22.1 | False | False | MoveForward 434, GoDown 66 |
| 17 | 16:13:39 | val_seen_small_roundA | cambridge_block_4 | 9f4b8a4e9bbe | 500 | False | False | 2267.2 | 88.5 | False | False | MoveForward 475, GoDown 25 |
| 18 | 16:14:02 | val_seen_transformer_m_roundA | cambridge_block_4 | 9f4b8a4e9bbe | 500 | False | False | 2262.2 | 87.9 | False | False | MoveForward 474, GoDown 26 |
| 19 | 16:23:26 | val_seen_small_roundA | birmingham_block_12 | 1cfd09de78f4 | 500 | False | False | 2255.1 | 7.1 | True | True | MoveForward 451, GoDown 49 |
| 20 | 16:24:50 | val_seen_transformer_m_roundA | birmingham_block_12 | 1cfd09de78f4 | 500 | False | False | 2255.1 | 7.1 | True | True | MoveForward 451, GoDown 49 |
| 21 | 16:25:53 | val_seen_transformer_m_roundA | cambridge_block_21 | 9dfe032ef575 | 500 | False | False | 2111.1 | 81.2 | False | False | MoveForward 470, GoDown 30 |
| 22 | 16:27:17 | val_seen_small_roundA | cambridge_block_21 | 9dfe032ef575 | 500 | False | False | 2136.2 | 85.1 | False | False | MoveForward 475, GoDown 25 |
| 23 | 16:29:42 | val_seen_small_roundA | cambridge_block_6 | 264a27d57345 | 500 | False | False | 1990.0 | 236.0 | False | False | MoveForward 448, GoDown 52 |
| 24 | 16:30:45 | val_seen_transformer_m_roundA | cambridge_block_6 | 264a27d57345 | 500 | False | False | 1975.1 | 236.1 | False | False | MoveForward 445, GoDown 55 |

Full per-step traces are in `runs/<run>/*.jsonl`. Test-Unseen is not used.
