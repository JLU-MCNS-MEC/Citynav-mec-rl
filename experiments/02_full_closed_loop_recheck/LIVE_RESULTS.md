# Experiment 02 live results (500-step budget)

- updated: 2026-10-29 19:24:18
- completed trajectories: **53**
- expected per run: 24; runs: 4 / 4 started

| run | done | SR | NE mean (m) | stop rate | entered 20m | left after entry | mean steps | actions (top 3) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| val_seen_small_roundA | 24/24 | 0.0% | 2140.0 | 0.0% | 12.5% | 12.5% | 500.0 | MoveForward 92%, GoDown 8%, Stop 0% |
| val_seen_transformer_m_roundA | 24/24 | 0.0% | 2135.6 | 0.0% | 12.5% | 12.5% | 500.0 | MoveForward 92%, GoDown 8%, Stop 0% |
| val_unseen_small_roundA | 3/24 | 0.0% | 2281.0 | 0.0% | 0.0% | 0.0% | 500.0 | MoveForward 93%, GoDown 7%, Stop 0% |
| val_unseen_transformer_m_roundA | 2/24 | 0.0% | 2224.8 | 0.0% | 0.0% | 0.0% | 500.0 | MoveForward 94%, GoDown 6%, Stop 0% |

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
| 25 | 16:32:11 | val_seen_small_roundA | cambridge_block_12 | 5bc36caf0515 | 500 | False | False | 2236.7 | 159.8 | False | False | MoveForward 459, GoDown 41 |
| 26 | 16:32:11 | val_seen_transformer_m_roundA | cambridge_block_23 | f4183efbde45 | 500 | False | False | 2233.8 | 61.3 | False | False | MoveForward 464, GoDown 36 |
| 27 | 16:33:35 | val_seen_transformer_m_roundA | cambridge_block_12 | 5bc36caf0515 | 500 | False | False | 2226.7 | 159.7 | False | False | MoveForward 457, GoDown 43 |
| 28 | 16:33:58 | val_seen_small_roundA | cambridge_block_23 | f4183efbde45 | 500 | False | False | 2238.8 | 61.5 | False | False | MoveForward 465, GoDown 35 |
| 29 | 16:45:42 | val_seen_transformer_m_roundA | cambridge_block_25 | 99869f7cf230 | 500 | False | False | 2367.8 | 49.7 | False | False | MoveForward 482, GoDown 18 |
| 30 | 16:47:26 | val_seen_transformer_m_roundA | cambridge_block_7 | 823a850ded5a | 500 | False | False | 2051.2 | 155.9 | False | False | MoveForward 468, GoDown 32 |
| 31 | 16:47:50 | val_seen_small_roundA | cambridge_block_25 | 99869f7cf230 | 500 | False | False | 2282.4 | 23.7 | False | False | MoveForward 465, GoDown 35 |
| 32 | 16:48:30 | val_seen_small_roundA | cambridge_block_7 | 823a850ded5a | 500 | False | False | 2061.2 | 155.9 | False | False | MoveForward 470, GoDown 30 |
| 33 | 16:59:16 | val_seen_transformer_m_roundA | birmingham_block_9 | aaa7b0d53aa3 | 500 | False | False | 1974.8 | 210.7 | False | False | MoveForward 436, GoDown 64 |
| 34 | 17:01:00 | val_seen_small_roundA | birmingham_block_9 | aaa7b0d53aa3 | 500 | False | False | 1979.7 | 210.7 | False | False | MoveForward 437, GoDown 63 |
| 35 | 17:16:29 | val_seen_transformer_m_roundA | cambridge_block_20 | 790a917bcb29 | 500 | False | False | 2021.7 | 32.3 | False | False | MoveForward 465, GoDown 35 |
| 36 | 17:17:52 | val_seen_small_roundA | cambridge_block_20 | 790a917bcb29 | 500 | False | False | 2026.7 | 32.4 | False | False | MoveForward 466, GoDown 34 |
| 37 | 17:21:36 | val_seen_small_roundA | cambridge_block_6 | f60d6658b501 | 500 | False | False | 2138.9 | 148.6 | False | False | MoveForward 449, GoDown 51 |
| 38 | 17:21:59 | val_seen_transformer_m_roundA | cambridge_block_6 | f60d6658b501 | 500 | False | False | 2114.0 | 149.3 | False | False | MoveForward 444, GoDown 56 |
| 39 | 17:33:02 | val_seen_transformer_m_roundA | birmingham_block_9 | 9463ec1cc8f9 | 500 | False | False | 2101.3 | 64.3 | False | False | MoveForward 444, GoDown 56 |
| 40 | 17:33:45 | val_seen_small_roundA | birmingham_block_9 | 9463ec1cc8f9 | 500 | False | False | 2111.3 | 64.4 | False | False | MoveForward 446, GoDown 54 |
| 41 | 17:36:05 | val_seen_small_roundA | cambridge_block_13 | f47866f51014 | 500 | False | False | 2289.3 | 39.3 | False | False | MoveForward 461, GoDown 39 |
| 42 | 17:37:28 | val_seen_transformer_m_roundA | cambridge_block_13 | f47866f51014 | 500 | False | False | 2284.3 | 39.4 | False | False | MoveForward 460, GoDown 40 |
| 43 | 17:39:11 | val_seen_transformer_m_roundA | cambridge_block_9 | 3e3d69644910 | 500 | False | False | 2001.8 | 53.7 | False | False | MoveForward 463, GoDown 37 |
| 44 | 17:39:34 | val_seen_small_roundA | cambridge_block_9 | 3e3d69644910 | 500 | False | False | 2011.9 | 54.9 | False | False | MoveForward 465, GoDown 35 |
| 45 | 18:06:41 | val_seen_small_roundA | cambridge_block_13 | 1e50e6983a50 | 500 | False | False | 2050.2 | 49.6 | False | False | MoveForward 452, GoDown 48 |
| 46 | 18:08:04 | val_seen_transformer_m_roundA | cambridge_block_13 | 1e50e6983a50 | 500 | False | False | 2050.2 | 49.6 | False | False | MoveForward 452, GoDown 48 |
| 47 | 18:08:27 | val_seen_transformer_m_roundA | cambridge_block_9 | 2330c6735d1f | 500 | False | False | 2227.5 | 5.1 | True | True | MoveForward 450, GoDown 50 |
| 48 | 18:08:47 | val_seen_small_roundA | cambridge_block_9 | 2330c6735d1f | 500 | False | False | 2237.5 | 8.3 | True | True | MoveForward 452, GoDown 48 |
| 49 | 19:19:24 | val_unseen_small_roundA | cambridge_block_10 | b8f07a313bd7 | 500 | False | False | 2472.5 | 71.5 | False | False | MoveForward 485, GoDown 15 |
| 50 | 19:21:08 | val_unseen_transformer_m_roundA | cambridge_block_10 | b8f07a313bd7 | 500 | False | False | 2477.5 | 72.2 | False | False | MoveForward 486, GoDown 14 |
| 51 | 19:23:14 | val_unseen_small_roundA | birmingham_block_5 | 4f7c86342b43 | 500 | False | False | 1977.0 | 142.8 | False | False | MoveForward 455, GoDown 45 |
| 52 | 19:23:58 | val_unseen_transformer_m_roundA | birmingham_block_5 | 4f7c86342b43 | 500 | False | False | 1972.0 | 142.7 | False | False | MoveForward 454, GoDown 46 |
| 53 | 19:24:18 | val_unseen_small_roundA | birmingham_block_5 | 0783b3d1a3d0 | 500 | False | False | 2393.5 | 149.5 | False | False | MoveForward 450, GoDown 50 |

Full per-step traces are in `runs/<run>/*.jsonl`. Test-Unseen is not used.
