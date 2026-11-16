# Stage 12 — baseline and RL association results

### Baselines (Val-Seen 2,498)

| group | policy | sr | e2e_mean_ms | e2e_p95_ms | inf_mean_ms | comm_mean_ms | encode_mean_ms |
|---|---|---|---|---|---|---|---|
| two_homo | random | 48.84% | 62.49 | 80.63 | 5.30 | 51.75 | 5.45 |
| two_homo | max_rate | 44.96% | 37.35 | 45.81 | 5.78 | 26.12 | 5.45 |
| two_homo | max_quality | 53.56% | 43.46 | 50.77 | 10.00 | 28.01 | 5.45 |
| two_homo | oracle | 60.33% | 36.65 | 44.75 | 3.21 | 27.99 | 5.45 |
| five_homo | random | 41.83% | 64.67 | 82.57 | 7.25 | 51.97 | 5.45 |
| five_homo | max_rate | 41.07% | 41.02 | 52.88 | 9.32 | 26.25 | 5.45 |
| five_homo | max_quality | 53.56% | 50.82 | 71.26 | 10.00 | 35.37 | 5.45 |
| five_homo | oracle | 76.02% | 41.84 | 55.16 | 4.68 | 31.71 | 5.45 |
| two_hetero | random | 48.84% | 62.30 | 80.32 | 5.11 | 51.75 | 5.45 |
| two_hetero | max_rate | 44.96% | 37.43 | 47.84 | 5.87 | 26.12 | 5.45 |
| two_hetero | max_quality | 53.56% | 41.65 | 50.53 | 7.04 | 29.16 | 5.45 |
| two_hetero | oracle | 60.33% | 36.03 | 44.70 | 2.54 | 28.03 | 5.45 |
| five_hetero | random | 41.83% | 63.71 | 81.55 | 6.30 | 51.97 | 5.45 |
| five_hetero | max_rate | 41.07% | 40.36 | 51.29 | 8.66 | 26.25 | 5.45 |
| five_hetero | max_quality | 53.56% | 53.24 | 75.32 | 11.01 | 36.78 | 5.45 |
| five_hetero | oracle | 76.02% | 41.03 | 53.90 | 3.82 | 31.76 | 5.45 |

### RL (Val-Seen 2,498)

| group | policy | sr | e2e_mean_ms | e2e_p95_ms | inf_mean_ms | comm_mean_ms | encode_mean_ms |
|---|---|---|---|---|---|---|---|
| two_homo | ppo-s42 | 36.15% | 37.79 | 49.11 | 1.67 | 30.67 | 5.45 |
| two_homo | ppo-s43 | 36.63% | 37.54 | 49.92 | 1.04 | 31.05 | 5.45 |
| two_homo | dqn-s42 | 46.60% | 54.57 | 77.09 | 6.69 | 42.44 | 5.45 |
| two_homo | dqn-s43 | 43.59% | 51.55 | 73.06 | 4.33 | 41.77 | 5.45 |
| five_homo | ppo-s42 | 31.35% | 41.34 | 56.65 | 2.37 | 33.52 | 5.45 |
| five_homo | ppo-s43 | 34.83% | 40.12 | 52.09 | 2.98 | 31.68 | 5.45 |
| five_homo | dqn-s42 | 37.95% | 58.06 | 80.24 | 5.15 | 47.46 | 5.45 |
| five_homo | dqn-s43 | 34.87% | 55.61 | 77.11 | 4.34 | 45.82 | 5.45 |
| two_hetero | ppo-s42 | 39.63% | 37.94 | 50.84 | 1.90 | 30.59 | 5.45 |
| two_hetero | ppo-s43 | 34.75% | 36.26 | 48.31 | 0.47 | 30.33 | 5.45 |
| two_hetero | dqn-s42 | 42.35% | 51.50 | 75.88 | 2.57 | 43.48 | 5.45 |
| two_hetero | dqn-s43 | 43.63% | 53.04 | 78.18 | 4.25 | 43.34 | 5.45 |
| five_hetero | ppo-s42 | 37.55% | 40.88 | 55.24 | 3.52 | 31.91 | 5.45 |
| five_hetero | ppo-s43 | 34.91% | 40.49 | 54.41 | 3.97 | 31.06 | 5.45 |
| five_hetero | dqn-s42 | 33.71% | 57.50 | 78.96 | 4.76 | 47.29 | 5.45 |
| five_hetero | dqn-s43 | 39.83% | 57.86 | 80.39 | 5.95 | 46.46 | 5.45 |

### Expert step share

| group | policy | Small | Medium | Transformer-S | Transformer-M | Large |
|---|---|---|---|---|---|---|
| two_homo | random | 49.9% | 0.0% | 0.0% | 50.1% | 0.0% |
| two_homo | max_rate | 44.6% | 0.0% | 0.0% | 55.4% | 0.0% |
| two_homo | max_quality | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| two_homo | oracle | 65.0% | 0.0% | 0.0% | 35.0% | 0.0% |
| five_homo | random | 19.8% | 20.1% | 20.1% | 19.9% | 20.1% |
| five_homo | max_rate | 15.0% | 12.9% | 17.2% | 25.8% | 29.0% |
| five_homo | max_quality | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| five_homo | oracle | 24.6% | 28.7% | 20.4% | 16.6% | 9.6% |
| two_hetero | random | 49.9% | 0.0% | 0.0% | 50.1% | 0.0% |
| two_hetero | max_rate | 44.6% | 0.0% | 0.0% | 55.4% | 0.0% |
| two_hetero | max_quality | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| two_hetero | oracle | 62.0% | 0.0% | 0.0% | 38.0% | 0.0% |
| five_hetero | random | 19.8% | 20.1% | 20.1% | 19.9% | 20.1% |
| five_hetero | max_rate | 15.0% | 12.9% | 17.2% | 25.8% | 29.0% |
| five_hetero | max_quality | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| five_hetero | oracle | 23.5% | 27.8% | 20.9% | 12.9% | 14.9% |
| two_homo | ppo-s42 | 86.5% | 0.0% | 0.0% | 13.5% | 0.0% |
| two_homo | ppo-s43 | 92.6% | 0.0% | 0.0% | 7.4% | 0.0% |
| two_homo | dqn-s42 | 35.8% | 0.0% | 0.0% | 64.2% | 0.0% |
| two_homo | dqn-s43 | 59.9% | 0.0% | 0.0% | 40.1% | 0.0% |
| five_homo | ppo-s42 | 39.2% | 34.1% | 20.7% | 1.4% | 4.6% |
| five_homo | ppo-s43 | 37.2% | 32.0% | 10.5% | 20.3% | 0.0% |
| five_homo | dqn-s42 | 15.5% | 30.8% | 21.0% | 25.2% | 7.4% |
| five_homo | dqn-s43 | 14.5% | 39.3% | 27.5% | 10.3% | 8.4% |
| two_hetero | ppo-s42 | 72.3% | 0.0% | 0.0% | 27.7% | 0.0% |
| two_hetero | ppo-s43 | 98.0% | 0.0% | 0.0% | 2.0% | 0.0% |
| two_hetero | dqn-s42 | 68.6% | 0.0% | 0.0% | 31.4% | 0.0% |
| two_hetero | dqn-s43 | 58.5% | 0.0% | 0.0% | 41.5% | 0.0% |
| five_hetero | ppo-s42 | 39.2% | 19.3% | 11.2% | 7.0% | 23.3% |
| five_hetero | ppo-s43 | 47.0% | 21.1% | 13.0% | 12.2% | 6.7% |
| five_hetero | dqn-s42 | 17.6% | 42.4% | 10.6% | 7.2% | 22.2% |
| five_hetero | dqn-s43 | 14.7% | 25.1% | 15.6% | 23.8% | 20.8% |
