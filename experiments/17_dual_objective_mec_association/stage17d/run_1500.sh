#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
D=stage17d/p2
run() { $PY stage17c/train17c.py --algorithm "$1" --episodes 1500 --eval-every 100 --out "$D/$1" > "logs/p2_$1.log" 2>&1; }
run dqn & run ddpg & wait; echo "P2_PAIR1 $(date)"
run td3 & run sac & wait; echo "P2_PAIR2 $(date)"
run ppo; echo "P2_PPO $(date)"
$PY stage17d/plot_ma100.py --root "$D" --out "$D/figures" > logs/p2_plot.log 2>&1
$PY stage17c/evaluate17.py --baselines random,min_latency,max_quality --out "$D/baselines_dev.json" > logs/p2_baselines.log 2>&1
echo "P2_DONE $(date)"
