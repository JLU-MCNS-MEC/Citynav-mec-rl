#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
D=stage17d/p0
run() { $PY stage17c/train17c.py --algorithm "$1" --episodes 100 --eval-every 25 --out "$D/$1" > "logs/p0_$1.log" 2>&1; }
run dqn & run ddpg & wait; echo "PAIR1 $(date)"
run td3 & run sac & wait; echo "PAIR2 $(date)"
run ppo; echo "PPO $(date)"
$PY stage17c/evaluate17.py --baselines random,min_latency,max_quality --out "$D/baselines_dev.json" > logs/p0_baselines.log 2>&1
echo "BASELINES $(date)"
echo P0_DONE
