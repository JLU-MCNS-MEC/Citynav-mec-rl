#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
D=stage17d/p1
$PY stage17c/train17c.py --algorithm ppo --episodes 500 --eval-every 50 --out "$D/ppo" > logs/p1_ppo.log 2>&1 &
$PY stage17c/train17c.py --algorithm sac --episodes 500 --eval-every 50 --out "$D/sac" > logs/p1_sac.log 2>&1 &
wait
echo "TRAIN_DONE $(date)"
$PY stage17c/evaluate17.py --baselines random,min_latency,max_quality --out "$D/baselines_dev.json" > logs/p1_baselines.log 2>&1
echo "BASELINES_DONE $(date)"
echo P1_DONE
