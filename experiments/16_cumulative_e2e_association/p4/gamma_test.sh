#!/usr/bin/env bash
# Stage 16: small-scale discount-factor comparison (DQN, 200 episodes each).
# No new 4000-episode training.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP16="$(cd "$HERE/.." && pwd)"
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
mkdir -p "$HERE/gamma/logs" "$HERE/gamma"
LOG="$HERE/gamma/logs/gamma.log"
: > "$LOG"
for g in 0.99 0.995 1.0; do
  out="$HERE/gamma/g${g}"
  rm -rf "$out"; mkdir -p "$out"
  echo "[$(date '+%F %T')] START gamma=$g" | tee -a "$LOG"
  STAGE16_TORCH_THREADS=4 "$PY" "$EXP16/p3/train16.py" \
      --algorithm dqn --episodes 200 --seed 42 --gamma "$g" \
      --out-dir "$out" > "$out/train.log" 2>&1
  echo "[$(date '+%F %T')] END gamma=$g rc=$?" | tee -a "$LOG"
done
echo "[$(date '+%F %T')] GAMMA_DONE" | tee -a "$LOG"
