#!/usr/bin/env bash
# Stage 16-P4: train four DRL agents, two in parallel, evaluate each as it finishes.
# Reward A (pure cumulative E2E), seed 42, 4000 episodes each.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP16="$(cd "$HERE/.." && pwd)"
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
mkdir -p "$HERE/logs" "$HERE/runs"
LOG="$HERE/logs/training.log"

run_alg() {
  local algo="$1"
  local out="$HERE/runs/${algo}/seed42"
  mkdir -p "$out"
  local t0=$(date +%s)
  echo "[$(date '+%F %T')] START $algo" | tee -a "$LOG"
  STAGE16_TORCH_THREADS=4 "$PY" "$EXP16/p3/train16.py" \
      --algorithm "$algo" --episodes 4000 --seed 42 \
      --out-dir "$out" > "$out/train.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] TRAIN_END $algo rc=$rc elapsed=$(( $(date +%s)-t0 ))s" | tee -a "$LOG"
  if [ "$rc" -eq 0 ] && [ -f "$out/final.pt" ]; then
    echo "[$(date '+%F %T')] EVAL_START $algo" | tee -a "$LOG"
    STAGE16_TORCH_THREADS=4 "$PY" "$HERE/eval_agent.py" \
        --ckpt "$out/final.pt" --algorithm "$algo" --out "$out/eval" \
        > "$out/eval.log" 2>&1
    echo "[$(date '+%F %T')] EVAL_END $algo rc=$?" | tee -a "$LOG"
  fi
}

echo "[$(date '+%F %T')] P4 start (reward A, seed 42, 4000 episodes, SERIAL)" | tee -a "$LOG"
# Serial because third-party GPU services already occupy ~11.7 GiB.
for algo in ppo dqn td3 sac; do
  run_alg "$algo"
done
echo "[$(date '+%F %T')] P4_ALL_DONE" | tee -a "$LOG"
