#!/usr/bin/env bash
# Stage 16: remaining DRL algorithms, serial, same config as DQN.
# TD3 -> SAC -> PPO (PPO numerics fixed).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP16="$(cd "$HERE/.." && pwd)"
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
mkdir -p "$HERE/logs" "$HERE/runs"
LOG="$HERE/logs/remaining.log"

run_alg() {
  local algo="$1"
  local out="$HERE/runs/${algo}/seed42"
  mkdir -p "$out"
  local t0=$(date +%s)
  echo "[$(date '+%F %T')] START $algo" | tee -a "$LOG"
  STAGE16_TORCH_THREADS=4 "$PY" "$EXP16/p3/train16.py" \
      --algorithm "$algo" --episodes 4000 --seed 42 --gamma 0.99 \
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

echo "[$(date '+%F %T')] REMAINING start (TD3 -> SAC -> PPO; reward A, gamma 0.99)" | tee -a "$LOG"
for algo in td3 sac ppo; do
  run_alg "$algo"
done
echo "[$(date '+%F %T')] REMAINING_ALL_DONE" | tee -a "$LOG"
