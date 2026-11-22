#!/usr/bin/env bash
# Stage 16: DQN and DDPG with the Stage-12 scheme-D reward, serial.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP16="$(cd "$HERE/.." && pwd)"
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
LOG="$HERE/logs/stage12_reward.log"
for algo in dqn ddpg; do
  out="$HERE/runs/${algo}/seed42"
  mkdir -p "$out"
  t0=$(date +%s)
  echo "[$(date '+%F %T')] START $algo (reward=stage12, more)" | tee -a "$LOG"
  STAGE16_TORCH_THREADS=4 "$PY" "$EXP16/p3/train16.py" \
      --algorithm "$algo" --episodes 4000 --seed 42 --gamma 0.99 \
      --reward stage12 --out-dir "$out" > "$out/train.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] TRAIN_END $algo rc=$rc elapsed=$(( $(date +%s)-t0 ))s" | tee -a "$LOG"
  if [ "$rc" -eq 0 ] && [ -f "$out/final.pt" ]; then
    echo "[$(date '+%F %T')] EVAL_START $algo" | tee -a "$LOG"
    STAGE16_TORCH_THREADS=4 "$PY" "$EXP16/p4/eval_agent.py" \
        --ckpt "$out/final.pt" --algorithm "$algo" --out "$out/eval" \
        > "$out/eval.log" 2>&1
    echo "[$(date '+%F %T')] EVAL_END $algo rc=$?" | tee -a "$LOG"
  fi
done
echo "[$(date '+%F %T')] STAGE12_MORE_DONE" | tee -a "$LOG"
