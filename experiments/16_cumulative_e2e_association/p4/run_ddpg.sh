#!/usr/bin/env bash
# Stage 16: DDPG queued after the remaining algorithms finish.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP16="$(cd "$HERE/.." && pwd)"
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
LOG="$HERE/logs/remaining.log"
out="$HERE/runs/ddpg/seed42"
mkdir -p "$out"
echo "[$(date '+%F %T')] START ddpg" | tee -a "$LOG"
STAGE16_TORCH_THREADS=4 "$PY" "$EXP16/p3/train16.py" --algorithm ddpg \
    --episodes 4000 --seed 42 --gamma 0.99 --out-dir "$out" > "$out/train.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] TRAIN_END ddpg rc=$rc" | tee -a "$LOG"
if [ "$rc" -eq 0 ] && [ -f "$out/final.pt" ]; then
  echo "[$(date '+%F %T')] EVAL_START ddpg" | tee -a "$LOG"
  STAGE16_TORCH_THREADS=4 "$PY" "$HERE/eval_agent.py" --ckpt "$out/final.pt" \
      --algorithm ddpg --out "$out/eval" > "$out/eval.log" 2>&1
  echo "[$(date '+%F %T')] EVAL_END ddpg rc=$?" | tee -a "$LOG"
fi
echo "[$(date '+%F %T')] DDPG_DONE" | tee -a "$LOG"
