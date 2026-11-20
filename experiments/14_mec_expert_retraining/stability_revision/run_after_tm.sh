#!/usr/bin/env bash
# Post-TM chain: when TM finishes 20 epochs, start the TM closed-loop eval
# and launch the Large continuation concurrently. If either side hits OOM,
# fall back to running them sequentially.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
PY_EVAL=/mnt/newhome/venvs/citynav-gpu/bin/python
export CITYNAV_DATA=/home/20261024_1/Workspace/DATA
export PYTORCH_ALLOC_CONF=expandable_segments:True

MANIFEST="$REPO/experiments/06_closed_loop_success_rate/results/seed42/manifest.json"
TASKS=/home/jlu-mcns/Workspace/zhangty/task_status/val_seen_ids.txt
NORM="$HERE/normalization_statistics.json"
TM_EVAL_OUT="$HERE/eval/tm/closed_loop"
mkdir -p "$TM_EVAL_OUT" "$HERE/logs"

run_tm_eval() {
  local log="$1"
  echo "[$(date '+%F %T')] TM closed-loop eval start -> $log"
  "$PY_EVAL" "$HERE/eval_val_seen_norm.py" \
      --model tm \
      --ckpt "$HERE/runs/tm/seed42/checkpoints/best.pt" \
      --manifest "$MANIFEST" \
      --task-file "$TASKS" \
      --out-dir "$TM_EVAL_OUT" \
      --step-cap 200 --device cuda \
      --norm-file "$NORM" \
      > "$log" 2>&1
  echo "[$(date '+%F %T')] TM eval rc=$? (log $log)"
}

D="$HERE/runs/tm/seed42/DONE.json"
until [ -f "$D" ] && python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if int(d.get('global_step', 0)) > 19608 else 1)
" "$D" 2>/dev/null; do sleep 30; done
echo "[$(date '+%F %T')] TM 20-epoch training DONE"
cat "$D"

# TM closed-loop eval starts first (small footprint), then Large training.
run_tm_eval "$HERE/logs/eval_tm.log" &
EV_PID=$!
sleep 20

# Large continuation; if it fails (e.g. OOM because eval shares the GPU),
# wait for the eval to finish and retry once.
if ! bash "$HERE/run_model_continuation.sh" large; then
  echo "[$(date '+%F %T')] Large continuation failed; waiting for TM eval then retrying"
  wait "$EV_PID" 2>/dev/null || true
  bash "$HERE/run_model_continuation.sh" large
fi

wait "$EV_PID" 2>/dev/null || true
if [ ! -f "$TM_EVAL_OUT/val_seen_summary.json" ]; then
  echo "[$(date '+%F %T')] TM eval did not complete during Large training; retrying on free GPU"
  run_tm_eval "$HERE/logs/eval_tm_retry.log"
fi
echo "[$(date '+%F %T')] POST_TM_CHAIN_DONE"
