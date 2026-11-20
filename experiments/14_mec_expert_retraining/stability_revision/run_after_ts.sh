#!/usr/bin/env bash
# Orchestrate the post-TS chain requested by the user:
#   1) wait for TS 20-epoch training to finish
#   2) start TM continuation (2-epoch ckpt -> 20 epochs)
#   3) once TM training is up and stable, start the TS Val-Seen closed loop
#   4) after TM finishes, retry the TS eval if it did not complete (e.g. OOM)
#   5) start Large continuation (gradient checkpointing)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
PY_EVAL=/mnt/newhome/venvs/citynav-gpu/bin/python
export CITYNAV_DATA=/home/20261024_1/Workspace/DATA

MANIFEST="$REPO/experiments/06_closed_loop_success_rate/results/seed42/manifest.json"
TASKS=/home/jlu-mcns/Workspace/zhangty/task_status/val_seen_ids.txt
NORM="$HERE/normalization_statistics.json"
EVAL_OUT="$HERE/eval/ts/closed_loop"
mkdir -p "$EVAL_OUT" "$HERE/logs"

run_ts_eval() {
  local log="$1"
  echo "[$(date '+%F %T')] TS closed-loop eval start -> $log"
  "$PY_EVAL" "$HERE/eval_val_seen_norm.py" \
      --model ts \
      --ckpt "$HERE/runs/ts/seed42/checkpoints/best.pt" \
      --manifest "$MANIFEST" \
      --task-file "$TASKS" \
      --out-dir "$EVAL_OUT" \
      --step-cap 200 --device cuda \
      --norm-file "$NORM" \
      > "$log" 2>&1
  echo "[$(date '+%F %T')] TS eval rc=$? (log $log)"
}

# 1) wait for TS
D="$HERE/runs/ts/seed42/DONE.json"
# NOTE: a DONE.json from the 2-epoch stability phase already exists
# (global_step=19608). Wait for the continuation to advance past it.
until [ -f "$D" ] && python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if int(d.get('global_step', 0)) > 19608 else 1)
" "$D" 2>/dev/null; do sleep 30; done
echo "[$(date '+%F %T')] TS 20-epoch training DONE"
cat "$D"

# 2) start TM continuation in the background
bash "$HERE/run_model_continuation.sh" tm &
TM_PID=$!

# wait until TM training is actually running (params line + a few steps)
for _ in $(seq 1 180); do
  if grep -q 'params=' "$HERE/logs/train_tm_continuation.log" 2>/dev/null; then break; fi
  kill -0 "$TM_PID" 2>/dev/null || { echo "TM continuation died before start"; wait "$TM_PID"; exit 1; }
  sleep 5
done
sleep 30

# 3) TS eval concurrently with TM training (eval is batch=1, ~2-4 GB)
run_ts_eval "$HERE/logs/eval_ts.log" &
EV_PID=$!

# 4) wait for TM
wait "$TM_PID"; TM_RC=$?
echo "[$(date '+%F %T')] TM continuation finished rc=$TM_RC"
if [ "$TM_RC" -ne 0 ]; then
  echo "TM continuation failed; not starting Large"
  exit "$TM_RC"
fi
wait "$EV_PID" 2>/dev/null || true
if [ ! -f "$EVAL_OUT/val_seen_summary.json" ]; then
  echo "[$(date '+%F %T')] TS eval did not complete during TM training; retrying on free GPU"
  run_ts_eval "$HERE/logs/eval_ts_retry.log"
fi

# 5) Large continuation
bash "$HERE/run_model_continuation.sh" large
echo "[$(date '+%F %T')] ALL CONTINUATIONS DONE"
