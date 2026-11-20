#!/usr/bin/env bash
# When the Large continuation finishes, automatically run its Val-Seen
# closed-loop evaluation with the same normalization as training.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
PY_EVAL=/mnt/newhome/venvs/citynav-gpu/bin/python
export CITYNAV_DATA=/home/20261024_1/Workspace/DATA
MANIFEST="$REPO/experiments/06_closed_loop_success_rate/results/seed42/manifest.json"
TASKS=/home/jlu-mcns/Workspace/zhangty/task_status/val_seen_ids.txt
NORM="$HERE/normalization_statistics.json"
OUT="$HERE/eval/large/closed_loop"
mkdir -p "$OUT" "$HERE/logs"

D="$HERE/runs/large/seed42/DONE.json"
until [ -f "$D" ] && python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if int(d.get('global_step', 0)) > 19606 else 1)
" "$D" 2>/dev/null; do sleep 30; done
echo "[$(date '+%F %T')] Large training DONE"
cat "$D"

for attempt in 1 2; do
  echo "[$(date '+%F %T')] Large closed-loop eval attempt $attempt"
  "$PY_EVAL" "$HERE/eval_val_seen_norm.py" \
      --model large \
      --ckpt "$HERE/runs/large/seed42/checkpoints/best.pt" \
      --manifest "$MANIFEST" \
      --task-file "$TASKS" \
      --out-dir "$OUT" \
      --step-cap 200 --device cuda \
      --norm-file "$NORM" \
      > "$HERE/logs/eval_large.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] Large eval rc=$rc"
  if [ -f "$OUT/val_seen_summary.json" ]; then break; fi
  sleep 60
done
echo "[$(date '+%F %T')] POST_LARGE_CHAIN_DONE"
