#!/bin/bash
# Stage 14-R1 queue: Small -> Medium -> Transformer-S -> Transformer-M -> Large.
# Per model: train -> verify best.pt -> Val-Seen 2498 closed loop -> report.
# Resume-safe: last.pt resumes training; closed-loop journals resume per task.
# Launch detached:  setsid bash run_stage14_queue.sh >> logs/queue.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export CITYNAV_DATA=${CITYNAV_DATA:-/home/20261024_1/Workspace/DATA}
PY_TRAIN=${PY_TRAIN:-/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python}
PY_EVAL=${PY_EVAL:-/mnt/newhome/venvs/citynav-gpu/bin/python}
MANIFEST="$REPO/experiments/06_closed_loop_success_rate/results/seed42/manifest.json"
TASKS=/home/jlu-mcns/Workspace/zhangty/task_status/val_seen_ids.txt
LOG="$HERE/logs"
mkdir -p "$LOG" "$HERE/figures" "$HERE/results"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/queue.log"; }
key_to_name() {
  case "$1" in
    small) echo Small;; medium) echo Medium;; ts) echo Transformer-S;;
    tm) echo Transformer-M;; large) echo Large;;
  esac
}

for f in "$PY_TRAIN" "$PY_EVAL" "$MANIFEST" "$TASKS" "$CITYNAV_DATA"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 1; }
done

log "queue start (pid $$) git=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
echo $$ > "$HERE/logs/queue.pid"
echo "$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" > "$HERE/logs/train_git_sha.txt"
"$PY_EVAL" "$HERE/progress_report.py" >> "$LOG/report.log" 2>&1 || true

# progress report refresh every 5 min while the queue is alive
(
  while kill -0 $$ 2>/dev/null; do
    "$PY_EVAL" "$HERE/progress_report.py" >> "$LOG/report.log" 2>&1 || true
    sleep 300
  done
) &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null; log "queue exiting"' EXIT

for key in small medium ts tm large; do
  name=$(key_to_name "$key")
  ckpt="$HERE/checkpoints/$name/best.pt"
  last="$HERE/checkpoints/$name/last.pt"
  rdir="$HERE/results/$name"
  cloop="$rdir/closed_loop"

  if pgrep -f "train_expert.py --model $key" >/dev/null; then
    log "$key: already training; abort duplicate launch"; exit 1
  fi
  if pgrep -f "eval_val_seen.py .*--model $key" >/dev/null; then
    log "$key: closed-loop already running; abort duplicate launch"; exit 1
  fi

  # ---- 1) training ----
  if [ -f "$ckpt" ]; then
    log "$key: best.pt exists, skip training"
  else
    resume=""
    [ -f "$last" ] && resume="--resume"
    log "$key: train start ${resume:-from-scratch} (log logs/train_${key}.log)"
    "$PY_TRAIN" "$HERE/train_expert.py" --model "$key" --seed 42 \
      --epochs 20 --patience 10 --device cuda $resume \
      >> "$LOG/train_${key}.log" 2>&1
    rc=$?
    log "$key: train rc=$rc"
    if [ $rc -ne 0 ] || [ ! -f "$ckpt" ]; then
      log "$key: TRAIN FAILED, queue stopped"; exit 1
    fi
    "$PY_TRAIN" "$HERE/progress_report.py" >> "$LOG/report.log" 2>&1 || true
  fi

  # ---- 2) strict checkpoint reload ----
  log "$key: verifying best.pt"
  if ! "$PY_EVAL" "$HERE/verify_ckpt.py" --model "$key" --ckpt "$ckpt" \
       >> "$LOG/verify_${key}.log" 2>&1; then
    log "$key: CHECKPOINT VERIFY FAILED, queue stopped"; exit 1
  fi

  # ---- 3) closed-loop Val-Seen 2498 ----
  complete=$("$PY_EVAL" - "$cloop" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "val_seen_summary.json"
try:
    d = json.loads(p.read_text())
    print("yes" if d.get("complete") and d.get("completed_episodes", 0) >= 2498 else "no")
except Exception:
    print("no")
PY
)
  if [ "$complete" = "yes" ]; then
    log "$key: closed-loop already complete, skip"
  else
    log "$key: closed-loop start (2498 tasks)"
    "$PY_EVAL" "$HERE/eval_val_seen.py" --model "$key" --ckpt "$ckpt" \
      --manifest "$MANIFEST" --task-file "$TASKS" \
      --out-dir "$cloop" --step-cap 200 --device cuda \
      >> "$LOG/eval_${key}.log" 2>&1
    rc=$?
    log "$key: closed-loop rc=$rc"
    if [ $rc -ne 0 ]; then
      log "$key: EVAL FAILED, queue stopped"; exit 1
    fi
    n=$("$PY_EVAL" - "$cloop" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "val_seen_summary.json"
try:
    d = json.loads(p.read_text()); print(d.get("completed_episodes", 0))
except Exception:
    print(0)
PY
)
    if [ "$n" -lt 2498 ]; then
      log "$key: INCOMPLETE closed loop ($n/2498), queue stopped"; exit 1
    fi
  fi

  # ---- 4) report ----
  "$PY_EVAL" "$HERE/progress_report.py" >> "$LOG/report.log" 2>&1 || {
    log "$key: report generation failed"; exit 1; }
  log "$key: MILESTONE complete (train + Val-Seen 2498 + report)"
done

log "ALL_MODELS_DONE"
