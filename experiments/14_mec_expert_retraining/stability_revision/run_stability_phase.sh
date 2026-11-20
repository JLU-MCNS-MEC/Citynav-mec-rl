#!/usr/bin/env bash
# Stage 14-R2 stability phase: run TS -> TM -> Large, at most 2 epochs each,
# one model at a time. Stops immediately if a model fails (non-zero rc).
# Does NOT launch full 20-epoch training and does NOT run Val-Seen evaluation.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/logs"
mkdir -p "$LOG"
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
export CITYNAV_DATA=/home/20261024_1/Workspace/DATA

for key in ts tm large; do
  echo "[$(date '+%F %T')] STABILITY START $key" | tee -a "$LOG/stability_phase.log"
  "$PY" "$HERE/train_stable.py" \
      --model "$key" --epochs 2 \
      --out-dir "$HERE/runs" \
      --norm-file "$HERE/normalization_statistics.json" \
      2>&1 | tee "$LOG/train_${key}_stability.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "[$(date '+%F %T')] STABILITY FAILED $key rc=$rc" | tee -a "$LOG/stability_phase.log"
    exit "$rc"
  fi
  echo "[$(date '+%F %T')] STABILITY DONE $key" | tee -a "$LOG/stability_phase.log"
done
echo "[$(date '+%F %T')] ALL_STABILITY_DONE" | tee -a "$LOG/stability_phase.log"
