#!/usr/bin/env bash
# Wait for prebuild_caches.py, verify it succeeded, then run the E1 20-step smoke.
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
BASE="$ROOT/experiments/visual_encoder_citynav/ablations/20261028"
PREBUILD_LOG="$BASE/reports/prebuild_caches.log"
RUN_NAME="abl_20261028_e1_smoke"
RUN_DIR="$ROOT/runs/visual_encoder_citynav/$RUN_NAME"
STATUS="$BASE/reports/smoke_pipeline_status.txt"

mkdir -p "$RUN_DIR"

while pgrep -f 'prebuild_caches.py' > /dev/null 2>&1; do
  sleep 10
done

if ! grep -q '^EXIT=0' "$PREBUILD_LOG" 2>/dev/null; then
  echo "prebuild did not finish with EXIT=0; smoke not started" > "$STATUS"
  exit 1
fi

cd "$ROOT" || exit 1
export CITYNAV_DATA_ROOT="/home/20261024_1/Workspace/DATA/sbfnav-data"
export CITYNAV_PLY_ROOT="/home/20261024_1/Workspace/DATA/original_block_ply"
export PYTHONPATH="$ROOT"

echo "prebuild ok; starting smoke at $(date -Is)" > "$STATUS"
"$PY" -u experiments/visual_encoder_citynav/train.py --smoke \
  --config "$BASE/configs/e1.yaml" --run-name "$RUN_NAME" \
  --num-workers 2 > "$RUN_DIR/smoke.log" 2>&1
RC=$?
echo "EXIT=$RC" >> "$RUN_DIR/smoke.log"
echo "smoke finished rc=$RC at $(date -Is)" >> "$STATUS"
exit "$RC"
