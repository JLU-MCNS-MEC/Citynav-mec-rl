#!/usr/bin/env bash
# Launch one ablation run with its own PID, log and status file.
# Usage: run_experiment.sh <e1|e2|e3|e4>
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
TAG="${1:?usage: run_experiment.sh <e1|e2|e3|e4>}"
BASE="$ROOT/experiments/visual_encoder_citynav/ablations/20261028"
CFG="$BASE/configs/$TAG.yaml"
NAME="abl_20261028_${TAG}_seed42"
RUN_DIR="$ROOT/runs/visual_encoder_citynav/$NAME"

mkdir -p "$RUN_DIR"
cd "$ROOT" || exit 1

EXTRA=""
if [ -f "$RUN_DIR/train_state.pt" ]; then
  EXTRA="--resume"
  echo "resume: found train_state.pt"
fi

export CITYNAV_DATA_ROOT="/home/20261024_1/Workspace/DATA/sbfnav-data"
export CITYNAV_PLY_ROOT="/home/20261024_1/Workspace/DATA/original_block_ply"
export PYTHONPATH="$ROOT"

nohup "$PY" -u experiments/visual_encoder_citynav/train.py \
  --config "$CFG" --run-name "$NAME" $EXTRA \
  > "$RUN_DIR/train.log" 2>&1 &
PID=$!
echo "$PID" > "$RUN_DIR/pid"
{
  echo "name=$NAME"
  echo "config=$CFG"
  echo "pid=$PID"
  echo "started=$(date -Is)"
  echo "resume=${EXTRA:-no}"
} > "$RUN_DIR/status.txt"
echo "started $NAME pid=$PID log=$RUN_DIR/train.log"
