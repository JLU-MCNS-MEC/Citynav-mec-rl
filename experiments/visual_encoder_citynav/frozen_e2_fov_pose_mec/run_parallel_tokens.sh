#!/usr/bin/env bash
# Experiment A: sharded token collection + merge + MEC training.
#
# Sizes (2026-10-28 decision):
#   train_seen : 625 trajectories -> 1250 windows
#   val_seen   :  88 requested, but the three protocol maps only hold 35
#                  eligible trajectories -> ~70 windows (hard cap, logged)
#   val_unseen : 140 trajectories -> ~280 windows
# Total validation is therefore ~350 windows.
#
# Stage 1 runs four collectors in parallel over disjoint record ranges.  Every
# episode is written to data_cache/t5_frozen_1000/tokens/shards immediately, so
# rerunning this script resumes instead of re-rendering.
# Stage 2 (merge) loads the shards in canonical order, fits the GAB threshold
# on train heatmaps, writes t5_tokens.npz and trains the four MEC models.
#
# Usage: run_parallel_tokens.sh [--collect-only|--merge-only]
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
CACHE="$ROOT/data_cache/t5_frozen_1000"
OUT="$ROOT/experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec/run_20261028"
ENC="$ROOT/runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/best_encoder.pt"
MODE="${1:---all}"

TRAIN=625
VAL_SEEN=88
VAL_UNSEEN=140
STEPS=8

export CITYNAV_DATA_ROOT="/home/20261024_1/Workspace/DATA/sbfnav-data"
export CITYNAV_PLY_ROOT="/home/20261024_1/Workspace/DATA/original_block_ply"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$ROOT" || exit 1
mkdir -p "$OUT"

BASE=(
  --device cuda --seed 42 --freeze-visual-encoder --encoder-checkpoint "$ENC"
  --train-trajectories "$TRAIN" --val-seen-trajectories "$VAL_SEEN"
  --val-unseen-trajectories "$VAL_UNSEEN" --steps-per-trajectory "$STEPS"
  --retrieval-train-per-map 16 --retrieval-val-per-map 12
  --cache-dir "$CACHE" --output-dir "$OUT"
)

collect() {
  local name="$1"; shift
  "$PY" -u scripts/t5_smoke.py "${BASE[@]}" "${COMMON_COLLECT[@]}" "$@" \
    > "$OUT/token_${name}.log" 2>&1 &
  echo $! > "$OUT/pid_token_${name}"
  echo "started ${name} pid=$! log=$OUT/token_${name}.log"
}

# --group-by-map only changes render order (episodes are identity-keyed); it
# avoids reloading a PLY every episode when records round-robin across maps.
COMMON_COLLECT=(--token-shards --reuse-retrieval --collect-only --group-by-map)

if [ "$MODE" = "--all" ] || [ "$MODE" = "--collect-only" ]; then
  # train collectors must not also collect the validation splits
  collect train0 --train-range 0:209 --val-seen-range 0:0 --val-unseen-range 0:0
  collect train1 --train-range 209:418 --val-seen-range 0:0 --val-unseen-range 0:0
  collect train2 --train-range 418:625 --val-seen-range 0:0 --val-unseen-range 0:0
  # validation split over two processes (seen maps are shared; unseen is cut)
  collect val0 --train-range 0:0 --val-seen-range 0:88 --val-unseen-range 0:70
  collect val1 --train-range 0:0 --val-seen-range 0:0 --val-unseen-range 70:140
  status=0
  for name in train0 train1 train2 val0 val1; do
    pid=$(cat "$OUT/pid_token_${name}")
    if wait "$pid"; then
      echo "collector ${name} finished OK"
    else
      echo "collector ${name} FAILED" >&2
      status=1
    fi
  done
  [ "$status" -eq 0 ] || exit "$status"
fi

if [ "$MODE" = "--all" ] || [ "$MODE" = "--merge-only" ]; then
  echo "merge: finalize tokens and train MEC"
  "$PY" -u scripts/t5_smoke.py "${BASE[@]}" --token-shards \
    --retrieval-steps 200 --mec-steps 2000 --mec-batch-size 16 \
    > "$OUT/merge_train.log" 2>&1
  echo "merge finished with exit=$?"
fi
