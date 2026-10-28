#!/usr/bin/env bash
# Closed-loop suite for experiment A.
#
# Budget (2026-10-28): 60 episodes per split, 8 shards, max 120 steps.  The
# retrieval ("full") models never emit Stop, so every episode runs to the step
# cap; that is reported, not hidden.  The map-free baselines stop normally.
#
# Each run is resumable through its per-episode journal.  Results are merged by
# aggregate_closed_loop.py into <split>_<model>_<variant>_summary.{json,csv}.
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
HERE="$ROOT/experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec"
OUT="$HERE/closed_loop/formal"

EPISODES="${EPISODES:-60}"
SHARDS="${SHARDS:-8}"
MAX_STEPS="${MAX_STEPS:-120}"

mkdir -p "$OUT"

run_one() {
  local split="$1" model="$2" variant="$3"
  echo "=== $(date -Is) start ${split} ${model} ${variant}"
  bash "$HERE/run_closed_loop.sh" "$split" "$model" "$variant" \
    "$EPISODES" "$SHARDS" "$OUT" --max-steps "$MAX_STEPS" \
    > "$OUT/${split}_${model}_${variant}_launch.log" 2>&1
  local status=$?
  echo "    closed-loop exit=$status"
  "$PY" "$HERE/aggregate_closed_loop.py" \
    --out-dir "$OUT" --split "$split" --model "$model" --variant "$variant" \
    --shards "$SHARDS" || echo "    aggregate failed"
}

# unseen first: only four maps, so it is the cheapest end-to-end check
run_one val_unseen small full
run_one val_unseen transformer_m full
run_one val_seen small full
run_one val_seen transformer_m full
run_one val_unseen small map_free
run_one val_unseen transformer_m map_free
run_one val_seen small map_free
run_one val_seen transformer_m map_free

echo "=== $(date -Is) suite finished"
