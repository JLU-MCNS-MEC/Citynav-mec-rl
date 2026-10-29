#!/usr/bin/env bash
# Closed-loop suite for the class-balanced MEC retraining (2026-10-29).
#
# Same protocol as the baseline formal suite (24 episodes per split, 8 shards,
# max 100 steps) so the two result sets are directly comparable.  Weights are
# read from data_cache/t5_frozen_1000/mec_balanced; outputs go to
# closed_loop/balanced so the baseline checkpoints and results stay untouched.
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
HERE="$ROOT/experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec"
MEC_SUBDIR="${MEC_SUBDIR:-mec_balanced}"
SUITE_NAME="${SUITE_NAME:-balanced}"
OUT="$HERE/closed_loop/$SUITE_NAME"
export MEC_ROOT="$ROOT/data_cache/t5_frozen_1000/$MEC_SUBDIR"

EPISODES="${EPISODES:-24}"
SHARDS="${SHARDS:-8}"
MAX_STEPS="${MAX_STEPS:-100}"

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

# Same order as the baseline suite: unseen first is the cheapest end-to-end check.
run_one val_unseen small full
run_one val_unseen transformer_m full
run_one val_seen small full
run_one val_seen transformer_m full
run_one val_unseen small map_free
run_one val_unseen transformer_m map_free
run_one val_seen small map_free
run_one val_seen transformer_m map_free

echo "=== $(date -Is) suite finished"
