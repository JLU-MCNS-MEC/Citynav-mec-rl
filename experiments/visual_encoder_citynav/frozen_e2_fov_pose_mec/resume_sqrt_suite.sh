#!/usr/bin/env bash
# Resume the single remaining sqrt-weight closed-loop run.
#
# The baseline suite (run_closed_loop_suite.sh) already completed 7/8 runs.
# The last run, val_seen / transformer_m / map_free, was interrupted after it
# started.  Episode journals are per-shard and resumable, so rerunning the same
# command only executes the episodes that are still missing.
#
# Protocol is identical to the completed sqrt runs:
#   24 episodes per split, 8 shards, max 100 steps, MEC_ROOT=mec_sqrt.
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
HERE="$ROOT/experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec"
OUT="$HERE/closed_loop/sqrt"
LOG="$HERE/closed_loop/sqrt_suite.log"

export MEC_ROOT="$ROOT/data_cache/t5_frozen_1000/mec_sqrt"

echo "=== $(date -Is) resume val_seen transformer_m map_free" >> "$LOG"
bash "$HERE/run_closed_loop.sh" val_seen transformer_m map_free 24 8 "$OUT" \
  --max-steps 100 > "$OUT/val_seen_transformer_m_map_free_launch.log" 2>&1
status=$?
echo "    closed-loop exit=$status" >> "$LOG"

if [ "$status" -eq 0 ]; then
  "$PY" "$HERE/aggregate_closed_loop.py" \
    --out-dir "$OUT" --split val_seen --model transformer_m \
    --variant map_free --shards 8 >> "$LOG" 2>&1
  agg_status=$?
  echo "    aggregate exit=$agg_status" >> "$LOG"
  if [ "$agg_status" -ne 0 ]; then
    status="$agg_status"
  fi
fi

echo "=== $(date -Is) sqrt suite finished (exit=$status)" >> "$LOG"
exit "$status"
