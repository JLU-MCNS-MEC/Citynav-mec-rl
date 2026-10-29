#!/usr/bin/env bash
# Experiment 02: full-model closed-loop recheck with a 500-step budget.
#
# Runs, for each split and round, small:full and transformer_m:full
# concurrently (8 shards each).  Per-step distance traces are written by
# closed_loop_eval.py; aggregate_experiment_02.py computes the
# 100/200/300/400/500-step sensitivity.
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
CLOSED="$ROOT/experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec"
HERE="$ROOT/experiments/02_full_closed_loop_recheck"
SEL="$HERE/selection"
OUT="$HERE/runs"
LOG="$HERE/suite.log"
MAX_STEPS=500
SHARDS=8
EPISODES=24

mkdir -p "$OUT"
echo "=== $(date -Is) experiment 02 suite start (max_steps=$MAX_STEPS)" >> "$LOG"

run_one() {
  local split="$1" model="$2" round="$3"
  local dir="$OUT/${split}_${model}_round${round}"
  local summary="$dir/${split}_${model}_full_summary.json"
  mkdir -p "$dir"
  if [ -f "$summary" ]; then
    echo "=== $(date -Is) skip ${split} ${model} round${round} (summary exists)" >> "$LOG"
    return 0
  fi
  echo "=== $(date -Is) start ${split} ${model} round${round}" >> "$LOG"
  bash "$CLOSED/run_closed_loop.sh" "$split" "$model" full "$EPISODES" "$SHARDS" "$dir" \
    --max-steps "$MAX_STEPS" \
    --episode-indices "$SEL/${split}_round${round}_indices.json" \
    > "$dir/suite_stdout.log" 2>&1
  local status=$?
  echo "    closed-loop exit=$status" >> "$LOG"
  if [ "$status" -eq 0 ]; then
    "$PY" "$CLOSED/aggregate_closed_loop.py" \
      --out-dir "$dir" --split "$split" --model "$model" \
      --variant full --shards "$SHARDS" >> "$dir/suite_stdout.log" 2>&1
    echo "    aggregate exit=$?" >> "$LOG"
  fi
  return "$status"
}

pair() {
  local split="$1" round="$2"
  run_one "$split" small "$round" &
  local p1=$!
  run_one "$split" transformer_m "$round" &
  local p2=$!
  wait "$p1"; local s1=$?
  wait "$p2"; local s2=$?
  echo "=== $(date -Is) pair ${split} round${round} done small=$s1 transformer_m=$s2" >> "$LOG"
}

pair val_seen A
pair val_unseen A

echo "=== $(date -Is) all runs finished; aggregating" >> "$LOG"
"$PY" "$HERE/scripts/aggregate_experiment_02.py" \
  --runs-dir "$OUT" --out-dir "$HERE/results" >> "$LOG" 2>&1
echo "=== $(date -Is) experiment 02 suite finished (aggregate exit=$?)" >> "$LOG"
