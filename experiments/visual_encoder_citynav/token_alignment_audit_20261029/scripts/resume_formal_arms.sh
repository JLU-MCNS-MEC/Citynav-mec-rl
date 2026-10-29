#!/usr/bin/env bash
# Resume the two unfinished formal audit arms.
#
# Completed already: h0-token (3 seeds, results/train_h0_token.json).
# Remaining:
#   * h16-token  : finish the 14 missing token shards, merge, then 3 seeds
#   * h0-pretrain: finish the missing re-render part 2, merge, then 3 seeds
#
# The H=16 collector reuses every existing episode shard, and the pretrain
# re-render is split into four independent parts; only the missing part is
# recomputed.  All caches stay under this audit directory.
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
AUD="$ROOT/experiments/visual_encoder_citynav/token_alignment_audit_20261029"
LOG="$AUD/logs/resume_formal_arms.log"

export CITYNAV_DATA_ROOT="/home/20261024_1/Workspace/DATA/sbfnav-data"
export CITYNAV_PLY_ROOT="/home/20261024_1/Workspace/DATA/original_block_ply"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CITYNAV_TORCH_THREADS=1

echo "=== $(date -Is) resume formal arms" >> "$LOG"

# 1) Finish the missing pretrain re-render part in the background while the
#    much slower H=16 token collection runs.
PRETRAIN_PID=""
if [ ! -f "$AUD/cache/subset_h0_pretrain/parts/part_02.npz" ]; then
  "$PY" -u "$AUD/scripts/retokenize_pretrain.py" \
    --shard-dir "$AUD/cache/subset_h0/tokens/shards" \
    --token-meta "$AUD/cache/subset_h0/tokens/t5_tokens.json" \
    --out-dir "$AUD/cache/subset_h0_pretrain/parts" \
    --part 2 --parts 4 > "$AUD/logs/retok_subset_h0_2.log" 2>&1 &
  PRETRAIN_PID=$!
  echo "  pretrain part 2 pid=$PRETRAIN_PID" >> "$LOG"
fi

# 2) Finish H=16 collection (existing shards are reused), then merge/train the
#    script's seed-42 report.
H=16 SHARDS=5 bash "$AUD/scripts/build_history_subset.sh" >> "$LOG" 2>&1
h16_status=$?
echo "  h16 build exit=$h16_status" >> "$LOG"

# 3) Formal 3-seed h16-token arm.
if [ "$h16_status" -eq 0 ]; then
  "$PY" -u "$AUD/scripts/train_from_tokens.py" \
    --cache-dir "$AUD/cache/subset_h16" --seeds 42 43 44 \
    --out "$AUD/results/train_h16_token.json" >> "$LOG" 2>&1
  echo "  h16 train exit=$?" >> "$LOG"
fi

# 4) Finish pretrain re-render, merge, and run the formal 3-seed arm.
if [ -n "$PRETRAIN_PID" ]; then
  wait "$PRETRAIN_PID"
  echo "  pretrain part 2 exit=$?" >> "$LOG"
fi
"$PY" -u "$AUD/scripts/merge_retokenized.py" \
  --base-cache "$AUD/cache/subset_h0" \
  --parts-dir "$AUD/cache/subset_h0_pretrain/parts" \
  --out-cache "$AUD/cache/subset_h0_pretrain" >> "$LOG" 2>&1
echo "  pretrain merge exit=$?" >> "$LOG"

"$PY" -u "$AUD/scripts/train_from_tokens.py" \
  --cache-dir "$AUD/cache/subset_h0_pretrain" --seeds 42 43 44 \
  --out "$AUD/results/train_h0_pretrain.json" >> "$LOG" 2>&1
echo "  pretrain train exit=$?" >> "$LOG"

# 5) Collect all three arms into CSV + markdown.
"$PY" "$AUD/scripts/summarize_arms.py" >> "$LOG" 2>&1
echo "  summarize exit=$?" >> "$LOG"

echo "=== $(date -Is) formal arms finished" >> "$LOG"
