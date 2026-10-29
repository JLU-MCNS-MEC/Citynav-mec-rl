#!/usr/bin/env bash
# Re-render an audit subset with the E2 pretraining protocol, in parallel parts.
# Usage: run_retokenize_pretrain.sh <base-cache> [parts] [out-cache]
set -u
ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
AUD="$ROOT/experiments/visual_encoder_citynav/token_alignment_audit_20261029"

BASE="${1:?base cache dir}"
PARTS="${2:-4}"
OUT="${3:-${BASE}_pretrain}"
TAG="$(basename "$BASE")"

export CITYNAV_DATA_ROOT="/home/20261024_1/Workspace/DATA/sbfnav-data"
export CITYNAV_PLY_ROOT="/home/20261024_1/Workspace/DATA/original_block_ply"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CITYNAV_TORCH_THREADS=1

mkdir -p "$OUT/parts" "$AUD/logs"
echo "=== $(date -Is) re-render $TAG with $PARTS part(s) -> $OUT"
for ((p = 0; p < PARTS; p++)); do
  "$PY" -u "$AUD/scripts/retokenize_pretrain.py" \
    --shard-dir "$BASE/tokens/shards" \
    --token-meta "$BASE/tokens/t5_tokens.json" \
    --out-dir "$OUT/parts" --part "$p" --parts "$PARTS" \
    > "$AUD/logs/retok_${TAG}_$p.log" 2>&1 &
  echo "  part $p pid=$!"
done
wait
"$PY" -u "$AUD/scripts/merge_retokenized.py" \
  --base-cache "$BASE" --parts-dir "$OUT/parts" --out-cache "$OUT" \
  > "$AUD/logs/retok_${TAG}_merge.log" 2>&1
echo "=== $(date -Is) merge exit=$?"
