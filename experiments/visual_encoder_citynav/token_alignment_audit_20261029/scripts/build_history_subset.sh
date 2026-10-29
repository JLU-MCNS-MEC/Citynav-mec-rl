#!/usr/bin/env bash
# Build an audit-only token subset with configurable history context.
#
# H=0 keeps the legacy 8-step-window history; H=16 renders up to 16 real
# preceding states per supervised window.  Supervised steps/labels never
# change.  Sharded collection: several processes write the same shard dir,
# then one merge process finalizes tokens and trains transformer-M.
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
AUD="$ROOT/experiments/visual_encoder_citynav/token_alignment_audit_20261029"

H="${H:-16}"
SHARDS="${SHARDS:-5}"
TRAIN="${TRAIN:-80}"
VAL_SEEN="${VAL_SEEN:-12}"
VAL_UNSEEN="${VAL_UNSEEN:-12}"
TRAIN_MAPS="birmingham_block_10 birmingham_block_12 cambridge_block_17 cambridge_block_25"
VAL_SEEN_MAPS="birmingham_block_13 birmingham_block_11 cambridge_block_25"
VAL_UNSEEN_MAPS="birmingham_block_5 cambridge_block_10 cambridge_block_2 cambridge_block_3"
CACHE="$AUD/cache/subset_h${H}"
OUT="$AUD/results/subset_h${H}"
LOG="$AUD/logs/subset_h${H}"

export CITYNAV_DATA_ROOT="/home/20261024_1/Workspace/DATA/sbfnav-data"
export CITYNAV_PLY_ROOT="/home/20261024_1/Workspace/DATA/original_block_ply"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CITYNAV_TORCH_THREADS=1

mkdir -p "$CACHE/retrieval" "$OUT" "$LOG"
if [ ! -f "$CACHE/retrieval/retriever.pt" ]; then
  cp "$ROOT/data_cache/t5_frozen_1000/retrieval/encoder.pt" "$CACHE/retrieval/"
  cp "$ROOT/data_cache/t5_frozen_1000/retrieval/retriever.pt" "$CACHE/retrieval/"
fi
if [ ! -f "$CACHE/language_features.npz" ]; then
  cp "$ROOT/data_cache/t5_frozen_1000/language_features.npz" "$CACHE/"
fi

common=(
  "$PY" -u "$ROOT/scripts/t5_smoke.py"
  --device cuda --seed 42 --freeze-visual-encoder
  --encoder-checkpoint "$ROOT/runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/best_encoder.pt"
  --train-trajectories "$TRAIN" --val-seen-trajectories "$VAL_SEEN"
  --val-unseen-trajectories "$VAL_UNSEEN" --steps-per-trajectory 8
  --cache-dir "$CACHE" --reuse-retrieval --token-shards
  --history-context "$H"
  --train-maps $TRAIN_MAPS --val-seen-maps $VAL_SEEN_MAPS
  --val-unseen-maps $VAL_UNSEEN_MAPS
)

echo "=== $(date -Is) collect H=$H with $SHARDS train shards + 1 val shard"
for ((i = 0; i < SHARDS; i++)); do
  a=$((i * TRAIN / SHARDS)); b=$(((i + 1) * TRAIN / SHARDS))
  "${common[@]}" --collect-only \
    --train-range "$a:$b" --val-seen-range 0:0 --val-unseen-range 0:0 \
    --output-dir "$OUT" > "$LOG/collect_train_$i.log" 2>&1 &
  echo "  train shard $i range $a:$b pid=$!"
done
"${common[@]}" --collect-only \
  --train-range 0:0 --val-seen-range "0:$VAL_SEEN" --val-unseen-range "0:$VAL_UNSEEN" \
  --output-dir "$OUT" > "$LOG/collect_val.log" 2>&1 &
echo "  val shard pid=$!"

wait
echo "=== $(date -Is) collect done; merging and training transformer_m:full"

"${common[@]}" \
  --mec-models transformer_m:full --mec-steps 1000 --mec-batch-size 16 \
  --output-dir "$OUT" > "$LOG/merge_train.log" 2>&1
echo "=== $(date -Is) merge/train exit=$?"
