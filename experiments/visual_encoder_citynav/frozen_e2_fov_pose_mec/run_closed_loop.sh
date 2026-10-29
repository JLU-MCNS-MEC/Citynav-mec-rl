#!/usr/bin/env bash
# Launch a sharded, resumable closed-loop evaluation.
#
# Usage:
#   run_closed_loop.sh <split> <model> <variant> <episodes> <shards> <out-dir> [extra args]
# Example:
#   run_closed_loop.sh val_seen small full 60 8 experiments/.../closed_loop
#
# variant is "full" (retrieval + map) or "map_free" (baseline weights).
# Each shard writes its own JSON + JSONL journal; rerunning the same command
# resumes finished episodes. Test-Unseen is rejected by the Python script.
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
SPLIT="${1:?split}"
MODEL="${2:?model}"
VARIANT="${3:?variant (full|map_free)}"
EPISODES="${4:?episodes}"
SHARDS="${5:?shards}"
OUT_DIR="${6:?out-dir}"
shift 6

CACHE="$ROOT/data_cache/t5_frozen_1000"
ENC="$ROOT/runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/best_encoder.pt"
MEC_ROOT="${MEC_ROOT:-$CACHE/mec}"
MEC="$MEC_ROOT/${MODEL}_${VARIANT}/mec.pt"
RET="$CACHE/retrieval/retriever.pt"

RETRIEVAL_ARGS=()
if [ "$VARIANT" = "full" ]; then
  RETRIEVAL_ARGS=(--use-retrieval --retriever-checkpoint "$RET")
fi

if [ ! -f "$MEC" ]; then
  echo "missing MEC checkpoint: $MEC" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

export CITYNAV_DATA_ROOT="/home/20261024_1/Workspace/DATA/sbfnav-data"
export CITYNAV_PLY_ROOT="/home/20261024_1/Workspace/DATA/original_block_ply"
export PYTHONPATH="$ROOT"
# Rendering is single-threaded; keep each shard to one core so eight shards do
# not oversubscribe the 24-core machine.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CITYNAV_TORCH_THREADS=1
cd "$ROOT" || exit 1

for ((i = 0; i < SHARDS; i++)); do
  OUT="$OUT_DIR/${SPLIT}_${MODEL}_${VARIANT}_shard${i}.json"
  "$PY" -u experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec/closed_loop_eval.py \
    --data-root "$CITYNAV_DATA_ROOT" \
    --ply-root "$CITYNAV_PLY_ROOT" \
    --split "$SPLIT" \
    --episodes "$EPISODES" \
    --shard-index "$i" --shards "$SHARDS" \
    --encoder-checkpoint "$ENC" \
    --mec-checkpoint "$MEC" \
    "${RETRIEVAL_ARGS[@]}" \
    --model "$MODEL" \
    --token-meta "$CACHE/tokens/t5_tokens.json" \
    --out "$OUT" "$@" \
    >> "$OUT_DIR/${SPLIT}_${MODEL}_${VARIANT}_shard${i}.log" 2>&1 &
  echo "started shard $i pid=$! log=$OUT_DIR/${SPLIT}_${MODEL}_${VARIANT}_shard${i}.log"
done
wait
echo "all $SHARDS shard(s) finished"
