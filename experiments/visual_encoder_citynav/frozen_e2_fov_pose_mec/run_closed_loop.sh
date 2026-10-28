#!/usr/bin/env bash
# Launch a sharded, resumable closed-loop evaluation.
#
# Usage:
#   run_closed_loop.sh <split> <model> <episodes> <shards> <out-dir> [extra args]
# Example:
#   run_closed_loop.sh val_seen small 200 4 experiments/.../closed_loop
#
# Each shard writes its own JSON + JSONL journal; rerunning the same command
# resumes finished episodes. Test-Unseen is rejected by the Python script.
set -u

ROOT="/home/20261024_1/Workspace/Citynav-mec-rl"
PY="$HOME/miniconda3/envs/DRL_migrated/bin/python"
SPLIT="${1:?split}"
MODEL="${2:?model}"
EPISODES="${3:?episodes}"
SHARDS="${4:?shards}"
OUT_DIR="${5:?out-dir}"
shift 5

CACHE="$ROOT/data_cache/t5_frozen_1000"
ENC="$ROOT/runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/best_encoder.pt"
MEC="$CACHE/mec/${MODEL}_full/mec.pt"
RET="$CACHE/retrieval/retriever.pt"

if [ ! -f "$MEC" ]; then
  echo "missing MEC checkpoint: $MEC" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

export CITYNAV_DATA_ROOT="/home/20261024_1/Workspace/DATA/sbfnav-data"
export CITYNAV_PLY_ROOT="/home/20261024_1/Workspace/DATA/original_block_ply"
export PYTHONPATH="$ROOT"
cd "$ROOT" || exit 1

for ((i = 0; i < SHARDS; i++)); do
  OUT="$OUT_DIR/${SPLIT}_${MODEL}_full_shard${i}.json"
  "$PY" -u experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec/closed_loop_eval.py \
    --data-root "$CITYNAV_DATA_ROOT" \
    --ply-root "$CITYNAV_PLY_ROOT" \
    --split "$SPLIT" \
    --episodes "$EPISODES" \
    --shard-index "$i" --shards "$SHARDS" \
    --encoder-checkpoint "$ENC" \
    --mec-checkpoint "$MEC" \
    --retriever-checkpoint "$RET" \
    --model "$MODEL" --use-retrieval \
    --token-meta "$CACHE/tokens/t5_tokens.json" \
    --out "$OUT" "$@" \
    >> "$OUT_DIR/${SPLIT}_${MODEL}_full_shard${i}.log" 2>&1 &
  echo "started shard $i pid=$! log=$OUT_DIR/${SPLIT}_${MODEL}_full_shard${i}.log"
done
wait
echo "all $SHARDS shard(s) finished"
