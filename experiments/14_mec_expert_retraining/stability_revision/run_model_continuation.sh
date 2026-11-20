#!/usr/bin/env bash
# Continue one Transformer from its 2-epoch stability checkpoint to 20 epochs.
# Backs up the 2-epoch checkpoints first.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:?usage: run_model_continuation.sh <ts|tm|large>}"
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
export CITYNAV_DATA=/home/20261024_1/Workspace/DATA
export PYTORCH_ALLOC_CONF=expandable_segments:True

RUN="$HERE/runs/$MODEL/seed42"
BK="$RUN/checkpoints_stability_2ep"
if [ ! -f "$BK/last.pt" ]; then
  mkdir -p "$BK"
  cp -a "$RUN/checkpoints/best.pt" "$RUN/checkpoints/last.pt" "$BK/"
  echo "[$(date '+%F %T')] backed up $MODEL 2-epoch checkpoints -> $BK"
fi

EXTRA=""
if [ "$MODEL" = "large" ]; then EXTRA="--grad-checkpoint"; fi

echo "[$(date '+%F %T')] $MODEL CONTINUATION START (epochs -> 20)${EXTRA:+ [$EXTRA]}"
"$PY" "$HERE/train_stable.py" \
    --model "$MODEL" --epochs 20 \
    --resume "$BK/last.pt" \
    --out-dir "$HERE/runs" \
    --norm-file "$HERE/normalization_statistics.json" \
    $EXTRA \
    2>&1 | tee "$HERE/logs/train_${MODEL}_continuation.log"
rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] $MODEL CONTINUATION EXIT rc=$rc"
exit "$rc"
