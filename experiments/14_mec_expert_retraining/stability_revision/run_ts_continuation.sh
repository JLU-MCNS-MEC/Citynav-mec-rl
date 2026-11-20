#!/usr/bin/env bash
# Continue Transformer-S from the 2-epoch stability checkpoint to the full
# 20-epoch run. Keeps the original 2-epoch checkpoints by first copying them
# to checkpoints_stability_2ep/.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
export CITYNAV_DATA=/home/20261024_1/Workspace/DATA
export PYTORCH_ALLOC_CONF=expandable_segments:True

TS="$HERE/runs/ts/seed42"
BK="$TS/checkpoints_stability_2ep"
if [ ! -f "$BK/last.pt" ]; then
  mkdir -p "$BK"
  cp -a "$TS/checkpoints/best.pt" "$TS/checkpoints/last.pt" "$BK/"
  echo "[$(date '+%F %T')] backed up 2-epoch checkpoints -> $BK"
fi

echo "[$(date '+%F %T')] TS CONTINUATION START (epoch 2 -> 20)"
"$PY" "$HERE/train_stable.py" \
    --model ts --epochs 20 \
    --resume "$BK/last.pt" \
    --out-dir "$HERE/runs" \
    --norm-file "$HERE/normalization_statistics.json" \
    2>&1 | tee "$HERE/logs/train_ts_continuation.log"
echo "[$(date '+%F %T')] TS CONTINUATION EXIT rc=${PIPESTATUS[0]}"
