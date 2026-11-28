#!/usr/bin/env bash
# Stage 14-R4: Large low-LR continuation (smoke, then 18 epochs, no early stop).
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.." || exit 1
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
export CITYNAV_DATA=${CITYNAV_DATA:-/home/20261024_1/Workspace/DATA}
R4=large_r4_lr_continuation
CKPT=stability_revision/runs/large/seed42/checkpoints_stability_2ep/last.pt

$PY stability_revision/train_stable.py --model large --epochs 3 \
  --scheduler cosine_cont --lr 5e-6 --lr-end 5e-7 --grad-checkpoint \
  --resume "$CKPT" --out-dir "$R4/smoke_runs" \
  --train-limit 64 --dev-limit 64 --max-steps 5 > "$R4/smoke.log" 2>&1
grep -q "epoch" "$R4/smoke.log" || { echo SMOKE_FAILED > "$R4/train.log"; exit 1; }
echo SMOKE_OK > "$R4/train.log"

$PY stability_revision/train_stable.py --model large --epochs 20 \
  --scheduler cosine_cont --lr 5e-6 --lr-end 5e-7 \
  --sched-total-steps 176472 --sched-offset-steps 0 \
  --patience 0 --min-epochs 0 --grad-checkpoint \
  --resume "$CKPT" --out-dir "$R4/runs" >> "$R4/train.log" 2>&1
echo "LARGE_R4_EXIT $?" >> "$R4/train.log"
