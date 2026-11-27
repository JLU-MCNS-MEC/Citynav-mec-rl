#!/usr/bin/env bash
# Stage 14-R3: Transformer-M low-LR continuation from the Epoch-1 checkpoint.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.." || exit 1   # experiments/14_mec_expert_retraining
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
export CITYNAV_DATA=${CITYNAV_DATA:-/home/20261024_1/Workspace/DATA}
R3=tm_r3_lr_continuation
# --- smoke: 5 optimizer steps on a 64-state slice, separate output dir ---
$PY stability_revision/train_stable.py \
    --model tm --epochs 3 \
    --scheduler cosine_cont --lr 5e-6 --lr-end 5e-7 \
    --resume stability_revision/runs/tm/seed42/checkpoints_stability_2ep/last.pt \
    --out-dir "$R3/smoke_runs" --train-limit 64 --dev-limit 64 --max-steps 5 \
    > "$R3/smoke.log" 2>&1
if ! grep -q "epoch" "$R3/smoke.log"; then
    echo "SMOKE_FAILED" >> "$R3/train.log"; exit 1
fi
echo "SMOKE_OK" >> "$R3/train.log"

$PY stability_revision/train_stable.py \
    --model tm --epochs 20 \
    --scheduler cosine_cont --lr 5e-6 --lr-end 5e-7 \
    --min-epochs 16 --patience 10 \
    --resume stability_revision/runs/tm/seed42/checkpoints_stability_2ep/last.pt \
    --out-dir "$R3/runs" \
    > "$R3/train.log" 2>&1
echo "TM_R3_TRAIN_EXIT $?" >> "$R3/train.log"
