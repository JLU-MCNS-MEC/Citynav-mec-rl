#!/usr/bin/env bash
# Stage 14-R3: Transformer-M low-LR continuation from the Epoch-1 checkpoint.
set -u
cd "$(dirname "$0")/../.." || exit 1        # experiments/14_mec_expert_retraining
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
R3=tm_r3_lr_continuation
$PY stability_revision/train_stable.py \
    --model tm --epochs 20 \
    --scheduler cosine_cont --lr 5e-6 --lr-end 5e-7 \
    --min-epochs 16 --patience 10 \
    --resume stability_revision/runs/tm/seed42/checkpoints_stability_2ep/last.pt \
    --out-dir "$R3/runs" \
    > "$R3/train.log" 2>&1
echo "TM_R3_TRAIN_EXIT $?" >> "$R3/train.log"
