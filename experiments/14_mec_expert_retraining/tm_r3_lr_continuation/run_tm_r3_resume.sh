#!/usr/bin/env bash
# Stage 14-R3 resume: continue from epoch 11 to the full 20-epoch budget with
# early stopping disabled and the original cosine schedule continued exactly
# (offset 98,040 = 10 completed epochs x 9,804 updates).
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.." || exit 1
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
export CITYNAV_DATA=${CITYNAV_DATA:-/home/20261024_1/Workspace/DATA}
R3=tm_r3_lr_continuation
$PY stability_revision/train_stable.py \
    --model tm --epochs 20 \
    --scheduler cosine_cont --lr 5e-6 --lr-end 5e-7 \
    --sched-total-steps 176472 --sched-offset-steps 98040 \
    --patience 0 --min-epochs 0 \
    --resume "$R3/runs/tm/seed42/checkpoints/last.pt" \
    --out-dir "$R3/runs" \
    >> "$R3/train_resume.log" 2>&1
echo "TM_R3_RESUME_EXIT $?" >> "$R3/train_resume.log"
