#!/usr/bin/env bash
# Stage 12 Oracle v2 re-evaluation: four deployments, Val-Seen 2,498 each.
# Only the Oracle action-agreement rule is run; no other baseline / RL job.
set -uo pipefail
cd /mnt/hdd2t/workspaces/Citynav-mec-rl-fixed50m
export CITYNAV_DATA=/home/20261024_1/Workspace/DATA
export STAGE12_TORCH_THREADS=4
export OMP_NUM_THREADS=4

PY=/home/tenant2/miniconda3/envs/citynav-mec/bin/python
OUT=experiments/12_hetero_mec_association/results/oracle_action_agreement_v2
mkdir -p "$OUT/logs"

for g in two_homo two_hetero five_homo five_hetero; do
  (
    "$PY" experiments/12_hetero_mec_association/scripts/run_association_closed_loop.py \
      --group "$g" \
      --strategy oracle_action_agreement_v2 \
      --seed 42 \
      --out-dir "oracle_action_agreement_v2/$g" \
      --device cuda \
      > "$OUT/logs/$g.log" 2>&1
  ) &
done

wait
echo "[all] Oracle v2 evaluations finished at $(date -Is)"
