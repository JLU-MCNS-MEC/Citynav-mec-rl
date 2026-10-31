#!/usr/bin/env bash
# 10A training launcher: runs train10a.py inside the citynav-mec env with the
# new-server data root, teeing stdout/stderr to results/<method>/train.log.
set -euo pipefail
METHOD="${1:?usage: launch_train.sh pq|ca}"
ROOT=/mnt/hdd2t/workspaces/Citynav-mec-rl-fixed50m
A="$ROOT/experiments/10_goal_query_mec_training/10A_point_query_cross_attention"
cd "$ROOT"
export CITYNAV_DATA=/mnt/hdd8t/Citynav-MEC/DATA
mkdir -p "$A/results/$METHOD"
exec /mnt/hdd2t/conda/envs/citynav-mec/bin/python -I -u \
  "$A/scripts/train10a.py" --method "$METHOD" \
  2>&1 | tee "$A/results/$METHOD/train.log"
