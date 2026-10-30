#!/usr/bin/env bash
# Stage 08B: run both geometric controllers over the full Stage-06 manifest.
# Resumable: each controller skips episodes already in its journal.
set -euo pipefail
PY=/home/20261024_1/miniconda3/envs/DRL_migrated/bin/python
EXP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXP"
mkdir -p logs
for c in top1 top5; do
  echo "=== controller ${c} | $(date '+%F %T') ==="
  "$PY" -I -u scripts/evaluate_geometry.py --controller "$c" 2>&1 | tee -a "logs/${c}.log"
done
echo "=== all controllers finished | $(date '+%F %T') ==="
