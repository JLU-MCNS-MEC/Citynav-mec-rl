#!/usr/bin/env bash
# 09C full closed-loop: continuous (c1) then continuous + observability gate (c2).
set -euo pipefail
PY=/home/20261024_1/miniconda3/envs/DRL_migrated/bin/python
EXP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXP"
mkdir -p logs
for m in c1 c2; do
  echo "=== mode ${m} | $(date '+%F %T') ==="
  "$PY" -I -u scripts/run_09c.py --mode "$m" 2>&1 | tee -a "logs/${m}.log"
done
echo "=== 09C finished | $(date '+%F %T') ==="
