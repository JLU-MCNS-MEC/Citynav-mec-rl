#!/usr/bin/env bash
# Formal Stage-05 training: Transformer-M, seed 42 only, 30 epochs.
#
# Seed 43 is deliberately absent: this stage trains exactly one model.  Each
# restart resumes from checkpoint.pt (mid-epoch batch position included).
# Run inside tmux session citynav05_seed42 so an SSH drop cannot kill it.
set -euo pipefail
PY=/home/20261024_1/miniconda3/envs/DRL_migrated/bin/python
EXP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXP"

EPOCHS=${EPOCHS:-30}
K=${K:-8}
WORKERS=${WORKERS:-12}
LOG_DIR="$EXP/logs/seed42"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/seed42_$(date +%Y%m%d_%H%M%S).log"

# tee keeps the tmux pane live and the file complete
exec > >(tee -a "$LOG") 2>&1
echo "=== Stage 05 | Transformer-M | seed 42 | ${EPOCHS} epochs ==="
echo "started: $(date '+%F %T') | host: $(hostname) | log: $LOG"
echo "python:  $PY"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

"$PY" -u scripts/train_v2.py \
  --seed 42 --epochs "$EPOCHS" --k-per-episode "$K" \
  --eval-episodes 500 --eval-per-episode 4 --workers "$WORKERS" \
  --tag seed42 --resume --save-every 100

echo "finished: $(date '+%F %T')"
