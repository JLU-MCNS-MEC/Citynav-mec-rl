#!/usr/bin/env bash
# Wait until the GPU has >=25 GB free (Stage 17 has priority), then start TM R3.
set -u
cd "$(dirname "$0")/../.." || exit 1
NEED=25000
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  drl=$(pgrep -fc "stage17c/train17c.py" || true)
  echo "$(date +%H:%M:%S) gpu_free=${free}MiB stage17_drl=${drl}" >> tm_r3_lr_continuation/queue.log
  if [ "$free" -ge "$NEED" ] && [ "${drl:-0}" -eq 0 ]; then
    echo "START $(date)" >> tm_r3_lr_continuation/queue.log
    bash tm_r3_lr_continuation/run_tm_r3.sh
    echo "DONE $(date)" >> tm_r3_lr_continuation/queue.log
    break
  fi
  sleep 300
done
