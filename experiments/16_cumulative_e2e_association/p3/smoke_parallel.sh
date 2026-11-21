#!/usr/bin/env bash
# Stage 16-P3: two-process parallel smoke test (20 episodes each).
# Each process has its own output dir and log; no shared files.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
mkdir -p "$HERE/smoke/logs" "$HERE/smoke/dqn" "$HERE/smoke/ppo"

echo "[$(date '+%F %T')] smoke start" | tee "$HERE/smoke/logs/smoke.log"
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$HERE/smoke/logs/smoke.log"

"$PY" "$HERE/train16.py" --algorithm dqn --episodes 20 --seed 42 \
    --out-dir "$HERE/smoke/dqn" > "$HERE/smoke/logs/dqn.log" 2>&1 &
P1=$!
"$PY" "$HERE/train16.py" --algorithm ppo --episodes 20 --seed 42 \
    --out-dir "$HERE/smoke/ppo" > "$HERE/smoke/logs/ppo.log" 2>&1 &
P2=$!

# sample GPU every 10 s
(
  while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader \
      >> "$HERE/smoke/logs/gpu.log"
    sleep 10
  done
) &
MON=$!

wait $P1; R1=$?
wait $P2; R2=$?
kill $MON 2>/dev/null || true
echo "[$(date '+%F %T')] dqn rc=$R1 ppo rc=$R2" | tee -a "$HERE/smoke/logs/smoke.log"
echo "gpu_peak_mib=$(awk -F, '{gsub(/ MiB/,"",$1); if($1>m)m=$1} END{print m}' "$HERE/smoke/logs/gpu.log")" \
  | tee -a "$HERE/smoke/logs/smoke.log"
