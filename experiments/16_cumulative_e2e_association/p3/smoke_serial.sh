#!/usr/bin/env bash
# Stage 16-P3: SERIAL smoke test. One algorithm at a time, 20 episodes each.
# Records GPU peak, throughput and reproducibility (two runs, same seed).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
mkdir -p "$HERE/smoke/logs"
LOG="$HERE/smoke/logs/serial.log"
: > "$LOG"

run_one() {
  local algo="$1" tag="$2"
  local out="$HERE/smoke/${algo}_${tag}"
  rm -rf "$out"; mkdir -p "$out"
  echo "[$(date '+%F %T')] START $algo $tag" | tee -a "$LOG"
  local t0=$(date +%s)
  ( while true; do
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
        >> "$out/gpu.log"; sleep 5
    done ) &
  local MON=$!
  "$PY" "$HERE/train16.py" --algorithm "$algo" --episodes 20 --seed 42 \
      --out-dir "$out" > "$out/train.log" 2>&1
  local rc=$?
  kill $MON 2>/dev/null || true
  local t1=$(date +%s)
  local peak=$(awk -F, '{gsub(/ /,"",$1); if($1+0>m)m=$1+0} END{print m+0}' "$out/gpu.log" 2>/dev/null)
  echo "[$(date '+%F %T')] END $algo $tag rc=$rc elapsed=$((t1-t0))s gpu_peak=${peak}MiB" | tee -a "$LOG"
}

for algo in dqn ppo; do
  run_one "$algo" run1
  run_one "$algo" run2   # reproducibility check (same seed)
done

echo "[$(date '+%F %T')] REPRODUCIBILITY" | tee -a "$LOG"
for algo in dqn ppo; do
  if cmp -s "$HERE/smoke/${algo}_run1/history.jsonl" "$HERE/smoke/${algo}_run2/history.jsonl"; then
    echo "  $algo: identical history.jsonl" | tee -a "$LOG"
  else
    echo "  $algo: DIFFERS (check nondeterminism)" | tee -a "$LOG"
  fi
done
echo "[$(date '+%F %T')] SERIAL_SMOKE_DONE" | tee -a "$LOG"
