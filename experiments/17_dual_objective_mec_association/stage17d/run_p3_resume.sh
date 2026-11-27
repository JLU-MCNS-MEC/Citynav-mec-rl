#!/usr/bin/env bash
# Resume Stage 17 P3 after the TM experiment finishes.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
SRC=stage17d/p2
DST=stage17d/p3
run() {
  a=$1
  mkdir -p "$DST/$a"
  if [ -s "$DST/$a/last.pt" ]; then
    ck="$DST/$a/last.pt"
  else
    cp -n "$SRC/$a/history.jsonl" "$DST/$a/" 2>/dev/null || true
    cp -n "$SRC/$a/dev.jsonl" "$DST/$a/" 2>/dev/null || true
    ck="$SRC/$a/last.pt"
  fi
  $PY stage17c/train17c.py --algorithm "$a" --episodes 4000 --eval-every 100 \
      --resume "$ck" --out "$DST/$a" > "logs/p3r_$a.log" 2>&1
}
run dqn & run ddpg & wait; echo "P3R_PAIR1 $(date)"
run td3 & run sac & wait; echo "P3R_PAIR2 $(date)"
run ppo; echo "P3R_PPO $(date)"
$PY stage17d/plot_ma100.py --root "$DST" --out "$DST/figures" > logs/p3r_plot.log 2>&1
$PY stage17c/evaluate17.py --baselines random,min_latency,max_quality --out "$DST/baselines_dev.json" > logs/p3r_baselines.log 2>&1
echo "P3R_DONE $(date)"
