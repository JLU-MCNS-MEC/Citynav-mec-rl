#!/usr/bin/env bash
# Stage 15-Extension G: unattended pipeline.
#   1. wait for the running PPO / SAC (slack 0.8, 4000 episodes)
#   2. train TD3 + DQN in parallel, then DDPG alone (GPU memory fits 2 runs)
#   3. closed-loop Dev evaluation of all five checkpoints
#   4. MA(100) curves (reward / cumulative E2E / SR / energy ratio / lambdas)
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python
ROOT=cmdp/out/slack08
BUD=cmdp/budget_reference_1000.json
LOGS=logs
mkdir -p "$ROOT" "$ROOT/eval" "$ROOT/figures"

wait_file() { while [ ! -s "$1" ]; do sleep 60; done; }

wait_file "$ROOT/ppo/final.pt"
wait_file "$ROOT/sac/final.pt"
echo "STAGE1_DONE $(date)"

$PY cmdp/train_cmdp_offpolicy.py --algorithm td3 --episodes 4000 --slack 0.8 \
    --sr-target 0.5 --budget-file "$BUD" --eval-every 100 \
    --out "$ROOT/td3" > "$LOGS/cmdp_s08_td3.log" 2>&1 &
$PY cmdp/train_cmdp_offpolicy.py --algorithm dqn --episodes 4000 --slack 0.8 \
    --sr-target 0.5 --budget-file "$BUD" --eval-every 100 \
    --out "$ROOT/dqn" > "$LOGS/cmdp_s08_dqn.log" 2>&1 &
wait
echo "STAGE2_DONE $(date)"

$PY cmdp/train_cmdp_offpolicy.py --algorithm ddpg --episodes 4000 --slack 0.8 \
    --sr-target 0.5 --budget-file "$BUD" --eval-every 100 \
    --out "$ROOT/ddpg" > "$LOGS/cmdp_s08_ddpg.log" 2>&1
echo "STAGE3_DONE $(date)"

for a in ppo sac td3 dqn ddpg; do
    $PY cmdp/eval_cmdp.py --ckpt "$ROOT/$a/final.pt" --budget-file "$BUD" \
        --slack 0.8 --out "$ROOT/eval/$a.json" > "$LOGS/eval_cmdp_$a.log" 2>&1
done
echo "EVAL_DONE $(date)"

$PY cmdp/plot_ma100.py --runs "$ROOT/ppo" "$ROOT/sac" "$ROOT/td3" "$ROOT/dqn" \
    "$ROOT/ddpg" --labels PPO SAC TD3 DQN DDPG --out-dir "$ROOT/figures" \
    > "$LOGS/plot_cmdp.log" 2>&1
echo "PLOT_DONE $(date)"
echo "AUTOPILOT_DONE"
