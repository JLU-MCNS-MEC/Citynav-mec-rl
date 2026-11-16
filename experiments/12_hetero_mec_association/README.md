# Stage 12 — commands

```bash
PY=/home/tenant2/miniconda3/envs/citynav-mec/bin/python
export CITYNAV_DATA=/home/20261024_1/Workspace/DATA
cd /mnt/hdd2t/workspaces/Citynav-mec-rl-fixed50m

# 1. protocol
$PY experiments/12_hetero_mec_association/scripts/measure_encoder_flops.py
$PY experiments/12_hetero_mec_association/scripts/build_deployments.py

# 2. baselines (4 groups x 4 strategies), 6 parallel
$PY experiments/12_hetero_mec_association/scripts/scheduler.py \
    --stage baselines --parallel-baseline 6 --report-every 60

# 3. PPO (homogeneous first, then heterogeneous), 4 parallel training +
#    4 parallel evaluation; each run is evaluated as soon as it finishes
$PY experiments/12_hetero_mec_association/scripts/scheduler.py \
    --stage ppo --parallel-train 4 --parallel-eval 4 --episodes 4000

# 4. DQN (after all PPO runs)
$PY experiments/12_hetero_mec_association/scripts/scheduler.py \
    --stage dqn --parallel-train 4 --parallel-eval 4 --episodes 4000

# live status
$PY experiments/12_hetero_mec_association/scripts/scheduler.py \
    --stage ppo --status

# 5. tables, bar charts, MA100 curves, Pareto
$PY experiments/12_hetero_mec_association/scripts/summarize.py

# tests
$PY -m pytest -q experiments/12_hetero_mec_association/tests
```

Outputs:

```
results/protocol/        # frozen deployments, encoder FLOPs, capability prior
results/baselines/       # <group>/<strategy>/seed42/{journal,episodes,summary}
results/rl/              # <group>/<algo>/seed<42|43>/{history,dev,best.pt}
results/eval/            # <group>_<algo>_seed<42|43>/{journal,summary}
results/scheduler/       # live status JSON + logs
results/summary/         # RESULTS.md + results.json
results/analysis/        # bars_<group>.png, m100_<group>.png, pareto_sr_e2e.png
```
