# Token alignment audit (2026-10-29)

Independent, read-only audit of the Experiment A token pipeline.  Nothing here
overwrites shared caches, MEC weights, the frozen E2-FOV-Pose encoder or the
closed-loop results.

## Contents

- `REPORT.md` — Chinese report: rendering consistency, teacher STOP vs 20 m,
  history context, small Transformer-M arms, encoder-retraining judgement.
- `scripts/`
  - `audit_render_consistency.py` — same-pose renders under pretraining /
    token / closed-loop protocols, with RGB-D and encoder-token metrics.
  - `audit_teacher_stop.py` — replays the teacher over a split and measures the
    3-D distance from the STOP pose to `target_positions[-1]`.
  - `audit_coverage.py` — full Train-Seen action distribution, trajectory
    lengths and sampled-token coverage.
  - `test_history_leakage.py` — causality/leakage tests for extended history.
  - `build_history_subset.sh` — sharded collection of the H=0 / H=16 subsets.
  - `retokenize_pretrain.py` + `merge_retokenized.py` — re-render an existing
    subset with the E2 pretraining protocol (4M cloud, 300 m, splat 1).
  - `train_from_tokens.py` — train transformer-M on a cached audit subset.
  - `summarize_arms.py` — CSV/markdown comparison of the model arms.
- `results/` — CSV/JSON/tables used in the report (small, committed).
- `figures/` — render panels and teacher-STOP histogram.
- `cache/`, `logs/` — heavy token caches and process logs (git-ignored).

## Reproduce

```bash
export CITYNAV_DATA_ROOT=/home/20261024_1/Workspace/DATA/sbfnav-data
export CITYNAV_PLY_ROOT=/home/20261024_1/Workspace/DATA/original_block_ply
export PYTHONPATH=$PWD

python experiments/visual_encoder_citynav/token_alignment_audit_20261029/scripts/audit_render_consistency.py
python experiments/visual_encoder_citynav/token_alignment_audit_20261029/scripts/audit_teacher_stop.py --split train_seen
python experiments/visual_encoder_citynav/token_alignment_audit_20261029/scripts/test_history_leakage.py

H=0  SHARDS=4 bash experiments/.../scripts/build_history_subset.sh
H=16 SHARDS=5 bash experiments/.../scripts/build_history_subset.sh
```
