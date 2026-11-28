# Archived: Stage 14 Small-1G experiment (removed 2026-11-28)

Removed from the working tree at the review's request. Nothing was rewritten in
git history; the complete experiment is recoverable from:

- Commit `0516f6ad03df6a3aecf07099e9527e257f0895c7`
  (`Stage 14 small_1g results: 0.998 GFLOPs model reaches Val-Seen SR 72.26%`)
- Tag `archive/stage14-small1g-0516f6a`

Recover with:

```bash
git checkout archive/stage14-small1g-0516f6a -- experiments/14_mec_expert_retraining/small_1g_experiment
```

Summary of the archived result (for reference only; NOT part of the current
expert pool):

| item | value |
|---|---|
| change | token MLP 2048->12288->2048 replaced by 2048->2432->2048 |
| params | 20,740,996 |
| FLOPs (2xMACs) | 0.997762 G |
| best Dev Macro-F1 | 0.5898 (epoch 6) |
| Val-Seen SR (2498) | 0.7226 (harness cross-checked on 100 common tasks) |

Server checkpoints of this experiment (2.9 GB) were deleted together with the
directory; they are not needed to reproduce the numbers above and are not part
of any Stage 17 deployment.
