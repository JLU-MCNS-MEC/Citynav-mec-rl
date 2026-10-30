# Stage 05 seed42 作废记录（2026-10-30 09:02–09:13）

## 运行身份

- 命令：`train_v2.py --seed 42 --epochs 30 --k-per-episode 8 --eval-episodes 500
  --eval-per-episode 4 --workers 12 --tag seed42 --resume`
- 进程：PID 2451779（父 2451776 `bash scripts/run_formal.sh`），PGID/SID 2451776，无 TTY
- 启动 09:02:47，停止 09:13（SIGINT → SIGTERM），完成 epoch 0–2
- 运行目录：`runs_v2/seed42/`（`config.json`、`history.jsonl`、`checkpoint.pt`、`best.pt`

## 判定：INVALID

`features_v2/{train_seen,val_seen,val_unseen}/meta.json` 三份都写着
`candidate_library = .../04_visual_goal_global_map_pipeline/cache_v2/candidates48`，
`dataset_v2.py` 也把 `CANDS` 硬编码为 `candidates48`。协议要求 Stage 05 统一使用
`cache_v2/candidates24`，并明确禁止混用旧的 48 m 候选缓存。该运行的 12 维目标信念
全部由 48 m 候选网格产生，与正式协议不一致。

数据来源本身是正确的：`03b_target_directed_teacher/results/teacher_train_seen.npz`
的 22,002 条 Train-Seen 直接目标导向教师轨迹（不是 Stage 02 的 high/medium/low
筛选，也不是 256 条人工轨迹）。

## 作废前的三个 epoch（仅作缺陷证据，不得作为结果引用）

| epoch | train loss | val acc | val Macro-F1 | STOP 召回 | val TurnLeft / TurnRight 召回 | 秒 |
|---|---|---|---|---|---|---|
| 0 | 0.2390 | 0.9184 | 0.7091 | 0.500 | 0.699 / 0.588 | 164.0 |
| 1 | 0.2224 | 0.9305 | 0.6663 | 0.167 | 0.688 / 0.553 | 165.8 |
| 2 | 0.2140 | 0.9345 | 0.6261 | 0.017 | 0.742 / 0.588 | 162.4 |

总体准确率上升、Macro-F1 与 STOP 召回同时下降，正是协议要求盯住的“退化为全 Forward”
信号；作废与这一趋势无关，作废原因是候选库错误。

## 归档

`runs_v2/INVALID_candidates48_20261030/`：`config.json`、`history.jsonl`、
`train_stdout.log`、三份 `*_meta.json`、`checkpoints.sha256`。
旧特征缓存整体改名为 `features_v2_candidates48_INVALID/`（4.4 GB，保留但不再使用）。
`runs_v2/seed42/` 的权重不作为任何报告依据。

## 后续处理

1. `dataset_v2.py` 改用 `cache_v2/candidates24`，缓存索引写入候选库版本，加载时校验。
2. 重提 episode 级特征缓存（原子写 + journal + SHA-256 校验）。
3. 归一化只保留 GPU 一条实现，缓存与训练路径特征 `max_abs_diff = 0`。
4. 训练改为 batch 级断点（每 100 batch）并在 tmux `citynav05_seed42` 中重启。

## 命令行片段，如需复查

```bash
ps -p 2451779 -o lstart,cmd          # 已结束
cat runs_v2/INVALID_candidates48_20261030/train_stdout.log
```
