# 测试日志摘要（2026-10-29）

原始进程日志在 `logs/`（`.gitignore` 排除 `*.log`）；这里保留可提交的测试证据。

## 1. 历史因果 / 防泄漏测试

命令：

```bash
python experiments/visual_encoder_citynav/token_alignment_audit_20261029/scripts/test_history_leakage.py
```

输出（`results/history_causality_tests.txt`）：

```
PASS test_slots_match_previous_actions
PASS test_short_history_is_causal
PASS test_future_actions_do_not_leak
PASS test_future_poses_do_not_leak
PASS test_current_pose_anchors_relative_features_causally
PASS test_visual_history_is_causal
all history causality tests passed
```

## 2. h=0 / h=16 构造一致性探针（64 帧）

同一 4 条轨迹、2 个 8 步窗口：

- `actions` / `step_id` / `split_id` / `goal_tokens` / `map_feat` /
  `language_features`：**逐元素相同**。
- `current_tokens`：最大绝对差 0.63，平均 4.9e-5，逐帧余弦最小 0.99997
  （GPU 卷积非确定性；监督观测未改变）。
- `history_mask` 有效长度：
  - h=0：`{0:8, 1:8, …, 7:8}`（窗口内位置）
  - h=16：`{16:64}`（全部满历史）
- 历史槽位动作示例（h=16，最近优先）：
  `[5,1,1,1,5,1,1,3,1,5,1,2,1,1,5,1]`，当前标签 1。

## 3. 预训练协议重渲染链路（64 帧探针）

```bash
python scripts/retokenize_pretrain.py --shard-dir cache/probe_h0/tokens/shards \
  --token-meta cache/probe_h0/tokens/t5_tokens.json \
  --out-dir cache/probe_h0_pretrain/parts
python scripts/merge_retokenized.py --base-cache cache/probe_h0 \
  --parts-dir cache/probe_h0_pretrain/parts --out-cache cache/probe_h0_pretrain
python scripts/train_from_tokens.py --cache-dir cache/probe_h0_pretrain \
  --model transformer_m:full --mec-steps 30 --output results/probe_h0_pretrain_train.json
```

结果：`part 0: 64 frames from 8 shard(s)` → `merged 64 rows` → 训练正常完成。
Token 特征对比（同一状态）：

- 平均余弦 **0.973**，最小 0.786，平均 L2 0.172；
- GAB 平均绝对差 0.0105。

## 4. 教师 STOP 审计

```bash
python scripts/audit_teacher_stop.py --split train_seen   # 22,002 条
python scripts/audit_teacher_stop.py --split val_seen     # 2,498 条
python scripts/audit_teacher_stop.py --split val_unseen   # 2,826 条
```

退出码均为 0；统计写入 `results/teacher_stop_<split>.{csv,json}`，
直方图 `figures/teacher_stop_distance.png`。
