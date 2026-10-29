# T5 + 冻结 BERT 地图检索 Smoke 报告

- 生成时间: 2026-10-28T23:53:02+00:00
- 设备: `cuda`
- 数据: train_seen / val_seen / val_unseen；test_unseen 未读取
- 原始数据快照未变化: `True`

## 1. 输入模态

| 模态 | 维度 |
|---|---|
| Current RGB tokens | 576 × 384 |
| Goal RGB tokens（Top-3 检索候选） | 3 × 384 |
| History visual | 16 × 768 |
| Pose history | 16 × 12（6 连续 + 6 动作 one-hot） |
| Action history | 16 × 6 |
| History mask | 16 |
| GAB | 20 |
| Top-5 map retrieval | 22 |
| Frozen BERT | 768 |

## 2. 候选库与检索

- 网格 stride: 48.0 m
- 尺度: [32.0, 64.0, 128.0] m
- 候选地图数: 12
- 候选总数: 1602
- 训练视觉编码器: False

| split | samples | with positive | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|

Top-3 Goal 不足时零填充并记录 `goal_valid`；Top-5 不足时 map 特征零填充。推理端不接收 target 坐标/ID/裁剪/教师动作/成功标签。

## 3. T5 Token 数据

- 帧数: 12800
- 轨迹数: {'train_seen': 1250, 'val_seen': 70, 'val_unseen': 280}
- GAB 阈值（仅 train 拟合）: 0.03880675001548639
- 动作计数: {'0': {'Stop': 625, 'MoveForward': 2965, 'TurnRight': 493, 'TurnLeft': 532, 'GoUp': 372, 'GoDown': 5013}, '1': {'Stop': 35, 'MoveForward': 145, 'TurnRight': 15, 'TurnLeft': 32, 'GoUp': 16, 'GoDown': 317}, '2': {'Stop': 140, 'MoveForward': 818, 'TurnRight': 98, 'TurnLeft': 88, 'GoUp': 68, 'GoDown': 1028}}

## 4. 小规模 MEC 训练

| model | retrieval | initial val_seen acc | final val_seen acc | final val_unseen acc | move acc (val_unseen) | Stop P/R (val_unseen) |
|---|:---:|---:|---:|---:|---:|---:|
| small | True | 0.0625 | 0.6161 | 0.5879 | 0.6271 | 0.000/0.000 |
| small | False | 0.0625 | 0.6696 | 0.6308 | 0.6148 | 0.953/0.871 |
| transformer_m | True | 0.0625 | 0.5946 | 0.5777 | 0.6148 | 0.150/0.021 |
| transformer_m | False | 0.0625 | 0.5179 | 0.5737 | 0.5514 | 0.907/0.907 |

## 5. 结论与限制

- 这是小规模检索/六模态/训练 Smoke，不是正式训练或泛化结论；
- Goal RGB 来自语言检索的固定网格候选，不使用 Ground Truth 目标裁剪；
- 候选检索训练仅用 train_seen 的 target_positions 生成正负标签；推理/验证排序不使用 target 坐标、ID 或裁剪；
- map-free baseline 去掉 Goal/GAB 地图证据和 Top-5 map 分支，使用 current/history/pose/language；
- 点云 FPV 有空洞，部分地图配准仍需更严格检查；
- 未启动完整训练、RL、DQN/PPO 或 GSM。
