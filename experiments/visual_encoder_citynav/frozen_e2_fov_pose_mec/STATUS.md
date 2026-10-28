# 实验 A：冻结 E2-FOV-Pose + MEC 闭环导航 — 状态

- 更新时间：2026-10-28 21:05
- 状态：**Token 生成进行中（分片断点已启用）**
- 分支：`exp/visual-encoder-ablation-20261028`

## 协议

| 项 | 值 |
|---|---|
| 视觉编码器 | **冻结** `runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/best_encoder.pt` |
| SHA-256 | `441de539f1d30f2be68d77a1c954daef5f3ab3bd3501f33b2c9798f89517011b`（已核验） |
| 数据规模 | 训练 **625 条轨迹 ≈ 1250 窗口**；Val-Unseen **140 条 ≈ 280 窗口**；Val-Seen 受 3 张协议地图限制为 **35 条 ≈ 70 窗口**（验证合计约 350）；每窗口 8 步 |
| 缓存 | `data_cache/t5_frozen_1000/` |
| 输出 | `run_20261028/` |
| 模型 | Small / Transformer-M，各含 `full`（检索+地图）与 `map_free` |
| MEC 步数 | 2000 / 模型，batch 16 |
| Test-Unseen | 未读取 |

## 阶段进度

| 阶段 | 状态 | 说明 |
|---|---|---|
| 候选库 + 语言特征 | 完成 | `data_cache/candidates/` |
| Retrieval 训练 | 完成（19:03） | `retrieval/{encoder.pt,retriever.pt}`，冻结编码器 |
| 语言特征 | 完成（21:28） | `data_cache/t5_frozen_1000/language_features.npz`，26,549 条，77 MB（磁盘） |
| Token 生成 | **完成（23:09）** | 1600 个窗口：train 1250、val_seen 70、val_unseen 280；分片 6.0 GB 在磁盘 |
| Token 合并 / MEC | **完成（23:16）** | `t5_tokens.npz` 5.29 GB；GAB 阈值 0.0388；4 个模型各 2000 步 |
| 闭环评估 | **进行中（00:25 启动）** | 每划分 40 条、8 分片、最多 120 步；见下 |

## MEC 结果（12800 帧：1250 train / 70 val-seen / 280 val-unseen）

| 模型 | VS acc | VS move-acc | VU acc | VU move-acc | VU Stop P/R |
|---|---:|---:|---:|---:|---:|
| small:full | 0.6161 | 0.6571 | 0.5732 | 0.6114 | 0.000 / 0.000 |
| small:map_free | 0.6929 | 0.6762 | 0.6464 | 0.6276 | 0.897 / 0.929 |
| transformer_m:full | 0.6429 | 0.6857 | 0.5938 | 0.6333 | 0.000 / 0.000 |
| transformer_m:map_free | 0.5786 | 0.5524 | 0.6022 | 0.5833 | 0.932 / 0.886 |

- 两个 full 模型在 argmax 决策下**从不预测 Stop**；闭环会跑满步数。
- 四个模型的 `TurnRight` 逐类准确率均为 0。
- 检索：Val-Seen R@1/5/10 = 0.174/0.609/0.609；Val-Unseen = 0.104/0.250/0.354。

## 闭环设置与已知代价

- 观测配置与 token 构建完全一致：336 px、120 m crop、无 splat。
- 单步渲染实测约 3.4 s（`birmingham_block_5`），与并发无关；
  因此 120 步约 7 min/条，每划分 40 条、8 分片，单轮约 35 min。
- 分片按地图排序后连续切块，每进程只覆盖少数地图；每进程强制单线程。
- 结果会同时报告 SR、oracle 成功率（rollout 最小距离 ≤20 m）、SPL、NE、nDTW、Stop 率。
| MEC 训练 | 待开始 | 4 个模型，权重写 `mec/<model>/{mec.pt}` |
| 闭环评估 | 待开始 | `closed_loop_eval.py`，分片 + journal 断点 |

### Token 断点

- 旧进程 19:55 收到 SIGTERM，当时 600 个窗口完成但未落盘（旧代码只在结尾整体写盘）。
- 本次增加 `--token-shards`：每完成一个只读 episode 即写
  `tokens/shards/*.npz`；用相同参数重启会按
  `(split, map, episode_id, start)` 复用，实测与一次性生成逐元素一致。
- 进程 PID 记录在 `run_20261028/pid_resume.txt`，日志
  `run_20261028/t5_smoke_resume.log`。
- 进度：`ls data_cache/t5_frozen_1000/tokens/shards | wc -l`（日志每 100 个打印一次）。

## 闭环评估约定

- 与 token 构建设置完全一致：336 px FPV、120 m crop、`fpv_splat_radius=0`、
  `map_cache_size=2`；GAB 阈值读取训练时拟合值（`t5_tokens.json`）。
- 动作由 MEC 输出（六类，含 Stop），逐步 `transition_pose` 后重渲染；
  目标位置只用于评估器计算 SR / NE / SPL / nDTW。
- 脚本：`closed_loop_eval.py`；分片启动 `run_closed_loop.sh`；
  汇总 `aggregate_closed_loop.py`。
- 已用随机 MEC 冒烟验证接口与断点 journal；正式规模待 MEC 训练完成后，
  按实测单步延迟与平均步数确定并记录。

## 已知风险

- CPU 渲染是主要成本（当前约 0.75 s/步，且与 Token 生成并行时会进一步变慢）；
  正式评估将分片并行并限制 `max_steps`，规模会在结果中注明。
- 单 seed，闭环结果只代表该检查点，不作显著性声明。
