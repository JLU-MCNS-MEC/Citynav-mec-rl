# 实验 A：冻结 E2-FOV-Pose + MEC 闭环导航 — 状态

- 更新时间：2026-10-29 12:50
- 状态：**实验 A 完成（8/8 闭环 run 结束）**；sqrt 权重闭环复核也已完成
  （8/8，SR 仍全为 0）。结论为负面结果，未建立 A2 > A0 式提升
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
| 闭环评估 | **完成（03:45）** | 8/8 run；每划分 24 条、8 分片、最多 100 步；`closed_loop/formal/` |

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
- 单步渲染实测约 3.4 s（`birmingham_block_5`，单进程）；8 个分片并行时受内存
  带宽影响约 6 s/步。最终预算：每划分 24 条、最多 100 步、8 分片，单轮约 30 min。
  样本量小，SR 的置信区间较宽，报告中会明确标注，不外推。
- 分片按地图排序后连续切块，每进程只覆盖少数地图；每进程强制单线程。
- 结果会同时报告 SR、oracle 成功率（rollout 最小距离 ≤20 m）、SPL、NE、nDTW、Stop 率。

## 闭环结果（2026-10-29 03:45 完成，8/8）

报告：`closed_loop/formal/CLOSED_LOOP_REPORT.md`、`closed_loop_comparison.csv`、
`closed_loop_rates.png`；逐条结果在各 `*_summary.json`。

| split | model | variant | n | SR | oracle SR | SPL | NE (m) | nDTW | steps | stop | s/step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_seen | small | full | 24 | 0.000 | 0.042 | 0.000 | 225.5 | 0.550 | 100.0 | 0.000 | 5.23 |
| val_seen | small | map_free | 24 | 0.000 | 0.000 | 0.000 | 230.2 | 0.407 | 17.0 | 1.000 | 4.56 |
| val_seen | transformer_m | full | 24 | 0.000 | 0.042 | 0.000 | 219.0 | 0.580 | 100.0 | 0.000 | 5.09 |
| val_seen | transformer_m | map_free | 24 | 0.000 | 0.042 | 0.000 | 222.9 | 0.277 | 15.1 | 1.000 | 6.90 |
| val_unseen | small | full | 24 | 0.000 | 0.083 | 0.000 | 195.8 | 0.577 | 100.0 | 0.000 | 6.91 |
| val_unseen | small | map_free | 24 | 0.000 | 0.042 | 0.000 | 202.5 | 0.476 | 18.0 | 1.000 | 4.98 |
| val_unseen | transformer_m | full | 24 | 0.000 | 0.083 | 0.000 | 189.5 | 0.595 | 100.0 | 0.000 | 6.83 |
| val_unseen | transformer_m | map_free | 24 | 0.000 | 0.083 | 0.000 | 194.8 | 0.283 | 17.7 | 1.000 | 7.13 |

### 行为诊断

- **动作坍缩**：闭环中所有模型的输出只有 `MoveForward` / `GoDown` / `Stop`，
  `TurnLeft`、`TurnRight`、`GoUp` 一次都没有出现。
  - `full`：MoveForward ≈ 0.53–0.57、GoDown ≈ 0.43–0.47，从不 Stop → 恒跑满 100 步。
  - `map_free`：GoDown ≈ 0.94（small）/ 0.22–0.27（transformer_m，另约 66–72% MoveForward），
    约 15–18 步后 Stop。
- 这与 token 离线验证一致：四个模型 `TurnRight` 逐类准确率为 0，`full` 的 Stop 召回为 0；
  训练标签中 GoDown 占 39%（5013/12800）、MoveForward 23%，转向类合计仅约 8%。
  因此 **SR=0 的主因是策略没有学会转向与停止，而不是渲染或评估接口故障**；
  oracle SR（0–8.3%）表明个别 rollout 曾偶然接近目标。
- 检索差异不足以改变行为：`full`（检索+地图）与 `map_free` 的 nDTW 有差别，
  但两者都没有可用的转向决策。

## 类别不平衡修正（2026-10-29，离线结果）

训练脚本新增 `--mec-balance {none,balanced,sqrt}`（移动类逆频率权重 + Stop 的
`pos_weight`），新权重分别写到 `data_cache/t5_frozen_1000/mec_balanced/` 与
`mec_sqrt/`，基线 `mec/` 未改动。

| 运行 | Stop pos_weight | small:full VU acc | TurnLeft | TurnRight | GoUp | 结论 |
|---|---:|---:|---:|---:|---:|---|
| baseline | 1 | 0.573 | 0.08 | 0.00 | 0.59 | 从不 Stop，转向近 0 |
| balanced | ≈15.0 | 0.079 | 0.03 | 0.00 | 0.50 | **退化：永远 Stop**（P 0.064 / R 1.000） |
| sqrt | ≈3.87 | 0.588 | 0.11 | 0.02 | 0.65 | 可用折中；Stop 召回仍为 0 |

- 两个 `full` 变体对 `balanced` 的 Stop 权重极敏感，arrival head 直接塌成全正；
  `map_free` 受影响较小（VU acc 0.646→0.592）。
- `sqrt` 下 `small:full` 保持/略升总准确率并首次出现非零转向召回；
  `transformer_m:full` 略降（0.594→0.578）；两个 `map_free` 均略低于基线。
- 离线报告：`balanced_run_20261029/t5_report.{md,json}`、
  `sqrt_run_20261029/t5_report.{md,json}`。
- 闭环：sqrt 权重的 24 条 × 8 分片同协议对比**已完成（8/8，12:49）**，
  输出 `closed_loop/sqrt/`，完整报告见
  `closed_loop/sqrt/SQRT_CLOSED_LOOP_REPORT.md`。
- sqrt 闭环结论：**8/8 run 的 SR 仍为 0**；`transformer_m:full` 首次出现少量
  Stop（8.3% / 4.2%），`small:full` 仍从不 Stop；两个 `full` 变体的 NE 变差
  约 90–145 m、nDTW 接近腰斩。sqrt 权重不是可用修复。

## 下一步候选

1. **类别不平衡修正**：已执行并完成闭环复核（见上节）。sqrt 仍 SR=0 且 `full`
   的 NE/nDTW 变差，**不再沿 sqrt 方向调参**；后续改为定向排查 `full` 的 Stop
   失效与转向监督，再考虑 focal loss / 过采样。
2. **检查 full 的 Stop 失效**：同一数据上 map_free 能 Stop、full 不能，
   需要对比两变体的输入（检索分数是否淹没钱特征）与 loss 曲线。
3. **扩大闭环规模**（当前每划分 24 条，置信区间很宽）：仅在策略行为改善后再做，
   否则只是把 SR=0 测得更精确。

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
