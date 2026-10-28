# 实验 A：冻结 E2-FOV-Pose + MEC 闭环导航 — 状态

- 更新时间：2026-10-28 21:05
- 状态：**Token 生成进行中（分片断点已启用）**
- 分支：`exp/visual-encoder-ablation-20261028`

## 协议

| 项 | 值 |
|---|---|
| 视觉编码器 | **冻结** `runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/best_encoder.pt` |
| SHA-256 | `441de539f1d30f2be68d77a1c954daef5f3ab3bd3501f33b2c9798f89517011b`（已核验） |
| 数据规模 | 训练 1000 条轨迹 × 2 窗口 × 8 步；Val-Seen 100、Val-Unseen 100 条 |
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
| Token 生成 | **进行中**（20:48 重启） | 每 episode 一个分片，见 `tokens/shards/` |
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
