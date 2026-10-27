# FPV 六动作小规模过拟合验证

- 日期：2026-10-27
- 目的：验证冻结 BERT + FPV 视觉编码器 + Small / Transformer-M + 六动作监督的完整训练链路
- 性质：**过拟合检查，不是正式训练结果，不是 val 泛化指标**
- 数据：仅 `train_seen`；未读取 `val_unseen` / `test_unseen`

## 1. 数据与缓存

命令：

```bash
python scripts/overfit_fpv_smoke.py \
  --trajectories 128 \
  --frames-per-trajectory 2 \
  --steps 300 \
  --batch-size 8 \
  --models small transformer_m \
  --device cuda \
  --render-size 336 \
  --crop-radius 120 \
  --max-points 4000000 \
  --splat-radius 0
```

| 项目 | 数值 |
|---|---:|
| 训练轨迹数 | 128 |
| FPV 帧数 | 256 |
| 地图数 | 8 |
| 动作计数 | Stop 32 / Forward 32 / Right 31 / Left 31 / Up 60 / Down 70 |
| 缓存大小 | 46.1 MB (`data_cache/fpv_overfit/frames_12407c0a9cca.npz`) |
| 缓存渲染耗时 | 46.3 s（256 帧，336×336，每图最多 400 万点） |
| 每图点数 | 2.35M–4.0M |
| 原始数据快照 | 未变化，摘要 `2c784870be858d7d801ee7c12e68c81b44b11cb7` |

标签来自官方 lookahead 教师：教师对 `marker_positions[-1]` 以 5 m 停止，评估成功另用 `target_positions[-1]` 的 20 m 半径。采样时显式保证六类覆盖，而不是只取前几个教师步骤。

## 2. 文本与视觉编码器

- 文本：本地 `bert-base-uncased`，109,482,240 参数，全部 `requires_grad=False`；输出 768 维池化特征；128 条唯一描述。
- 视觉：MEC-V8.2 `shared_encoder.pt` 初始化，3,997,056 参数；本次过拟合允许更新。
- Small MEC：695,542 参数。
- Transformer-M MEC：2,920,310 参数。
- 语言投影层和 MEC 主干可训练；BERT 不进入优化器。

## 3. 过拟合结果

300 步 AdamW，batch=8；训练与评估使用同一批 256 帧。

| model | initial acc | final acc | final move acc | Stop P | Stop R | train s | peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 0.1250 | **0.6133** | 0.5938 | 0.857 | 0.750 | 7.72 | 1,307.7 MB |
| Transformer-M | 0.1250 | **0.5977** | 0.5804 | 0.742 | 0.719 | 9.45 | 1,436.8 MB |

初始评估时模型几乎全部预测 Stop（32/256 = 0.125）；训练后六类和 Stop 均明显改善。训练损失：

| model | step 0 | step 100 | step 299 |
|---|---:|---:|---:|
| Small | 2.236 | 1.313 | 0.914 |
| Transformer-M | 2.672 | 1.987 | 1.087 |

六类最终 per-class accuracy：

| model | Stop | Forward | Right | Left | Up | Down |
|---|---:|---:|---:|---:|---:|---:|
| Small | 0.750 | 0.625 | 0.226 | 0.452 | 0.633 | 0.771 |
| Transformer-M | 0.719 | 0.781 | 0.548 | 0.290 | 0.633 | 0.586 |

Right/Left 仍较难区分，符合两类动作在单帧视觉上高度相似；Forward/Down/Stop 已能较好拟合。

## 4. 梯度与冻结检查

| model | encoder grad norm | MEC grad norm | encoder ΔL2 | MEC ΔL2 | BERT grad params | BERT changed |
|---|---:|---:|---:|---:|---:|:---:|
| Small | 0.355 | 0.935 | 0.7444 | 3.1568 | 0 | 否 |
| Transformer-M | 0.369 | 0.929 | 0.6292 | 5.4850 | 0 | 否 |

- 视觉编码器和 MEC 的梯度均有限，参数确实更新；
- BERT 所有参数 `requires_grad=False`，反向传播后无梯度，参数指纹前后完全一致；
- 训练吞吐：Small 311 帧/s，Transformer-M 254 帧/s（RTX 5090，batch=8）；
- 峰值显存：1.31 / 1.44 GB。

## 5. 结论与限制

结论：六动作标签、五类移动头 + 二元 Stop 头、六类概率组合、冻结 BERT、可训练视觉编码器与 MEC 的联合训练链路已贯通，且可以在 256 帧上明显过拟合。

限制：

- 这是同一批训练帧上的过拟合，不代表 val_seen / val_unseen 泛化；
- 只用了 128 条轨迹和 8 个地图，不是完整训练集；
- 其中部分地图的点云/正射颜色相关性较低（见 FPV 报告），正式实验前需要逐地图几何配准检查；
- 网络状态仍是确定性 5 维占位向量；
- 没有完整训练配方（V8.2 的 class balance、arrival hard states、checkpoint 选择）和正式 SR/SPL/NE 评测；
- 仍然没有 RL 环境、DQN/PPO 或 GSM。

## 6. 产物

- 训练缓存：`data_cache/fpv_overfit/frames_12407c0a9cca.npz`（本地忽略）
- Checkpoint：`data_cache/overfit_checkpoints/{small,transformer-m}/`（本地忽略）
- 完整指标：`reports/overfit/overfit_report.json`（本地忽略）
- 相关命令与数据快照记录在报告中；没有把权重、缓存或日志提交到 GitHub。

## 7. 下一步

1. 对正式使用的地图做逐 block 几何配准/可视化检查，排除低相关地图；
2. 用完整 `train_seen` 训练集做监督基线，在 `val_seen` / `val_unseen` 评测 SR/SPL/NE；
3. 移植 V8.2 的 movement class balance、arrival hard states 和 checkpoint 指标；
4. 替换占位网络状态为真实/模拟 MEC 状态；
5. 之后再考虑 RL 闭环和 FPV 缓存规模化。
