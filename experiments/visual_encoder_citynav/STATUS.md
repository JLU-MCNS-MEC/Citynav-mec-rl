# Visual Encoder CityNav — 当前状态

- 更新时间：2026-10-27
- 状态：**训练完成并已核验，勿重跑**
- 运行名：`full_20261027_masks_b8`
- 运行目录：`runs/visual_encoder_citynav/full_20261027_masks_b8/`
- 环境：`/home/ubuntu5/miniconda3/envs/AirVLN/bin/python`（torch 2.9.1+cu128）

## 训练

| 项 | 值 |
|---|---|
| 数据 | Train-Seen 22,002 条，24 图，单 epoch |
| 步数 | 2751 / 2751 |
| batch size | 8（首次 batch=32 在第 390 步中断，未用于结果） |
| 训练目标 | action CE + 0.5 × xview InfoNCE + 0.5 × retrieval InfoNCE |
| 冻结 | BERT 冻结；不训练 Small/Transformer-M，不做 RL |
| 耗时 | 2348.937 s（39.15 min） |
| GPU 峰值分配 | 5612.14 MiB |
| 参数变化 | L2 1.0670，相对初始化 1.52% |

## 验证（最佳 step 2751）

| 指标 | Val-Seen | Val-Unseen |
|---|---:|---:|
| action acc | 0.4297 | 0.4336 |
| action macro-F1 | 0.3536 | 0.3699 |
| xview Top-1 / Top-5 | 0.0562 / 0.1968 | 0.0317 / 0.1786 |
| retrieval R@1 / @5 / @10 | 0.0820 / 0.2891 / 0.4648 | 0.0547 / 0.2070 / 0.3359 |

- 选模规则：Val-Seen `0.4*macro-F1 + 0.3*xview_top1 + 0.3*R@5`，最佳分数 0.2450。
- Val-Unseen 只诊断，不参与选模；Test-Unseen 封存。
- 主要短板：TurnRight/TurnLeft 几乎不被预测；检索/跨视角绝对指标偏低。

## Checkpoint

- 最佳：`runs/visual_encoder_citynav/full_20261027_masks_b8/best_encoder.pt`
- SHA-256：`322cfa82a06ee926b74541ecde30f7ecd20b15c0a6ddd687647a0cd80d27b23b`
- 大小/参数：15.26 MiB，3,997,056 参数，29 个 encoder 张量
- 严格加载：1.0000，缺失/多余/形状不匹配 0，无辅助头 key
- final 与 best 逐张量一致（最大绝对差 0.0）
- 最终：`encoder_final.pt`，SHA-256
  `b66862b84cbfc97cf5a91f72a74890319d3a82380a77273ec8b2c4ef17b0f569`
- 其他：`checkpoints/encoder_step_{000688,001376,002063,002751}.pt`

## 已生成结果

- `REPORT.md`：完整中文报告
- `results/full_20261027_masks_b8_{train_metrics,loss_components,validation_summary,per_map}.csv`
- `results/full_20261027_masks_b8_{analysis,best_metrics,val_history,gradient_checks,summary,checkpoint_verification}.json`
- `figures/full_20261027_masks_b8_*.png`（12 张）
- 原始数据快照未变化；`test_unseen_read=false`

## 后续

- 不要重跑本 run，不要覆盖已有 checkpoint。
- 若要继续：先读 `REPORT.md` 和 `PROTOCOL.md`；优先更高点云密度/表面重建、
  侧向动作监督、相同协议多 seed。不要引入 target 泄漏，不要打开 Test-Unseen。
