# 四组消融运行状态（2026-10-28，全部完成）

- 分支：`exp/visual-encoder-ablation-20261028`
- 启动：2026-10-28 12:10；最后一组完成：15:10
- 初始化权重：`data_cache/weights/shared_encoder.pt`
  SHA-256 `03ee1295f058b1982125e5d9a4647cee87934e7dde697e9be8756a69167ec0f7`
  （重新生成；与旧基线 `8ee6194e…` 不同）
- 数据：28/28 开发 block，28.32 GB；Test-Unseen 6 个 block 未下载、未读取
- 统一：seed 42、2751 步、batch 8、`L_action + 0.5 L_xview + 0.5 L_retrieval`

## 结果总表

| 组 | 配置差异 | 完成 | 最佳步 | score | VS acc | VS F1 | VS xv1 | VS R@5 | VU acc | VU F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 审计基线 | 2751/2751 | 2751 | 0.2281 | 0.4023 | 0.3141 | 0.0602 | 0.2812 | 0.4336 | 0.3488 |
| E2 | splat=1 | 2751/2751 | 2063 | **0.2288** | **0.4102** | **0.3247** | 0.0602 | 0.2695 | **0.4414** | **0.3638** |
| E3 | 3 教师步/轨迹 | 2751/2751 | 688 | 0.1990 | 0.3086 | 0.2620 | 0.0482 | 0.2656 | 0.3242 | 0.2732 |
| E4 | splat=1 + 3 步 | 2751/2751 | 2751 | 0.2062 | 0.3438 | 0.2774 | 0.0361 | 0.2812 | 0.3633 | 0.2965 |

- 用时：E1 124.8 min、E2 170.6、E3 133.3、E4 180.6；GPU 峰值均 5,621 MiB
- 原始数据未变、Test-Unseen 未读取（四组均为 True）

## 关键结论

1. E2 在动作指标与 Val-Unseen 上略优于 E1，但综合分差距仅 0.0007，单种子，不构成显著改进。
2. E3/E4 的多步采样未带来收益：最佳 Checkpoint 提前，训练损失几乎不下降。
3. 转向与升降存在此消彼长：E3 学会转（0.35/0.33）但升降为 0；E1/E2 相反。
4. 四组与旧基线不可同条件比较（旧基线 seed 0 + 不同初始化）。

详见 [`REPORT.md`](REPORT.md)。

## 产物

- `results/ablation_comparison_20261028.{csv,json}`
- `results/e1_final_summary.json` … `e4_final_summary.json`
- `figures/ablation_validation_20261028.png`、`ablation_curves_20261028.png`、
  `ablation_training_losses_20261028.png`
- 权重仅在服务器 `runs/visual_encoder_citynav/abl_20261028_*_seed42/best_encoder.pt`

## 可恢复性

- 每个 Checkpoint 同时写 `train_state.pt`；`--resume` 已验证
- 每组独立 `pid`、`status.txt`、`train.log`、`train_metrics.csv`
