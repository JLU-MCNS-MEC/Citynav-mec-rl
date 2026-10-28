# 四组消融运行状态（2026-10-28）

- 分支：`exp/visual-encoder-ablation-20261028`
- 启动：2026-10-28 12:10（Asia/Shanghai）
- 初始化权重：`data_cache/weights/shared_encoder.pt`
  SHA-256 `03ee1295f058b1982125e5d9a4647cee87934e7dde697e9be8756a69167ec0f7`
  （重新生成；与旧基线 `8ee6194e…` 不同）
- 数据：28/28 开发 block，28.32 GB；Test-Unseen 6 个 block 未下载、未读取
- 统一：seed 42、2751 步、batch 8、`L_action + 0.5 L_xview + 0.5 L_retrieval`

## 运行进度

| 组 | 配置差异 | 当前步数 | 第一个 Checkpoint（688） |
|---|---|---:|---|
| E1 | 审计基线 | 780 | score 0.1936 / F1 0.2668 / xview@1 0.0161 / R@5 0.2734 |
| E2 | splat=1 | 540 | 未到 |
| E3 | 3 教师步/轨迹 | 690 | score 0.1990 / F1 0.2620 / xview@1 0.0482 / R@5 0.2656 |
| E4 | splat=1 + 3 步 | 450 | 未到 |

## E1 完成（2751/2751）

| 指标 | Val-Seen | Val-Unseen |
|---|---:|---:|
| action accuracy | 0.4023 | 0.4336 |
| macro-F1 | 0.3141 | 0.3488 |
| xview Top-1 / Top-5 | 0.0602 / 0.2490 | 0.0437 / 0.1508 |
| R@1 / 5 / 10 | 0.0820 / 0.2812 / 0.4258 | 0.0781 / 0.1875 / 0.3281 |

- best step 2751，score 0.2281
- 用时 124.8 分钟，GPU 峰值 5621 MiB
- best_encoder SHA-256 `23765e13e91d5be45c6958caf55cb558e9e236a4e4c323e76b8225e5080c26ea`
- encoder_final SHA-256 `1716772788c6b435a31cdea472ebd0d601831d5de569530f27c7b2fe586d0f39`
- 原始数据未变：True；Test-Unseen 未读取：True
- 逐类动作（Val-Seen）：{'Stop': 0.8372093023255814, 'MoveForward': 0.5, 'TurnRight': 0.0, 'TurnLeft': 0.0, 'GoUp': 0.2692307692307692, 'GoDown': 0.6938775510204082}

> 注意：TurnRight / TurnLeft 的逐类准确率仍为 0，与旧基线的已知失败模式一致；
> 这是辅助动作头的诊断结果，不是导航成功率。

## E3 完成（2751/2751，最佳在 688）

| 指标 | Val-Seen | Val-Unseen |
|---|---:|---:|
| action accuracy | 0.3086 | 0.3242 |
| macro-F1 | 0.2620 | 0.2732 |
| xview Top-1 / Top-5 | 0.0482 / 0.1847 | 0.0635 / 0.1508 |
| R@1 / 5 / 10 | 0.1016 / 0.2656 / 0.3867 | 0.0430 / 0.1602 / 0.2188 |

- best step 688，score 0.1990；用时 133.3 分钟
- best_encoder SHA-256 `899c7a62d8a3691334b6cbfe1b169f169e41a41386bf646d477dbf8db6a93a0e`
- 逐类动作（Val-Seen，最佳步）：{'Stop': 0.813953488372093, 'MoveForward': 0.2692307692307692, 'TurnRight': 0.34782608695652173, 'TurnLeft': 0.325, 'GoUp': 0.0, 'GoDown': 0.02040816326530612}
- 与 E1 的差异：E3 学会转向（TurnLeft 0.488 / TurnRight 0.146），但升降为 0；
  E1 相反（TurnLeft/Right 0，GoUp 0.269 / GoDown 0.694）

## 资源

- 四组并行，GPU 显存约 28.5 GB / 97.9 GB
- 瓶颈为 CPU 渲染：四组并行时单样本渲染 0.65–1.23 s（单跑 0.14–0.30 s）
- 预计完成：E1 约 14:35，E3 约 14:55，E2 约 15:45，E4 约 16:40（以实测速率外推）

## 可恢复性

- 每个 Checkpoint（688/1376/2063/2751）同时写 `train_state.pt`
  （模型、优化器、调度器、步数、最佳指标、RNG）
- 启动脚本检测到 `train_state.pt` 时自动加 `--resume`，不从头重跑
- 每个 run 独立 `pid`、`status.txt`、`train.log`、`train_metrics.csv`

## 验证指标说明

- E1–E4 内部可比（同种子、同初始化、同数据、同步数）
- 与旧基线 `full_20261027_masks_b8` 的比较混入两种差异：种子 0→42、
  初始化权重不同；报告中单列，不作为同条件对照
