# E2-FOV-Pose 延长训练（2751 → 5502）— 状态

- 更新时间：2026-10-28 20:55
- 状态：**训练与验证已完成**
- 运行目录：`runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_ext5502_seed42/`
- 配置：`experiments/visual_encoder_citynav/ablations/20261028/configs/e2_fov_pose_ext5502.yaml`
- 分支：`exp/visual-encoder-ablation-20261028`

## 训练

| 项 | 值 |
|---|---|
| 协议 | 与 `full_20261027_masks_b8` 相同，seed 42 |
| 初始化 | `data_cache/weights/shared_encoder.pt`（SHA-256 `03ee1295…ec0f7`），与 E2-FOV-Pose 原始运行一致 |
| 恢复点 | `train_state.pt` @ step 2751（含模型/优化器/调度器/CPU+CUDA RNG） |
| 总步数 | **5502 / 5502**（含原有 2751，不是额外 5502） |
| batch size | 8 |
| 验证点 | 4127、5502（`checkpoint_steps_abs`） |
| 断点保存 | 每 200 步 `train_state.pt` |
| 耗时 | 5013.54 s（约 83.6 min，自 2751 起） |
| GPU 峰值 | 5327.26 MiB |
| 原始数据 | 未修改（snapshot digest `22d64d73…`）；Test-Unseen 未读取 |

## 验证结果（Val-Seen 选模）

选模规则：`0.4*macro-F1 + 0.3*xview_top1 + 0.3*R@5`（Val-Seen）。
Val-Unseen 仅诊断，不参与选模。

| step | split | action CE | xview loss | retrieval loss | acc | macro-F1 | xview T1 | xview T5 | R@1 | R@5 | R@10 | score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2751 | Val-Seen | 1.1492 | 3.7326 | 3.0512 | 0.5312 | 0.5326 | 0.0643 | 0.2570 | 0.0859 | 0.2656 | 0.4062 | 0.3120 |
| 4127 | Val-Seen | 1.1342 | 3.6846 | 2.9911 | 0.5234 | 0.5268 | 0.0643 | 0.2249 | 0.0742 | 0.2812 | 0.4648 | 0.3144 |
| **5502** | Val-Seen | 1.1805 | 3.7519 | 3.0302 | 0.5430 | **0.5533** | 0.0522 | 0.1847 | 0.1172 | **0.3359** | **0.4805** | **0.3378** |
| 2751 | Val-Unseen | 1.0113 | 3.9686 | 3.3152 | 0.5742 | 0.5694 | 0.0516 | 0.1746 | 0.0703 | 0.1641 | 0.3242 | 0.2924 |
| 4127 | Val-Unseen | 0.9912 | 3.9150 | 3.1811 | 0.5781 | 0.5796 | 0.0437 | 0.1944 | 0.0938 | 0.2461 | 0.3672 | 0.3188 |
| 5502 | Val-Unseen | 1.0219 | 3.9941 | 3.2561 | 0.5781 | **0.6021** | 0.0516 | 0.1587 | 0.0664 | **0.2617** | **0.3906** | **0.3348** |

六类动作逐类 F1 见 `REPORT_EXT5502.md` 与 `val_comparison.csv`。

## Checkpoint（均为独立文件，不覆盖实验 A 的冻结权重）

| 文件 | step | SHA-256 |
|---|---:|---|
| `best_encoder.pt` | 5502 | `8807035e050d86126aba2cd13d6e99c388d719a5fd087427e1ebceb636ed2fb3` |
| `encoder_final.pt` | 5502 | `2ab54a054b2c88eefc2aab028e779a4c798f597390766b76d1473e4e175b0e17` |
| `checkpoints/encoder_step_004127.pt` | 4127 | — |
| `checkpoints/encoder_step_005502.pt` | 5502 | — |
| `original_best_encoder.pt` | 2751（实验 A 冻结权重副本） | `441de539f1d30f2be68d77a1c954daef5f3ab3bd3501f33b2c9798f89517011b` |
| `train_state.pt` | 5502（模型+优化器+调度器+RNG） | — |

Checkpoint 核验（`verify_checkpoint.py`，结果在
`experiments/visual_encoder_citynav/results/abl_20261028_e2_fov_pose_ext5502_seed42_checkpoint_verification.json`）：

- best / final 严格加载：1.0000，缺失/多余/形状不匹配均为 0，无辅助头 key；
- best 与 final 逐张量一致（最大绝对差 0.0）；
- 相对初始化 L2 变化 1.7189，相对值 2.44%；
- step 2751 的检查点不在本 run 目录，保留在原始 run
  `runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/checkpoints/`。

## 结论

1. 延长到 5502 步后，Val-Seen 选模分数 0.3120 → 0.3378，macro-F1 0.5326 → 0.5533，
   R@5 0.2656 → 0.3359，R@10 0.4062 → 0.4805；Val-Unseen 同向（F1 0.5694 → 0.6021）。
2. 收益主要来自语言-地图检索和动作分类；cross-view 匹配未同步改善
   （Val-Seen Top-1 0.0643 → 0.0522，Top-5 0.2570 → 0.1847）。
3. action CE 在 4127 最低（Val-Seen 1.1342），5502 回升到 1.1805，但选模指标仍最高；
   说明选模规则与 CE 不完全一致，报告时需同时列出。
4. 单 seed，不能据此声称显著改进。

## 产物

- 本目录：`REPORT_EXT5502.md`、`val_comparison.csv`、`comparison_summary.json`、三张 PNG
- `experiments/visual_encoder_citynav/results/abl_20261028_e2_fov_pose_ext5502_seed42_*.{json,csv}`
- 复现：`python experiments/visual_encoder_citynav/ablations/20261028/compare_extended.py`
- 权重不提交（`.gitignore`），仅保留在本机 `runs/`。
