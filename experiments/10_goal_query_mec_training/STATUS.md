# Stage 10：目标查询式 MEC 训练 — 阶段状态

- 分支：`exp/citynav-mec-fixed50m-topdown`
- 目录：`experiments/10_goal_query_mec_training/`
- 起点：`223f11d`（M0 新服务器检查与同步）
- 研究路线（用户 2026-10-31 确认）：
  - **09C-1 连续目标定位（C1）保留为正式基线**，历史结果不修改、不重新解释
  - **C2 永久排除**：不再引入/调试/验证硬门控，仅作负结果归档
  - 在 C1 框架上对比 **Point Query（PQ）** 与 **Cross-Attention（CA）** 两种视觉匹配
  - 10B 闭环比较选出最终方案 → 10C 确定 MEC 输入模态并生成训练数据（含 cross-fitting 泄露审计）
  - 10D 只训练一次 Transformer-M（seed 42，30 epoch，early stopping）
  - 10E 最终闭环验证；完成后停止，等待用户指示
- 固定条件：冻结 ResNet-50（layer2 空间特征为主）、不引入 C2 硬门控、
  不改变 09C-1 连续信念更新主框架、不用 Test-Unseen 调参
- 数据协议：Train-Seen 训练/校准（2,000 状态校准子集，seed 20261030）、
  Val-Seen + Val-Unseen 评测（5,324 任务），沿用 09A 状态采样清单

## 阶段状态

| 阶段 | 名称 | 状态 | 关键结论 | commit |
|---|---|---|---|---|
| 10A | Point Query / Cross-Attention 视觉匹配对比 | **完成（v2 公平版，v1 作废）**：公平性契约下 CA 达 C1 量级（median 3.74 vs 3.27 m）但未超过、门控更弱（fp_oov 23% vs 4.3%）；PQ 因 MSE 监督塌陷不可用（修复需用户同意）；**推荐 C1 进入 10B，CA 可加跑闭环作证** | `10A_point_query_cross_attention/REPORT.md` | 待填 |
| 10B | 几何控制器闭环比较（C1 复用 09C-1 / CA 全量跑证） | **完成**：C1 SR 64.7% vs CA 26.0%（Δ −38.7pp，McNemar p≈0，n=5,324 配对）；CA 过早 STOP 73.9%、视野外误接受 23% 毒化信念；**最终方案 = C1**，PQ/CA 负结果归档 | `10B_geometry_comparison/REPORT.md` | 待填 |
| 10C | MEC 输入模态与训练数据生成 | **完成**：契约 = layer4 当前视觉 + 16 步因果历史 + C1 目标描述子(512) + 09C 信念(10-d) + C1 匹配(5-d)；训练 250,000 状态（教师 200k 逐状态均匀采样 + C1 恢复 50k，DAgger 式）；泄露审计通过（C1 无参数 → cross-fitting 不适用，已论证） | `10C_mec_input_dataset/REPORT.md` + `INPUT_CONTRACT.md` + `LEAK_AUDIT.md` | 待填 |
| 10D | Transformer-M 正式训练（seed 42，仅一次） | 未开始（依赖 10C） | — | — |
| 10E | 最终闭环验证（5 种控制器对比） | 未开始（依赖 10D） | — | — |

## 运行环境备忘

- 服务器：新服务器 `ubuntu`（RTX 5090 32 GB）
- conda 环境：`citynav-mec`（/mnt/hdd2t/conda/envs/citynav-mec）
- 数据根：`CITYNAV_DATA=/mnt/hdd8t/Citynav-MEC/DATA`（pipeline.py 已支持该环境变量）
- tmux 会话：`citynav10_pointquery_mec`
- 10A 训练图像缓存：`10A.../results/cache` → NVMe `/home/tenant2/dataext/stage10/10A_image_cache`（28 GB）
