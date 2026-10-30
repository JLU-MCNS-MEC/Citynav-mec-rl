# 实验总索引（CityNav-MEC 阶段二 · 固定 50 m 俯视视觉目标导航）

本目录按固定顺序组织 **2026-10-29 起**在分支 `exp/citynav-mec-fixed50m-topdown` 上的实验。
总览与决策见 [`STATUS.md`](STATUS.md)。

> **历史索引**：2026-10-29 之前在旧分支（`exp/visual-encoder-ablation-20261028`）上做的
> Token / FPV / BERT 六动作实验，其索引是那份文档里的旧表格，结论汇总在
> [`STATUS_2026-10-29.md`](STATUS_2026-10-29.md)。旧目录（`01_dataset_length_and_horizon_audit`、
> `02_full_closed_loop_recheck`、`03_v82_citynav_dataflow_audit`、
> `06_token_sampling_design_open_issue`）**保留原地、不覆盖**，仅作历史证据。

## 固定原则

1. UAV 固定离地 **50 m**（`z = GROUND_LEVEL[map] + 50`），俯视 RGB 观测，不再逐步渲染 PLY 第一视角。
2. 冻结 ImageNet 预训练 ResNet-50；不训练视觉编码器。
3. **无语言输入、无 BERT、无网络状态**；不输入真实目标坐标、目标 ID、目标距离、未来动作。
4. 动作四类：`STOP` / `Forward 5 m` / `TurnLeft 30°` / `TurnRight 30°`。
5. 只训练 Transformer-M，固定 **seed 42**（seed 43 不执行）；不使用 Small；不使用类别加权、pos_weight、focal loss。
6. 教师使用实验 03B 的**直接目标导向教师**；人工轨迹只提供起点位姿，不提供中间路径。
7. 真实目标坐标只用于离线标签与评测，不进入 MEC 推理。
8. SR 是主要指标；`test_unseen` 于 2026-10-30 解封，仅用于**已训练模型**的闭环评测，不参与训练或调参。
9. 不覆盖已有报告：修正一律新增 `ADDENDUM_*.md`。

## 实验索引

| 编号 | 目录 | 目的 | 依赖 | 状态 | 结果 |
|---|---|---|---|---|---|
| 00 | `00_archive_and_token_cleanup/` | 归档旧 Token 数据并删除（用户确认后执行） | 无 | **完成** | 删除 13.49 GB，残留 0；见 `REPORT.md` |
| 00R | `00_archive_and_token_cleanup/RECOVERY_2026-10-30.md` | 中断恢复检查与文档一致性修复 | 00 | **完成** | 无遗留进程；文档与 STATUS 已对齐 |
| 01 | `01_topdown_asset_audit/` | 俯视栅格 / GROUND_LEVEL / PLY 资产审计 | 无 | **完成** | 34 栅格、28 图 PLY、固定高度约定；见 `REPORT.md` |
| 02 | `02_trajectory_quality_audit/` | Train-Seen 人工轨迹质量审计（50 m 口径） | 01 | **完成** | 高 14,237 / 中 4,284 / 低 3,481；见 `REPORT.md` |
| 03 | `03_fixed50m_four_action_teacher/` | 沿人工轨迹的四动作教师（旧教师，保留） | 02 | **完成** | 19,105 / 22,002 可重放；见 `REPORT.md` |
| 03B | `03b_target_directed_teacher/` | **直接目标导向教师**（只看起点位姿 + 最终目标） | 01、02 | **完成** | 22,002/22,002 成功；修复 15° 极限环；见 `REPORT.md` |
| 04 | `04_visual_goal_global_map_pipeline/` | 视觉目标 + 全局地图候选 + 目标信念输入管线 | 01 | **完成（修正）** | 空间特征 7×7×2048、高程掩码、Recall@K 已修正；`ADDENDUM_2026-10-30.md` |
| 05 | `05_transformer_m_training/` | Transformer-M 正式训练（seed 42，30 epochs） | 03B、04 | **完成** | best Macro-F1 0.784（epoch 1）；Val-Seen STOP 0.485；见 `REPORT.md` |
| 06 | `06_closed_loop_success_rate/` | 闭环 SR 验证（Val-Seen / Val-Unseen） | 05 | **完成** | 5,324 条，SR 23.7%，OSR 29.6%；见 `REPORT.md` |
| 07 | `07_sampling_coverage_open_issue/` | 采样与覆盖率未解决问题（保持 OPEN，已完成诊断） | 03B–06 | **OPEN — DIAGNOSIS COMPLETED** | 见 `AUDIT_REPORT.md`；主因：协变量偏移 + argmax 坍缩 |
| 08 | `08_belief_only_geometric_navigation/` | 仅用 12 维目标信念的几何导航基线 | 04、06 | **完成** | Top-1 44.9% / Top-5 40.2% vs Transformer-M 23.7%；见 `REPORT.md` |

## 依赖关系

```
00 ──> 01 ──> 02 ──> 03（旧教师，保留）
        │      │
        │      └──> 03B（直接目标导向教师）──┐
        │                                    ├──> 05 ──> 06 ──> 07（OPEN）
        └──> 04（输入管线，已修正）──────────┘
```

## 汇报与提交约定

每个实验完成后必须：

1. 在对应目录写 `REPORT.md`：假设、设置、数据规模、指标、结果、失败案例、结论、下一步；
   明确区分已证实事实 / 未验证假设 / 待实验问题。
2. 更新 [`STATUS.md`](STATUS.md) 与本索引的状态列。
3. 运行相关测试，检查 `git status`，提交并推送代码与报告（不提交缓存、权重、原始数据和大日志）。
4. 回复中给出 commit SHA、GitHub 链接和阶段结论。
