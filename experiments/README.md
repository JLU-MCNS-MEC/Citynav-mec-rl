# 实验总索引（CityNav-MEC 阶段二）

本目录按固定顺序组织 2026-10-29 起的复核与诊断实验。**实验 06 是
`OPEN / BLOCKED` 的未解决问题，任何涉及 Token 生成、采样、压缩或训练覆盖率的
后续任务都必须先读它。**

## 固定原则

1. 新训练实验只训练 Transformer-M MEC，固定 seed 42/43；不再训练 Small。
   Small 仅使用已有 checkpoint 作为对照。
2. 使用已有 Token 数据和冻结的 E2-FOV-Pose 编码器；暂不重训视觉编码器。
3. 闭环只测 `small:full` 与 `transformer_m:full`，不跑 `map_free` 闭环。
4. 不使用类别加权、pos_weight、focal loss、人工动作标签或额外目标距离输入。
5. `target_positions[-1]` 的真实距离只用于监督标签构建和评测，不能作为
   MEC 推理输入，也不能用于闭环强制 STOP。
6. 不读取 `test_unseen`；不覆盖原始数据、Token 缓存、checkpoint 和已有结果。

## 实验索引

| 编号 | 目录 | 目的 | 依赖 | 状态 | 结果 |
|---|---|---|---|---|---|
| 01 | `01_dataset_length_and_horizon_audit/` | 官方口径、轨迹长度、教师动作数、步数上限 | 无 | **完成** | 100 步只覆盖约 20%；建议固定上限 500；见 `REPORT.md` |
| 02 | `02_full_closed_loop_recheck/` | small:full / transformer_m:full 单轮分层闭环复测（每划分 24 条，固定 500 步） | 01 | **运行中**（预算 4–5 h） | 待填写 |
| 03 | `03_v82_citynav_dataflow_audit/` | MEC-V8.2 与 CityNav 数据流逐项对照 | 01–02 | 计划中 | 待填写 |
| 04 | `04_stop_history_consistency/` | STOP 监督、历史因果、GAB 状态一致性修复与诊断缓存 | 03 | 计划中 | 待填写 |
| 05 | `05_transformer_m_overfit_diagnostic/` | Transformer-M 小规模过拟合与决策诊断 | 04 | 计划中 | 待填写 |
| 06 | `06_token_sampling_design_open_issue/` | Token 采样与动作覆盖率的未解决问题 | 02–05 的证据 | **OPEN / BLOCKED** | 见 `OPEN_ISSUE.md` |

## 依赖关系

```
01 ──> 02 ──> 03 ──> 04 ──> 05
                         │
                         └──> 06（持续未解决；阻止 Token v2 生成）
```

## 汇报与提交约定

每个实验完成后必须：

1. 在对应目录写 `REPORT.md`：假设、设置、数据规模、指标、结果、失败案例、
   结论、下一步；明确区分已证实事实 / 未验证假设 / 待实验问题。
2. 更新本索引中的状态和结果。
3. 运行相关测试，检查 `git status`，提交并推送代码与报告（不提交缓存、权重、
   原始数据和 GPU 日志）。
4. 回复中给出 commit SHA、GitHub 链接和阶段结论。
