# Stage 07 审计报告：数据、训练、输入与闭环失败

- 状态：**OPEN — DIAGNOSIS COMPLETED**（采样/覆盖问题没有被解决，只是被诊断清楚）
- 范围：Stage 00–06 的静态审计 + Stage 06 已存数据的再分析
- **本阶段没有重新训练、没有修改模型或输入、没有重跑 Stage 06、没有读取 Test-Unseen**
- 子报告：
  [`DATA_LINEAGE.md`](DATA_LINEAGE.md) ·
  [`TRAINING_SAMPLING_AUDIT.md`](TRAINING_SAMPLING_AUDIT.md) ·
  [`FEATURE_LEAKAGE_AUDIT.md`](FEATURE_LEAKAGE_AUDIT.md) ·
  [`CLOSED_LOOP_FAILURE_AUDIT.md`](CLOSED_LOOP_FAILURE_AUDIT.md) ·
  [`LITERATURE_PROTOCOL_COMPARISON.md`](LITERATURE_PROTOCOL_COMPARISON.md)

## 1. 一句话结论

Stage 06 的 **SR 23.7%** 主要不是"STOP 不会停"，而是**模型在偏离教师轨迹后不会修正**：
闭环里 70.4% 的任务从未进入目标 20 m，越界占 43.3%；越界前 5 步目标平均在身后 **128.7°**，
而模型实际转向率只有 **0.04%**。训练数据里目标在身后的状态只占 **0.65%**，
闭环里却占 **20.6%**——这是典型的模仿学习协变量偏移，叠加一个被 91.6% Forward 先验主导的 `argmax`
策略。与此同时，12 维目标信念在匹配成功时几乎是"目标偏移真值"（方向误差中位 4.4°、距离误差约 8 m），
这既是任务的重大简化，也是模型主要依赖的信号。

## 2. 已确认的事实（按严重程度）

| # | 结论 | 证据 | 严重程度 | 性质 |
|---|---|---|---|---|
| 1 | 闭环失败的主因是"到不了目标"，不是 STOP | OSR 29.6%；进入 20 m 的 1,578 条中 1,260 条成功（79.9%） | **高** | 已确认 |
| 2 | 训练/闭环状态分布严重不匹配：目标在身后 0.65% vs 20.6% | `CLOSED_LOOP_FAILURE_AUDIT.md` §2 | **高** | 已确认 |
| 3 | 越界前目标在身后，模型仍几乎总选 Forward | 转向概率质量 13–18%，实际转向率 0.04–0.23% | **高** | 已确认 |
| 4 | 12 维信念近似目标偏移真值 | 与真实 dx/dy/距离相关 0.987–0.996，方向误差中位 4.4° | **高（解释性）** | 已确认 |
| 5 | 模型强依赖信念 | 信念置零后准确率 0.942→0.132（推理期扰动，非消融） | 中 | 有证据的推断 |
| 6 | 低有效目标 patch <0.5 的任务全部失败 | 0/218；≥0.75 子集 SR 26.2% | 中 | 已确认（相关性） |
| 7 | 训练采样偏向短 episode / 近目标状态 | 短任务占抽样 33.0%（池中 15.6%）；>200 m 状态占 13.9%（池中 21.7%） | 中 | 已确认 |
| 8 | best checkpoint 的选择被评估噪声主导 | 每 epoch 仅 60 个 STOP；epoch 1 高于平台期约 3σ | 中 | 有证据的推断 |
| 9 | 03 的人工路径跟随标签已被 03B 取代，02 的质量分未参与训练筛选 | 05 缓存 `episodes_total=22002`；03 标签下游引用为 0 | 低 | 已确认 |
| 10 | nodata 一律按"可飞"处理 | 03B/04 共同假设 | 中 | 未验证的假设 |
| 11 | 低空避障没有被检验 | 5,324 条只有 1 次碰撞；5 个 1 m 单元高于巡航高度 | 低（对本任务） | 已确认 |
| 12 | 训练/闭环特征路径一致，无坐标/距离/未来动作泄漏 | 归一化 max_abs_diff=0；因果性测试；输入键检查 | — | 已确认（无问题） |

## 3. 哪些属于"合理但必须写清的任务简化"

1. **目标图像来自同一张地图、同一比例尺、以真实目标为中心裁剪。** 这是任务定义的一部分，
   不是数据泄漏；但它让"目标定位"退化为同图模板检索。
2. **12 维信念 ≈ 带噪声的目标偏移向量。** 模型实际上拿到了一个误差约 8 m 的 GPS-to-goal。
   Stage 06 的 SR **不能**与官方 CityNav 的语言指令导航 SR 直接比较（见
   `LITERATURE_PROTOCOL_COMPARISON.md`）。
3. **教师是"已知终点"的规划器。** 训练标签是近最优路径，学生面对的是部分可观测问题；
   这不是错误，但"教师 100% 成功"不能当作难度证据。
4. **固定 50 m 事实上越过几乎所有建筑。** 避障分支在 03B 只触发 1 条、闭环只有 1 次碰撞；
   本任务实际上不检验避障。

## 4. 仍然没有充分证据的问题

- 若把目标图像换成跨视角/跨地图，"信念近似 GPS"的优势会消失多少？
- 如果加入"目标在身后"的监督（DAgger / 数据增强），SR 能提升多少？——**未做实验**。
- nodata 区域是否藏有未测绘结构？——只有共同假设，没有独立数据。
- 转向率下降有多少来自 `argmax` 的类别先验、多少来自视觉/信念信息不足？——需要消融。
- 模型对历史、当前视觉、目标嵌入的因果重要性——**本次没有做重新训练的消融**，
  只有推理期置零的提示（`FEATURE_LEAKAGE_AUDIT.md` §5），不能据此下结论。

## 5. 下一轮消融矩阵（建议，未执行）

固定 seed 42、30 epoch、同一数据与超参；每个配置只改一个输入，报告教师强制指标 + 闭环 SR/OSR/STOP 条件成功率。

| ID | 配置 | 要回答的问题 |
|---|---|---|
| A0 | 当前完整模型（基线） | — |
| A1 | belief 置零 + 可学习的 null 向量（重新训练） | 去掉近 GPS 信号后还能导航吗？ |
| A2 | 只保留 belief（去掉当前视觉与历史） | 信念单独是否足以完成？ |
| A3 | 去掉当前视觉 | 局部视觉对 STOP/转向的贡献 |
| A4 | 去掉历史（mask 全 0） | 时序上下文贡献 |
| A5 | 去掉目标嵌入 | 目标图像是否被信念完全替代 |
| A6 | 目标图像来自**另一张地图**的同位置裁剪 | 同图便利消失后 SR 下降多少 |
| A7 | 教师数据加入"目标在身后"的恢复片段（DAgger 式重标注） | 协变量偏移是否是主因 |
| A8 | oracle belief（真实目标偏移） | SR 上界，用于校准信念误差的影响 |

注意：A1–A8 都需要重新训练，**本次不执行**；不得根据结果人为挑选"最优模态"再回改 Stage 05/06 的结论。

## 6. 可复现脚本

```bash
python -I -u experiments/07_sampling_coverage_open_issue/scripts/audit_training_sampling.py
python -I -u experiments/07_sampling_coverage_open_issue/scripts/audit_closed_loop_failure.py
python -I -u experiments/07_sampling_coverage_open_issue/scripts/probe_modalities.py
python -I -u experiments/07_sampling_coverage_open_issue/scripts/make_lineage_figure.py
```

结果：`results/training_sampling.json`、`results/failure_audit.json`、
`results/modality_probe.json`、`results/sampling_tables.csv`、`results/failure_tables.csv`；
图：`figures/data_lineage.png`、`figures/sampling_coverage.png`、`figures/failure_audit.png`。

## 7. 优先级建议

1. **P0 — 修正训练分布**：在"目标在身后 / 偏离后恢复"的状态上获得监督（DAgger 式教师重标注，
   或按方位分层的重采样）。这是唯一直接针对主因的方向。
2. **P1 — 重新定义评估子集**：checkpoint 选择不要再依赖每 epoch 60 个 STOP 的子集；
   用全量验证或至少 4× 更大的 STOP 样本。
3. **P2 — 明确任务边界**：在论文/报告中把"同图目标图像 + 候选匹配"的便利写清楚，
   与官方 VLN 分开比较；跨地图目标图像作为更强的设定单独评测。
4. **P3 — 采样均衡**：按长度/距离分层采样，避免短任务状态被 2 倍过采样。
5. **P4 — 数据质量假设**：对 nodata 区域做一次独立抽检；低有效目标任务单列。

**Stage 07 不因本报告而关闭**：问题已被诊断，但没有被解决。
