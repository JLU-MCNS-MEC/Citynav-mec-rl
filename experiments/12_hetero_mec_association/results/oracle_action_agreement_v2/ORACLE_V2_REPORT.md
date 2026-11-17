# Oracle 关联规则修正报告（v2：仅动作一致）

生成日期：2026-11-17　分支：`exp/citynav-mec-fixed50m-topdown`
基线提交：`bc6ed854`（Stage 12 原始结果保留）
新结果目录：`experiments/12_hetero_mec_association/results/oracle_action_agreement_v2/`

---

## 1. 修正原因与定义

Stage 12 原始 Oracle（`oracle_e2e_tiebreak_v1`）在两处使用了 E2E 时延：

1. 多个专家动作与教师动作完全一致时，选择瞬时 E2E 最低的节点；
2. 无专家完全一致时，先按 `P(a*)` 排序，概率相同时再用 E2E 作平局裁决。

这与研究定义不符：Oracle 只应依据“专家导航动作与教师参考动作的一致程度”关联，
不得使用通信/计算/E2E/距离/质量先验/RL。

新版 `oracle_action_agreement_v2` 的规则：

```python
agreement_i = int(argmax(P_i) == a*)
score_i     = (agreement_i, P_i(a*))
# 按 score 字典序最大化；最终平局按固定节点编号 G1 < G2 < ... < G10
```

两处 E2E tie-break 均已删除。选择函数（`scripts/oracle_selection.py`）不接收
E2E、通信、推理、距离或质量字段；这些字段只在选择完成后写入 journal，用于
事后统计。教师动作仍由 Stage-03B Direct Teacher 从真实状态计算，未修改教师算法。

## 2. 审计字段与测试

每个 Oracle v2 决策步记录：

`teacher_action`、`selected_node`、`selected_model`、`selected_action`、
`selected_teacher_action_probability`、`selected_agreement`、
`available_candidate_count`、`exact_agreement_candidate_count`、
`selected_node_compute_tflops`、`selected_node_comm_ms`、
`selected_node_inf_ms`、`selected_node_e2e_ms`、`tie_break_used`。

逐任务结果：

- `journal.part00.jsonl`：每 episode 一行（汇总指标）；
- `steps.part00.jsonl`：每决策步一行（含全部审计字段）；
- `episodes/<id>.json`：完整 episode 结果与 trajectory。

单元测试 `tests/test_oracle_selection_v2.py`：9/9 通过，覆盖多专家一致、
无专家一致、概率平局（两处）、延时/距离/质量字段不影响选择、算力档位不影响
选择、动作分布变化切换专家、审计字段与空候选。Stage 12 全部测试 22/22 通过。

## 3. 四组 Oracle v2 结果（Val-Seen 2,498，seed 42，20 m，200 步）

| Group | SR | OSR | SPL | NE mean | E2E mean / P50 / P95 / P99 | MEC Inf mean / P50 / P95 | Comm mean / P50 / P95 | UAV Enc |
|---|---:|---:|---:|---:|---|---|---|---:|
| two_homo | 60.33% | 60.85% | 0.528 | 51.94 | 72.58 / 71.89 / 106.13 / 116.09 | 4.23 / 4.30 / 7.29 | 62.90 / 62.51 / 96.72 | 5.45 |
| two_hetero | 60.33% | 60.85% | 0.528 | 51.94 | 72.87 / 72.27 / 106.52 / 116.32 | 4.53 / 4.60 / 7.43 | 62.90 / 62.51 / 96.72 | 5.45 |
| five_homo | 76.02% | 76.42% | 0.656 | 37.01 | 69.68 / 68.50 / 98.11 / 106.32 | 8.07 / 7.80 / 12.69 | 56.16 / 54.65 / 84.45 | 5.45 |
| five_hetero | 76.02% | 76.42% | 0.656 | 37.01 | 68.54 / 67.33 / 96.93 / 105.70 | 6.93 / 6.58 / 11.77 | 56.16 / 54.65 / 84.45 | 5.45 |

（时延单位 ms；P50/P95 由每 episode 的逐步均值跨任务统计。）

专家调用与动作一致率：

| Group | Small | Medium | Transformer-S | Transformer-M | Large | 精确一致率 | fallback | tie-break | 平均一致候选数 | 平均可用候选数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| two_homo | 58.3% | 0.0% | 0.0% | 41.7% | 0.0% | 47.13% | 52.87% | 100.00% | 4.58 | 9.98 |
| two_hetero | 58.3% | 0.0% | 0.0% | 41.7% | 0.0% | 47.13% | 52.87% | 100.00% | 4.58 | 9.98 |
| five_homo | 19.5% | 20.9% | 21.2% | 12.2% | 26.2% | 51.94% | 48.06% | 99.49% | 4.94 | 9.98 |
| five_hetero | 19.5% | 20.9% | 21.2% | 12.2% | 26.2% | 51.94% | 48.06% | 99.49% | 4.94 | 9.98 |

节点选择集中度：two 组为 G1 57.6% / G2 41.6% / G3 0.7% / G4 0.1%；
five 组为 G5 26.2% / G3 21.2% / G2 20.9% / G1 19.1% / G4 12.2% / G6 0.4% / G9 0.1%。
原因是固定编号规则在每个模型类型内部总是选择最小 G 编号节点。

Failure taxonomy（v2 与 v1 完全相同）：

| Group | success | out_of_bounds | premature_stop | never_stop | reached_but_not_stopped | collision |
|---|---:|---:|---:|---:|---:|---:|
| two_homo / two_hetero | 1507 | 367 | 380 | 230 | 13 | 1 |
| five_homo / five_hetero | 1899 | 156 | 202 | 230 | 10 | 1 |

## 4. 新旧 Oracle 对比

| Group | v1 SR | v2 SR | ΔSR | v1 E2E | v2 E2E | ΔE2E | v1 Inf | v2 Inf | v1 Comm | v2 Comm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| two_homo | 60.33% | 60.33% | **0.00** | 36.65 | 72.58 | **+35.93** | 3.21 | 4.23 | 27.99 | 62.90 |
| two_hetero | 60.33% | 60.33% | **0.00** | 36.03 | 72.87 | **+36.84** | 2.54 | 4.53 | 28.03 | 62.90 |
| five_homo | 76.02% | 76.02% | **0.00** | 41.84 | 69.68 | **+27.84** | 4.68 | 8.07 | 31.71 | 56.16 |
| five_hetero | 76.02% | 76.02% | **0.00** | 41.03 | 68.54 | **+27.51** | 3.82 | 6.93 | 31.76 | 56.16 |

**SR、OSR、SPL、NE、成功数、failure taxonomy 在 v1 与 v2 之间逐任务完全一致**
（1,507 / 1,899 成功；越界 367/156；过早停 380/202；从未停 230/230；
到达未停 13/10；碰撞 1/1）。只有时延与节点/专家比例变化。

专家比例变化：

| Group | v1 | v2 |
|---|---|---|
| two_homo | Small 65.0 / TM 35.0 | Small 58.3 / TM 41.7 |
| two_hetero | Small 62.0 / TM 38.0 | Small 58.3 / TM 41.7 |
| five_homo | Large 9.6 / Medium 28.7 / Small 24.6 / TS 20.4 / TM 16.6 | Large 26.2 / TS 21.2 / Medium 20.9 / Small 19.5 / TM 12.2 |
| five_hetero | Large 14.9 / Medium 27.8 / Small 23.5 / TS 20.9 / TM 12.9 | Large 26.2 / TS 21.2 / Medium 20.9 / Small 19.5 / TM 12.2 |

## 5. 八个分析问题

1. **去掉 E2E tie-break 后 SR 变化？**
   **完全没有变化**：two 组 60.33% → 60.33%，five 组 76.02% → 76.02%，
   逐任务成功/失败集合完全相同。原因：v1 的两处 tie-break 只在“动作已等于
   a*”的节点之间或“同一 `P(a*)` 最大值模型”的节点之间选择，执行的导航动作
   不变；v2 改变的是选哪个节点，不是动作。
2. **E2E 是否明显上升？**
   是。+35.93 / +36.84 / +27.84 / +27.51 ms（two_homo / two_hetero /
   five_homo / five_hetero）。
3. **MEC 推理时延是否变化？**
   上升 1.02 / 1.99 / 3.39 / 3.11 ms。v2 更偏向 Large（five 组 26.2%）和
   TM（two 组 41.7%），推理均值升高。
4. **Communication 是否变化？**
   上升 34.91 / 34.87 / 24.45 / 24.40 ms。固定编号规则把选择集中到
   G1/G2（two 组）或 G1–G5（five 组），这些节点通常比 v1 选中的近处节点远。
5. **Oracle 是否更多选择某些专家？**
   two 组：TM 35.0→41.7%（two_homo）、38.0→41.7%（two_hetero）；
   five 组：Large 9.6→26.2%（five_homo）、14.9→26.2%（five_hetero），
   Medium/Small/TM 下降。原因是一致集合中 `P(a*)` 最高的模型发生变化，
   且节点顺序固定；执行动作仍为 a*。
6. **动作一致率是否保持？**
   是。v2 精确一致率 47.13%（two）/ 51.94%（five），与 v1 的
   “至少一个专家一致”率完全相同；fallback 52.87% / 48.06%，
   tie-break 100% / 99.49%（同模型多节点必然共享最高 score）。
7. **两模型与五模型变化是否一致？**
   方向一致：SR 不变、E2E/通信/推理上升。五模型的推理增幅更大
   （+3.39/+3.11ms，因 Large 使用率上升），但通信增幅更小
   （+24.4ms vs +34.9ms），因为 v1 五模型本身已选较远节点。
8. **与 Max-rate、Max-quality、PPO、DQN 的差距如何变化？**
   - **SR**：v2 60.33%（two）/ 76.02%（five），比 max_rate（44.96/41.07）
     高 **+15.37/+34.95pp**，比 max_quality（53.56/53.56）高
     **+6.77/+22.46pp**，比最佳 PPO（39.63/37.55）高
     **+20.70/+38.47pp**，比最佳 DQN（46.60/39.83）高
     **+13.73/+36.19pp**。
   - **E2E**：v2 68.5–72.9ms，比 max_rate（37.4–41.0）高 **+27.5–35.4ms**，
     比 max_quality（43.5–53.2）高 **+15.3–29.1ms**，比最佳 PPO
     （36.3–40.5）高 **+28.0–35.0ms**，比最佳 DQN（51.5–57.5）高
     **+11.0–21.0ms**。
   - 因此 v2 是**质量上界**，不是时延上界；v1 的 E2E 接近静态策略是
     E2E tie-break 带来的“时延优化”，而非关联规则本身。

## 6. 特权信息与不可部署性

Oracle v2 使用：真实 UAV 位置与航向、真实目标坐标、高程障碍图、Stage-03B
教师规则、所有可用专家对当前输入的动作概率分布。它不使用真实目标坐标之外的
任务信息，也不使用未来轨迹。由于目标坐标与教师动作对模型不可见，v2 不可部署。
它给出的是“在固定冻结专家池与教师动作偏好下，关联层可达到的动作一致率上界”，
不是严格 SR 数学上界，也不是 E2E 下界。

## 7. 产物

- 代码：`scripts/oracle_selection.py`、`scripts/run_association_closed_loop.py`
  （新增 `oracle_action_agreement_v2` 分支）、`scripts/analyze_oracle_v2.py`、
  `scripts/run_oracle_v2_all.sh`
- 测试：`tests/test_oracle_selection_v2.py`
- 四组逐任务结果：`results/oracle_action_agreement_v2/<group>/`
  - `journal.part00.jsonl`、`steps.part00.jsonl`、`episodes/*.json`、
    `summary.json`
- 汇总：`results/oracle_action_agreement_v2/summary/oracle_v2_summary.csv`、
  `oracle_v1_vs_v2.csv`、`oracle_v2_failures.csv`
- 图：`results/oracle_action_agreement_v2/figures/`
  - `oracle_sr_e2e.png`、`oracle_inf_comm.png`、
    `oracle_expert_share.png`、`oracle_agreement.png`
- 日志：`results/oracle_action_agreement_v2/logs/<group>.log`

复现：

```bash
bash experiments/12_hetero_mec_association/scripts/run_oracle_v2_all.sh
$PY experiments/12_hetero_mec_association/scripts/analyze_oracle_v2.py
```

## 8. 结论

1. **v1 的 E2E tie-break 从未改变导航**：v1 与 v2 的 SR/OSR/SPL/NE/成功数/
   failure taxonomy 逐任务完全相同。它的唯一作用是在“动作已与教师一致”的
   节点中挑更近的一个，从而降低通信时延。
2. **v2 是更严格的关联定义**：只按动作一致性与 `P(a*)` 关联，时延完全作为
   结果而非选择依据。代价是 E2E 上升约 27.5–36.8ms，通信上升约 24.4–34.9ms。
3. **质量上界不变**：two 组 60.33%、five 组 76.02%；五模型的动作多样性仍然
   是最大的关联上界来源。
4. **v1 与 v2 不可混为同一算法的重复实验**：它们共享教师与专家池，但关联
   规则不同；本报告将 v1 标记为 `oracle_e2e_tiebreak_v1`，v2 标记为
   `oracle_action_agreement_v2`。
5. 如果论文需要“纯关联上界”，应引用 v2 的 SR 与时延；如果需要“在关联
   正确的前提下尽量降低时延”的特权参照，可引用 v1，但必须明确其 E2E 来自
   时延 tie-break。
