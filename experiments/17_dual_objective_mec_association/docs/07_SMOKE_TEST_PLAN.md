# Stage 17-B Smoke Test 验收方案（仅编写，**未执行**）

> 本文件是计划，不是结果。Stage 17-A 禁止执行以下任何一项。

## 0. 前置条件

- Stage 17-A 的代码、公式、参数经人工审核通过；
- 待审问题（`08_REVIEW_QUESTIONS.md`）中 P1/P2 级问题全部关闭；
- 固定 `configs/default.json` 的哈希，任何改动都要重新审核。

## 1. 单元级（不加载专家模型，纯公式）

| 编号 | 检查 | 通过标准 |
|---|---|---|
| U1 | $D_{enc}$ 与 Stage 16 一致 | = 5.449515 ms（容差 1e-9） |
| U2 | $D_{up}(d)$ 与 `protocol.comm_latency_ms` 逐点一致 | 500 m 内 20 个采样点完全相等 |
| U3 | $D_{inf}$ 五模型 | 0.495 / 1.498 / 3.967 / 10.003 / 19.972 ms |
| U4 | $D_{down}$ 单调性与量级 | 距离↑ 时延↑；8 B 载荷下 < 1 ms |
| U5 | $P_{fly}(v)$ 三项分解 | $v=0$ 时退化为 $P_0+P_i=168.49$ W |
| U6 | $E_{forward}(v=3)$ | 与手算一致（记录数值） |
| U7 | $E_{turn}$ | = 168.49 J |
| U8 | 单位一致性 | 所有能量字段为 J，所有时延字段为 ms |

## 2. 环境一致性（10 条任务，固定 seed）

| 编号 | 检查 | 通过标准 |
|---|---|---|
| E1 | 归一化/checkpoint 正确性 | 同一状态重放动作与 Stage 16 完全一致 |
| E2 | `preview_action` 与父类实际动作 | `preview_mismatch_count == 0` |
| E3 | 覆盖掩码强制 | 构造覆盖外节点动作 → 必须 `invalid_mec_action`，不得执行 |
| E4 | 截止时间强制 | 构造 $D_{coll}>D_{max}$ 的配置 → 该节点 `deadline_valid=0` |
| E5 | 能量预算强制 | 调小 `task_energy_budget_j` → 出现 `energy_budget_exhausted` 且 SR 不增加 |
| E6 | 无可行节点 | 极限配置下必须 `no_feasible_mec`，不得静默选节点 |
| E7 | 计费正确性 | $\sum E_{total}^{step}$ = episode 的 `cumulative_energy_j`（逐 episode 对账） |
| E8 | 终止后不计费 | Stop 之后无额外步与能量 |
| E9 | 旧指标保留 | `legacy_cumulative_e2e_ms` 与 Stage 16 同状态同动作一致 |

## 2b. R2 新增检查

| 编号 | 检查 | 通过标准 |
|---|---|---|
| R-U1 | IEEE 公式独立实现对比 | 3 m/s = 157.555 W / 262.593 J；5 m/s = 143.614 W / 143.613 J |
| R-U2 | Layer2 FLOPs | 3,318,841,344 FLOPs → 2.212561 ms |
| R-U3 | 上行负载 | 4,281,664 bit；Relation 不在上传清单 |
| R-U4 | 通信几何 | $d_{3D}(500)=501.597$ m，仰角 4.574° |
| R-E1 | 单次前向 | 每个决策步的专家前向次数 == 1（不得因校验能量重复推理） |
| R-E2 | 轨迹等价 | 关闭约束时 Stage 17 与 Stage 16 的逐 step 动作/位置/朝向/成功完全一致 |
| R-E3 | 碰撞计费 | `motion_aborted=true` 时前进能耗为 0，且仍计服务能耗 |
| R-E4 | 越界计费 | 同 R-E3 |
| R-E5 | 无可行节点 | `step(None)` 不得抛 `int(None)`，必须记 `no_feasible_mec` |

## 3. 阈值可行性扫描（只做统计，不做训练）

对 Train/Dev 池各 100 个任务，用 Min-Latency 与 Max-Quality 两个静态策略：

- 统计 $D_{coll}$ 分布（p50/p90/p95/max）；
- 统计单任务 $E_{sum}$ 分布；
- 对一组候选 $D_{max}\in\{50,100,150,200\}$ ms 与 $E_{budget}\in\{10,20,40,80\}$ kJ，
  输出"可行任务比例 / 命中 `no_feasible_mec` 的比例 / SR 变化"；
- **只有存在同时满足 SR ≥ 0.5 且两条约束都有实际约束力（而非 0% 或 100% 违约）的阈值，才允许进入 Stage 17-C 训练。**

## 4. 不做的事情

- 不训练任何 DRL 智能体；
- 不做超参数搜索；
- 不使用 Val-Seen 调整任何参数（Val-Seen 只用于最终报告）。

## 5. 输出物（Stage 17-B）

- `smoke/unit_checks.json`
- `smoke/env_consistency.csv`（10 任务逐步记录）
- `smoke/feasibility_grid.csv`（$D_{max}\times E_{budget}$ 网格）
- `smoke/SMOKE_REPORT.md`（含 U/E 逐项结论与失败原因）
