# Stage 17 · 能量预算与截止时间约束

**约束集合只有两条**（2026-11-26 审核决定）：

1. 单个 episode 的 UAV 能耗 ≤ 电池容量；
2. 单次端边协同推理服务时延 ≤ $D_{max}$。

**SR 不是约束，是优化目标。** nDTW 不作为报告指标；OSR 以 20 m 成功半径报告。

---

## 1. 能量预算

### 1.1 配置项

| 配置 | 默认 | 含义 |
|---|---:|---|
| `battery_capacity_wh` | 100 Wh | 电池标称容量 |
| `initial_energy_j` | 360,000 J | 可用能量（= 100 Wh） |
| `battery_safety_margin_j` | 0 J | 安全余量 |
| `task_energy_budget_j` | 360,000 J | 与电池容量一致（审核决定：不设独立任务预算） |
| `energy_budget_j` | 运行时 | $=\min(E_{initial}-E_{margin},\ E_{task})$ |
| `remaining_energy_j` | 运行时 | $E_{budget}-E_{cum}$ |
| `cumulative_energy_j` | 运行时 | $\sum_t E_{total}^{step}$ |

> 采用 `min()` 是为了让"电池容量"与"任务预算"两个旋钮不会静默冲突：
> 任一更小者生效。默认两者相等（360,000 J）。
>
> **物理含义说明**：100 Wh 的单任务约束非常宽松（Stage 15-Extension 在 v=3 m/s 下测得
> 单任务典型能耗约 16.9 kJ，约为容量的 4.7%），因此该约束在默认配置下几乎不会触发。
> 这符合审核意见"能耗不高于电池容量"，但必须在该约束是否作为"有约束力"的项出现在论文中时
> 由 Stage 17-B 的可行性扫描给出定量说明。

### 1.2 两级检查

1. **服务前检查**：若 $E_{coll}(m) > E_{remaining}$，该节点从 `feasible_mec_mask` 剔除。
2. **推理后、执行前的运动检查**：
   $$
   E_{remaining}-E_{coll}-E_{motion}(a)<0 \Rightarrow \text{终止}
   $$
   不执行该导航动作，原因记为 `energy_budget_exhausted`，**不记为成功**。

### 1.3 失败终止规则

| 情况 | `terminated_reason` | 计入 SR | 能量计数 |
|---|---|---|---|
| 无任何可行节点 | `no_feasible_mec` | ❌ | 本步不计费 |
| 选择了不可行节点 | `invalid_mec_action` | ❌ | 本步不计费，记违规 |
| 服务可负担、运动不可负担 | `energy_budget_exhausted` | ❌ | 记 $E_{coll}$，不记运动 |

### 1.4 记录项

- `energy_budget_violation_count`、`remaining_energy_j`（step / episode）
- 分项：`e_hover_j / e_forward_j / e_turn_j / e_enc_j / e_tx_j / e_rx_j`

## 2. 单次服务截止时间

$$
D_{coll}(t,m_t)\le D_{max}
$$

| 配置 | 默认 | 状态 |
|---|---:|---|
| `d_max_ms` | 150 ms | **PROVISIONAL** |
| `include_downlink` | `false` | 审核决定：忽略动作返回时延 |

**R2 重要更新**：加入 Layer2 时延（+2.21 ms）、完整多模态上行（4.28 Mbit）与
50/10 m 通信几何后，$D_{coll}$ 在水平距离 50/100/200/500 m 处分别约为
**71 / 104 / 172 / 362 ms**（Small 模型）。因此：

- 水平距离 ≲ 175 m 时 150 ms 可行；
- 距离 ≳ 175 m 时**所有节点都不可行**，episode 以 `no_feasible_mec` 终止。

150 ms 已不再是一个"几乎不排除节点"的中性取值。**本轮不改动它**（规范要求保持可配置），
但其最终取值必须在 Stage 17-B 前由人工确定（见 docs/12 §4）。

### 2.1 每步候选计算

```python
candidate_service_latency[m]  # D_coll(distance_m, model_type)
deadline_valid_mask[m]        # candidate_service_latency[m] <= d_max_ms
coverage_valid_mask[m]        # distance <= coverage_m
candidate_energy_cost[m]      # E_coll(m) —— 仅服务成本，不含尚未产生的运动能耗
feasible_mec_mask[m]          # coverage ∧ deadline ∧ (E_coll <= remaining)
```

`candidate_energy_cost` 只含决策时**已知**的服务成本，**不含运动能耗**（它依赖 MEC 输出的动作，
决策时尚不存在）。这是因果性要求。

### 2.2 无可行节点

```
if feasible_mec_mask.sum() == 0:
    terminated_reason = "no_feasible_mec"
    success = False
```

不允许静默选择不可行节点，也不允许把约束"软化"进奖励。

## 3. 关于 SR

SR 是优化目标之一，不是约束。由于"最小化 $D_{sum}$"天然偏好提前停止，
SR 必须在**双目标层面**得到保证（例如 Pareto 权衡、约束序优化或 Stage 17-C 选定的其他机制），
而不是靠一条硬约束。这一点在 Stage 17-C 的方法设计中必须显式回答。
