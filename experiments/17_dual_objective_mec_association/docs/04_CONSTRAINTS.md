# Stage 17 · 能量预算与截止时间约束

## 1. 能量预算

### 1.1 配置项（电池容量与任务预算严格分离）

| 配置 | 默认 | 含义 |
|---|---:|---|
| `battery_capacity_wh` | 100 Wh | 电池**标称容量**（仅作参考量，不是任务阈值） |
| `initial_energy_j` | 360,000 J | 任务开始时可用能量（= 100 Wh） |
| `battery_safety_margin_j` | 0 J | 安全余量，从初始能量中扣除 |
| `task_energy_budget_j` | 20,000 J | **任务级允许消耗上限** |
| `remaining_energy_j` | 运行时 | $E_{initial}-E_{margin}-E_{cum}$ |
| `cumulative_energy_j` | 运行时 | $\sum E_{total}^{step}$ |

> 100 Wh **不能**直接当作任务阈值。默认 20,000 J 的来历：Stage 15-Extension 在 v=3 m/s、含推进的模型下测得 Min-Latency 单任务平均约 16.9 kJ，
> 本阶段能耗模型（服务悬停 + 前进 + 转向）与其同量级，故先取 20 kJ ≈ 1.2 倍典型值。**该值是 PROVISIONAL，必须由人工审核或由后续 Smoke Test 的分布重新标定。**

### 1.2 两级检查

1. **服务前检查**：对每个候选节点 $m$，若 $E_{coll}(m) > E_{remaining}$，则该节点从 `feasible_mec_mask` 中剔除（`energy_valid = False`）。
2. **推理后、执行前的运动检查**：MEC 返回导航动作 $a$ 后，
   $$
   E_{remaining}-E_{coll}-E_{motion}(a)<0 \;\Rightarrow\; \text{终止}
   $$
   此时**不执行**该导航动作，终止原因记为 `energy_budget_exhausted`，并**不允许记为成功**。

### 1.3 失败终止规则

| 情况 | `terminated_reason` | 计入 SR | 能量计数 |
|---|---|---|---|
| 无任何可行节点 | `no_feasible_mec` | ❌ | 本步不计费 |
| 选择了不可行节点 | `invalid_mec_action` | ❌ | 本步不计费，记违规 |
| 服务可负担、运动不可负担 | `energy_budget_exhausted` | ❌ | 记 $E_{coll}$，不记运动 |

### 1.4 记录项

- `energy_budget_violation_count`（episode 级）
- `remaining_energy_j`（step 级与 episode 级）
- 分项能耗：`e_hover_j / e_forward_j / e_turn_j / e_enc_j / e_tx_j / e_rx_j`

## 2. 单次服务截止时间

$$
D_{coll}(t,m_t)\le D_{max}
$$

| 配置 | 默认 | 状态 |
|---|---:|---|
| `d_max_ms` | 150 ms | **PROVISIONAL** |

**取值依据（待审核）**：Stage 16 各分量的解析范围是 $D_{enc}=5.45$ ms、$D_{up}\in[24,126]$ ms（500 m 内）、$D_{inf}\in[0.5,20]$ ms，
因此 $D_{coll}$ 的解析上界约 152 ms。150 ms 只是"不实质排除任何节点"的起点，用于先验证管线；真正的阈值应在 Stage 17-B 用实际时延分布标定。
**本阶段不声称该阈值有物理依据。**

### 2.1 每步候选计算

环境每一步都计算全部 10 个候选：

```python
candidate_service_latency[m]  # D_coll(distance_m, model_type)
deadline_valid_mask[m]        # candidate_service_latency[m] <= d_max_ms
coverage_valid_mask[m]        # distance <= coverage_m
candidate_energy_cost[m]      # E_coll(m)  —— 仅服务成本，不含尚未产生的动作
feasible_mec_mask[m]          # coverage ∧ deadline ∧ (E_coll <= remaining)
```

`candidate_energy_cost` 只包含决策时**已经可知**的服务成本（编码 + 上行 + 推理 + 下行的悬停/收发能耗），
**不包含**运动能耗——运动能耗依赖 MEC 推理输出的动作，决策时还不存在。这是因果性要求。

### 2.2 无可行节点

```
if feasible_mec_mask.sum() == 0:
    terminated_reason = "no_feasible_mec"
    success = False
```

不允许静默选择不可行节点，也不允许把约束"软化"进奖励。

## 3. 与 SR 约束的关系

Stage 17-A 只提供约束通道；是否把 `SR ≥ 0.5` 显式写成第三条约束，在 Stage 17-B 决定。
注意：**如果只最小化 $D_{sum}$，会自然倾向"提前停止"**（早停的累计时延更低）。因此 SR 约束必须存在，
否则策略会利用这个捷径。Stage 16 的失败模式已验证这一点。

## 4. 约束阈值的可行性

两条约束可能同时不可行：高质量专家（TS/Large）服务时延与能耗更高，若 `d_max_ms` 与 `task_energy_budget_j` 同时收紧，
则"SR 达标"与"约束满足"可能无法同时成立。Stage 17-B 的 Smoke Test 必须在开始训练前先做阈值可行性扫描
（见 `07_SMOKE_TEST_PLAN.md`），而不是直接训练。
