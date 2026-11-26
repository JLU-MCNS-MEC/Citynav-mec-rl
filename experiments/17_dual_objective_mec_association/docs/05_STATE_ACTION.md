# Stage 17 · 状态与动作接口变更

## 1. 状态：74 → 127 维

Stage 16 状态原样保留（`train16.state_features`，74 维 = 4 全局 + 10 节点 × 7 特征），其后追加 53 维约束特征：

| 区间 | 字段 | 归一化 |
|---|---|---|
| `[0:74]` | Stage-16 状态（UAV 位置、步数、累计 E2E 均值；每节点 dx, dy, 距离, 通信时延, 推理时延, 单步 E2E, 可用标志） | 同 Stage 16 |
| `[74]` | `normalized_remaining_energy` = 剩余 / 初始 | ✅ |
| `[75]` | `normalized_consumed_energy` = 累计 / 任务预算 | ✅ |
| `[76]` | `remaining_energy_j / 1e5` | 360 kJ 预算映射为 3.6 |
| `[77:87]` | `candidate_service_latency[m] / D_max`（10） | ✅ |
| `[87:97]` | `candidate_energy_cost[m] / task_budget`（10） | ✅ |
| `[97:107]` | `deadline_valid_mask`（10） | 0/1 |
| `[107:117]` | `coverage_valid_mask`（10） | 0/1 |
| `[117:127]` | `feasible_mec_mask`（10） | 0/1 |

代码：`src/state17.py`（`build_state17`、`STATE17_LAYOUT`、`STATE17_DIM=127`）。

## 2. 因果性检查（强制）

| 禁止项 | 本实现 |
|---|---|
| 真实目标坐标 | ❌ 未进入状态（Stage 16 本就不含；`target` 只在环境内部判定） |
| 真实目标距离 | ❌ 未进入状态 |
| 未来轨迹 / 未来动作 | ❌ |
| 导航成功标签 | ❌ 只在 episode 结束后生成 |
| MEC 推理结果或运动能耗 | ❌ `candidate_energy_cost` 只含服务成本；运动能耗在执行前才计算，且不进入状态 |

## 3. 动作空间

- 离散 10 维（节点索引 0–9），与 Stage 16 完全一致。
- **新增约束**：动作必须落在 `feasible_mec_mask` 内。

### 3.1 统一可行投影接口（`src/action17.py`）

审计结论（`AUDIT_STAGE14_16.md` G 节）：Stage 16 **不存在**"连续动作 → 节点索引"的离散映射；
TD3/SAC 从 softmax 采样，DDPG/DQN 取 argmax，基线直接选节点。因此 Stage 17 只需统一"如何在可行集上采样"：

| 算法 | Stage 16 做法 | Stage 17 做法 |
|---|---|---|
| DQN | Q argmax | `greedy_node(mask_logits(Q, feasible_mask))` |
| PPO / SAC | softmax 采样 | `sample_node(logits, feasible_mask)` |
| TD3 | 采样目标策略分布 | `sample_node(...)`；目标网络同样掩码 |
| DDPG | argmax 策略 | `greedy_node(...)` |
| Random | 均匀选节点 | 在可行集上均匀 |
| Min-Latency / Max-Quality | 直接选节点 | 在可行集上按原准则选，无可行则 `None` |

无可行节点时所有接口返回 `None`，环境记 `no_feasible_mec` 并终止——不静默选非法节点。

## 4. 状态中不含的字段（有意为之）

- 当前专家置信度、专家动作：Stage 16 状态也没有，保持可比性。
- 已选择的专家历史：会引入与动作空间重复的信息，本阶段不加入。


## 5. 执行顺序（R3 最终）

```
候选表（覆盖 / 截止时间 / 服务能耗可行性）   ← 决策前可算，不含动作信息
  ↓ feasible_mec_mask
若为空 → no_feasible_mec（service_attempt 记录，无推理）
  ↓
关联策略选节点
  ↓
UAV 视觉处理（layer4 + layer2 + matching，一次）
  ↓
MEC 冻结专家前向（一次）
  ↓
运动能量检查（remaining − E_coll − E_motion ≥ 0）？
  ├─ 否 → 拒绝动作：只计服务能量/时延，终止 energy_budget_exhausted
  └─ 是 → 执行动作（含碰撞/越界判定）、更新历史与状态、计费
```

候选时延/能耗只用**决策前已知**的量（距离、模型 FLOPs、上传负载），
不使用尚未产生的导航动作。

## 6. 事件计数

| 计数 | 含义 | 递增时机 |
|---|---|---|
| `service_attempt_count` | 尝试发起一次协同服务（含非法/不可行） | 每次 `step()` 调用 |
| `completed_inference_count` | 完成本地计算 + 上行 + MEC 推理并得到动作 | 专家前向返回后 |
| `executed_navigation_action_count` | 真正执行了 Forward/Turn/Stop | transition 返回 executed=True |

关系：`attempts ≥ inferences ≥ executed`。碰撞/越界的中止 Forward 计入 inferences，
但**不计入** executed；该动作的运动能耗**仍然计费**。

## 7. 奖励接口（R3）

`env.step()` 返回标量 **0.0**，所有信号在 `info` 中：

| 信号 | 含义 |
|---|---|
| `terminal_success_reward` | 终止且成功为 1 |
| `navigation_quality_reward` | 0.0（Stage 17-C 定义） |
| `collaborative_service_latency_cost` | $D_{coll}/D_{max}$ |
| `energy_consumption_cost` | $E_{step}/E_{budget}$ |
| `energy_budget_violation` | 动作被预算拒绝 |
| `deadline_violation` | 请求了超截止节点 |
| `legacy_stage16_reward` | $-D_{legacy}/100$，仅作对照 |

Stage 16 的奖励不再作为 Stage 17 的默认目标。
