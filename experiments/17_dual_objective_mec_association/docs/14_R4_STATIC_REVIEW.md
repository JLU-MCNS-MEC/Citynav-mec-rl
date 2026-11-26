# Stage 17-A-R4 · 静态审核报告

日期：2026-11-26
基准 commit：`0558fccb09ceac7dc6a08891ed6e96dfebd2f65d`
范围：仅修复 R3 审核发现的环境计费、事件记录与 Smoke Test 断言问题。

**本轮未加载 MEC Checkpoint、未创建导航环境、未运行 episode、未执行 Smoke Test、
未训练、未评测。** 仅执行 `py_compile` 语法检查。

**公式与参数全部保持不变**：`d_max_ms=400`、100 Wh、飞行/悬停/转向能耗、
Layer4/Layer2/Matching FLOPs、4,281,664 bit 上行负载、50/10 m 几何、500 m 覆盖、
五种专家与导航判定均未改动。

---

## 1. 修复项

### 1.1 Step Record 补齐 `e_coll_j` / `e_motion_j`

| 位置 | 修复前 | 修复后 |
|---|---|---|
| `env17._record()`（正常步骤） | 只有 `e_hover/e_enc/e_tx/e_rx/e_forward/e_turn/e_total` | 新增 `e_coll_j`、`e_motion_j` |
| `env17._record_attempt_only()`（非法/无可行请求） | 无这两个字段 | 两者显式置 0 |
| 请求失败但服务已发生（能量拒绝分支） | 无 `e_coll_j` | 记录实际服务能耗，`e_motion_j=0` |

不变量：`e_total_step_j == e_coll_j + e_motion_j`（E5 中逐条断言，容差 1e-6）。

### 1.2 能量拒绝分支的执行顺序

修复前：`_record()` 先写入，`_terminate()` 后设置原因 → Step Record 的
`terminated_reason` 为空，与 Episode 记录不一致。

修复后顺序：

1. 完成 MEC 推理（`completed_inference_count += 1`）；
2. `model_selection[node] += 1`（**此前遗漏**，导致 `sum(model_selection) != completed_inference_count`）；
3. 计算已发生的服务能耗与时延；
4. 判定运动不可支付；
5. 拒绝动作，`e_motion_j = 0`；
6. 累计时延、能耗、剩余电量；
7. `_terminate("energy_budget_exhausted")`；
8. 最后写 Step Record。

Step Record 与 Episode Record 的终止原因现在一致。

### 1.3 事件计数语义

| 计数 | 定义 | 维护位置 |
|---|---|---|
| `service_attempt_count` | 每次 `step()` 调用（含非法/不可行） | step 入口 |
| `completed_inference_count` | 专家前向返回动作 | `decide()` 之后 |
| `executed_navigation_action_count` | transition 返回 `executed=True` | transition 之后 |
| `model_selection[node]` | 完成推理的节点使用次数 | 正常分支与能量拒绝分支 |

不变量：`attempts ≥ inferences ≥ executed`、`sum(model_selection) == inferences`、
`attempts == len(step_records)`。`step_idx` 仍是 CityNav 的导航决策步语义，
**未为匹配日志条数而增加**。

### 1.4 E5 测试重写

修复前：用 0 m 处 Small 的服务能耗估算预算，与实际候选节点不匹配；
且把 `no_feasible_mec` 也算作通过。

修复后：
1. 从真实 `candidate_table()` 取第一个可行节点；
2. 预算 = **该节点实际 `service_energy_j`** + 0.5×`E_turn`；
3. 执行该节点；若专家输出 Stop（零运动）则换下一个任务（不构成测试情形）；
4. 断言：`energy_budget_exhausted`、推理计数=1、执行动作计数=0、
   `motion_executed=False`、`e_coll_j>0`、`e_motion_j==0`、能量平衡、
   `cumulative ≤ budget`、Step 与 Episode 终止原因一致。

### 1.5 E15 测试改为确定性构造

修复前：依赖自然导航碰巧碰撞/越界，未发生则返回 `None`。

修复后：
1. 从 `info["blocked"]` 中找"本格阻塞、西侧自由"的格子，把 UAV 放在西侧自由格中心、朝向 +x，
   直接调用 **真实** `transition(1, percept)` → 断言 `collision` 且未执行；
2. 把 UAV 放在 `right-1.0 m` 处、朝向 +x，同样调用 `transition(1, percept)`
   → 断言 `out_of_bounds` 且未执行；
3. 用与 `step()` 相同的 `step_energy_j(1, …)` 断言 `e_forward_j == 262.593 J`
   且 `e_motion_j > 0`；
4. 三项同时为真才 `pass=True`，**不再返回 None**。

导航判定逻辑来自 Stage 16 的 `segment_status`，未修改。

### 1.6 E8 与退出码

- E8 不再要求 `len(step_records) == step_idx`（非法请求/无可行/能量拒绝会使两者不同）；
  改为校验终止后再次 `step()` 抛 `RuntimeError`，且累计能耗、累计时延、
  尝试计数与记录条数**全部不变**。
- 脚本结尾：任一检查 `pass is not True` → 打印失败清单并 `sys.exit(1)`；
  全部通过才 `sys.exit(0)`。不存在"未执行却标记通过"的分支。

### 1.7 日志字段完整性

每个完成服务的 Step Record 现在包含：
`selected_mec / selected_model / navigation_action / d_coll_ms /
e_coll_j / e_motion_j / e_total_step_j / battery_remaining_j /
terminated_reason / service_attempt_count / completed_inference_count /
executed_navigation_action_count`。

Episode Record 增加三个计数（`logger17`），并保持
`cumulative_collaborative_service_latency_ms`、`cumulative_energy_j`、
`remaining_energy_j`、三类违规计数、`model_selection_frequency`、`terminated_reason`。

## 2. 修改文件

| 文件 | 修改 |
|---|---|
| `src/env17.py` | 记录字段、能量拒绝分支顺序、`model_selection` 计数 |
| `src/logger17.py` | Episode 记录增加三个事件计数（R3 已加，本轮随记录字段核对确认） |
| `stage17b/env_consistency.py` | E5 / E8 / E13 / E15 重写；失败退出码 |
| `docs/14_R4_STATIC_REVIEW.md` | 本文件 |

## 3. 未解决问题（不阻塞 Smoke Test）

| # | 问题 | 说明 |
|---|---|---|
| 1 | Target Embedding 每步计费 | 缓存协议未实现，偏保守 |
| 2 | `history_mask` 按 512 bit 计 | 实际可压缩 |
| 3 | Layer2 FLOPs 未含 BN/ReLU | 需与 preflight 口径确认 |
| 4 | 碰撞前飞行距离不建模 | 固定动作成本，能耗略高估 |
| 5 | 约束活跃率 | 属后续实验分析，本轮不讨论 |

## 4. 结论

R4 的六项修复全部完成；公式、参数、模型与导航判定零改动。
代码可进入人工审核。**未执行 Smoke Test**，需审核通过后再批准 Stage 17-B。
