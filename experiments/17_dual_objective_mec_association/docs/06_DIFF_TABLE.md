# Stage 16 → Stage 17 文件差异表

**Stage 17-A 没有修改 Stage 16 的任何文件。** 全部新增代码位于
`experiments/17_dual_objective_mec_association/`。

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `audit/AUDIT_STAGE14_16.md` | Stage 14/16 审计（含文件路径与行号） |
| `configs/default.json` | 全部参数与默认值 |
| `src/config17.py` | 配置 dataclass 与加载/覆盖/导出 |
| `src/latency17.py` | 四分量协同服务时延 |
| `src/energy17.py` | Zeng 飞行功率 + 服务能耗 + 运动能耗 |
| `src/env17.py` | `EpisodeEnv17(EpisodeEnv16)`：约束、掩码、计费 |
| `src/state17.py` | 127 维状态与布局 |
| `src/action17.py` | 可行集采样/投影的统一接口 |
| `src/reward17.py` | 双目标与约束成本信号 |
| `src/logger17.py` | step / episode 记录 |
| `docs/*.md` | 问题定义、时延、能耗、约束、状态动作、本表、测试方案、待审问题 |

## 2. 复用而非复制的 Stage 16 组件

| 组件 | 来源 | 用法 |
|---|---|---|
| `EpisodeEnv16` | `experiments/16.../p1/env16.py` | 被继承；导航、碰撞、Stop、成功判定全部沿用父类 |
| `state_features` | `experiments/16.../p3/train16.py` | 直接 import，构成状态前 74 维 |
| `MODEL_GFLOPS_MEASURED` | `env16.py` | 推理时延 |
| `build_deployment_by_map / load_models / load_norm` | `env16.py` | 节点部署与模型加载 |
| `uplink_rate_bps / comm_latency_ms / encoder_flops` | Stage 12 `protocol.py` | 上行、编码 |

## 3. 行为差异（相对 Stage 16）

| 行为 | Stage 16 | Stage 17 |
|---|---|---|
| 覆盖检查 | 仅在状态里给掩码，`step()` 不校验 | `feasible_mec_mask` 强制；选了不可行节点 → `invalid_mec_action` |
| 动作执行 | 先感知→前向→执行（一步一次） | **perceive → decide → transition**：同样只做一次编码与一次专家前向，但把生成与转移拆开，以便在执行前校验运动能量 |
| 单步时延 | `enc(5.45ms)+up(1.6Mbit)+inf` | `D_UAV(7.66ms)+up(4.28Mbit)+inf`，通信几何 50/10 m |
| 累计指标 | `cumulative_e2e_ms` | 新增 `cumulative_collaborative_service_latency_ms`，旧指标保留为 `legacy_*` |
| 能量 | 无 | 服务 + 运动；episode 能耗 ≤ 电池容量 |
| 终止原因 | success / premature_stop / collision / out_of_bounds / step_cap | 上述 + `no_feasible_mec` / `invalid_mec_action` / `energy_budget_exhausted` |
| 奖励 | `-e2e/100` | 保留原值作为 `navigation_reward`，另给双目标与约束成本信号（不改变标量） |

## 4. 关键实现说明

### 4.1 动作预览（为什么需要）

"不得允许超过预算的导航动作继续执行"要求在执行前知道动作。Stage 16 的 `step()` 把
"算动作 + 执行"绑定在一起，因此 `EpisodeEnv17.preview_action()` 在一份 **belief 深拷贝**上
复现父类的感知与专家前向（`env16.EpisodeEnv16.step` 的感知段），只读取动作、不改变任何状态。
随后仍由父类执行真实一步（包含真实的 belief 更新）。

一致性由模型确定性的 argmax 保证；代码用 `preview_mismatch_count` 统计任何不一致，
Smoke Test 中该项必须为 0。

### 4.2 计费位置

- 服务计费在父类 `step()` 返回后，用**父类实际执行的动作**计算运动能耗（不是预览值）。
- `energy_budget_exhausted` 分支不调用父类 `step()`，只记服务能耗。

## 5. 未改动清单（明确声明）

- 五种专家 Checkpoint：未加载、未重训、未修改。
- `experiments/14_*`、`experiments/16_*`、`experiments/12_*`：**零改动**。
- Stage 15-Extension 的 `extension/`：零改动。
- 任何历史结果、配置、日志：未覆盖。
