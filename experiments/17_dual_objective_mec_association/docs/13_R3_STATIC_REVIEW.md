# Stage 17-A-R3 · 静态审核报告

日期：2026-11-26
基准 commit：`f0cb45ee2d3d486e416eccfc7cd8b3fd20937491`
范围：仅代码、配置、文档与静态公式核对。
**未加载 MEC Checkpoint、未创建导航环境、未运行 episode、未做 Smoke Test、未训练、未评测。**
唯一执行的脚本是 `stage17b/unit_checks.py`（纯算术，U1–U11），结果 11/11 通过。

---

## 1. 逐项修改对照

| # | 修改 | 文件 / 函数 | 修改前 | 修改后 |
|---|---|---|---|---|
| 1 | 单次服务截止时间 | `config17.Stage17Config.d_max_ms`、`configs/default.json` | 150 ms | **400 ms** |
| 2 | 通信精度统一 FP32 | `config17.feature_bits`、`multimodal17` | 已为 32（R2 起） | 保持 32，文档明确"计算 fp16 ≠ 传输 fp16" |
| 3 | Matching 计算量 | 新增 `compute17.matching_flops()` | `None`（按 0 计） | **4.0e6 FLOPs（解析上界）→ 0.002667 ms** |
| 4 | 时延公式统一 | `latency17.service_latency_ms` | $D_{UAV}=7.662$ | $D_{UAV}=7.664743$；`d_matching_verified=True` |
| 5 | 可行性检查顺序 | `env17.step` | 已是"先算候选表/掩码，再视觉与推理" | 显式写进 docstring 与 `docs/05` |
| 6 | 事件计数 | `env17.start/_record_attempt_only/step` | 无 | `service_attempt_count` / `completed_inference_count` / `executed_navigation_action_count` |
| 7 | 碰撞/越界能耗 | `energy17.e_motion_j` | 中止的动作计 0 | **按完整离散动作计费** |
| 8 | 双目标接口 | `env17.step`、`reward17.py` | 返回 `-legacy/100` | 返回 0.0；七个信号独立导出 |
| 9 | Smoke 脚本 | `stage17b/env_consistency.py` | E1–E9 有自比较/`pass:None` | E1–E15，逐项断言 |
| 10 | 文档 | `docs/*`、`README.md`、本文件 | — | 全部同步 |

## 2. 公式对照

### 2.1 服务时延

$$
D_{coll}=\underbrace{D_{layer4}+D_{layer2}+D_{matching}}_{D_{UAV}}+D_{up}+D_{inf}
$$

| 项 | 修改前 | 修改后 |
|---|---|---|
| $D_{layer4}$ | 5.449515 ms | 不变 |
| $D_{layer2}$ | 2.212561 ms | 不变 |
| $D_{matching}$ | 0（未验证） | **0.002667 ms（4.0e6 FLOPs，已验证）** |
| $D_{UAV}$ | 7.662076 ms | **7.664743 ms** |
| $D_{up}$ | 4,281,664 bit / 新几何 | 不变 |
| $D_{inf}$ | 继承 | 不变 |
| 累计指标 | `legacy_stage16_e2e_ms` | `cumulative_collaborative_service_latency_ms` |

### 2.2 能耗

$$
E_{coll}=P_{hover}D_{coll}+P_{enc}D_{UAV}+P_{tx}D_{up},\qquad
E_{step}=E_{coll}+E_{motion}
$$

| 项 | 修改前 | 修改后 |
|---|---|---|
| 前进 | $P(3)\cdot5/3=262.593$ J | 不变（R2 已修复） |
| 转向 | 168.49 J | 不变 |
| 碰撞/越界动作 | 运动能耗置 0 | **完整计费** |
| 预算 | 360,000 J | 不变 |

### 2.3 奖励

| 修改前 | 修改后 |
|---|---|
| `reward = -legacy_e2e/100` | `reward = 0.0`；`terminal_success_reward`、`navigation_quality_reward`、`collaborative_service_latency_cost`、`energy_consumption_cost`、`energy_budget_violation`、`deadline_violation`、`legacy_stage16_reward` 全部独立 |

## 3. 主要计算结果

| 量 | 值 |
|---|---:|
| $D_{UAV}$ | 7.664743 ms |
| Matching 占视觉计算 | 0.035%（< 1% 判据，30 倍余量） |
| 上行负载 | 4,281,664 bit（535,208 B） |
| $D_{up}$ @50 / 200 / 500 m | 63.16 / 163.66 / 354.16 ms |
| $D_{coll}$ Small @50 / 200 / 500 m | 71.32 / 171.82 / 362.32 ms |
| $D_{coll}$ Large @500 m | **381.80 ms** |
| $E_{forward}$ / $E_{turn}$ | 262.593 J / 168.49 J |
| 单步能耗（前进 + 服务 @200 m） | ≈291.7 J（Small）– 295.0 J（Large） |
| 200 步能量上界 | ≈58–65 kJ（360 kJ 的 16–18%） |

## 4. Matching FLOPs 审核结论

**计算量**：一次 `subpixel_match` 的主算子为 512×28×28 特征图上的 box filter（≈1.61 M 次加法）、
通道 L2 范数（≈0.80 M）、逐格余弦点积（784×512 MACs ≈0.80 M），加上 argmax/质心/margin 与
K≤8 的 belief 更新（合计 <0.02 M）。取保守上界 **4.0e6 FLOPs**。

**时延**：$4.0\text{e}6/1.5\text{e}12=0.002667$ ms。

**是否可忽略**：$0.002667/7.662=0.035\%$，远低于建议判据 1%，因此**在时延上可视为可忽略**。
但代码**没有把它置零**：`matching_flops = 4.0e6` 显式参与计算，`d_matching_verified=True`、
`matching_negligible=True` 写入记录。

**剩余建模误差**：belief 的 `_merge` 为 O(K²)，K≤8，未逐算子展开；误差上界 <1e4 FLOPs
（时延 7e-9 ms），不改变结论。

## 5. 400 ms 截止时间的理论可行性

| 距离 | 最慢节点（Large） | ≤400 ms |
|---:|---:|:--:|
| 50 m | 90.80 ms | ✅ |
| 200 m | 191.29 ms | ✅ |
| 500 m | **381.80 ms** | ✅（余量 18.2 ms） |

即：只要节点在 500 m 水平覆盖圆内，就同时满足覆盖与截止时间。
**但这不代表每个位置都存在满足全部约束的节点**——还要叠加"剩余电量能否支付该次服务"，
以及 10 个节点的实际几何分布。四类节点需在 17-B 分别统计：

1. 无线覆盖之外；
2. 有覆盖但超截止时间（当前配置下为空集，除非负载/几何再变）；
3. 有覆盖且满足截止时间；
4. 有覆盖、满足截止时间，但剩余能量不足以支付服务。

> 400 ms 是**当前参数下的候选值**，不是 IEEE 标准参数。若 Matching 或负载口径变化，
> 必须重新核算，**不得擅自继续调大**。

## 6. 继承性检查

| 项目 | 是否继承 Stage 14/16 |
|---|---|
| 五种专家模型 / Checkpoint / 动作预测 | 完全继承，未改动 |
| 10 节点部署、每模型两份、10 TFLOPS | 继承 |
| 500 m 水平覆盖 | 继承（改为读取 config） |
| 导航规则（5 m / 30° / 20 m / 200 步） | 继承 |
| 1.5 TFLOPS UAV 算力 | 继承 Stage 12 |
| Stage 16 E2E 指标 | 保留为 `legacy_*`，不参与目标 |

Stage 12 / 14 / 16 文件：**零改动**（`git status` 复核）。

## 7. 尚未解决的问题

| # | 问题 | 影响 |
|---|---|---|
| 1 | Target Embedding 缓存协议未实现（每步计费） | 负载偏保守，不影响正确性 |
| 2 | `history_mask` 按 16×FP32 计（512 bit），可压到 1 字节 | 负载高估 <0.02% |
| 3 | Layer2 FLOPs 为 conv-only 2×MACs，未含 BN/ReLU | 与 preflight 口径需确认 |
| 4 | 碰撞前实际飞行距离不建模（固定动作成本） | 能耗略高估 |
| 5 | 能量约束在 200 步下很可能不活跃 | Stage 17-C 必须统计活跃率 |
| 6 | 400 ms 与当前负载强绑定 | 参数变化需重新核算 |

以上均**不阻塞** Smoke Test；它们影响的是数值口径而非实现正确性。

## 8. 是否建议进入 Stage 17-B

**建议进入，但需先确认两点**：

1. 400 ms 是否被接受为本次阶段的固定值；
2. "碰撞/越界动作按完整离散动作计费"的口径是否确认。

确认后，Stage 17-B 执行 `stage17b/unit_checks.py`（U1–U11，已在 R3 通过）与
`stage17b/env_consistency.py`（E1–E15），不涉及训练与闭环评测。

## 9. 本轮不声称的内容

- 不声称环境已验证正确（E1–E15 未执行）；
- 不声称具备训练条件；
- 不声称 400 ms 有文献依据。
