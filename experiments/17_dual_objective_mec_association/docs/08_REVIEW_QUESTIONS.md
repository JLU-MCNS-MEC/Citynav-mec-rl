# Stage 17-A · 审核结论与剩余待确认项

## 一、已由审核解决（2026-11-26）

| # | 问题 | 审核结论 | 代码状态 |
|---|---|---|---|
| 1 | MEC 动作返回时延 / 下行负载 / MEC 发射功率 | **忽略**（有文献支撑，引用待补） | `include_downlink=False` → `D_down = 0`；接口保留用于敏感性分析 |
| 2 | $P_{rx}$ 接收增量功率 | 同上，恒为 0 | 接口保留 |
| 3 | $P_{enc}=15$ W | 有文献支撑，无需再论证 | 标注 LITERATURE（引用待插入） |
| 4 | 任务能量预算 | **等于电池容量**（不设独立任务预算） | `task_energy_budget_j = 360000` J = 100 Wh |
| 5 | SR 是否作为约束 | **不是约束，是优化目标** | 约束集合仅：单次服务时延上限、episode 能耗 ≤ 电池容量 |
| 6 | nDTW | **不需要** | 已从 episode 记录中移除 |
| 7 | OSR / 成功半径 | 保留 OSR，成功半径 20 m | `logger17.episode_record` 输出 `osr` |

## 一-B、R2 新增待确认（P1）

| # | 问题 | 现状 |
|---|---|---|
| R1 | **`d_max_ms = 150` 与新负载/几何不匹配**：距离 ≳175 m 时全部节点不可行 | 未改，等待决定（建议 250–400 ms 或压缩负载） |
| R2 | **Matching FLOPs 未核定** | `matching_flops=None`，`d_matching_verified=false`，按 0 计 |
| R3 | 碰撞/越界前实际飞行距离的能耗 | 不计（`motion_aborted=true`），需确定口径 |
| R4 | Target Embedding 缓存协议 | v1 每步计费，缓存接口抛 NotImplementedError |
| R5 | `history_mask` 传输量 | 按 16×float32=512 bit 计；可压缩为 1 字节 |

## 二、仍待确认（不影响公式正确性，但影响后续口径）

| # | 问题 | 当前实现 | 需要你确认 |
|---|---|---|---|
| A | $D_{max}$ 取值 | 150 ms（R2 后距离 >≈175 m 全部不可行） | 见 R1 |
| B | 电池容量 100 Wh | PROVISIONAL | 是否有确定平台型号/容量？ |
| C | SPL 分母 | $5\times n_{forward}$ | 是否与 Stage 16 修正后的 SPL 口径一致？ |
| D | OSR 定义 | 成功 或 `min_dist ≤ 20 m` | 是否与 Stage 16 报告一致？ |
| E | 覆盖强制 | Stage 17 强制，Stage 16 不强制 | 这会令 Stage 16/17 指标不可直接比较，是否接受并在报告中声明？ |
| F | `encoder_flops.json` 的 1.5 TFLOPS 依据 | 继承 Stage 12 常量 | 是否有硬件来源可引用？ |

## 三、明确未做 / 未采用

- 未采用 nDTW；
- 未采用 SR 硬约束；
- 未执行 Smoke Test、训练、评测（Stage 17-B 及以后）；
- 未修改 Stage 12/14/16 任何文件。
