# Stage 17-A-R2 · 静态审核报告

审核日期：2026-11-26
上一轮审核 commit：`5fb2ffa83aa43cde6629e544656b20ba6050bc33`
**本轮未执行 Smoke Test、未加载 MEC Checkpoint、未创建环境、未运行 episode、未训练、未评测。**

## 1. 已执行的静态检查

| 检查 | 方法 | 结果 |
|---|---|---|
| 语法 | `python -m py_compile src/*.py stage17b/*.py` | 通过 |
| 公式算术 | `stage17b/unit_checks.py`（U1–U10） | **10/10 通过** |
| IEEE 公式独立性 | U5 用本文件内独立实现的 eq.(12) 对比 `energy17.fly_power_w` | 一致（<1e-9） |
| 状态维度 | 74 + 3 + 10×5 = 127 | 与 `STATE17_DIM` 一致 |
| Stage 12/14/16 | `git status` | 零改动 |

## 2. R2 修复项与对应代码

| # | 修复 | 位置 | 验证 |
|---|---|---|---|
| 1 | 诱导功率改为 $\big(\sqrt{1+x^2}-x\big)^{1/2}$ | `src/energy17.py:fly_power_w` | U5/U6：3 m/s = 157.555 W / 262.593 J；5 m/s = 143.614 W / 143.613 J |
| 2 | 电池 100 Wh 与预算约束保留 | `config17` + `env17.start` | 预算 = min(初始−余量, 任务预算) = 360 kJ |
| 3 | UAV 高度 80→50 m，MEC 10 m | `src/comm17.py` | U9：$d_{3D}(500)=\sqrt{500^2+40^2}=501.597$ m |
| 4 | 10 MEC / 500 m 覆盖，改为读取 cfg | `env17.candidate_table` | `coverage_valid = distance <= cfg.coverage_m` |
| 5 | 多模态负载重算 | `src/multimodal17.py` | U8：**4,281,664 bit（535,208 B）** |
| 6 | Layer2 匹配链路计入 UAV 时延 | `src/compute17.py` | U2：3,318,841,344 FLOPs → 2.212561 ms |
| 7 | 上行不再用 1,605,632 bit | `latency17` + `comm17` | U8/U10 |
| 8 | 动作生成与执行拆分，单次推理 | `env17.perceive/decide/transition` | 静态审查；E1 待 17-B 验证 |
| 9 | 可行性约束、计费、异常终止 | `env17.step` | 静态审查；E3–E8 待 17-B |
| 10 | 公式/配置/文献/审核更新 | 全部文档 | 本文件 + docs/10/11 |

## 3. 关键新数值（公式计算，非仿真）

| 量 | 值 |
|---|---:|
| $D_{UAV}$ = layer4 + layer2 + matching | 7.662076 ms（matching 未核定，按 0 计并标记） |
| 上行负载 | 4,281,664 bit |
| $D_{up}$ @50 m / 200 m / 500 m | 63.16 / 163.66 / 354.16 ms |
| $D_{coll}$（Small）@50 / 100 / 200 / 500 m | 71.32 / 103.86 / 171.81 / 362.32 ms |
| $E_{forward}$（3 m/s） | 262.593 J |
| $E_{turn}$ | 168.49 J |
| 单步能耗（前进 + 服务 @200 m） | ≈291.7 J（Small）– 295.0 J（Large） |
| 200 步能量上界 | ≈58–65 kJ，仅占 360 kJ 的 16–18% |

## 4. P1 发现：150 ms 截止时间与 R2 负载不匹配

- 水平距离 ≳ 175 m 时，**所有 10 个节点**的 $D_{coll}$ 都超过 150 ms；
- 在这些位置上 `feasible_mec_mask` 全为 0，episode 立即以 `no_feasible_mec` 终止；
- 后果：agent 会被人为地限制在距最近节点 ~175 m 的范围内，SR 与 $D_{sum}$ 的权衡失去意义。

**处置建议（需审核决定，本轮未改）**：
1. 将 `d_max_ms` 提到 250–400 ms 量级（覆盖 500 m 半径）；
2. 或压缩上行负载（例如对 history/current 采用 FP16，负载减半）；
3. 或复核 50 m 高度与 LoS 参数是否应针对本场景重新标定。

## 5. P2 待确认

| # | 项目 | 现状 |
|---|---|---|
| 1 | Matching FLOPs | `None`，`d_matching_verified=false`，时延按 0 计 |
| 2 | 碰撞/越界前实际飞行距离的能耗 | 不计（`motion_aborted`），需确定口径 |
| 3 | Target Embedding 缓存协议 | v1 每步计费；缓存接口抛 NotImplementedError |
| 4 | `history_mask` 传输 | 建模为 16×float32=512 bit；可用 1 字节表示 |
| 5 | Layer2 FLOPs 口径 | conv-only 2×MACs，需与 preflight 计数口径对齐确认 |
| 6 | $D_{max}$ 最终取值 | 待 17-B 可行性扫描后确定（不得用 Val-Seen） |

## 6. 对 Stage 16 的零改动声明

- `experiments/12_*`、`experiments/14_*`、`experiments/16_*`：**未修改任何文件**；
- 五种专家 Checkpoint：未加载、未重训、未修改；
- Stage 15-Extension 的 `extension/`：未修改。

## 7. 本轮交付状态

代码修正完成、文档更新完成、静态公式检查通过、Commit 已推送。
**不声称环境已验证正确，也不声称具备训练条件**——需先完成 17-B 的 E1–E9 一致性检查，
并解决 P1（截止时间）与 P2 的口径问题。
