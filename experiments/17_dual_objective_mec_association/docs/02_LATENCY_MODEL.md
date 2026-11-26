# Stage 17-A-R2 · 端边协同推理服务时延模型

## 1. 定义

$$
D_{coll}(t,m)=D_{UAV}(t)+D_{up}(t,m)+D_{inf}(m)
\qquad
D_{sum}=\sum_{t}D_{coll}(t,m_t)
$$

MEC→UAV 的动作返回时延按审核决定忽略（`include_downlink=False`），对应的悬停能耗也不计。
导航动作本身的物理时间（5 m 前进、30° 转向）不计入 $D_{sum}$。

## 2. $D_{UAV}$：UAV 本地计算（R2 新增 Layer2 与 Matching）

$$
D_{UAV}=D_{layer4}+D_{layer2}+D_{matching}
$$

| 分量 | 计算 | 值 | 说明 |
|---|---|---:|---|
| $D_{layer4}$ | 8,174,272,512 / 1.5e12 | **5.449515 ms** | FrozenEncoder（resnet50 全网络 fp16，224×224），产出 49×2048 token |
| $D_{layer2}$ | 3,318,841,344 / 1.5e12 | **2.212561 ms** | `matching.Encoder` stem = conv1+layer1+layer2（fp32，224×224） |
| $D_{matching}$ | 未核定 | **0 ms（未验证）** | subpixel 相关 + belief 更新；`matching_flops=None`，记录里标 `d_matching_verified=false` |
| **合计** | | **7.662076 ms** | |

Layer2 的 FLOPs 由算子表逐层计算（`src/compute17.py`，2×MACs、仅卷积，与 Stage-16 preflight 同口径）：

| 算子块 | 输出 | FLOPs |
|---|---|---:|
| conv1 (3→64, 7×7, s2) | 112×112 | 236,027,904 |
| layer1 (3 bottlenecks, 64→256) | 56×56 | 1,181,745,152 |
| layer2 (4 bottlenecks, 128→512) | 28×28 | 1,901,068,288 |
| **合计** | | **3,318,841,344** |

> **不能**用完整模型的 8.174 GFLOPs 代替 Layer2：那是 conv1→layer4 的全网络前向。
> 该 3.32 G 是 conv-only 2×MACs 的理论值，不含 BN/ReLU，需人工复核。

## 3. $D_{up}$：上行（R2 改为完整多模态负载 + 新几何）

$$
D_{up}=\frac{L_{up}}{R_{up}(d_h)}\times10^{3}\ \text{ms}
$$

- $L_{up}$ = **4,281,664 bit**（535,208 B），由 `src/multimodal17.py` 逐模态累加，
  明细见 `docs/10_MULTIMODAL_AUDIT.md`。Stage 16 的 1,605,632 bit 只覆盖 Current Visual。
- $R_{up}$ 使用 Stage 17 几何：UAV 50 m、MEC 天线 10 m、$d_{3D}=\sqrt{d_h^2+40^2}$，
  仰角 $\theta=\arctan(40/d_h)$，LoS 概率/路径损耗/带宽/载波/噪声结构与 Stage 12 完全相同
  （`src/comm17.py`，**未修改** `protocol.py`）。

## 4. $D_{inf}$：MEC 推理（继承）

| 模型 | GFLOPs | $D_{inf}$ |
|---|---:|---:|
| Small | 4.954039 | 0.495404 ms |
| Medium | 14.978528 | 1.497853 ms |
| Transformer-S | 39.671126 | 3.967113 ms |
| Transformer-M | 100.025418 | 10.002542 ms |
| Large | 199.723981 | 19.972398 ms |

## 5. 静态服务时延表（公式计算，非仿真）

$D_{UAV}=7.662$ ms 恒定；$D_{down}=0$。

| 水平距离 | $D_{up}$ | Small | Medium | TS | TM | Large | ≤150 ms |
|---:|---:|---:|---:|---:|---:|---:|---|
| 50 m | 63.16 | 71.32 | 72.32 | 74.79 | 80.83 | 90.80 | ✅ 全部 |
| 100 m | 95.70 | 103.86 | 104.86 | 107.33 | 113.37 | 123.34 | ✅ 全部 |
| 200 m | 163.66 | 171.81 | 172.82 | 175.29 | 181.32 | 191.29 | ❌ 全部 |
| 300 m | 222.83 | 230.99 | 231.99 | 234.46 | 240.50 | 250.47 | ❌ 全部 |
| 400 m | 285.17 | 293.33 | 294.33 | 296.80 | 302.83 | 312.80 | ❌ 全部 |
| 500 m | 354.16 | 362.32 | 363.32 | 365.79 | 371.83 | 381.80 | ❌ 全部 |

**关键结论（P1，需人工决策）**：新负载与 50/10 m 几何使 $D_{up}$ 从 Stage-16 的
约 24–126 ms 升至 **63–354 ms**。在当前 `d_max_ms = 150 ms` 下，
**只有水平距离 ≲ 175 m 的节点可行**；超过该距离时所有节点都不可行，
episode 会以 `no_feasible_mec` 终止。这不是实现错误，而是 150 ms 与 R2 负载/几何不匹配。

> 本阶段**不修改**该阈值（规范要求保持为可配置临时参数）。可选处置：
> (a) 提高 $D_{max}$ 至约 200–400 ms 量级；(b) 压缩上行负载（如 FP16 或在 UAV 端缓存
> Current Visual 之外的模态）；(c) 重新审视 50 m 高度假设。三者都需要审核决定。

## 6. 旧指标保留

`legacy_stage16_e2e_ms(d, m)` 仍按 Stage-16 定义（1,605,632 bit、80/10 m、enc+up+inf）
计算并写入 `legacy_cumulative_e2e_ms`，用于与新指标对照，不参与 Stage 17 目标。
