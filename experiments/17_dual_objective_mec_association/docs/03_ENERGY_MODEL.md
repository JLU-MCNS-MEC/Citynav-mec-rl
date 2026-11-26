# Stage 17-A-R2 · UAV 能耗模型

全部能量单位为焦耳 (J)。实现：`src/energy17.py`。

## 1. 单步总能耗

$$
E_{step}=E_{coll}+E_{motion}
$$

## 2. 协同服务能耗

$$
E_{coll}=P_{hover}D_{coll}+P_{enc}D_{UAV}+P_{tx}D_{up}
$$

（$D_{down}=0$ 且 $P_{rx}$ 项按审核决定恒为 0；不另加板级/待机功率。）

| 符号 | 值 | 来源 |
|---|---:|---|
| $P_{hover}$ | 168.49 W | Zeng 2019：$P_0+P_i$ |
| $P_{enc}$ | 15 W | IEEE GRSL 2026 机载视觉计算平台（文献标称，非实测） |
| $P_{tx}$ | 0.1 W | Stage 12 `protocol.TX_POWER_W` |

## 3. 前进飞行功率（R2 修复）

Zeng, Xu & Zhang, IEEE TWC 2019, DOI 10.1109/TWC.2019.2902559：

$$
P(v)=P_0\left(1+\frac{3v^2}{U_{tip}^2}\right)
+P_i\left(\sqrt{1+\frac{v^4}{4v_0^4}}-\frac{v^2}{2v_0^2}\right)^{1/2}
+\frac12 d_0\rho sAv^3
$$

**R2 修复点**：诱导功率是"平方根再开方"的形式
$\big(\sqrt{1+x^2}-x\big)^{1/2}$，其中 $x=v^2/(2v_0^2)$。
Stage 17-A 误实现为单层平方根 $\sqrt{1+v^4/(4v_0^4)}$，导致 3 m/s 下功率偏高约 9%。

| 参数 | 值 | 单位 | 来源 |
|---|---:|---|---|
| $P_0$ | 79.86 | W | Zeng 2019 |
| $P_i$ | 88.63 | W | Zeng 2019 |
| $U_{tip}$ | 120 | m/s | Zeng 2019 |
| $v_0$ | 4.03 | m/s | Zeng 2019 |
| $d_0$ | 0.6 | — | Zeng 2019 |
| $\rho$ | 1.225 | kg/m³ | ISA 海平面 |
| $s$ | 0.05 | — | Zeng 2019 |
| $A$ | 0.503 | m² | Zeng 2019 |

### 3.1 修复后的参考值

| 速度 | $P(v)$ | 前进 5 m 能耗 |
|---|---:|---:|
| 3 m/s | **157.555 W** | **262.593 J** |
| 5 m/s | **143.614 W** | **143.613 J** |
| 0 m/s | 168.49 W（= 悬停，退化检查） | — |

> Stage 17-A 的 172.2285 W / 287.0476 J 是错误值，所有文档与代码已替换。

前进能耗：

$$
E_{forward}=P(v)\frac{5}{v}
$$

**禁止**使用 $E_{forward}=P_{hover}\cdot 5/v$。

## 4. 原地转向

$$
T_{turn}=\frac{30°}{30°/s}=1\ \text{s},
\qquad
E_{turn}=P_{hover}\cdot1=168.49\ \text{J}
$$

## 5. Stop

$E_{motion}=0$，但生成 Stop 之前的完整 $E_{coll}$ 仍计费。

## 6. 碰撞 / 越界动作的计费口径（R2 明确）

**R3 规则（最终）**：只要 MEC 输出了 Forward / TurnLeft / TurnRight，就按完整离散动作计费，
**包括**该动作触发 `collision` 或 `out_of_bounds` 终止的情况。
这是"固定离散动作成本"的仿真抽象：失败的 Forward 仍按一次完整 5 m 前进计 262.593 J，
不模拟碰撞前实际飞过的距离。

- 与"剩余能量不足、动作被拒绝"是两种不同情况：后者**不执行**动作、也不计运动能耗，
  只计已经发生的服务能耗。
- Step record 用 `motion_aborted` 标注该动作以碰撞/越界结束。

## 7. 能量预算约束

$$
\sum_t E_{step}\le E_{battery}
$$

| 配置 | 默认 |
|---|---:|
| `battery_capacity_wh` | 100 Wh（文献参考：DJI TB47D 99.9 Wh；IEEE Asilomar 2023 用 130 Wh） |
| `initial_energy_j` | 360,000 J |
| `task_energy_budget_j` | 360,000 J |
| `battery_safety_margin_j` | 0 J |

**当前 200 步上限下，该约束很可能始终不活跃**：单步能耗约
$262.6$ J（前进）+ 服务 $12–64$ J ≈ **275–327 J**，200 步上界约 65 kJ，
仅为 360 kJ 的 18%。Stage 17-C 必须统计电池约束的实际活跃率，
**不得虚构其对 MEC 关联策略的优化作用**。

## 8. 计费范围

| 项目 | 计入 | 字段 |
|---|---|---|
| 服务期间悬停 | ✅ | `e_hover_j` |
| 前进推进 | ✅ | `e_forward_j` |
| 转向推进 | ✅ | `e_turn_j` |
| UAV 视觉计算（Layer4+Layer2+Matching） | ✅ | `e_enc_j` |
| 上行通信 | ✅ | `e_tx_j` |
| 下行接收 | 0（审核决定忽略返回） | `e_rx_j` |
| 辅助/板级 | ❌（`p_aux_w=0`） | `e_aux_j` |

## 9. 单位核对

| 量 | 单位 | 检查 |
|---|---|---|
| $P$ | W = J/s | 三项均为 W |
| $E_{forward}$ | J | W × s |
| $E_{coll}$ | J | W × s（$D$ 由 ms 转 s 后再乘） |
| 预算 | J（100 Wh = 360,000 J） | 不写 mAh |
