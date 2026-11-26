# Stage 17 · UAV 能耗模型

所有能量统一为 **焦耳 (J)**。实现见 `src/energy17.py`。

## 1. 单步总能耗

$$
E_{total}^{step}=E_{coll}+E_{motion}
$$

## 2. 协同服务能耗

$$
E_{coll}=P_{hover}\cdot D_{coll}+P_{enc}\cdot D_{enc}+P_{tx}\cdot D_{up}+P_{rx}\cdot D_{down}
$$

| 符号 | 值 | 来源 | 说明 |
|---|---:|---|---|
| $P_{hover}$ | 168.49 W | Zeng 2019：$P_0+P_i=79.86+88.63$ | 悬停诱导 + 型阻功率 |
| $P_{enc}$ | 15 W | **LITERATURE**（引用待补） | 机载视觉编码负载 |
| $P_{tx}$ | 0.1 W | Stage 12 `protocol.TX_POWER_W` | 上行射频 |
| $P_{rx}$ | 0.0 W | 下行默认忽略，故为 0 | 接口保留，见下 |

**关于 $P_{rx}=0$**：2026-11-26 审核决定忽略 MEC 动作返回时延及其悬停能耗，
因此 $D_{down}=0$，$P_{rx}\cdot D_{down}$ 恒为 0。`p_rx_w` 接口保留，
只有在 `include_downlink=True` 做敏感性分析时才需要赋值。

**不重复计算**：本模型只包含三类功率——悬停推进、编码负载、射频收发。没有另加"板级功耗/空闲功率/标称功率"项（`p_aux_w` 默认 0，且与上述三项不重叠）。

## 3. 匀速前进

每个 Forward 动作移动 5 m，主实验速度 $v=3$ m/s（备用 5 m/s）：

$$
E_{forward}=P_{fly}(v)\cdot\frac{5}{v}
$$

飞行功率采用 Zeng, Xu & Zhang, *Energy Minimization for Wireless Communication with Rotary-Wing UAV*, **IEEE TWC 2019** 的完整旋翼模型：

$$
P_{fly}(v)=
\underbrace{P_0\left(1+\frac{3v^2}{U_{tip}^2}\right)}_{\text{型阻}}
+\underbrace{P_i\sqrt{1+\frac{v^4}{4v_0^4}}}_{\text{诱导}}
+\underbrace{\frac{1}{2}d_0\rho sAv^3}_{\text{寄生}}
$$

| 参数 | 值 | 单位 | 来源 |
|---|---:|---|---|
| $P_0$ | 79.86 | W | Zeng 2019 |
| $P_i$ | 88.63 | W | Zeng 2019 |
| $U_{tip}$ | 120 | m/s | Zeng 2019 |
| $v_0$ | 4.03 | m/s | Zeng 2019 |
| $d_0$ | 0.6 | — | Zeng 2019 |
| $\rho$ | 1.225 | kg/m³ | 标准大气（ISA 海平面） |
| $s$ | 0.05 | — | Zeng 2019（旋翼实度） |
| $A$ | 0.503 | m² | Zeng 2019（旋翼盘面积） |

> **禁止**用 $E_{forward}=P_{hover}\cdot 5/v$ 近似；代码中直接调用 `fly_power_w()`。

## 4. 原地转向

每次转 30°，角速度 30°/s：

$$
T_{turn}=\frac{30°}{30°/s}=1\ \text{s},
\qquad
E_{turn}=P_{hover}\cdot T_{turn}=168.49\ \text{J}
$$

不引入偏航动力学模型。

## 5. Stop

Stop 不产生运动能耗（$E_{motion}=0$），但**产生 Stop 之前的编码、上行、推理、下行仍然计费**——`env17.step()` 对 Stop 动作同样累计 $E_{coll}$。

## 6. 计费范围（必须记录）

| 项目 | 是否计入 | 代码字段 |
|---|---|---|
| 悬停推进（服务期间） | ✅ | `e_hover_j` |
| 匀速前进推进 | ✅ | `e_forward_j` |
| 原地转向推进 | ✅ | `e_turn_j` |
| 视觉编码 | ✅ | `e_enc_j` |
| 上行通信 | ✅ | `e_tx_j` |
| 下行接收 | 接口保留，默认 0 | `e_rx_j` |
| 辅助/板级 | ❌（默认 0） | `e_aux_j` |

## 7. 因果性与计费规则

1. 每一步只对**实际执行的节点**计费；被拒绝的动作不计入 $D_{sum}$ / $E_{sum}$（只记违规标志）。
2. 任务终止后不再计费。
3. 运动能耗在执行前校验（见 `04_CONSTRAINTS.md`）。

## 8. 量纲核对

| 量 | 单位 | 公式检查 |
|---|---|---|
| $P_{fly}$ | W = J/s | 三项均为 W |
| $E_{forward}$ | J | W × s |
| $E_{turn}$ | J | W × s |
| $E_{coll}$ | J | W × s（$D$ 由 ms 转 s） |

## 9. 已知局限

- $P_{enc}=15$ W 无实测依据（PROVISIONAL）。
- 速度假设 $v=3$ m/s 由用户指定；能量对速度高度敏感（Stage 15-Extension 实测：1/3/5 m/s 下推进能耗相差约 4 倍），因此所有绝对能量结论都必须与该速度绑定声明。
