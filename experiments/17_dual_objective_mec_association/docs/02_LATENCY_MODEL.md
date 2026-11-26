# Stage 17 · 端边协同推理服务时延模型

## 1. 定义

$$
D_{coll}(t,m)=D_{enc}(t)+D_{up}(t,m)+D_{inf}(m)+D_{down}(t,m)
\qquad
D_{sum}=\sum_{t=1}^{T}D_{coll}(t,m_t)
$$

所有量在代码中统一以 **毫秒** 输出（`latency17.py`）。

## 2. 各分量

### 2.1 $D_{enc}$ — UAV 视觉编码（继承）

来源：`experiments/12_hetero_mec_association/scripts/protocol.py:199-205`

$$
D_{enc}=\frac{\text{encoder FLOPs}}{\text{UAV 算力}}
=\frac{8\,174\,272\,512}{1.5\times10^{12}}\ \text{s}=5.449515\ \text{ms}
$$

| 项 | 值 | 来源 |
|---|---|---|
| encoder FLOPs | 8,174,272,512 | `results/protocol/encoder_flops.json`（resnet50 layer4, fp16, 1×3×224×224） |
| UAV 算力 | 1.5 TFLOPS | `protocol.py:139` `UAV_TFLOPS = 1.5e12` |

> 该 1.5 TFLOPS 是 Stage 12 协议常量，Stage 14/16 一直沿用；Stage 17 原样继承，**未新设**。
> 其硬件依据在仓库中无记录 → 列为待核实项（见 `08_REVIEW_QUESTIONS.md`）。

### 2.2 $D_{up}$ — UAV → MEC 上行（继承）

$$
D_{up}=\frac{\text{PAYLOAD\_BITS}}{\text{rate}_{up}(d)}\times 10^{3}\ \text{ms}
$$

| 参数 | 值 | 来源 |
|---|---|---|
| 上行数据量 | 1,605,632 bit = $49\times2048\times16$ | `protocol.py:128` |
| 带宽 | 5 MHz | `protocol.py:123` |
| UAV 发射功率 | 0.1 W | `protocol.py:124` |
| 载波 / 噪声 / NF | 2 GHz / -174 dBm/Hz / 9 dB | `protocol.py:131-133` |
| 视距概率与额外损耗 | A=9.61, B=0.16, 1.0 / 20.0 dB | 同上 |
| 高度 | MEC 10 m, UAV 80 m | `protocol.py:129` |

`rate_up` 由 Stage 12 的 `uplink_rate_bps(distance)` 给出，Stage 17 直接调用，不重写。

### 2.3 $D_{inf}$ — MEC 推理（继承）

$$
D_{inf}(m)=\frac{\text{FLOPs}(m)}{10\ \text{TFLOPS}}
$$

| 模型 | FLOPs（2×MACs） | $D_{inf}$ |
|---|---:|---:|
| Small | 4.954039 G | 0.495 ms |
| Medium | 14.978528 G | 1.498 ms |
| Transformer-S | 39.671126 G | 3.967 ms |
| Transformer-M | 100.025418 G | 10.003 ms |
| Large | 199.723981 G | 19.972 ms |

来源：`env16.py:66-69`（preflight 实测），10 TFLOPS 同构。

### 2.4 $D_{down}$ — MEC → UAV 下行（**默认忽略**）

> **2026-11-26 审核决定**：MEC 推理动作返回时延及其对应的 UAV 悬停能耗**可忽略**，
> 该结论有参考文献支撑（引用待补）。默认 `include_downlink=False`，因此
> $D_{down}=0$，$D_{coll}=D_{enc}+D_{up}+D_{inf}$。
>
> 模型实现保留在 `latency17.d_down_ms()` 中，仅在 `include_downlink=True` 时启用，
> 用于敏感性分析；**不是**把下行"偷偷设为 0"，而是显式配置项。

启用时的定义如下（Stage 16 没有下行，审计 H.1）：

$$
D_{down}=\frac{8\times\text{payload\_bytes}}{\text{rate}_{down}(d)}\times10^3\ \text{ms}
$$

信道**复用 Stage 12 同一模型**，仅发射端由 UAV 换成 MEC：

$$
\mathrm{SNR}_{down}=\mathrm{SNR}_{up}\cdot\frac{P_{mec}}{P_{uav}}
\qquad
\text{rate}_{down}=B\log_2\left(1+\mathrm{SNR}_{down}\right)
$$

| 参数 | 默认值 | 状态 |
|---|---:|---|
| `downlink_payload_bytes` | 8 B | **PROVISIONAL** |
| `mec_tx_power_w` | 0.1 W | **PROVISIONAL**（与 UAV 对称假设） |

**下行负载量 8 字节的推导**：MEC 需要回传的信息只有"4 类导航动作中的哪一个"，语义最小为 2 bit。8 B 的构成假设为 **1 B 动作编码 + 3 B 消息序号 + 4 B 校验/帧头**。这是一个工程假设，不是实测协议；**必须在 Stage 17-B 之前由人工确认，或替换为真实协议格式**。不同假设对 $D_{down}$ 的影响请用参数敏感性表格评估（载荷每增加 8 B，时延约翻倍，但因速率在 Mbps 量级，绝对增量仍在微秒级）。

> 明确声明：Stage 17-A **没有**把 $D_{down}$ 设为 0，也没有把下行负载设为 0。所有取值集中在 `configs/default.json`。

## 3. 单位与转换检查

| 量 | 单位 | 代码位置 |
|---|---|---|
| FLOPs | 次浮点运算（无量纲计数） | `MODEL_GFLOPS_MEASURED` 以 **G** 为单位 |
| 算力 | TFLOPS = $10^{12}$ FLOP/s | `uav_tflops = 1.5e12`, `mec_tflops = 10.0` |
| 载荷 | bit（下行由 bytes×8 得到） | `latency17.d_down_ms` |
| 时延 | 统一 ms | 所有函数返回 ms |

$D_{inf}$ 的实现为 `GFLOPS / 10.0`，量纲上等价于 `FLOPs / (10e12 FLOP/s) × 1e3 ms`，与 Stage 16 一致。

## 4. 与 Stage 16 的差异记录（不得静默修改）

| 项目 | Stage 16 | Stage 17 | 处理方式 |
|---|---|---|---|
| 单步时延名称 | `e2e_ms` | `d_coll_ms` | 新指标独立命名，旧指标保留为 `legacy_cumulative_e2e_ms` |
| 下行 | 无 | 有 | 新增，不改写 Stage 16 代码 |
| 累计指标 | `cumulative_e2e_ms` | `cumulative_collaborative_service_latency_ms` + 旧指标 | 两者同时输出，便于对照 |

## 5. 未纳入的部分

- MEC 排队时延、MEC 上行回传的重传、UAV 端结果解析时间：**未建模**（Stage 12/16 同样未建模）。
- UAV→MEC 的编码结果缓存复用：Stage 16 每步都重新编码；Stage 17 沿用该行为并按实际编码调用计费，不做缓存优化。
