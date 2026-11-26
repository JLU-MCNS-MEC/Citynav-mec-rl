# Stage 17 · 优化问题定义

## 1. 决策问题

一个 CityNav 城市空中视觉导航任务，每一步 $t$ 上：

- UAV 观测当前视角图像，本地视觉编码器产出特征（固定 FLOPs）；
- UAV 选择把本次推理卸载到 10 个 MEC 节点中的一个 $m_t$；
- 节点 $m_t$ 上运行五种冻结专家之一，输出 4 类导航动作 $a_t$（Stop / Forward / TurnLeft / TurnRight）；
- UAV 执行 $a_t$（前进 5 m、转向 30° 或停止）。

关联策略 $\pi(m_t \mid s_t)$ 只决定"在哪台 MEC 上推理"，不决定导航动作本身——导航动作由被选中节点上的专家产生。这是 Stage 16 已确立、Stage 17 继承的设定。

## 2. 双目标

$$
\max_{\pi}\ \mathrm{SR}(\pi), \qquad
\min_{\pi}\ D_{sum}(\pi)=\sum_{t=1}^{T} D_{coll}(t,m_t)
$$

- $\mathrm{SR}$：任务级导航成功率（成功半径 20 m，Stop 时判定，规则继承 Stage 16）；
- $D_{coll}$：单次端边协同推理服务时延（编码 + 上行 + 推理 + 下行）。

**导航动作本身的物理时间（Forward 5 m、Turn 30°）不计入 $D_{sum}$。** 这一点与 Stage 16 的"累计 E2E"口径一致，但 Stage 16 缺下行分量（见审计 H.1）。

## 3. 硬约束（仅两条）

$$
\text{(C1)}\quad \sum_{t=1}^{T} E_{total}(t) \le E_{battery}
$$

$$
\text{(C2)}\quad D_{coll}(t,m_t) \le D_{max},\quad \forall t
$$

- **(C1)** 单个 episode 的 UAV 能耗不得超过**电池容量**（默认 100 Wh = 360,000 J）。
- **(C2)** 单次端边协同推理服务时延不得超过 $D_{max}$。

两条都是**硬约束**，通过可行节点掩码与任务终止实现，不得只用奖励惩罚近似。

> **2026-11-26 审核决定**：SR **不是约束**，它是优化目标（与 $D_{sum}$ 并列）。
> 约束集合里没有 SR 项。为保证 SR 不被"早停省时延"的捷径牺牲，Stage 17-C 的
> 训练方式必须在双目标层面处理 SR，而不是靠第三条硬约束。

## 4. CMDP 形式

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\ \mathcal{C}_{E},\mathcal{C}_{D},\ \mathbf{d})
$$

| 记号 | 含义 |
|---|---|
| $\mathcal{S}$ | Stage 17 状态（Stage 16 的 74 维 + 约束特征，共 127 维，见 `STATE_ACTION.md`） |
| $\mathcal{A}$ | $\{0,\dots,9\}$ 节点索引，受 `feasible_mec_mask` 约束 |
| $r$ | 保留 Stage 16 的 $r_t=-D_{legacy}(t)/100$；Stage 17-A 不决定最终标量化形式 |
| $\mathcal{C}_E$ | 任务级能量消耗成本 |
| $\mathcal{C}_D$ | 单次服务时延成本 |
| $\mathbf{d}$ | 约束阈值 $(E_{battery},\ D_{max})$ |

## 5. 本阶段范围

**做**：环境、时延模型、能耗模型、约束机制、状态接口、日志接口、参数配置、文档与静态审核材料。

**不做**：Smoke Test、正式训练、闭环评测、超参数搜索、奖励权重选择。

## 6. 与 Stage 16 的差异一览

| 维度 | Stage 16 | Stage 17-A |
|---|---|---|
| 目标 | 单目标：最小化累计 E2E | 双目标：SR 与 $D_{sum}$ |
| 时延 | enc + up + inf | enc + up + inf + **down** |
| 能量 | 无 | 服务能耗 + 运动能耗 + 任务预算 |
| 截止时间 | 无 | 单次服务 $D_{max}$ 约束 |
| 覆盖 | 仅提供掩码，不强制 | **强制**，越界节点不可选 |
| 状态 | 74 维 | 127 维（+53 维约束特征） |
| 动作 | 10 维节点索引 | 同一动作空间，但受可行掩码约束 |
