# Stage 17-C · 双目标奖励分析

## 1. 论文目标与训练奖励的对应

论文目标（未折扣）：

$$
\max J_Q=\mathbb E[S(\tau)],\qquad
\min J_D=\mathbb E\Big[\sum_{t=0}^{T-1}D_t^{coll}\Big]
$$

训练奖励（标量化）：

$$
r_t=\lambda_Q\Big(\alpha_p\,\mathrm{prog}_t+\mathbb{1}\{t=T,S=1\}\Big)
-\lambda_D\frac{D_t^{coll}}{D_{ref}}
$$

$$
\mathrm{prog}_t=\mathrm{clip}\!\left(\frac{d_{t-1}-d_t}{5\ \mathrm m},-1,1\right)
$$

于是未折扣回报

$$
\sum_t r_t=\lambda_Q\Big(\alpha_p\sum_t \mathrm{prog}_t+\sum_t\mathbb 1\{\text{success}\}\Big)
-\frac{\lambda_D}{D_{ref}}\sum_t D_t^{coll}
$$

其中第二项与 $J_D$ 精确对应（$\lambda_D/D_{ref}$ 为常数缩放）；第一项在
$\sum_t\mathrm{prog}_t\propto (d_0-d_T)$ 时近似任务级进展，成功指示项直接对应 $J_Q$。

## 2. 与 Stage 16 奖励的区别

| 项目 | Stage 16 | Stage 17-C |
|---|---|---|
| 奖励 | $r_t=-e^{legacy}_t/100$ | 进展 + 成功 − 时延成本 |
| 时延口径 | 旧 E2E（1.6 Mbit、80/10 m） | 新 $D_{coll}$（4.28 Mbit、50/10 m、含 Layer2） |
| 成功信号 | 无显式项 | 终止成功 +1 |
| 能耗 | 无 | **不进奖励**，仅作为硬约束 |
| 目标 | 单目标（最小化时延） | 双目标（SR 与 $D_{sum}$） |

## 3. 分项定义

| 字段 | 公式 | 用途 |
|---|---|---|
| `terminal_success_reward` | 终止且成功 +1，否则 0（`failure_penalty` 默认 0） | 日志 / 训练 |
| `navigation_progress_reward` | $\mathrm{clip}(\Delta d/5,-1,1)$ | 日志 / 训练 |
| `collaborative_service_latency_cost` | $D_{coll}/D_{ref}$ | 训练（负向） |
| `energy_consumption_cost` | $E_{step}/E_{budget}$ | 仅日志（约束由环境执行） |
| `constraint_violation` | 非法/无可行/能量终止记 1 | 日志 |
| `total_training_reward` | 上式 | 训练 |

**目标距离只作为环境内部反馈**：127 维状态中不含真实目标坐标或距离，
`prog` 只用于计算奖励。

## 4. 归一化与参考量

| 量 | 取值 | 依据 |
|---|---:|---|
| $\lambda_Q$ | 1.0 | 配置 |
| $\lambda_D$ | 1.0 | 配置 |
| $D_{ref}$ | **6000 ms** | 一次真实 44 步 Min-Latency episode 的 $D_{sum}=5665$ ms；取整后作为固定参考 |
| $\alpha_p$ | 0.5 | 进展项与成功项大致同量级（一次完美推进 1.0 vs 成功 1.0） |
| γ | 0.99 | Stage 16 值，本轮不改 |

$D_{ref}$ **不在 episode 内动态计算**，否则会改变目标含义；必须在 Stage 17-D 开始前
用 Train 上 Min-Latency 基线的实测 $D_{sum}$ 复核并**冻结**，所有算法共用同一个值。

## 5. 折扣因子（重要审核项）

- 论文目标是**未折扣**累计时延与成功率；
- DRL 使用 $G_t=\sum_k\gamma^{k-t}r_k$，$\gamma=0.99$ 时有效视野约 100 步，
  而 episode 长度约 40–70 步，折扣对早期步骤的时延权重下降约 0.99^50≈0.61；
- 因此训练代理目标与 $J_D$ 并非严格一致，属已知偏差；
- 可选方案：
  1. 保持 $\gamma=0.99$（Stage 16 兼容，本轮默认）；
  2. 有限时域 $\gamma=1$（与目标严格一致，但需重新验证五种算法的稳定性）；
- **本轮不修改任何算法的折扣配置**；是否切换由审核决定，切换后需在 17-D 单独做对照。

## 6. 自检

`stage17c/check_reward_math.py`（纯算术，无环境/模型）覆盖：
进展计算、成功项、失败零惩罚、进度裁剪、$\lambda_D$ 缩放。**5/5 通过**。
