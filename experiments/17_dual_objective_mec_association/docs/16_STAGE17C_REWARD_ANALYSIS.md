# Stage 17-C-R1 · 双目标奖励分析（更新版）

> 本文件取代 Stage 17-C 初版（该版使用 $D_{ref}=6000$ ms 与导航进展奖励，均已删除）。

## 1. 奖励定义

$$
r_t=\lambda_Q\,\mathbb 1\{t\ \text{成功终止}\}-\lambda_D\frac{D_t^{coll}}{D_{\max}},
\qquad D_{\max}=400\ \mathrm{ms}
$$

Episode 回报：

$$
G(\tau)=\sum_t r_t=\lambda_Q S(\tau)-\lambda_D\frac{D_{sum}(\tau)}{D_{\max}}
$$

与论文目标的关系：$J_Q=\mathbb E[S]$、$J_D=\mathbb E[D_{sum}]$。
因为 $D_{\max}$ 是**常数**，

$$
G=\lambda_Q J_Q-\frac{\lambda_D}{D_{\max}}J_D
$$

即奖励的正是在最大化 $J_Q$、最小化 $J_D$ 的加权和，**不引入任何新的累计时延阈值或约束**。

## 2. 删除项与原因

| 删除项 | 原因 |
|---|---|
| $D_{ref}=6000$ ms | 由单个 episode 推得，无统计依据；且引入未经验证的尺度 |
| 导航进展奖励 $\alpha_p\cdot\Delta d/5$ | 未经确认的额外 shaping，且与成功项作用重复 |
| `progress_weight` / `progress_clip` | 随进展项删除 |
| 终局奖励系数 / 失败惩罚 | 保持"成功=1、失败=0"的最简形式 |

目标坐标仍只用于环境成功判定；127 维状态中不含目标信息。

## 3. 归一化的量级分析（P1）

$D_{\max}=400$ ms 让单步成本落在 $O(1)$：典型 $D_{coll}$ 约 71–382 ms → 0.18–0.96。

但**累计项随 episode 长度线性增长**：

| 情形 | 步数 | $D_{sum}$ | 时延项 $D_{sum}/400$ | 成功项（最大） |
|---|---:|---:|---:|---:|
| 短 episode | 20 | ≈2.6 s | 6.5 | 1 |
| 典型（17-B 示例） | 44 | 5.67 s | 14.2 | 1 |
| 长 episode | 70 | ≈9.0 s | 22.5 | 1 |

在 $\lambda_Q=\lambda_D=1$ 下，成功只占总回报的约 4–7%。
这意味着训练更像"最小化累计时延、顺带成功"，而不是均衡的双目标；
更糟的是**早停**（`premature_stop`）会同时压低 $D_{sum}$、只损失 1 的成功项，
存在被利用的风险（Stage 16 已出现过类似失败模式）。

**因此 $\lambda_Q:\lambda_D$ 必须在 Stage 17-D 之前由审核确定。**
本阶段只实现权重接口，不做调整。可选做法：

1. 提高 $\lambda_Q$ 到 10–20（使成功项与典型时延项同量级）；
2. 或把时延项改为**每步平均** $D_{coll}/D_{\max}/T$（改变目标含义，需谨慎）；
3. 或用约束形式处理 SR（但审核已明确 SR 是目标，不是约束）。

## 4. γ = 1.0

- 论文目标未折扣 ⇒ $\gamma=1$；
- 五种算法统一读取同一 γ，终止状态 $1-d$ 截断；
- PPO 的 `gae_lambda=0.95` 保留（GAE 平滑，不是 γ）；
- Checkpoint 记录生效 γ；
- γ=1 下五种算法的数值稳定性未经验证 → 17-D 需先做 100-episode 冒烟。

## 5. 分项字段

| 字段 | 用途 |
|---|---|
| `terminal_success_reward` | 训练 + 日志 |
| `collaborative_service_latency_cost` | 训练（负向） |
| `energy_consumption_cost` | 仅日志（硬约束由环境执行） |
| `energy_budget_violation` / `deadline_violation` | 日志与约束统计 |
| `total_training_reward` | 训练 |

## 6. 自检

`stage17c/checks_r1.py` R1/R2/R3/R9 覆盖公式、旧项删除、γ 与选择逻辑；RB 见 docs/18。
