# Stage 17-C-R1 · 奖励 / 算法修复与静态审核

日期：2026-11-26
基准 commit：`8161c8405c53e0c21921a32a55a0339b3757b695`
**本轮未加载 MEC Checkpoint、未创建环境、未运行 episode、未训练。**
执行的只有：`py_compile`、`stage17c/checks_r1.py`（纯算术与虚拟张量，R1–R9）、
`stage17c/check_reward_math.py`（R1 前的旧自检已删除，由 checks_r1 取代）。

---

## 1. 最终训练奖励

$$
r_t=\lambda_Q\,\mathbb 1\{t\ \text{成功终止}\}-\lambda_D\frac{D_t^{coll}}{D_{\max}},
\qquad D_{\max}=400\ \mathrm{ms}
$$

Episode 回报：

$$
G(\tau)=\lambda_Q S(\tau)-\lambda_D\frac{D_{sum}(\tau)}{D_{\max}}
$$

- 只有两项：成功指示 + 归一化单步服务时延；
- **没有**导航进展奖励、**没有** $D_{ref}$、**没有**失败惩罚、**没有**终局奖励系数；
- 能耗**不进入**奖励（硬约束由环境执行），仅在日志中输出。

## 2. 删除 $D_{ref}$ 的原因

$D_{ref}=6000$ ms 是上一轮由单个 44 步 episode 的 5665 ms 推得的临时值，
没有训练集统计依据，且会在算法间引入未经验证的尺度。R1 直接删除该参数：
累计项的分母改为**已验证的固定常数** $D_{\max}=400$ ms（Stage 17-B 的 15/15 项已确认）。
由于分母是常数，最小化 $\sum D_t/D_{\max}$ 与最小化 $\sum D_t$ **严格等价**，
不改变优化目标，也不需要额外的累计时延阈值。

## 3. 用单次服务截止时间归一化的数学解释

$D_{\max}$ 只是**无量纲化因子**，不是新的约束：
$D_{sum}/D_{\max}$ 是一个纯数，其最小值与 $D_{sum}$ 同一排序；
它同时让每步成本落在 $O(1)$（典型 130–380 ms → 0.33–0.95），
避免累计项在数值上吞掉成功项之外的尺度问题。

**重要提醒（P1）**：归一化只解决单步尺度，不解决累计项与成功项的**相对量级**。
见 §10 第 1 条。

## 4. γ = 1.0

论文目标是**未折扣**的成功率与累计服务时延，因此：

- `train_config.json`：`gamma = 1.0`
- `train17c.DEFAULTS["gamma"] = 1.0`
- 五种算法的 Bellman/GAE 全部读取同一个 `cfg["gamma"]`
- Checkpoint 保存生效的 `gamma`
- 终止状态一律 $1-d$ 截断（$d=1$ ⇒ 无 bootstrap）
- `gae_lambda = 0.95` 保留（GAE 平滑，与 γ 不同）

## 5. 五种算法关键更新公式（R1）

| 算法 | Critic / 回报 | Actor |
|---|---|---|
| DQN | $y=r+\gamma(1-d)\,\max_{m\in\mathcal F'}Q'(s',m)$，空掩码取 0 | — |
| DDPG(discrete) | $y=r+\gamma(1-d)\,Q'(s',\arg\max_{m\in\mathcal F'}\pi'(m\|s'))$ | $-\mathbb E\sum_m\pi(m\|s)Q(s,m)$（可微） |
| TD3(discrete) | 同上，双 Q 取 min，目标动作从掩码策略采样（平滑） | $-\mathbb E\sum_m\pi(m\|s)\min(Q_1,Q_2)$（**R1 修复**） |
| SAC(discrete) | $V(s')=\sum_m\pi(m\|s')[\min Q'-\alpha\log\pi]$，$y=r+\gamma(1-d)V$（**R1 修复**） | $\mathbb E\sum_m\pi\,[\alpha\log\pi-Q]$ |
| PPO | GAE，$\gamma=1$ | clip 目标，掩码 Categorical |

$\mathcal F'$ 为下一状态的 `feasible_mec_mask`。

## 6. TD3 / SAC 修复前后

**TD3 Actor**

```python
# 修复前（不可微）
p = masked_softmax(lg, bm); a_pi = torch.multinomial(p.detach(), 1); loss = -minQ(bs)[a_pi]
# 修复后（策略期望，可微）
p = masked_softmax(lg, bm); loss_pi = -(p * torch.min(q1(bs), q2(bs))).sum(1).mean()
```

**SAC Soft Value**

```python
# 修复前（max，错）
qn = torch.min(q1t, q2t) - alpha * logp2;  qn = mask_logits(qn, bm2).max(1).values
# 修复后（期望，对）
qmin = torch.min(q1t(bs2), q2t(bs2))
v = (p2 * (qmin - alpha * logp2)).sum(1)
tgt = r + gamma * (1 - d) * v
```

两者都保留双 Critic、Target 网络与软更新（τ=0.005），TD3 的 Actor 更新仍按
`target_sync_steps = 200`（**单位是环境交互步**，沿用 Stage 16；已在文档中写明）。

## 7. 离散动作适配声明

DDPG / TD3 的标准形式是连续动作。本实现用 10 路 softmax 策略 + 期望/采样，
属于 **Discrete-DDPG / Discrete-TD3 variant**，不声称复现标准连续版本。
SAC 同样为离散版本（标准离散 SAC 形式）。DQN 与 PPO 本身即离散。

## 8. 统一动作掩码与终止处理

- 五算法共用 `feasible_mec_mask`（覆盖 ∧ 截止 ∧ 服务能量）；
- `masked_softmax` 在全零掩码时返回全零（不产生 NaN、不产生均匀分布）；
- `masked_max` 在空掩码返回 0，避免对 $-10^{18}$ 取 max；
- Replay 存**实际执行**的节点索引；
- 终止状态 $1-d=0$，不参与 bootstrap；
- 无可行节点时策略返回 `None`，环境按既定规则终止。

## 9. 配置与 Checkpoint

- 优先级：**CLI > `train_config.json` > 代码默认值**；启动打印生效配置并写入
  `effective_config.json`；
- 奖励的 $D_{\max}$ 从 `configs/default.json` 读取，不在奖励模块重复硬编码；
- Checkpoint：`best.pt` / `last.pt` / `history.jsonl` / `dev_history.jsonl`；
- 选择准则（**评分，不是新约束**）：
  $J_{Dev}=\lambda_Q SR_{Dev}-\lambda_D\bar D_{sum,Dev}/400$；
- 保存：网络、Target、优化器状态、γ、权重、环境摘要、episode/global step、Dev 指标；
- 不使用 Val-Seen/Test 选择。

## 10. 测试结果（`smoke/checks_r1.json`，9/9 通过）

| 编号 | 内容 | 结果 |
|---|---|---|
| R1 | 奖励公式与 episode 求和 | ✅ |
| R2 | 旧奖励标识符（AST 扫描）已消失 | ✅ |
| R3 | γ 默认 1.0、JSON 与 CLI 读取 | ✅ |
| R4 | TD3 Actor 梯度：范数、有限性、参数变化 | ✅ |
| R5 | SAC soft value 与手算一致（1e-6） | ✅ |
| R6 | 终止 bootstrap = 0；空掩码 max = 0 | ✅ |
| R7 | 掩码排除非法动作 | ✅ |
| R8 | 配置优先级 | ✅ |
| R9 | Best/Last 选择逻辑 | ✅ |

> R5 的容差为 1e-6：本机 torch 的 double `log` 与 libm 相差约 4e-8，属数值库差异，不影响公式正确性。

## 11. 尚未解决 / 待审核

| # | 问题 | 数据 |
|---|---|---|
| **1** | **λ_Q=λ_D=1 时奖励被时延项主导** | 44 步示例：$D_{sum}/400=14.16$，成功项 ≤1 ⇒ 成功只占约 6–7%。若不加权，策略会倾向"少走步、早停"以压低累计时延 |
| 2 | Best 选择准则同样被时延主导 | 同上，$\lambda_Q$ 需与 $\lambda_D$ 一起审 |
| 3 | `target_entropy = -0.5·ln10` 沿用 Stage 16 | 离散 SAC 常见取 $-1$；两者均可，但需在 17-D 前定 |
| 4 | TD3 actor delay 单位 | 已明确为环境交互步（每 200 步一次），与 Stage 16 一致 |
| 5 | γ=1 的稳定性 | 五种算法均未在 γ=1 下跑过，17-D 需先做 100-episode 冒烟 |

第 1 条是**进入 17-D 前必须决定的**：是提高 $\lambda_Q$（例如 10–20）使成功项与
时延项同量级，还是保持 1:1 并接受"以时延为主、成功为次"的目标。本阶段不擅自调权重。
