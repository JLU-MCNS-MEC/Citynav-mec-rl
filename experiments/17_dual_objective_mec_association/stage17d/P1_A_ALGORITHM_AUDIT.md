# Stage 17-D-P1-A · 算法修复与诊断审计

日期：2026-11-26
基准：P0 commit `ab31af849`
**本文件对应 Commit 1（算法修复 + 诊断），尚未包含 P1-B 训练结果。**
未加载 Checkpoint、未建环境、未跑 episode；只执行 `py_compile` 与
`stage17c/checks_p1a.py`（T1–T9，纯张量/算术），**9/9 通过**。

## 1. TD3 Actor 更新频率（P0 核心问题）

**P0 现象**：3,692 次 Critic 更新，Actor 仅约 18 次（绑定 `total_steps % 200`）。

**修复**：引入独立计数器，Actor 每 `policy_delay=2` 次 **Critic 更新**执行一次，
Target 网络同步跟随 Actor 更新。DQN 的 `target_sync_steps=200` 保持不变。

```python
ag["critic_updates"] += 1
if ag["critic_updates"] % policy_delay == 0:
    update_actor()            # 可微策略期望
    soft_update(targets)      # tau=0.005
```

新增日志：`critic_updates / actor_updates / target_updates / grad_q / grad_pi`。

**验证（T1）**：10 次 Critic 更新 → Actor 5 次、Target 5 次，比例精确 2:1。

## 2. DDPG 梯度诊断（只加观测，不改算法）

P0 中 Actor 梯度 ≈ `1.3e-7`。新增以下诊断字段：

```text
actor_logits_std, actor_max_prob, actor_entropy,
q_action_std, q_action_range, actor_grad_norm, actor_param_delta
```

用途：区分三种假设——(a) softmax 饱和（max_prob→1、logits_std 大）；
(b) Critic 对各动作 Q 近似相同（q_action_std ≈ 0）；(c) 计算图/优化器问题
（grad 正常但 param_delta ≈ 0）。

**本轮不调整 DDPG 的学习率、不加熵奖励**；下一轮 DDPG 训练将先输出这些指标再决定。

## 3. SAC 数值诊断与稳定性

新增日志：

```text
mean_env_reward, mean_entropy_bonus(=alpha*H), mean_bellman_target,
mean_q1, mean_q2, mean_q_min, critic_td_error,
alpha, entropy, sac_target_entropy
```

**稳定性修复（单独记录）**：Critic 梯度裁剪 `sac_critic_clip = 1.0`。
理由：P0 后期 Critic loss 达 6–15、梯度较大；γ=1 下熵项在长序列中累积，
会放大 TD 误差。裁剪只作用于 Critic，不改目标函数、α 初值或奖励权重。

Soft Bellman 目标保持策略期望形式（R1 已修复），本轮只在日志中记录
`mean_entropy_bonus` 与 `mean_env_reward` 的相对尺度。

## 4. PPO 策略不变化诊断

P0 中 Dev 在 25/50/75/100 完全一致（SR 0.617、D_sum 12,497）。

新增 Dev 侧诊断：策略熵、最大动作概率、动作改变比例（与上次 Dev 对齐逐步比较）、
各节点选择直方图；训练侧新增 `approx_kl`、`clip_frac`、`policy_entropy`、
`actor_param_delta`。

据此可判断是"参数变了但贪心不变"、"过早集中到 TS"，还是"Clip 限制过度"。

## 5. 训练任务抽样

P0：`train_tasks[(ep-1) % N]` → 前 100 episode 只覆盖前 100 个任务。

P1：固定种子（42）的随机排列，遍历全部 Train 后再重排；记录
`task_index` 与 `perm_cycle`。五种算法使用同一生成规则。
**注意：任务顺序改变后，P1 曲线与原 P0 不是同一随机轨迹，不能直接续接。**

## 6. 检查结果（`smoke/checks_p1a.json`）

| 编号 | 内容 | 结果 |
|---|---|---|
| T1 | TD3 Actor/Critic 更新比例 = 1:2 | ✅ 10/5/5 |
| T2 | TD3 Actor 梯度非零且有限 | ✅ |
| T3 | DDPG 七个诊断量可计算且有限 | ✅ |
| T4 | SAC Soft Value 与手算一致（1e-6） | ✅ |
| T5 | SAC 十个诊断量齐全 | ✅ |
| T6 | PPO KL / Clip / 熵 / Δ参数 可记录 | ✅ |
| T7 | 种子 42 的任务排列可复现，43 不同 | ✅ |
| T8 | 奖励权重 γ=1、λ=1/0.5 全局一致 | ✅ |
| T9 | 400 ms / 360 kJ / 500 m / 200 步 / FP32 / 50-10 m 未变 | ✅ |

## 7. 尚未解决的问题

| # | 问题 | 状态 |
|---|---|---|
| 1 | DDPG 梯度消失的根因 | 已有观测字段，待下一次 DDPG 训练量化 |
| 2 | SAC 熵项是否主导价值 | 已有 `mean_entropy_bonus` vs `mean_env_reward`，待 P1-B |
| 3 | PPO 是否过早塌缩到 TS | 已有 Dev 直方图与 change_ratio，待 P1-B |
| 4 | γ=1 的长期稳定性 | 本轮仍为 γ=1，仅 Critic 梯度裁剪 |
| 5 | TD3 修复后未训练 | 按规范本轮不训练 TD3 |

## 8. 进入 P1-B 的前提

T1–T9 全部通过、环境参数未变、奖励未变。P1-B 仅训练 **PPO 与 SAC**，
各最多 500 episodes，Dev 每 50 次评估。
