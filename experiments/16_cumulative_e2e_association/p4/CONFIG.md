# Stage 16 · DRL 配置（已固定）

日期：2026-11-21

| 项目 | 值 |
|---|---|
| 奖励 | A：`r_t = -e2e_t / 100` |
| 折扣因子 | γ = 0.99 |
| 训练 episodes | 4000（每个算法） |
| Seed | 42 |
| 优化器 | Adam（DQN 1e-4；PPO/TD3/SAC 3e-4） |
| Batch / Replay | DQN/TD3/SAC replay 50k，sample 128；PPO rollout 1024 |
| 目标网络更新 | DQN 每 200 steps；TD3/SAC τ=0.005 |
| 训练池 | 11A Train 4,000（seed 20261022），Dev-60 仅诊断 |
| Val-Seen | 仅最终 2498 任务评测 |
| 评测权重 | **收敛后的 `final.pt`**（不保存/不挑选 Dev 最佳） |
| 专家 | Stage 14 五种冻结模型 |
| 部署 | 10 节点，5 模型 ×2，10 TFLOPS |
| 终止 | Stop / 碰撞 / 越界 / 200 步 |

算法顺序（串行，避免 OOM）：DQN（已完成）→ TD3 → SAC → PPO。

> γ 对照实验已按用户要求取消（200 episodes 不足以区分）。
