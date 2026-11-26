# Stage 17-C · 静态审核报告

日期：2026-11-26
基准 commit：`6d29d71a6`（Stage 17-B 15/15 通过）
**本轮未加载 Checkpoint、未创建导航环境、未运行 episode、未训练任何算法。**

## 1. 执行的静态检查

| 检查 | 方法 | 结果 |
|---|---|---|
| 语法 | `python -m py_compile stage17c/*.py` | 通过 |
| 奖励代数 | `stage17c/check_reward_math.py`（纯算术，无环境无模型） | **5/5 通过** |
| 状态维度 | `STATE17_DIM = 127` 与 `src/state17.py` 一致 | 一致 |
| Stage 16 零改动 | 仅 import 思路参考，未修改任何文件 | 确认 |

## 2. 新增文件

| 文件 | 作用 |
|---|---|
| `stage17c/nets17.py` | 127 维 QNet / ActorCritic（结构与 Stage 16 相同） |
| `stage17c/reward17c.py` | 双目标奖励与五个分项 |
| `stage17c/train17c.py` | DQN/PPO/TD3/SAC/DDPG 训练器（代码复用于 Stage 16 更新公式） |
| `stage17c/evaluate17.py` | DRL 评估 + Random/Min-Latency/Max-Quality 新口径基线 |
| `stage17c/train_config.json` | 权重、$D_{ref}$、γ、全部超参 |
| `stage17c/check_reward_math.py` | 奖励代数自检 |
| `docs/15, 16, 17` | 算法设计、奖励分析、本报告 |
| `smoke/reward_math_check.json` | 自检结果 |

## 3. 关键设计决定（供审核）

| # | 决定 | 理由 |
|---|---|---|
| 1 | 训练奖励改为 `λ_Q·(α_p·progress + success) − λ_D·D_coll/D_ref` | 与论文双目标一一对应；不使用 Stage 16 的 `-legacy_e2e/100` |
| 2 | 能耗**不进入奖励** | 能量是硬约束，由环境执行；避免"早停省能"的捷径 |
| 3 | 动作掩码五算法统一 | Replay 存的是实际执行的节点，避免投影歧义 |
| 4 | 状态扩到 127 维，Stage 16 网络不动 | Stage 16 硬编码 74 维 |
| 5 | 基线 Min-Latency 使用 Stage 17 新时延 | 与 DRL 同口径 |
| 6 | γ 保持 0.99 | 不擅自改折扣；分析见 docs/16 §5 |

## 4. 尚未解决的设计问题

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| 1 | **$D_{ref}=6000$ ms 是临时值**（来自一次 44 步 Min-Latency episode 的 5665 ms） | 影响奖励尺度 | 17-D 开始前用 Train 上 Min-Latency 基线实测均值复核并**冻结** |
| 2 | **γ=0.99 与未折扣 $J_D$ 不一致** | 训练目标与论文目标有已知偏差 | 审核决定是否增加 γ=1 对照 |
| 3 | `failure_penalty=0` | 失败只损失 +1 成功项 | 若训练出现"提前停下"可在 17-D 用配置调整，不改代码 |
| 4 | 进展奖励权重 α_p=0.5 | 经验值 | 17-D 短程对照后再定 |
| 5 | PPO 单 episode rollout | 沿用 Stage 16 | 无改动 |

以上 1、2 需要在 17-D 前由审核确认；3–5 可在短程训练中通过配置验证。

## 5. 与 Stage 17-B 环境的接口一致性

- 环境 `step()` 返回 `reward=0.0` 与完整 `info`；训练器**只使用** `info` 中的
  `d_coll_ms`、`e_total_step_j`、`terminated_reason` 与 `env` 的 `pos/target/success`；
- 状态与动作掩码来自 `build_state17` / `env.masks()`，与 17-B 验证过的接口一致；
- 训练器**不会**调用 Stage 16 的 `train16.py`，也不会覆盖其 checkpoint。

## 6. 是否具备 Stage 17-D 短程训练条件

**代码层面具备**：五个训练器可编译、奖励代数自检通过、接口与 17-B 环境一致。

**仍需审核确认三点**：
1. 是否接受 $D_{ref}=6000$ ms 作为 17-D 的固定参考（或给出复核流程）；
2. γ 是否保持 0.99；
3. 短程规模（建议每算法 100 episodes、Dev-60 评估）。

在审核通过前不会执行任何训练。
