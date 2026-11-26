# Stage 17-C · 算法设计与 Stage 16 继承表

日期：2026-11-26
本轮**只写代码与文档，不训练、不加载 Checkpoint、不运行 episode**。

## 1. Stage 16 审计（算法与超参）

| 项目 | Stage 16 实现 | 证据 |
|---|---|---|
| 状态维度 | 74 = 4 + 10×7 | `p3/train16.py:45,50-65` |
| 动作 | 10 个 MEC 索引 | `N_ACT = 10` |
| 网络 | QNet / ActorCritic，hidden 256，2 隐层 ReLU | `train16.py:68-89` |
| 奖励 | `r_t = -e2e_t/100`（A）或 Stage-12 scheme D | `env16.py:296-299`、`train16.py:118-130` |
| γ | 0.99 | `train16.py:493` |
| DQN | lr 1e-4，ε 1.0→0.05 / 60k 步，target 每 200 步 | `train16.py:158,279,297-298` |
| PPO | lr 3e-4，4 epochs，clip 0.2，GAE 0.95，单 episode rollout | `train16.py:163,427-460` |
| TD3 | twin Q + target，actor delay 200 步，τ 0.005，采样式目标策略 | `train16.py:165-176,314-352` |
| SAC | twin soft-Q，α 自动调节，target entropy −0.5·ln10，τ 0.005 | `train16.py:187-199,384-420` |
| DDPG | Q + 确定性 argmax 目标策略，τ 0.005 | `train16.py:177-186,353-383` |
| Replay | 50,000，batch 128，warm-up 2,000 | `train16.py:92-116,286,322` |
| 训练规模 | 4000 episodes，seed 42 | `P5_FINAL_REPORT.md` |
| 基线 | Random / Min-Latency / Max-Quality | `p2/baselines.py` |

## 2. Stage 16 → Stage 17-C 继承表

| 组件 | 处理 | 说明 |
|---|---|---|
| DQN/PPO/TD3/SAC/DDPG 更新公式 | **原样复用** | 逐行对应 `train16.py`，无算法改动 |
| 优化器 / 学习率 / batch / replay / warm-up | **原样复用** | 见上表 |
| 探索机制 | **原样复用** | ε 线性衰减（DQN/TD3/DDPG）；softmax 采样（PPO/SAC） |
| TD3 actor delay（200 步）与 τ | **原样复用** | `update_offpolicy` 内 |
| 网络结构 | 复用层宽/激活，**输入 74 → 127** | Stage 16 硬编码 STATE_DIM=74，故新建 `nets17.py` |
| 状态 | **新增** `build_state17` | 74 继承 + 53 约束特征 |
| 动作屏蔽 | **新增** 统一 `feasible_mec_mask` | 见 §4 |
| 奖励 | **替换** 为 `reward17c` | 不再使用 `-legacy_e2e/100` |
| 环境 | `EpisodeEnv17` | Stage 17-B 已验证 |
| Stage 16 代码/Checkpoint | **零改动** | 未 import 修改，也未覆盖 |

## 3. 状态与网络

- 状态：127 维（`src/state17.py`），前 74 维与 Stage 16 完全一致；
- 网络：`stage17c/nets17.py` 的 `QNet17` / `ActorCritic17`，`in_features=127`，其余结构同 Stage 16；
- 五种算法共享同一状态与屏蔽语义。

## 4. 合法 MEC 动作处理（五算法统一）

| 算法 | 训练时 | 评估时 |
|---|---|---|
| DQN | Q 值在不可行节点置 `-1e18` 后取 argmax / ε-greedy 仅在可行集内 | 同左，ε=0 |
| PPO | `masked_softmax` 采样；log-prob 用同一个掩码分布计算 | argmax |
| TD3 | 目标策略同样 `masked_softmax` 采样（与 Stage 16 的离散 TD3 语义一致） | 掩码 argmax |
| SAC | 目标 Q 用 `masked_softmax`；α 项同样掩码 | argmax |
| DDPG | 目标策略掩码 argmax | 掩码 argmax |

- Replay 中存储的动作就是**实际执行**的节点索引（掩码后的 argmax/采样），不存在"连续动作→投影"的歧义；
- 无可行节点时策略返回 `None`，环境记 `no_feasible_mec` 并终止（Stage 17-B E6 已验证）；
- 掩码只依赖当前状态（覆盖/截止/电量），不引入未来信息。

## 5. 奖励接口

见 `docs/16`。实现：`stage17c/reward17c.py`。
`DualSignals` 输出的五个分项与 `train17c.py` 日志一一对应。

## 6. Train / Dev 与模型选择

- 训练：CityNav Train（Stage 16 同源 `load_tasks_and_split()`）；
- 模型选择：Dev（60 任务）每 100 episodes 评估，保存 `checkpoint.pt`；
- **Val-Seen / Test 不参与任何选择**；
- 最终 `final.pt` 与最后一个 `checkpoint.pt` 都保留。

## 7. 基线（Stage 17 新时延口径）

| 基线 | 规则 |
|---|---|
| Random | 在 `feasible_mec_mask` 内均匀采样 |
| Min-Latency | 可行节点中 `candidate_service_latency_ms` 最小（**Stage 17 新公式**） |
| Max-Quality | 可行 TS 节点中时延最小（Stage 14 质量最优模型） |

## 8. 目录

```
stage17c/
  nets17.py            127 维网络与掩码工具
  reward17c.py         双目标奖励
  train17c.py          五种算法训练器
  evaluate17.py        DRL + 三基线评估
  train_config.json    权重、γ、超参
  check_reward_math.py 纯数学自检（已运行 5/5）
```

## 9. Stage 17-D 短程 Smoke 验收方案（仅方案，未执行）

1. 每种算法 100 episodes，seed 42，`d_ref=6000 ms`；
2. 断言：无 NaN、每步 1 次专家前向、`attempts ≥ inferences ≥ executed`、
   `ΣE == cumulative`、Dev SR 可计算；
3. 记录：reward 分项均值、D_sum、能耗比、模型选择分布、约束违反次数；
4. 与三条基线在 Dev-60 上对比；
5. 全部通过后再讨论 4000-episode 正式训练。
