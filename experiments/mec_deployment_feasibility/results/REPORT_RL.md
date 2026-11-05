# 三种 10-MEC 部署下的 DQN / PPO 关联智能体 —— 训练与 Val-Seen 闭环评测报告

生成时间: 2026-11-06　分支: `exp/citynav-mec-fixed50m-topdown`
协议来源: mec-v8.2 `experiment/v83-spatial-foundation` Stage-2r/2o/2p（状态/奖励/算法）
**本轮未修改研究目标：RL 仅优化 E2E 时延（方案 D），SR/OSR/nDTW/SPL 为独立评价指标。**

## 一、实验设置

| 项 | 值 |
|---|---|
| 部署 | A: 10×Small；B: 10×Transformer-M；C: 5×Small+5×TM（同位置/同信道，见 `deployment_configs.json`） |
| 状态 | Pure-MEC v2 = **4 + 10×19**（相对位置/距离/算力/带宽/上行速率/通信/推理/E2E/SNR + Dev 能力先验 + 5 类模型 one-hot） |
| 奖励 | 方案 D：`u_t = 1 − e2e_t/100`；`r_t = u_t/400`；终止补 `(1−T/400)·J`；`R = 1 − mean_E2E/100` |
| 时延口径 | **解析推理**（用户指示：暂不使用实测推理时延）：`T_inf = FLOPs/0.2 TFLOPS` + 通信 + 编码 1.68ms |
| 算法 | 标准 DQN（194→256→256→10，γ=1.0，ε 1.0→0.05/200k，replay 50k，batch 128，target 1k，lr 1e-4）<br>标准 PPO（194→256→256→{10,1}，γ=1.0，GAE 0.95，clip 0.2，lr 3e-4，rollout 2048，minibatch 256，10 epochs） |
| 数据 | **11A Train 18,521** → Dev-60 + Train-4,000（seed 20261022，与 Val-Seen 无任务级重叠） |
| 能力先验 | Dev-60 最近节点策略实测：Small SR 36.7%/SPL 0.306；TM SR 63.3%/SPL 0.525（nDTW 以 SPL 代用，注明） |
| 预算 | 每 run 4,000 episodes，每 200 episodes Dev-60 验证；best = **最低 Dev E2E**（平局比 SR、更早 episode）；seed 11（历史首种子） |
| 评测 | Val-Seen 2,498 任务独立闭环，贪心策略，与 Baseline 同协议 |

## 二、训练结果（Dev-60，最佳 checkpoint）

| 配置 | 算法 | **最佳 Dev E2E** | 最佳 Episode | Dev SR | 最终 Dev E2E |
|---|---|---|---|---|---|
| A 10×Small | **PPO** | **31.94 ms** | 3600 | 36.7% | 33.03 |
| A 10×Small | DQN | 43.09 ms | 400 | 36.7% | 49.25 |
| B 10×TM | **PPO** | **34.27 ms** | 2400 | 63.3% | 35.58 |
| B 10×TM | DQN | 47.31 ms | 2000 | 63.3% | 57.44 |
| C 5+5 | **PPO** | **33.33 ms** | 3800 | 48.3% | 34.08 |
| C 5+5 | DQN | 47.41 ms | 600 | 48.3% | 60.08 |

**静态 nearest（Dev-60，参考基准）**：A 28.50ms / B 31.26ms / C 30.27ms（SR 36.7%/63.3%/53.3%）。

## 三、Val-Seen 2,498 闭环评测（RL）

| 配置/策略 | SR | OSR | NE (m) | SPL | E2E Mean | E2E P95 | E2E P99 | 推理占比 |
|---|---|---|---|---|---|---|---|---|
| A / RL-PPO | 34.51% | 35.03% | 91.4 | 0.325 | **30.87** | 40.23 | 51.82 | 0.07% |
| A / RL-DQN | 34.51% | 35.03% | 91.4 | 0.325 | 41.07 | 61.10 | 72.56 | 0.06% |
| B / RL-PPO | 53.56% | 54.12% | 55.4 | 0.468 | **34.23** | 44.64 | 56.68 | 8.29% |
| B / RL-DQN | 53.56% | 54.12% | 55.4 | 0.468 | 49.12 | 73.83 | 86.26 | 6.01% |
| C / RL-PPO | 41.43% | 41.91% | 78.0 | 0.377 | **32.42** | 41.72 | 51.44 | 3.13% |
| C / RL-DQN | 45.60% | 46.16% | 73.0 | 0.404 | 48.96 | 71.04 | 83.51 | 3.02% |

## 四、与静态 Baseline 的对比（同 Val-Seen 协议）

| 配置 | RL 最佳 E2E | 静态最优 E2E | 静态策略 | RL 最佳 SR | 静态最优 SR | 静态策略 |
|---|---|---|---|---|---|---|
| A | 30.87（PPO） | **27.79** | nearest | 34.51% | 34.67% | 任一（同质） |
| B | 34.23（PPO） | **30.69** | nearest | 53.56% | 53.60% | 任一（同质） |
| C | 32.42（PPO） | **29.21** | min_latency | 45.60%（DQN） | **53.56%** | best_quality |

完整静态表见 `assoc_summary.json`。

## 五、结论（诚实汇总）

1. **PPO 全面优于 DQN**：三种部署的 Val-Seen E2E 分别低 24.9%（A）、30.3%（B）、33.8%（C）；DQN 均在早期（ep400-2000）达到最优后退化，历史 Stage-2r 的"DQN 不稳定/未严格收敛"在本轮复现。
2. **RL 未超越静态非特权 Baseline**：三种配置下 RL 的最佳 E2E 均高于静态最优（A +3.1ms、B +3.5ms、C +3.2ms）；C 配置的 SR 也低于 best_quality（45.6% vs 53.6%）。
3. **RL 未学到"最近节点"这一显式最优规则**：在它自己的 Dev-60 选择集上，RL 最佳（31.94/34.27/33.33ms）也分别劣于 nearest（28.50/31.26/30.27ms）——不是过拟合，而是**未学会简单规则**。
4. **纯时延奖励下的质量-时延权衡**（C 配置）：PPO 更偏向 Small 节点（推理占比 3.13%，E2E 32.4ms，SR 41.4%）；DQN 更多使用 TM（SR 45.6%，E2E 49.0ms）。**两者均未同时超过 nearest（45.0%@29.5ms）或 best_quality（53.6%@32.7ms）**。
5. 与历史 Stage-2r 结论一致：**学习策略尚未优于全部非特权 Baseline**。

## 六、下一步建议（不改变奖励与研究目标）

- 增加训练预算与多种子（Stage-2r 用 3 seeds × 4000；本轮仅 seed 11），观察稳定性与最优值；
- 提升 Dev 划分规模（当前 60 个任务，验证噪声大）或增加验证频率；
- 检查状态/网络容量与探索策略（ε 衰减 200k 步在 4000 episodes 内只覆盖部分衰减）；
- 用 lane cache 加速同质部署的 A/B 训练（历史 2r 做法），把算力集中到异质 C 的在线训练。

## 七、证据与复现

- 训练脚本：`scripts/train_association_rl.py`（含断点续训、best.pt 独立保存）
- 评测脚本：`scripts/eval_association_rl.py`
- 训练产物：`results/rl_{A,B,C}_{dqn,ppo}_s11/{history.jsonl,dev.jsonl,best_meta.json,train.log,best.pt}`
- 评测产物：`results/eval_rl_{A,B,C}_{dqn,ppo}_s11/{journal.jsonl,summary.json}`
- 对比汇总：`results/rl_vs_static.json`、`results/static_nearest_dev60.json`、`results/assoc_summary.json`
- 复现命令：
  ```bash
  PY=~/miniconda3/envs/citynav-mec/bin/python
  $PY experiments/mec_deployment_feasibility/scripts/train_association_rl.py \
      --config C --algorithm ppo --seed 11 --episodes 4000 --dev-every 200 \
      --out-dir rl_C_ppo_s11
  $PY experiments/mec_deployment_feasibility/scripts/eval_association_rl.py \
      --run-dir rl_C_ppo_s11 --out-dir eval_rl_C_ppo_s11
  ```
- 备注：`results/pre_fix_20261105/` 为 best.pt 保存缺陷修复前的探索性 run（仅保留曲线，不用于任何结论）。
