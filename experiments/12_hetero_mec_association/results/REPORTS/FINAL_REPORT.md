# Stage 12 — 异构算力 MEC 关联策略与 RL 实验报告

生成日期：2026-11-17　分支：`exp/citynav-mec-fixed50m-topdown`

## 1. 实验目标

在不重新训练任何 CityNav 导航模型、不修改权重与输入模态的前提下，只改变
MEC 节点托管的**任务计算量**与节点**算力档位**，研究：

- 不同关联策略（Random / Max-rate / Max-quality / Oracle）对真实闭环
  SR、E2E 时延及推理时延的影响；
- DQN / PPO 关联智能体在纯 E2E 奖励下的行为；
- 两模型混合（5 Small + 5 Transformer-M）与五模型等量部署（各 2 个）的差异；
- 同构 10 TFLOPS 与异构 5/10/20/30 TFLOPS 的差异。

导航专家完全冻结：Small、Medium、Transformer-S、Transformer-M、Large
（Stage-11 的五个 `best.pt`；运行前后 SHA256 不变）。

## 2. 冻结协议

### 2.1 计算量与时延

- 时延建模计算量（用户给定，非实测）：Small 5 / Medium 15 /
  Transformer-S 40 / Transformer-M 100 / Large 200 GFLOPs。
- UAV 视觉编码器实测 **8.174 GFLOPs**
  （冻结 ResNet-50 layer4，224×224，2×MACs），按 1.5 TFLOPS 折算
  **T_encode = 5.450 ms**。
- MEC 推理：`T_inf = GFLOPs / 节点 TFLOPS`。
- 通信完全沿用 CityNav/mec-v8.2：payload 49×2048×16 = 1,605,632 bit，
  2 GHz、5 MHz、0.1 W、500 m，LOS/NLOS + Shannon。
- `T_E2E = T_encode + T_uplink + T_inference`；下行忽略。

### 2.2 四组部署（每图 10 节点，24 张地图）

| 组 | 模型放置 | 算力 |
|---|---|---|
| two_homo | G1/G3/G5/G7/G9=Small；G2/G4/G6/G8/G10=TM | 10 TFLOPS |
| two_hetero | 同上 | G1=5, G2=10, G3=20, G4=30, G5=10, G6=5, G7=30, G8=20, G9=10, G10=10 |
| five_homo | S,M,TS,TM,L,S,M,TS,TM,L | 10 TFLOPS |
| five_hetero | 同上 | G1=5, G2=10, G3=20, G4=30, G5=10, G6=30, G7=20, G8=10, G9=5, G10=20 |

### 2.3 Baseline

- **random**：覆盖集内均匀抽样；逐任务 RNG `[seed, task_index]`。
- **max_rate**：Shannon 上行速率最大（距离单调 → 等价最近）。
- **max_quality**：质量先验最大的模型；平局取最低瞬时 E2E。先验为训练侧
  Val Macro-F1（Small 0.7243 / Medium 0.7103 / TS 0.7258 / TM 0.7475 /
  Large 0.7340），不含 Val-Seen。
- **oracle（特权上界，不可部署）**：每步用真实目标坐标 + 高程障碍图调用
  Stage-03B 教师得到 `a*`；在动作等于 `a*` 的可用节点中取最低 E2E；若无节点
  同意，取 `P(a*)` 最大者（E2E 平局裁决）。它界定的是“固定冻结专家池下
  关联层的动作正确率上界”，不是严格 SR 数学上界，也不是 E2E 下界。

### 2.4 评测与 RL

- Val-Seen 全部 **2,498** 任务；成功半径 20 m；最大 200 步；逐步真实闭环。
- RL 状态 4 + 10×19；动作 10 节点；奖励为方案 D（纯 E2E）：
  `u_t=1−e2e_t/100`，`r_t=u_t/400`，终止修正；SR 不进入奖励。
- 训练池：11A Train 18,521 → Dev-60 + Train-4,000；Val-Seen 不参与训练。
- 每个组/算法 2 个种子（42, 43），4,000 episodes；
  best 按最低 Dev-60 平均 E2E 选择。

## 3. Baseline 结果（2,498 任务）

| 组 | 策略 | SR | E2E mean | P95 | MEC Inf | Comm | UAV Enc | 专家选择 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| two_homo | random | 48.84% | 62.49 | 80.63 | 5.30 | 51.75 | 5.45 | Transformer-M 50.1%, Small 49.9% |
| two_homo | max_rate | 44.96% | 37.35 | 45.81 | 5.78 | 26.12 | 5.45 | Transformer-M 55.4%, Small 44.6% |
| two_homo | max_quality | 53.56% | 43.46 | 50.77 | 10.00 | 28.01 | 5.45 | Transformer-M 100.0% |
| two_homo | oracle | 60.33% | 36.65 | 44.75 | 3.21 | 27.99 | 5.45 | Small 65.0%, Transformer-M 35.0% |
| five_homo | random | 41.83% | 64.67 | 82.57 | 7.25 | 51.97 | 5.45 | Large 20.1%, Medium 20.1%, Transformer-S 20.1%, Transformer-M 19.9%, Small 19.8% |
| five_homo | max_rate | 41.07% | 41.02 | 52.88 | 9.32 | 26.25 | 5.45 | Large 29.0%, Transformer-M 25.8%, Transformer-S 17.2%, Small 15.0%, Medium 12.9% |
| five_homo | max_quality | 53.56% | 50.82 | 71.26 | 10.00 | 35.37 | 5.45 | Transformer-M 100.0% |
| five_homo | oracle | 76.02% | 41.84 | 55.16 | 4.68 | 31.71 | 5.45 | Medium 28.7%, Small 24.6%, Transformer-S 20.4%, Transformer-M 16.6%, Large 9.6% |
| two_hetero | random | 48.84% | 62.30 | 80.32 | 5.11 | 51.75 | 5.45 | Transformer-M 50.1%, Small 49.9% |
| two_hetero | max_rate | 44.96% | 37.43 | 47.84 | 5.87 | 26.12 | 5.45 | Transformer-M 55.4%, Small 44.6% |
| two_hetero | max_quality | 53.56% | 41.65 | 50.53 | 7.04 | 29.16 | 5.45 | Transformer-M 100.0% |
| two_hetero | oracle | 60.33% | 36.03 | 44.70 | 2.54 | 28.03 | 5.45 | Small 62.0%, Transformer-M 38.0% |
| five_hetero | random | 41.83% | 63.71 | 81.55 | 6.30 | 51.97 | 5.45 | Large 20.1%, Medium 20.1%, Transformer-S 20.1%, Transformer-M 19.9%, Small 19.8% |
| five_hetero | max_rate | 41.07% | 40.36 | 51.29 | 8.66 | 26.25 | 5.45 | Large 29.0%, Transformer-M 25.8%, Transformer-S 17.2%, Small 15.0%, Medium 12.9% |
| five_hetero | max_quality | 53.56% | 53.24 | 75.32 | 11.01 | 36.78 | 5.45 | Transformer-M 100.0% |
| five_hetero | oracle | 76.02% | 41.03 | 53.90 | 3.82 | 31.76 | 5.45 | Medium 27.8%, Small 23.5%, Transformer-S 20.9%, Large 14.9%, Transformer-M 12.9% |

## 4. PPO / DQN 结果（2,498 任务）

| 组 | 策略 | SR | E2E mean | P95 | MEC Inf | Comm | 专家选择 |
|---|---|---:|---:|---:|---:|---:|---|
| two_homo | ppo-s42 | 36.15% | 37.79 | 49.11 | 1.67 | 30.67 | Small 86.5%, Transformer-M 13.5% |
| two_homo | ppo-s43 | 36.63% | 37.54 | 49.92 | 1.04 | 31.05 | Small 92.6%, Transformer-M 7.4% |
| two_homo | dqn-s42 | 46.60% | 54.57 | 77.09 | 6.69 | 42.44 | Transformer-M 64.2%, Small 35.8% |
| two_homo | dqn-s43 | 43.59% | 51.55 | 73.06 | 4.33 | 41.77 | Small 59.9%, Transformer-M 40.1% |
| five_homo | ppo-s42 | 31.35% | 41.34 | 56.65 | 2.37 | 33.52 | Small 39.2%, Medium 34.1%, Transformer-S 20.7%, Large 4.6%, Transformer-M 1.4% |
| five_homo | ppo-s43 | 34.83% | 40.12 | 52.09 | 2.98 | 31.68 | Small 37.2%, Medium 32.0%, Transformer-M 20.3%, Transformer-S 10.5% |
| five_homo | dqn-s42 | 37.95% | 58.06 | 80.24 | 5.15 | 47.46 | Medium 30.8%, Transformer-M 25.2%, Transformer-S 21.0%, Small 15.5%, Large 7.4% |
| five_homo | dqn-s43 | 34.87% | 55.61 | 77.11 | 4.34 | 45.82 | Medium 39.3%, Transformer-S 27.5%, Small 14.5%, Transformer-M 10.3%, Large 8.4% |
| two_hetero | ppo-s42 | 39.63% | 37.94 | 50.84 | 1.90 | 30.59 | Small 72.3%, Transformer-M 27.7% |
| two_hetero | ppo-s43 | 34.75% | 36.26 | 48.31 | 0.47 | 30.33 | Small 98.0%, Transformer-M 2.0% |
| two_hetero | dqn-s42 | 42.35% | 51.50 | 75.88 | 2.57 | 43.48 | Small 68.6%, Transformer-M 31.4% |
| two_hetero | dqn-s43 | 43.63% | 53.04 | 78.18 | 4.25 | 43.34 | Small 58.5%, Transformer-M 41.5% |
| five_hetero | ppo-s42 | 37.55% | 40.88 | 55.24 | 3.52 | 31.91 | Small 39.2%, Large 23.3%, Medium 19.3%, Transformer-S 11.2%, Transformer-M 7.0% |
| five_hetero | ppo-s43 | 34.91% | 40.49 | 54.41 | 3.97 | 31.06 | Small 47.0%, Medium 21.1%, Transformer-S 13.0%, Transformer-M 12.2%, Large 6.7% |
| five_hetero | dqn-s42 | 33.71% | 57.50 | 78.96 | 4.76 | 47.29 | Medium 42.4%, Large 22.2%, Small 17.6%, Transformer-S 10.6%, Transformer-M 7.2% |
| five_hetero | dqn-s43 | 39.83% | 57.86 | 80.39 | 5.95 | 46.46 | Medium 25.1%, Transformer-M 23.8%, Large 20.8%, Transformer-S 15.6%, Small 14.7% |

## 5. 每组对比（SR / E2E mean）

| 组 | max_rate | max_quality | oracle | 最佳 PPO | 最佳 DQN |
|---|---|---|---|---|---|
| two_homo | 44.96% / 37.35 | 53.56% / 43.46 | 60.33% / 36.65 | 36.63% / 37.54 | 43.59% / 51.55 |
| five_homo | 41.07% / 41.02 | 53.56% / 50.82 | 76.02% / 41.84 | 34.83% / 40.12 | 34.87% / 55.61 |
| two_hetero | 44.96% / 37.43 | 53.56% / 41.65 | 60.33% / 36.03 | 34.75% / 36.26 | 42.35% / 51.50 |
| five_hetero | 41.07% / 40.36 | 53.56% / 53.24 | 76.02% / 41.03 | 34.91% / 40.49 | 33.71% / 57.50 |

## 6. 关键发现

### 6.1 导航轨迹只由“专家序列”决定，算力只改变时延

同一策略在 HOMO 与 HETERO 下 SR 逐百分点一致：
random 48.84%/41.83%，max_rate 44.96%/41.07%，max_quality 53.56%，
oracle 60.33%/76.02%（two/five）。节点位置与模型放置相同，算力档位不影响
模型输入或动作。因此 HETERO 的 SR 差异全部来自“策略是否因算力改变选路”，
而不是算力本身改变导航。

### 6.2 算力异构的时延效应可正可负

- `two_hetero/max_quality`：E2E 43.46 → **41.65 ms（−1.81）**，因为 TM 有
  G4=30、G8=20 的高档节点，平局裁决会选“远一点但推理更快”的 TM。
- `five_hetero/max_quality`：E2E 50.82 → **53.24 ms（+2.42）**，因为五模型
  部署中 TM 只有 G4=30 与 G9=5 两个节点，近处常常是 5 TFLOPS 的 G9
  （20 ms 推理），质量先验被锁死在低档 TM 上。
- Oracle：two_hetero 36.65 → 36.03（−0.62），five_hetero 41.84 → 41.03
  （−0.81）。

**结论：异构算力的收益取决于“质量模型是否分布在高档节点”，不是有异构就更好。**

### 6.3 专家多样性是最大的关联上界来源

- 两模型 Oracle：SR **60.33%**，E2E 36.65 ms（two_homo）/36.03 ms（two_hetero）。
- 五模型 Oracle：SR **76.02%**，E2E 41.84/41.03 ms；教师动作一致率仅从
  47.13% 升到 51.94%，但 SR 提升 15.69 pp——关键步的一致性被轨迹放大。
- 五模型 Oracle 中被选最多的是 **Medium（27.8–28.7%）**，而 Medium 单独闭环
  SR 仅 16.67%。**关联上界取决于专家动作的多样性，而不是单个专家的质量。**

### 6.4 静态策略的 Pareto 前沿

- 最低 E2E：`two_hetero/oracle` **36.03 ms（SR 60.33%）**。
- 最高 SR：`five_hetero/oracle` **76.02%（E2E 41.03 ms）**。
- `max_rate` 是“非特权”最强时延策略：37.35–41.02 ms，SR 41.07–44.96%。
- `max_quality` 是非特权最强 SR：53.56%（所有部署），E2E 41.65–53.24 ms。

### 6.5 PPO：纯 E2E 奖励收敛到 Small，SR 被牺牲

- two_homo：SR 36.15/36.63%，E2E 37.79/37.54 ms；Small 占 86.5–92.6%。
  被 `max_rate`（44.96% / 37.35）同时支配。
- two_hetero：s43 E2E **36.26 ms**（接近 Oracle 36.03），但 SR 34.75%
  （Small 98%）；s42 SR 39.63%、E2E 37.94 ms（TM 27.7%）。
- five_homo：SR 31.35/34.83%，E2E 41.34/40.12 ms；s43 E2E 略优于 max_rate
  41.02，但 SR 低 6.24 pp。
- five_hetero：s42 SR **37.55%**、E2E 40.88 ms，通过高档位 **Large 23.3%**
  把 SR 从 HOMO s42 的 31.35% 提升 6.20 pp，同时 E2E −0.46 ms——异构确实
  可以让“高质量模型”变得可负担；但 s43 没有获得同样收益（SR 34.91%，
  Large 6.7%），种子方差大。
- **没有任何 PPO run 在 SR 与 E2E 上同时超过 max_rate**；最佳 PPO E2E
  36.26 ms 距 Oracle 36.03 仅 0.23 ms，但 SR 低 25.58 pp。

### 6.6 DQN：不稳定，时延显著恶化

- two_homo：SR 46.60/43.59%，E2E 54.57/51.55 ms；TM 使用率 64.2%/40.1%。
  SR 高于 PPO，但被 `max_quality`（53.56% / 43.46）完全支配。
- five_homo：SR 37.95/34.87%，E2E 58.06/55.61 ms；通信 45.8–47.5 ms，
  说明选了远处节点。
- 异构组同样不稳定：two_hetero s42 E2E −3.07 ms、s43 +1.49 ms；
  five_hetero s42 E2E −0.56 ms 但 SR −4.24 pp。
- **DQN 的主要问题不是算力感知，而是选路质量**：推理已降到 2.6–4.8 ms，
  但通信仍 41.8–47.5 ms，总 E2E 远高于 PPO 与静态策略。

### 6.7 统计检验

配对 McNemar（SR）与 paired bootstrap（E2E，10,000 次）见
`results/summary/paired_stats.md`。代表性结果：

- `max_rate vs oracle`（two_homo）：ΔSR −15.37 pp，p=5.5e-68；
  ΔE2E +0.71 ms（oracle 更低）。
- `max_quality vs oracle`：ΔSR −6.77 pp，p=4.2e-17；ΔE2E +6.81 ms。
- `max_rate vs ppo-s42`：ΔSR +8.81 pp，p=1.5e-23；ΔE2E −0.44 ms
  （PPO 的 E2E 略高，SR 显著更低）。

## 7. 限制

1. 专家 GFLOPs 是建模假设，不是实测；与 Stage-11 实测 FLOPs 不可混用。
2. 无 MEC 排队、无切换开销；每架 UAV 独立闭环，不建模多 UAV 竞争。
3. 定位匹配（layer-2 编码器）为 UAV 侧控制开销，未计入 E2E（与 Stage-11 同口径）。
4. 每个 RL 配置 2 个种子、4,000 episodes；五模型 × 四档位的动作/算力联合空间
   对 PPO/DQN 仍偏大，种子方差显著。
5. 5/10/20/30 节点分配为本次预注册假设；仓库中不存在更早的具体分配。
6. 仅评测 Val-Seen；Test-Unseen 未使用。

## 8. 结论

- **只优化 E2E 的 RL 会系统性放弃质量模型**：PPO 收敛到 ~90% Small，
  DQN 则不稳定地在 Small/TM/Medium 间摆动。SR 作为独立指标揭示了这一代价。
- **静态 `max_rate` 是极强的 E2E 基线**；`max_quality` 是极强的 SR 基线；
  学习策略没有同时超过两者。
- **专家多样性是未被利用的资源**：五模型 Oracle 达 76.02% SR，而最佳 PPO
  仅 39.63%。若 SR 重要，下一步应引入 SR/质量约束、多目标奖励或
  “动作多样性”先验，而不是继续加长纯 E2E 训练。
- **异构算力的价值取决于部署设计**：把质量模型放在高档节点（two_hetero）
  可降低 E2E；把质量模型锁在低档节点（five_hetero 的 G9=5 TFLOPS）会伤害
  max_quality。RL 可以利用档位，但收益对种子与部署敏感。

## 9. 产物与复现

- 代码与协议：`experiments/12_hetero_mec_association/{scripts,tests,PROTOCOL.md,README.md}`
- 冻结部署与编码 FLOPs：`results/protocol/`
- Baseline 逐任务结果：`results/baselines/<group>/<strategy>/seed42/`
- RL 训练：`results/rl/<group>/<algo>/seed<42|43>/`
- RL 评测：`results/eval/<group>_<algo>_seed<seed>/`
- 汇总表：`results/summary/RESULTS.md`、`paired_stats.md`
- 图：`results/analysis/bars_<group>.png`、`m100_<group>.png`、`pareto_sr_e2e.png`

复现命令见 `README.md`；逐 episode 原始 journal 因体积较大不随仓库推送。
