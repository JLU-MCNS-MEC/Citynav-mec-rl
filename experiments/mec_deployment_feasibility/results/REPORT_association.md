# 阶段 7：三配置 × Baseline 关联策略闭环评测报告

生成时间: 2026-11-05
任务: Val-Seen 完整 2,498 任务（统一 manifest，SHA `2da37816…`）；STOP 半径 20m、
最大 200 步；无候选任务计入分母（本轮 Val-Seen 无候选 = 0）。
策略 = mec-v8.2 `association_strategies.py` 实际实现：random / nearest / max_rate /
min_latency / best_quality（oracle 为特权上界、需真实最优动作，不可部署，未跑）。

## 一、主结果表

| 配置 | 策略 | SR | E2E Mean | E2E P95 | 推理均值 | 通信均值 | 推理占比 |
|---|---|---|---|---|---|---|---|
| A 10×Small | random | 34.67% | 54.3 ms | 97.1 | 0.023 | 52.6 | 0.05% |
| A 10×Small | nearest | 34.67% | **27.9 ms** | 37.0 | 0.023 | 26.2 | 0.08% |
| A 10×Small | max_rate | 34.67% | 27.9 | 37.0 | 0.023 | 26.2 | 0.08% |
| A 10×Small | min_latency | 34.67% | 27.9 | 37.0 | 0.023 | 26.2 | 0.08% |
| A 10×Small | best_quality | 34.67% | 27.9 | 37.0 | 0.023 | 26.2 | 0.08% |
| B 10×TM | random | 53.60% | 57.4 ms | 100.7 | 2.780 | 53.0 | 5.68% |
| B 10×TM | nearest | 53.60% | **30.7 ms** | 40.0 | 2.780 | 26.2 | 9.21% |
| B 10×TM | max_rate | 53.60% | 30.7 | 40.0 | 2.780 | 26.2 | 9.21% |
| B 10×TM | min_latency | 53.60% | 30.7 | 40.0 | 2.780 | 26.2 | 9.21% |
| B 10×TM | best_quality | 53.60% | 30.7 | 40.0 | 2.780 | 26.2 | 9.21% |
| C 5+5 | random | 46.20% | 56.4 ms | 99.5 | 1.404 | 53.3 | 2.90% |
| C 5+5 | nearest | 44.96% | **29.5 ms** | 39.2 | 1.550 | 26.3 | 5.08% |
| C 5+5 | max_rate | 44.96% | 29.5 | 39.2 | 1.550 | 26.3 | 5.08% |
| C 5+5 | min_latency | **42.47%** | 29.2 | 39.1 | 1.012 | 26.5 | 3.28% |
| C 5+5 | best_quality | **53.56%** | 32.7 ms | 43.3 | 2.780 | 28.3 | 8.69% |

## 二、关键发现

1. **同质配置（A/B）中除 random 外所有策略等价**：模型相同 → 质量打平 →
   best_quality 退化为最近；距离单调 → max_rate = nearest；推理时延差（≤2.8ms）
   远小于节点间通信差 → min_latency ≈ nearest。**随机选路的代价极大**：
   E2E 翻倍（A: 54.3 vs 27.9ms；B: 57.4 vs 30.7ms）。
2. **异质配置（C）中策略选择真正影响任务成功率**：
   - **best_quality（选 TM）SR 53.56%**，与 B（全 TM）的 53.60% 几乎相同，
     而 E2E 仅比最优时延策略高 **+3.2ms**（32.7 vs 29.5ms）——**用 5 个 TM
     就能获得全 TM 的闭环质量，代价是多 3ms 延迟**。
   - **min_latency（偏好 Small）SR 42.47%**，是所有策略中最差——对时延的
     单目标优化会牺牲近 11 个百分点的成功率。
   - nearest/max_rate（空间选路，不看模型类型）SR 44.96%，介于两者之间。
   - random SR 46.20% 且 E2E 56.4ms —— 双差。
3. **SR 与 E2E 的帕累托前沿**：C/best_quality = (53.56%, 32.7ms) 支配
   C/random (46.20%, 56.4ms)；A 系列虽有最低 E2E（27.9ms）但 SR 仅 34.67%。
   **没有任何单一静态策略同时达到最高 SR 与最低 E2E**——这正是 RL 的
   优化空间（Pareto 前沿的形状需要在训练中学习）。
4. **推理占比 0.05%-9.2%**：即使 best_quality 也只有 8.69%，通信始终主导
   （85-94%）。这与阶段 5 结论一致。

## 三、对 RL 训练的启示（研究目标固定：仅优化 E2E，SR 独立评价）

- **目标固定**：RL 仅优化 E2E 时延（方案 D：`1 − mean_E2E/100`）；SR/OSR/nDTW/SPL
  为独立评价指标，不进奖励、不改多目标设计、本阶段不训练。
- **基线参照**：C/best_quality (53.56%, 32.7ms)、C/nearest (44.96%, 29.5ms)、
  C/min_latency (42.47%, 29.2ms)、A/B 同质配置 27.9/30.7ms —— RL 的时延目标
  是逼近/低于这些静态策略的 E2E。
- **观察（不构成奖励修改）**：C 中 min_latency 的 SR（42.47%）低于
  best_quality（53.56%），说明时延最优与质量最优并不重合；后续训练中 SR
  将作为并列独立指标报告，用于观察时延优化对成功率的影响。
- **状态接口**：历史 Pure-MEC v2 状态已含每候选的模型类型 one-hot 与算力
  归一化（19-D/候选），RL 状态沿用即可，无需新增字段。
- 训练数据只能用 Train（11A 18,521 eps / 1,279,882 states）；Val-Seen 仅用于
  独立验证，不参与训练或调参。

## 四、证据与复现

- 脚本：
  - `scripts/run_association_loop.py`（C 配置真实逐步切换模型闭环；断点续跑）
  - `scripts/replay_association.py`（A/B 等价重放：同质配置下选路不改变轨迹）
  - `scripts/summarize_association.py`（汇总 + 图表）
- 结果：`results/assoc_summary.json`、`results/assoc_{A,B}_{strategy}.json`、
  `results/assoc_C_{strategy}/{journal.jsonl,episodes/}`、`fig_assoc_sr.png`、
  `fig_assoc_e2e.png`
- 复现命令：
  ```
  python scripts/replay_association.py --config A --strategy nearest
  python scripts/run_association_loop.py --config C --strategy best_quality \
      --out-dir assoc_C_best_quality
  python scripts/summarize_association.py
  ```
- 注：A/B 的策略级结果由阶段 1 同协议闭环轨迹（small / v5fix）重放得到；
  因同质配置下所选 MEC 不影响模型输入与动作，该重放与重跑 5 遍闭环完全等价
  （v8.2 语义：model type / compute capacity 对 MEC 输入不可见）。
