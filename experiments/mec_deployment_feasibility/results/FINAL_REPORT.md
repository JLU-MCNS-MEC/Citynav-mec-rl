# CityNav-MEC × MEC-v8.2：RL 训练前时延与部署可行性 —— 最终交付报告

生成时间: 2026-11-05　分支: `exp/citynav-mec-fixed50m-topdown`
**本轮只做部署规模、实测推理时延、理论计算时延、通信时延、E2E 时延及 Baseline
闭环测试；未启动任何 DQN/PPO 训练。**

---

## 1. MEC 部署数量与位置建议表

| K | 可通行区覆盖 | 任务可达率 | 最近距离 Mean / P95 | 任务点可用数 均值/最小 | 最近分配负载σ | 通信 P95 |
|---|---|---|---|---|---|---|
| 5 | 100% | 100% | 98 / 207 m | 5.0 / 4 | 1.3 | 9.7 ms |
| 8 | 100% | 100% | 75 / 168 m | 8.0 / 7 | 2.2 | 7.9 ms |
| **10（推荐，固定）** | **100%** | **100%** | **65 / 142 m** | **10.0 / 8** | **2.8** | **6.7 ms** |
| 12 | 100% | 100% | 59 / 144 m | 12.0 / 11 | 3.3 | 6.8 ms |
| 15 | 100% | 100% | 52 / 124 m | 15.0 / 14 | 4.3 | 6.0 ms |
| 20 | 100% | 100% | 44 / 107 m | 20.0 / 17 | 5.7 | 5.5 ms |

- 统计范围：Train 22,002 + Val-Seen 2,498 + Val-Unseen 2,826 = **27,326 个任务起点**，
  28 张真实 CityNav 地图；可通行区 = 0.1m 高程 TIF 的 1m 有效掩码。
- **结论：10 个 MEC 足够并固定**（100% 覆盖/可达；每任务 ≥8 个可用节点；
  再增加仅把通信 P95 从 6.7 降到 5.5ms，成本线性增长而收益边际）。
- 位置：`results/mec_positions_k10.json`（28 图 × 10 节点，均匀部署）。
- 注：v8.2 的 120m 最小间距在 13/28 张真实地图需放宽（真实街区碎片化程度
  高于合成环境），逐图记录 `spacing_relaxed`。

## 2. 三种部署配置图

- 图：`fig_deployment_k.png`（K vs 覆盖/可达/距离/通信）
- 配置：`results/deployment_configs.json`
  - **A: 10×Small**、**B: 10×Transformer-M**、**C: 5×Small + 5×Transformer-M**
  - 同位置、同带宽 5MHz、同发射功率 0.1W、同信道模型、同覆盖半径 500m、
    同任务集；仅模型类型不同（主分析算力同为 0.2 TFLOPS/节点以隔离模型效应）

## 3. 实测推理时延表 + 柱形图（RTX 5090）

batch=1、warmup 50 次、300 次重复、每次 `torch.cuda.synchronize()`。
输入 = val_seen:00000 真实一步（与 `run_closed_loop.py` 完全一致的 V5 Fix 契约）。

| 段 | Mean | P50 | P95 | P99 |
|---|---|---|---|---|
| 视觉编码（FrozenEncoder → 49×2048） | 1.680 ms | 1.671 | 1.732 | — |
| 输入预处理（CPU 装配） | 0.003 ms | — | — | — |
| Small 前向 | **2.761 ms** | 2.705 | 3.186 | 3.937 |
| Medium 前向 | **2.937 ms** | 2.903 | 3.015 | 3.107 |
| Transformer-M 前向 | **3.355 ms** | 3.348 | 3.420 | 3.460 |

- 图：`fig_measured_vs_theoretical.png`
- 关键：Small→TM 实测差仅 **0.594ms**、Medium→TM 差 **0.418ms**；batch=1 小模型
  在 5090 上是 launch-bound（~2.7ms 平台），**真实硬件不能代理 MEC 算力档位**。

## 4. 理论推理时延表 + 柱形图（算力为人为建模假设）

FLOPs（运行期 hook，2×MACs 口径，与 v8.2 一致）：Small **4.5M** / Medium **55.1M** /
Transformer-M **556.0M**（123× / 10.1×）。

| 算力 | Small | Medium | Transformer-M | Δ Small→TM | Δ Medium→TM | 理论/实测(S/M/TM) |
|---|---|---|---|---|---|---|
| 0.1 TFLOPS | 0.04 ms | 0.55 ms | 5.56 ms | 5.51 | 5.01 | 0.02/0.19/1.66 |
| **0.2 TFLOPS** | 0.02 ms | 0.28 ms | **2.78 ms** | **2.76** | **2.50** | 0.01/0.09/0.83 |
| 0.3 TFLOPS | 0.01 ms | 0.18 ms | 1.85 ms | 1.84 | 1.67 | 0.01/0.06/0.55 |
| 0.5 TFLOPS | 0.01 ms | 0.11 ms | 1.11 ms | 1.10 | 1.00 | 0.00/0.04/0.33 |
| 1.0 TFLOPS | 0.00 ms | 0.06 ms | 0.56 ms | 0.55 | 0.50 | 0.00/0.02/0.17 |

- 图：`fig_theoretical_latency.png`、`fig_share.png`
- **算力档位是人为实验建模假设，非硬件测量值**；历史 0.2 TFLOPS 出处不可追溯
  （GitHub mec-v8.2 仅单提交、无 Stage-2k/2l 报告），故做全档位扫描。
- 结论：任何档位下 Small→TM 与 Medium→TM 的差都相近（FLOPs 比值固定）；
  要复现"推理主导 E2E"的设定需算力 <0.05 TFLOPS 或更小 payload。

## 5. 通信 / 推理 / E2E 时延及占比表

通信（27,326 任务 × 最近 MEC，真实 CityNav payload 1.6 Mbit）：
**Mean 26.40 / P50 24.69 / P95 35.94 / P99 46.67 ms**

| 配置 | 推理 | 通信 | 编码 | E2E Mean | E2E P95 | E2E P99 | 推理占比 | 通信占比 |
|---|---|---|---|---|---|---|---|---|
| A 10×Small | 0.023 | 26.40 | 1.68 | 28.1 | 37.6 | 48.4 | 0.1% | 93.8% |
| B 10×TM | 2.780 | 26.40 | 1.68 | 30.9 | 40.4 | 51.1 | 9.2% | 85.3% |
| C 5+5 | 1.537 | 26.40 | 1.68 | 29.6 | 39.6 | 51.0 | 5.0% | 89.2% |

- 图：`fig_e2e_breakdown.png`
- 下行时延：动作整数回传可忽略（与 v8.2 同口径，v8.2 total 亦不含下行）。

## 6. Baseline E2E / SR 对比表 + 图

| 配置 | 策略 | SR | E2E Mean | E2E P95 | 推理占比 |
|---|---|---|---|---|---|
| A 10×Small | random | 34.67% | 54.3 | 97.1 | 0.05% |
| A 10×Small | nearest=max_rate=min_latency=best_quality | 34.67% | **27.9** | 37.0 | 0.08% |
| B 10×TM | random | 53.60% | 57.4 | 100.7 | 5.68% |
| B 10×TM | nearest=max_rate=min_latency=best_quality | 53.60% | **30.7** | 40.0 | 9.21% |
| C 5+5 | random | 46.20% | 56.4 | 99.5 | 2.90% |
| C 5+5 | nearest / max_rate | 44.96% | **29.5** | 39.2 | 5.08% |
| C 5+5 | min_latency | **42.47%** | 29.2 | 39.1 | 3.28% |
| C 5+5 | **best_quality** | **53.56%** | 32.7 | 43.3 | 8.69% |

- 图：`fig_assoc_sr.png`、`fig_assoc_e2e.png`
- 同质配置（A/B）除 random 外策略等价（模型相同，质量打平）；
  **异质配置（C）中 best_quality 仅用 5 个 TM 即达到全 TM 的 SR（53.56% vs 53.60%），
  代价仅 +3.2ms E2E**；**min_latency 最差（42.47%）——纯时延优化牺牲 11pp 成功率**。

## 7. 推理时延差异对 E2E 的贡献分析

- Small→TM 推理差 **2.758 ms = E2E 的 9.09%**；Medium→TM 差 **2.504 ms = 8.25%**
  （@0.2 TFLOPS，逐任务均值）。
- 通信是 TM 推理的 9.5 倍、Small 推理的 1150 倍；同配置内仅距离抖动就 ±11ms，
  **大于任何模型间推理差异 → 通信主导，掩盖模型算力差异**。
- 算力敏感性：即使 0.1 TFLOPS，TM 推理占比也只有 16.8%；0.2→1.0 TFLOPS 区间
  推理占比 9.2%→2.0%。

## 8. 是否值得进入 RL 训练的结论

**值得，但奖励设计与 RL 状态接口是关键前提：**

1. **静态策略已有强基线**：C/best_quality = (SR 53.56%, E2E 32.7ms)；
   RL 必须超越它（例如在 SR≈53% 下把 E2E 压向 29ms，或在 E2E≈30ms 下提高 SR）。
2. **纯时延奖励不可行**：min_latency SR 42.47%，比 best_quality 低 11pp；
   奖励必须包含任务成功/质量，或作为硬约束。
3. **RL 状态必须显式包含节点模型类型与计算能力**（v8.2 中这些对 MEC 输入
   不可见）——否则无法区分"近的 Small 节点"与"稍远的 TM 节点"，而本实验证明
   这一区分正是成功率的决定因素。
4. 通信占 E2E 85-94%，时延优化的主要杠杆在**选近节点**，而质量优化的杠杆在
   **选强模型**；RL 的价值在于逐任务/逐步权衡二者（Pareto 前沿），而非单一优化。

## 9. 完整实验配置、数据来源、代码路径与复现命令

**数据来源**
- 任务：`experiments/06_closed_loop_success_rate/results/seed42/manifest.json`
  （Val-Seen 2,498 / Val-Unseen 2,826）；
  `citynav_train_seen.json`（Train 22,002，从保留的 `official/data.tar.gz` 恢复；
  见注意事项）
- 地图：`dataext/htnav-repro/data/rgbd/*.tif`（0.1m 高程）+ `*.png`
- 教师池（仅用于本报告的任务定义参考）：11A 18,521 eps / 1,279,882 states
- 模型权重：`runs_small`、`runs_medium`、`runs_11c_v5fix` 的 `seed42/best.pt`
  （SHA256 见 `11B_transformer_m_v4/results/SHA256_MANIFEST.md`）

**代码路径**（全部在 `experiments/mec_deployment_feasibility/`）
- `scripts/analyze_deployment.py` — 阶段 2 部署数量分析
- `scripts/measure_inference_latency.py` — 阶段 3 实测时延
- `scripts/estimate_flops_capacity.py` — 阶段 4 FLOPs 与敏感性
- `scripts/analyze_e2e.py` — 阶段 5+6 E2E 与三配置
- `scripts/run_association_loop.py` — 阶段 7 C 配置真实闭环
- `scripts/replay_association.py` — 阶段 7 A/B 等价重放
- `scripts/summarize_association.py` — 汇总与图表

**复现命令**
```bash
PY=~/miniconda3/envs/citynav-mec/bin/python
cd /mnt/hdd2t/workspaces/Citynav-mec-rl-fixed50m
$PY experiments/mec_deployment_feasibility/scripts/analyze_deployment.py
$PY experiments/mec_deployment_feasibility/scripts/measure_inference_latency.py
$PY experiments/mec_deployment_feasibility/scripts/estimate_flops_capacity.py
$PY experiments/mec_deployment_feasibility/scripts/analyze_e2e.py
for s in random nearest max_rate min_latency best_quality; do
  $PY experiments/mec_deployment_feasibility/scripts/replay_association.py --config A --strategy $s
  $PY experiments/mec_deployment_feasibility/scripts/replay_association.py --config B --strategy $s
  $PY experiments/mec_deployment_feasibility/scripts/run_association_loop.py --config C --strategy $s --out-dir assoc_C_$s
done
$PY experiments/mec_deployment_feasibility/scripts/summarize_association.py
```

**注意事项（数据缺失/协议不一致）**
1. `mec-v8.2` GitHub 仓库仅 1 个提交（`d988a9a`，MAIN v8.2 input-redesign），
   **无 Stage-2k/2l 报告、无 DQN/PPO 代码、无历史 10-MEC 配置**；0.2 TFLOPS
   与历史拓扑不可追溯 → 按用户指示改为**均匀部署**，算力做全档位扫描并标注
   为建模假设。历史 10-MEC 版本/位置/拓扑**无法核实**。
2. 清理任务删除 `official/extract` 时连带删除了 `sbfnav-data/original_citynav`
   软链的目标（删除前核对对象正是软链自身，未解析到真实目录）→ 已从保留的
   `data.tar.gz` 完整恢复（754M，Train 22,002 集可用）。
3. 本轮未跑 oracle 策略（需真实最优动作，属特权上界、不可部署）。
4. 阶段 7 的 C 配置逐 step 记录（选择的 MEC / 模型类型 / 推理时延 / 通信时延 /
   E2E）已完整保存于 `results/assoc_C_*/episodes/`（**2,498 任务 × 5 策略**，
   合计 12,490 个 episode 文件 + 5 个 journal；此前"2,499"为笔误，已按
   task-file 与 manifest 双向核对修正），
   约 285MB，已提交仓库；仓库 pack 因此增大至约 460MB）。
5. Val-Seen 仅用于独立验证与 Baseline 闭环，未参与任何训练或调参。
