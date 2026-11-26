# Stage 14 / Stage 16 代码审计（Stage 17-A 前置）

审计日期：2026-11-26
审计方式：只读源码与已有结果文件。**未运行任何仿真、训练或评测。**
原则：所有数值均给出文件路径与行号；找不到证据的条目一律写“未核实”，不推测。

---

## A. Stage 14 · 五专家模型

| 项目 | 值 | 证据 |
|---|---|---|
| 模型定义 | `SmallPreset` / `MediumPreset` / `Transformer-S` / `TransformerM12Fix` / `Large` | `experiments/14_mec_expert_retraining/models.py:35-194` |
| Small 结构 | per-token MLP 2048→12288→2048，D=512，head 2048，8×D 拼接 | `models.py:36-70` |
| Medium 结构 | goal cross-attention（8 头）+ per-token MLP 2048→36864→2048，D=512，FFN 4096 | `models.py:73-120` |
| Transformer-S | 104-token，D=1408，L=8，H=8 | `models.py:5-9`（模块 docstring） |
| Transformer-M | `TransformerM12Fix`，D=2048，L=10，H=16 | 同上 |
| Large | 104-token，D=2560，L=12，H=16 | 同上 |
| 参数动作空间 | 4 类：Stop / Forward / TurnLeft / TurnRight | 各 head 输出 4 维；`protocol.py` CLASSES |
| 冻结 Checkpoint（Small/Medium） | `checkpoints/{Small,Medium}/best.pt` | 目录实测存在（244 MB / 664 MB） |
| 冻结 Checkpoint（TS/TM/Large） | `stability_revision/runs/{ts,tm,large}/seed42/checkpoints/best.pt` | Stage 16 `p1/env16.py:75-81` CKPTS |
| 输入协议 | 104 token 多模态：current 49×2048 + history 16×三件套 + belief + match + landmarks + states + target_emb | `env16.py:246-259`、`models.py` forward |
| 归一化 | ts/tm/large 使用 Stage 14-R2 标准化；small/medium 不归一化 | `env16.py:82` `NORMALIZED = {"ts","tm","large"}` |
| Dev Macro-F1 | Small 0.5771 / Medium 0.6025 / TS 0.6500 / TM 0.6177 / Large 0.6105 | `env16.py:71-74` QUALITY_DEV_F1 |
| 闭环 SR（2498 Val-Seen） | Small 0.5392 / Medium 0.3195 / TS 0.6301 / TM 0.4251 / Large 0.3351 | `experiments/14_mec_expert_retraining/stability_revision/TRANSFORMER_STABILITY_REVISION.md:302-310` |

**Stage 14 闭环 SR 的原始报告**：上述文件第 302 行起的表格（Small 0.5392 / TS 0.6301 / TM 0.4251 / Large 0.3351 / Medium 0.3195）。

## B. 计算复杂度（实测，非估算）

| 模型 | GFLOPs（2×MACs） | 证据 |
|---|---:|---|
| Small | 4.954039 | `experiments/16_cumulative_e2e_association/p1/env16.py:66-69` |
| Medium | 14.978528 | 同上 |
| Transformer-S | 39.671126 | 同上 |
| Transformer-M | 100.025418 | 同上 |
| Large | 199.723981 | 同上 |
| 测量脚本与原始记录 | `experiments/16_cumulative_e2e_association/preflight/model_flops_audit.{json,csv}`、`measure_final_flops.py` | 目录实测存在 |

> 口径：Linear/MHA hook，**不含** LayerNorm / GELU / Softmax（`env16.py` docstring 与 preflight 审计文件）。
> 作为新脚本 `experiments/17.../` 的输入时，必须原样继承，不得重测或改口径。

## C. MEC 部署与关联动作

| 项目 | 值 | 证据 |
|---|---|---|
| 节点数 | 10 / 地图 | `env16.py:318` `build_deployment_by_map` |
| 位置来源 | `experiments/mec_deployment_feasibility/results/mec_positions_k10.json` | `env16.py:320-321` |
| 模型映射 | `DEPLOY = MODEL_ORDER * 2`，即 G1..G5 = small..large，G6..G10 = small..large | `env16.py:63-64` |
| 计算能力 | 全部 10 TFLOPS（同构） | `env16.py:84` `MEC_TFLOPS = 10.0`；`build_deployment_by_map` 写入 `compute_tflops` |
| 覆盖半径 | 500 m | `env16.py:85` `COVERAGE_M = 500.0` |
| 动作编码 | 离散 10 维 = 节点索引；`action` 直接作为 `metrics[action]` | `env16.py:201-203` |
| 覆盖掩码 | `coverage_mask()` 返回 10 维 0/1，依据 `NodeModel.available = distance <= 500` | `env16.py:113-122, 198-199` |

**注意**：Stage 16 的 `step()` **不校验覆盖**——动作若指向不可用节点仍会执行。Stage 17 必须显式用 `feasible_mec_mask` 约束，否则违反新约束定义。

## D. 时延模型（Stage 16 的“累计 E2E”）

| 分量 | 公式 | 证据 |
|---|---|---|
| 编码 `D_enc` | `encoder_flops / 1.5e12 × 1e3` | `protocol.py:199-205`；`encoder_flops.json`：8,174,272,512 FLOPs → 5.449515 ms |
| 上行 `D_up` | `PAYLOAD_BITS / uplink_rate(d) × 1e3`，`PAYLOAD_BITS = 49*2048*16 = 1,605,632 bit` | `protocol.py:128, 197-200` |
| 推理 `D_inf` | `MODEL_GFLOPS / 10.0`（ms） | `env16.py:121` |
| **下行 `D_down`** | **不存在** | `protocol.py` 仅有 `uplink_rate_bps`；`env16.py` 未计算下行 |
| 单步 E2E | `enc + comm + inf` | `env16.py:122` `self.e2e_ms = enc_ms + comm_ms + inf_ms` |
| 累计 E2E | `sum(episode_e2e)` | `env16.py:204, 313` |

**信道参数**（`protocol.py:119-134`）：

| 参数 | 值 |
|---|---|
| 带宽 | 5.0 MHz |
| UAV 发射功率 | 0.1 W |
| 载波 | 2.0 GHz |
| 噪声 PSD / 噪声系数 | -174 dBm/Hz / 9 dB |
| 路径损耗 | 自由空间 + 概率视距（A=9.61, B=0.16, 额外损耗 1.0 / 20.0 dB） |
| 高度 | MEC 10 m，UAV 80 m |
| 上传数据量 | 1,605,632 bit / 决策 |

## E. UAV 编码端

| 项目 | 值 | 证据 |
|---|---|---|
| 编码器 | torchvision resnet50 IMAGENET1K_V2 (layer4, fp16) | `encoder_flops.json` |
| 输入 | 1×3×224×224 uint8（100 m × 100 m heading crop） | 同上 |
| FLOPs | 8.174272512 G | 同上 |
| **UAV 算力** | **1.5 TFLOPS** | `protocol.py:139` `UAV_TFLOPS = 1.5e12` |
| 编码时延 | 5.4495 ms（模型）/ 3.095 ms（RTX5090 实测） | `encoder_flops.json` |
| 是否缓存复用 | Stage 16 每步重新编码（`enc4(...)` 每步调用一次） | `env16.py:260-261` |

> 审计结论：1.5 TFLOPS **不是本次新设**，它是 Stage 12 协议常量，Stage 14/16 一直沿用。Stage 17 继承该值并在参数表中标注其来源。

## F. 导航与终止规则

| 规则 | 值 | 证据 |
|---|---|---|
| 前进 | 5 m / 步 | `closed_loop.py FORWARD_M`，`env16.py` import |
| 转向 | ±30° / 步 | `TURN_RAD = radians(30)` |
| 成功半径 | 20 m | `protocol.py:143` `STOP_RADIUS = 20.0` |
| 步数上限 | 200 | `env16.py:86 MAX_STEPS` |
| 终止原因 | `success / premature_stop / collision / out_of_bounds / step_cap` | `env16.py:270-294` |
| SR 判定 | `a == 0 且 dist <= STOP_RADIUS` | `env16.py:270-273` |

## G. Stage 16 DRL 与基线

| 项目 | 值 | 证据 |
|---|---|---|
| 状态维度 | `4 + 10×7 = 74` | `train16.py:45 STATE_DIM`；`state_features` 第 50-65 行 |
| 动作维度 | 10（节点索引） | `train16.py:46 N_ACT = 10` |
| 奖励 A | `r_t = -e2e_t / 100` | `env16.py:296-299` |
| 算法 | DQN / PPO / TD3(discrete) / SAC(discrete) / DDPG(discrete) | `train16.py:141-201` |
| 连续算法的动作处理 | 用 softmax 分布采样（TD3/SAC）或 argmax（DDPG），**不存在连续动作离散化映射** | `train16.py:314-420` |
| 基线 | Random / Min-Latency / Max-Quality | `p2/baselines.py` |
| 训练规模 | 4000 episodes，seed 42，Train 4000，Dev 60 | `P5_FINAL_REPORT.md:31` |
| 评测 | 2498 Val-Seen | 同上 |

**Stage 16 最终结果（2498 Val-Seen，`P5_FINAL_REPORT.md:36-55`）**

| 指标 | Random | Min-Lat | Max-Qual | DQN | TD3 | SAC | PPO | DDPG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SR | 0.4932 | 0.4844 | **0.6289** | 0.4339 | 0.6157 | 0.4219 | 0.6249 | 0.4287 |
| 累计 E2E (ms) | 4193.98 | 2245.07 | 4024.69 | 3376.57 | 4517.00 | **2223.79** | 5358.97 | 3116.92 |

## H. 审计发现的问题（Stage 17 必须处理）

1. **无下行时延**：Stage 16 的 E2E 只有 enc+up+inf，缺少 MEC→UAV 的动作回传。Stage 17 必须新增并显式声明参数来源。
2. **覆盖未被强制**：`env16.step()` 不检查 `available`，关联到覆盖外节点不会失败。Stage 17 必须用 `feasible_mec_mask` 强制。
3. **无能量、无截止时间**：Stage 16 完全没有能量与 deadline 概念。
4. **动作到执行的动作不一致风险**：Stage 16 的 `step()` 内部计算动作并立即执行；Stage 17 要做“先算能耗、后执行”的前置检查，必须用**同状态预览动作**并保证与实际执行一致（模型确定性 argmax）。
5. **编码器算力常量分散**：`UAV_TFLOPS` 只在 Stage 12 protocol 中定义，Stage 16 直接 `flops/1.5e12` 硬编码（`env16.py:150`）。Stage 17 统一从配置读取并注明来源为 Stage 12 协议。

## I. 未核实项

| 项目 | 状态 |
|---|---|
| MEC 回传数据的确切协议格式与字节数 | **未核实**（Stage 16 未建模；Stage 17 需新增并标注为待审核参数） |
| MEC 下行发射功率 | **未核实**（Stage 12/16 只有 UAV 上行 0.1 W） |
| `encoder_flops.json` 中 1.5 TFLOPS 的物理依据 | **未核实**（文件中仅记录数值，无硬件型号说明） |
| Stage 16 Val-Seen 的 nDTW / SPL 原始数值 | 未在 `P5_FINAL_REPORT.md` 主表列出；SAC/其他算法的 SPL 曾因 10 倍错误修正，未复核最新值 → **未核实** |
