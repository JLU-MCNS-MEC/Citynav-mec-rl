# Stage 13-R3：五种 MEC 预设架构的显存、吞吐与前向时延可行性

日期：2026-11-17　状态：**只读可行性审计 + 随机输入 Benchmark；未训练、未改模型/Checkpoint**

依据：`revision_r1/PRESET_SCHEME.md`、`results/preset_scheme.csv`、`scripts/preset_scheme.py`
（R2 预设），以及本目录的随机输入 benchmark。

---

## 0. 结论摘要

1. **RTX 5090 32 GB 可以训练全部五种预设模型**，但 **Large 不能直接用 batch 128**：
   - Small / Medium / Transformer-S / Transformer-M：**batch 128 可行**，无需
     gradient checkpointing；
   - Large（1.03B 参数）：**batch 64 + 梯度累积 2（等效 128）** 可行；
     batch 128 在标准 AMP + AdamW 下估算 **33.2 GiB**，超过 32 GB。
2. **Medium 的 36,864 维 per-token MLP 不是瓶颈**：166M 参数，batch 128 的
   fwd+bwd 峰值仅 **2.66 GiB**，训练峰值约 **4.39 GiB**，fwd+bwd 每步 31.1 ms。
   36,864 宽 GEMM 在 5090 上效率很好；无需修改 R2 预设。
3. **Large 的 1B 参数是可行的**，但必须降 batch：batch 64 训练峰值约
   **24.0 GiB**，fwd+bwd 每步 250.6 ms；20 epochs 约 **37.2 h**，
   15 epochs 早停约 **27.9 h**。
4. **R2 的训练时间估算偏保守约 2 倍**：实测 BF16 有效吞吐约
   **150 TFLOPS**，不是 R2 假设的 50 TFLOPS。20 epochs 顺序训练从 R2 的
   196 h 下调到 **约 94 h**（数据加载按串行相加的保守口径）；
   15 epochs 早停约 **70 h**；若数据加载与计算重叠，可降至
   **约 51 h（20 ep）/ 39 h（15 ep）**。
5. 所有模型在推荐 batch 下顺序训练即可；单卡并行训练会争用显存与算力，
   不建议与 Large 并行。

---

## 1. 环境与方法

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32,607 MiB（约 31.8 GiB 可用） |
| 驱动 / CUDA | 580.159.03 / CUDA 12.8 |
| PyTorch | 2.9.1+cu128 |
| BF16 | `torch.cuda.is_bf16_supported() = True` |
| 计算能力 | sm_120（Blackwell） |
| 其他占用 | benchmark 前约 2.3 GB（桌面/其他进程） |

Benchmark 方法：

- 独立脚本 `scripts/benchmark_memory_latency.py`；构造五种 R2 预设模型
  （随机初始化，不加载 Checkpoint）。
- 随机输入：current 49×2048、target 512、history 16×2048 / 16×3 / 16×4、
  state 12、landmark 8×3、relation 由几何层生成。
- **标准 AMP 口径**：模型参数 FP32，`torch.autocast('cuda', bfloat16)`；
  不做 optimizer step，不加载数据，不训练。
- 计时：CUDA Event；forward 5 次、forward+backward 3 次，warmup 2 次。
- 显存：`torch.cuda.max_memory_allocated()`；每模型/每 batch 后释放缓存。
- 训练峰值估算：`fwd+bwd 峰值 + AdamW m/v (8P) + 0.5 GiB`。
- 非矩阵运算（LayerNorm/GELU/Softmax）不计入 FLOPs，但计入显存/时延。

注意：

- forward-only 的峰值在 TS/TM 上高于 fwd+bwd，是 eval fast path/allocator
  差异；训练以 fwd+bwd 峰值为准。
- 顺序 benchmark 中 Large B=128 forward-only 曾 OOM；**独立进程复测成功**：
  154.9 ms、7.28 GiB 峰值。该 OOM 是 allocator 碎片，不是真实容量不足；
  但 Large B=128 的训练峰值（含 AdamW）仍超过 32 GiB。

---

## 2. 表 1：显存组成（batch 128，标准 AMP + AdamW）

单位 GiB（1 GiB = 1024 MiB）。FP32 master 列表示若使用独立 master copy；
标准 AMP 的 FP32 参数本身即 master，不重复计。

| 模型 | 参数 | BF16 权重 | FP16 权重 | FP32 权重 | BF16 梯度 | FP32 梯度 | AdamW FP32 m/v | FP32 master | 标准 AMP 总状态 | 前向峰值 | Fwd+Bwd 峰值 | 训练峰值估算 | ≤32 GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Small | 61.1 M | 0.12 | 0.12 | 0.24 | 0.12 | 0.24 | 0.48 | 0.24 | 0.96 | 1.18 | 1.07 | **2.05** | 是 |
| Medium | 166.0 M | 0.32 | 0.32 | 0.65 | 0.32 | 0.65 | 1.30 | 0.65 | 2.59 | 3.21 | 2.60 | **4.39** | 是 |
| Transformer-S | 211.2 M | 0.41 | 0.41 | 0.83 | 0.41 | 0.83 | 1.65 | 0.83 | 3.30 | 13.2 | 8.14 | **10.29** | 是 |
| Transformer-M | 498.6 M | 0.97 | 0.97 | 1.95 | 0.97 | 1.95 | 3.90 | 1.95 | 7.79 | 24.0 | 14.80 | **19.19** | 是 |
| Large | 1,034.7 M | 2.02 | 2.02 | 4.04 | 2.02 | 4.04 | 8.09 | 4.04 | 16.17 | 7.28 | 24.61 | **33.19** | **否** |

补充：

- 若采用 **纯 BF16 参数 + BF16 梯度 + FP32 AdamW（无 FP32 master）**，
  状态为 12P；Large 的 B=128 训练峰值可降到约 **29.2 GiB**，理论可放入
  32 GB，但余量仅约 2.5 GiB，且无 FP32 master 可能影响收敛；**不推荐**
  为了维持 batch 128 而采用该方案。
- 若使用 **FP32 master + BF16 参数/梯度**，状态仍为 16P，与标准 AMP 相同。
- 训练峰值估算的 0.5 GiB 覆盖 pinned memory、CUDA workspace 与碎片余量；
  实际可能 ±1–2 GiB。

## 3. 表 2：Batch Size 可行性（训练峰值，GiB）

| 模型 | B=16 | B=32 | B=64 | B=128 | 推荐 B | 32 GB 可行 | 需要 Gradient Checkpointing |
|---|---:|---:|---:|---:|---:|---|---|
| Small | 1.53 | 1.62 | 1.71 | 2.05 | **128** | 是 | 否 |
| Medium | 3.27 | 3.28 | 3.60 | 4.39 | **128** | 是 | 否 |
| Transformer-S | 4.33 | 5.17 | 6.87 | 10.29 | **128** | 是 | 否 |
| Transformer-M | 8.93 | 10.39 | 13.33 | 19.19 | **128** | 是 | 否（余量约 12.6 GiB） |
| Large | 17.12 | 19.40 | 24.04 | **33.19** | **64 + 累积 2** | B≤64 是；B=128 否 | B=64 否；B=128 是 |

- Small / Medium / TS / TM 全部可直接 B=128。
- Large 推荐 **B=64 + 梯度累积 2**（等效 batch 128），不需要 gradient
  checkpointing；训练峰值 24.04 GiB，余量约 7.8 GiB。
- 若坚持 Large B=128：需要 gradient checkpointing（或 8-bit Adam/CPU
  offload），属于额外优化；本轮不推荐把它作为默认方案。

## 4. 表 3：单次前向时延（随机输入，BF16 autocast，实测）

每样本毫秒（ms/sample）；括号内为该 batch 的总前向耗时。

| 模型 | B=1 | B=16 | B=32 | B=64 | B=128 |
|---|---:|---:|---:|---:|---:|
| Small | 3.70 | 0.240 | 0.126 | 0.085 | **0.054** (6.89 ms) |
| Medium | 3.92 | 0.329 | 0.207 | 0.140 | **0.111** (14.23 ms) |
| Transformer-S | 5.43 | 0.791 | 0.491 | 0.345 | **0.323** (41.40 ms) |
| Transformer-M | 6.20 | 1.126 | 0.868 | 0.723 | **0.614** (78.54 ms) |
| Large | 8.75 | 1.817 | 1.525 | 1.318 | **1.210** (154.89 ms，独立进程) |

- 前向时延随 batch 增大显著下降（GEMM 利用率提升）；B=1 主要受 kernel
  launch/小矩阵效率限制。
- 理论核对：按实测有效吞吐约 150 TFLOPS，Large 200 GFLOPs ≈ 1.33 ms/
  sample，实测 1.21 ms；TM 100 G ≈ 0.67 ms，实测 0.614 ms；TS 40 G ≈ 0.27，
  实测 0.323；Medium 15 G ≈ 0.10，实测 0.111；Small 5 G ≈ 0.033，实测 0.054。
- 这些是**模型单次前向**，不含数据加载、通信、UAV 编码或闭环控制；不能直接
  当作导航决策的 E2E 时延。

## 5. 表 4：训练时间（实测 fwd+bwd + 数据开销）

| 模型 | 推荐 B | steps/epoch | fwd+bwd ms/step | 计算 h/epoch | 数据 h/epoch | 每 epoch | 20 epochs | 15 epochs 早停 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small | 128 | 10,000 | 12.91 | 0.036 | 0.26 | 0.296 | **5.92 h** | 4.44 h |
| Medium | 128 | 10,000 | 31.06 | 0.086 | 0.48 | 0.566 | **11.33 h** | 8.49 h |
| Transformer-S | 128 | 10,000 | 131.68 | 0.366 | 0.47 | 0.836 | **16.72 h** | 12.54 h |
| Transformer-M | 128 | 10,000 | 248.15 | 0.689 | 0.45 | 1.139 | **22.79 h** | 17.09 h |
| Large | 64 | 19,999 | 250.55 | 1.392 | 0.47 | 1.862 | **37.24 h** | 27.93 h |
| **合计** | — | — | — | — | — | — | **93.95 h（3.9 天）** | **70.49 h（2.9 天）** |

- 数据 h/epoch 沿用 Stage 11 的经验值（Small 0.26、Medium 0.48、TS 0.47、
  TM 0.45、Large 0.47），并与计算时间**相加**（保守口径）。
- 若 DataLoader 与 GPU 计算重叠，计算-only 总时间约
  **51.4 h（20 ep）/ 38.6 h（15 ep）**；实际预计在两者之间，
  约 **52–94 h（20 ep）/ 39–70 h（15 ep）**。
- 早停按 15 epochs 估算；Stage 11 的 best epoch 为 1–12，patience=10，
  实际可能 10–18 个 epoch，时间相应在 39–84 h 之间。
- **与 R2 对比**：R2 的 196 h 偏保守约 2 倍（Small 7.3→5.9、Medium
  16.0→11.3、TS 26.3→16.7、TM 51.6→22.8、Large 94.5→37.2）。
  主要原因是实测 BF16 有效吞吐约 150 TFLOPS，而不是 R2 假设的 50 TFLOPS；
  数据开销在大型模型上占比更小。

## 6. 重点检查：Medium 的 36,864 维 per-token MLP

| 项 | 值 |
|---|---|
| Token MLP 参数 | 2048×36,864 + 36,864 + 36,864×2,048 + 2,048 ≈ **151.0 M** |
| 每 token 计算量 | 4×2048×36,864 ≈ **302 MFLOPs** |
| 49 token 总计算量 | 49 × 302 M ≈ **14.80 GFLOPs** |
| 中间激活（B=128） | 128×49×36,864×2 B ≈ **462 MiB**（单张量） |
| B=128 前向峰值 | **3.21 GiB**（含整个模型） |
| B=128 fwd+bwd 峰值 | **2.60 GiB** |
| B=128 训练峰值 | **4.39 GiB**（含 AdamW） |
| B=128 前向/步 | 14.23 ms（0.111 ms/sample） |
| B=128 fwd+bwd/步 | 31.06 ms（0.243 ms/sample） |
| 是否瓶颈 | **否**：显存余量约 27 GiB；36,864 宽 GEMM 效率高 |

结论：**Medium 的 36,864 维 per-token MLP 是合理的**。它的参数和激活都在
32 GB 范围内，吞吐也正常。若未来想进一步降低激活，可改成 3 层
`2048→8192→13312→2048`（FLOPs 约 153M MACs/token，接近原 151M；
峰值中间激活约 8192/13312 而非 36864），但本阶段不需要改。

## 7. 重点检查：Large 的 1.03B 参数

| 项 | 值 |
|---|---|
| 参数量 | 1,034.7 M |
| BF16 权重 | 2.02 GiB |
| FP32 权重（标准 AMP） | 4.04 GiB |
| FP32 梯度 | 4.04 GiB |
| AdamW FP32 m/v | 8.09 GiB |
| 标准 AMP 总状态 | 16.17 GiB |
| B=64 fwd+bwd 峰值 | 15.83 GiB |
| B=64 训练峰值 | **24.04 GiB**（推荐，余量约 7.8 GiB） |
| B=128 fwd+bwd 峰值 | 24.61 GiB |
| B=128 训练峰值 | **33.19 GiB**（不推荐；超过 32 GB） |
| B=128 前向 | 154.9 ms（1.21 ms/sample，7.28 GiB） |
| B=64 fwd+bwd | 250.6 ms/step |
| B=64 训练时间 | 20 ep ≈ 37.2 h；15 ep ≈ 27.9 h |

结论：**Large 可行，但必须用 B=64 + 梯度累积 2**。B=128 在标准 AMP +
AdamW 下估算 33.2 GiB，超过 32 GB；不要为了维持 B=128 而改用纯 BF16
无 master 或 8-bit Adam 等非默认方案。若一定需要 B=128，可启用 gradient
checkpointing（激活显著下降），但本轮推荐 B=64。

## 8. 九个问题的回答

1. **RTX 5090 32 GB 能否训练全部五种？** 能，顺序训练，使用推荐 batch。
2. **哪些可以用 B=128？** Small、Medium、Transformer-S、Transformer-M。
3. **哪些必须降 batch？** Large：B=64（+累积 2，等效 128）。
4. **哪些需要 Gradient Checkpointing？** 推荐配置下都不需要；只有强制
   Large B=128 时才需要。
5. **Medium 的 36,864 MLP 是否合理？** 合理；166M 参数、4.39 GiB 训练峰值、
   31 ms/step（B=128），不是瓶颈。
6. **TM / Large 显存是否可接受？** TM B=128 19.19 GiB，可接受；
   Large B=64 24.04 GiB，可接受；Large B=128 33.19 GiB，不可接受。
7. **真实单次前向时延？** 见表 3：B=128 时 Small 0.054、Medium 0.111、
   TS 0.323、TM 0.614、Large 1.210 ms/sample。
8. **20 epoch / early stop 总时间？** 保守（数据相加）约 **94 h / 70 h**；
   计算-only 约 **51 h / 39 h**；实际预计 **52–94 h / 39–70 h**。
9. **哪些模型需要调整预设？** 架构不需要调整；只有 Large 的训练 batch
   需要从 128 改为 64 + 累积 2。Medium 36,864 无需改动。

## 9. 并行训练

- 单卡 RTX 5090；顺序训练最稳。
- Small / Medium / TS 理论上可同时放入显存（约 2.05+4.39+10.29 ≈
  16.7 GiB），但会争用 GPU 算力，总墙钟不一定缩短；不建议在 Large/TM
  训练时并行其他模型。
- Large 单独训练（B=64）占用约 24 GiB，剩余显存不足以再放 TM（19 GiB）。

## 10. 产物

```
hardware_feasibility/
├── HARDWARE_FEASIBILITY_REPORT.md
├── results/
│   ├── benchmark_results.csv
│   ├── memory_estimation.csv
│   ├── batch_size_comparison.csv
│   ├── forward_latency_estimation.csv
│   └── training_time_estimation.csv
├── figures/
│   ├── gpu_memory_comparison.png
│   ├── batch_size_feasibility.png
│   ├── forward_latency_comparison.png
│   └── training_time_comparison.png
└── scripts/
    ├── benchmark_memory_latency.py
    └── analyze_hardware.py
```

本阶段未训练、未加载数据/Checkpoint、未修改模型或 Stage 12 结果。
等待你确认推荐 batch 与训练计划后再启动正式训练。
