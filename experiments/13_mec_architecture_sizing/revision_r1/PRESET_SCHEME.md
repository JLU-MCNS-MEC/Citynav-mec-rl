# Stage 13-R2 预设方案：104-token 统一 + 接近 5/15/40/100/200 GFLOPs

日期：2026-11-17　状态：**待确认，尚未修改模型或训练**

本方案基于 R1 的 104-token 契约，不要求权重继承（全部从头训练）。目标值
允许偏高或偏低，优先选择可训练、参数规模合理的架构。

## 1. 预设架构表

| 模型 | 架构 | hidden | layers | heads | FFN / 关键宽度 | 预测 GFLOPs | 目标 | 偏差 | 参数量 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| Small | MLP + per-token MLP | 512 | — | — | fusion H=2,048；token MLP 2048→12,288→2048 | **4.95** | 5 | **−0.9%** | 61.1 M |
| Medium | goal cross-attention + per-token MLP | 512 | — | 8 | F_gc=4,096；fusion H=2,048；token MLP 2048→36,864→2048 | **14.98** | 15 | **−0.1%** | 166.0 M |
| Transformer-S | 104-token Transformer | 1,408 | 8 | 8 | FFN=5,376 | **39.67** | 40 | **−0.8%** | 211.1 M |
| Transformer-M | 104-token Transformer | 2,048 | 10 | 16 | FFN=7,424 | **100.03** | 100 | **+0.03%** | 498.6 M |
| Large | 104-token Transformer | 2,560 | 12 | 16 | FFN=10,240 | **199.72** | 200 | **−0.14%** | 1,034.7 M |

Small / Medium 保持“非 Transformer”结构：Small 仍是 mean-pool + late-fusion
MLP；Medium 仍是 1 个 goal query 对 49 个 current token 的 cross-attention。
新增的 per-token visual MLP 只作用于 49 个 current token 的特征（在 mean-pool /
cross-attention 之前），不改变输入模态语义，也不是 Transformer encoder。

> 为什么 Small/Medium 要加 per-token MLP：如果只改 hidden 或 fusion 宽度，
> 5/15 GFLOPs 需要 Small hidden≈12.4k / head≈1.22M（约 2.5B 参数）、
> Medium hidden≈7.4k / gc_FFN≈14.6M（约 1.25B+ 参数），不可训练。per-token
> MLP 是保持 MLP/cross-attention 语义、且参数量可接受的最小扩算力方式。

## 2. 预计训练时间（单卡 RTX 5090，20 epochs 上限）

假设：11A 教师池 1,279,882 states；batch 128（约 10k steps/epoch）；
bf16/fp16 混合精度；有效吞吐 30–70 TFLOPS（中心值 50）；
数据加载/CPU 开销沿用 Stage 11 经验（Small≈0.26 h/epoch，Medium≈0.48，
TS≈0.47，TM≈0.45，Large≈0.47）。

| 模型 | 每 epoch 计算 | +数据 | 20 epochs 中心值 | 20 epochs 范围 |
|---|---:|---:|---:|---:|
| Small | 0.11 h | 0.37 h | **~7.3 h** | 6.7–8.7 h |
| Medium | 0.32 h | 0.80 h | **~16.0 h** | 14.2–20.2 h |
| Transformer-S | 0.85 h | 1.32 h | **~26.3 h** | 21.5–37.6 h |
| Transformer-M | 2.13 h | 2.58 h | **~51.6 h** | 39.4–80.0 h |
| Large | 4.25 h | 4.72 h | **~94.5 h** | 70.1–151.1 h |
| **合计** | — | — | **~196 h（约 8.2 天）** | 152–298 h |

- 早期停止（patience 10，max 20 epochs）：Stage 11 的 best epoch 为
  1–12，预计实际 10–18 个 epoch，训练时间可减少约 **20%–40%**；
  合计约 **120–160 h（5–7 天）**。
- TM/Large 建议开启 gradient checkpointing，必要时 batch 64 + 梯度累积；
  会再增加约 20%–30% 时间，但可避免 32 GB 显存 OOM。
- 该估计是“单卡顺序训练”；若并行训练多个模型，单卡显存/吞吐会互相争用，
  总墙钟不一定缩短。

## 3. 代码修改要点（确认后执行）

- **Transformer-S / Large**：history 改为 16 个独立 token（89→104）；
  新增 `hist_time`（16×h）与统一 `type_emb`（7×h）；mask 前缀 53→68；
  hidden/layers/FFN 按上表设置。
- **Transformer-M**：已是 104-token；仅改 D=2048、L=10、heads=16、FFN=7424。
- **Small**：新增 `current_token_mlp = Linear(2048,12288)+GELU+Linear(12288,2048)`，
  对 49 个 current token 逐个执行后再 mean-pool；D=512、fusion H=2048。
- **Medium**：新增 `current_token_mlp = Linear(2048,36864)+GELU+Linear(36864,2048)`，
  在 `cur_proj` 之前对 49 个 current token 执行；D=512、gc_FFN=4096、
  fusion H=2048。
- 五种模型均从头训练（不加载旧 Checkpoint，不做权重继承）；输入仍然是
  49×2048 current、512 target、16 步 history、12 state、8 landmark、
  28 relation，V5fix 几何/ mask / 归一化 / 四动作定义不变。
- 训练协议沿用 Stage 11：11A 教师池 18,521 eps / 1,279,882 states，
  batch 128，lr 3e-4，weight_decay 0.01，OneCycleLR，≤20 epochs，
  patience 10，Val-Seen Macro-F1 选 best（或按闭环 SR 选 best，需你确认）。

## 4. 需要你确认

1. 是否采用上表五个预设（Small/Medium 的 per-token MLP + 三个 Transformer
   的 104-token 尺寸）？
2. 训练 20 epochs + patience 10 早停，是否同意？
3. best checkpoint 的选择指标：Val Macro-F1（Stage 11 原协议）还是
   Dev 闭环 SR（更贴近导航目标）？
4. 确认后我再修改模型定义、启动训练；本轮不训练、不改 Checkpoint。
