# Stage 14 · Small-1G 实验最终报告

日期：2026-11-28
目的：把 Small 的 token MLP 从 `2048→12288→2048` 改为 `2048→2432→2048`（唯一改动），
验证 1/5 算力能否保持甚至超过 Small-5G 的性能。

**结论：参数量 61.1M → 20.7M、FLOPs 4.954G → 0.998G，Val-Seen 闭环 SR 从 53.92% 提升到 72.26%（+18.34 pp）。**

---

## 1. 模型与 FLOPs（真实前向，2×MACs）

| 模型 | 参数量 | 总 FLOPs | Token MLP 占比 | 10 TFLOPS 时延 |
|---|---:|---:|---:|---:|
| Small-5G | 61,121,028 | 4.954039 G | 4.9325 G（99.6%） | 0.4954 ms |
| **Small-1G** | **20,740,996** | **0.997762 G** | 0.9762 G（97.8%） | **0.0998 ms** |

计数器用同一份 hook 对原 SmallPreset 复现出 4.954039 GFLOPs（与 Stage 16 preflight 实测一致），
证实口径正确。

## 2. 训练（按用户指示在 Epoch 7 后停止）

协议与 Stage 14 Small 完全一致：lr 3e-4、wd 0.01、OneCycle warmup 0.03、bf16、
batch 128、clip 1.0、seed 42、随机初始化、Dev Macro-F1 选 best。
优化：DataLoader 增加 `pin_memory=True`、workers 8→16（数据吞吐 625 → 1,602 samples/s），
不影响数值结果。

| Epoch | Small-5G Dev F1 | **Small-1G Dev F1** | Δ | Loss | 耗时 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.4911 | **0.5019** | +1.08 pp | 0.6897 | 23.6 min |
| 1 | 0.5579 | 0.5579 | +0.01 pp | 0.6250 | 23.2 min |
| 2 | 0.5482 | 0.5568 | +0.87 pp | 0.6078 | 23.3 min |
| 3 | 0.5693 | 0.5712 | +0.19 pp | 0.5945 | 23.2 min |
| 4 | 0.5740 | 0.5788 | +0.48 pp | 0.5723 | 24.1 min |
| 5 | 0.5481 | 0.5504 | +0.23 pp | 0.5200 | 25.0 min |
| **6** | 0.5771（5G 最佳） | **0.5898（最佳）** | **+1.26 pp** | 0.4371 | 24.8 min |
| 7 | 0.5671 | 0.5780 | +1.09 pp | 0.3489 | 24.1 min |

前 8 个 epoch **每一轮都高于同 epoch 的 Small-5G**。用户指示在 Epoch 7 后停止，
采用 **best.pt = Epoch 6（Dev Macro-F1 0.5898）** 做闭环评测。

## 3. Val-Seen 闭环（2498 任务，分母 2498）

协议：`eval_val_seen.py`（Small 系不做输入归一化，与 Stage 14 Small-5G 的 eval_config 一致）、
同一 manifest（SHA `2da37816…`）、200 步、20 m。2498 条按奇偶拆两半并行（1249 + 1249），
用 `merge_eval.py` 合并去重。

| 模型 | FLOPs | **SR** | OSR | SPL | NE (m) | 平均步数 | Stop 率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 11 旧 Small | 0.00447 G | 0.3467 | — | — | — | — | — |
| Stage 14 Small-5G | 4.954 G | 0.5392 | 0.5460 | 0.4773 | 75.87 | 64.4 | 0.670 |
| **Small-1G（新）** | **0.998 G** | **0.7226** | **0.7326** | **0.6054** | **48.15** | 74.8 | **0.837** |
| （参考）Transformer-S | 39.67 G | 0.6301 | 0.6397 | 0.5450 | 64.23 | 78.9 | 0.718 |

失败结构（Small-1G）：OOB 189、never_stop 202、premature_stop 276、reached_but_not_stopped 25、collision 1。

**交叉校验（防口径错误）**：在同一 100 条 val_seen 上分别跑两个模型：

| 模型 | SR（100 条） | 与全量一致性 |
|---|---:|---|
| Small-5G | 0.570 | 与 53.92% 同量级 ✓ |
| Small-1G | 0.700 | 与 72.26% 同量级 ✓ |

两者在同任务上的 13 pp 差距与全量一致，说明差异不是评测脚本造成的。

## 4. 解读

1. **1/5 FLOPs 反而显著更强**：Dev F1 只高 1.3 pp，闭环 SR 却高 18.3 pp。这说明
   闭环收益不是来自"分类更准"，而是来自**泛化与停止行为**：
   - Stop 率 0.670 → **0.837**（更会在该停时停），Stop 精度 0.864；
   - NE 75.87 → **48.15 m**，SPL 0.477 → **0.605**（路径更短更直接）；
   - OOB 从 732 降到 **189**（失败模式大幅减少）。
2. 一种合理解释：`12288` 宽的 token MLP 容量过大，容易记住教师轨迹的分布细节，
   在闭环中表现为过度 Forward、错过 Stop；`2432` 宽反而起到正则化作用。
   **这是假设，需进一步实验（如 Dev F1 饱和下的闭环敏感性）验证。**
3. Dev Macro-F1 在本例中**严重低估**了闭环差异（+1.3 pp vs +18.3 pp），
   说明"用 Dev F1 选模型"对闭环 SR 的指导性有限——这一点对 Stage 17 的模型选择也有影响。
4. Small-1G（72.26%）已超过 Transformer-S（63.01%），若成立，则是 Stage 14 目前
   Val-Seen SR 最高的专家，且算力只有 TS 的 1/40。

## 5. 风险与待核实

| # | 问题 | 说明 |
|---|---|---|
| 1 | 结果超出预期 18 pp | 已用同任务交叉校验排除口径问题；仍建议独立复跑一次（第二个种子）确认 |
| 2 | 仅 1 个种子 | 模型选择与评测都是 seed 42 |
| 3 | Dev F1 与闭环脱节 | 若以闭环为目标，模型选择准则需要重新设计 |
| 4 | 训练只到 Epoch 7 | 按用户指示提前停止，后续 epoch 是否更优未知 |

## 6. 复现

```bash
CITYNAV_DATA=/home/20261024_1/Workspace/DATA \
python train_stage14.py --model small1g --workers 16 --out-root small_1g_experiment/runs

python eval_val_seen.py --ckpt small_1g_experiment/runs/small1g/seed42/best.pt \
  --model small1g --manifest ../06_closed_loop_success_rate/results/seed42/manifest.json \
  --task-file tm_r3_lr_continuation/val_seen_ids.txt \
  --out-dir small_1g_experiment/eval_full
```

产物：`config.json`、`FLOPS_AUDIT.md`、`flops_audit.json`、`PROGRESS_REPORT.md`、
`eval_summary.json`（2498 分母）、`eval_episodes.csv`、`figures/`。

## 7. 状态

- 未替换 Stage 17 的任何专家，未启动新的 DRL 训练；
- Stage 17 P3 仍暂停（DQN 3341 / DDPG 3262，last.pt 已保存）。
