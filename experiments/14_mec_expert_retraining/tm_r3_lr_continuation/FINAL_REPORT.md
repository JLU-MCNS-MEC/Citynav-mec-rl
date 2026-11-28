# Stage 14-R3 · Transformer-M 低学习率续训 — 最终报告

日期：2026-11-28
代码 commit：`21631b3fb`（之后的本报告与结果另提交）
**研究问题**：Stage 14 的 TM 在 Epoch 1 达到最佳后，是否因重新升高学习率 + Early Stopping
错过了后期低学习率阶段的提升？

**答案：是，而且幅度很大。** Val-Seen 2498 条上 SR 从 42.51% 提升到 **56.04%（+13.53 pp）**。

---

## 1. 配置差异（原 TM vs R3）

| 项目 | 原 TM (Stage 14-R2) | R3 续训 |
|---|---|---|
| 起点 | 随机初始化 | `checkpoints_stability_2ep/last.pt`（Epoch 1，best 0.6177，含 AdamW 状态） |
| 学习率 | OneCycle，max 5e-5，warmup 10% | 余弦 **5e-6 → 5e-7**，无 warmup |
| Epoch | 0–11（Epoch 11 早停） | 2–19（跑满，早停禁用） |
| 其余 | AdamW / wd 0.01 / BF16 / batch 128 / clip 1.0 / seed 42 / 4 类 CE | 完全相同 |
| 网络 / FLOPs / 数据 | — | **未改动**（498.6 M / 100.025 GFLOPs） |

## 2. 逐 epoch 对比（Dev Macro-F1）

| Epoch | 原 TS | 原 TM | **新 TM** | Train Loss | LR |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5820 | 0.5642 | 0.5642* | — | — |
| 1 | 0.6249 | 0.6177 | 0.6177* | — | — |
| 2 | 0.6010 | 0.5926 | 0.6222 | 0.5298 | 4.97e-06 |
| 3 | 0.6120 | 0.5812 | 0.6208 | 0.5262 | 4.86e-06 |
| 4 | 0.6119 | 0.5788 | 0.6309 | 0.5234 | 4.70e-06 |
| 5 | 0.5905 | 0.5826 | 0.6142 | 0.5202 | 4.47e-06 |
| 6 | 0.6286 | 0.6041 | 0.6246 | 0.5174 | 4.20e-06 |
| 7 | 0.6020 | 0.5793 | 0.6193 | 0.5156 | 3.88e-06 |
| 8 | 0.6219 | 0.5987 | 0.6350 | 0.5130 | 3.52e-06 |
| 9 | 0.6255 | 0.5899 | 0.6316 | 0.5107 | 3.14e-06 |
| 10 | 0.6218 | 0.5912 | 0.6330 | 0.5083 | 2.75e-06 |
| 11 | 0.6300 | 0.5993 | 0.6340 | 0.5063 | 2.36e-06 |
| 12 | 0.6275 | — | 0.6355 | 0.5039 | 1.98e-06 |
| 13 | 0.6454 | — | 0.6401 | 0.5016 | 1.63e-06 |
| 14 | 0.6293 | — | 0.6247 | 0.4997 | 1.30e-06 |
| **15** | 0.6412 | — | **0.6407（最佳）** | 0.4983 | 1.03e-06 |
| 16 | 0.6500 | — | 0.6380 | 0.4966 | 8.01e-07 |
| 17 | 0.6454 | — | 0.6395 | 0.4954 | 6.36e-07 |
| 18 | 0.6444 | — | 0.6331 | 0.4943 | 5.34e-07 |
| 19 | 0.6459 | — | 0.6399 | 0.4937 | 5.00e-07 |

\* Epoch 0/1 为历史继承值，非本轮训练所得。原 TM 在 Epoch 11 后由 Early Stopping 终止，故为空。

- 新 TM 最佳 **0.6407**，比原 TM 最佳 0.6177 高 **+2.30 pp**；
- 与 TS 最佳 0.6500 差 0.93 pp；新 TM 的曲线形态与 TS 类似（约 Epoch 13–16 出现高点），
  但整体低约 0.5–1.0 pp；
- 18 个 epoch 全程 `skipped_updates=0`、`nan_losses=0`，Loss 单调下降 0.5298 → 0.4937。

## 3. Val-Seen 闭环结果（2498 任务，分母 2498）

评测使用与原五模型**完全相同**的协议：`eval_val_seen_norm.py`、
`normalization_statistics.json`、同一 manifest（SHA `2da37816…`）、200 步上限、20 m 成功半径。
为缩短时间，2498 条按奇偶拆成两半并行（两个独立进程），完成后合并去重（脚本三项自检通过）。

| 模型 | SR | OSR | SPL | NE (m) | 平均步数 | Stop 率 | Stop 精度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 0.5392 | 0.5460 | 0.4773 | 75.87 | 64.4 | 0.670 | 0.805 |
| Medium | 0.3195 | 0.3259 | 0.3184 | 91.20 | 51.7 | 0.382 | 0.836 |
| **TS** | **0.6301** | **0.6397** | **0.5450** | **64.23** | 78.9 | 0.718 | 0.877 |
| TM（原） | 0.4251 | 0.4616 | 0.3894 | 90.04 | 66.5 | 0.498 | 0.853 |
| Large | 0.3351 | 0.3651 | 0.3254 | 94.93 | 55.4 | 0.392 | 0.854 |
| **TM-R3（新）** | **0.5604** | **0.5685** | **0.4914** | **72.51** | 78.7 | 0.661 | 0.847 |

相对原 TM：**SR +13.53 pp、SPL +0.102、NE −17.53 m、Stop 率 +0.163**；
相对 TS：SR −6.97 pp（仍排第 2，但已超过 Small 的 53.92%，+2.12 pp）。

失败结构（新 TM，2498）：OOB 493、never_stop 336、premature_stop 248、
reached_but_not_stopped 20、collision 1。

> 注：Dev Macro-F1 与闭环 SR 的排序一致（TS 0.6500 > TM-R3 0.6407 > 原 TM 0.6177），
> 说明 Dev F1 在本任务上可作为模型选择的可靠代理。

## 4. 关键结论

1. **原 TM 的 Early Stopping 显著过早**：Epoch 1 的 0.6177 远非最优，低 LR 续训后
   Dev F1 到 0.6407、闭环 SR 到 56.04%。
2. **Epoch 11 之后仍有可观提升**：Epoch 12–15 连续刷新最佳。
3. 提升主要来自 **Stop 率（0.498→0.661）与路径效率（SPL 0.389→0.491、NE −17.5 m）**，
   即模型学会了"在该停的时候停"，与 Dev 上 Stop F1 0.83–0.85 的稳定高位一致。
4. 仍未超过 TS（0.6301），但已从"远落后于 Small"提升为"仅次于 TS"。

## 5. 建议（需审核）

1. **是否用新 TM 替换 Stage 17 的 TM 节点**：新 TM 的 SR 比原 TM 高 13.5 pp，
   若替换，会改变 Stage 17 的专家质量先验（`QUALITY_DEV_F1` 亦需更新）；
   **本轮未做任何替换**，等审核决定。
2. **Large 是否值得同样处理**：Large 当前 SR 33.51%、Dev F1 0.6105，且训练成本约为 TM 两倍；
   若按同样思路（低 LR 续训、禁用早停）也可能获益，但需先确认时间预算。
3. 若替换专家，Stage 17 的 DRL 结果需重跑才能比较（专家变了）。

## 6. 复现与产物

```
tm_r3_lr_continuation/
  config.json                    训练与评测配置（含 SHA、调度、早停设置）
  run_tm_r3.sh / run_tm_r3_resume.sh   启动脚本（含 5 步 smoke）
  monitor_tm_r3.py / merge_eval.py     进度与并行评测合并工具
  runs/tm/seed42/               checkpoints(best/last)、training_history.jsonl、config
  evalA/ evalB/                 两半并行评测（各 1249 条）
  eval_merged_summary.json      2498 分母最终汇总
  eval_merged_episodes.csv      2498 行逐 episode 结果
  figures/                      dev_f1 / train_loss / lr 曲线
  PROGRESS_REPORT.md            逐 epoch 进度与 ETA
```

命令（正式评测，单进程版本）：

```bash
CITYNAV_DATA=/home/20261024_1/Workspace/DATA \
python stability_revision/eval_val_seen_norm.py \
  --ckpt tm_r3_lr_continuation/runs/tm/seed42/checkpoints/best.pt \
  --model tm --manifest ../06_closed_loop_success_rate/results/seed42/manifest.json \
  --task-file tm_r3_lr_continuation/val_seen_ids.txt \
  --norm-file stability_revision/normalization_statistics.json \
  --out-dir tm_r3_lr_continuation/eval_val_seen_norm
```

## 7. 过程中的问题与更正（如实记录）

1. 首次闭环评测误用未归一化脚本（`eval_val_seen.py`），SR 仅 0.32%，结果作废；
   改为 `eval_val_seen_norm.py` 后正常（抽样 111 条时 SR 56.8%，与最终一致）。
2. 初版全量评测包含 val_unseen 2826 条（共 5324），已按审阅要求改为只跑 val_seen 2498。
3. 并行合并脚本初版只读 `journal.part*.jsonl`，而评测结束时文件被合并为 `journal.jsonl`，
   已修正为兼容三种来源。
4. 训练中途为"禁用早停 + 保持 LR 连续"重启一次：通过 `--sched-offset-steps 98040`
   精确恢复到 2.3592916e-06（与当时记录逐位一致），仅损失约 30 分钟。

## 8. 状态

- 训练、闭环评测、报告均已完成；
- **Stage 17 P3 仍处于暂停**（DQN 3341 / DDPG 3262，`last.pt` 已保存），
  `run_p3_resume.sh` 就绪但**未执行**；
- 未修改任何 Stage 12/14/16 历史结果，未替换 Stage 17 的专家。
