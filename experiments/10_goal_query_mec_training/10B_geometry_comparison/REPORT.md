# 10B 报告：几何控制器闭环比较（C1 vs CA）

- 日期：2026-10-31
- 服务器：新服务器 `ubuntu`（RTX 5090 32 GB）
- 分支：`exp/citynav-mec-fixed50m-topdown`
- 目录：`experiments/10_goal_query_mec_training/10B_geometry_comparison/`

## 1. 范围（按用户决定）

- **C1**：直接复用 09C-1 已提交的 5,324 条 journal（零重跑，历史结果不动）；
- **CA**：10A v2 训练的最优 checkpoint，通过 10B harness 跑全量闭环作证；
- **PQ**：跳过（10A 中因 MSE 监督塌陷不可用，修复需用户同意后另起一轮）。

## 2. 协议

| 项目 | 内容 |
|---|---|
| 任务 | Stage-06 manifest 5,324 条（Val-Seen 2,498 + Val-Unseen 2,826） |
| 闭环环境 | `run_10b.py`：与 `continuous_env.run_episode`（c1 分支）**逐行一致的复刻**，唯一替换是匹配模块调用（CA: 10A CrossAttnMatcher，tau=0.13） |
| 不变部分 | 24 m 候选先验、ContinuousBelief Kalman 更新（09C 校准的 sigma_vis_m/eta/rho 冻结）、agree_m=10 m 两帧门、几何控制（STOP 20 m / align 25° / fwd 5 m / turn 30°）、步数上限 200 |
| 不使用 | C2 硬门控；Test-Unseen |
| Harness 验证 | `--mode c1 --limit 30` 与 09c_c1 journal **30/30 条完全一致**（success/终点距/步数/更新次数），证明环境复刻无偏差 |
| 断点续跑 | journal 按 manifest SHA + mode 键控，原子保存 |

## 3. 结果

### 3.1 主指标（5,324 条，成功率 + Wilson 95% CI）

| 指标 | C1（09C-1） | CA（10B） |
|---|---|---|
| **SR** | **64.7%** [63.4, 66.0] | 26.0% [24.9, 27.3] |
| OSR | 64.7% | 26.2% |
| Val-Seen SR | 65.5% [63.6, 67.3] | 30.5% [28.7, 32.3] |
| Val-Unseen SR | 64.0% [62.2, 65.8] | 22.1% [20.6, 23.7] |
| Short / Medium / Long | 70.5% / 64.0% / 60.0% | 45.1% / 21.3% / 13.0% |
| 过早 STOP | 35.0% | **73.9%** |
| 越界率 | 0.30% | 0.04% |
| 平均步数 | 39.9 | 24.9 |
| 平均终点误差 | 30.4 m | **109.4 m** |
| 动作分布 (S/F/L/R) | 2.5/91.6/3.0/3.0% | 4.0/85.0/5.5/5.4% |

### 3.2 配对比较（McNemar，同一 5,324 条任务）

| 对比 | n | 参考 SR | 对比 SR | Δ (pp) | McNemar p | CA 独胜 | C1 独胜 |
|---|---|---|---|---|---|---|---|
| CA vs C1 | 5,324 | 64.7% | 26.0% | **−38.67** | **≈0（<1e-300）** | 223 | 2,282 |

### 3.3 可视化

- `figures/10b_sr_bars.png` — SR 柱状图（含 CI）
- `figures/10b_breakdown.png` — Seen/Unseen 与 Short/Medium/Long 分解
- `figures/10b_trajectories.png` — CA 闭环轨迹示例（3 成功 + 3 失败）

视频：本阶段未生成 MP4——比较结果差异极大（Δ −38.7 pp，p≈0），轨迹图已
足够说明结论；如后续需要可补。

## 4. 失败机制诊断

CA 离线 in-view 定位（median 3.74 m）接近 C1（3.27 m），闭环却崩塌
（−38.7 pp）。原因链：

1. **过早 STOP 73.9%**（C1 35.0%）：CA 的 softmax 峰值分数即使匹配质量一般
   也接近 1.0，tau=0.13 门控几乎总通过 → 信念被大量低质量测量更新 →
   错误模式快速坍缩 → 在错误位置自信地 STOP（平均终点误差 109.4 m）。
2. **视野外误接受 23.0%**（C1 4.3%）：目标在视野外时 CA 峰值仍高，
   把信念拉向错误位置；C1 的余弦峰值天然压低视野外响应。
3. 离线评测只看 in-view 误差，掩盖了分数校准与视野外行为的差距；
   **闭环才是视觉匹配方案的最终裁判**（与实验设计预期一致）。

## 5. 结论

**CA 未能在闭环中超过 C1，且大幅落后（SR 26.0% vs 64.7%，McNemar p≈0）。
按既定原则（PQ/CA 无显著优势则保留 C1、不强行上复杂结构），
最终视觉匹配方案 = C1（09C-1 固定中心块余弦匹配 + 亚像素质心）。**

10C 的"目标相关视觉特征"将基于 C1 的匹配特征设计；MEC 输入模态不含
PQ/CA 组件。PQ/CA 作为负结果归档（10A v2 + 本报告）。

## 6. 复现

```bash
export CITYNAV_DATA=/mnt/hdd8t/Citynav-MEC/DATA
cd experiments/10_goal_query_mec_training/10B_geometry_comparison
# CA 闭环（两个 split 可并行；断点续跑）
python -I scripts/run_10b.py --mode ca --task-file /tmp/10b_val_seen_ids.txt \
  --out-dir results/10b_ca_seen
python -I scripts/run_10b.py --mode ca --task-file /tmp/10b_val_unseen_ids.txt \
  --out-dir results/10b_ca_unseen
# 合并 + 分析
cat results/10b_ca_seen/journal.jsonl results/10b_ca_unseen/journal.jsonl \
  > results/10b_ca/journal.jsonl
python -I scripts/analyze_10b.py
```

## 7. 运行记录

- CA 闭环：5,324/5,324 条，双进程并行各 ~4.7 分钟（11.5-13.4 ep/s，RTX 5090）
- Harness 验证：c1 模式 30 条，与 09c_c1 逐条一致（30/30）
- 异常：run_10b.py 两处路径 bug（仓库根层级、10A 校准/checkpoint 路径）已修复
