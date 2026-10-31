# 10A 报告：Point Query 与 Cross-Attention 视觉匹配对比

- 日期：2026-10-31
- 服务器：新服务器 `ubuntu`（RTX 5090 32 GB）
- 分支：`exp/citynav-mec-fixed50m-topdown`
- 目录：`experiments/10_goal_query_mec_training/10A_point_query_cross_attention/`

## 1. 目标

在保留 09C-1 连续目标定位（C1）框架的前提下，回答一个问题：
**把 C1 的"固定中心块余弦匹配"换成可学习的 Point Query（PQ）或
Cross-Attention（CA）匹配头，离线定位精度与置信度门控是否更好？**

固定条件（与用户约定一致）：
- 冻结 ResNet-50（IMAGENET1K_V2），只用 **layer2** 空间特征（512×28×28）；
- 不重新训练 ResNet-50；不引入 C2 硬门控；不改变 09C-1 连续信念更新主框架；
- 不使用 Test-Unseen 调参；只跑 seed 42（沿用项目"只训练 seed 42"约定）。

## 2. 公平对比协议

| 项目 | 内容 |
|---|---|
| 训练数据 | Train-Seen 教师状态（`teacher_train_seen.npz`），09A 距离桶分层采样 ≤12 状态/条，共 **193,635** 状态（in-view 93,419 / out-of-view 100,216，3,945 个去重目标）；**剔除**与 09A 校准完全一致的 2,000 状态校准子集（rng seed 20261030） |
| 校准数据 | 与 `calibrate_09a.py` 相同的 2,000 个 Train-Seen 状态 + 同图错目标负控制；tau 网格搜索最大化"10 m 内定位"F1 |
| 评测数据 | 与 09A 相同的 `09a_match_manifest.csv`：**58,891** 个状态（Stage-06 闭环轨迹采样，覆盖 5,324 条任务；Val-Seen 27,608 / Val-Unseen 31,283；in-view 22,280 / out-of-view 36,611） |
| 输入 | 三方案完全相同的输入：北向上 224px 观测图 + 北向上目标图，冻结 ResNet-50 layer2 特征（fp32） |
| 输出接口 | 三方案输出相同的 `(col, row, score, margin)`（北向上 224px 帧），可直接替换 `subpixel_match` |
| 门控 | C1 用冻结的历史校准 tau=0.875（09a_calibration.json，layer2）；PQ/CA 用各自 10A 校准 tau（同协议） |
| 评测指标 | MAE / Median / P90（in-view）、Precision@5/10/20m、Seen/Unseen、Short/Medium/Long、可见/不可见（fp_rate_out_of_view）、AUROC、参数量、推理耗时 |

## 3. 方法设计

### 3.1 C1（正式基线）
09C-1 的 `subpixel_match`：目标 layer2 中心 2×2 块描述子与观测 28×28 单元做
稠密余弦相似度，取峰值的 3×3 加权质心做亚像素细化。零训练参数。

### 3.2 PQ（Learnable Point Query）
- **12 个可学习 query token**（512 维），各自带可学习 2D anchor；
- query 由目标全局描述子的投影（goal_proj）条件化；
- query 与观测 28×28 单元（+正弦 2D 位置编码）做一次交叉注意力；
- 每个 query 用 MLP 头输出一个候选 `(col, row, score_logit)`；
- 取 score 最高候选；margin = 最高分 − 次高分。

### 3.3 CA（Cross-Attention）
- 目标侧 token：目标特征的 **3×3 分块均值描述子 + 全局描述子（10 个 token）**
  + 可学习位置嵌入；
- 目标 token 与观测单元（+位置编码）做交叉注意力；
- 随后 10 个 token 之间做自注意力互相修正；
- 每个 token 用 MLP 头输出候选 `(col, row, score_logit)`；取最高分候选。

### 3.4 训练损失（两方案相同）
- Smooth-L1 回归 `(col,row)`（像素，仅对 in-view 状态生效；out-of-view 偏移
  超出有界输出头，只训练置信度）；
- BCE 训练 score：标签 = "该候选预测误差 ≤ 10 m"（由模型自身预测计算，detach）；
- AdamW lr=1e-3，weight_decay=1e-4，batch 128，12 epoch，seed 42；
- 训练集不含校准子集 → 校准无污染。

## 4. 结果

### 4.1 参数量与推理耗时

| 方法 | 参数量 | 匹配耗时 (ms/状态) | 特征提取耗时 (ms/状态，三方案共享) |
|---|---|---|---|
| C1 | 0（无参数） | 2.409（CPU scipy） | 0.567 |
| PQ | 1,454,107 | 0.138（GPU） | 0.567 |
| CA | 2,240,515 | 0.063（GPU） | 0.567 |

### 4.2 离线定位误差（in-view 状态，58,891 状态）

| 方法 | MAE (m) | Median (m) | P90 (m) | Prec@5m | Prec@10m | Prec@20m |
|---|---|---|---|---|---|---|
| **C1** | **8.54** | **3.27** | **18.38** | **84.5%** | **87.8%** | **90.4%** |
| PQ | 28.65 | 26.66 | 51.08 | 3.6% | 12.2% | 34.2% |
| CA | 34.24 | 32.99 | 56.70 | 1.6% | 6.0% | 21.3% |

C1 的 10A 复测（median 3.27 m、P@10m 87.8%）与 09A 历史报告（layer2
中位 3.6 m、P@10m 0.878）一致，确认复测管线正确。

### 4.3 Seen / Unseen

| 方法 | Val-Seen median (m) | Val-Unseen median (m) | Val-Seen P@10m | Val-Unseen P@10m |
|---|---|---|---|---|
| **C1** | **3.31** | **3.23** | **87.3%** | **88.3%** |
| PQ | 23.93 | 29.39 | 15.8% | 8.6% |
| CA | 32.03 | 34.02 | 6.7% | 5.4% |

### 4.4 距离桶 / 长度组 / 可见性（MAE, m；in-view）

| 分组 | C1 | PQ | CA |
|---|---|---|---|
| 距离 0-20 m | 3.40 | 16.50 | 18.07 |
| 距离 20-40 m | 3.39 | 24.61 | 29.14 |
| 距离 40-60 m | 12.05 | 35.42 | 43.20 |
| 距离 60-80 m | 48.22 | 49.25 | 59.21 |
| Short | 5.53 | 26.49 | 31.32 |
| Medium | 10.10 | 30.48 | 36.09 |
| Long | 11.33 | 29.61 | 36.47 |
| 有效性 <0.5 | 8.75 | 26.87 | 35.65 |
| 有效性 ≥0.5 | 8.52 | 28.77 | 34.14 |

### 4.5 置信度门控（Train-Seen 校准 2,000 状态 + 负控制）

| 方法 | tau | 校准 F1@10m | 校准 P@10m | 负控制误接受率 | 评测视野外误接受率 (fp_oov) | AUROC（评测 in-view） |
|---|---|---|---|---|---|---|
| C1 | 0.875（历史冻结） | —（09A 冻结） | — | — | **4.3%** | **0.924** |
| PQ | 0.23 | 0.36 | 25.5% | 9.3% | 7.8% | 0.645 |
| CA | 0.09 | 0.17 | 13.5% | 44.4% | 37.6% | 0.599 |

CA 的 tau 被校准到极低（0.09），导致 37.6% 的视野外状态通过门控
（fp_oov），门控几乎失效；PQ 的 fp_oov（7.8%）尚可但正检率极低。

### 4.6 可视化

- `figures/10a_error_cdf.png` — 误差 CDF（Seen/Unseen）
- `figures/10a_metrics_bars.png` — 指标对比柱状图
- `figures/10a_breakdown.png` — 距离桶/长度组分解
- `figures/10a_score_calibration.png` — 分数-误差散点与 tau
- `figures/10a_examples.png` — 三方案典型定位案例（val_unseen）
- `figures/10a_attention.png` — PQ query anchor / CA 注意力可视化

## 5. 诊断（为什么 PQ/CA 差这么多）

1. **不是崩溃性 bug**：诊断运行（400 个 in-view 训练状态）显示两者均未
   坍缩到"预测画面中心"（仅 9-11% 预测落在中心 20px 内），PQ 预测与真值的
   相关系数 0.67/0.73 —— 模型学到了**粗方向**，但无法做**精确定位**。
2. **注意力不够锐利 + 纯回归头精度上限**：PQ/CA 用 MLP 从交叉注意力 token
   直接回归绝对像素坐标；C1 的余弦相关 + 亚像素质心在 28×28 网格上有极强的
   定位归纳偏置。学习到的注意力分布弥散，回归只能到"十几米"量级。
3. **训练未收敛**：12 epoch 结束时 train loss 仍在单调下降
   （PQ 38.0→25.8，P@10 0.07→0.27 仍在上升），模型处于欠拟合状态。
4. **PQ 的目标条件化信息不足**：PQ 只用目标全局均值描述子；C1 用的是目标
   中心块（真正的目标区域）描述子。CA 虽然用了 3×3 分块，但位置嵌入与
   "哪块是中心"脱钩，且结果更差。
5. **结论**：本实现/预算下（12 epoch、单层注意力、绝对回归头），PQ/CA
   离线定位远逊于 C1。这不是"接近但略差"，而是数量级差距
   （median 3.3 m vs 27-33 m）。

## 6. 结论与推荐

**10A 结论：PQ 与 CA（本轮实现与训练预算）显著劣于 C1，不满足替换条件。
推荐 C1（09C-1 固定中心块余弦匹配 + 亚像素质心）作为 10B 的正式方案。**

依据：
- 离线定位 median 误差：C1 3.27 m ≪ PQ 26.7 m ≪ CA 33.0 m；
- 门控质量：C1 F1@10 0.945 / AUROC 0.924，PQ 0.253/0.645，CA 0.141/0.599；
- CA 校准 tau 0.09 使 37.6% 视野外状态误通过；
- 推理耗时：PQ/CA 虽快 17-38 倍（0.06-0.14 ms vs 2.41 ms/状态），但在
  0.57 ms 的特征提取面前无意义（总耗时 0.63-0.71 ms vs 2.98 ms）。

**关于 v2 改进轮**：诊断指出三条可修复的设计缺陷（① 位置头改为注意力
soft-argmax + 残差，替代绝对回归；② PQ 目标条件化改用中心块描述子；
③ 训练延长至 loss 收敛）。按约定，任何新训练轮需用户明确同意后才会启动；
本轮（12 epoch）为获批协议内的唯一一轮，结果如实归档。

## 7. 复现

```bash
export CITYNAV_DATA=/mnt/hdd8t/Citynav-MEC/DATA
cd experiments/10_goal_query_mec_training/10A_point_query_cross_attention
# 1) 训练缓存（已完成：results/cache -> NVMe 28 GB）
python -I scripts/build_cache10a.py
# 2) 训练（seed 42；checkpoint 每 epoch 原子保存，--resume 断点续跑）
python -I scripts/train10a.py --method pq
python -I scripts/train10a.py --method ca
# 3) 校准（Train-Seen 2,000 状态子集，与 09A 完全一致）
python -I scripts/calibrate10a.py --method pq
python -I scripts/calibrate10a.py --method ca
# 4) 评测（58,891 状态，三方案同批同输入；断点续跑）
python -I scripts/eval10a.py
# 5) 图表
python -I scripts/make_figures10a.py
```

## 8. 运行记录

- 训练：PQ 4,897 s（82 min，best epoch 11 loss 25.82）；CA 4,907 s（82 min，
  best epoch 11 loss 34.17）；两方法并行（GPU 55-68% 利用率，显存 ≤8 GB）
- 校准：pq/ca 各 28.5 s
- 评测：三方案同批 58,891 状态；特征 0.567 ms/状态；匹配耗时
  C1 2.409 / PQ 0.138 / CA 0.063 ms/状态；总评测 ~7 min（含 OOM 修复后重续）
- 异常记录：① HDD 随机读瓶颈 → 训练缓存迁移 NVMe（28 GB）；② 评测第一轮
  因跨 batch 缓存 5,000 个 goal 特征（RAM+VRAM 各 ~8 GB）被系统 OOM 杀掉
  → 改为 batch 内去重，截断 gzip 用 zlib 宽容模式抢救 37,056 行后断点续跑；
  ③ torch.load(weights_only=True) 拒绝含 numpy 标量的 best.pt → 加载器回退修复
