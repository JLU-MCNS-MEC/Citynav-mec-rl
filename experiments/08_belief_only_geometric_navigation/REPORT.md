# Stage 08：全局目标信念的几何导航基线

- 状态：**完成**
- 分支：`exp/citynav-mec-fixed50m-topdown`
- 任务：与 Stage 06 **完全相同的 5,324 条**（Val-Seen 2,498 + Val-Unseen 2,826），
  同一 manifest、同一起点/目标、同一障碍与边界、200 步上限、20 m 成功半径
- 回答的问题：**不用实时局部 RGB、不用历史、不用 Transformer，只靠 12 维目标信念的
  简单几何控制器，能达到什么 SR？**

## 1. 主结果（配对任务，同一评估协议）

| 方法 | Val-Seen SR | Val-Unseen SR | **总 SR** | 95% CI | OSR | STOP 率 | NE 中位 | 步数均值 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Transformer-M（Stage 06，复用） | 24.9% | 22.5% | **23.7%** | [22.5, 24.8] | 29.6% | 51.2% | 37.7 m | 50.0 |
| **Top-1 Geometry** | 47.6% | 42.4% | **44.9%** | [43.5, 46.2] | 46.0% | 99.7% | 21.4 m | 40.9 |
| **Top-5 Weighted Geometry** | 37.5% | 42.6% | **40.2%** | [38.9, 41.5] | 76.3% | 45.1% | 13.4 m | 129.6 |

**阶段判定：A —— Geometry 明显高于 Transformer-M。** 在完全相同的输入协议下，
一个只看 12 维信念、25° 对齐阈值、20 m 估计距离 STOP 的控制器，SR 比 Transformer-M
高 **21.2 个百分点**（配对差值 95% CI [19.7, 22.6]，McNemar p ≈ 4.6e-167）。

### 1.1 逐 episode 配对

| 比较 | n | 两者成功 | 仅前者 | 仅后者 | 都失败 | 差值 | 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Transformer vs **Top-1** | 5,324 | 922 | 338 | **1,466** | 2,598 | **+21.19 pp** | [19.73, 22.64] | 4.6e-167 |
| Transformer vs **Top-5** | 5,324 | 640 | 620 | **1,499** | 2,565 | **+16.51 pp** | [14.87, 18.15] | 1.6e-83 |
| Top-1 vs Top-5 | 5,324 | 1,160 | 1,228 | 979 | 1,957 | Top-1 +4.68 pp | [2.95, 6.40] | 1.3e-7 |

## 2. 分组结果

| 划分 / 长度 | Transformer-M | Top-1 | Top-5 |
|---|---:|---:|---:|
| Seen / Short | 42.9% | **51.3%** | 33.2% |
| Seen / Medium | 19.2% | **46.2%** | 38.2% |
| Seen / Long | 11.3% | **45.0%** | 41.4% |
| Unseen / Short | 41.6% | **47.2%** | 46.1% |
| Unseen / Medium | 20.3% | 43.1% | **44.8%** |
| Unseen / Long | 9.5% | **38.0%** | 37.6% |

Top-1 相对 Transformer 的配对提升：Short +8.4 pp（Seen）/ +5.6 pp（Unseen），
Medium +27.0 / +22.8 pp，**Long +33.7 / +28.6 pp**。差距几乎全部来自中长距离任务。

## 3. 行为分析

| 指标 | Transformer-M | Top-1 | Top-5 |
|---|---:|---:|---:|
| 越界（任意时刻） | 48.8% | **0.3%** | 0.11% |
| 转向占执行步数 | 2.75% | 5.25% | 58.6% |
| **目标在身后时的转向率** | **2.2%** | **26.0%** | 84.9% |
| 目标在身后的状态数 | 54,703 | 5,123 | 236,288 |
| 过早 STOP | 27.0% | **53.7%** | 4.4% |
| 进入 20 m 未停 | 6.0% | 1.1% | **36.1%** |
| 到 200 步上限 | 0 | 0 | **19.2%** |
| 碰撞 | 1 | 2 | 0 |

- **Top-1 的失败几乎只有一种**：过早 STOP（2,861 条，53.7%）。它几乎不会飞丢
  （越界 0.3%），但会因为"估计距离 ≤20 m"而在真实距离 20–60 m 处停下。
- **Top-5 的失败是另一种**：它进入 20 m 的比例高达 76.3%，却只有 45.1% 会停；
  36.1% 进入后继续飞、19.2% 冲到 200 步上限；转向占步数 58.6%，是一种左右振荡。
- **STOP 决策误差**：Top-1 的估计距离平均**偏低 9.7 m**（中位绝对误差 7.0 m），
  2,861 次过早 STOP 全部是"估计 ≤20 m 而真实 >20 m"；Top-5 平均**偏高 8.0 m**
  （中位绝对误差 9.1 m），所以它更少早停、更多过了 20 m 也不停。
  偏低的来源是候选网格的离散化：24 m 网格上，离真实目标最近的候选可以比目标近十几米。

## 4. 协议中的八个问题

1. **能否达到 Transformer-M 的 23.7%？** 能，而且大幅超过：Top-1 **44.9%**、Top-5 **40.2%**。
2. **是否更容易纠正偏航？** 是。目标在身后时的转向率：2.2%（T）→ 26.0%（Top-1）。
3. **是否更少直飞越界？** 是。越界 48.8% → 0.3%（Top-1）/ 0.11%（Top-5）。
4. **目标转到机头后方时能否正确转向？** Top-1 在 26% 的身后状态转向（其余是停机边缘或
   刚好越过阈值）；Top-5 会转（84.9%），但左右振荡，步数从 41 增到 130。
5. **是否有大量由候选定位误差造成的错误 STOP？** 有，而且这是 Top-1 的主要失败：
   2,861 条过早 STOP 全部来自低估距离（平均 −10.4 m）。
6. **Top-1 与 Top-5 哪个更稳定？** Top-1：SR 高 4.7 pp、步数少 3 倍、失败类型单一。
   Top-5 的 OSR 更高（76.3%）但 STOP 决策差，属于"到了不停"。
7. **长距离任务能否显著优于 Transformer？** 能：Long SR 从 11.3%/9.5% 提升到
   45.0%/38.0%，配对提升 +33.7/+28.6 pp；这正是 Transformer 越界最严重的区间。
8. **Transformer 成功而 Geometry 失败的 338 条有什么共同点？** 主要是**短任务**
   （184/338）、起点离目标较近（中位 134.6 m）。几何控制器在这些任务里因为候选距离
   低估而提前 STOP；Transformer 的实时视觉/历史在"最后几米"提供了更细的停止判断。
   这类任务合计只有 338 条（6.3%），远小于几何控制器多赢的 1,466 条。

## 5. 案例

- 轨迹对照图（12 条，真实地图，红色=Transformer，蓝色=Top-1，橙色=几何控制器的估计目标）：
  `figures/trajectory_cases.png`
  - T 失败 / G 成功 4 条（长任务为主）
  - T 成功 / G 失败 4 条（短任务，几何早停）
  - 两者都成功 2 条、都失败 2 条
- 对比视频（2 条，红蓝轨迹同步生长）：
  - `videos/val_unseen_00154_compare.mp4`（T 越界，G 成功）
  - `videos/val_seen_00465_compare.mp4`（T 成功，G 提前停）

## 6. 结论

按协议的三档判定，本实验落入 **A：Geometry SR 明显高于 Transformer-M**。

含义（有证据的部分）：

- 在**当前输入协议**下（目标图像同图同尺度裁剪 + 24 m 候选匹配），12 维信念已经携带了
  足够准确的"目标方向 + 距离"（Stage 07 实测方向误差中位 4.4°、距离 MAE 约 7.5 m），
  一个固定阈值的几何控制器即可解释并超过 Transformer-M 的吞吐。
- Transformer-M 的主要损失不是"感知不够"，而是**策略在分布外状态上不修正**：
  目标在身后 54,703 个状态里只有 2.2% 转向，最终越界 48.8%。
- 因此"复杂视觉融合带来导航收益"这一说法，在本协议下**没有得到证据支持**；
  重新审视实时视觉/历史独立贡献是必要的（Stage 07 的消融矩阵 A1–A8）。

**不能**由此得出任何"数据泄漏"结论。Geometry 用的是同一套目标参考图像与候选检索；
本实验只说明：在当前信息协议下，目标位置估计 + 简单方向控制可以解释并超过现有模型的 SR。
几何控制器的 44.9% 本身也受候选距离偏差限制（53.7% 过早 STOP）。

## 7. 限制

- 控制器阈值（25°、20 m）为协议**预先指定**，没有根据任何 split 调整；Top-5 是预先
  登记的敏感性对照，不是事后挑选。
- 几何控制器不使用当前视觉与历史；它**使用**目标参考图像（经 ResNet 编码）与 24 m 候选库，
  这与 Stage 06 完全相同。因此这不是"无视觉"实验，而是"无实时局部视觉/无序列模型"实验。
- 迟到/偏差：候选网格 24 m，最近候选与真实目标可差十几米，直接造成 Top-1 的提前 STOP。
- 目标图像有效比例 <0.5 的任务仍保留在主结果中（与 Stage 06 相同规则）。

## 8. 复现

```bash
# 08A 契约与冒烟（8 项测试 + 12 条任务）
python -I -m pytest experiments/08_belief_only_geometric_navigation/scripts/test_geometry_contract.py -q
python -I -u experiments/08_belief_only_geometric_navigation/scripts/evaluate_geometry.py \
  --controller top1 --task-file experiments/08_belief_only_geometric_navigation/results/smoke_ids.txt \
  --out-dir experiments/08_belief_only_geometric_navigation/results/smoke_top1

# 08B 全量（tmux citynav08_geometry）
bash experiments/08_belief_only_geometric_navigation/scripts/run_full.sh
python -I -u experiments/08_belief_only_geometric_navigation/scripts/analyze_geometry.py
python -I -u experiments/08_belief_only_geometric_navigation/scripts/make_figures.py
python -I -u experiments/08_belief_only_geometric_navigation/scripts/make_case_figures.py
python -I -u experiments/08_belief_only_geometric_navigation/scripts/make_comparison_videos.py
```

运行时间：Top-1 **1.08 分钟**（5,324 条），Top-5 **2.25 分钟**；分析与图约 1 分钟。
结果：`results/{top1,top5}/journal.jsonl`、`results/comparison.json`、`comparison_tables.csv`、
`paired_outcomes.csv`；逐条轨迹在服务器 `results/{top1,top5}/episodes/`（不入库）。
