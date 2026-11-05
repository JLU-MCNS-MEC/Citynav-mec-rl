# 阶段 2：真实 CityNav 地图 MEC 部署数量分析报告

生成时间: 2026-11-05（审计修正版 v2，commit 见 AUDIT_2026-11-05.md 审计项 4）
方法: 每张城市地图在**可通行区**（0.1m 高程 TIF 的 1m 有效掩码，`pipeline.py MapRaster`）
规则网格**均匀部署**（用户指示：无历史记录则均匀部署）；
**正式任务集（official）** = 11A 质量审计 Train **18,521** + Val-Seen **2,498** = **21,019 个任务起点**，覆盖 **24 张有任务的地图**；
参考任务集（all，仅部署规划参考）= 原始 Train 22,002 + Val-Seen 2,498 + Val-Unseen 2,826 = 27,326
（`deployment_analysis_all_tasks.json`）。
覆盖半径 500m、通信模型 = mec-v8.2 `latency_model.a2g_channel`（LOS 概率路径损耗 +
Shannon，载频 2GHz/GBS 10m/UAV 80m/带宽 5MHz/功率 0.1W），payload = 真实
CityNav 视觉 token 49×2048×16 bit = 1.6 Mbit（2026-11-05 审计统一口径）。

## 结果总表（official，21,019 任务）

| K | 可通行区覆盖 | 任务可达率 | 最近距离均值 | 距离 P95 | 任务点可用数(均值/最小) | 最近分配负载σ | 通信时延 P95 |
|---|---|---|---|---|---|---|---|
| 5 | 100% | 100%（21019/21019） | 98m | 208m | 5.0 / 4 | 1.3 | 52.0 ms |
| 8 | 100% | 100%（21019/21019） | 75m | 170m | 8.0 / 7 | 2.1 | 42.6 ms |
| 10 | 100% | 100%（21019/21019） | 65m | 144m | 10.0 / 9 | 2.7 | 36.3 ms |
| 12 | 100% | 100%（21019/21019） | 59m | 144m | 12.0 / 11 | 3.3 | 36.3 ms |
| 15 | 100% | 100%（21019/21019） | 52m | 123m | 15.0 / 14 | 4.2 | 32.0 ms |
| 20 | 100% | 100%（21019/21019） | 43m | 107m | 20.0 / 17 | 5.6 | 29.4 ms |

参考（all，27,326 任务，含原始 22,002 与 Val-Unseen）：
| K | 距离 P95 | 可用数最小 | 通信 P95 |
|---|---|---|---|
| 5 | 207m | 4 | 51.7 ms |
| 8 | 168m | 7 | 42.1 ms |
| 10 | 142m | 8 | 35.9 ms |
| 12 | 144m | 11 | 36.3 ms |
| 15 | 124m | 14 | 32.2 ms |
| 20 | 107m | 17 | 29.3 ms |

## 结论与建议

1. **覆盖充分性在 K=5 即满足**（100% 覆盖/可达，每任务最少 4 个可用节点）；
   **K=10 的推荐理由不是覆盖，而是通信时延优化与关联多样性**：
   通信 P95 由 52.0ms 降至 36.3ms（省 15.7ms），
   每任务可用节点由最少 4 个增至最少 9 个（给关联策略/RL 提供候选），
   且与历史 Stage-2r 的 10-MEC 设定及本轮简报一致。K>10 的边际收益递减
   （K=20 仅再降 6.9ms），成本线性增长。
2. 部署位置已存 `mec_positions_k10.json`（official）/ `mec_positions_k10_all_tasks.json`（all），
   均匀部署、规则网格 + 有效区过滤。
3. **注意**：v8.2 的 120m 最小间距在真实小尺度街区地图上无法处处满足
   （K=10 时部分地图需放宽；K≥12 时全部需放宽）——真实地图可通行区碎片化
   程度远高于合成环境，逐图记录 `spacing_relaxed` 标志。
4. **任务集口径（审计项 4）**：Train 22,002 是原始 CityNav 人工轨迹；11A 教师池
   18,521 是经 Stage-02 质量审计（High+Medium）与 Stage-03 四动作教师成功
   筛选后的训练池（排除 3,481 条），也是 RL 唯一允许使用的 Train 数据。
   本报告的 official 口径已改用 18,521；22,002 版本仅作部署规划参考保留。

## 证据文件

- 分析脚本：`scripts/analyze_deployment.py`（--task-set official|all）
- 结果：`results/deployment_analysis.json`、`deployment_analysis_all_tasks.json`、
  `mec_positions_k10.json`、`mec_positions_k10_all_tasks.json`
- 地图源：`dataext/htnav-repro/data/rgbd/*.tif`（0.1m 高程）+ `*.png`
- 任务源：06 manifest（Val-Seen 2,498 / Val-Unseen 2,826）+ 11A 教师清单（18,521）
  + 原始 `citynav_train_seen.json`（22,002）
- 通信模型：mec-v8.2 `latency_model.py`（commit `d988a9a`）同款公式
