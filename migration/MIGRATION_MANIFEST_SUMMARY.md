# MIGRATION_MANIFEST_SUMMARY

> 迁移数据清单摘要（脱敏版）。全部条目经 SHA-256 逐文件校验，源/目标 0 差异。

| 数据项 | 文件数 | 大小 | SHA-256 校验 |
|---|---:|---:|---|
| teacher_03b（03B 教师轨迹） | 114 | 28M | ✅ 一致 |
| candidates_04/cache_v2（candidates24/48） | 56 | 15M | ✅ 一致 |
| candidates_04/cache（v1 候选缓存） | 28 | 3.4M | ✅ 一致 |
| checkpoints/runs（seed42/43） | 10 | 252M | ✅ 一致 |
| checkpoints/runs_v2（seed42 全 epoch） | 53 | 788M | ✅ 一致 |
| DATA/original_citynav | 13 | 754M | ✅ 一致 |
| DATA/cityrefer | 2 | 27M | ✅ 一致 |
| DATA/cityrefer_original | 2 | 27M | ✅ 一致 |
| DATA/rgbd（34 块 png+tif） | 68 | 3.4G | ✅ 一致 |
| results_06（Stage 06 全量闭环） | 10,799 | 267M | ✅ 一致 |
| results_08（Stage 08 Top-1/Top-5） | 10,728 | 268M | ✅ 一致 |
| **合计** | **21,873** | **~6.0G** | **全部 ✅** |

## 条数核验

| 数据 | 期望 | 实测 |
|---|---:|---:|
| 教师轨迹 Train-Seen | 22,002 | 22,002 ✅ |
| 教师轨迹 Val-Seen | 2,498 | 2,498 ✅ |
| 教师轨迹 Val-Unseen | 2,826 | 2,826 ✅ |
| Stage 06 闭环（seed42） | 5,324 | 5,324 ✅ |
| Stage 06 闭环（test_unseen） | — | 5,311 ✅ |
| Stage 08 comparison / Top-1 / Top-5 | 5,324 ×3 | 5,324 ×3 ✅ |

## 旧服务器删除清单（用户最终授权范围）

| 项 | 文件数 | 大小 | 说明 |
|---|---:|---:|---|
| mec-v8.2 本地目录 | 33 | 87M | 代码已全量推送 GitHub（main@d988a9a），仓库干净 |
| 主线分支工作区（visual-encoder-ablation checkout，保留 .git） | 1,755 | 6.5G | 代码已推送（ahead=0/behind=0）；1 个未跟踪文件已先行保存到新服务器 |

其余旧服务器数据（fixed50m 工作区、DATA/PLY、审计记录、gsam 等）按用户要求**保留**。
