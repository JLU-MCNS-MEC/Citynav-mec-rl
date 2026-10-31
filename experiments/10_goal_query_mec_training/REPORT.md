# Stage 10 总报告：目标查询式 MEC 训练

> 本文件在 10E 完成后汇总 10A–10E 全部结论；各阶段独立报告见子目录 REPORT.md。

- 分支：`exp/citynav-mec-fixed50m-topdown`
- 起始 commit：`223f11d`（M0）
- 服务器：新服务器 `ubuntu`（RTX 5090 32 GB）

## 阶段索引

| 阶段 | 报告 | 状态 |
|---|---|---|
| 10A | `10A_point_query_cross_attention/REPORT.md` | 进行中 |
| 10B | `10B_geometry_comparison/REPORT.md` | 未开始 |
| 10C | `10C_mec_input_dataset/REPORT.md` | 未开始 |
| 10D | `10D_transformer_m_training/REPORT.md` | 未开始 |
| 10E | `10E_final_closed_loop/REPORT.md` | 未开始 |

## 最终待回答的问题（10E 后填写）

1. Point Query 和 Cross-Attention 哪个更适合当前任务？_待填_
2. 它们是否显著优于 C1？_待填_
3. 最终确定的 MEC 输入模态是什么？_待填_
4. 是否存在训练/推理不一致或信息泄露？_待填_
5. 新 Transformer-M 是否真正学会了更好的导航策略？_待填_
6. SR 比旧版提升了多少？_待填_
7. 与 09C-1 连续几何控制器相比，新 MEC 还差在哪里？_待填_
8. 后续是否还值得继续做更复杂模块？_待填_
