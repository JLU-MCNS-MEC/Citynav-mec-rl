# Stage 08A：契约与小规模验证

- 日期：2026-10-30
- 状态：**完成**（12 条任务 × 2 个控制器 + 8 项契约测试）
- 本阶段只检查程序正确性；**没有根据 SR 调整任何阈值**（25° / 20 m 为协议预先指定）

## 1. 被验证的东西

| 项 | 结论 | 证据 |
|---|---|---|
| 控制器只接收 12 维信念 | ✅ | `act(belief)` 签名；传 11 维或 (1,12) 直接报错 |
| 控制器不接触任务/目标 | ✅ | AST 检查：`geometry_controller.py` 只 import `math`/`numpy`，标识符中无 `task`/`target`/`episode_id`/`manifest` |
| 不调用实时局部 RGB | ✅ | `geometry_env.py` 的 AST 中无 `heading_crop`、`history_block`、`TransformerM`、`torch` |
| 目标信念来源与 Stage 06 一致 | ✅ | 目标嵌入与 Stage 05 缓存 `target_emb` **FP16 逐位相同** |
| 四个动作规则 | ✅ | 单元测试覆盖 STOP / ±25° / Forward 边界 |
| Top-5 控制器使用加权分量 | ✅ | 用 belief[6,7,8] 的单元测试 |
| STOP / 碰撞 / 越界 / 200 步 | ✅ | 复用 Stage 06 的 `segment_status`、`obstacle_for`、`classify`（Stage 06 契约测试仍通过） |
| 同一任务重复执行一致 | ✅ | 12/12 episode 的逐步轨迹完全相同 |

测试命令：`python -I -m pytest scripts/test_geometry_contract.py -q` → **8 passed**。

## 2. 冒烟任务与结果

12 条任务 = Val-Seen / Val-Unseen × Short/Medium/Long 各 2 条（cohort A 的前两条），
与 Stage 06 小规模验证使用同一批 episode。

| 控制器 | SR | 失败构成 | 平均步数 | 每条耗时 |
|---|---:|---|---:|---:|
| Top-1 Geometry | 41.7%（5/12） | premature_stop 7 | 38.2 | 0.120 s |
| Top-5 Weighted | 41.7%（5/12） | reached_but_not_stopped 6、premature_stop 1 | **124.3** | 0.133 s |

初步观察（仅 12 条，不作结论）：
- Top-1 常常在估计距离刚过 20 m 时停下（7 条过早 STOP）；
- Top-5 的加权距离把 5 个候选混在一起，估计值更平滑但更保守：它很少停，
  常常进入真实 20 m 后仍在飞（6 条），平均步数 124。

## 3. 耗时估算

- 实测 0.120–0.133 s/episode（单进程，无 Transformer、无局部视觉编码）。
- 全量 5,324 条：**约 10.6 分钟 / 控制器**；Top-1 + Top-5 合计 **约 21 分钟**。
- 目标嵌入按 (map, 目标坐标) 缓存，重复目标不会重复编码。

## 4. 防作弊的代码结构

```
evaluate_geometry.py
      └── geometry_env.run_episode(task, controller, ...)
                belief = belief_features(lib, tvec, pos, yaw, sims)
                action = controller.act(belief)          # 唯一输入
                # task / true target 只用于：构建目标图像、评分
```

控制器模块不可 import 任务或评测代码（AST 测试强制）；`act` 的形参只有 `belief`。

## 5. 下一步

08B：在完全相同的 5,324 条任务上运行 Top-1 与 Top-5，与 Stage 06 的 Transformer-M 结果
做逐 episode 配对比较（McNemar、配对差值 CI、失败类型分解、轨迹对照图）。
