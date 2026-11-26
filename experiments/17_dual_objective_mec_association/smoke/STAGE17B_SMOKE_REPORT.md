# Stage 17-B · 环境一致性与约束机制 Smoke Test 报告

日期：2026-11-26
基准 commit：`7067e0b054dc3eee3fb7638197396cdfb3652a59`
范围：Train 池固定 10 条任务（`train_tasks[:10]`），未使用 Val-Seen / Val-Unseen / Test。
**未执行任何 DRL 训练，未重训 MEC 专家，未修改任何模型权重。**

## 0. 前置检查

| 项目 | 结果 |
|---|---|
| 5 个 Stage-14 `best.pt` 存在 | ✅ Small 244 MB / Medium 664 MB / TS 845 MB / TM 1.99 GB / Large 4.14 GB |
| 10 节点部署（`mec_positions_k10.json`） | ✅ 5 模型 ×2 |
| 模型 `eval()`、`requires_grad=False` | ✅（`env16.load_models`） |
| 任务集合 Stage 16/17 相同 | ✅ 同一 `load_tasks_and_split()` 前 10 条 |
| 随机种子 | ✅ 每个 episode `numpy.random.default_rng(7)` |
| 输出目录 | ✅ `experiments/17_dual_objective_mec_association/smoke/` |

## 1. 逐项结果

| 检查 | 关键数值 | 结果 |
|---|---|:--:|
| E1 Stage 16/17 轨迹一致 | 10/10 任务的 动作/位置/朝向/终止原因/SR/步数 完全一致 | ✅ |
| E2 单次专家前向 | expert_forwards 499 == completed_inferences 499 | ✅ |
| E3 覆盖参数 | `coverage_m=1.0` 隔离配置下全部节点超半径 → `no_feasible_mec`；正式配置 500 m | ✅ |
| E4 截止时间 | `d_max_ms=1.0` 下全部节点 `deadline_valid=0` → `no_feasible_mec` | ✅ |
| E5 服务已付、运动被拒 | task 6370 / node 0：服务 51.25 J，预算 135.49 J，运动 0 J，`energy_budget_exhausted`，推理 1、执行 0 | ✅ |
| E6 无可行 MEC | `step(None)` 无异常，正常终止 | ✅ |
| E7 能耗对账 | 正常轨迹 ΣE=40,366.41 J == cumulative；1 J 预算分支 ΣE=0 == cumulative | ✅ |
| E8 终止后不计费 | 再次 `step()` 抛 `RuntimeError`；能耗/时延/计数/记录均不变 | ✅ |
| E9 Legacy E2E 独立核对 | 10/10 任务逐 step 与 Stage 16 独立计算一致 | ✅ |
| E10 上行负载 | 4,281,664 bit；Relation 不在上传清单 | ✅ |
| E11 IEEE 飞行功率 | 3 m/s 157.555 W / 262.593 J；5 m/s 143.614 W / 143.613 J（`unit_checks` U5/U6） | ✅ |
| E12 Matching | 4.0e6 FLOPs；占视觉计算 0.035%（<1%） | ✅ |
| E13 事件计数 | attempts 141 ≥ inferences 141 ≥ executed 140；`Σ model_selection = 141`；记录对账一致 | ✅ |
| E14 新时延目标 | 200 m/TS：D_coll 175.289 ms ≠ legacy 59.462 ms，且等于分项之和 | ✅ |
| E15 碰撞/越界计费 | 确定性碰撞（`birmingham_block_4`）与越界均触发；Forward 计满 **262.593 J** | ✅ |

**15/15 PASS**（脚本以 `sys.exit(0)` 退出；任一失败会 `sys.exit(1)`）。

## 2. 调用计数（示例 episode）

| 计数 | 值 |
|---|---:|
| 决策步数 | 44 |
| UAV Layer4 编码调用（`enc4`） | **44**（每步 1 次） |
| Layer2 匹配编码调用（`matcher.features`） | **45**（1 次目标编码 + 44 次观测编码） |
| MEC 专家前向（10 任务合计） | **499** |
| completed_inference（10 任务合计） | **499** |

即：每个决策步恰好 1 次 Layer4 + 1 次 Layer2 + 1 次专家前向，无重复推理。

## 3. 一个真实步骤的完整计算（task 0，map birmingham_block_9，step 2）

| 项目 | 值 |
|---|---:|
| 选中 MEC / 模型 | node 8 / **tm** |
| 导航动作 | 1（Forward） |
| 水平距离 / 3D 距离 / 仰角 | 133.623 m / 139.482 m / 16.665° |
| 上行负载 | 4,281,664 bit（535,208 B） |
| 上行速率 | 35.393 Mbit/s |
| $D_{up}$ | 120.974 ms |
| $D_{layer4}$ / $D_{layer2}$ / $D_{matching}$ | 5.4495 / 2.2126 / 0.0027 ms |
| $D_{UAV}$ | 7.6647 ms |
| $D_{inf}$（TM） | 10.0025 ms |
| **$D_{coll}$** | **138.641 ms** |
| 悬停能耗 $P_{hover}D_{coll}$ | 23.360 J |
| 编码能耗 $P_{enc}D_{UAV}$ | 0.115 J |
| 通信能耗 $P_{tx}D_{up}$ | 0.012 J |
| $E_{coll}$ | 23.487 J |
| $E_{forward}$ | 262.593 J |
| $E_{step}$ | **286.079 J** |
| 剩余电量 | 359,521.94 J / 360,000 J |
| 可行节点数 | 10 |

Episode 汇总（44 步，成功）：

| 指标 | 值 |
|---|---:|
| 累计协同服务时延 $D_{sum}$ | 5,665.109 ms |
| 累计能耗 | 12,157.432 J |
| 剩余电量 | 347,842.568 J |
| Σ step 能耗 / Σ step 时延 | 12,157.432 J / 5,665.109 ms（与累计完全一致） |
| attempts / inferences / executed | 44 / 44 / 44 |
| 节点使用 | small 0/5:0, TM 6/7: 13+13, Large 9: 18 |
| Legacy Stage-16 E2E（对照） | 2,189.078 ms |

**注意**：Legacy E2E 2,189 ms 与新定义 $D_{sum}$ 5,665 ms 不同是预期的——后者包含 Layer2（2.21 ms/步）与 4.28 Mbit 上行（约 121 ms/步），而 Legacy 仍是 Stage-16 的 1.6 Mbit + 80/10 m 口径。

## 4. 约束执行结果

| 约束 | 观测 |
|---|---|
| 覆盖 500 m | 正式配置 10 节点全部覆盖；隔离配置下半径改为 1 m 后全部不可行 |
| 截止时间 400 ms | 正式配置下所有步骤 `deadline_valid=True`（最慢 138 ms）；隔离配置 1 ms 时全部不可行 |
| 能量预算 360 kJ | 44 步示例消耗 12.16 kJ（3.4%），全程未触发；E5 用构造预算验证了触发路径 |

## 5. 已知口径与说明

1. **目标编码为一次性成本**：`env.start()` 时对目标做 1 次 Layer2 编码（示例中 45 次匹配调用中的第 1 次）。该一次性开销**未计入** $D_{sum}$；其量级为 2.21 ms/任务（约 0.04%）。
2. **E7 的 1 J 预算分支**终止于 `no_feasible_mec`（服务成本高于预算），这是预期行为，与 E5 的"服务可付、运动不可付"是两种不同情形。
3. **碰撞/越界按完整动作计费**（固定动作成本抽象），已在 E15 用确定性构造验证。
4. **不活跃的约束不代表无效**：本次 10 条任务中电池约束未触发，属后续实验分析范围。

## 6. 结论

- E1–E15 全部通过，环境与 Stage 16 的导航行为逐步一致，计费、约束、计数与日志自洽。
- 无 Traceback、无 NaN、无重复推理。
- 建议进入人工审核；审核通过后可讨论 Stage 17-C（双目标 DRL）的设计，但本轮不启动训练。

## 7. 产物

- `smoke/env_consistency_r4.json`（15 项结果）
- `smoke/example_step.json`（完整示例步骤与 episode 汇总）
- `logs/stage17b_smoke4.log`、`logs/capture_example.log`
- `stage17b/capture_example_step.py`（计数与示例抓取）
