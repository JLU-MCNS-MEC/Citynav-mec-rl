# CHANGELOG

## [0.15] - 2026-11-17
### Features
- Oracle v2 关联规则修正（`oracle_action_agreement_v2`）：新增纯函数 `scripts/oracle_selection.py`，只按 `(agreement, P_i(a*))` 选择，最终平局按固定节点编号 G1<G2<…；删除旧版两处 E2E tie-break（多专家一致时、无专家一致时）。
- 四组 Val-Seen 2,498 完整重评（two_homo / two_hetero / five_homo / five_hetero），新结果独立保存于 `results/oracle_action_agreement_v2/`；旧版结果保留为 `oracle_e2e_tiebreak_v1`。
- 新增逐步审计字段、`steps.part00.jsonl`、单元测试（9 项）、对比 CSV、4 张图与 `ORACLE_V2_REPORT.md`。
### Design Rationale
- 研究定义要求 Oracle 仅依据专家导航动作与教师参考动作的一致性；通信/计算/E2E/距离/质量先验/RL 均不得参与关联。两处 E2E tie-break 已全部移除。
- 教师仍使用 Stage-03B Direct Teacher（真实位置/航向/目标/障碍图），未修改教师算法；实际动作仍来自所选 MEC 专家。
- v1 与 v2 不可混为同一算法的重复实验，报告与 CSV 均显式区分版本。
### Notes & Caveats
- 去掉 E2E tie-break 后 SR/OSR/SPL/NE/成功数/failure taxonomy **逐任务完全不变**（two 组 1,507 成功、five 组 1,899 成功）；v1 的 tie-break 只改变选中的同动作节点，从未改变导航动作。
- E2E 上升 +35.93 / +36.84 / +27.84 / +27.51 ms；通信上升 +34.91 / +34.87 / +24.45 / +24.40 ms；推理上升 +1.02 / +1.99 / +3.39 / +3.11 ms。
- v2 SR 60.33%（two）/ 76.02%（five），质量上界不变；v2 是质量上界而非时延上界。逐 episode 与逐步原始文件保留在本地（约 1.1GB），未推送。

## [0.14] - 2026-11-17
### Features
- Stage 12 异构算力 MEC 关联实验（`experiments/12_hetero_mec_association/`）：四组 10 节点部署（2 模型/5 模型 × 同构 10 TFLOPS / 异构 5-10-20-30 TFLOPS），16 个真实闭环 Baseline（random / max_rate / max_quality / oracle），PPO 与 DQN 各 4 组 × 2 种子，全部在 Val-Seen 2,498 任务上评测。
- 新增冻结协议、部署构造、真实逐步关联闭环、特权 Oracle 教师、RL 训练/评测、并行调度（自动评测 + ETA + 实时状态表）、汇总/图表/配对统计与最终报告生成。
### Design Rationale
- 五个 Stage-11 专家完全冻结；只改变时延建模计算量（Small 5 / Medium 15 / TS 40 / TM 100 / Large 200 GFLOPs）与节点算力（10 或 5/10/20/30 TFLOPS）。
- UAV 编码时延 = 实测编码器 FLOPs（8.174 GFLOPs）/ 1.5 TFLOPS = **5.450 ms**；只有编码器是实测的，五个专家计算量使用预设建模值。
- Oracle 沿用 v8.2 Stage-2k 特权定义：真实目标 + 障碍图 + 03B 教师动作，选择“动作与教师一致”节点中瞬时 E2E 最低者；不可部署，与 03B 教师在自身路径上 100% 一致（608/608 步）。
- 所有策略均执行全新闭环，禁止复用旧轨迹；HOMO/HETERO 使用相同节点位置与模型放置，以隔离算力效应。
### Notes & Caveats
- Baseline 关键结果：max_rate SR 44.96%（两模型）/41.07%（五模型），E2E 37.35/41.02 ms；max_quality SR 53.56%（所有部署）：oracle SR **60.33%（两模型）/76.02%（五模型）**，E2E 36.03–41.84 ms。
- 算力异构的时延效应可正可负：`two_hetero/max_quality` E2E −1.81 ms（TM 有高档节点）；`five_hetero/max_quality` E2E +2.42 ms（TM 被锁在 G9=5 TFLOPS）。
- PPO（纯 E2E 奖励）收敛到 ~90% Small：E2E 36.26–41.34 ms，SR 31.35–39.63%；DQN 不稳定：E2E 51.50–58.06 ms，SR 33.71–46.60%。**没有任何学习策略同时超过 max_rate 的 SR 与 E2E**；Oracle 上界与最佳 PPO 的 SR 差 36.4 pp。
- 无排队/切换开销；定位编码器未计入 E2E；每配置 2 种子、4,000 episodes；5/10/20/30 节点分配为本次预注册假设。

## [0.13] - 2026-11-04
### Features
- Transformer-S（`TransformerS4`，192/1 层/4 头/ff×3）与 Large（`TransformerL4`，352/2 层/8 头/ff×4）四动作实现：mec-v8.2 `ExplicitReasoningTransformer` 原样骨架（视觉投影/type+pos 编码/cls_token/goal→current 交叉注意力块/fusion TransformerEncoder norm_first+gelu+ff_mult/final_norm），只适配输入投影与融合（89 token 序列、几何 mask 并入 encoder padding）；注册 `train_11b.py --model ts|large` 与 `run_closed_loop build_model`。
- 五模型 Val-Seen 闭环全量完成：统一对比报告 `COMPARISON_five_models.md` + 各模型 REPORT.md + 权重 SHA256 清单（6 个权重）。
### Design Rationale
- 保留 v8.2 原尺寸原骨架（GBS_MODEL_CONFIGS 原值）；六动作协议的 heatmap/network 模态移除，代之以固定协议模态；几何特征复用 `TransformerM12Fix.landmark_geometry`（与 V5fix 同款修正）。
- TDD：test_explicit_models.py 12/12（RED=ModuleNotFoundError 已目睹）；旧权重保护新增显式测试。
### Notes & Caveats
- 闭环 SR：Small 34.67% / Medium 23.98% / Transformer-S 37.47% / Large 43.55% / V5fix 53.60%（基线：旧 v4 46.76%、旧 v5 60.09%）。
- SPL 的 L* 为欧氏直线距离（历史口径，局限已在审计报告标注，未修改）。
- TS/Large 的 step_cap 与早停类失败偏高（338/266），终点附近停止决策不足，见五模型对比分析。


## [0.12] - 2026-11-03
### Features
- V5 六项修正（`TransformerM12Fix`，`model11c_fix.py`）：无效几何 token 注意力 mask + 特征置零；关系角改为 heading 帧 bearing 差（wrap、旋转不变）并归一化（dist/50、角/π）；relation 由单 56 维 token 改为 28 个 per-pair token（`[d/50, Δ/π, valid]`，无效对 mask）；Top-8 槽位按权重降序稳定重排（无需重建特征）；空候选 blank 合成（5% episode/epoch 确定性轮换）消除训练/推理 OOD；推理侧当前帧单次编码（bit-identical）。
- Small（`SmallMEC4`，MeanPoolMLP）与 Medium（`MediumMEC4`，goal→current 交叉注意力）四动作实现，适配修正后 V5 完整多模态输入；统一训练入口 `train_11b.py --model small|medium|v5fix`（run 目录防覆盖）。
- 效率优化：`materialize_current_11b.py` 物化冻结 ResNet 当前帧特征（`current.npy` f16，约 257G），DataLoader 快路径免每 Epoch 重复视觉编码；SSD memmap + 分块 DataLoader。
- 闭环评测工厂：`run_closed_loop.py build_model` 覆盖 v3/v4/v5/v5fix/small/medium。
### Design Rationale
- `model11c.py` 字节级冻结：旧 V5 权重（runs_11c/best.pt）的 state_dict 仅能由旧类加载（rel_proj 56→3），冻结保证旧权重永久可用与对照评测可复现；修正实现放新文件避免任何结构变化。
- 共享契约常量 `v5fix_contract.py`：训练 blank 与推理空候选分支引用同一 numpy 常量，一致性由构造保证。
- 六项修正全部落在模型/加载/推理侧，物化特征语义不变 → 无需重新生成 11B 特征（CACHE_VERSION 不 bump）。
### Notes & Caveats
- 修正协议基线（Val-Seen 2,498，STOP 20m/200 步）：旧 v4 SR=46.76%、旧 v5 SR=60.09%（与历史 11D 的旧协议数字不可直接比）。
- 训练 blank 使 loss 略升属预期（5% 帧空信念）；Val 不 blank，协议可比。
- `run_11d.py` 为旧协议 harness（历史序/绝对 pose bug），仅用于旧协议复现，已 docstring 标注。
