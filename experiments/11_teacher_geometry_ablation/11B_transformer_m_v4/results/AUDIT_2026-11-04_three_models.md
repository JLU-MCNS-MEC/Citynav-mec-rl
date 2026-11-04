# Small / Medium / V5 Fix 三模型结果只读审计报告

生成时间: 2026-11-04（审计期间未重启训练、未重新生成特征、未重跑闭环；全部只读）
审计范围: 已完成并推送 GitHub 的三个模型（Val-Seen，修正协议，seed 42）

## 一、训练侧审计（三个模型一致）

| 项 | 值 | 证据 |
|---|---|---|
| Seed | 42 | 各 run 目录 `config.json`（runs_small / runs_medium / runs_11c_v5fix / seed42） |
| 训练轨迹数 | **18,521** episodes | `11A_teacher_dataset_audit/results/teacher_pool_audit.json`（final_episodes=18521）+ `teacher_episode_manifest.csv` 18,521 数据行 |
| 训练状态数 | **1,279,882** states | 同上（decision_states.total）+ train 日志 `train states 1279882 (memmap)` |
| 协议 | batch 128 / 30 ep / patience 10 / lr 3e-4 / blank 5% / OneCycleLR | 三份 config.json 逐项一致 |

训练结果（最佳 epoch 指标，取自各 REPORT.md 与 history.jsonl）：

| 模型 | 最佳 Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro-F1 | 各动作 F1（Stop/Fwd/TL/TR） | 训练耗时 |
|---|---|---|---|---|---|---|---|---|
| Small | 6 | 0.579 | 77.8% | 0.552 | 79.8% | 0.7243 | 0.959 / 0.829 / 0.565 / 0.545 | 4.4 h（17 ep） |
| Medium | 1 | 0.625 | 76.8% | 0.585 | 79.1% | 0.7103 | 0.951 / 0.823 / 0.556 / 0.511 | 5.8 h（12 ep） |
| V5fix | 3 | 0.527 | 78.9% | 0.502 | 81.0% | 0.7475 | 0.957 / 0.830 / 0.607 / 0.510 | 6.3 h（14 ep） |

（各动作 F1 来源：train_{model}.log 最佳 epoch 行的 per_class_f1；Medium ep1 = Stop 0.9505 / Forward 0.8234 / TurnLeft 0.5559 / TurnRight 0.5112）

## 二、闭环审计（三个模型一致通过）

| 检查项 | Small | Medium | V5fix | 证据 |
|---|---|---|---|---|
| Manifest val_seen 条目 | **2,498** | 2,498 | 2,498 | `06_closed_loop_success_rate/results/seed42/manifest.json`（tasks 5,324 = val_seen 2,498 + val_unseen 2,826；counts 短870+中833+长795=2,498） |
| 三模型 manifest 一致 | ✅ 同一 SHA | ✅ | ✅ | 三份 `eval_config.json` 的 manifest_sha 均为 `2da37816…` |
| task-file / limit | 用 task-file（2,498 行），**无 --limit** | 同 | 同 | `task_status/val_seen_ids.txt` 2,498 行；`eval_one.sh` 仅传 `--task-file`，无 `--limit` |
| journal 行数 / 唯一 episode | 2,498 / 2,498 | 同 | 同 | 各 `journal.jsonl` |
| 重复 / 漏测 | **0 / 0** | 0 / 0 | 0 / 0 | journal 唯一 id 集合 vs task-file 逐项比对 |
| episodes 文件数 | 2,498 | 2,498 | 2,498 | 各 `episodes/*.json` |
| 异常 / 未执行 | **0 / 0** | 0 / 0 | 0 / 0 | journal execution_status 全 ok、exception 全空 |
| SR 分母 / OSR 分母 | **2,498 / 2,498** | 同 | 同 | success 计数除以 journal 行数 |
| 无候选任务 | **0** | **0** | **0** | candidate_status 全为 ok、candidate_count 全 >0 |

**无候选任务专项结论**：Val-Seen 全部 2,498 任务均有候选库条目，空候选推理路径
（EMPTY_BELIEF/EMPTY_MATCH/EMPTY_LANDMARKS）在本轮三个闭环中**未被任何任务触发**；
空候选输入覆盖仅来自训练侧 5% blank 合成。因此"空候选闭环表现"在 Val-Seen 上
**无经验数据点**，三模型闭环数字不受其影响。

## 三、V5 Fix 目标几何信息流审计（只读代码核对）

证据: `experiments/v4_closed_loop_revalidation/scripts/run_closed_loop.py`
（eval 侧）与 `11B_transformer_m_v4/scripts/build_features_11b.py`（训练侧）。

**推理数据流**（按执行顺序）：
1. `target`（真实目标坐标，仅此处使用）→ `matching.prepare_goal(raster, target)` 目标裁剪图 → `matcher_enc.features(…, layer2)` → 中心 2×2 均值归一化 **gdesc → target_emb**（**目标视觉参考**，协议定义的任务规格输入）。
2. 候选库（CandidateStore，按 block 加载）→ 各候选嵌入 `lib.embeddings`。
3. **belief 初始化**：`enc4.embed(north_crop(target, 48m))`（目标裁剪视觉特征）与候选嵌入余弦相似度 → 初始 mixture。**未使用坐标**。
4. **belief 更新**：每步 `subpixel_match`（当前观测 vs 目标视觉），因果更新（η 接受门控）。**未使用坐标**。
5. **landmarks**：`landmark_inputs(belief, empty)` = belief.means/weights 的 Top-8；空候选时 EMPTY_LANDMARKS。**未使用坐标**。
6. history：`history_inputs`（相对 pose/视觉/动作，slot0=最近）。current：heading 裁剪单次编码（修正 5b）。states = 智能体位置+朝向。
7. 真实目标坐标的**唯一用途**：目标裁剪图、belief 初始化目标视觉特征、成功判定（stop 半径 20m）与 SPL/NE 指标——**均不进入模型输入张量**。

**结论**：
- landmarks 只由候选库 + 目标视觉特征 + 当前/历史视觉观测 + 因果信念更新得到 ✅
- **模型输入中不存在直接或间接使用真实目标坐标的通道** ✅
- 训练侧（build_features_11b.causal_episode）与推理侧输入定义一致，且由
  `test_train_inference_contract.py`（6/6）以契约测试固化（belief 回放对拍、
  历史四路一致、collate↔推理、空模式契约）✅
- 无候选任务仍保留目标视觉参考（target_emb 在 empty 分支外公共构造）——任务
  规格合法输入；因 Val-Seen 无此类任务，对三模型数字无影响，属协议性质记录。

## 四、统一对比表（实际分母 2,498）

| 模型 | 训练耗时 | 最佳 Epoch | Val Macro-F1 | SR | OSR | NE | SPL | 平均步数 | 失败分类（never_stop/越界/早停/到达未停） |
|---|---|---|---|---|---|---|---|---|---|
| Small | 4.4h | 6 | 0.7243 | 34.67% (866) | 35.19% (879) | 91.6m | 0.327 | 63.0 | OOB 1203 / model_stop 1214 / step_cap 81 |
| Medium | 5.8h | 1 | 0.7103 | 23.98% (599) | 25.62% (640) | 105.1m | 0.238 | 54.9 | OOB 1707 / model_stop 789 / step 1 / collision 1 |
| V5fix | 6.3h | 3 | 0.7475 | 53.60% (1339) | 54.08% (1351) | 56.2m | 0.469 | 71.1 | never_stop 226 / OOB 368 / premature 553 / reached_not_stopped 12 |

## 五、SPL 的 L* 定义审计

- 证据：journal 行 `l_star_m` 数值抽检——Small ep0: start[33.64,561.31]→
  target[368.22,440.78]，`l_star_m=355.62` = √(334.58²+120.52²) ✅
- **结论：L* = 起点到目标的欧氏直线距离**（`l_star_m`），SPL = L*/实际路径长
  （成功时），失败为 0（截断指标）。
- **局限（仅标注，未修改历史结果）**：欧氏 L* 不感知障碍与拓扑，绕行即被
  判"非最优"；且失败任务 SPL 一律 0，使 SPL 与 SR 强耦合，不独立衡量路径
  质量。NE 同为直线距离。建议后续版本若需路径质量指标，另增"有效路径长度
  L*（沿图最短路径）"作补充指标，历史数字保持原样。

## 六、证据清单与哈希

- Checkpoint SHA256（= 各 eval_config.ckpt_sha = SHA256_MANIFEST.md）：
  - Small best.pt `79b01bd5…d4584d`
  - Medium best.pt `3966c3c3…8ce22f`
  - V5fix best.pt `18e92e0f…894e777`
- Manifest SHA256：`2da3781677e1376c7ff37ee029956c3a8d54aa0a85f81feca5858fe611123446`
  （三模型 eval_config.manifest_sha 一致）
- GitHub 提交（分支 exp/citynav-mec-fixed50m-topdown）：
  - 实现链：`0274c498`（case studies）→ `3b30fc2c` `e34aca17` `e5c9c234`
    `7354d550`（V5 修正+契约测试）→ `49d4d69a`（Small/Medium 四动作）→
    `f97edb26`（current 物化）→ `9dde5eaa`（闭环工厂）→ `b60c1248`
    `77bccb3e`（Small 报告）→ `5330fdfc`（Medium/V5fix 报告+对比表+SHA 清单）
  - 本审计报告提交见其后的新 commit SHA
- 关键路径：
  - 训练证据：`experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/results/{runs_small,runs_medium,runs_11c_v5fix}/seed42/{config.json,history.jsonl,REPORT.md,best.pt}`
  - 闭环证据：`experiments/v4_closed_loop_revalidation/results/{small,medium,v5fix}_val_seen_20261103/{eval_config.json,journal.jsonl,episodes/}`
  - 教师池：`experiments/11_teacher_geometry_ablation/11A_teacher_dataset_audit/results/`
  - 日志：`task_status/logs/{train_small,train_medium,train_v5fix,eval_*}.log`

## 七、发现的问题与建议

1. **空候选闭环无经验数据**（提示性）：Val-Seen 无候选任务为 0，空候选推理
   路径未被验证。建议：只读统计 val_unseen 各 block 的候选数，若有空候选 block
   可在后续阶段补充评测（需用户批准，不自动执行）。
2. **manifest.candidate_library 为旧路径**（文档性）：`/home/20261024_1/…`
   已失效，实际由 CandidateStore 自身配置解析（评测正常运行）。建议更新该
   字段或注明"展示用，实际解析走 CandidateStore 配置"。
3. **`failure` 字段语义**（文档性）：journal 的 `failure` 是**标签**（成功行
   值为 `"success"`），不是布尔标志；使用时应以布尔字段 `success` 为准。
   建议在 run_closed_loop 文档字符串注明。
4. **manifest.seed 为日期串**（文档性）：`seed: 20261030` 是生成日期，与
   训练 seed=42 无关，易混淆；建议改名或注释。
5. **SPL 的 L\* 局限**（见第五节）：按用户要求仅标注，不修改历史结果。
