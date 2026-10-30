# 研究规范核对：本项目 vs CityNav 官方协议 vs 模仿学习文献

> 每条外部事实都给出 URL；每条关于本仓库的判断都给出代码/结果路径。
> 标注：**[已核实]** 有源文件或外部 URL；**[推断]** 由已核实事实推出；**[需实验]** 尚无证据。

## 1. 官方 CityNav 是什么

- 论文：*CityNav: A Large-Scale Dataset for Real-World Aerial Navigation*，
  Jungdae Lee, Taiki Miyanishi, Shuhei Kurita, Koya Sakamoto, Daichi Azuma,
  Yutaka Matsuo, Nakamasa Inoue，**ICCV 2025**（arXiv v1 2024-06，v3 2025-08）。
  - https://arxiv.org/abs/2406.14240
  - https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_CityNav_A_Large-Scale_Dataset_for_Real-World_Aerial_Navigation_ICCV_2025_paper.pdf
- 观测：第一视角 RGB（224×224，90° FOV）+ 深度（256×256）+ **语言描述**；可选
  Geographic Semantic Map 作为辅助输入。**不提供目标图像。**
  - https://arxiv.org/html/2406.14240v3
- 动作：六类离散动作 —— Stop、Forward 5 m、TurnLeft 30°、TurnRight 30°、
  **Ascend 2 m、Descend 2 m**。 - https://arxiv.org/html/2406.14240v3
- 指标：NE、SR（停止点距目标 < 20 m）、OSR（轨迹曾进入目标 20 m）、SPL。
  - https://arxiv.org/html/2406.14240v3
- 划分：train 22,002 / 24 scenes；val-seen 2,498 / 24；val-unseen 2,826 / 4；
  test-unseen 5,311 / 6。 - https://arxiv.org/html/2406.14240v3
- 目标坐标**不作为输入**（"the absolute coordinates of the goal are not available"），
  只用于监督与评测。 - https://arxiv.org/html/2406.14240v3
- 起点：同一 3D 城市场景内、距目标 500 m 内的随机位置；起始高度 100–150 m，
  高度可通过 ascend/descend 改变。 - https://arxiv.org/html/2406.14240v3

官方代码：https://github.com/water-cookie/citynav-arxiv
- `--success_dist` 默认 20.0；SR 用 **3D 距离 < 20 m**，OSR 用 **xy 平面距离 ≤ 20 m**。
  - https://raw.githubusercontent.com/water-cookie/citynav-arxiv/main/vlnce/parser.py
  - https://raw.githubusercontent.com/water-cookie/citynav-arxiv/main/vlnce/evaluate.py
- `--eval_max_timestep` 默认 **200**（README 的快速评测示例用 20）。
  - https://raw.githubusercontent.com/water-cookie/citynav-arxiv/main/vlnce/parser.py
  - https://raw.githubusercontent.com/water-cookie/citynav-arxiv/main/README.md
- 回合在策略输出 STOP 时结束。 - .../vlnce/evaluate.py

## 2. 本项目改了什么

| 维度 | 官方 CityNav | 本项目（固定 50 m） | 影响 |
|---|---|---|---|
| 目标规范 | 语言描述 | **目标参考图像**（同图、同尺度、以真实目标为中心）+ 候选检索 | 目标定位显著变易（见 §3） |
| 观测 | 第一视角 RGB + 深度 | **俯视 RGB**（100 m×100 m 机头朝前 224²） | 看到的是屋顶/道路，不是立面 |
| 动作 | 6 类（含升降 2 m） | **4 类**（STOP/F/L/R） | 去掉高度决策 |
| 高度 | 可变（初始 100–150 m） | **固定 GROUND_LEVEL + 50 m** | 障碍规避基本消失（见 §4） |
| 历史 | 策略自行维护 | 16 步因果历史 + 16×2048 视觉/位姿/动作 | 显式时序输入 |
| 全局信息 | 可选 GSM | 固定的 24 m 候选库 + 12 维信念 | 见 §3 |
| 训练方式 | 论文基线为 seq2seq/CMA 等 | 行为克隆（教师轨迹） | 协变量偏移（见 §5） |
| 划分 | 22,002 / 2,498 / 2,826 / 5,311 | **完全相同的前三个划分** | 任务集合可直接对照 |

结论 **[已核实]**：本项目是"CityNav 数据 + 固定高度 + 俯视观测 + 目标图像 + 四动作"的
**衍生任务**，不是官方 CityNav 任务本身。数据划分一致，但输入与动作空间不同。

## 3. 目标图像 + 同图候选检索使任务明显简化

已核实的外部事实：
- 官方只给语言描述，明确不提供目标坐标，也不给目标图像。 - https://arxiv.org/html/2406.14240v3
- 官方起点在目标 500 m 内、同一场景。 - https://arxiv.org/html/2406.14240v3

本仓库的已测量事实（`FEATURE_LEAKAGE_AUDIT.md` §2）：
- 目标图像从**同一张地图**、**同一比例尺**、以真实目标坐标为中心裁剪；
- 候选库也从同一张地图按 24 m 网格切出，同一冻结 ResNet 编码；
- 因此 12 维信念与真实目标偏移的相关为 **0.987–0.996**，方向误差中位 **4.4°**、
  距离误差约 **8 m**——近似 GPS-to-goal。

**[推断]** 这使"目标定位"从官方的语言落地（需要理解描述并在第一视角里找）退化为
同图模板检索 + 局部控制。因此：

- 本项目的 SR **不能**与官方 CityNav 的 SR 直接比较；
- 若要与官方口径对齐，应改成语言目标、跨视角目标图像，或至少禁止同图候选匹配。

**[需实验]** 把目标图像换成"另一张地图的同位置裁剪"或"不同视角渲染"后，SR 会下降多少。
建议作为下一轮的 A6 配置（见 `AUDIT_REPORT.md` §5）。

## 4. 固定 50 m 改变了障碍规避问题

已核实：
- 官方高度可变（ascend/descend），城市建筑高度是任务的一部分。 - https://arxiv.org/html/2406.14240v3
- 本仓库 `z = GROUND_LEVEL[map] + 50 m`（`03b/obstacles.py`），1 m 高程栅格上
  28 张开发地图只有 **5 个单元**高于巡航高度（`03B REPORT` §1.1）。
- 03B 的 A* 分支全量只触发 1 条任务；Stage 06 闭环 5,324 条只有 **1 次碰撞**。

**[已核实]** 在这个高度上，"避障"几乎不构成挑战；本项目实际上检验的是
"视觉目标导航 + 停止"，不是"城市低空避障"。
官方 README 的评测命令确实传了 `--altitude 50`，但该参数如何影响轨迹高度**未核实**
（不在 `parser.py` 中）——因此不能据此说官方也是固定 50 m。
- https://raw.githubusercontent.com/water-cookie/citynav-arxiv/main/README.md

## 5. 与行为克隆 / DAgger 文献的对应

- Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning*, AISTATS 2011（DAgger）。
  - https://proceedings.mlr.press/v15/ross11a.html ；https://arxiv.org/abs/1011.0686
  - 算法：用当前策略采样状态、由专家标注、聚合数据后重训；理论上界
    `J(π̂) ≤ J(π*) + uTε_N + O(1)`，需要 β_i → 0。
  - https://ar5iv.labs.arxiv.org/html/1011.0686
- Ross & Bagnell, *Efficient Reductions for Imitation Learning*, AISTATS 2010：
  把模仿当作 i.i.d. 监督学习会产生**随 horizon 二次增长**的复合误差。
  - https://proceedings.mlr.press/v9/ross10a.html

与本项目的对应 **[已核实的事实 + 推断]**：

| 文献概念 | 本项目的证据 |
|---|---|
| 训练分布 = 专家状态分布 | 教师状态里目标在身后仅 **0.65%**（`CLOSED_LOOP_FAILURE_AUDIT.md` §2） |
| 闭环访问分布偏移 | 模型访问状态里目标在身后 **20.6%**，转向率从 5.9% 降到 2.75% |
| 复合误差 | 越界前 5 步目标在身后 128.7°，模型转向率 0.04% |
| DAgger 的动机 | 训练集里没有"偏离后如何恢复"的样本（教师从不偏离） |

**[推断]** 观测到的失败模式与 Ross & Bagnell 的复合误差描述一致；但本项目**没有做**
DAgger 实验，因此"加入恢复样本能提升 SR"仍是假设，**[需实验]**。

## 6. 指标口径差异

| 指标 | 官方 | 本项目 | 可比性 |
|---|---|---|---|
| SR | 停止且 **3D** 距离 < 20 m | 停止且**水平**距离 ≤ 20 m | **不可直接比较**（高度固定 50 m 时 3D 距离必然 > 50 m） |
| OSR | 轨迹曾进入 xy 20 m | 同 | 可比 |
| NE | 停止点距离 | 同 | 可比 |
| SPL | 官方实现记录 | 本项目成功加权 SPL 0.235 | 定义近似，参考路径不同 |
| 最大步数 | `eval_max_timestep` 默认 200 | 200 | 一致 |
| nDTW | 论文与代码未使用（**未核实**是否存在） | 未计算 | — |

**[已核实]** 本项目报告中的 SR 是"水平 20 m"，必须在报告里持续标注，
不能与官方 3D SR 并列。

## 7. 哪些结论有证据、哪些是推断、哪些需要实验

**有证据（可直接引用）**
- 数据划分与官方一致；训练/闭环特征路径一致、无坐标数值泄漏；
- 闭环 OSR 29.6%、越界 43.3%、条件成功率 79.9%（Stage 06）；
- 教师状态目标在身后 0.65% vs 闭环 20.6%；
- 12 维信念与真实目标偏移相关 0.987–0.996；
- 5,324 条只有 1 次碰撞，5 个高程单元高于巡航高度。

**推断（有证据支撑，但非因果）**
- 主要失败机制是协变量偏移 + argmax 先验；
- 信念是模型的主导输入（推理期置零的提示）；
- 同图目标图像显著降低任务难度。

**需要重新实验**
- 去掉/替换信念、目标图像、历史后的 SR 变化（A1–A8）；
- DAgger 式重标注能否提升 SR；
- 跨地图目标图像下的难度；
- 官方口径（语言目标、可变高度）下的可比 SR。

## 8. 引用清单

1. Lee et al., *CityNav: A Large-Scale Dataset for Real-World Aerial Navigation*, ICCV 2025.
   https://arxiv.org/abs/2406.14240
2. CityNav 官方代码。 https://github.com/water-cookie/citynav-arxiv
3. Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured Prediction to
   No-Regret Online Learning*, AISTATS 2011. https://proceedings.mlr.press/v15/ross11a.html
4. Ross, Bagnell, *Efficient Reductions for Imitation Learning*, AISTATS 2010.
   https://proceedings.mlr.press/v9/ross10a.html
5. Liu et al., *AerialVLN: Vision-and-Language Navigation for Aerial Vehicles*, ICCV 2023.
   https://arxiv.org/abs/2308.06735
