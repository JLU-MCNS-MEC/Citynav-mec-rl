# Stage 17-A · 双目标受约束 MEC 动态关联环境

**状态：仅完成代码与文档，未执行任何 Smoke Test / 训练 / 评测。**

研究目标：在 CityNav 城市空中视觉导航任务上，最大化长期成功率（SR）的同时最小化累计端边协同推理服务时延 $D_{sum}$，
并满足 UAV 任务级能量预算与单次推理服务截止时间两条硬约束。

## 目录

```
17_dual_objective_mec_association/
├── README.md                      本文件
├── audit/AUDIT_STAGE14_16.md      Stage 14/16 审计（含文件路径与行号）
├── configs/default.json           全部参数与默认值
├── docs/
│   ├── 01_OPTIMIZATION_PROBLEM.md 优化问题与 CMDP 形式
│   ├── 02_LATENCY_MODEL.md        D_enc / D_up / D_inf / D_down
│   ├── 03_ENERGY_MODEL.md         悬停 + 前进(Zeng) + 转向 + 服务能耗
│   ├── 04_CONSTRAINTS.md          能量预算、截止时间、掩码、终止规则
│   ├── 05_STATE_ACTION.md         74→127 维状态、可行动作投影
│   ├── 06_DIFF_TABLE.md           Stage16→17 文件与行为差异
│   ├── 07_SMOKE_TEST_PLAN.md      Stage 17-B 验收方案（未执行）
│   ├── 08_REVIEW_QUESTIONS.md     待审参数与问题（P1/P2/P3）
│   └── 09_STATIC_REVIEW.md        静态审核记录
└── src/
    ├── config17.py                配置 dataclass
    ├── latency17.py               时延模型
    ├── energy17.py                能耗模型
    ├── env17.py                   EpisodeEnv17(EpisodeEnv16)
    ├── state17.py                 状态构造与布局
    ├── action17.py                可行集采样/投影
    ├── reward17.py                双目标与约束成本信号
    └── logger17.py                step / episode 记录
```

## 继承与禁止

**继承**：五种 Stage 14 冻结专家与 Checkpoint、104-token 输入协议、10 节点部署（5 模型 ×2）、
10 TFLOPS、500 m 覆盖、Stage 12 上行/编码模型、CityNav 任务划分与导航规则、Stage 16 的 DRL/基线接口。

**禁止且未做**：重训/微调专家、改输入模态、改成功判定、覆盖 Stage 16 结果、修改 Stage 12/14/16 任何文件。

## 快速开始（Stage 17-B 之后才允许）

```python
from src.config17 import load_config
from src.env17 import EpisodeEnv17
cfg = load_config("configs/default.json")
# env = EpisodeEnv17(by_map, device, models, norm, cfg)
# env.start(task); s, meta = build_state17(env, cfg); env.step(node)
```

## 关键统计量（设计期口径，非实测）

| 指标 | 定义 |
|---|---|
| $D_{sum}$ | $\sum_t (D_{enc}+D_{up}+D_{inf}+D_{down})$ |
| 能量 | $E_{coll}=P_{hover}D_{coll}+P_{enc}D_{enc}+P_{tx}D_{up}+P_{rx}D_{down}$；$E_{motion}\in\{E_{forward},E_{turn},0\}$ |
| 硬约束 | $\sum E \le E_{budget}$；$D_{coll}\le D_{max}$ |

所有 PROVISIONAL 参数见 `docs/08_REVIEW_QUESTIONS.md`，**审核通过前不得进入下一阶段。**
