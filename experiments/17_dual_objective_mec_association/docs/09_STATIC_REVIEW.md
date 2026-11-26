# Stage 17-A · 静态审核记录

审核日期：2026-11-26
**本阶段未执行任何仿真、训练、评测或超参数搜索。** 下列数值均为按公式手工计算或从仓库文件读取，不是运行结果。

## 1. 已执行的静态检查

| 检查 | 方法 | 结果 |
|---|---|---|
| 语法 | `python -m py_compile src/*.py stage17b/*.py` | 通过 |
| 公式算术 | `stage17b/unit_checks.py` U1–U8（纯算术，不加载模型、不建环境、不跑 episode） | 8/8 通过 |
| 状态维度算术 | 74 + 3 + 10×5 = 127 | 与 `STATE17_DIM` 一致 |
| 时延常量 | 读取 `encoder_flops.json`、`env16.MODEL_GFLOPS_MEASURED` | 见表 2 |
| 单位 | 逐函数人工核对（见 `02_LATENCY_MODEL.md` §3、`03_ENERGY_MODEL.md` §8） | 一致 |
| 继承关系 | 逐个 import 对照 Stage 16 源文件 | 仅 import，未修改 |
| Stage 16 零改动 | `git status experiments/16_* experiments/14_* experiments/12_*` | 无改动 |

## 2. 手工计算的公式值（用于 Stage 17-B 的 U1–U7 对照）

| 量 | 计算 | 值 |
|---|---|---:|
| $D_{enc}$ | 8,174,272,512 / 1.5e12 × 1e3 | **5.449515 ms** |
| $D_{inf}$ Small | 4.954039 / 10 | 0.495404 ms |
| $D_{inf}$ Medium | 14.978528 / 10 | 1.497853 ms |
| $D_{inf}$ TS | 39.671126 / 10 | 3.967113 ms |
| $D_{inf}$ TM | 100.025418 / 10 | 10.002542 ms |
| $D_{inf}$ Large | 199.723981 / 10 | 19.972398 ms |
| $P_{hover}$ | 79.86 + 88.63 | 168.49 W |
| $P_{fly}(3)$ | 型阻 80.010 + 诱导 91.970 + 寄生 0.250 | **≈172.23 W** |
| $E_{forward}(v=3)$ | 172.23 × (5/3) | **≈287.05 J** |
| $E_{turn}$ | 168.49 × (30/30) | **168.49 J** |
| $E_{coll}$ 范围 | 168.49 × D_coll，D_coll ∈ [30,151.5] ms（下行忽略） | ≈5.1–25.5 J |

推导细节：

- 型阻 $=79.86\times(1+3\times3^2/120^2)=79.86\times1.001875=80.010$
- 诱导 $=88.63\times\sqrt{1+3^4/(4\times4.03^4)}=88.63\times\sqrt{1+81/1055.1}=88.63\times1.03768=91.970$
- 寄生 $=0.5\times0.6\times1.225\times0.05\times0.503\times27=0.250$

## 3. 代码级核对要点

### 3.1 时延
- `latency17.d_enc_ms` 调用 `P12.encoder_flops()`，与 Stage 16 的 `env16.enc_ms` 同源；
- `d_up_ms` 直接调用 `P12.comm_latency_ms`，无重写；
- `d_inf_ms` 使用 `MODEL_GFLOPS_MEASURED / mec_tflops`，与 `env16.NodeModel` 一致；
- `d_down_ms` 用 `SNR_down = SNR_up × (P_mec/P_uav)` 复用同一信道模型，未另造路径损耗。

### 3.2 能耗
- `fly_power_w` 三项与 Zeng 2019 公式逐项对应，$v\to0$ 时退化为 $P_0+P_i$；
- `e_forward_j` 用 `P_fly(v)×5/v`，**未**用 $P_{hover}\times5/v$；
- `e_turn_j` 用 $P_{hover}\times1$ s；
- `e_coll_j` 四项与规范一致，`p_aux_w` 默认 0 保证不重复计费。

### 3.3 约束
- `feasible_mec_mask = coverage ∧ deadline ∧ (E_coll ≤ remaining)`；
- `candidate_energy_cost` 只含服务成本，不含运动能耗（因果性）；
- `energy_budget_exhausted` 分支不调用父类 `step()`，因此不会执行越预算的导航动作；
- `no_feasible_mec` 优先于一切，保证不静默选非法节点。

### 3.4 继承安全
- `EpisodeEnv17.preview_action` 在 `copy.deepcopy(self.belief)` 上操作，`finally` 恢复，
  不污染父类状态；
- 真实转移仍由 `EpisodeEnv16.step()` 完成，导航/碰撞/Stop/成功判定未被重写。

## 4. 已知的实现性风险（需 Stage 17-B 验证）

| 风险 | 说明 |
|---|---|
| 预览动作与实际动作不一致 | 由确定性 argmax 保证；代码计 `preview_mismatch_count`，Smoke Test 必须为 0 |
| 每步重复计算候选表 | `candidate_table()` 会被状态构造与 step 各调用一次，性能开销约 2×；不影响正确性 |
| 状态含 10 维原始能量量级 | `remaining_energy_j/1e4` 量级约 0–3.6，与其他特征可比性需在训练前检查 |
| `ndtw` 不可得 | 输出 `None`，需先定义参考轨迹 |

## 5. 未执行清单（声明）

- 未加载任何专家 Checkpoint；
- 未构建环境实例；
- 未跑任何 episode；
- 未训练任何算法；
- 未做任何评测或阈值扫描。

（唯一执行过的脚本是 `stage17b/unit_checks.py`：只对公式做算术核对，
不 import 环境、不加载权重、不产生任何轨迹数据。）

以上全部属于 Stage 17-B，需先完成人工审核。
