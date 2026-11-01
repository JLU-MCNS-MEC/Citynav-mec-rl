# success_examples_5：Transformer-M v3 双视角导航回放视频

- 分支：`exp/citynav-mec-fixed50m-topdown`
- 目录：`experiments/10_goal_query_mec_training/10E_final_closed_loop/videos/success_examples_5/`
- 生成日期：2026-11-01

## 这是什么

基于 **Stage 10E 已保存的闭环轨迹**制作的离散动作回放视频（**不是重新运行的
仿真实验**，不重新评测、不重新训练）。每帧对应一个已保存的决策状态，每个
状态保持 3 帧（10 FPS）；末帧停留更久。

布局严格参考 Stage 06 的 `make_videos.py` 与 `success_compilation.mp4`：

- **左侧（760×640）**：城市俯视 RGB 地图；20 m 成功判定圆；起点；真实目标
  （星标）；已飞行轨迹；当前 UAV 位置与航向箭头。
- **右侧（480×640）**：上方为当前 heading-forward 局部视觉观测（与模型推理
  完全相同的 `heading_crop`，标注 "UAV view (heading-forward, 100 m)"）；
  下方为当前步数、动作类别、目标距离与四类动作概率条。
- **底部信息栏**：Episode ID、Split、Short/Medium/Long、当前动作、当前目标
  距离、四类动作概率、最终 SUCCESS / FAIL、停止距离、最小距离、总步数。

## 概率来源（重要）

10E 的 Episode JSON 未保存 softmax 概率。本目录中的概率由 **离线重放**
（`../../scripts/replay_10e.py`）产生：加载 10D `best.pt`，严格复用
`run_10e.py` 的 C1 连续信念更新、当前视觉编码、16 步因果历史与模型输入构造，
对每个已保存决策状态重新前向，得到**真实 softmax 概率**。

校验结果（全部 5 条 episode，共 192 步）：

- 重算 argmax 与原始保存 action **100% 一致**（0 处不一致）；
- 概率和 = 1（误差 < 1e-6）；
- 概率为真实 softmax 输出，**不是** one-hot 冒充。

目标距离显示使用保存的 `true_dist_m`（仅可视化，不是模型输入——10C 契约
无真实距离输入）。

## 案例清单

| Episode | 长度组 | 新 M-v3 | 旧 M-v2 | 视频 |
|---|---|---|---|---|
| val_seen:00018 | short | ✅ SUCCESS | ❌ fail | `individual/val_seen_00018.mp4` |
| val_seen:00591 | medium | ✅ SUCCESS | ❌ fail | `individual/val_seen_00591.mp4` |
| val_seen:00629 | medium | ✅ SUCCESS | ❌ fail | `individual/val_seen_00629.mp4` |
| val_seen:00776 | long | ✅ SUCCESS | ❌ fail | `individual/val_seen_00776.mp4` |
| val_seen:00960 | long | ❌ **FAIL（对照）** | ✅ SUCCESS | `individual/val_seen_00960.mp4` |

00960 是**对照案例**（旧 M 成功、新 M 因过早 STOP 失败），在视频与
MANIFEST 中均如实标记，未标为成功案例。

## 合集

- `compilations/m_v3_success_compilation_4.mp4`：仅拼接新 M-v3 成功的 4 条
  （00018 → 00591 → 00629 → 00776），连续连接无转场，48.0 s。
- `compilations/selected_5_cases_compilation.mp4`：按
  00018 → 00591 → 00629 → 00776 → 00960 顺序拼接全部 5 条（00960 正确标记
  为 FAIL 对照），63.6 s。

## 编码与元数据

- H.264（libx264）、1280×720、10 FPS、yuv420p、CRF 26、每个状态 3 帧
- 每条视频的渲染元数据（源/重放 SHA-256、checkpoint SHA-256、时长、大小）
  见 `meta/*.json` 与 `MANIFEST.json`
- checkpoint SHA-256：`5f01a52dd5bbb53e361a`（10D best.pt）

## 完整性检查（全部通过）

1. 五条 Episode ID 与 MANIFEST 完全匹配 ✓
2. SUCCESS / FAIL 与 10E journal 一致 ✓
3. 无黑帧 / 损坏帧（抽样解码均值检查）✓
4. 当前 UAV 位置与轨迹一致（直接绘制自保存状态）✓
5. 20 m 目标圆与目标位置正确 ✓
6. 动作概率和 = 1 ✓
7. 重算 argmax 与原始 action 一致（192/192）✓
8. 两条合集拼接顺序正确（时长 = 各条之和：48.0 / 63.6 s）✓
9. ffprobe 验证分辨率/FPS/时长/编码 ✓
10. Contact Sheet：`contact_sheet_5.png` ✓

## 复现

```bash
export CITYNAV_DATA=/mnt/hdd8t/Citynav-MEC/DATA
cd experiments/10_goal_query_mec_training/10E_final_closed_loop
python -I scripts/replay_10e.py                     # 概率重放（断点续跑）
python -I scripts/make_success_videos.py           # 逐条渲染（断点续跑）
# python -I scripts/make_success_videos.py --only val_seen:00960   # 单条
```
