# 本目录结果标记为 historical / incomplete

- 原因：Stage 11D 的闭环脚本沿用了 Stage 10E harness，存在两处训练/推理输入不一致：
  1. 历史 token 顺序反转（推理槽 0 = 最旧，训练槽 0 = 最近）
  2. 位姿编码不一致（推理用绝对 x,y,yaw，训练用当前朝向相对位姿）
- 2026-11-03 起正式结果以修正版为准：
  - experiments/v4_closed_loop_revalidation/（V4 修正版闭环）
  - experiments/v5_closed_loop_evaluation/（V5 修正版闭环）
  - reports/v4_v5_comparison/（对比报告）
- 本目录内容保留作历史对照，不得作为正式指标。
