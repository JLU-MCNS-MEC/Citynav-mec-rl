#!/usr/bin/env python3
"""V5 修正共享契约常量（纯 numpy，无 torch）。

训练（train_11b blank 合成）、推理（run_closed_loop 空候选分支）、
模型侧（model11c_fix）三方共用，空候选表示的训练/推理一致性由构造保证。
"""
from __future__ import annotations

import math

import numpy as np

# 空候选库表示（与 run_closed_loop 空分支一致；unc=50、since=99 为推理约定值）
EMPTY_BELIEF = np.asarray([0, 0, 0, 0, 0, 50, 0, 99, 0, 0], np.float32)
EMPTY_MATCH = np.zeros(5, np.float32)
EMPTY_LANDMARKS = np.zeros((8, 3), np.float32)

# 几何特征归一化尺度（固定常数：推理单样本确定、无 batch 依赖）
DIST_SCALE_M = 50.0     # 距离 ÷ 50 m
ANGLE_SCALE = math.pi   # 角度 ÷ π → [-1, 1]

# 训练侧空候选合成比例（每 epoch 确定性轮换 5% episode 全程 blank）
BLANK_FRAC = 0.05
