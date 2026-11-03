# CHANGELOG

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
