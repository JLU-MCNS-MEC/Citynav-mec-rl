# CityNav 视觉编码器四组消融（2026-10-28）

- 协议：[`PROTOCOL.md`](PROTOCOL.md)
- 配置：[`configs/e1.yaml`](configs/e1.yaml) … `e4.yaml`
- 审计脚本：`audit_labels.py`（只读）
- 结果：`results/`，报告：`reports/`

## 运行

```bash
cd /home/20261024_1/Workspace/Citynav-mec-rl
export CITYNAV_DATA_ROOT=/home/20261024_1/Workspace/DATA/sbfnav-data
export CITYNAV_PLY_ROOT=/home/20261024_1/Workspace/DATA/original_block_ply

# E1 标签审计（不训练）
python experiments/visual_encoder_citynav/ablations/20261028/audit_labels.py \
  --data-root "$CITYNAV_DATA_ROOT" --ply-root "$CITYNAV_PLY_ROOT" \
  --split val_seen --out experiments/visual_encoder_citynav/ablations/20261028/results/e1_audit_val_seen_full.json

# Smoke（20 步）
python experiments/visual_encoder_citynav/train.py --smoke \
  --config experiments/visual_encoder_citynav/ablations/20261028/configs/e1.yaml \
  --run-name abl_e1_smoke

# 正式训练（2751 步）
python experiments/visual_encoder_citynav/train.py \
  --config experiments/visual_encoder_citynav/ablations/20261028/configs/e1.yaml \
  --run-name abl_20261028_e1_seed42
```

E2/E4 首次运行会重建 `max_points=8M` 的点云缓存；E3/E4 首次运行会生成
`*_s3` 的多步 manifest。两者都写在 `data_cache/`，不入库。

## 约束

- 原始 JSON / CityRefer / RGB / TIFF / PLY 只读，不复制、不改写。
- Test-Unseen 封存。
- 权重、缓存、完整日志不提交 GitHub。
