# 实验 01：俯视地图资产审计

- 日期：2026-10-29
- 状态：**完成**（只读审计）
- 依赖：无；实验 00 的归档已完成（Token 删除待确认，不影响本审计）
- 数据：`DATA/rgbd`（俯视 RGB PNG + 高程 TIF）、`DATA/original_block_ply`、
  `DATA/sbfnav-data/original_citynav`（只读 Val-Seen / Val-Unseen）
- 脚本：`scripts/audit_assets.py`；结果：`results/map_assets.csv`、`results/summary.json`
- 未读取：`test_unseen`

## 1. 官方约定（从 water-cookie/citynav 源码核对）

| 约定 | 官方实现 | 本实验采用的规则 |
|---|---|---|
| 栅格分辨率 | `rasterize.py`：PDAL `writers.gdal`，`resolution: 0.1` | 0.1 m/px |
| 高程图 | `z.tif`：`dimension=Z`、`output_type=max`、nodata `-9999`，事后把 nodata 填成 `min_z-1` | 逐像素最高表面高度 |
| RGB 图 | 红/绿/蓝三个 `mean` 栅格拼成 PNG（北向上） | 直接使用 |
| 固定高度 | `mturk_trajectory.fix_altitude(altitude_from_ground)`：`new_z = GROUND_LEVEL[map] + altitude_from_ground` | **z = GROUND_LEVEL + 50 m** |
| 局部观测 | `space.view_area_corners`：以 `altitude_from_ground = pose.z - ground_level` 为半边长、按 yaw 旋转的正方形；`cropclient` 用透视变换裁到图像 | 以 50 m 为半边长（视场 100 m × 100 m），按航向旋转到机头朝前 |
| 地图坐标 | TIF 无 CRS，像素坐标即米制世界坐标（`raster.index(x, y)` 直接用轨迹 x/y） | 沿用同一坐标，无需投影转换 |

关键结论：**“固定 50 m”指离地 50 m，不是世界坐标 z=50**。高程基准取自官方
`GROUND_LEVEL` 表（40 个条目，覆盖全部 34 张有栅格的地图）。

## 2. 资产盘点

| 资产 | 数量 | 大小 | 结论 |
|---|---:|---:|---|
| 俯视 RGB PNG | 34 | 0.60 GB | 每图与 TIF 同尺寸 |
| 高程 TIF | 34 | 2.98 GB | float64、nodata `-9999` |
| 俯视 RGBD 合计 | 34 对 | **3.59 GB** | 完整，可直接复用 |
| PLY 点云 | 28 | **28.32 GB** | 28 张开发地图全部有 PLY；缺的 6 张不在开发划分内 |
| CityNav 原始 JSON | 4 个开发 split | 4.3 GB（整个 sbfnav-data） | 只读 |

- 34 张地图全部通过：PNG/TIF 尺寸一致（`size_mismatch = []`）、分辨率均为
  `(0.1, -0.1)`、`GROUND_LEVEL` 全部可查。
- 开发划分 28 张地图的栅格资产**无缺失**（`dev_maps_missing_assets = []`）。
- 无 PLY 的 6 张图（`birmingham_block_0/6`、`cambridge_block_14/26/32/8`）都不在 28 张
  开发地图内，本实验不需要它们。
- PLY 顶点数：最少 235 万、中位 7,289 万、最多 1.37 亿。

### 2.1 坐标系与边界抽查

以 `birmingham_block_1` 为例（Val-Seen 第一条）：

| 项 | 值 |
|---|---|
| 栅格 bounds (m) | x 15.2–400.1，y 400.0–800.0 |
| 轨迹 x / y 范围 | 31.6–399.0 / 400.3–561.3 |
| 目标点 | (368.2, 440.8) |
| 结论 | 轨迹、目标均在栅格范围内，JSON 的 x/y 与栅格像素坐标同一坐标系 |

### 2.2 高程与空洞

- 有效高程范围：**1.84 m – 37.84 m**（各图 `z_min` 的最小/最大值）。
- nodata 比例：多数地图 20–30%，最高的 `cambridge_block_4` 为 **74.7%**
  （点云覆盖不到区域被填成 `min_z-1`，在局部裁剪时需按有效掩码处理）。
- 高程图是**地表最高面**（建筑屋顶等），不是裸地；`GROUND_LEVEL` 是官方给出的
  地图基准高度，两者不可混用。

## 3. 是否重新栅格化

**不需要。** 现有 `DATA/rgbd` 与官方 `rasterize.py` 的口径一致（0.1 m、max-Z、
mean-RGB、nodata 处理方式），且 34/34 张图完整。后续实验只读取这两类栅格，
不再从 PLY 逐步渲染三维点云，也不生成旧版第一视角图像。

唯一需要额外处理的是：局部裁剪时携带高程有效掩码（`z != nodata 且 z > min_z`），
避免把填充值当作地面。

## 4. 结论与下一步

1. 俯视资产完备：28 张开发地图 + 6 张非开发地图，3.59 GB，可直接用。
2. 固定高度必须按 `GROUND_LEVEL + 50 m` 计算；局部观测是以 50 m 为半边长的正方形。
3. 下一步（实验 02）：对 Train-Seen 人工轨迹做质量审计，产出高/中/低三组、
   可视化与筛选前后统计；Val-Seen / Val-Unseen 保持完整，不参与筛选。

## 5. 复现

```bash
python experiments/01_topdown_asset_audit/scripts/audit_assets.py \
  --mapdata /tmp/citynav-official/gsamllavanav/mapdata.py
```

结果文件：`results/map_assets.csv`（逐图 31 列）、`results/summary.json`。
