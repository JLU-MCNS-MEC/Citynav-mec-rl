#!/usr/bin/env python3
"""阶段 2：真实 CityNav 地图 MEC 部署数量分析（5/8/10/12/15/20）。

- 可通行区域 = MapRaster.validity（0.1m 高程 TIF 的 1m 有效掩码，含 nodata/
  最小值填充台地剔除，出处 pipeline.py MapRaster.load）。
- 任务集合 = Val-Seen 2,498 + Val-Unseen 2,826（06 manifest）+ Train
  18,521（11A teacher_episode_manifest.csv，start_target_m 为 "x1,y1;x2,y2"）。
- 均匀部署（用户指示：无历史记录则均匀部署）：每个地图在有效区包围盒内以
  step=sqrt(A_valid/K) 规则网格取点，过滤到有效像素，不足 K 时用更密网格补齐；
  最小间距 120m（mec-v8.2 NETWORK.min_spacing_m）。
- 通信时延 = mec-v8.2 latency_model.a2g_channel 公式（LOS 概率路径损耗 +
  Shannon 速率；载频 2GHz/GBS 10m/UAV 80m/带宽 5MHz/功率 0.1W/NF 9dB），
  payload 用 49×384×16bit 视觉 token（真实 CityNav 输入尺度）。
输出：results/deployment_analysis.json + 报告 markdown。
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
assert (REPO / "experiments").is_dir(), REPO
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline"
                       "/scripts"))
from pipeline import MapRaster, RGBD  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
RES.mkdir(parents=True, exist_ok=True)

# ---- mec-v8.2 常量（commit d988a9a, config.py NetworkConfig）----
GBS_HEIGHT_M = 10.0
UAV_ALTITUDE_M = 80.0
CARRIER_HZ = 2.0e9
BANDWIDTH_HZ = 5.0e6
TX_POWER_W = 0.1
NOISE_PSD_DBM_PER_HZ = -174.0
NOISE_FIGURE_DB = 9.0
LOS_A, LOS_B = 9.61, 0.16
EXCESS_LOS_DB, EXCESS_NLOS_DB = 1.0, 20.0
COVERAGE_RADIUS_M = 500.0
MIN_SPACING_M = 120.0
# 统一时延口径（2026-11-05 审计修正）：真实 CityNav 视觉 token =
# FrozenEncoder(ResNet50-layer4 池化) 49 token × 2048 维 × 16 bit。
# 旧值 49×384×16 是 mec-v8.2 ViT 配置的 token 维度，与本项目输入不符，
# 会使通信时延低估 5.33 倍；本脚本与 analyze_e2e.py 现统一为 2048 维。
VISUAL_TOKEN_BITS = 49 * 2048 * 16

K_LIST = [5, 8, 10, 12, 15, 20]
SPEED_OF_LIGHT = 299_792_458.0


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def a2g_rate_bps(horizontal_m: float) -> float:
    """mec-v8.2 latency_model.a2g_channel 的同款公式（上行速率）。"""
    h = max(float(horizontal_m), 0.0)
    vertical = abs(UAV_ALTITUDE_M - GBS_HEIGHT_M)
    d3 = max(math.sqrt(h * h + vertical * vertical), 1.0)
    elev = math.degrees(math.atan2(vertical, max(h, 1e-15)))
    p_los = 1.0 / (1.0 + LOS_A * math.exp(-LOS_B * (elev - LOS_A)))
    fsl = 20.0 * math.log10(4.0 * math.pi * CARRIER_HZ * d3 / SPEED_OF_LIGHT)
    path_loss = fsl + p_los * EXCESS_LOS_DB + (1 - p_los) * EXCESS_NLOS_DB
    noise_dbm = NOISE_PSD_DBM_PER_HZ + 10.0 * math.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB
    snr = TX_POWER_W * 10.0 ** (-path_loss / 10.0) / max(dbm_to_watt(noise_dbm), 1e-15)
    return max(BANDWIDTH_HZ * math.log2(1.0 + snr), 1e-15)


def load_tasks(task_set: str = "official") -> list[dict]:
    """任务集（审计项 4 修正）。

    official = Train 18,521（11A 质量审计后的训练池，RL 唯一训练数据）
             + Val-Seen 2,498（独立验证） = 21,019
    all      = 原始 Train 22,002（含 3,481 条低质量/教师失败，仅作部署规划参考）
             + Val-Seen 2,498 + Val-Unseen 2,826 = 27,326
    """
    mani = json.loads((REPO / "experiments/06_closed_loop_success_rate/results"
                       "/seed42/manifest.json").read_text())
    splits = ("val_seen",) if task_set == "official" else ("val_seen", "val_unseen")
    tasks = [{"split": t["split"], "map": t["map"],
              "start": [float(t["start_x"]), float(t["start_y"])],
              "target": [float(t["target_x"]), float(t["target_y"])]}
             for t in mani["tasks"] if t["split"] in splits]
    raw = json.loads(Path("/mnt/hdd2t/Citynav-MEC/DATA/sbfnav-data"
                          "/original_citynav/citynav_train_seen.json").read_text())
    if task_set == "official":
        # 11A 教师池索引已实证：manifest index 对应 raw[index]（18,521/18,521 匹配）
        rows = list(csv.DictReader(open(
            REPO / "experiments/11_teacher_geometry_ablation"
            "/11A_teacher_dataset_audit/results/teacher_episode_manifest.csv")))
        chosen = [raw[int(r["index"])] for r in rows]
        assert len(chosen) == 18521
    else:
        chosen = raw
    for e in chosen:
        traj = e["trajectory"]
        traj = json.loads(traj) if isinstance(traj, str) else traj
        tgt = e["target_positions"]
        tgt = json.loads(tgt) if isinstance(tgt, str) else tgt
        tasks.append({"split": "train",
                      "map": f"{e['area']}_block_{e['block']}",
                      "start": [traj[0][0], traj[0][1]],
                      "target": [tgt[0][0], tgt[0][1]]})
    return tasks


def uniform_deploy(mask: np.ndarray, left: float, bottom: float, res: float,
                   k: int) -> tuple[np.ndarray, float]:
    """规则网格均匀部署 k 个点（仅有效像素，间距优先 120m 逐步放宽）。

    返回 (positions (k,2), 实际采用的最小间距)。
    """
    ys, xs = np.nonzero(mask)
    if k == 0 or len(xs) == 0:
        return np.zeros((0, 2), np.float64), 0.0
    xmin = left + xs.min() * res
    xmax = left + xs.max() * res
    ymin = bottom + ys.min() * res
    ymax = bottom + ys.max() * res
    area = (xmax - xmin) * (ymax - ymin)
    step = math.sqrt(area / max(k, 1))
    spacings = (120.0, 90.0, 60.0, 30.0, 0.0)
    for mult in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2):
        s = step * mult
        gx = np.arange(xmin, xmax + s, s)
        gy = np.arange(ymin, ymax + s, s)
        gx, gy = np.meshgrid(gx, gy)
        cols = np.round((gx - left) / res).astype(int)
        rows = np.round((gy - bottom) / res).astype(int)
        ok = ((rows >= 0) & (rows < mask.shape[0])
              & (cols >= 0) & (cols < mask.shape[1]))
        pts = [(left + c * res, bottom + r * res)
               for r, c in zip(rows[ok].ravel(), cols[ok].ravel())
               if mask[r, c]]
        for sp in spacings:
            chosen = []
            for p in pts:
                if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= sp * sp
                       for q in chosen):
                    chosen.append(p)
                if len(chosen) == k:
                    return np.asarray(chosen), sp
    raise RuntimeError(f"无法放置 {k} 个点（有效区过小）")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-set", choices=["official", "all"], default="official")
    args = ap.parse_args()
    maps = sorted(p.stem for p in Path(RGBD).glob("*.png"))
    tasks = load_tasks(args.task_set)
    per_map_tasks: dict[str, list[np.ndarray]] = {}
    for t in tasks:
        per_map_tasks.setdefault(t["map"], []).append(
            np.asarray(t["start"][:2], np.float64))
    used_maps = sorted(set(per_map_tasks) & set(maps))
    print(f"maps={len(maps)} | 任务总数={len(tasks)} | 覆盖地图={len(used_maps)}")

    results = {"task_set": args.task_set, "tasks_total": len(tasks),
               "maps": len(maps),
               "note": ("official = 11A Train 18,521 + Val-Seen 2,498"
                        if args.task_set == "official" else
                        "all = raw Train 22,002 + Val-Seen 2,498 + Val-Unseen 2,826")}
    for k in K_LIST:
        cov_fracs, reach_s, reach_t, dists, n_avail, loads, lat_ms = [], [], [], [], [], [], []
        spacing_relaxed: list[bool] = []
        deployments = {}
        for m in used_maps:
            mr = MapRaster.load(m)
            pts, sp_used = uniform_deploy(mr.validity, mr.left, mr.bottom,
                                          mr.valid_scale * mr.res, k)
            deployments[m] = pts.tolist()
            spacing_relaxed.append(sp_used < MIN_SPACING_M)
            # 可通行区覆盖（1m 采样）
            ys, xs = np.nonzero(mr.validity)
            valid_xy = np.stack([mr.left + xs * mr.valid_scale * mr.res,
                                 mr.bottom + ys * mr.valid_scale * mr.res],
                                axis=1)
            sample = valid_xy[:: max(1, len(valid_xy) // 4000)]
            d_grid = np.linalg.norm(sample[:, None] - pts[None], axis=2)
            cov_fracs.append((d_grid.min(axis=1) <= COVERAGE_RADIUS_M).mean())
            # 任务可达性
            for t_xy in per_map_tasks.get(m, []):
                d = np.linalg.norm(pts - t_xy, axis=1)
                dn = d.min()
                dists.append(dn)
                reach_s.append(dn <= COVERAGE_RADIUS_M)
                n_avail.append(int((d <= COVERAGE_RADIUS_M).sum()))
                loads.append(int(np.argmin(d)))
                lat_ms.append(VISUAL_TOKEN_BITS / a2g_rate_bps(dn) * 1e3)
        dists = np.asarray(dists)
        loads = np.asarray(loads)
        lat = np.asarray(lat_ms)
        load_std = loads.std() if len(loads) else 0.0
        results[f"K{k}"] = {
            "coverage_frac_mean": float(np.mean(cov_fracs)),
            "task_reach_mean": float(np.mean(reach_s)),
            "task_reach_total": (int(np.sum(reach_s)), int(len(reach_s))),
            "dist_mean_m": float(dists.mean()), "dist_p95_m": float(np.percentile(dists, 95)),
            "n_available_mean": float(np.mean(n_avail)),
            "n_available_min": int(np.min(n_avail)) if len(n_avail) else -1,
            "nearest_mec_assign_std": float(load_std),
            "spacing_relaxed_maps": int(np.sum(spacing_relaxed)),
            "comm_latency_mean_ms": float(lat.mean()),
            "comm_latency_p95_ms": float(np.percentile(lat, 95)),
        }
        print(f"K={k:2d}: 覆盖 {results[f'K{k}']['coverage_frac_mean']:.3f} | "
              f"任务可达 {np.mean(reach_s):.3f} | 距离均值 {dists.mean():.0f}m "
              f"P95 {np.percentile(dists, 95):.0f}m | 可用数均值 {np.mean(n_avail):.1f} "
              f"最小 {np.min(n_avail)} | 负载σ {load_std:.1f} | 通信 P95 {np.percentile(lat, 95):.1f}ms")
        if k == 10:
            pos_name = ("mec_positions_k10.json" if args.task_set == "official"
                        else "mec_positions_k10_all_tasks.json")
            (RES / pos_name).write_text(json.dumps(deployments, indent=2))
    out_name = ("deployment_analysis.json" if args.task_set == "official"
                else "deployment_analysis_all_tasks.json")
    (RES / out_name).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved results/{out_name} + positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
