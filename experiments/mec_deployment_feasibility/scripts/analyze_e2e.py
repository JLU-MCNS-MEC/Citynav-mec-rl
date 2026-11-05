#!/usr/bin/env python3
"""阶段 5+6：通信/E2E 时延分析 + 三种 10-MEC 公平对照配置。

复用 mec-v8.2（commit d988a9a）语义：
- 通信：`latency_model.a2g_channel`（LOS 概率路径损耗 + Shannon 速率；
  载频 2GHz、GBS 10m、UAV 80m、带宽 5MHz、发射功率 0.1W、NF 9dB）
- 上行 payload：仅视觉 token（history 由 MEC 侧缓存，不重复上行，同 v8.2 语义），
  适配真实 CityNav 输入 = 49 token × 2048 维 × 16 bit
- 解析（推理）：T_inf = FLOPs / MEC 算力（阶段 4 运行期计数：
  Small 4.5M / Medium 55.1M / Transformer-M 556.0M）
- UAV 侧视觉编码：阶段 3 实测 1.68 ms（真实 CityNav 裁剪图 → 49×2048，
  FrozenEncoder）；同时给出 v8.2 解析式（UAV 235.8 GFLOPS）作为对照

配置（阶段 6，用户简报 §6）：
  A: 10×Small        B: 10×Transformer-M      C: 5×Small + 5×Transformer-M
  三配置同位置（阶段 2 均匀部署）、同带宽 5MHz、同发射功率 0.1W、同信道模型；
  主分析统一算力 0.2 TFLOPS/节点（人为建模假设，隔离"模型类型"效应），
  另给算力敏感性（每模型类型不同档位）。

输出：results/e2e_latency.json、results/deployment_configs.json、
      results/fig_e2e_breakdown.png、results/fig_share.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"

# ---- mec-v8.2 常量（config.py NetworkConfig, commit d988a9a）----
GBS_HEIGHT_M, UAV_ALTITUDE_M = 10.0, 80.0
CARRIER_HZ, BANDWIDTH_HZ, TX_POWER_W = 2.0e9, 5.0e6, 0.1
NOISE_PSD, NF_DB = -174.0, 9.0
LOS_A, LOS_B = 9.61, 0.16
EXCESS_LOS, EXCESS_NLOS = 1.0, 20.0
COVERAGE_RADIUS_M = 500.0
SPEED_OF_LIGHT = 299_792_458.0
PAYLOAD_BITS = 49 * 2048 * 16      # 真实 CityNav 视觉 token（无 depth 模态）

# ---- 阶段 3/4 结果 ----
ENCODE_MEASURED_MS = 1.680
FLOP = {"small": 4.5e6, "medium": 55.1e6, "transformer_m": 556.0e6}
CAP_MAIN = 0.2e12                 # 主分析统一算力（人为建模假设）


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def uplink_rate_bps(horizontal_m: float) -> float:
    h = max(float(horizontal_m), 0.0)
    v = abs(UAV_ALTITUDE_M - GBS_HEIGHT_M)
    d3 = max(math.sqrt(h * h + v * v), 1.0)
    elev = math.degrees(math.atan2(v, max(h, 1e-15)))
    p_los = 1.0 / (1.0 + LOS_A * math.exp(-LOS_B * (elev - LOS_A)))
    fsl = 20.0 * math.log10(4.0 * math.pi * CARRIER_HZ * d3 / SPEED_OF_LIGHT)
    pl = fsl + p_los * EXCESS_LOS + (1.0 - p_los) * EXCESS_NLOS
    noise = dbm_to_watt(NOISE_PSD + 10.0 * math.log10(BANDWIDTH_HZ) + NF_DB)
    snr = TX_POWER_W * 10.0 ** (-pl / 10.0) / max(noise, 1e-15)
    return max(BANDWIDTH_HZ * math.log2(1.0 + snr), 1e-15)


def load_tasks() -> list[dict]:
    mani = json.loads((REPO / "experiments/06_closed_loop_success_rate/results"
                       "/seed42/manifest.json").read_text())
    tasks = [{"split": t["split"], "map": t["map"],
              "start": [float(t["start_x"]), float(t["start_y"])]}
             for t in mani["tasks"] if t["split"] in ("val_seen", "val_unseen")]
    train = json.loads(Path("/mnt/hdd2t/Citynav-MEC/DATA/sbfnav-data"
                            "/original_citynav/citynav_train_seen.json").read_text())
    for e in train:
        traj = e["trajectory"]
        traj = json.loads(traj) if isinstance(traj, str) else traj
        tasks.append({"split": "train",
                      "map": f"{e['area']}_block_{e['block']}",
                      "start": [traj[0][0], traj[0][1]]})
    return tasks


def build_configs(positions: dict[str, list[list[float]]]) -> dict:
    """三种配置：同位置/同带宽/同功率，仅模型类型与算力不同。"""
    def nodes(kind: str) -> list[dict]:
        out = []
        for m, pts in positions.items():
            for i, p in enumerate(pts):
                if kind == "A":
                    mt, cap = "small", CAP_MAIN
                elif kind == "B":
                    mt, cap = "transformer_m", CAP_MAIN
                else:  # C: 5 Small + 5 TM，按索引交替
                    mt = "small" if i % 2 == 0 else "transformer_m"
                    cap = CAP_MAIN
                out.append({"map": m, "gbs_id": f"G{i+1}", "x": p[0], "y": p[1],
                            "model_type": mt,
                            "compute_capacity_flops_per_s": cap,
                            "bandwidth_hz": BANDWIDTH_HZ,
                            "tx_power_w": TX_POWER_W,
                            "coverage_radius_m": COVERAGE_RADIUS_M})
        return out

    return {
        "protocol": {
            "source": "mec-v8.2 latency_model.py / config.py (commit d988a9a)",
            "positions": "phase2 uniform deployment (mec_positions_k10.json)",
            "bandwidth_hz": BANDWIDTH_HZ, "tx_power_w": TX_POWER_W,
            "compute_capacity_note": "0.2 TFLOPS 为人为建模假设（历史出处不可追溯）",
            "payload_bits": PAYLOAD_BITS,
            "task_sets": "Train 22,002 + Val-Seen 2,498 + Val-Unseen 2,826",
        },
        "A_10xSmall": nodes("A"),
        "B_10xTransformerM": nodes("B"),
        "C_5Small_5TM": nodes("C"),
    }


def summarize_sec(arr: np.ndarray) -> dict:
    """输入秒，输出毫秒统计。"""
    return {"mean_ms": float(arr.mean()) * 1e3,
            "p50_ms": float(np.percentile(arr, 50)) * 1e3,
            "p95_ms": float(np.percentile(arr, 95)) * 1e3,
            "p99_ms": float(np.percentile(arr, 99)) * 1e3}


def summarize_ms(arr_ms: np.ndarray) -> dict:
    """输入已是毫秒。"""
    return {"mean_ms": float(arr_ms.mean()),
            "p50_ms": float(np.percentile(arr_ms, 50)),
            "p95_ms": float(np.percentile(arr_ms, 95)),
            "p99_ms": float(np.percentile(arr_ms, 99))}


def main() -> int:
    positions = json.loads((RES / "mec_positions_k10.json").read_text())
    configs = build_configs(positions)
    (RES / "deployment_configs.json").write_text(
        json.dumps(configs, indent=1, ensure_ascii=False))

    tasks = load_tasks()
    print(f"tasks={len(tasks)}")

    # 每任务最近 MEC 距离（同位置，三配置相同）
    per_map_pos = {m: np.asarray(p, np.float64) for m, p in positions.items()}
    dists = []
    for t in tasks:
        pts = per_map_pos[t["map"]]
        d = np.linalg.norm(pts - np.asarray(t["start"], np.float64), axis=1)
        dists.append(d.min())
    dists = np.asarray(dists)

    rate = np.array([uplink_rate_bps(d) for d in dists])
    comm_s = PAYLOAD_BITS / rate
    encode_ms = np.full(len(tasks), ENCODE_MEASURED_MS)

    out = {"n_tasks": len(tasks),
           "encode_measured_ms": ENCODE_MEASURED_MS,
           "distance": summarize_sec(dists),
           "comm": summarize_sec(comm_s),
           "configs": {}}
    print("comm:", out["comm"])

    for cfg_name, kind in [("A_10xSmall", "small"),
                           ("B_10xTransformerM", "transformer_m"),
                           ("C_5Small_5TM", None)]:
        if kind is not None:
            inf_ms = np.full(len(tasks), FLOP[kind] / CAP_MAIN * 1e3)
            label = kind
        else:
            # 5+5：最近 MEC 的模型类型决定；按最近点索引奇偶（与 build_configs 一致）
            inf_ms = np.empty(len(tasks))
            for i, t in enumerate(tasks):
                pts = per_map_pos[t["map"]]
                d = np.linalg.norm(pts - np.asarray(t["start"], np.float64), axis=1)
                idx = int(np.argmin(d))
                mt = "small" if idx % 2 == 0 else "transformer_m"
                inf_ms[i] = FLOP[mt] / CAP_MAIN * 1e3
            label = "mixed"
        e2e_ms = encode_ms + comm_s * 1e3 + inf_ms
        out["configs"][cfg_name] = {
            "model_mix": label,
            "inference": summarize_ms(inf_ms),
            "e2e": summarize_ms(e2e_ms),
            "inference_share_mean": float((inf_ms / e2e_ms).mean()),
            "communication_share_mean": float((comm_s * 1e3 / e2e_ms).mean()),
            "encode_share_mean": float((encode_ms / e2e_ms).mean()),
        }
        print(cfg_name, "inference:", out["configs"][cfg_name]["inference"],
              "share(inf/comm/enc)=%.3f/%.3f/%.3f" % (
                  out["configs"][cfg_name]["inference_share_mean"],
                  out["configs"][cfg_name]["communication_share_mean"],
                  out["configs"][cfg_name]["encode_share_mean"]))

    # 模型间差异占 E2E 的比例
    dS = FLOP["small"] / CAP_MAIN * 1e3
    dTM = FLOP["transformer_m"] / CAP_MAIN * 1e3
    dM = FLOP["medium"] / CAP_MAIN * 1e3
    e2eS = encode_ms + comm_s * 1e3 + dS
    e2eTM = encode_ms + comm_s * 1e3 + dTM
    out["deltas"] = {
        "small_to_tm_inf_ms": dTM - dS,
        "medium_to_tm_inf_ms": dTM - dM,
        "small_to_tm_e2e_share": float(((dTM - dS) / e2eTM).mean()),
        "medium_to_tm_e2e_share": float(((dTM - dM) / e2eTM).mean()),
    }
    print("deltas:", out["deltas"])

    # 算力敏感性：不同算力档下 TM 的推理/E2E
    out["capacity_sensitivity"] = {}
    for cap in (0.1e12, 0.2e12, 0.3e12, 0.5e12, 1.0e12):
        row = {}
        for name in ("small", "medium", "transformer_m"):
            inf = FLOP[name] / cap * 1e3
            e2e = encode_ms + comm_s * 1e3 + inf
            row[name] = {"inference_mean_ms": float(inf),
                         "e2e_mean_ms": float(e2e.mean()),
                         "inference_share_mean": float((inf / e2e).mean())}
        out["capacity_sensitivity"][f"{cap/1e12:.1f} TFLOPS"] = row

    (RES / "e2e_latency.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    # ---- 图：E2E 分段（A/B/C）----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["A_10xSmall", "B_10xTransformerM", "C_5Small_5TM"]
    labels = ["A: 10×Small", "B: 10×Transformer-M", "C: 5+5"]
    enc_m = out["encode_measured_ms"]
    com_m = out["comm"]["mean_ms"]
    inf_m = [out["configs"][n]["inference"]["mean_ms"] for n in names]
    ax.bar(labels, [enc_m] * 3, label="UAV encode (measured)")
    ax.bar(labels, [com_m] * 3, bottom=[enc_m] * 3, label="uplink comm")
    bottom = [enc_m + com_m] * 3
    ax.bar(labels, inf_m, bottom=bottom, label="MEC inference @0.2 TFLOPS")
    ax.set_ylabel("E2E latency mean (ms)")
    ax.set_title("E2E latency breakdown, 10-MEC uniform deployment")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RES / "fig_e2e_breakdown.png", dpi=150)

    # ---- 图：占比随算力 ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    caps = [c / 1e12 for c in (0.1e12, 0.2e12, 0.3e12, 0.5e12, 1.0e12)]
    for name, label in [("small", "Small"), ("medium", "Medium"),
                        ("transformer_m", "Transformer-M")]:
        ys = [out["capacity_sensitivity"][f"{c} TFLOPS"][name]
              ["inference_share_mean"] * 100 for c in caps]
        ax.plot(caps, ys, marker="o", label=label)
    ax.set_xlabel("MEC compute capacity (TFLOPS, modeling assumption)")
    ax.set_ylabel("Inference share of E2E (%)")
    ax.set_title("Inference share vs MEC capacity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "fig_share.png", dpi=150)
    print("saved e2e_latency.json + deployment_configs.json + 2 figs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
