#!/usr/bin/env python3
"""Stage 12 frozen protocol: heterogeneous MEC compute + association policies.

Scope of this stage (2026-11-16):
- Navigation experts are the five frozen Stage-11 models.  No navigation model is
  trained, fine-tuned or modified here.
- Only the *hosted task compute* and the *node compute capacity* change.
- Association selects one MEC node per decision step; the node's hosted expert
  produces the navigation action, so different associations follow different
  real closed-loop trajectories.

Frozen numbers (user-provided, treated as modelling assumptions, not measurements):
  Small 5, Medium 15, Transformer-S 40, Transformer-M 100, Large 200 GFLOPs.
  UAV visual encoding: measured encoder FLOPs / 1.5 TFLOPS.
  MEC: homogeneous 10 TFLOPS, or heterogeneous 5/10/20/30 TFLOPS tiers.
  Communication: unchanged from the CityNav / mec-v8.2 model
  (payload 49 x 2048 x 16 bit, 5 MHz, 0.1 W, 2 GHz, 500 m coverage).

All deployment tables are pre-registered in this file.  Nothing here reads
Val-Seen statistics.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

# The canonical data root used by every Stage-04..11 script on this host.
os.environ.setdefault("CITYNAV_DATA", "/home/20261024_1/Workspace/DATA")

HERE = Path(__file__).resolve().parent
EXP = HERE.parent                      # experiments/12_hetero_mec_association
REPO = EXP.parents[1]
RES = EXP / "results"
PROTOCOL_DIR = RES / "protocol"

MEC_RES = REPO / "experiments/mec_deployment_feasibility/results"
POSITIONS_PATH = MEC_RES / "mec_positions_k10.json"
CKPT_BASE = (REPO / "experiments/11_teacher_geometry_ablation"
             "/11B_transformer_m_v4/results")

# ---------------------------------------------------------------------------
# Experts
# ---------------------------------------------------------------------------
MODEL_ORDER = ("small", "medium", "transformer_s", "transformer_m", "large")
MODEL_LABEL = {
    "small": "Small",
    "medium": "Medium",
    "transformer_s": "Transformer-S",
    "transformer_m": "Transformer-M",
    "large": "Large",
}
#: run_closed_loop.build_model names
RUNNER_MODEL = {
    "small": "small",
    "medium": "medium",
    "transformer_s": "ts",
    "transformer_m": "v5fix",
    "large": "large",
}
CKPTS = {
    "small": "runs_small/seed42/best.pt",
    "medium": "runs_medium/seed42/best.pt",
    "transformer_s": "runs_transformer_s/seed42/best.pt",
    "transformer_m": "runs_11c_v5fix/seed42/best.pt",
    "large": "runs_large/seed42/best.pt",
}
#: user-frozen latency-modelling compute (GFLOPs per decision)
MODEL_GFLOPS = {
    "small": 5.0,
    "medium": 15.0,
    "transformer_s": 40.0,
    "transformer_m": 100.0,
    "large": 200.0,
}
#: leak-free quality prior (training-side Val Macro-F1, Stage-11 audit item 3)
QUALITY_MACRO_F1 = {
    "small": 0.7243,
    "medium": 0.7103,
    "transformer_s": 0.7258,
    "transformer_m": 0.7475,
    "large": 0.7340,
}

# ---------------------------------------------------------------------------
# Deployments: 10 nodes per map, by gbs_id G1..G10
# ---------------------------------------------------------------------------
TWO_MODELS = ["small", "transformer_m"] * 5
FIVE_MODELS = list(MODEL_ORDER) * 2
HOMO = [10.0] * 10

GROUP_SPECS = {
    "two_homo": {
        "label": "2-model mixed | homogeneous 10 TFLOPS",
        "models": TWO_MODELS,
        "tflops": HOMO,
    },
    "two_hetero": {
        "label": "2-model mixed | heterogeneous 5/10/20/30 TFLOPS",
        "models": TWO_MODELS,
        "tflops": [5.0, 10.0, 20.0, 30.0, 10.0,
                   5.0, 30.0, 20.0, 10.0, 10.0],
    },
    "five_homo": {
        "label": "5-model equal | homogeneous 10 TFLOPS",
        "models": FIVE_MODELS,
        "tflops": HOMO,
    },
    "five_hetero": {
        "label": "5-model equal | heterogeneous 5/10/20/30 TFLOPS",
        "models": FIVE_MODELS,
        "tflops": [5.0, 10.0, 20.0, 30.0, 10.0,
                   30.0, 20.0, 10.0, 5.0, 20.0],
    },
}
GROUPS = tuple(GROUP_SPECS)
HOMO_GROUPS = ("two_homo", "five_homo")
HETERO_GROUPS = ("two_hetero", "five_hetero")

BANDWIDTH_HZ = 5.0e6
TX_POWER_W = 0.1
COVERAGE_RADIUS_M = 500.0

# ---------------------------------------------------------------------------
# Communication model (identical to mec-v8.2 / Stage-11 feasibility)
# ---------------------------------------------------------------------------
PAYLOAD_BITS = 49 * 2048 * 16          # 1,605,632 bit per decision
GBS_H, UAV_H = 10.0, 80.0
CARRIER, BW_HZ, TXP = 2.0e9, BANDWIDTH_HZ, TX_POWER_W
NOISE_PSD, NF = -174.0, 9.0
LOS_A, LOS_B, EXC_LOS, EXC_NLOS = 9.61, 0.16, 1.0, 20.0
C_LIGHT = 299_792_458.0

# ---------------------------------------------------------------------------
# UAV encoding (measured once by measure_encoder_flops.py)
# ---------------------------------------------------------------------------
UAV_TFLOPS = 1.5e12
ENCODER_FLOPS_PATH = PROTOCOL_DIR / "encoder_flops.json"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
STOP_RADIUS = 20.0
FORWARD_M = 5.0
TURN_RAD = math.radians(30.0)
MAX_STEPS = 200
EVAL_SEED = 42
MANIFEST = (REPO / "experiments/06_closed_loop_success_rate/results"
            "/seed42/manifest.json")
MANIFEST_SHA = "2da3781677e1376c7ff37ee029956c3a8d54aa0a85f81feca5858fe611123446"
TRAIN_MANIFEST = (REPO / "experiments/11_teacher_geometry_ablation"
                  "/11A_teacher_dataset_audit/results/teacher_episode_manifest.csv")
RAW_TRAIN = Path("/mnt/hdd2t/Citynav-MEC/DATA/sbfnav-data/original_citynav"
                 "/citynav_train_seen.json")

# ---------------------------------------------------------------------------
# RL state scaling (re-calibrated for the new compute range)
# ---------------------------------------------------------------------------
TAU_MS, H_MAX = 100.0, 400
POS_SCALE = 500.0
COMPUTE_SCALE = 30.0e12
BW_SCALE = 30.0e6
RATE_SCALE = 200.0e6
LAT_SCALE = 100.0
INF_SCALE = 25.0
E2E_CLIP = 1.0
SNR_SCALE = 40.0
STATE_DIM = 4 + 10 * 19
N_ACT = 10


# ---------------------------------------------------------------------------
# Communication helpers
# ---------------------------------------------------------------------------
def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def uplink_rate_bps(h_m: float) -> tuple[float, float]:
    """Return (uplink_rate_bps, linear_SNR) for horizontal distance h_m."""
    h = max(float(h_m), 0.0)
    v = abs(UAV_H - GBS_H)
    d3 = max(math.sqrt(h * h + v * v), 1.0)
    elev = math.degrees(math.atan2(v, max(h, 1e-15)))
    p_los = 1.0 / (1.0 + LOS_A * math.exp(-LOS_B * (elev - LOS_A)))
    fsl = 20.0 * math.log10(4.0 * math.pi * CARRIER * d3 / C_LIGHT)
    pl = fsl + p_los * EXC_LOS + (1 - p_los) * EXC_NLOS
    noise = dbm_to_watt(NOISE_PSD + 10.0 * math.log10(BW_HZ) + NF)
    snr = TXP * 10.0 ** (-pl / 10.0) / max(noise, 1e-15)
    return max(BW_HZ * math.log2(1.0 + snr), 1e-15), snr


def comm_latency_ms(distance_m: float) -> float:
    rate, _ = uplink_rate_bps(distance_m)
    return PAYLOAD_BITS / rate * 1e3


# ---------------------------------------------------------------------------
# Encoding latency
# ---------------------------------------------------------------------------
def encoder_flops() -> float:
    if not ENCODER_FLOPS_PATH.exists():
        raise FileNotFoundError(
            f"missing {ENCODER_FLOPS_PATH}; run measure_encoder_flops.py first")
    rec = json.loads(ENCODER_FLOPS_PATH.read_text(encoding="utf-8"))
    return float(rec["flops"])


def encode_ms() -> float:
    return encoder_flops() / UAV_TFLOPS * 1e3


# ---------------------------------------------------------------------------
# Deployment construction
# ---------------------------------------------------------------------------
def load_positions() -> dict[str, list[list[float]]]:
    return json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))


def deployment(group: str) -> dict[str, list[dict]]:
    """Return {map: [10 nodes]} for one frozen group."""
    spec = GROUP_SPECS[group]
    positions = load_positions()
    out: dict[str, list[dict]] = {}
    for name, xy in positions.items():
        if len(xy) != 10:
            raise ValueError(f"{name}: expected 10 positions, got {len(xy)}")
        nodes = []
        for i, (x, y) in enumerate(xy):
            nodes.append({
                "map": name,
                "gbs_id": f"G{i + 1}",
                "x": float(x),
                "y": float(y),
                "model_type": spec["models"][i],
                "compute_capacity_flops_per_s": spec["tflops"][i] * 1e12,
                "compute_tflops": float(spec["tflops"][i]),
                "bandwidth_hz": BANDWIDTH_HZ,
                "tx_power_w": TX_POWER_W,
                "coverage_radius_m": COVERAGE_RADIUS_M,
            })
        out[name] = nodes
    return out


def flat_deployment(group: str) -> list[dict]:
    return [n for nodes in deployment(group).values() for n in nodes]


# ---------------------------------------------------------------------------
# Per-node metrics
# ---------------------------------------------------------------------------
def node_metrics(node: dict, pos: np.ndarray, enc_ms: float | None = None) -> dict:
    d = float(np.linalg.norm(np.asarray([node["x"], node["y"]], np.float64)
                             - np.asarray(pos, np.float64)))
    rate, snr = uplink_rate_bps(d)
    comm = PAYLOAD_BITS / rate * 1e3
    inf = MODEL_GFLOPS[node["model_type"]] * 1e9 / max(
        float(node["compute_capacity_flops_per_s"]), 1.0) * 1e3
    enc = encode_ms() if enc_ms is None else float(enc_ms)
    return {
        "distance_m": d,
        "rate_bps": rate,
        "snr": snr,
        "comm_ms": comm,
        "inf_ms": inf,
        "encode_ms": enc,
        "e2e_ms": enc + comm + inf,
        "available": d <= float(node["coverage_radius_m"]),
    }


def available_nodes(nodes: list[dict], pos: np.ndarray) -> list[dict]:
    avail = [n for n in nodes
             if np.linalg.norm(np.asarray([n["x"], n["y"]], np.float64)
                               - np.asarray(pos, np.float64))
             <= float(n["coverage_radius_m"])]
    if avail:
        return avail
    # Fallback: the single nearest node (Stage-11 minimum was 8 available).
    return [min(nodes, key=lambda n: np.linalg.norm(
        np.asarray([n["x"], n["y"]], np.float64) - np.asarray(pos, np.float64)))]


# ---------------------------------------------------------------------------
# Static association strategies (Oracle lives in oracle_teacher.py)
# ---------------------------------------------------------------------------
def select_node(nodes, pos, strategy, rng, quality=None, enc_ms=None):
    quality = QUALITY_MACRO_F1 if quality is None else quality
    avail = available_nodes(nodes, pos)
    if strategy == "random":
        return avail[int(rng.integers(0, len(avail)))]
    if strategy == "max_rate":
        return max(avail, key=lambda n: uplink_rate_bps(node_metrics(
            n, pos, enc_ms)["distance_m"])[0])
    if strategy == "max_quality":
        return max(avail, key=lambda n: (
            quality[n["model_type"]],
            -node_metrics(n, pos, enc_ms)["e2e_ms"]))
    raise ValueError(f"unknown static strategy {strategy}")


STRATEGIES = ("random", "max_rate", "max_quality", "oracle")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def group_dir(group: str) -> Path:
    if group not in GROUP_SPECS:
        raise ValueError(f"unknown group {group}; choices {GROUPS}")
    return RES / group


def ckpt_path(model_type: str) -> Path:
    return CKPT_BASE / CKPTS[model_type]


def eval_tasks(limit: int = 0) -> list[dict]:
    mani = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = [t for t in mani["tasks"] if t["split"] == "val_seen"]
    if limit:
        tasks = tasks[:limit]
    return tasks


__all__ = [
    "GROUP_SPECS", "GROUPS", "HOMO_GROUPS", "HETERO_GROUPS", "MODEL_ORDER",
    "MODEL_LABEL", "MODEL_GFLOPS", "QUALITY_MACRO_F1", "CKPTS", "RUNNER_MODEL",
    "STRATEGIES", "PAYLOAD_BITS", "STOP_RADIUS", "MAX_STEPS", "EVAL_SEED",
    "MANIFEST", "MANIFEST_SHA", "TRAIN_MANIFEST", "RAW_TRAIN",
    "TAU_MS", "H_MAX", "POS_SCALE", "COMPUTE_SCALE", "BW_SCALE",
    "RATE_SCALE", "LAT_SCALE", "INF_SCALE", "E2E_CLIP", "SNR_SCALE",
    "STATE_DIM", "N_ACT", "UAV_TFLOPS",
    "uplink_rate_bps", "comm_latency_ms", "encoder_flops", "encode_ms",
    "deployment", "flat_deployment", "node_metrics", "available_nodes",
    "select_node", "group_dir", "ckpt_path", "eval_tasks",
    "PROTOCOL_DIR", "RES", "REPO", "EXP", "CKPT_BASE",
]
