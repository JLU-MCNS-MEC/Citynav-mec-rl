"""Protocol invariants for Stage 12 (no GPU, no data access)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402


def test_deployments_are_10_nodes_per_map():
    for g in P.GROUPS:
        dep = P.deployment(g)
        assert len(dep) == 24
        for nodes in dep.values():
            assert len(nodes) == 10
            assert [n["gbs_id"] for n in nodes] == [f"G{i}" for i in range(1, 11)]


def test_two_model_placement_alternates():
    dep = P.deployment("two_homo")
    for nodes in dep.values():
        assert [n["model_type"] for n in nodes] == ["small", "transformer_m"] * 5
        assert sum(n["model_type"] == "small" for n in nodes) == 5


def test_five_model_equal_placement():
    dep = P.deployment("five_hetero")
    for nodes in dep.values():
        counts = {m: sum(n["model_type"] == m for n in nodes)
                  for m in P.MODEL_ORDER}
        assert counts == {m: 2 for m in P.MODEL_ORDER}


def test_homogeneous_tiers_are_10():
    for g in P.HOMO_GROUPS:
        for nodes in P.deployment(g).values():
            assert all(n["compute_tflops"] == 10.0 for n in nodes)


def test_heterogeneous_tiers_use_all_four_levels():
    for g in P.HETERO_GROUPS:
        seen = set()
        for nodes in P.deployment(g).values():
            seen |= {n["compute_tflops"] for n in nodes}
        assert seen == {5.0, 10.0, 20.0, 30.0}


def test_model_and_tier_are_balanced_in_two_model_hetero():
    dep = P.deployment("two_hetero")
    by_model = {m: [] for m in ("small", "transformer_m")}
    for nodes in dep.values():
        for n in nodes:
            by_model[n["model_type"]].append(n["compute_tflops"])
    assert sorted(by_model["small"][:5]) == sorted(by_model["transformer_m"][:5])
    assert sorted(by_model["small"][:5]) == [5.0, 10.0, 10.0, 20.0, 30.0]


def test_inference_latency_uses_frozen_gflops():
    node = {"x": 0.0, "y": 0.0, "model_type": "transformer_m",
            "compute_capacity_flops_per_s": 10e12,
            "coverage_radius_m": 500.0}
    m = P.node_metrics(node, np.zeros(2), enc_ms=0.0)
    assert m["inf_ms"] == pytest.approx(10.0)
    node["compute_capacity_flops_per_s"] = 5e12
    assert P.node_metrics(node, np.zeros(2), enc_ms=0.0)["inf_ms"] == \
        pytest.approx(20.0)


def test_communication_decreases_with_distance():
    near = P.comm_latency_ms(20.0)
    far = P.comm_latency_ms(400.0)
    assert far > near > 0


def test_encoder_latency_is_measured_value():
    # measure_encoder_flops.py must have been run
    assert P.ENCODER_FLOPS_PATH.exists()
    assert P.encode_ms() == pytest.approx(P.encoder_flops() / 1.5e12 * 1e3)


def test_max_quality_prefers_transformer_m():
    dep = P.deployment("two_homo")["birmingham_block_1"]
    pos = np.asarray([dep[0]["x"], dep[0]["y"]], np.float64)
    node = P.select_node(dep, pos, "max_quality", np.random.default_rng(0),
                         enc_ms=0.0)
    assert node["model_type"] == "transformer_m"


def test_max_rate_matches_nearest_when_channel_is_monotone():
    dep = P.deployment("two_homo")["birmingham_block_1"]
    pos = np.asarray([300.0, 500.0], np.float64)
    rng = np.random.default_rng(0)
    r = P.select_node(dep, pos, "max_rate", rng, enc_ms=0.0)
    nearest = min(dep, key=lambda n: math.hypot(n["x"] - pos[0],
                                                n["y"] - pos[1]))
    assert r["gbs_id"] == nearest["gbs_id"]


def test_state_layout_is_194():
    assert P.STATE_DIM == 4 + 10 * 19
    assert P.N_ACT == 10
