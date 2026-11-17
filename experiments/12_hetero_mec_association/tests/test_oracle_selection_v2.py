"""Unit tests for the Oracle v2 rule: action agreement only.

These tests exercise the pure selector; no model, data or GPU is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(HERE))
from oracle_selection import Candidate, select_oracle_action_agreement  # noqa: E402


def node(gid, **extra):
    return {"gbs_id": gid, "model_type": "small", **extra}


def cand(gid, probs, **extra):
    return Candidate(node(gid, **extra), probs)


def test_multiple_exact_agreements_pick_highest_probability():
    # teacher action 1; G1 and G2 both predict 1, G2 has higher P(1)
    out = select_oracle_action_agreement([
        cand("G1", [0.1, 0.6, 0.2, 0.1]),
        cand("G2", [0.05, 0.8, 0.10, 0.05]),
        cand("G3", [0.1, 0.1, 0.7, 0.1]),
    ], teacher_action=1)
    assert out["node"]["gbs_id"] == "G2"
    assert out["selected_agreement"] == 1
    assert out["selected_action"] == 1
    assert out["selected_teacher_action_probability"] == pytest.approx(0.8)
    assert out["exact_agreement_candidate_count"] == 2


def test_no_exact_agreement_pick_highest_probability():
    # teacher 1; no expert predicts 1; G2 has higher P(1)
    out = select_oracle_action_agreement([
        cand("G1", [0.05, 0.35, 0.55, 0.05]),
        cand("G2", [0.05, 0.45, 0.48, 0.02]),
        cand("G3", [0.05, 0.25, 0.65, 0.05]),
    ], teacher_action=1)
    assert out["node"]["gbs_id"] == "G2"
    assert out["selected_agreement"] == 0
    assert out["selected_action"] == 2
    assert out["selected_teacher_action_probability"] == pytest.approx(0.45)
    assert out["exact_agreement_candidate_count"] == 0


def test_probability_tie_uses_fixed_node_order_exact_agreement():
    # G4 and G2 both predict 1 with P(1)=0.7 -> G2 (lower index)
    out = select_oracle_action_agreement([
        cand("G4", [0.05, 0.7, 0.2, 0.05]),
        cand("G2", [0.05, 0.7, 0.2, 0.05]),
        cand("G7", [0.05, 0.7, 0.2, 0.05]),
    ], teacher_action=1)
    assert out["node"]["gbs_id"] == "G2"
    assert out["tie_break_used"] is True


def test_probability_tie_uses_fixed_node_order_fallback():
    # teacher 1; none predict 1; G5/G3 both have P(1)=0.5 -> G3
    out = select_oracle_action_agreement([
        cand("G5", [0.05, 0.45, 0.48, 0.02]),
        cand("G3", [0.05, 0.45, 0.48, 0.02]),
        cand("G1", [0.05, 0.35, 0.55, 0.05]),
    ], teacher_action=1)
    assert out["node"]["gbs_id"] == "G3"
    assert out["selected_agreement"] == 0
    assert out["tie_break_used"] is True


def test_latency_and_quality_fields_do_not_affect_selection():
    base = [
        cand("G1", [0.1, 0.6, 0.2, 0.1], e2e_ms=10.0, comm_ms=1.0,
             inf_ms=2.0, distance_m=5.0, quality=0.1),
        cand("G2", [0.05, 0.8, 0.10, 0.05], e2e_ms=100.0, comm_ms=50.0,
             inf_ms=20.0, distance_m=400.0, quality=0.9),
        cand("G3", [0.1, 0.1, 0.7, 0.1], e2e_ms=1.0, comm_ms=0.1,
             inf_ms=0.1, distance_m=1.0, quality=0.99),
    ]
    out = select_oracle_action_agreement(base, teacher_action=1)
    assert out["node"]["gbs_id"] == "G2"

    # invert every latency / quality field; the rule must not change
    changed = [
        cand("G1", [0.1, 0.6, 0.2, 0.1], e2e_ms=1000.0, comm_ms=900.0,
             inf_ms=90.0, distance_m=900.0, quality=0.99),
        cand("G2", [0.05, 0.8, 0.10, 0.05], e2e_ms=0.1, comm_ms=0.1,
             inf_ms=0.1, distance_m=0.1, quality=0.01),
        cand("G3", [0.1, 0.1, 0.7, 0.1], e2e_ms=2000.0, comm_ms=1000.0,
             inf_ms=200.0, distance_m=1000.0, quality=0.5),
    ]
    out2 = select_oracle_action_agreement(changed, teacher_action=1)
    assert out2["node"]["gbs_id"] == out["node"]["gbs_id"]


def test_compute_capacity_does_not_affect_selection():
    base = [
        cand("G1", [0.1, 0.6, 0.2, 0.1], compute_tflops=5.0),
        cand("G2", [0.05, 0.8, 0.10, 0.05], compute_tflops=30.0),
    ]
    out = select_oracle_action_agreement(base, teacher_action=1)
    assert out["node"]["gbs_id"] == "G2"

    changed = [
        cand("G1", [0.1, 0.6, 0.2, 0.1], compute_tflops=30.0),
        cand("G2", [0.05, 0.8, 0.10, 0.05], compute_tflops=5.0),
    ]
    out2 = select_oracle_action_agreement(changed, teacher_action=1)
    assert out2["node"]["gbs_id"] == out["node"]["gbs_id"]


def test_distribution_change_switches_expert():
    teacher = 1
    a = [
        cand("G1", [0.1, 0.7, 0.1, 0.1]),   # exact, P=0.7
        cand("G2", [0.1, 0.6, 0.2, 0.1]),   # exact, P=0.6
    ]
    assert select_oracle_action_agreement(a, teacher)["node"]["gbs_id"] == "G1"
    b = [
        cand("G1", [0.1, 0.6, 0.2, 0.1]),
        cand("G2", [0.05, 0.9, 0.03, 0.02]),
    ]
    assert select_oracle_action_agreement(b, teacher)["node"]["gbs_id"] == "G2"


def test_audit_fields():
    out = select_oracle_action_agreement([
        cand("G1", [0.1, 0.6, 0.2, 0.1]),
        cand("G2", [0.05, 0.8, 0.10, 0.05]),
        cand("G3", [0.1, 0.1, 0.7, 0.1]),
    ], teacher_action=1)
    assert out["available_candidate_count"] == 3
    assert out["exact_agreement_candidate_count"] == 2
    assert 0.0 <= out["selected_teacher_action_probability"] <= 1.0
    assert out["selected_action"] == 1
    assert out["selected_agreement"] == 1


def test_empty_candidates_rejected():
    with pytest.raises(ValueError):
        select_oracle_action_agreement([], teacher_action=1)
