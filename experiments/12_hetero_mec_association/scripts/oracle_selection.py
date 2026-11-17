#!/usr/bin/env python3
"""Oracle v2 association rule: action agreement only.

The rule is deliberately tiny and pure so it can be unit-tested without any
model, data or latency code:

    agreement = int(argmax(P_i) == teacher_action)
    score     = (agreement, P_i(teacher_action))
    select the lexicographically largest score;
    final ties are broken by fixed node id order (G1 < G2 < ...).

E2E, communication latency, inference latency, distance, model quality and any
RL signal are **not** inputs to this function and must never be added.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Candidate:
    node: Any
    probs: Sequence[float]


def node_index(node: Any) -> int:
    """Fixed deployment order G1 < G2 < ...; unknown ids sort last."""
    gid = str(node.get("gbs_id", "")) if isinstance(node, dict) else ""
    if gid.startswith("G") and gid[1:].isdigit():
        return int(gid[1:])
    return 10**6


def _score(candidate: Candidate, teacher_action: int) -> tuple[int, float, int]:
    probs = candidate.probs
    pred = int(max(range(len(probs)), key=lambda i: float(probs[i])))
    agreement = int(pred == int(teacher_action))
    p = float(probs[int(teacher_action)])
    return agreement, p, pred


def select_oracle_action_agreement(
    candidates: Sequence[Candidate], teacher_action: int,
) -> dict:
    """Select a node by (agreement, P(a*)) only; ties by node id order."""
    if not candidates:
        raise ValueError("no available candidate nodes")
    ta = int(teacher_action)
    scored = [(_score(c, ta), c) for c in candidates]
    # agreement desc, probability desc, then fixed node order asc
    scored.sort(key=lambda item: (-item[0][0], -item[0][1],
                                  node_index(item[1].node)))
    best_score, best_candidate = scored[0]
    top = (best_score[0], best_score[1])
    tied = [item for item in scored
            if (item[0][0], item[0][1]) == top]
    exact_count = sum(1 for item in scored if item[0][0] == 1)
    agree, p, pred = best_score
    return {
        "node": best_candidate.node,
        "selected_action": pred,
        "selected_teacher_action_probability": p,
        "selected_agreement": agree,
        "available_candidate_count": len(candidates),
        "exact_agreement_candidate_count": exact_count,
        "tie_break_used": len(tied) > 1,
    }
