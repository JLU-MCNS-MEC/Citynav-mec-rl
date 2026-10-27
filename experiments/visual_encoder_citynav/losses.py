"""Losses for CityNav visual-encoder pretraining.

All three losses are kept separate.  Current-pose cross-view positives and
target-position retrieval positives are different masks and are reported
separately; they must never be silently merged.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def multi_positive_infonce(
    query: torch.Tensor,
    candidates: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_valid: torch.Tensor | None = None,
    *,
    temperature: float = 0.07,
    logit_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """InfoNCE with any number of positive candidates per row.

    ``candidate_valid`` excludes zero-padded candidates from the denominator.
    Rows without a positive and a valid candidate are skipped.
    """
    if query.ndim != 2 or candidates.ndim != 3:
        raise ValueError("query must be [B,D] and candidates [B,K,D]")
    if query.size(0) != candidates.size(0):
        raise ValueError("batch mismatch")
    if positive_mask.shape != candidates.shape[:2]:
        raise ValueError("positive_mask must be [B,K]")
    q = F.normalize(query.float(), dim=-1)
    c = F.normalize(candidates.float(), dim=-1)
    logits = torch.einsum("bd,bkd->bk", q, c) / float(temperature)
    if logit_scale is not None:
        logits = logits * logit_scale.exp().clamp(max=100.0)
    positive = positive_mask.bool()
    if candidate_valid is not None:
        valid = candidate_valid.bool()
        positive = positive & valid
        logits = logits.masked_fill(~valid, float("-inf"))
    row_valid = positive.any(dim=1) & torch.isfinite(logits).any(dim=1)
    if not bool(row_valid.any()):
        return query.sum() * 0.0, {
            "valid_rows": 0,
            "top1": None,
            "top5": None,
        }
    log_denom = torch.logsumexp(logits[row_valid], dim=1)
    positive_logits = logits[row_valid].masked_fill(
        ~positive[row_valid], float("-inf")
    )
    log_pos = torch.logsumexp(positive_logits, dim=1)
    loss = -(log_pos - log_denom).mean()
    with torch.no_grad():
        rows = logits[row_valid]
        top1 = rows.argmax(dim=1)
        top5 = rows.topk(k=min(5, rows.size(1)), dim=1).indices
        pos_rows = positive[row_valid]
        top1_hit = pos_rows.gather(1, top1[:, None]).squeeze(1).float().mean()
        top5_hit = pos_rows.gather(1, top5).any(dim=1).float().mean()
    return loss, {
        "valid_rows": int(row_valid.sum().item()),
        "top1": float(top1_hit.item()),
        "top5": float(top5_hit.item()),
    }


def weighted_action_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.float(), actions.long().view(-1), weight=class_weights
    )
