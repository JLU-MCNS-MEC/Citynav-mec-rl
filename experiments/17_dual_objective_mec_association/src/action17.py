#!/usr/bin/env python3
"""Stage 17-A unified feasible-node action interface.

Audit result (AUDIT_STAGE14_16.md, section G): Stage 16 does not convert a
continuous action into a node index.  TD3/SAC sample from a softmax over the
10 node logits, DDPG takes its argmax, DQN takes the argmax of Q, and the
baselines pick a node directly.  Every algorithm therefore already shares one
discrete action space of size 10.

The problem Stage 17 introduces is different: a node can now be *infeasible*
(no coverage / deadline / energy).  If each algorithm handled the mask on its
own, the comparison would be unfair.  This module defines the single masking
and projection rule all seven actors must use:

    logits' = -inf where feasible_mec_mask == 0
    sample  : multinomial(softmax(logits'))
    greedy  : argmax(logits')

and, when the feasible set is empty, `select` returns None so the environment
records `no_feasible_mec` instead of silently picking an illegal node.
"""
from __future__ import annotations

import numpy as np
import torch

NEG_INF = -1e18


def mask_logits(logits: torch.Tensor, feasible_mask) -> torch.Tensor:
    """Return logits with infeasible entries set to -inf (no renormalisation)."""
    mask = torch.as_tensor(feasible_mask, device=logits.device, dtype=torch.float32)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return logits.masked_fill(mask < 0.5, NEG_INF)


def masked_probs(logits: torch.Tensor, feasible_mask) -> torch.Tensor:
    """Numerically safe softmax over the feasible set."""
    m = mask_logits(logits, feasible_mask)
    p = torch.softmax(m, dim=-1)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return p / (p.sum(dim=-1, keepdim=True) + 1e-12)


def sample_node(logits: torch.Tensor, feasible_mask, generator=None) -> int | None:
    if float(np.sum(feasible_mask)) < 0.5:
        return None
    p = masked_probs(logits, feasible_mask)
    return int(torch.multinomial(p[0], 1, generator=generator).item())


def greedy_node(logits: torch.Tensor, feasible_mask) -> int | None:
    if float(np.sum(feasible_mask)) < 0.5:
        return None
    return int(torch.argmax(mask_logits(logits, feasible_mask), dim=-1).item())


def project_index(action: int, feasible_mask) -> int | None:
    """Guard for algorithms that emit an index directly (e.g. replay actions).

    Returns None when the stored action became infeasible, so the caller can
    treat it as `no_feasible_mec` rather than executing an illegal node.
    """
    if feasible_mask[int(action)] > 0.5:
        return int(action)
    return None
