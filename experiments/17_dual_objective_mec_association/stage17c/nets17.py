#!/usr/bin/env python3
"""Stage 17-C networks: the Stage-16 architectures with a 127-dim input.

Stage 16 hard-codes STATE_DIM = 74 in `train16.QNet/ActorCritic`.  Stage 17
needs 127 (74 inherited + 53 constraint features), so the same layer stack is
re-instantiated with a configurable input dimension.  The layer widths,
activations and heads are identical to Stage 16; only `in_features` changes.

Stage 16's file and checkpoints are NOT touched.
"""
from __future__ import annotations

import torch
import torch.nn as nn

STATE17_DIM = 127
N_ACT = 10
NEG_INF = -1e18


class QNet17(nn.Module):
    """Stage-16 QNet with state_dim=127."""

    def __init__(self, state_dim: int = STATE17_DIM, hidden: int = 256,
                 n_act: int = N_ACT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_act))

    def forward(self, x):
        return self.net(x)


class ActorCritic17(nn.Module):
    """Stage-16 ActorCritic with state_dim=127."""

    def __init__(self, state_dim: int = STATE17_DIM, hidden: int = 256,
                 n_act: int = N_ACT):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, n_act)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(logits.device, torch.float32)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return logits.masked_fill(mask < 0.5, NEG_INF)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask_logits(logits, mask)
    p = torch.softmax(m, -1)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return p / (p.sum(-1, keepdim=True) + 1e-12)
