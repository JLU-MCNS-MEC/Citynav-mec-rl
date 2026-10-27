"""Shared MEC components copied from MEC-V8.2 ``gbs_model_v3.py``.

The movement/arrival factorization, heatmap context encoder and arrival head are
kept structurally identical, except that the movement head now has five classes
(Forward / Right / Left / Up / Down) and is combined with the binary arrival
head into the official six-action CityNav interface.  The only other new
component is a frozen-language projection attached by the Small and
Transformer-M wrappers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import EncoderConfig, MECConfig
from src.data.actions import MOVE_ACTION_IDS, STOP


ENCODER = EncoderConfig()
MEC = MECConfig()
ARRIVAL_STOP_THRESHOLD = float(MEC.arrival_stop_threshold)


def goal_current_similarity(
    current_tokens: torch.Tensor,
    goal_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return [B, S_goal, N_current] cosine similarity maps."""
    if current_tokens.ndim != 3 or goal_tokens.ndim != 3:
        raise ValueError("current_tokens and goal_tokens must be rank-3")
    current = F.normalize(current_tokens.float(), dim=-1)
    goal = F.normalize(goal_tokens.float(), dim=-1)
    return torch.einsum("bnd,bsd->bsn", current, goal)


class HeatmapNetworkContextEncoder(nn.Module):
    def __init__(
        self,
        heatmap_feature_dim: int = MEC.heatmap_feature_dim,
        network_feature_dim: int = MEC.network_feature_dim,
    ):
        super().__init__()
        self.heatmap_net = nn.Sequential(
            nn.Conv2d(MEC.heatmap_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(
                16 * MEC.heatmap_grid * MEC.heatmap_grid,
                heatmap_feature_dim,
            ),
            nn.GELU(),
        )
        self.network_net = nn.Sequential(
            nn.Linear(MEC.network_state_dim, network_feature_dim),
            nn.GELU(),
            nn.Linear(network_feature_dim, network_feature_dim),
            nn.GELU(),
        )

    def forward(
        self, heatmap: torch.Tensor, network_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if heatmap.ndim != 4:
            raise ValueError(f"heatmap must be [B,3,8,8], got {tuple(heatmap.shape)}")
        if network_state.ndim != 2:
            raise ValueError(
                f"network_state must be [B,5], got {tuple(network_state.shape)}"
            )
        heat_feat = self.heatmap_net(heatmap.float())
        net_feat = self.network_net(network_state.float())
        return heat_feat, net_feat


class ArrivalReasoningHead(nn.Module):
    def __init__(self, token_dim: int = ENCODER.token_dim):
        super().__init__()
        hidden = int(MEC.arrival_hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(
                8 + MEC.heatmap_feature_dim + MEC.network_feature_dim,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def extract_features(
        self,
        current_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        heatmap_feature: torch.Tensor,
        network_feature: torch.Tensor,
    ) -> torch.Tensor:
        sim = goal_current_similarity(current_tokens, goal_tokens)
        patch_best = sim.max(dim=1).values
        b, n = patch_best.shape

        temperature = float(MEC.arrival_match_temperature)
        attention = torch.softmax(patch_best / max(temperature, 1e-4), dim=-1)

        g = int(ENCODER.grid_size)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, g, device=current_tokens.device),
            torch.linspace(-1.0, 1.0, g, device=current_tokens.device),
            indexing="ij",
        )
        xx = xx.reshape(1, n)
        yy = yy.reshape(1, n)

        max_match = patch_best.max(dim=-1).values
        topk = min(10, n)
        topk_mean = torch.topk(patch_best, k=topk, dim=-1).values.mean(dim=-1)
        mean_match = patch_best.mean(dim=-1)
        entropy = -(attention * (attention.clamp_min(1e-8)).log()).sum(dim=-1)
        weighted_x = (attention * xx).sum(dim=-1)
        weighted_y = (attention * yy).sum(dim=-1)
        best_idx = patch_best.argmax(dim=-1)
        best_x = xx.expand(b, -1).gather(1, best_idx[:, None]).squeeze(1)
        best_y = yy.expand(b, -1).gather(1, best_idx[:, None]).squeeze(1)

        match_features = torch.stack(
            [
                max_match,
                topk_mean,
                mean_match,
                entropy,
                weighted_x,
                weighted_y,
                best_x,
                best_y,
            ],
            dim=-1,
        )
        return torch.cat(
            [match_features, heatmap_feature, network_feature], dim=-1
        )

    def forward(
        self,
        current_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        heatmap_feature: torch.Tensor,
        network_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(
            current_tokens, goal_tokens, heatmap_feature, network_feature
        )
        logit = self.net(features).squeeze(-1)
        return logit, features


def combine_move_and_arrival(
    move_logits: torch.Tensor, arrival_logit: torch.Tensor
) -> torch.Tensor:
    """Return six log-probabilities in official action-ID order.

    Column 0 is STOP; columns 1..5 are Forward / Right / Left / Up / Down.
    """
    log_move_gate = F.logsigmoid(-arrival_logit).unsqueeze(-1)
    log_stop = F.logsigmoid(arrival_logit).unsqueeze(-1)
    move_log_prob = F.log_softmax(move_logits, dim=-1) + log_move_gate
    return torch.cat([log_stop, move_log_prob], dim=-1)


def arrival_gated_decision(
    move_logits: torch.Tensor,
    arrival_logit: torch.Tensor,
    stop_threshold: float = ARRIVAL_STOP_THRESHOLD,
) -> dict:
    threshold = float(stop_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"stop_threshold must be in (0,1), got {threshold}")
    arrival_probability = torch.sigmoid(arrival_logit)
    move_probabilities = torch.softmax(move_logits, dim=-1)
    move_head = move_probabilities.argmax(dim=-1)
    move_id_table = torch.as_tensor(
        MOVE_ACTION_IDS, dtype=move_head.dtype, device=move_head.device
    )
    move_action = move_id_table[move_head]
    stop_action = torch.full_like(move_action, int(STOP))
    action = torch.where(arrival_probability >= threshold, stop_action, move_action)
    joint_move = (1.0 - arrival_probability).unsqueeze(-1) * move_probabilities
    joint_probabilities = torch.cat(
        [arrival_probability.unsqueeze(-1), joint_move], dim=-1
    )
    return {
        "action": action,
        "joint_probabilities": joint_probabilities,
        "arrival_probability": arrival_probability,
        "move_probabilities": move_probabilities,
        "stop_threshold": threshold,
    }


def build_spatial_heatmap(
    current_tokens: torch.Tensor,
    goal_tokens: torch.Tensor,
    *,
    grid: int = MEC.heatmap_grid,
) -> torch.Tensor:
    """Scale-specific cosine heatmap [B,S,grid,grid] from frozen tokens."""
    sim = goal_current_similarity(current_tokens, goal_tokens)
    b, s, n = sim.shape
    if s != MEC.heatmap_channels:
        raise ValueError(f"goal scales {s} != heatmap channels {MEC.heatmap_channels}")
    maps = sim.reshape(b, s, ENCODER.grid_size, ENCODER.grid_size)
    maps = maps.reshape(b * s, 1, ENCODER.grid_size, ENCODER.grid_size)
    pooled = F.adaptive_avg_pool2d(maps, (grid, grid))
    return pooled.reshape(b, s, grid, grid)


class LanguageProjector(nn.Module):
    def __init__(self, text_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(text_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
            nn.GELU(),
        )

    def forward(self, language_features: torch.Tensor) -> torch.Tensor:
        if language_features.ndim != 2:
            raise ValueError(
                f"language_features must be [B,D], got {tuple(language_features.shape)}"
            )
        return self.net(language_features.float())


def placeholder_network_state(seed: int, dim: int = MEC.network_state_dim) -> np.ndarray:
    """Deterministic 5-D placeholder.

    The real 20-MEC deployment/association study is intentionally out of scope
    for this data-adaptation stage.  This vector has the same interface and
    normalization range, but it is **not** a measured network state.
    """
    rng = np.random.default_rng(int(seed))
    return rng.random(int(dim), dtype=np.float32)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def check_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains non-finite values")
