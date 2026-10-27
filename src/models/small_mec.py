"""Small MEC: the V8.2 late-fusion MLP with a frozen-language interface.

Only change relative to MEC-V8.2 ``MeanPoolMLP``:
  * an optional projected language vector is concatenated with the existing
    Current / Goal / heatmap / network features before the movement head.
The movement head still predicts Forward/Left/Right, and the binary Arrival
head still predicts Stop.  No other architecture change is made.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.config import MECConfig
from src.models.mec_common import (
    ARRIVAL_STOP_THRESHOLD,
    ArrivalReasoningHead,
    HeatmapNetworkContextEncoder,
    LanguageProjector,
    arrival_gated_decision,
    combine_move_and_arrival,
)
from src.models.vision_encoder import count_parameters


MEC = MECConfig()


class SmallMEC(nn.Module):
    def __init__(
        self,
        token_dim: int = 384,
        hidden_dim: int = MEC.small_hidden_dim,
        text_dim: int = 768,
        use_language: bool = True,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.use_language = bool(use_language)
        self.context_encoder = HeatmapNetworkContextEncoder()
        self.arrival_head = ArrivalReasoningHead(token_dim=self.token_dim)
        self.language_proj = (
            LanguageProjector(text_dim, self.token_dim)
            if self.use_language
            else None
        )
        fusion_dim = (
            2 * self.token_dim
            + MEC.heatmap_feature_dim
            + MEC.network_feature_dim
            + (self.token_dim if self.use_language else 0)
        )
        self.fusion_dim = int(fusion_dim)
        self.move_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim), MEC.num_move_actions),
        )

    def _resolve_inputs(
        self,
        current_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        heatmap: torch.Tensor,
        network_state: torch.Tensor,
        language_features: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if current_tokens is None or goal_tokens is None:
            raise ValueError("Current and Goal tokens are required")
        if heatmap is None or network_state is None:
            raise ValueError("Spatial heatmap and network_state are required")
        if self.use_language and language_features is None:
            raise ValueError("language_features are required when use_language=True")
        return (
            current_tokens.float(),
            goal_tokens.float(),
            heatmap.float(),
            network_state.float(),
            None if language_features is None else language_features.float(),
        )

    def forward_components(
        self,
        current_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        heatmap: torch.Tensor,
        network_state: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> dict:
        (
            current_tokens,
            goal_tokens,
            heatmap,
            network_state,
            language_features,
        ) = self._resolve_inputs(
            current_tokens, goal_tokens, heatmap, network_state, language_features
        )
        heat_feat, net_feat = self.context_encoder(heatmap, network_state)
        current = current_tokens.mean(dim=1)
        goal = goal_tokens.mean(dim=1)
        parts = [current, goal, heat_feat, net_feat]
        if self.language_proj is not None:
            parts.append(self.language_proj(language_features))
        move_logits = self.move_classifier(torch.cat(parts, dim=-1))
        arrival_logit, arrival_features = self.arrival_head(
            current_tokens, goal_tokens, heat_feat, net_feat
        )
        return {
            "move_logits": move_logits,
            "arrival_logit": arrival_logit,
            "arrival_features": arrival_features,
            "attention": {},
        }

    def forward(
        self,
        current_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        heatmap: torch.Tensor,
        network_state: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        components = self.forward_components(
            current_tokens,
            goal_tokens,
            heatmap,
            network_state,
            language_features=language_features,
            return_attention=return_attention,
        )
        return combine_move_and_arrival(
            components["move_logits"], components["arrival_logit"]
        )

    @torch.no_grad()
    def predict_action(
        self,
        current_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        heatmap: torch.Tensor,
        network_state: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
        stop_threshold: float = ARRIVAL_STOP_THRESHOLD,
    ) -> dict:
        components = self.forward_components(
            current_tokens,
            goal_tokens,
            heatmap,
            network_state,
            language_features=language_features,
        )
        decision = arrival_gated_decision(
            components["move_logits"],
            components["arrival_logit"],
            stop_threshold=stop_threshold,
        )
        decision["move_logits"] = components["move_logits"]
        decision["arrival_logit"] = components["arrival_logit"]
        return decision

    def parameter_count(self) -> int:
        return count_parameters(self)
