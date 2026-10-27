"""Transformer-M MEC: V8.2 explicit multi-modal reasoning transformer.

Only changes relative to MEC-V8.2 ``ExplicitReasoningTransformer``:
  * one projected language token (with its own type embedding) is appended to
    the fused sequence ``[CLS, current(576), matched-goal(3), heat(1), net(1)]``;
  * the movement head has five classes (Forward/Right/Left/Up/Down), combined
    with the binary Arrival head into the official six-action distribution.
The 576-token spatial current representation and the arrival reasoning head
are unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.config import EncoderConfig, MECConfig
from src.models.mec_common import (
    ARRIVAL_STOP_THRESHOLD,
    ArrivalReasoningHead,
    HeatmapNetworkContextEncoder,
    LanguageProjector,
    arrival_gated_decision,
    combine_move_and_arrival,
)
from src.models.vision_encoder import count_parameters


ENCODER = EncoderConfig()
MEC = MECConfig()


class TransformerMMEC(nn.Module):
    def __init__(
        self,
        token_dim: int = ENCODER.token_dim,
        hidden_dim: int = MEC.transformer_m_hidden_dim,
        num_layers: int = MEC.transformer_m_layers,
        num_heads: int = MEC.transformer_m_heads,
        text_dim: int = 768,
        use_language: bool = True,
        ff_mult: int = MEC.transformer_m_ff_mult,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_language = bool(use_language)

        self.context_encoder = HeatmapNetworkContextEncoder()
        self.arrival_head = ArrivalReasoningHead(token_dim=self.token_dim)

        self.visual_proj = nn.Linear(self.token_dim, self.hidden_dim)
        self.current_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.heat_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.net_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.lang_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.current_pos = nn.Parameter(
            torch.zeros(1, ENCODER.num_tokens, self.hidden_dim)
        )
        self.goal_pos = nn.Parameter(
            torch.zeros(1, MEC.heatmap_channels, self.hidden_dim)
        )

        self.goal_current_attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.goal_current_norm = nn.LayerNorm(self.hidden_dim)
        self.goal_current_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * ff_mult, self.hidden_dim),
        )
        self.goal_current_ffn_norm = nn.LayerNorm(self.hidden_dim)

        self.heat_proj = nn.Linear(MEC.heatmap_feature_dim, self.hidden_dim)
        self.net_proj = nn.Linear(MEC.network_feature_dim, self.hidden_dim)
        self.language_proj = (
            LanguageProjector(text_dim, self.hidden_dim)
            if self.use_language
            else None
        )

        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=self.hidden_dim * ff_mult,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.move_head = nn.Linear(self.hidden_dim, MEC.num_move_actions)

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
        batch = current_tokens.size(0)

        current = self.visual_proj(current_tokens) + self.current_pos
        goal = self.visual_proj(goal_tokens) + self.goal_pos

        kv_current = torch.cat(
            [current, torch.zeros(batch, 1, current.size(-1), device=current.device)],
            dim=1,
        )
        matched, attn_gc = self.goal_current_attn(
            query=goal,
            key=kv_current,
            value=kv_current,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        matched = self.goal_current_norm(goal + matched)
        matched = self.goal_current_ffn_norm(
            matched + self.goal_current_ffn(matched)
        )

        heat_feat, net_feat = self.context_encoder(heatmap, network_state)
        heat_token = self.heat_proj(heat_feat).unsqueeze(1)
        net_token = self.net_proj(net_feat).unsqueeze(1)

        cls = self.cls_token.expand(batch, -1, -1)
        current = current + self.current_type
        matched = matched + self.goal_type
        heat_token = heat_token + self.heat_type
        net_token = net_token + self.net_type

        fusion_parts = [cls, current, matched, heat_token, net_token]
        if self.language_proj is not None:
            lang_token = self.language_proj(language_features).unsqueeze(1)
            fusion_parts.append(lang_token + self.lang_type)
        fusion = torch.cat(fusion_parts, dim=1)
        encoded = self.fusion_encoder(fusion)
        cls_feat = self.final_norm(encoded[:, 0])
        move_logits = self.move_head(cls_feat)

        arrival_logit, arrival_features = self.arrival_head(
            current_tokens, goal_tokens, heat_feat, net_feat
        )
        return {
            "move_logits": move_logits,
            "arrival_logit": arrival_logit,
            "arrival_features": arrival_features,
            "attention": {"goal_to_current": attn_gc},
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
