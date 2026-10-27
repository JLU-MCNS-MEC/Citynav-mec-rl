"""T5 + language MEC models for CityNav.

The full model receives:
  Current tokens [576,384], retrieved Goal tokens [3,384], 16-step visual
  history [16,768], pose history [16,12], action history [16,6], history mask
  [16], GAB [20], Top-5 map-retrieval features [22], frozen BERT [768].

The map-free baseline receives Current tokens, the same history/pose/mask and
language features, but no Goal tokens, no GAB evidence and no map features.
It has a separate current+history+language Arrival head because the shared
Goal-Current match head is undefined without Goal tokens.  The baseline never
reads target coordinates, target IDs, future actions or success labels.

The Small late-fusion MLP and Transformer-M explicit-reasoning backbone are
the same families as the earlier MEC-V8.2 ports; only the observable input
adapter and six-action heads are new.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import MECConfig
from src.models.mec_common import (
    ARRIVAL_STOP_THRESHOLD,
    LanguageProjector,
    arrival_gated_decision,
    combine_move_and_arrival,
    goal_current_similarity,
)


MEC = MECConfig()
T5_HISTORY_DIM = 128


def match_features(
    current_tokens: torch.Tensor,
    goal_tokens: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """The 8 observable Current/Goal match features for the Arrival head."""
    sim = goal_current_similarity(current_tokens, goal_tokens)
    patch_best = sim.max(dim=1).values
    batch, n = patch_best.shape
    attention = torch.softmax(
        patch_best / max(float(temperature), 1e-4), dim=-1
    )
    grid = int(round(n ** 0.5))
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, grid, device=current_tokens.device),
        torch.linspace(-1.0, 1.0, grid, device=current_tokens.device),
        indexing="ij",
    )
    xx = xx.reshape(1, n)
    yy = yy.reshape(1, n)
    max_match = patch_best.max(dim=-1).values
    topk = min(10, n)
    topk_mean = torch.topk(patch_best, k=topk, dim=-1).values.mean(dim=-1)
    mean_match = patch_best.mean(dim=-1)
    entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1)
    weighted_x = (attention * xx).sum(dim=-1)
    weighted_y = (attention * yy).sum(dim=-1)
    best_idx = patch_best.argmax(dim=-1)
    best_x = xx.expand(batch, -1).gather(1, best_idx[:, None]).squeeze(1)
    best_y = yy.expand(batch, -1).gather(1, best_idx[:, None]).squeeze(1)
    return torch.stack(
        [max_match, topk_mean, mean_match, entropy,
         weighted_x, weighted_y, best_x, best_y],
        dim=-1,
    )


class MatchArrivalHead(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, current_tokens, goal_tokens):
        features = match_features(current_tokens, goal_tokens)
        return self.net(features).squeeze(-1), features


class BaselineArrivalHead(nn.Module):
    """Map-free Stop head using only current visual, language and history."""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, current, history, language):
        features = torch.cat([current, history, language], dim=-1)
        return self.net(features).squeeze(-1), features


class T5HistoryEncoder(nn.Module):
    """Causal 16-step visual + pose + six-action history encoder."""

    def __init__(
        self,
        steps: int = 16,
        visual_dim: int = 768,
        pose_dim: int = 12,
        action_dim: int = 6,
        hidden_dim: int = T5_HISTORY_DIM,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.steps = int(steps)
        self.visual_proj = nn.Linear(int(visual_dim), hidden_dim)
        self.pose_proj = nn.Linear(int(pose_dim), hidden_dim)
        self.action_proj = nn.Linear(int(action_dim), hidden_dim)
        self.step_pos = nn.Parameter(torch.zeros(1, self.steps, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, history_vis, history_pose, history_action, history_mask):
        if history_vis.ndim != 3 or history_vis.shape[1:] != (
            self.steps, 768
        ):
            raise ValueError(
                f"history_vis must be [B,{self.steps},768], "
                f"got {tuple(history_vis.shape)}"
            )
        if history_pose.shape[1:] != (self.steps, 12):
            raise ValueError(
                f"history_pose must be [B,{self.steps},12], "
                f"got {tuple(history_pose.shape)}"
            )
        if history_action.shape[1:] != (self.steps, 6):
            raise ValueError(
                f"history_action must be [B,{self.steps},6], "
                f"got {tuple(history_action.shape)}"
            )
        if history_mask.shape != (history_vis.size(0), self.steps):
            raise ValueError(
                f"history_mask must be [B,{self.steps}], "
                f"got {tuple(history_mask.shape)}"
            )
        mask = history_mask.to(history_vis.dtype).unsqueeze(-1)
        x = (
            self.visual_proj(history_vis.float())
            + self.pose_proj(history_pose.float())
            + self.action_proj(history_action.float())
        ) * mask
        x = x + self.step_pos * mask
        cls = self.cls_token.expand(x.size(0), -1, -1)
        encoded = self.encoder(torch.cat([cls, x], dim=1))
        return self.norm(encoded[:, 0])


def _validate_history(
    current_tokens,
    history_vis,
    history_pose,
    history_action,
    history_mask,
    language_features,
):
    if current_tokens.ndim != 3 or current_tokens.shape[1:] != (576, 384):
        raise ValueError(
            f"current_tokens must be [B,576,384], got {tuple(current_tokens.shape)}"
        )
    if language_features.ndim != 2 or language_features.shape[-1] != 768:
        raise ValueError(
            f"language_features must be [B,768], got {tuple(language_features.shape)}"
        )
    batch = current_tokens.size(0)
    for name, value, shape in (
        ("history_vis", history_vis, (16, 768)),
        ("history_pose", history_pose, (16, 12)),
        ("history_action", history_action, (16, 6)),
    ):
        if value.ndim != 3 or tuple(value.shape[1:]) != shape:
            raise ValueError(f"{name} must be [B,{shape}], got {tuple(value.shape)}")
        if value.size(0) != batch:
            raise ValueError(f"{name} batch mismatch")
    if history_mask.shape != (batch, 16):
        raise ValueError(
            f"history_mask must be [B,16], got {tuple(history_mask.shape)}"
        )


def _validate_retrieval_inputs(
    goal_tokens, gab, map_feat, batch, *, require_map: bool = True
):
    if (
        goal_tokens is None
        or goal_tokens.ndim != 3
        or goal_tokens.shape[1:] != (3, 384)
    ):
        raise ValueError(
            "goal_tokens must be [B,3,384], got "
            f"{None if goal_tokens is None else tuple(goal_tokens.shape)}"
        )
    if goal_tokens.size(0) != batch:
        raise ValueError("goal_tokens batch mismatch")
    if gab is None or gab.ndim != 2 or gab.shape[-1] != 20:
        raise ValueError(
            f"gab must be [B,20], got {None if gab is None else tuple(gab.shape)}"
        )
    if gab.size(0) != batch:
        raise ValueError("gab batch mismatch")
    if require_map:
        if map_feat is None or map_feat.ndim != 2 or map_feat.shape[-1] != 22:
            raise ValueError(
                "map_feat must be [B,22], got "
                f"{None if map_feat is None else tuple(map_feat.shape)}"
            )
        if map_feat.size(0) != batch:
            raise ValueError("map_feat batch mismatch")
    elif map_feat is not None:
        raise ValueError("use_map=False requires map_feat=None")


class T5SmallMEC(nn.Module):
    """Small late-fusion MLP over T5 + language (full or map-free)."""

    architecture = "t5_late_fusion_mlp"

    def __init__(
        self,
        token_dim: int = 384,
        hidden_dim: int = MEC.small_hidden_dim,
        text_dim: int = 768,
        use_retrieval: bool = True,
        use_map: bool = True,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.use_retrieval = bool(use_retrieval)
        self.use_map = bool(use_map)
        if not self.use_retrieval and self.use_map:
            raise ValueError("map branch requires retrieval")
        self.history_encoder = T5HistoryEncoder()
        self.language_proj = LanguageProjector(text_dim, self.token_dim)
        fusion_dim = self.token_dim + T5_HISTORY_DIM + self.token_dim
        if self.use_retrieval:
            fusion_dim += self.token_dim + 20
        if self.use_map:
            fusion_dim += 22
        self.fusion_dim = int(fusion_dim)
        self.move_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim), MEC.num_move_actions),
        )
        if self.use_retrieval:
            self.arrival_head = MatchArrivalHead()
        else:
            self.arrival_head = BaselineArrivalHead(
                self.token_dim + T5_HISTORY_DIM + self.token_dim
            )

    def forward_components(
        self,
        current_tokens,
        history_vis,
        history_pose,
        history_action,
        history_mask,
        language_features,
        goal_tokens=None,
        gab=None,
        map_feat=None,
        return_attention: bool = False,
    ):
        _validate_history(
            current_tokens, history_vis, history_pose, history_action,
            history_mask, language_features,
        )
        current = current_tokens.float().mean(dim=1)
        history = self.history_encoder(
            history_vis, history_pose, history_action, history_mask
        )
        language = self.language_proj(language_features.float())
        parts = [current, history, language]
        if self.use_retrieval:
            _validate_retrieval_inputs(
                goal_tokens, gab, map_feat, current.size(0),
                require_map=self.use_map,
            )
            parts.extend([goal_tokens.float().mean(dim=1), gab.float()])
        elif any(value is not None for value in (goal_tokens, gab, map_feat)):
            raise ValueError(
                "map-free baseline must not receive goal, GAB or map features"
            )
        if self.use_map:
            parts.append(map_feat.float())
        move_logits = self.move_classifier(torch.cat(parts, dim=-1))
        if self.use_retrieval:
            arrival_logit, arrival_features = self.arrival_head(
                current_tokens.float(), goal_tokens.float()
            )
        else:
            arrival_logit, arrival_features = self.arrival_head(
                current, history, language
            )
        return {
            "move_logits": move_logits,
            "arrival_logit": arrival_logit,
            "arrival_features": arrival_features,
            "attention": {},
        }

    def forward(self, *args, **kwargs):
        components = self.forward_components(*args, **kwargs)
        return combine_move_and_arrival(
            components["move_logits"], components["arrival_logit"]
        )

    @torch.no_grad()
    def predict_action(self, *args, stop_threshold=ARRIVAL_STOP_THRESHOLD, **kwargs):
        components = self.forward_components(*args, **kwargs)
        decision = arrival_gated_decision(
            components["move_logits"],
            components["arrival_logit"],
            stop_threshold=stop_threshold,
        )
        decision["move_logits"] = components["move_logits"]
        decision["arrival_logit"] = components["arrival_logit"]
        return decision


class T5TransformerMMEC(nn.Module):
    """Transformer-M explicit reasoning over current, goal, history, GAB, map, language."""

    architecture = "t5_explicit_transformer"

    def __init__(
        self,
        token_dim: int = 384,
        hidden_dim: int = MEC.transformer_m_hidden_dim,
        num_layers: int = MEC.transformer_m_layers,
        num_heads: int = MEC.transformer_m_heads,
        ff_mult: int = MEC.transformer_m_ff_mult,
        text_dim: int = 768,
        use_retrieval: bool = True,
        use_map: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_retrieval = bool(use_retrieval)
        self.use_map = bool(use_map)
        if not self.use_retrieval and self.use_map:
            raise ValueError("map branch requires retrieval")
        self.history_encoder = T5HistoryEncoder()
        self.visual_proj = nn.Linear(self.token_dim, self.hidden_dim)
        self.current_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.history_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.gab_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.map_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.lang_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.current_pos = nn.Parameter(torch.zeros(1, 576, self.hidden_dim))
        self.goal_pos = nn.Parameter(torch.zeros(1, 3, self.hidden_dim))
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
        self.history_proj = nn.Linear(T5_HISTORY_DIM, self.hidden_dim)
        self.gab_proj = nn.Linear(20, self.hidden_dim)
        self.map_proj = nn.Linear(22, self.hidden_dim)
        self.language_proj = LanguageProjector(text_dim, self.hidden_dim)
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
        if self.use_retrieval:
            self.arrival_head = MatchArrivalHead()
        else:
            self.arrival_head = BaselineArrivalHead(
                self.token_dim + T5_HISTORY_DIM + text_dim
            )

    def forward_components(
        self,
        current_tokens,
        history_vis,
        history_pose,
        history_action,
        history_mask,
        language_features,
        goal_tokens=None,
        gab=None,
        map_feat=None,
        return_attention: bool = False,
    ):
        _validate_history(
            current_tokens, history_vis, history_pose, history_action,
            history_mask, language_features,
        )
        batch = current_tokens.size(0)
        current = current_tokens.float()
        history = self.history_encoder(
            history_vis, history_pose, history_action, history_mask
        )
        language = self.language_proj(language_features.float())
        history_token = self.history_proj(history).unsqueeze(1)
        language_token = language.unsqueeze(1)
        fusion = [
            self.cls_token.expand(batch, -1, -1),
            self.visual_proj(current) + self.current_pos + self.current_type,
        ]
        attention = {}
        if self.use_retrieval:
            _validate_retrieval_inputs(
                goal_tokens, gab, map_feat, batch,
                require_map=self.use_map,
            )
            goal = (
                self.visual_proj(goal_tokens.float()) + self.goal_pos
            )
            kv_current = torch.cat(
                [
                    self.visual_proj(current) + self.current_pos,
                    torch.zeros(batch, 1, self.hidden_dim, device=current.device),
                ],
                dim=1,
            )
            matched, attention["goal_to_current"] = self.goal_current_attn(
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
            fusion.extend(
                [
                    matched + self.goal_type,
                    history_token + self.history_type,
                    self.gab_proj(gab.float()).unsqueeze(1) + self.gab_type,
                ]
            )
            if self.use_map:
                fusion.append(
                    self.map_proj(map_feat.float()).unsqueeze(1) + self.map_type
                )
        else:
            if any(value is not None for value in (goal_tokens, gab, map_feat)):
                raise ValueError(
                    "map-free baseline must not receive goal, GAB or map features"
                )
            fusion.append(history_token + self.history_type)
        fusion.append(language_token + self.lang_type)
        encoded = self.fusion_encoder(torch.cat(fusion, dim=1))
        cls_feat = self.final_norm(encoded[:, 0])
        move_logits = self.move_head(cls_feat)
        if self.use_retrieval:
            arrival_logit, arrival_features = self.arrival_head(
                current, goal_tokens.float()
            )
        else:
            arrival_logit, arrival_features = self.arrival_head(
                current.mean(dim=1),
                history,
                language_features.float(),
            )
        return {
            "move_logits": move_logits,
            "arrival_logit": arrival_logit,
            "arrival_features": arrival_features,
            "attention": attention,
        }

    def forward(self, *args, **kwargs):
        components = self.forward_components(*args, **kwargs)
        return combine_move_and_arrival(
            components["move_logits"], components["arrival_logit"]
        )

    @torch.no_grad()
    def predict_action(self, *args, stop_threshold=ARRIVAL_STOP_THRESHOLD, **kwargs):
        components = self.forward_components(*args, **kwargs)
        decision = arrival_gated_decision(
            components["move_logits"],
            components["arrival_logit"],
            stop_threshold=stop_threshold,
        )
        decision["move_logits"] = components["move_logits"]
        decision["arrival_logit"] = components["arrival_logit"]
        return decision


def build_t5_model(
    name: str,
    *,
    use_retrieval: bool = True,
    use_map: bool = True,
    text_dim: int = 768,
):
    name = str(name).lower()
    if name == "small":
        return T5SmallMEC(
            text_dim=text_dim,
            use_retrieval=use_retrieval,
            use_map=use_map,
        )
    if name in {"transformer_m", "transformer-m"}:
        return T5TransformerMMEC(
            text_dim=text_dim,
            use_retrieval=use_retrieval,
            use_map=use_map,
        )
    raise ValueError(f"unknown T5 model: {name}")
