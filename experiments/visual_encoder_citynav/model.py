"""Visual encoder plus temporary auxiliary pretraining heads.

Only ``VisualTokenEncoder`` is saved as the final artifact.  The action,
cross-view and retrieval heads are auxiliary and are not official MEC models.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vision_encoder import VisualTokenEncoder


class PretrainModel(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int = 384,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        text_dim: int = 768,
    ):
        super().__init__()
        self.encoder = VisualTokenEncoder()
        self.token_dim = int(token_dim)
        self.text_dim = int(text_dim)
        self.projection_dim = int(projection_dim)
        self.action_head = nn.Sequential(
            nn.Linear(2 * self.token_dim + self.text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 6),
        )
        self.xview_query = nn.Sequential(
            nn.Linear(2 * self.token_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.xview_candidate = nn.Sequential(
            nn.Linear(self.token_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.retrieval_query = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.retrieval_candidate = nn.Sequential(
            nn.Linear(self.token_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.xview_logit_scale = nn.Parameter(torch.tensor(0.0))
        self.retrieval_logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_current(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder(images.float())
        global_feature = torch.cat(
            [tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1
        )
        return tokens, global_feature

    def encode_candidate_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Encode [B,K,3,96,96] candidate images to [B,K,384]."""
        if images.ndim != 5:
            raise ValueError(f"candidate images must be [B,K,3,H,W], got {images.shape}")
        batch, k, channels, height, width = images.shape
        flat = images.reshape(batch * k, channels, height, width).float()
        if (height, width) != (336, 336):
            flat = F.interpolate(
                flat, size=(336, 336), mode="bilinear", align_corners=False
            )
        tokens = self.encoder(flat)
        return tokens.mean(dim=1).reshape(batch, k, -1)

    def encode_candidate_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode [N,3,96,96] candidate images to [N,384]."""
        if images.ndim != 4:
            raise ValueError(f"candidate images must be [N,3,H,W], got {images.shape}")
        if images.shape[-2:] != (336, 336):
            images = F.interpolate(
                images.float(), size=(336, 336), mode="bilinear",
                align_corners=False,
            )
        return self.encoder(images.float()).mean(dim=1)

    def forward(
        self,
        current_images: torch.Tensor,
        candidate_images: torch.Tensor,
        language_features: torch.Tensor,
    ) -> dict:
        _, current_global = self.encode_current(current_images)
        candidate_tokens = self.encode_candidate_batch(candidate_images)
        action_logits = self.action_head(
            torch.cat([current_global, language_features.float()], dim=-1)
        )
        xview_query = self.xview_query(current_global)
        xview_candidate = self.xview_candidate(candidate_tokens)
        retrieval_query = self.retrieval_query(language_features.float())
        retrieval_candidate = self.retrieval_candidate(candidate_tokens)
        return {
            "action_logits": action_logits,
            "current_global": current_global,
            "candidate_tokens": candidate_tokens,
            "xview_query": xview_query,
            "xview_candidate": xview_candidate,
            "retrieval_query": retrieval_query,
            "retrieval_candidate": retrieval_candidate,
            "xview_logit_scale": self.xview_logit_scale,
            "retrieval_logit_scale": self.retrieval_logit_scale,
        }

    def encoder_parameters(self):
        return list(self.encoder.parameters())


def load_pretrained_encoder(
    checkpoint: Path | str,
    device: torch.device,
) -> tuple[VisualTokenEncoder, dict]:
    """Strictly load shared_encoder.pt and report the actual loaded ratio."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"pretrained encoder checkpoint missing: {path}; refusing random init"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"encoder checkpoint is not a state dict: {type(state)}")
    encoder = VisualTokenEncoder()
    model_state = encoder.state_dict()
    matching = 0
    total = 0
    missing = []
    unexpected = []
    shape_mismatch = []
    for key, value in model_state.items():
        total += int(value.numel())
        if key not in state:
            missing.append(key)
            continue
        if tuple(state[key].shape) != tuple(value.shape):
            shape_mismatch.append(key)
            continue
        matching += int(value.numel())
    for key in state:
        if key not in model_state:
            unexpected.append(key)
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "encoder checkpoint incompatible: "
            f"missing={missing[:5]} unexpected={unexpected[:5]} "
            f"shape_mismatch={shape_mismatch[:5]}"
        )
    encoder.load_state_dict(state, strict=True)
    encoder.to(device)
    ratio = matching / max(total, 1)
    return encoder, {
        "checkpoint": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "loaded_parameter_ratio": float(ratio),
        "loaded_parameters": int(matching),
        "total_parameters": int(total),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
    }


def encoder_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_encoder_only(encoder: nn.Module, path: Path | str) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), path)
    return {
        "path": str(path),
        "sha256": encoder_sha256(path),
        "parameters": int(sum(p.numel() for p in encoder.parameters())),
    }
