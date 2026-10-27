"""Language-to-map candidate retrieval.

Training labels are generated only from training target positions.  The
inference API :func:`retrieve_candidates` accepts language features and
candidate embeddings only; it has no target coordinate, target ID, target
crop, teacher action or success argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.candidates import (
    CandidateLibrary,
    encode_candidate_images,
    positive_mask_for_target,
    recall_at_k,
    topk_indices,
)


@dataclass
class RetrievalSample:
    split: str
    episode_id: str
    map_name: str
    language_features: np.ndarray
    library: CandidateLibrary
    target_xy: tuple[float, float]  # training labels only; never passed to inference


class LanguageMapRetriever(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        visual_dim: int = 384,
        hidden_dim: int = 512,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.text_dim = int(text_dim)
        self.visual_dim = int(visual_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.visual_dim),
            nn.LayerNorm(self.visual_dim),
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / float(temperature)))
        )

    def forward(
        self,
        language_features: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        query = F.normalize(
            self.projector(language_features.float()), dim=-1
        )
        candidates = F.normalize(
            candidate_embeddings.float(), dim=-1
        )
        if query.ndim == 2 and query.size(0) == 1:
            scores = query @ candidates.T
            return scores[0] * self.logit_scale.exp()
        if query.ndim == 2 and candidates.ndim == 2:
            return query @ candidates.T * self.logit_scale.exp()
        raise ValueError("query and candidates must be rank-2 tensors")


def retrieve_candidates(
    language_features: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    retriever: LanguageMapRetriever,
    *,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Inference API: no target coordinate/ID/crop/action/success argument."""
    with torch.no_grad():
        scores = retriever(language_features, candidate_embeddings)
    values = scores.detach().cpu().numpy().reshape(-1)
    indices = topk_indices(values, int(top_k))
    return indices, values


def _candidate_tokens(
    sample: RetrievalSample,
    encoder: torch.nn.Module | None,
    device: torch.device,
    *,
    use_encoder: bool,
    requires_grad: bool,
) -> torch.Tensor:
    if use_encoder:
        if encoder is None:
            raise ValueError("use_encoder requires an encoder")
        return encode_candidate_images(
            sample.library.images,
            encoder,
            device,
            requires_grad=requires_grad,
        )
    if sample.library.embeddings is None:
        raise ValueError(
            f"candidate embeddings missing for {sample.library.map_name}"
        )
    return torch.from_numpy(sample.library.embeddings).to(device)


def _loss_for_sample(
    sample: RetrievalSample,
    retriever: LanguageMapRetriever,
    encoder: torch.nn.Module | None,
    device: torch.device,
    *,
    train_encoder: bool,
) -> tuple[torch.Tensor, dict] | None:
    targets = positive_mask_for_target(
        sample.library.centers, sample.target_xy
    )
    if not bool(targets.any()):
        return None
    candidates = _candidate_tokens(
        sample, encoder, device,
        use_encoder=train_encoder,
        requires_grad=train_encoder,
    )
    language = torch.from_numpy(
        np.asarray(sample.language_features, dtype=np.float32)
    ).to(device).reshape(1, -1)
    scores = retriever(language, candidates)
    labels = torch.from_numpy(targets.astype(np.float32)).to(device)
    positives = float(labels.sum().item())
    negatives = float(len(labels) - positives)
    pos_weight = torch.tensor(
        max(negatives / max(positives, 1.0), 1.0), device=device
    )
    loss = F.binary_cross_entropy_with_logits(
        scores, labels, pos_weight=pos_weight
    )
    with torch.no_grad():
        order = topk_indices(scores.detach().cpu().numpy(), 5)
        recalls = recall_at_k(
            scores.detach().cpu().numpy(), targets, (1, 5, 10)
        )
    return loss, {
        "positives": int(positives),
        "candidates": int(len(labels)),
        "top5_hit": bool(targets[order].any()),
        **recalls,
    }


@torch.no_grad()
def evaluate_retrieval(
    samples: Sequence[RetrievalSample],
    retriever: LanguageMapRetriever,
    encoder: torch.nn.Module | None,
    device: torch.device,
    *,
    use_encoder: bool = False,
) -> dict:
    retriever.eval()
    if encoder is not None:
        encoder.eval()
    per_split: dict[str, list[dict]] = {}
    for sample in samples:
        targets = positive_mask_for_target(
            sample.library.centers, sample.target_xy
        )
        if not bool(targets.any()):
            per_split.setdefault(sample.split, []).append(
                {"positive": 0, "recall@1": None, "recall@5": None, "recall@10": None}
            )
            continue
        candidates = _candidate_tokens(
            sample, encoder, device,
            use_encoder=use_encoder,
            requires_grad=False,
        )
        language = torch.from_numpy(
            np.asarray(sample.language_features, dtype=np.float32)
        ).to(device).reshape(1, -1)
        scores = retriever(language, candidates).detach().cpu().numpy()
        result = recall_at_k(scores, targets, (1, 5, 10))
        result["top5_indices"] = topk_indices(scores, 5).tolist()
        result["positive_count"] = int(targets.sum())
        per_split.setdefault(sample.split, []).append(result)
    summary = {}
    for split, rows in per_split.items():
        valid = [row for row in rows if row["positive"] != 0]
        summary[split] = {
            "samples": len(rows),
            "with_positive": len(valid),
            "no_positive": len(rows) - len(valid),
            "recall@1": (
                float(np.mean([row["recall@1"] for row in valid]))
                if valid else None
            ),
            "recall@5": (
                float(np.mean([row["recall@5"] for row in valid]))
                if valid else None
            ),
            "recall@10": (
                float(np.mean([row["recall@10"] for row in valid]))
                if valid else None
            ),
        }
    return summary


def train_retriever(
    train_samples: Sequence[RetrievalSample],
    val_samples: Sequence[RetrievalSample],
    *,
    encoder: torch.nn.Module | None,
    device: torch.device,
    steps: int = 100,
    lr: float = 1e-3,
    encoder_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    train_encoder: bool = False,
    seed: int = 0,
    log_every: int = 20,
    eval_every: int = 20,
) -> tuple[LanguageMapRetriever, dict, dict]:
    """Train the retriever; target labels are used only inside the loss."""
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    retriever = LanguageMapRetriever().to(device)
    params = list(retriever.parameters())
    groups = [{"params": params, "lr": float(lr)}]
    if train_encoder:
        if encoder is None:
            raise ValueError("train_encoder requires encoder")
        encoder.requires_grad_(True)
        groups.append({"params": list(encoder.parameters()), "lr": float(encoder_lr)})
    else:
        if encoder is not None:
            encoder.requires_grad_(False)
            encoder.eval()
    optimizer = torch.optim.AdamW(groups, weight_decay=float(weight_decay))
    best_recall = -1.0
    best_state = None
    best_encoder_state = None
    best_val_summary = {}
    trace = []
    usable = [
        sample for sample in train_samples
        if bool(positive_mask_for_target(
            sample.library.centers, sample.target_xy
        ).any())
    ]
    if not usable:
        raise ValueError("no training sample has a positive candidate")
    for step in range(int(steps)):
        sample = usable[int(rng.integers(0, len(usable)))]
        retriever.train()
        if train_encoder and encoder is not None:
            encoder.train()
        optimizer.zero_grad(set_to_none=True)
        result = _loss_for_sample(
            sample, retriever, encoder, device, train_encoder=train_encoder
        )
        if result is None:
            continue
        loss, metrics = result
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in params if p.grad is not None]
            + (
                [p for p in encoder.parameters() if p.grad is not None]
                if train_encoder and encoder is not None else []
            ),
            1.0,
        )
        optimizer.step()
        if step % max(1, int(log_every)) == 0 or step == int(steps) - 1:
            trace.append(
                {
                    "step": int(step),
                    "loss": round(float(loss.item()), 6),
                    "map": sample.map_name,
                    **metrics,
                }
            )
        should_evaluate = (
            (step + 1) % max(1, int(eval_every)) == 0
            or step == int(steps) - 1
        )
        if should_evaluate and val_samples:
            summary = evaluate_retrieval(
                val_samples, retriever, encoder, device,
                use_encoder=train_encoder,
            )
            recalls = [
                values["recall@1"]
                for values in summary.values()
                if values["recall@1"] is not None
            ]
            mean_recall = float(np.mean(recalls)) if recalls else 0.0
            if mean_recall > best_recall:
                best_recall = mean_recall
                best_val_summary = summary
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in retriever.state_dict().items()
                }
                if train_encoder and encoder is not None:
                    best_encoder_state = {
                        key: value.detach().cpu().clone()
                        for key, value in encoder.state_dict().items()
                    }
    if best_state is not None:
        retriever.load_state_dict(best_state)
    if best_encoder_state is not None and encoder is not None:
        encoder.load_state_dict(best_encoder_state)
    if not best_val_summary:
        best_val_summary = evaluate_retrieval(
            val_samples, retriever, encoder, device,
            use_encoder=train_encoder,
        )
    return retriever, best_val_summary, {
        "trace": trace,
        "best_recall@1": best_recall,
        "best_retriever_state": best_state,
        "best_encoder_state": best_encoder_state,
        "train_samples": len(usable),
        "val_samples": len(val_samples),
    }
