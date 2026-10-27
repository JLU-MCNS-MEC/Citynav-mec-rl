"""Language feature interface for the MEC models.

The MEC-V8.2 models had no language branch.  This module supplies a frozen text
feature extractor, so the only trainable language component is a small linear
projection inside each MEC model.  Two backends are provided:

* ``bert``   - a frozen ``bert-base-uncased`` from the local HuggingFace cache.
* ``hashing``- a deterministic hashed word/character n-gram encoder that needs
  no model download.  It is the fallback for offline smoke tests.

Both return a fixed-dimensional vector per instruction.  The MEC models project
it to their own token dimension; the text encoder itself is never fine-tuned.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.config import TextConfig


_WORD_RE = re.compile(r"[a-z0-9]+")


class HashingTextEncoder:
    backend = "hashing"

    def __init__(self, dim: int = 384, char_weight: float = 0.5):
        self.dim = int(dim)
        self.char_weight = float(char_weight)
        self.output_dim = self.dim

    @staticmethod
    def _hash(feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % 10_000_003
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = _WORD_RE.findall(text.lower())
        features: list[tuple[str, float]] = []
        for word in words:
            features.append((f"w:{word}", 1.0))
            padded = f"_{word}_"
            for index in range(max(1, len(padded) - 2)):
                features.append((f"c:{padded[index:index + 3]}", self.char_weight))
        for left, right in zip(words[:-1], words[1:]):
            features.append((f"b:{left}_{right}", 1.0))
        return features

    def encode_numpy(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(str(text)):
                index, sign = self._hash(feature)
                out[row, index % self.dim] += sign * weight
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out

    def encode_tensor(
        self, texts: Sequence[str], device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        return torch.from_numpy(self.encode_numpy(texts)).to(device)

    def close(self) -> None:
        return None


class BertTextEncoder:
    backend = "bert"

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        max_length: int = 48,
        device: str | torch.device = "cpu",
        local_files_only: bool = True,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.output_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode_tensor(
        self, texts: Sequence[str], device: str | torch.device | None = None
    ) -> torch.Tensor:
        target = torch.device(device) if device is not None else self.device
        batch = self.tokenizer(
            [str(text) for text in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(target) for key, value in batch.items()}
        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)

    def encode_numpy(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_tensor(texts).cpu().numpy()

    def close(self) -> None:
        return None


def build_language_adapter(
    config: TextConfig | None = None,
    *,
    device: str | torch.device = "cpu",
) -> tuple[object, int, str, str | None]:
    """Return ``(adapter, output_dim, actual_backend, warning)``."""
    config = config or TextConfig()
    backend = str(config.backend).lower()
    warning: str | None = None

    if backend in {"bert", "auto"}:
        try:
            adapter = BertTextEncoder(
                config.model_name,
                max_length=config.max_length,
                device=device,
                local_files_only=config.local_files_only,
            )
            return adapter, adapter.output_dim, "bert", warning
        except Exception as exc:  # noqa: BLE001 - offline fallback is intentional
            if backend == "bert":
                raise
            warning = (
                "BERT text encoder unavailable; falling back to hashing "
                f"({type(exc).__name__}: {exc})"
            )
            warnings.warn(warning, RuntimeWarning, stacklevel=2)

    if backend not in {"hashing", "auto"}:
        raise ValueError(f"unknown text backend: {config.backend}")
    adapter = HashingTextEncoder(dim=config.hashing_dim)
    return adapter, adapter.output_dim, "hashing", warning
