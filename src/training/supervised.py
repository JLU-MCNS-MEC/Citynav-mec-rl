"""Supervised movement/arrival loss and a one-batch training step."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TrainingConfig
from src.data.actions import ACTION_NAMES, NUM_ACTIONS, STOP
from src.models.mec_common import combine_move_and_arrival


_MODEL_INPUT_KEYS = (
    "current_tokens",
    "goal_tokens",
    "heatmap",
    "network_state",
    "language_features",
    "history_vis",
    "history_pose",
    "history_action",
    "history_mask",
    "gab",
    "map_feat",
)


def model_inputs_from_batch(batch: dict) -> dict:
    """Select model kwargs from a supervision batch.

    The batch also carries labels such as ``actions``; those must not be
    forwarded to ``forward_components``.  Keeping this filtering in one place
    prevents a supervision-only key from reaching the model signature.
    """
    if "history_vis" in batch:
        required = (
            "current_tokens",
            "history_vis",
            "history_pose",
            "history_action",
            "history_mask",
            "language_features",
        )
    else:
        required = (
            "current_tokens",
            "goal_tokens",
            "heatmap",
            "network_state",
        )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError("batch is missing model input(s): " + ", ".join(missing))
    return {key: batch[key] for key in _MODEL_INPUT_KEYS if key in batch}


def factorized_loss(
    components: dict,
    actions: torch.Tensor,
    *,
    arrival_loss_weight: float = 1.0,
    move_class_weights: Optional[torch.Tensor] = None,
    arrival_pos_weight: Optional[torch.Tensor] = None,
) -> dict:
    """Movement CE on non-stop labels + binary arrival BCE on every label."""
    move_logits = components["move_logits"]
    arrival_logit = components["arrival_logit"]
    actions = torch.as_tensor(
        actions, dtype=torch.long, device=move_logits.device
    ).view(-1)
    if actions.numel() != move_logits.size(0):
        raise ValueError("actions and model outputs have different batch sizes")
    if bool(((actions < 0) | (actions >= NUM_ACTIONS)).any()):
        raise ValueError(f"actions must be in [0, {NUM_ACTIONS - 1}]")
    if move_logits.size(-1) != NUM_ACTIONS - 1:
        raise ValueError(
            "movement head must have five classes for the six-action interface"
        )

    move_mask = actions != int(STOP)
    if bool(move_mask.any()):
        # Movement head order: Forward, Right, Left, Up, Down.  Official action
        # IDs are 1..5, so subtract one to obtain the head target.
        move_targets = actions[move_mask] - 1
        move_loss = F.cross_entropy(
            move_logits[move_mask],
            move_targets,
            weight=move_class_weights,
        )
    else:
        move_loss = move_logits.sum() * 0.0

    arrival_target = (actions == int(STOP)).float()
    arrival_loss = F.binary_cross_entropy_with_logits(
        arrival_logit,
        arrival_target,
        pos_weight=arrival_pos_weight,
    )
    loss = move_loss + float(arrival_loss_weight) * arrival_loss

    joint_logits = combine_move_and_arrival(move_logits, arrival_logit)
    predictions = joint_logits.argmax(dim=-1)
    accuracy = float((predictions == actions).float().mean().item())

    if bool(move_mask.any()):
        move_accuracy = float(
            (predictions[move_mask] == actions[move_mask]).float().mean().item()
        )
    else:
        move_accuracy = 0.0

    stop_target = actions == int(STOP)
    stop_predictions = predictions == int(STOP)
    tp = int((stop_predictions & stop_target).sum().item())
    fp = int((stop_predictions & ~stop_target).sum().item())
    fn = int((~stop_predictions & stop_target).sum().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    per_class_counts = {
        ACTION_NAMES[action]: int((actions == int(action)).sum().item())
        for action in ACTION_NAMES
    }
    per_class_correct = {
        ACTION_NAMES[action]: int(
            ((actions == int(action)) & (predictions == int(action))).sum().item()
        )
        for action in ACTION_NAMES
    }
    per_class_accuracy = {
        name: (
            per_class_correct[name] / per_class_counts[name]
            if per_class_counts[name]
            else 0.0
        )
        for name in per_class_counts
    }

    return {
        "loss": loss,
        "move_loss": move_loss.detach(),
        "arrival_loss": arrival_loss.detach(),
        "logits": joint_logits,
        "predictions": predictions,
        "accuracy": accuracy,
        "move_accuracy": move_accuracy,
        "stop_precision": precision,
        "stop_recall": recall,
        "num_stop": int(stop_target.sum().item()),
        "num_move": int((~stop_target).sum().item()),
        "per_class_counts": per_class_counts,
        "per_class_correct": per_class_correct,
        "per_class_accuracy": per_class_accuracy,
    }


def gradient_norm(module: nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().double().pow(2).sum().item())
    return math.sqrt(squared)


def gradients_are_finite(module: nn.Module) -> bool:
    for parameter in module.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def parameter_fingerprint(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        total += float(parameter.detach().double().sum().item())
    return total


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    config: TrainingConfig | None = None,
    *,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    clip_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    move_class_weights: Optional[torch.Tensor] = None,
    arrival_pos_weight: Optional[torch.Tensor] = None,
) -> dict:
    config = config or TrainingConfig()
    clip_parameters = (
        list(clip_parameters)
        if clip_parameters is not None
        else list(model.parameters())
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    components = model.forward_components(**model_inputs_from_batch(batch))
    losses = factorized_loss(
        components,
        batch["actions"],
        arrival_loss_weight=config.arrival_loss_weight,
        move_class_weights=move_class_weights,
        arrival_pos_weight=arrival_pos_weight,
    )
    loss = losses["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite training loss: {loss.item()}")
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            clip_parameters, float(config.grad_clip)
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            clip_parameters, float(config.grad_clip)
        )
        optimizer.step()
    if not gradients_are_finite(model):
        raise FloatingPointError("non-finite gradients after backward")
    return {
        "loss": float(loss.detach().item()),
        "move_loss": float(losses["move_loss"].item()),
        "arrival_loss": float(losses["arrival_loss"].item()),
        "accuracy": losses["accuracy"],
        "move_accuracy": losses["move_accuracy"],
        "stop_precision": losses["stop_precision"],
        "stop_recall": losses["stop_recall"],
        "grad_norm": float(norm.detach().item() if torch.is_tensor(norm) else norm),
        "num_stop": losses["num_stop"],
        "num_move": losses["num_move"],
        "per_class_counts": losses["per_class_counts"],
        "per_class_correct": losses["per_class_correct"],
        "per_class_accuracy": losses["per_class_accuracy"],
    }


@torch.no_grad()
def evaluate_components(model: nn.Module, batch: dict) -> dict:
    model.eval()
    components = model.forward_components(**model_inputs_from_batch(batch))
    losses = factorized_loss(components, batch["actions"])
    return {
        "loss": float(losses["loss"].item()),
        "accuracy": losses["accuracy"],
        "move_accuracy": losses["move_accuracy"],
        "stop_precision": losses["stop_precision"],
        "stop_recall": losses["stop_recall"],
        "per_class_counts": losses["per_class_counts"],
        "per_class_correct": losses["per_class_correct"],
        "per_class_accuracy": losses["per_class_accuracy"],
    }
