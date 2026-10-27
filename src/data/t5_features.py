"""T5 + language feature construction for CityNav.

T5 base modalities (ToyEnv V8.2 Stage-2i/2j), adapted to six actions:

  Current RGB tokens        [576, 384]
  Goal RGB tokens           [3, 384]   (retrieved map candidates)
  History visual            [16, 768]  (mean+max pooled past current tokens)
  Pose history              [16, 12]   (6 continuous + 6 action one-hot)
  Action history            [16, 6]    (executed six-action one-hot)
  History mask              [16]
  GAB                       [20]
  Global map retrieval      [22]       (Top-5 anonymous candidates + 2 summary)
  Language features         [768]      (frozen BERT)

The no-goal / no-map baseline omits Goal, GAB evidence and map features; its
history/pose/language/current inputs are the same.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from src.data.actions import (
    FORWARD_STEP_M,
    NUM_ACTIONS,
    TURN_STEP_DEG,
)
from src.data.trajectory_adapter import Pose5D, wrap_angle


HISTORY_STEPS = 16
VISUAL_DESCRIPTOR_DIM = 768
POSE_FEATURE_DIM = 12
ACTION_HISTORY_DIM = 6
GAB_FEATURE_DIM = 20
MAP_FEATURE_DIM = 22
LANGUAGE_DIM = 768
TURN_STEP_RAD = math.radians(float(TURN_STEP_DEG))
RELATIVE_POSITION_SCALE_M = 100.0

T5_FEATURE_DIMS = {
    "current_tokens": (576, 384),
    "goal_tokens": (3, 384),
    "history_vis": (HISTORY_STEPS, VISUAL_DESCRIPTOR_DIM),
    "history_pose": (HISTORY_STEPS, POSE_FEATURE_DIM),
    "history_action": (HISTORY_STEPS, ACTION_HISTORY_DIM),
    "history_mask": (HISTORY_STEPS,),
    "gab": (GAB_FEATURE_DIM,),
    "map_feat": (MAP_FEATURE_DIM,),
    "language_features": (LANGUAGE_DIM,),
}


@torch.no_grad()
def visual_descriptor(tokens: torch.Tensor) -> torch.Tensor:
    """Mean + max pooled visual descriptor, matching ToyEnv T5."""
    if tokens.ndim != 3 or tokens.shape[-1] != 384:
        raise ValueError(
            f"tokens must be [B,576,384], got {tuple(tokens.shape)}"
        )
    return torch.cat(
        [tokens.float().mean(dim=1), tokens.float().max(dim=1).values],
        dim=-1,
    )


def action_one_hot(action: int) -> np.ndarray:
    one_hot = np.zeros(ACTION_HISTORY_DIM, dtype=np.float32)
    action = int(action)
    if 0 <= action < NUM_ACTIONS:
        one_hot[action] = 1.0
    return one_hot


def pose_action_feature(
    previous_pose: Pose5D,
    current_pose: Pose5D,
    reference_pose: Pose5D,
    action: int,
) -> np.ndarray:
    """6 continuous relative-motion features + six-action one-hot.

    Slot features are relative to the stated reference pose, so no absolute
    UAV or target coordinate enters the MEC.
    """
    dx = float(current_pose.x) - float(previous_pose.x)
    dy = float(current_pose.y) - float(previous_pose.y)
    cos_p, sin_p = math.cos(previous_pose.yaw), math.sin(previous_pose.yaw)
    motion_forward = (
        dx * cos_p + dy * sin_p
    ) / float(FORWARD_STEP_M)
    motion_right = (
        -dx * sin_p + dy * cos_p
    ) / float(FORWARD_STEP_M)
    motion_yaw = wrap_angle(current_pose.yaw - previous_pose.yaw) / TURN_STEP_RAD

    gx = float(previous_pose.x) - float(reference_pose.x)
    gy = float(previous_pose.y) - float(reference_pose.y)
    cos_r, sin_r = math.cos(reference_pose.yaw), math.sin(reference_pose.yaw)
    relative_forward = (
        gx * cos_r + gy * sin_r
    ) / RELATIVE_POSITION_SCALE_M
    relative_right = (
        -gx * sin_r + gy * cos_r
    ) / RELATIVE_POSITION_SCALE_M
    relative_yaw = wrap_angle(previous_pose.yaw - reference_pose.yaw) / math.pi

    return np.concatenate(
        [
            np.asarray(
                [
                    motion_forward,
                    motion_right,
                    motion_yaw,
                    relative_forward,
                    relative_right,
                    relative_yaw,
                ],
                dtype=np.float32,
            ),
            action_one_hot(action),
        ]
    ).astype(np.float32)


def history_arrays(
    poses: Sequence[Pose5D],
    actions: Sequence[int],
    index: int,
    visuals: np.ndarray,
    *,
    steps: int = HISTORY_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal history for state ``index``; missing slots are masked zeros."""
    if len(poses) != len(actions):
        raise ValueError("poses and actions must have the same length")
    if len(visuals) != len(poses):
        raise ValueError("visuals must have one descriptor per state")
    if not 0 <= int(index) < len(poses):
        raise IndexError(index)
    visual = np.zeros((int(steps), VISUAL_DESCRIPTOR_DIM), dtype=np.float32)
    pose = np.zeros((int(steps), POSE_FEATURE_DIM), dtype=np.float32)
    action = np.zeros((int(steps), ACTION_HISTORY_DIM), dtype=np.float32)
    mask = np.zeros((int(steps),), dtype=np.float32)
    reference = poses[int(index)]
    for offset in range(1, int(steps) + 1):
        source = int(index) - offset
        if source < 0:
            continue
        visual[offset - 1] = np.asarray(visuals[source], dtype=np.float32)
        pose[offset - 1] = pose_action_feature(
            poses[source],
            poses[source + 1],
            reference,
            actions[source],
        )
        action[offset - 1] = action_one_hot(actions[source])
        mask[offset - 1] = 1.0
    return visual, pose, action, mask


def history_mask_is_causal(mask: np.ndarray) -> bool:
    """A valid causal history mask is a prefix of ones, then zeros."""
    values = np.asarray(mask, dtype=np.float32).reshape(-1)
    return bool(
        np.all(values[: int(values.sum())] == 1.0)
        and np.all(values[int(values.sum()):] == 0.0)
    )
