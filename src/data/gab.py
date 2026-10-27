"""Goal-Anchor Belief (GAB) for the six-action T5 interface.

Adapted from ToyEnv V8.2 Stage-2h/2i.  The belief is causal, reset per
episode, and built only from an observable 3x8x8 Current/Goal similarity
heatmap plus the previously executed action.  No true target coordinate,
distance, future action, success label or object ID is read.

The only protocol change is the turn step: 30 degrees instead of ToyEnv's
15 degrees, matching the official CityNav six-action interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.actions import MOVE_FORWARD, TURN_LEFT, TURN_RIGHT


BINS = 8
FOV_DEG = 90.0
TURN_DEG = 30.0
ALPHA = 0.50
BETA = 0.25
FORWARD_BETA = 0.15
MAX_AGE = 16
GAB_FEATURE_DIM = 20


def uniform_belief() -> np.ndarray:
    return np.full(BINS, 1.0 / BINS, dtype=np.float64)


def normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = values / max(values.sum(), 1e-12)
    entropy = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum()
    return float(entropy / math.log(len(values)))


def current_evidence(maps: np.ndarray, temperature: float = 0.10) -> dict:
    """Bearing evidence from a [3,8,8] observable similarity heatmap."""
    heatmaps = np.asarray(maps, dtype=np.float64)
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]
    if heatmaps.shape != (3, BINS, BINS):
        raise ValueError(f"heatmap must be [3,8,8], got {heatmaps.shape}")
    heat = heatmaps.max(axis=0).reshape(-1)
    logits = heat / float(temperature)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= max(probabilities.sum(), 1e-12)
    grid = probabilities.reshape(BINS, BINS)
    bins = grid.sum(axis=0)
    bins /= max(bins.sum(), 1e-12)
    return {
        "bins": bins.astype(np.float64),
        "confidence": float(probabilities.max()),
        "entropy": normalized_entropy(bins),
    }


def shift_belief(
    probabilities: np.ndarray,
    previous_action: int,
    *,
    turn_deg: float = TURN_DEG,
    fov_deg: float = FOV_DEG,
) -> np.ndarray:
    """Rotate an egocentric belief after a yaw action.

    Positive bin index is rightward in the image.  Turning LEFT moves a
    world-fixed target to the right, so the prior shifts positively.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    action = int(previous_action)
    if action not in (int(TURN_LEFT), int(TURN_RIGHT)):
        return values.copy()
    bin_width = float(fov_deg) / BINS
    shift = float(turn_deg) / bin_width
    if action == int(TURN_RIGHT):
        shift = -shift
    source = np.arange(BINS, dtype=np.float64) - shift
    lower = np.floor(source).astype(int)
    fraction = source - lower
    output = (
        (1.0 - fraction) * values[lower % BINS]
        + fraction * values[(lower + 1) % BINS]
    )
    return output / max(output.sum(), 1e-12)


@dataclass(frozen=True)
class GABConfig:
    reliability_threshold: float
    alpha: float = ALPHA
    beta: float = BETA
    forward_beta: float = FORWARD_BETA
    max_age: int = MAX_AGE
    bins: int = BINS
    fov_deg: float = FOV_DEG
    turn_deg: float = TURN_DEG


class GABState:
    def __init__(self, config: GABConfig):
        self.config = config
        self.belief = uniform_belief()
        self.last_reliable = uniform_belief()
        self.confidence = 0.0
        self.last_seen_age = 0
        self.valid = 0.0

    def update(self, maps: np.ndarray, previous_action: int) -> dict:
        prior = shift_belief(
            self.belief, previous_action,
            turn_deg=self.config.turn_deg,
            fov_deg=self.config.fov_deg,
        )
        evidence = current_evidence(maps)
        reliable = evidence["confidence"] >= self.config.reliability_threshold
        if reliable:
            self.belief = (
                (1.0 - self.config.alpha) * prior
                + self.config.alpha * evidence["bins"]
            )
            self.belief /= max(self.belief.sum(), 1e-12)
            self.last_reliable = evidence["bins"].copy()
            self.confidence = (
                (1.0 - self.config.alpha) * self.confidence
                + self.config.alpha * evidence["confidence"]
            )
            self.last_seen_age = 0
            self.valid = 1.0
        else:
            decay = (
                self.config.forward_beta
                if int(previous_action) == int(MOVE_FORWARD)
                else self.config.beta
            )
            self.belief = (
                (1.0 - decay) * prior + decay * uniform_belief()
            )
            self.belief /= max(self.belief.sum(), 1e-12)
            self.confidence *= (1.0 - decay)
            self.last_seen_age = min(
                self.last_seen_age + 1, self.config.max_age
            )
            if self.last_seen_age >= self.config.max_age:
                self.valid = 0.0
        return evidence

    def features(self) -> np.ndarray:
        return np.concatenate(
            [
                self.belief,
                self.last_reliable,
                np.asarray(
                    [
                        self.confidence,
                        normalized_entropy(self.belief),
                        self.last_seen_age / max(self.config.max_age, 1),
                        self.valid,
                    ],
                    dtype=np.float64,
                ),
            ]
        ).astype(np.float32)


def fit_reliability_threshold(
    heatmaps: Sequence[np.ndarray],
    percentile: float = 60.0,
) -> float:
    confidences = np.asarray(
        [current_evidence(np.asarray(h, dtype=np.float64))["confidence"]
         for h in heatmaps],
        dtype=np.float64,
    )
    if len(confidences) == 0:
        raise ValueError("at least one heatmap is required")
    return float(np.percentile(confidences, float(percentile)))


def build_gab_sequence(
    heatmaps: Sequence[np.ndarray],
    actions: Sequence[int],
    config: GABConfig,
    *,
    step_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Causal GAB features for one ordered episode sequence."""
    if len(heatmaps) != len(actions):
        raise ValueError("heatmaps and actions must have the same length")
    state = GABState(config)
    features = np.zeros((len(heatmaps), GAB_FEATURE_DIM), dtype=np.float32)
    previous_action = -1
    previous_step = None
    for index in range(len(heatmaps)):
        if step_ids is not None:
            step = int(step_ids[index])
            if previous_step is not None and step != previous_step + 1:
                state = GABState(config)
                previous_action = -1
            previous_step = step
        state.update(np.asarray(heatmaps[index]), previous_action)
        features[index] = state.features()
        previous_action = int(actions[index])
    return features


def zero_gab_sequence(length: int) -> np.ndarray:
    """Map-free baseline GAB: uniform belief, valid=0, no map evidence."""
    state = GABState(GABConfig(reliability_threshold=1e9))
    state.valid = 0.0
    state.confidence = 0.0
    state.last_seen_age = state.config.max_age
    return np.repeat(state.features()[None, :], int(length), axis=0)
