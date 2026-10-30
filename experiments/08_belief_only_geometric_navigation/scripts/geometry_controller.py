#!/usr/bin/env python3
"""Belief-only geometric controllers (Stage 08).

The controller sees **one thing**: the 12-d global goal belief produced by
``pipeline.belief_features``.  It has no reference to the task, the map, the
true target, the evaluator, the Transformer, or any observation encoder.

Belief layout (see pipeline.py):
    0  top-1 similarity          1  top-5 mean similarity   2  entropy
    3  top-1 forward offset (m)  4  top-1 left offset (m)   5  top-1 distance (m)
    6  top-5 weighted forward    7  top-5 weighted left     8  weighted distance
    9  spread                   10 candidate-count proxy    11 constant 1

Controller A (primary, pre-registered): top-1 geometry.
Controller B (control): top-5 weighted geometry.
Both use the fixed 25 deg alignment threshold (consistent with the 30 deg
discrete turn of the teacher) and a 20 m estimated-distance STOP radius.
Thresholds are fixed by the protocol and were not tuned on any split.
"""

from __future__ import annotations

import math

import numpy as np

ALIGN_DEG = 25.0
ALIGN_RAD = math.radians(ALIGN_DEG)
STOP_RADIUS_M = 20.0
STOP, FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


class _Base:
    """All the controller is allowed to touch: a (12,) belief vector."""

    name = "base"

    def _components(self, belief: np.ndarray) -> tuple[float, float, float]:
        raise NotImplementedError

    def act(self, belief: np.ndarray) -> int:
        if belief.shape != (12,):
            raise ValueError(f"controller expects a 12-d belief, got {belief.shape}")
        forward, left, distance = self._components(belief)
        if distance <= STOP_RADIUS_M:
            return STOP
        bearing = math.atan2(float(left), float(forward))
        if bearing > ALIGN_RAD:
            return TURN_LEFT
        if bearing < -ALIGN_RAD:
            return TURN_RIGHT
        return FORWARD

    def diagnostics(self, belief: np.ndarray) -> dict:
        forward, left, distance = self._components(belief)
        return {"est_forward_m": float(forward), "est_left_m": float(left),
                "est_distance_m": float(distance),
                "est_bearing_deg": math.degrees(math.atan2(float(left), float(forward)))}


class Top1Geometry(_Base):
    """Primary baseline: the single best visual candidate."""

    name = "top1"

    def _components(self, belief):
        return belief[3], belief[4], belief[5]


class Top5WeightedGeometry(_Base):
    """Sensitivity control: similarity-weighted top-5 geometry."""

    name = "top5"

    def _components(self, belief):
        return belief[6], belief[7], belief[8]


CONTROLLERS = {"top1": Top1Geometry, "top5": Top5WeightedGeometry}


def make(name: str) -> _Base:
    if name not in CONTROLLERS:
        raise KeyError(f"unknown controller {name!r}; choose from {sorted(CONTROLLERS)}")
    return CONTROLLERS[name]()
