#!/usr/bin/env python3
"""Stage 17-A-R2 multimodal deployment / uplink payload audit.

Every entry below was checked against the actual tensors the five frozen
experts consume (`env16.EpisodeEnv16.step`, `models*.forward`), not inferred
from names:

    current        (49, 2048) float32   FrozenEncoder layer4 tokens
    target_emb     (512,)     float32   layer2 goal descriptor, per episode
    history_vis    (16, 2048) float32   pooled history tokens
    history_pose   (16, 3)    float32   relative pose, 16 slots
    history_action (16, 4)    float32   one-hot action, 16 slots
    history_mask   (16,)      float32   1.0 for valid slots
    belief         (10,)      float32   belief features
    match           (5,)      float32   dx, dy, score, margin, sharpness
    landmarks      (8, 3)     float32   [mean_x, mean_y, weight] x 8
    states          (3,)      float32   x, y, yaw
    relation       NOT uploaded -- the MEC derives it from landmarks + states
                     inside TransformerM12Fix.landmark_geometry

Serialisation precision is a *communication* choice and is independent of the
compute precision (the encoders run in fp16 but their outputs are float32).
The first version therefore uploads FP32 (`feature_bits = 32`); an FP16
option exists for sensitivity studies but is not enabled.

Caching policy: `none` (v1).  Nothing is assumed to be pre-placed on the MEC,
so every modality that a node does not already hold is charged every step.
A future static cache (e.g. target_emb replicated to all 10 nodes) must be
defined together with its establishment cost and invalidation rule; until
then it is not free.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from config17 import Stage17Config


@dataclass(frozen=True)
class Modality:
    name: str
    shape: tuple
    elements: int
    source: str
    generated_by: str
    needed_by_mec: bool
    uploaded: bool
    cacheable: bool
    note: str = ""


def modality_spec(cfg: Stage17Config) -> list[Modality]:
    return [
        Modality("current_visual", (49, 2048), 49 * 2048,
                 "FrozenEncoder(layer4) on the current 224x224 crop", "UAV", True, True,
                 False, "main visual input of every expert"),
        Modality("target_emb", (512,), 512,
                 "layer2 goal descriptor, computed once per episode", "UAV", True, True,
                 True, "constant during an episode; cacheable in principle, charged in v1"),
        Modality("history_visual", (16, 2048), 16 * 2048,
                 "pooled layer4 tokens of the previous 16 steps", "UAV", True, True,
                 False, "rebuilt every step, so re-sent every step"),
        Modality("history_pose", (16, 3), 16 * 3,
                 "relative pose of the previous 16 steps", "UAV", True, True, False, ""),
        Modality("history_action", (16, 4), 16 * 4,
                 "one-hot previous actions", "UAV", True, True, False, ""),
        Modality("history_mask", (16,), 16,
                 "validity mask of the 16 slots", "UAV", True, True, False,
                 "could be 8 bits (a count); modelled as 16 x float32"),
        Modality("belief", (10,), 10,
                 "belief_features()", "UAV", True, True, False, ""),
        Modality("match", (5,), 5,
                 "subpixel_match() -> dx, dy, score, margin, sharpness", "UAV",
                 True, True, False, ""),
        Modality("landmarks", (8, 3), 8 * 3,
                 "landmark_inputs()", "UAV", True, True, False, ""),
        Modality("uav_state", (3,), 3,
                 "UAV x, y, yaw", "UAV", True, True, False, ""),
        Modality("relation", (1, 8, 3), 24,
                 "landmark_geometry(landmarks, states)", "MEC", True, False,
                 False, "derived on the MEC, therefore NOT uploaded"),
    ]


def calculate_uplink_payload_bits(cfg: Stage17Config,
                                  caching_policy: str = "none") -> dict:
    """Return per-modality bits, totals and flags.

    `caching_policy` is intentionally limited to "none": any other value would
    require a cache-establishment protocol that does not exist yet.
    """
    if caching_policy != "none":
        raise NotImplementedError(
            f"caching policy {caching_policy!r} has no implemented protocol; "
            "use 'none' or define the cache establishment cost first")
    bits_per_elem = cfg.feature_bits
    rows, total = [], 0
    for m in modality_spec(cfg):
        bits = m.elements * bits_per_elem if m.uploaded else 0
        total += bits
        rows.append({**asdict(m), "bits_per_element": bits_per_elem,
                     "payload_bits": int(bits),
                     "cache_hit": False})
    return {
        "modalities": rows,
        "uplink_payload_bits": int(total),
        "uplink_payload_bytes": int(total) // 8,
        "bits_per_element": bits_per_elem,
        "caching_policy": caching_policy,
        "derived_not_uploaded": [m["name"] for m in rows if not m["uploaded"]],
    }
