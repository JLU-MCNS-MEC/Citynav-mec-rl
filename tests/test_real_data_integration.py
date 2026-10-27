"""Integration test against the read-only CityNav tree on this server.

It is skipped automatically when no local data configuration is available.
The test only reads raw data; it never writes into the shared tree.
"""

from __future__ import annotations

import pytest
import torch

from src.config import ObservationConfig, TextConfig, load_data_config
from src.data.citynav_dataset import CityNavDataset
from src.data.fpv_observation import FPVObservationAdapter
from src.data.language_adapter import build_language_adapter
from src.models import SmallMEC
from src.models.token_utils import encode_scene
from src.models.vision_encoder import VisualTokenEncoder


def _available_config():
    try:
        config = load_data_config()
        config.validate()
    except Exception as exc:  # noqa: BLE001 - absent server data is a skip
        pytest.skip(f"CityNav data is not configured: {type(exc).__name__}: {exc}")
    if not config.split_path("val_unseen").exists():
        pytest.skip("citynav_val_unseen.json is missing")
    return config


def test_val_unseen_episode_observation_encoder_and_small_model() -> None:
    config = _available_config()
    dataset = CityNavDataset(config, "val_unseen", max_episodes=3)
    assert len(dataset) > 0
    record = dataset[0]
    if not config.ply_path(record.map_name).exists():
        pytest.skip(f"PLY is missing for {record.map_name}")
    assert record.object_info is not None
    assert record.description_match is True
    assert record.teacher().steps >= 1

    observation_config = ObservationConfig(
        fpv_width=336,
        fpv_height=336,
        fpv_crop_radius_m=100.0,
        fpv_splat_radius=0,
    )
    adapter = FPVObservationAdapter(config, observation_config)
    try:
        observation = adapter.render(
            record.map_name,
            record.poses[0],
            record.object_info,
            crop_radius_m=100.0,
        )
    finally:
        adapter.close()

    assert observation["current_rgb"].shape == (336, 336, 3)
    assert observation["depth"].shape == (336, 336, 1)
    assert observation["depth_quality"]["valid_pixel_ratio"] > 0.05
    assert observation["goal_rgbs"].shape == (3, 96, 96, 3)

    encoder = VisualTokenEncoder().eval()
    scene = encode_scene(
        encoder,
        observation["current_rgb"][None],
        observation["goal_rgbs"][None],
    )
    assert scene["current_tokens"].shape == (1, 576, 384)
    assert scene["goal_tokens"].shape == (1, 3, 384)
    assert scene["heatmap"].shape == (1, 3, 8, 8)

    text_adapter, text_dim, backend, _warning = build_language_adapter(
        TextConfig(backend="hashing")
    )
    assert backend == "hashing"
    language = text_adapter.encode_tensor([record.description])
    assert language.shape == (1, text_dim)

    model = SmallMEC(text_dim=text_dim).eval()
    with torch.no_grad():
        logits = model(
            scene["current_tokens"],
            scene["goal_tokens"],
            scene["heatmap"],
            torch.zeros(1, 5),
            language,
        )
    assert logits.shape == (1, 6)
    assert torch.isfinite(logits).all()
