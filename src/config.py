"""Configuration objects for CityNav data adaptation and smoke tests.

No server absolute path is stored in tracked files.  Data locations are read
from a git-ignored YAML file or from the ``CITYNAV_DATA_ROOT`` /
``CITYNAV_PLY_ROOT`` environment variables.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.data.actions import NUM_ACTIONS, NUM_MOVE_ACTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_UNRESOLVED_ENV = re.compile(r"\$\{[^}]+\}")


def _expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_value(item) for key, item in value.items()}
    return value


def load_yaml(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    payload = _expand_value(payload)
    unresolved = _find_unresolved(payload)
    if unresolved:
        raise ValueError(
            "unresolved environment variable(s) in config: "
            + ", ".join(sorted(unresolved))
        )
    return payload


def _find_unresolved(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.update(_UNRESOLVED_ENV.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            found.update(_find_unresolved(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_find_unresolved(item))
    return found


def _get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = mapping.get(key, default)
    if value is None and default is not None:
        return default
    return value


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    ply_root: Path
    citynav_subdir: str = "original_citynav"
    cityrefer_subdir: str = "cityrefer"
    rgbd_subdir: str = "rgbd"
    train_split: str = "train_seen"
    val_splits: tuple[str, ...] = ("val_seen", "val_unseen")
    test_split: str = "test_unseen"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DataConfig":
        data_root = mapping.get("data_root")
        ply_root = mapping.get("ply_root")
        if not data_root or not ply_root:
            raise ValueError(
                "data_root and ply_root are required; set CITYNAV_DATA_ROOT / "
                "CITYNAV_PLY_ROOT or use configs/data_config.local.yaml"
            )
        return cls(
            data_root=Path(str(data_root)).expanduser().resolve(),
            ply_root=Path(str(ply_root)).expanduser().resolve(),
            citynav_subdir=str(_get(mapping, "citynav_subdir", "original_citynav")),
            cityrefer_subdir=str(_get(mapping, "cityrefer_subdir", "cityrefer")),
            rgbd_subdir=str(_get(mapping, "rgbd_subdir", "rgbd")),
            train_split=str(_get(mapping, "train_split", "train_seen")),
            val_splits=tuple(_get(mapping, "val_splits", ("val_seen", "val_unseen"))),
            test_split=str(_get(mapping, "test_split", "test_unseen")),
        )

    @classmethod
    def from_env(cls) -> "DataConfig":
        mapping = {
            "data_root": os.environ.get("CITYNAV_DATA_ROOT"),
            "ply_root": os.environ.get("CITYNAV_PLY_ROOT"),
            "citynav_subdir": os.environ.get(
                "CITYNAV_CITYNAV_SUBDIR", "original_citynav"
            ),
            "cityrefer_subdir": os.environ.get(
                "CITYNAV_CITYREFER_SUBDIR", "cityrefer"
            ),
            "rgbd_subdir": os.environ.get("CITYNAV_RGBD_SUBDIR", "rgbd"),
        }
        return cls.from_mapping(mapping)

    @property
    def citynav_dir(self) -> Path:
        return self.data_root / self.citynav_subdir

    @property
    def cityrefer_dir(self) -> Path:
        return self.data_root / self.cityrefer_subdir

    @property
    def rgbd_dir(self) -> Path:
        return self.data_root / self.rgbd_subdir

    def split_path(self, split: str) -> Path:
        return self.citynav_dir / f"citynav_{split}.json"

    def rgb_path(self, map_name: str) -> Path:
        return self.rgbd_dir / f"{map_name}.png"

    def height_path(self, map_name: str) -> Path:
        return self.rgbd_dir / f"{map_name}.tif"

    def ply_path(self, map_name: str) -> Path:
        return self.ply_root / f"{map_name}.ply"

    def validate(self) -> None:
        required = [
            self.citynav_dir,
            self.cityrefer_dir,
            self.rgbd_dir,
            self.ply_root,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "missing read-only data director(ies): " + ", ".join(missing)
            )
        objects = self.cityrefer_dir / "objects.json"
        if not objects.exists():
            raise FileNotFoundError(objects)

    def redacted(self) -> dict:
        """Return a path-free summary suitable for logs / reports."""
        return {
            "data_root": "<CITYNAV_DATA_ROOT>",
            "ply_root": "<CITYNAV_PLY_ROOT>",
            "citynav_subdir": self.citynav_subdir,
            "cityrefer_subdir": self.cityrefer_subdir,
            "rgbd_subdir": self.rgbd_subdir,
            "train_split": self.train_split,
            "val_splits": list(self.val_splits),
            "test_split": self.test_split,
        }


@dataclass(frozen=True)
class ObservationConfig:
    image_size: int = 336
    depth_size: int = 336
    goal_crop_size: int = 96
    goal_scales: tuple[int, ...] = (14, 28, 56)
    map_cache_size: int = 2

    # First-person point-cloud renderer.  The camera is a pinhole with a square
    # 90-degree horizontal FOV; depth is metric forward depth.
    fpv_width: int = 512
    fpv_height: int = 512
    fpv_hfov_deg: float = 90.0
    fpv_near_m: float = 0.5
    fpv_far_m: float = 300.0
    fpv_crop_radius_m: float = 300.0
    fpv_splat_radius: int = 1
    # "trajectory" uses the pitch from CityNav direction vectors; "fixed" is
    # available for controlled diagnostics.
    fpv_pitch_mode: str = "trajectory"
    fpv_fixed_pitch_deg: float = -30.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ObservationConfig":
        mapping = mapping or {}
        return cls(
            image_size=int(_get(mapping, "image_size", 336)),
            depth_size=int(_get(mapping, "depth_size", 336)),
            goal_crop_size=int(_get(mapping, "goal_crop_size", 96)),
            goal_scales=tuple(int(v) for v in _get(mapping, "goal_scales", (14, 28, 56))),
            map_cache_size=int(_get(mapping, "map_cache_size", 2)),
            fpv_width=int(_get(mapping, "fpv_width", 512)),
            fpv_height=int(_get(mapping, "fpv_height", 512)),
            fpv_hfov_deg=float(_get(mapping, "fpv_hfov_deg", 90.0)),
            fpv_near_m=float(_get(mapping, "fpv_near_m", 0.5)),
            fpv_far_m=float(_get(mapping, "fpv_far_m", 300.0)),
            fpv_crop_radius_m=float(_get(mapping, "fpv_crop_radius_m", 300.0)),
            fpv_splat_radius=int(_get(mapping, "fpv_splat_radius", 1)),
            fpv_pitch_mode=str(_get(mapping, "fpv_pitch_mode", "trajectory")),
            fpv_fixed_pitch_deg=float(_get(mapping, "fpv_fixed_pitch_deg", -30.0)),
        )


@dataclass(frozen=True)
class TeacherConfig:
    # Official CityNav lookahead teacher defaults.
    forward_step_m: float = 5.0
    turn_step_deg: float = 30.0
    vertical_step_m: float = 2.0
    # Official episode generation uses LookaheadTeacherParams(lookahead=1).
    lookahead: int = 1
    stop_distance_m: float = 5.0
    success_radius_m: float = 20.0
    turn_threshold_deg: float = 30.0
    vertical_threshold_m: float = 1.0
    # Safety bound for the otherwise unbounded official while-loop.
    max_steps: int = 400

    @property
    def turn_step_rad(self) -> float:
        return float(self.turn_step_deg) * 3.141592653589793 / 180.0

    @property
    def turn_threshold_rad(self) -> float:
        return float(self.turn_threshold_deg) * 3.141592653589793 / 180.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "TeacherConfig":
        mapping = mapping or {}
        return cls(
            forward_step_m=float(_get(mapping, "forward_step_m", 5.0)),
            turn_step_deg=float(_get(mapping, "turn_step_deg", 30.0)),
            vertical_step_m=float(_get(mapping, "vertical_step_m", 2.0)),
            lookahead=int(_get(mapping, "lookahead", 1)),
            stop_distance_m=float(_get(mapping, "stop_distance_m", 5.0)),
            success_radius_m=float(_get(mapping, "success_radius_m", 20.0)),
            turn_threshold_deg=float(_get(mapping, "turn_threshold_deg", 30.0)),
            vertical_threshold_m=float(_get(mapping, "vertical_threshold_m", 1.0)),
            max_steps=int(_get(mapping, "max_steps", 400)),
        )


@dataclass(frozen=True)
class EncoderConfig:
    image_size: int = 336
    patch_size: int = 14
    token_dim: int = 384
    num_layers: int = 2
    num_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.10
    # Local-only path to the MEC-V8.2 shared_encoder.pt.  Never commit weights.
    checkpoint: str | None = None

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_tokens(self) -> int:
        return self.grid_size ** 2

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "EncoderConfig":
        mapping = mapping or {}
        checkpoint = _get(mapping, "checkpoint", None)
        return cls(
            image_size=int(_get(mapping, "image_size", 336)),
            patch_size=int(_get(mapping, "patch_size", 14)),
            token_dim=int(_get(mapping, "token_dim", 384)),
            num_layers=int(_get(mapping, "num_layers", 2)),
            num_heads=int(_get(mapping, "num_heads", 6)),
            mlp_ratio=int(_get(mapping, "mlp_ratio", 4)),
            dropout=float(_get(mapping, "dropout", 0.10)),
            checkpoint=None if not checkpoint else str(checkpoint),
        )


@dataclass(frozen=True)
class MECConfig:
    small_hidden_dim: int = 256
    transformer_m_hidden_dim: int = 256
    transformer_m_layers: int = 2
    transformer_m_heads: int = 8
    transformer_m_ff_mult: int = 4
    heatmap_grid: int = 8
    heatmap_channels: int = 3
    heatmap_feature_dim: int = 64
    network_feature_dim: int = 32
    network_state_dim: int = 5
    arrival_hidden_dim: int = 64
    arrival_match_temperature: float = 0.10
    arrival_stop_threshold: float = 0.50
    num_actions: int = NUM_ACTIONS
    num_move_actions: int = NUM_MOVE_ACTIONS


@dataclass(frozen=True)
class TextConfig:
    # Formal training uses frozen pretrained BERT.  ``hashing`` is only for
    # offline unit / smoke tests, never for a formal training run.
    backend: str = "bert"  # bert | hashing (auto is retained for diagnostics)
    model_name: str = "bert-base-uncased"
    local_files_only: bool = True
    max_length: int = 48
    hashing_dim: int = 384

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "TextConfig":
        mapping = mapping or {}
        return cls(
            backend=str(_get(mapping, "backend", "bert")),
            model_name=str(_get(mapping, "model_name", "bert-base-uncased")),
            local_files_only=bool(_get(mapping, "local_files_only", True)),
            max_length=int(_get(mapping, "max_length", 48)),
            hashing_dim=int(_get(mapping, "hashing_dim", 384)),
        )


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    lr: float = 1e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 1e-4
    arrival_loss_weight: float = 1.0
    grad_clip: float = 1.0
    amp: bool = False
    train_vision_encoder: bool = True
    freeze_text_encoder: bool = True
    seed: int = 0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "TrainingConfig":
        mapping = mapping or {}
        return cls(
            batch_size=int(_get(mapping, "batch_size", 4)),
            lr=float(_get(mapping, "lr", 1e-4)),
            encoder_lr=float(_get(mapping, "encoder_lr", 1e-5)),
            weight_decay=float(_get(mapping, "weight_decay", 1e-4)),
            arrival_loss_weight=float(_get(mapping, "arrival_loss_weight", 1.0)),
            grad_clip=float(_get(mapping, "grad_clip", 1.0)),
            amp=bool(_get(mapping, "amp", False)),
            train_vision_encoder=bool(_get(mapping, "train_vision_encoder", True)),
            freeze_text_encoder=bool(_get(mapping, "freeze_text_encoder", True)),
            seed=int(_get(mapping, "seed", 0)),
        )


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    mec: MECConfig = field(default_factory=MECConfig)
    text: TextConfig = field(default_factory=TextConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cpu"
    seed: int = 0

    def redacted(self) -> dict:
        return {
            "data": self.data.redacted(),
            "observation": asdict(self.observation),
            "teacher": asdict(self.teacher),
            "encoder": asdict(self.encoder),
            "mec": asdict(self.mec),
            "text": asdict(self.text),
            "training": asdict(self.training),
            "device": self.device,
            "seed": self.seed,
        }


def load_data_config(
    path: Path | str | None = None,
    *,
    data_root: str | None = None,
    ply_root: str | None = None,
) -> DataConfig:
    if data_root or ply_root:
        mapping = {
            "data_root": data_root or os.environ.get("CITYNAV_DATA_ROOT"),
            "ply_root": ply_root or os.environ.get("CITYNAV_PLY_ROOT"),
        }
        return DataConfig.from_mapping(mapping)
    if path is not None:
        return DataConfig.from_mapping(load_yaml(path))
    local = PROJECT_ROOT / "configs" / "data_config.local.yaml"
    if local.exists():
        return DataConfig.from_mapping(load_yaml(local))
    return DataConfig.from_env()


def load_app_config(
    *,
    data_config_path: Path | str | None = None,
    train_config_path: Path | str | None = None,
    data_root: str | None = None,
    ply_root: str | None = None,
    device: str | None = None,
) -> AppConfig:
    data = load_data_config(
        data_config_path, data_root=data_root, ply_root=ply_root
    )
    train_mapping: dict = {}
    if train_config_path is not None:
        train_mapping = load_yaml(train_config_path)
    observation = ObservationConfig.from_mapping(train_mapping.get("observation"))
    teacher = TeacherConfig.from_mapping(train_mapping.get("teacher"))
    text = TextConfig.from_mapping(train_mapping.get("text"))
    training = TrainingConfig.from_mapping(train_mapping.get("training"))
    encoder = EncoderConfig.from_mapping(train_mapping.get("encoder"))
    seed = int(train_mapping.get("seed", training.seed))
    return AppConfig(
        data=data,
        observation=observation,
        teacher=teacher,
        encoder=encoder,
        mec=MECConfig(),
        text=text,
        training=training,
        device=str(device or train_mapping.get("device", "cpu")),
        seed=seed,
    )
