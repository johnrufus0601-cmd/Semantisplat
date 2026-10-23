"""
SemantiSplat Configuration Loader.

Loads config/default.yaml, merges with CLI/env overrides, and exposes
typed dataclasses for each pipeline stage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ============================================================================
# Typed config dataclasses
# ============================================================================

def _get_default_colmap_bin() -> str:
    env_bin = os.getenv("COLMAP_BIN")
    if env_bin:
        return env_bin
    
    # Check local bundled colmap_bin folder
    local_bin = Path(__file__).parent.parent / "colmap_bin" / "COLMAP.bat"
    if local_bin.exists():
        return str(local_bin.resolve())
        
    return "colmap"


@dataclass
class ColmapConfig:
    """COLMAP structure-from-motion settings."""
    binary_path: str = field(default_factory=_get_default_colmap_bin)
    camera_model: str = "OPENCV"
    use_gpu: bool = True
    matcher_type: str = "exhaustive"
    registration_threshold: float = 0.8


@dataclass
class IngestionConfig:
    """FR-1: Video ingestion & pose estimation settings."""
    frame_extraction_fps: float = 2.0
    max_frames: int = 300
    frame_format: str = "png"
    frame_quality: int = 95
    downsample_resolution: int = 0      # 0 means no downsampling
    colmap: ColmapConfig = field(default_factory=ColmapConfig)


@dataclass
class SAMConfig:
    """Segment Anything Model settings."""
    model_type: str = "vit_b"
    checkpoint: str = "sam_vit_b_01ec64.pth"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    min_mask_area: int = 100
    max_masks_per_frame: int = 100


@dataclass
class CLIPConfig:
    """CLIP vision-language model settings."""
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    embedding_dim: int = 512
    batch_size: int = 64


@dataclass
class SemanticConfig:
    """FR-2: Semantic feature extraction settings."""
    mixed_precision: str = "fp32"       # "fp32", "fp16", or "bf16"
    sam: SAMConfig = field(default_factory=SAMConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)


@dataclass
class LearningRateConfig:
    """Per-parameter learning rates for Gaussian training."""
    position: float = 0.00016
    feature: float = 0.0025
    opacity: float = 0.05
    scaling: float = 0.005
    rotation: float = 0.001
    semantic: float = 0.001
    compressor: float = 0.001


@dataclass
class LossWeightsConfig:
    """Loss function weights."""
    photometric: float = 1.0
    ssim: float = 0.2
    semantic: float = 0.1


@dataclass
class DensificationConfig:
    """Gaussian densification/pruning schedule."""
    start_iter: int = 500
    end_iter: int = 15000
    interval: int = 100
    grad_threshold: float = 0.0002
    min_opacity: float = 0.005
    max_gaussians: int = 0              # 0 means no hard cap


@dataclass
class ReconstructionConfig:
    """FR-3: 3D Gaussian Splatting reconstruction settings."""
    semantic_dim: int = 16
    compression_type: str = "linear"
    mixed_precision: str = "fp32"       # "fp32", "fp16", or "bf16"
    max_iterations: int = 30000
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    densification: DensificationConfig = field(default_factory=DensificationConfig)
    checkpoint_interval: int = 5000
    output_format: str = "ply"


@dataclass
class SpatialClusteringConfig:
    """DBSCAN clustering for grouping matched Gaussians into regions."""
    enabled: bool = True
    eps: float = 0.05
    min_samples: int = 10


@dataclass
class RetrievalConfig:
    """FR-4: Natural-language query & retrieval settings."""
    top_k: int = 5
    similarity_threshold: float = 0.25
    spatial_clustering: SpatialClusteringConfig = field(default_factory=SpatialClusteringConfig)


@dataclass
class APIConfig:
    """Web API settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_upload_size_mb: int = 2048
    static_dir: str = "frontend"


@dataclass
class SemantiSplatConfig:
    """Top-level configuration container for the entire pipeline."""
    project_name: str = "SemantiSplat"
    data_dir: str = "data"
    device: str = "cuda"
    seed: int = 42
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    api: APIConfig = field(default_factory=APIConfig)


# ============================================================================
# Config loading
# ============================================================================

def _merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_dataclass(cls, data: dict[str, Any]):
    """Recursively convert a nested dict to typed dataclasses."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            continue  # skip unknown keys

        field_type = field_types[key]

        # Resolve string type annotations
        if isinstance(field_type, str):
            # Handle forward references / string annotations
            import importlib
            module = importlib.import_module(cls.__module__)
            field_type = getattr(module, field_type, field_type)

        # If the field type is itself a dataclass, recurse
        if isinstance(field_type, type) and hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(
    config_path: str | Path = "config/default.yaml",
    overrides: dict[str, Any] | None = None,
) -> SemantiSplatConfig:
    """
    Load configuration from YAML file, merge overrides, return typed config.

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional dict of overrides to merge on top.

    Returns:
        Fully typed SemantiSplatConfig instance.
    """
    config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # Merge overrides
    if overrides:
        raw = _merge_dict(raw, overrides)

    # Apply environment variable overrides (SEMANTISPLAT_DEVICE, etc.)
    env_device = os.environ.get("SEMANTISPLAT_DEVICE")
    if env_device:
        raw["device"] = env_device

    env_data_dir = os.environ.get("SEMANTISPLAT_DATA_DIR")
    if env_data_dir:
        raw["data_dir"] = env_data_dir

    return _dict_to_dataclass(SemantiSplatConfig, raw)


def get_data_path(config: SemantiSplatConfig, *subpaths: str) -> Path:
    """Get a path under the data directory, creating it if needed."""
    path = Path(config.data_dir).joinpath(*subpaths)
    path.mkdir(parents=True, exist_ok=True)
    return path
