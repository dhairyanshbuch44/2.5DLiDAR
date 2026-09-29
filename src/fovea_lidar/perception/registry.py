"""Model registry and factory for semantic segmentation backends."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fovea_lidar.perception.base import SegmentationModel
from fovea_lidar.perception.mock_model import MockModel
from fovea_lidar.utils.config import Config, load_config
from fovea_lidar.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_segmentation_model(
    config: Optional[Union[Config, Dict[str, Any], Path, str]] = None,
) -> SegmentationModel:
    """Build and return an instantiated SegmentationModel based on system configuration.

    Parameters
    ----------
    config : Config, dict, Path, or str, optional
        System configuration containing 'perception' settings.

    Returns
    -------
    SegmentationModel
        Constructed segmentation model backend.

    Raises
    ------
    NotImplementedError
        If an unimplemented backend ('pretrained', 'custom') is requested.
    ValueError
        If an unknown backend identifier is specified.
    """
    if config is None:
        full_cfg = load_config()
    elif isinstance(config, (str, Path)):
        full_cfg = load_config(config)
    elif isinstance(config, (Config, dict)):
        full_cfg = config
    else:
        raise TypeError(f"Unsupported config type: {type(config).__name__}")

    percep_cfg = full_cfg.get("perception", {})
    if isinstance(percep_cfg, Config):
        backend = str(percep_cfg.get("backend", "mock")).strip().lower()
    elif isinstance(percep_cfg, dict):
        backend = str(percep_cfg.get("backend", "mock")).strip().lower()
    else:
        backend = "mock"

    if backend == "mock":
        return MockModel.load(config=full_cfg)

    elif backend in ("pretrained", "pointnet", "pointnet++"):
        raise NotImplementedError(
            f"Segmentation backend '{backend}' is not yet implemented — "
            "deep learning inference models will be integrated in Advanced Prompt A1."
        )

    elif backend == "custom":
        raise NotImplementedError(
            "Custom segmentation backend is not yet configured — "
            "implement and register your model backend in perception/registry.py."
        )

    else:
        raise ValueError(
            f"Unknown perception backend '{backend}'. Supported options: 'mock', 'pretrained', 'custom'."
        )
