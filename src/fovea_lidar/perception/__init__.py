"""Pluggable semantic perception module for Fovea-LiDAR."""

from fovea_lidar.perception.base import SegmentationModel
from fovea_lidar.perception.mock_model import MockModel
from fovea_lidar.perception.registry import build_segmentation_model

__all__ = [
    "SegmentationModel",
    "MockModel",
    "build_segmentation_model",
]
