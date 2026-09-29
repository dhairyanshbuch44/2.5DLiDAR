"""Abstract base class for semantic segmentation perception models in Fovea-LiDAR."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.contracts.semantic import SemanticPoint
from fovea_lidar.utils.config import Config, load_config


class SegmentationModel(ABC):
    """Abstract model interface for 3D point cloud semantic segmentation."""

    @abstractmethod
    def predict(self, frame: PointCloudFrame) -> List[SemanticPoint]:
        """Perform semantic point-wise prediction on an input point cloud frame.

        Parameters
        ----------
        frame : PointCloudFrame
            LiDAR scan frame with shape (N, 3) or (N, 4).

        Returns
        -------
        List[SemanticPoint]
            List of classified 3D points matching the input cloud.
        """
        raise NotImplementedError("Subclasses must implement predict()")

    @classmethod
    @abstractmethod
    def load(cls, config: Optional[Union[Config, Dict[str, Any], Path, str]] = None) -> SegmentationModel:
        """Instantiate the segmentation model from configuration.

        Parameters
        ----------
        config : Config, dict, Path, or str, optional
            Configuration containing perception parameters.

        Returns
        -------
        SegmentationModel
            Configured model instance.
        """
        raise NotImplementedError("Subclasses must implement load()")
