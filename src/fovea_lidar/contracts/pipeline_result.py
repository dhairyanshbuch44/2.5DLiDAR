"""Pipeline execution result contract for Fovea-LiDAR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fovea_lidar.contracts.mapping import AdaptiveMap, TraversabilityMap
from fovea_lidar.contracts.objects import DetectedObject
from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.contracts.risk import RiskRegion
from fovea_lidar.contracts.semantic import SemanticPoint
from fovea_lidar.contracts.tracking import TrackedObject


@dataclass
class PipelineResult:
    """Encapsulates the end-to-end output of a single processed LiDAR scan frame.

    Attributes
    ----------
    frame_index : int
        Zero-based frame sequence index.
    timestamp : float
        Timestamp of the processed frame in seconds (s).
    raw_frame : PointCloudFrame
        Original unadulterated point cloud input.
    cleaned_frame : PointCloudFrame
        Filtered, downsampled, and ground-segmented point cloud.
    semantic_points : List[SemanticPoint]
        Per-point semantic classification predictions.
    detected_objects : List[DetectedObject]
        Spatial clusters classified into candidate objects with bounding boxes.
    tracked_objects : List[TrackedObject]
        Objects with persistent track IDs, Kalman velocity, and history.
    risk_regions : List[RiskRegion]
        Evaluated spatial collision risk hazard regions.
    active_fovea_regions : List[Dict[str, Any]]
        List of active high-resolution foveation patches from FoveaManager.
    adaptive_map : Any
        Reference to the updated 2.5D stateful adaptive map.
    terrain_data : Optional[Dict[Any, Any]]
        Optional terrain traversability classification and hazard map.
    traversability_map : Optional[TraversabilityMap]
        Consolidated dynamic traversability map accounting for terrain and moving obstacles.
    metrics : Dict[str, Any]
        Per-frame stage timings (ms), counts, and performance measurements.
    """

    frame_index: int
    timestamp: float
    raw_frame: PointCloudFrame
    cleaned_frame: PointCloudFrame
    semantic_points: List[SemanticPoint]
    detected_objects: List[DetectedObject]
    tracked_objects: List[TrackedObject]
    risk_regions: List[RiskRegion]
    active_fovea_regions: List[Dict[str, Any]]
    adaptive_map: Any
    terrain_data: Optional[Dict[Any, Any]] = None
    traversability_map: Optional[TraversabilityMap] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize high-level summary of pipeline result to dictionary."""
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "raw_point_count": int(self.raw_frame.points.shape[0]),
            "cleaned_point_count": int(self.cleaned_frame.points.shape[0]),
            "semantic_point_count": len(self.semantic_points),
            "detected_object_count": len(self.detected_objects),
            "tracked_object_count": len(self.tracked_objects),
            "risk_region_count": len(self.risk_regions),
            "active_fovea_count": len(self.active_fovea_regions),
            "total_map_cells": self.adaptive_map.total_cells,
            "metrics": dict(self.metrics),
        }
