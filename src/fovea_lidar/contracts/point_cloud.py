"""Point cloud frame contract for Fovea-LiDAR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union
import numpy as np


class FrameSource(str, Enum):
    """Origin source of the point cloud frame."""

    REPLAY = "REPLAY"
    MOCK = "MOCK"
    LIVE = "LIVE"


@dataclass
class PointCloudFrame:
    """Represents a single LiDAR scan frame.

    Attributes
    ----------
    points : np.ndarray
        Array of shape (N, 3) or (N, 4).
        Columns 0, 1, 2: X, Y, Z coordinates in meters (m).
        Column 3 (optional): Intensity value (normalized [0.0, 1.0] or sensor raw).
    timestamp : float
        Timestamp of the scan acquisition in seconds (s).
    frame_id : str
        Coordinate frame identifier (e.g., 'lidar_top', 'base_link').
    source : FrameSource
        Source origin of the frame (REPLAY, MOCK, LIVE).
    metadata : dict
        Arbitrary auxiliary metadata (sensor calibration, temperature, sequence index).
    """

    points: np.ndarray
    timestamp: float
    frame_id: str
    source: FrameSource = FrameSource.MOCK
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.points, np.ndarray):
            self.points = np.asarray(self.points, dtype=np.float32)

        if self.points.ndim != 2 or self.points.shape[1] not in (3, 4):
            raise ValueError(
                f"Points array must be of shape (N, 3) or (N, 4), got shape {self.points.shape}"
            )

        if isinstance(self.source, str):
            self.source = FrameSource(self.source)

        self.timestamp = float(self.timestamp)
        self.frame_id = str(self.frame_id)
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize frame to JSON-serializable dictionary."""
        return {
            "points": self.points.tolist(),
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "source": self.source.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PointCloudFrame:
        """Construct PointCloudFrame from dictionary representation."""
        points = np.asarray(data["points"], dtype=np.float32)
        source = FrameSource(data["source"]) if isinstance(data["source"], str) else data["source"]
        return cls(
            points=points,
            timestamp=float(data["timestamp"]),
            frame_id=str(data["frame_id"]),
            source=source,
            metadata=dict(data.get("metadata", {})),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointCloudFrame):
            return False
        return (
            self.timestamp == other.timestamp
            and self.frame_id == other.frame_id
            and self.source == other.source
            and self.metadata == other.metadata
            and np.allclose(self.points, other.points, rtol=1e-5, atol=1e-5, equal_nan=True)
        )
