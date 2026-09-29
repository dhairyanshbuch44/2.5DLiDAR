"""Detected object contracts for Fovea-LiDAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union


@dataclass
class DetectedObject:
    """Represents an object detected in a 3D LiDAR point cloud.

    NOTE ON BOUNDING REGION:
    The bounding_region field represents an Axis-Aligned Bounding Box (AABB)
    defined as (min_x, min_y, min_z, max_x, max_y, max_z) in meters (m).
    Oriented 3D bounding boxes (OBB with center, dimensions, and yaw/pitch/roll
    quaternions) are planned as a future architectural upgrade in downstream prompts.

    Attributes
    ----------
    object_id : Union[int, str]
        Unique identifier assigned to the detection.
    semantic_class : str
        Predicted semantic class label (e.g., 'VEHICLE', 'PEDESTRIAN').
    position : Tuple[float, float, float]
        Center/centroid position (x, y, z) in meters (m).
    bounding_region : Tuple[float, float, float, float, float, float]
        Axis-aligned bounding box (min_x, min_y, min_z, max_x, max_y, max_z) in meters (m).
    confidence : float
        Detection confidence score in range [0.0, 1.0] (dimensionless).
    velocity : Optional[Tuple[float, float, float]]
        Estimated instantaneous velocity (vx, vy, vz) in meters per second (m/s).
        Defaults to None prior to multi-object tracking association.
    dynamic : bool
        Whether the object is classified as dynamic/moving (dimensionless boolean).
    timestamp : float
        Detection timestamp in seconds (s).
    """

    object_id: Union[int, str]
    semantic_class: str
    position: Tuple[float, float, float]
    bounding_region: Tuple[float, float, float, float, float, float]
    confidence: float
    velocity: Optional[Tuple[float, float, float]] = None
    dynamic: bool = False
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError(
                f"Position must contain exactly 3 components (x, y, z), got {len(self.position)}"
            )
        self.position = (
            float(self.position[0]),
            float(self.position[1]),
            float(self.position[2]),
        )

        if len(self.bounding_region) != 6:
            raise ValueError(
                f"Bounding region must be (min_x, min_y, min_z, max_x, max_y, max_z), got length {len(self.bounding_region)}"
            )
        self.bounding_region = (
            float(self.bounding_region[0]),
            float(self.bounding_region[1]),
            float(self.bounding_region[2]),
            float(self.bounding_region[3]),
            float(self.bounding_region[4]),
            float(self.bounding_region[5]),
        )

        self.confidence = float(self.confidence)
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence score must be in range [0.0, 1.0], got {self.confidence}"
            )

        if self.velocity is not None:
            if len(self.velocity) != 3:
                raise ValueError(
                    f"Velocity must contain exactly 3 components (vx, vy, vz), got {len(self.velocity)}"
                )
            self.velocity = (
                float(self.velocity[0]),
                float(self.velocity[1]),
                float(self.velocity[2]),
            )

        self.dynamic = bool(self.dynamic)
        self.timestamp = float(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DetectedObject to JSON-serializable dictionary."""
        return {
            "object_id": self.object_id,
            "semantic_class": self.semantic_class,
            "position": list(self.position),
            "bounding_region": list(self.bounding_region),
            "confidence": self.confidence,
            "velocity": list(self.velocity) if self.velocity is not None else None,
            "dynamic": self.dynamic,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DetectedObject:
        """Construct DetectedObject from dictionary representation."""
        pos = data["position"]
        bbox = data["bounding_region"]
        vel = data.get("velocity")
        return cls(
            object_id=data["object_id"],
            semantic_class=str(data["semantic_class"]),
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            bounding_region=(
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                float(bbox[4]),
                float(bbox[5]),
            ),
            confidence=float(data["confidence"]),
            velocity=(float(vel[0]), float(vel[1]), float(vel[2])) if vel is not None else None,
            dynamic=bool(data.get("dynamic", False)),
            timestamp=float(data.get("timestamp", 0.0)),
        )
