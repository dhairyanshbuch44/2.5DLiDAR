"""Tracked object contracts for Fovea-LiDAR."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from fovea_lidar.contracts.objects import DetectedObject


@dataclass
class TrackedObject(DetectedObject):
    """Extends DetectedObject with persistent tracking state, kinematic velocity, and history.

    Attributes
    ----------
    track_id : Union[int, str]
        Persistent identifier assigned by the multi-object tracking filter across frames.
    velocity : Tuple[float, float, float]
        Kinematic 3D velocity vector (vx, vy, vz) in meters per second (m/s).
        REQUIRED for TrackedObject (cannot be None).
    direction : Tuple[float, float, float]
        Normalized 3D heading direction vector (dx, dy, dz) (dimensionless unit vector).
    position_history : List[Tuple[float, float, float]]
        Bounded history of sequential (x, y, z) positions in meters (m).
    last_seen : float
        Timestamp when the tracked object was most recently matched in seconds (s).
    track_confidence : float
        Track confirmation score in range [0.0, 1.0] (dimensionless). Decays over time
        when coasting without sensor detections.
    max_history_len : int
        Maximum number of historic positions retained in position_history.
    """

    track_id: Union[int, str] = ""
    # In TrackedObject, velocity is required and must not be None
    velocity: Optional[Tuple[float, float, float]] = None
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    position_history: List[Tuple[float, float, float]] = field(default_factory=list)
    last_seen: float = 0.0
    track_confidence: float = 1.0
    max_history_len: int = 50
    is_coasting: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        # Velocity is mandatory on TrackedObject
        if self.velocity is None:
            raise ValueError("velocity is required for TrackedObject and cannot be None")

        if len(self.velocity) != 3:
            raise ValueError(
                f"Velocity must contain exactly 3 components (vx, vy, vz), got {len(self.velocity)}"
            )
        self.velocity = (
            float(self.velocity[0]),
            float(self.velocity[1]),
            float(self.velocity[2]),
        )

        if len(self.direction) != 3:
            raise ValueError(
                f"Direction must be a 3D unit vector (dx, dy, dz), got length {len(self.direction)}"
            )
        self.direction = (
            float(self.direction[0]),
            float(self.direction[1]),
            float(self.direction[2]),
        )

        # Normalize position history items
        normalized_history: List[Tuple[float, float, float]] = []
        for p in self.position_history:
            if len(p) != 3:
                raise ValueError(f"Each historical position must have 3 coordinates (x, y, z), got {p}")
            normalized_history.append((float(p[0]), float(p[1]), float(p[2])))
        self.position_history = normalized_history[-self.max_history_len :]

        self.last_seen = float(self.last_seen)
        self.track_confidence = float(self.track_confidence)
        if not (0.0 <= self.track_confidence <= 1.0):
            raise ValueError(
                f"Track confidence must be in range [0.0, 1.0], got {self.track_confidence}"
            )

    def add_position(self, pos: Tuple[float, float, float]) -> None:
        """Append a new (x, y, z) position in meters and enforce history bounds."""
        if len(pos) != 3:
            raise ValueError(f"Position must be a 3-tuple (x, y, z), got {pos}")
        self.position_history.append((float(pos[0]), float(pos[1]), float(pos[2])))
        if len(self.position_history) > self.max_history_len:
            self.position_history.pop(0)

    def decay_confidence(self, current_time: float, decay_rate: float = 0.5) -> float:
        """Exponentially decay track confidence based on elapsed time without observation.

        Full multi-object Kalman filter tracking and coasting logic is implemented
        in Prompt 10 (Multi-Object Tracking).

        Parameters
        ----------
        current_time : float
            Current system timestamp in seconds (s).
        decay_rate : float
            Exponential decay rate per second (s^-1), default 0.5.

        Returns
        -------
        float
            Updated track confidence score in range [0.0, 1.0].
        """
        dt = max(0.0, float(current_time) - self.last_seen)
        self.track_confidence = max(0.0, min(1.0, self.track_confidence * math.exp(-decay_rate * dt)))
        return self.track_confidence

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TrackedObject to JSON-serializable dictionary."""
        d = super().to_dict()
        d.update(
            {
                "track_id": self.track_id,
                "velocity": list(self.velocity) if self.velocity is not None else None,
                "direction": list(self.direction),
                "position_history": [list(p) for p in self.position_history],
                "last_seen": self.last_seen,
                "track_confidence": self.track_confidence,
                "max_history_len": self.max_history_len,
                "is_coasting": self.is_coasting,
            }
        )
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrackedObject:
        """Construct TrackedObject from dictionary representation."""
        pos = data["position"]
        bbox = data["bounding_region"]
        vel = data.get("velocity")
        if vel is None:
            raise ValueError("Field 'velocity' is required for TrackedObject")

        direction = data.get("direction", (1.0, 0.0, 0.0))
        history_raw = data.get("position_history", [])
        history = [(float(p[0]), float(p[1]), float(p[2])) for p in history_raw]

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
            velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
            dynamic=bool(data.get("dynamic", True)),
            timestamp=float(data.get("timestamp", 0.0)),
            track_id=data.get("track_id", data["object_id"]),
            direction=(float(direction[0]), float(direction[1]), float(direction[2])),
            position_history=history,
            last_seen=float(data.get("last_seen", data.get("timestamp", 0.0))),
            track_confidence=float(data.get("track_confidence", 1.0)),
            max_history_len=int(data.get("max_history_len", 50)),
            is_coasting=bool(data.get("is_coasting", False)),
        )
