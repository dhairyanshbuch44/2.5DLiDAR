"""Semantic point contract for Fovea-LiDAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from fovea_lidar.contracts.semantic_classes import validate_class


@dataclass
class SemanticPoint:
    """Represents a single 3D point labeled with semantic class and confidence.

    Attributes
    ----------
    coordinates : Tuple[float, float, float]
        X, Y, Z coordinates in meters (m).
    semantic_class : str
        Semantic class label, validated against the semantic class registry
        (e.g., 'ROAD', 'VEHICLE', 'PEDESTRIAN').
    confidence : float
        Classification confidence score in range [0.0, 1.0] (dimensionless).
    timestamp : float
        Timestamp of the point observation in seconds (s).
    """

    coordinates: Tuple[float, float, float]
    semantic_class: str
    confidence: float
    timestamp: float

    def __post_init__(self) -> None:
        # Validate coordinates
        if len(self.coordinates) != 3:
            raise ValueError(
                f"Coordinates must contain exactly 3 components (x, y, z), got {len(self.coordinates)}"
            )
        self.coordinates = (
            float(self.coordinates[0]),
            float(self.coordinates[1]),
            float(self.coordinates[2]),
        )

        # Validate semantic class against dynamic registry
        self.semantic_class = validate_class(self.semantic_class)

        # Validate confidence
        self.confidence = float(self.confidence)
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence score must be in range [0.0, 1.0], got {self.confidence}"
            )

        self.timestamp = float(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SemanticPoint to JSON-serializable dictionary."""
        return {
            "coordinates": list(self.coordinates),
            "semantic_class": self.semantic_class,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SemanticPoint:
        """Construct SemanticPoint from dictionary representation."""
        coords = data["coordinates"]
        return cls(
            coordinates=(float(coords[0]), float(coords[1]), float(coords[2])),
            semantic_class=str(data["semantic_class"]),
            confidence=float(data["confidence"]),
            timestamp=float(data["timestamp"]),
        )
