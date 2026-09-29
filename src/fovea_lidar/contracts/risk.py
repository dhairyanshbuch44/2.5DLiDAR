"""Collision risk contracts for Fovea-LiDAR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union


class RiskLevel(str, Enum):
    """Categorical collision risk severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskRegion:
    """Represents a spatial region assessed with dynamic collision risk.

    Attributes
    ----------
    region_bounds : Tuple[float, float, float, float] or Tuple[float, float, float, float, float, float]
        Spatial bounds (min_x, min_y, max_x, max_y) in meters (m) for 2D, or
        (min_x, min_y, min_z, max_x, max_y, max_z) in meters (m) for 3D.
    risk_score : float
        Continuous hazard score in range [0.0, 1.0] (dimensionless).
    risk_level : RiskLevel
        Categorical severity level (LOW, MEDIUM, HIGH, CRITICAL).
    associated_object_id : Optional[Union[int, str]]
        ID of the dynamic obstacle or track inducing this risk zone (or None).
    timestamp : float
        Timestamp of risk computation in seconds (s).
    """

    region_bounds: Union[
        Tuple[float, float, float, float],
        Tuple[float, float, float, float, float, float],
    ]
    risk_score: float
    risk_level: RiskLevel
    associated_object_id: Optional[Union[int, str]] = None
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if len(self.region_bounds) not in (4, 6):
            raise ValueError(
                f"region_bounds must have length 4 (2D) or 6 (3D), got {len(self.region_bounds)}"
            )
        self.region_bounds = tuple(float(x) for x in self.region_bounds)  # type: ignore

        self.risk_score = float(self.risk_score)
        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError(f"risk_score must be in range [0.0, 1.0], got {self.risk_score}")

        if isinstance(self.risk_level, str):
            self.risk_level = RiskLevel(self.risk_level.upper())

        self.timestamp = float(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize RiskRegion to JSON-serializable dictionary."""
        return {
            "region_bounds": list(self.region_bounds),
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "associated_object_id": self.associated_object_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RiskRegion:
        """Construct RiskRegion from dictionary representation."""
        bounds = data["region_bounds"]
        level = RiskLevel(data["risk_level"]) if isinstance(data["risk_level"], str) else data["risk_level"]
        return cls(
            region_bounds=tuple(float(b) for b in bounds),  # type: ignore
            risk_score=float(data["risk_score"]),
            risk_level=level,
            associated_object_id=data.get("associated_object_id"),
            timestamp=float(data.get("timestamp", 0.0)),
        )
