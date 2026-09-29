"""Preprocessing metrics data contract for Fovea-LiDAR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PreprocessingMetrics:
    """Quantitative performance and point reduction metrics from preprocessing.

    Attributes
    ----------
    input_point_count : int
        Number of raw input points.
    output_point_count : int
        Number of points remaining after filtering and downsampling.
    filtering_ratio : float
        Fraction of points removed: (input - output) / input in range [0.0, 1.0] (dimensionless).
    processing_time_ms : float
        Real measured wall-clock processing time in milliseconds (ms).
    stage_timings_ms : Dict[str, float]
        Real measured execution duration per individual preprocessing stage in milliseconds (ms).
    """

    input_point_count: int
    output_point_count: int
    filtering_ratio: float
    processing_time_ms: float
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to JSON-serializable dictionary."""
        return {
            "input_point_count": self.input_point_count,
            "output_point_count": self.output_point_count,
            "filtering_ratio": self.filtering_ratio,
            "processing_time_ms": self.processing_time_ms,
            "stage_timings_ms": self.stage_timings_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreprocessingMetrics:
        """Construct PreprocessingMetrics from dictionary."""
        return cls(
            input_point_count=int(data["input_point_count"]),
            output_point_count=int(data["output_point_count"]),
            filtering_ratio=float(data["filtering_ratio"]),
            processing_time_ms=float(data["processing_time_ms"]),
            stage_timings_ms=dict(data.get("stage_timings_ms", {})),
        )
