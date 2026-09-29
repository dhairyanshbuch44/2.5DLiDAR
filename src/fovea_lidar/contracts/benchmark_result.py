"""Benchmark execution result and experiment reproducibility contract for Fovea-LiDAR."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkResult:
    """Standardized record of a benchmark evaluation run across an ablation mode and scenario.

    Adheres to project brief sections 33-35: includes reproducibility metadata,
    runtime hardware environment, aggregate performance metrics, accuracy metrics,
    distance-wise cell density distribution, and per-frame telemetry.

    Attributes
    ----------
    mode_name : str
        Ablation configuration mode (e.g., 'mode_1_uniform', 'mode_5_full').
    scenario_name : str
        Evaluated scenario dataset name (e.g., 'multi_object_dense').
    num_frames : int
        Total number of processed LiDAR frames.
    hardware_info : Dict[str, Any]
        Runtime system environment (platform, CPU cores, RAM).
    config_snapshot : Dict[str, Any]
        Deep copy of the exact pipeline configuration used for the run.
    resolution_profile : str
        Human-readable summary of the resolution zone structure.
    mean_fps : float
        Mean effective frames per second.
    mean_latency_ms : float
        Mean total frame latency in milliseconds.
    p95_latency_ms : float
        95th percentile total frame latency in milliseconds.
    mean_active_cells : float
        Mean number of populated map cells per frame.
    peak_memory_mb : float
        Peak memory allocation/RSS measured during execution in MB.
    point_throughput_pps : float
        Point processing throughput in points per second (PPS).
    detection_precision : float
        Object detection precision evaluated against synthetic ground truth in [0.0, 1.0].
    detection_recall : float
        Object detection recall evaluated against synthetic ground truth in [0.0, 1.0].
    tracking_id_switches : int
        Count of identity switches across tracked objects during sequence.
    distance_density : Dict[str, int]
        Final cell count distribution across distance bands ('0-10m', '10-30m', '30-60m', '60-100m').
    frame_metrics : List[Dict[str, Any]]
        Per-frame metrics dictionaries captured from MetricsCollector.
    timestamp : float
        Timestamp of benchmark run completion in epoch seconds.
    """

    mode_name: str
    scenario_name: str
    num_frames: int
    hardware_info: Dict[str, Any]
    config_snapshot: Dict[str, Any]
    resolution_profile: str
    mean_fps: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_active_cells: float
    peak_memory_mb: float
    point_throughput_pps: float
    detection_precision: float
    detection_recall: float
    tracking_id_switches: int
    distance_density: Dict[str, int]
    frame_metrics: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforce that all metric fields contain real, valid, non-null measurements.

        Raises
        ------
        ValueError
            If any metric field is missing, None, or NaN.
        """
        required_numeric = [
            ("mean_fps", self.mean_fps),
            ("mean_latency_ms", self.mean_latency_ms),
            ("p95_latency_ms", self.p95_latency_ms),
            ("mean_active_cells", self.mean_active_cells),
            ("peak_memory_mb", self.peak_memory_mb),
            ("point_throughput_pps", self.point_throughput_pps),
            ("detection_precision", self.detection_precision),
            ("detection_recall", self.detection_recall),
        ]

        for name, val in required_numeric:
            if val is None or math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"BenchmarkResult metric '{name}' must be a valid real number, got {val}"
                )

        if not isinstance(self.tracking_id_switches, int) or self.tracking_id_switches < 0:
            raise ValueError(
                f"tracking_id_switches must be non-negative integer, got {self.tracking_id_switches}"
            )

        if not isinstance(self.distance_density, dict) or not self.distance_density:
            raise ValueError("distance_density must be a non-empty dictionary")

    def to_dict(self) -> Dict[str, Any]:
        """Convert BenchmarkResult to JSON-serializable dictionary."""
        return {
            "mode_name": self.mode_name,
            "scenario_name": self.scenario_name,
            "num_frames": self.num_frames,
            "hardware_info": dict(self.hardware_info),
            "config_snapshot": dict(self.config_snapshot),
            "resolution_profile": self.resolution_profile,
            "mean_fps": round(float(self.mean_fps), 2),
            "mean_latency_ms": round(float(self.mean_latency_ms), 2),
            "p95_latency_ms": round(float(self.p95_latency_ms), 2),
            "mean_active_cells": round(float(self.mean_active_cells), 1),
            "peak_memory_mb": round(float(self.peak_memory_mb), 2),
            "point_throughput_pps": round(float(self.point_throughput_pps), 1),
            "detection_precision": round(float(self.detection_precision), 3),
            "detection_recall": round(float(self.detection_recall), 3),
            "tracking_id_switches": int(self.tracking_id_switches),
            "distance_density": {str(k): int(v) for k, v in self.distance_density.items()},
            "frame_metrics": list(self.frame_metrics),
            "timestamp": float(self.timestamp),
        }

    def to_flattened_dict(self) -> Dict[str, Any]:
        """Convert aggregate metrics to a flat dictionary for CSV export."""
        flat = {
            "mode_name": self.mode_name,
            "scenario_name": self.scenario_name,
            "num_frames": self.num_frames,
            "resolution_profile": self.resolution_profile,
            "mean_fps": round(float(self.mean_fps), 2),
            "mean_latency_ms": round(float(self.mean_latency_ms), 2),
            "p95_latency_ms": round(float(self.p95_latency_ms), 2),
            "mean_active_cells": round(float(self.mean_active_cells), 1),
            "peak_memory_mb": round(float(self.peak_memory_mb), 2),
            "point_throughput_pps": round(float(self.point_throughput_pps), 1),
            "detection_precision": round(float(self.detection_precision), 3),
            "detection_recall": round(float(self.detection_recall), 3),
            "tracking_id_switches": int(self.tracking_id_switches),
            "cells_0_10m": int(self.distance_density.get("0-10m", 0)),
            "cells_10_30m": int(self.distance_density.get("10-30m", 0)),
            "cells_30_60m": int(self.distance_density.get("30-60m", 0)),
            "cells_60_100m": int(self.distance_density.get("60-100m", 0)),
            "cpu_cores": self.hardware_info.get("cpu_count", 0),
            "platform": self.hardware_info.get("platform", ""),
        }
        return flat

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkResult:
        """Reconstruct BenchmarkResult from dictionary."""
        return cls(
            mode_name=str(data["mode_name"]),
            scenario_name=str(data["scenario_name"]),
            num_frames=int(data["num_frames"]),
            hardware_info=dict(data.get("hardware_info", {})),
            config_snapshot=dict(data.get("config_snapshot", {})),
            resolution_profile=str(data.get("resolution_profile", "")),
            mean_fps=float(data["mean_fps"]),
            mean_latency_ms=float(data["mean_latency_ms"]),
            p95_latency_ms=float(data["p95_latency_ms"]),
            mean_active_cells=float(data["mean_active_cells"]),
            peak_memory_mb=float(data["peak_memory_mb"]),
            point_throughput_pps=float(data["point_throughput_pps"]),
            detection_precision=float(data.get("detection_precision", 1.0)),
            detection_recall=float(data.get("detection_recall", 1.0)),
            tracking_id_switches=int(data.get("tracking_id_switches", 0)),
            distance_density=dict(data.get("distance_density", {})),
            frame_metrics=list(data.get("frame_metrics", [])),
            timestamp=float(data.get("timestamp", 0.0)),
        )
