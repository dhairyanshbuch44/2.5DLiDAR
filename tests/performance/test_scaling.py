"""Performance benchmarks evaluating FoveaPipeline scaling across point counts and object counts."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.pipeline import FoveaPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SCALING_PERF_FILE = EXPERIMENTS_DIR / "scaling_perf.json"


def test_pipeline_scaling_performance():
    """Evaluate FoveaPipeline scaling with increasing point counts and object counts.

    Measures real, wall-clock pipeline processing latency across:
    1. Point scaling: varying point cloud density (500, 1500, 3000 ground points)
    2. Object scaling: varying concurrent tracked objects (2, 4, 8 dynamic obstacles)

    Records quantitative measurements to experiments/scaling_perf.json.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    gen = SyntheticSceneGenerator(seed=42)

    point_counts = [500, 1500, 3000]
    object_counts = [2, 4, 8]

    scaling_results: Dict[str, Any] = {
        "benchmark": "fovea_pipeline_scaling",
        "description": "Full-pipeline wall-clock timing across point density and concurrent object counts",
        "timestamp": time.time(),
        "point_scaling_runs": [],
        "object_scaling_runs": [],
    }

    # 1. Point Count Scaling (fixed 4 objects)
    for pts_count in point_counts:
        pipeline = FoveaPipeline()
        frames, _ = gen.generate_multi_object_dense(
            num_frames=3,
            num_ground_points=pts_count,
            num_objects=4,
            seed=42,
        )

        frame_latencies: List[float] = []
        point_totals: List[int] = []
        cell_totals: List[int] = []

        for frame in frames:
            t0 = time.perf_counter()
            res = pipeline.process_frame(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            frame_latencies.append(elapsed_ms)
            point_totals.append(int(frame.points.shape[0]))
            cell_totals.append(len(res.adaptive_map.cells))

        mean_pts = float(np.mean(point_totals))
        mean_lat = float(np.mean(frame_latencies))
        mean_cells = float(np.mean(cell_totals))

        scaling_results["point_scaling_runs"].append(
            {
                "ground_points_parameter": pts_count,
                "mean_total_points": round(mean_pts, 1),
                "mean_latency_ms": round(mean_lat, 3),
                "mean_active_cells": round(mean_cells, 1),
                "throughput_pps": round(mean_pts / max(1e-4, mean_lat / 1000.0), 1),
            }
        )

    # 2. Object Count Scaling (fixed 500 ground points)
    for obj_count in object_counts:
        pipeline = FoveaPipeline()
        frames, _ = gen.generate_multi_object_dense(
            num_frames=3,
            num_ground_points=500,
            num_objects=obj_count,
            seed=42,
        )

        frame_latencies = []
        extraction_times = []
        tracking_times = []
        fovea_counts = []

        for frame in frames:
            t0 = time.perf_counter()
            res = pipeline.process_frame(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            frame_latencies.append(elapsed_ms)
            extraction_times.append(res.metrics.get("extraction_ms", 0.0))
            tracking_times.append(res.metrics.get("tracking_ms", 0.0))
            fovea_counts.append(len(res.active_fovea_regions))

        scaling_results["object_scaling_runs"].append(
            {
                "simultaneous_objects_parameter": obj_count,
                "mean_latency_ms": round(float(np.mean(frame_latencies)), 3),
                "mean_extraction_ms": round(float(np.mean(extraction_times)), 3),
                "mean_tracking_ms": round(float(np.mean(tracking_times)), 3),
                "mean_active_foveas": round(float(np.mean(fovea_counts)), 1),
            }
        )

    # Persist results to experiments/scaling_perf.json
    with open(SCALING_PERF_FILE, "w", encoding="utf-8") as f:
        json.dump(scaling_results, f, indent=2)

    # Verification assertions
    assert SCALING_PERF_FILE.is_file(), "scaling_perf.json was not created"
    assert len(scaling_results["point_scaling_runs"]) == len(point_counts)
    assert len(scaling_results["object_scaling_runs"]) == len(object_counts)

    for run in scaling_results["point_scaling_runs"]:
        assert run["mean_latency_ms"] > 0.0
        assert run["throughput_pps"] > 0.0
        assert not np.isnan(run["mean_latency_ms"])

    for run in scaling_results["object_scaling_runs"]:
        assert run["mean_latency_ms"] > 0.0
        assert run["mean_extraction_ms"] >= 0.0
        assert not np.isnan(run["mean_latency_ms"])
