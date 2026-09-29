"""Performance benchmarks for the Fovea-LiDAR preprocessing pipeline."""

import json
from pathlib import Path
import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame
from fovea_lidar.preprocessing.pipeline import Preprocessor

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
PERF_OUTPUT_FILE = EXPERIMENTS_DIR / "preprocessing_perf.json"


def test_preprocessing_performance_scaling():
    """Benchmark preprocessing pipeline over increasing point cloud sizes (1k, 10k, 100k).

    Records real, measured wall-clock latencies to experiments/preprocessing_perf.json.
    Does not assert hard latency thresholds to remain robust across differing hardware.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    point_counts = [1_000, 10_000, 100_000]
    rng = np.random.RandomState(42)
    preprocessor = Preprocessor()

    benchmark_results = {
        "benchmark": "preprocessing_latency_scaling",
        "description": "Measured wall-clock timing for Preprocessor.process() across point loads",
        "runs": [],
    }

    for count in point_counts:
        # Generate realistic points in sensor field of view
        r = rng.uniform(1.0, 45.0, size=count).astype(np.float32)
        theta = rng.uniform(-np.pi / 3, np.pi / 3, size=count).astype(np.float32)
        x = (r * np.cos(theta)).astype(np.float32)
        y = (r * np.sin(theta)).astype(np.float32)
        z = rng.uniform(-1.5, 2.5, size=count).astype(np.float32)
        intensity = rng.uniform(0.1, 0.9, size=count).astype(np.float32)
        pts = np.column_stack([x, y, z, intensity])

        frame = PointCloudFrame(
            points=pts,
            timestamp=0.0,
            frame_id="bench_lidar",
            source=FrameSource.MOCK,
        )

        cleaned_frame, metrics = preprocessor.process(frame)

        # Assert pipeline completion and real measured metrics
        assert metrics.input_point_count == count
        assert metrics.output_point_count == len(cleaned_frame.points)
        assert metrics.processing_time_ms > 0.0

        run_entry = {
            "input_point_count": count,
            "output_point_count": metrics.output_point_count,
            "filtering_ratio": round(metrics.filtering_ratio, 4),
            "processing_time_ms": round(metrics.processing_time_ms, 3),
            "stage_timings_ms": {k: round(v, 3) for k, v in metrics.stage_timings_ms.items()},
        }
        benchmark_results["runs"].append(run_entry)

    # Persist measured timing to experiments/preprocessing_perf.json
    with open(PERF_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    assert PERF_OUTPUT_FILE.is_file()
    assert len(benchmark_results["runs"]) == 3
    for run in benchmark_results["runs"]:
        assert run["processing_time_ms"] > 0.0
        assert "stage_timings_ms" in run
