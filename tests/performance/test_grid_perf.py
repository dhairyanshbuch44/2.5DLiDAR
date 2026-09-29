"""Performance benchmarks for AdaptiveGridEngine point projection and cell allocation."""

import json
from pathlib import Path
import time
import numpy as np
import pytest

from fovea_lidar.mapping.grid_engine import AdaptiveGridEngine

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
GRID_PERF_FILE = EXPERIMENTS_DIR / "grid_perf.json"


def test_grid_projection_performance():
    """Benchmark point-to-cell projection throughput across scaling point counts (1k, 10k, 100k).

    Measures real wall-clock execution time and records results to experiments/grid_perf.json.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    point_counts = [1_000, 10_000, 100_000]
    rng = np.random.RandomState(42)
    engine = AdaptiveGridEngine()

    benchmark_data = {
        "benchmark": "grid_projection_throughput",
        "description": "Measured wall-clock timing for AdaptiveGridEngine.project_points()",
        "runs": [],
    }

    for count in point_counts:
        # Generate realistic points in sensor field of view
        r = rng.uniform(0.5, 95.0, size=count).astype(np.float32)
        theta = rng.uniform(-np.pi, np.pi, size=count).astype(np.float32)
        x = (r * np.cos(theta)).astype(np.float32)
        y = (r * np.sin(theta)).astype(np.float32)
        z = rng.uniform(-1.5, 2.0, size=count).astype(np.float32)
        pts = np.column_stack([x, y, z])

        # Measure projection time
        t0 = time.perf_counter()
        proj = engine.project_points(pts)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        occupied_cells = len(proj)
        total_assigned = sum(len(idx) for idx in proj.values())

        assert total_assigned == count, "Zero data loss violated in benchmark"
        assert elapsed_ms > 0.0

        run_entry = {
            "input_point_count": count,
            "occupied_cells_count": occupied_cells,
            "projection_time_ms": round(elapsed_ms, 3),
            "points_per_second": round(count / (elapsed_ms / 1000.0), 1),
        }
        benchmark_data["runs"].append(run_entry)

    # Persist results
    with open(GRID_PERF_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    assert GRID_PERF_FILE.is_file()
    assert len(benchmark_data["runs"]) == 3
    for run in benchmark_data["runs"]:
        assert run["projection_time_ms"] > 0.0
