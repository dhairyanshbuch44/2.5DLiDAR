"""Integration tests for BenchmarkRunner harness and ablation study evaluation."""

from pathlib import Path
import pytest

from fovea_lidar.benchmarking.harness import BenchmarkRunner
from fovea_lidar.contracts.benchmark_result import BenchmarkResult
from scripts.benchmark import run_benchmarks


def test_uniform_vs_full_adaptive_cell_reduction():
    """Verify project core claim: mode_1_uniform produces measurably higher active cell count than mode_5_full."""
    runner = BenchmarkRunner()

    res_uniform = runner.run(
        scenario_name="multi_object_dense",
        mode_name="mode_1_uniform",
        num_frames=5,
    )
    res_full = runner.run(
        scenario_name="multi_object_dense",
        mode_name="mode_5_full",
        num_frames=5,
    )

    assert isinstance(res_uniform, BenchmarkResult)
    assert isinstance(res_full, BenchmarkResult)

    # Core claim assertion: Uniform high-res allocates more cells than adaptive foveation
    print(f"\n[Ablation Test] mode_1_uniform Mean Cells: {res_uniform.mean_active_cells:.1f}")
    print(f"[Ablation Test] mode_5_full    Mean Cells: {res_full.mean_active_cells:.1f}")
    reduction_ratio = res_uniform.mean_active_cells / res_full.mean_active_cells
    print(f"[Ablation Test] Measured Cell Reduction Ratio: {reduction_ratio:.2f}x")

    assert res_uniform.mean_active_cells > res_full.mean_active_cells, (
        f"mode_1_uniform ({res_uniform.mean_active_cells}) should have strictly more cells than mode_5_full ({res_full.mean_active_cells})"
    )
    assert reduction_ratio >= 1.15, (
        f"Expected at least 1.15x cell reduction in adaptive mode, got {reduction_ratio:.2f}x"
    )

    # In far field (30-60m), uniform allocates drastically more cells than coarse far-field band
    assert res_uniform.distance_density["30-60m"] > res_full.distance_density["30-60m"], (
        f"Uniform 30-60m ({res_uniform.distance_density['30-60m']}) must exceed adaptive 30-60m ({res_full.distance_density['30-60m']})"
    )


def test_benchmark_result_validity_and_reproducibility():
    """Verify BenchmarkResult validates real metrics and captures runtime hardware snapshot."""
    runner = BenchmarkRunner()
    res = runner.run(
        scenario_name="pedestrian_at_distance",
        mode_name="mode_2_distance_only",
        num_frames=3,
    )

    # Ensure validation passes
    res.validate()

    # Hardware metadata assertions
    hw = res.hardware_info
    assert isinstance(hw, dict)
    assert "platform" in hw and len(hw["platform"]) > 0
    assert "cpu_count" in hw and hw["cpu_count"] >= 1
    assert "ram_total_gb" in hw and hw["ram_total_gb"] > 0.0

    # Real non-placeholder performance measurements
    assert res.mean_fps > 0.0
    assert res.mean_latency_ms > 0.0
    assert res.p95_latency_ms >= res.mean_latency_ms * 0.5
    assert res.peak_memory_mb > 0.0
    assert res.point_throughput_pps > 0.0

    # Distance density bands non-empty
    assert sum(res.distance_density.values()) > 0


def test_benchmark_export_json_csv_and_plots(tmp_path: Path):
    """Run benchmark runner on a fast subset, verify JSON, CSV, and plot generation."""
    out_dir = tmp_path / "experiments"
    plots_dir = out_dir / "plots"

    results = run_benchmarks(
        scenarios=["pedestrian_at_distance"],
        modes=["mode_1_uniform", "mode_5_full"],
        num_frames=3,
        output_dir=out_dir,
        plots_dir=plots_dir,
        skip_plots=False,
    )

    assert len(results) == 2

    # Verify JSON file
    json_file = out_dir / "benchmark_results.json"
    assert json_file.exists()
    assert json_file.stat().st_size > 0

    # Verify CSV file
    csv_file = out_dir / "benchmark_results.csv"
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    # Verify generated plot images
    expected_plots = [
        "active_cells_vs_mode.png",
        "fps_vs_mode.png",
        "memory_vs_mode.png",
        "distance_breakdown_vs_mode.png",
        "accuracy_vs_mode.png",
    ]
    for plot_name in expected_plots:
        p_file = plots_dir / plot_name
        assert p_file.exists(), f"Expected plot file {plot_name} was not generated in {plots_dir}"
        assert p_file.stat().st_size > 1000, f"Plot file {plot_name} is empty or corrupt"
