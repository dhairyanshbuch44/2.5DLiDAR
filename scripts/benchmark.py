#!/usr/bin/env python
"""Command-line benchmark and ablation runner for Fovea-LiDAR.

Executes evaluations across ablation study modes and scenarios, writes
reproducibility JSON/CSV records, and generates publication-grade comparison plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from fovea_lidar.benchmarking.harness import BenchmarkRunner
from fovea_lidar.contracts.benchmark_result import BenchmarkResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("benchmark_cli")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "experiments"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots"


def load_benchmark_config() -> Dict[str, Any]:
    """Load default benchmark configuration file if present."""
    cfg_file = CONFIGS_DIR / "benchmark.yaml"
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def generate_plots(
    results: List[BenchmarkResult],
    output_dir: Path,
) -> List[Path]:
    """Generate comparison plots from benchmark results and save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files: List[Path] = []

    if not results:
        logger.warning("No results to plot.")
        return plot_files

    # Apply clean publication-ready style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # Group metrics across modes (averaged over evaluated scenarios)
    mode_order = [
        "mode_1_uniform",
        "mode_2_distance_only",
        "mode_3_distance_semantic",
        "mode_4_distance_semantic_risk",
        "mode_5_full",
    ]

    # Filter modes present in results
    modes = [m for m in mode_order if any(r.mode_name == m for r in results)]
    if not modes:
        modes = sorted(list({r.mode_name for r in results}))

    mode_labels = {
        "mode_1_uniform": "Mode 1\nUniform (0.05m)",
        "mode_2_distance_only": "Mode 2\nDistance Only",
        "mode_3_distance_semantic": "Mode 3\n+ Semantic",
        "mode_4_distance_semantic_risk": "Mode 4\n+ Risk",
        "mode_5_full": "Mode 5\nFull Adaptive",
    }

    x_indices = np.arange(len(modes))
    display_names = [mode_labels.get(m, m) for m in modes]

    # Compute mode averages
    avg_cells = [
        float(np.mean([r.mean_active_cells for r in results if r.mode_name == m]))
        for m in modes
    ]
    avg_fps = [
        float(np.mean([r.mean_fps for r in results if r.mode_name == m]))
        for m in modes
    ]
    avg_mem = [
        float(np.mean([r.peak_memory_mb for r in results if r.mode_name == m]))
        for m in modes
    ]

    # --- Plot 1: Active Cell Count vs Mode ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        x_indices,
        avg_cells,
        color=["#e11d48", "#f59e0b", "#3b82f6", "#6366f1", "#10b981"][: len(modes)],
        edgecolor="#1e293b",
        linewidth=1.2,
        width=0.55,
    )
    ax.set_ylabel("Mean Active Map Cells", fontsize=12, fontweight="bold")
    ax.set_title("Memory Efficiency: Active Cell Count vs. Ablation Mode", fontsize=13, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(display_names, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h):,}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    p1 = output_dir / "active_cells_vs_mode.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    plot_files.append(p1)

    # --- Plot 2: FPS vs Mode ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        x_indices,
        avg_fps,
        color=["#0284c7", "#0ea5e9", "#38bdf8", "#7dd3fc", "#10b981"][: len(modes)],
        edgecolor="#1e293b",
        linewidth=1.2,
        width=0.55,
    )
    ax.set_ylabel("Effective Frames Per Second (FPS)", fontsize=12, fontweight="bold")
    ax.set_title("Real-Time Throughput: Pipeline FPS vs. Ablation Mode", fontsize=13, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(display_names, fontweight="semibold")
    ax.axhline(15.0, color="#ef4444", linestyle=":", linewidth=1.8, label="Real-Time Target (15 FPS)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    p2 = output_dir / "fps_vs_mode.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    plot_files.append(p2)

    # --- Plot 3: Memory Usage vs Mode ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        x_indices,
        avg_mem,
        color=["#8b5cf6", "#a855f7", "#c084fc", "#d8b4fe", "#10b981"][: len(modes)],
        edgecolor="#1e293b",
        linewidth=1.2,
        width=0.55,
    )
    ax.set_ylabel("Peak Memory Allocation (MB)", fontsize=12, fontweight="bold")
    ax.set_title("Memory Footprint: Peak RAM (MB) vs. Ablation Mode", fontsize=13, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(display_names, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f} MB",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    p3 = output_dir / "memory_vs_mode.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    plot_files.append(p3)

    # --- Plot 4: Distance-wise Density Comparison (Brief Section 33) ---
    # Compare Mode 1 (Uniform) vs Mode 5 (Full Adaptive)
    u_results = [r for r in results if r.mode_name == "mode_1_uniform"]
    f_results = [r for r in results if r.mode_name == "mode_5_full"]

    if u_results and f_results:
        bands = ["0-10m", "10-30m", "30-60m", "60-100m"]
        u_densities = [
            float(np.mean([r.distance_density.get(b, 0) for r in u_results]))
            for b in bands
        ]
        f_densities = [
            float(np.mean([r.distance_density.get(b, 0) for r in f_results]))
            for b in bands
        ]

        b_idx = np.arange(len(bands))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(
            b_idx - width / 2,
            u_densities,
            width,
            label="Mode 1: Uniform (0.05m)",
            color="#e11d48",
            edgecolor="#1e293b",
        )
        ax.bar(
            b_idx + width / 2,
            f_densities,
            width,
            label="Mode 5: Full Fovea-LiDAR",
            color="#10b981",
            edgecolor="#1e293b",
        )

        ax.set_ylabel("Cell Count", fontsize=12, fontweight="bold")
        ax.set_title("Distance-Wise Cell Density Breakdown (Brief Section 33)", fontsize=13, fontweight="bold")
        ax.set_xticks(b_idx)
        ax.set_xticklabels(bands, fontweight="semibold")
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        p4 = output_dir / "distance_breakdown_vs_mode.png"
        fig.savefig(p4, dpi=200)
        plt.close(fig)
        plot_files.append(p4)

    # --- Plot 5: Accuracy Metrics vs Mode ---
    avg_prec = [
        float(np.mean([r.detection_precision for r in results if r.mode_name == m]))
        for m in modes
    ]
    avg_rec = [
        float(np.mean([r.detection_recall for r in results if r.mode_name == m]))
        for m in modes
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.35
    ax.bar(
        x_indices - width / 2,
        [p * 100.0 for p in avg_prec],
        width,
        label="Detection Precision (%)",
        color="#3b82f6",
        edgecolor="#1e293b",
    )
    ax.bar(
        x_indices + width / 2,
        [r * 100.0 for r in avg_rec],
        width,
        label="Detection Recall (%)",
        color="#10b981",
        edgecolor="#1e293b",
    )

    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Perception Accuracy: Precision & Recall vs. Mode", fontsize=13, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(display_names, fontweight="semibold")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    p5 = output_dir / "accuracy_vs_mode.png"
    fig.savefig(p5, dpi=200)
    plt.close(fig)
    plot_files.append(p5)

    return plot_files


def run_benchmarks(
    scenarios: List[str],
    modes: List[str],
    num_frames: int = 10,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    skip_plots: bool = False,
) -> List[BenchmarkResult]:
    """Execute benchmark matrix and save CSV, JSON, and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner()

    results: List[BenchmarkResult] = []
    total_runs = len(scenarios) * len(modes)
    current_run = 0

    print("=" * 80)
    print(f"Fovea-LiDAR Automated Benchmarking & Ablation Study ({total_runs} total runs)")
    print(f"Scenarios ({len(scenarios)}): {scenarios}")
    print(f"Ablation Modes ({len(modes)}): {modes}")
    print(f"Frames per run: {num_frames}")
    print("=" * 80)

    for scen in scenarios:
        for mode in modes:
            current_run += 1
            print(f"[{current_run:02d}/{total_runs:02d}] Running mode '{mode}' on scenario '{scen}'...", end=" ", flush=True)

            res = runner.run(
                pipeline_config=None,
                scenario_name=scen,
                mode_name=mode,
                num_frames=num_frames,
            )
            results.append(res)
            print(f"Done! (FPS: {res.mean_fps:.1f}, Active Cells: {int(res.mean_active_cells):,}, Latency: {res.mean_latency_ms:.1f}ms)")

    # Print Per-Class Semantic Perception Accuracy Table
    print("\n" + "=" * 84)
    print("       SEMANTIC PERCEPTION ACCURACY (PER-CLASS PRECISION / RECALL / F1)")
    print("=" * 84)
    print(f"{'Scenario':<26} {'Mode':<16} {'Class':<16} {'Precision':<10} {'Recall':<10} {'F1':<8}")
    print("-" * 84)

    has_class_metrics = False
    for r in results:
        pcm = getattr(r, "per_class_metrics", {})
        if not pcm:
            continue
        has_class_metrics = True
        for cls_name, m in sorted(pcm.items()):
            prec_str = f"{m['precision']*100:.1f}%"
            rec_str = f"{m['recall']*100:.1f}%"
            f1_str = f"{m['f1']*100:.1f}%"
            print(f"{r.scenario_name:<26} {r.mode_name:<16} {cls_name:<16} {prec_str:<10} {rec_str:<10} {f1_str:<8}")
    if not has_class_metrics:
        print("  (No objects present to evaluate class metrics)")
    print("=" * 84 + "\n")

    # 1. Export Raw JSON Results
    json_path = output_dir / "benchmark_results.json"
    json_payload = {
        "benchmark_metadata": {
            "total_runs": len(results),
            "scenarios": scenarios,
            "modes": modes,
            "num_frames": num_frames,
            "hardware": results[0].hardware_info if results else {},
        },
        "results": [
            {
                **r.to_dict(),
                "per_class_metrics": getattr(r, "per_class_metrics", {}),
            }
            for r in results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    logger.info("Exported JSON results to: %s", json_path)

    # 2. Export Flattened CSV Results
    csv_path = output_dir / "benchmark_results.csv"
    if results:
        fieldnames = list(results[0].to_flattened_dict().keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_flattened_dict())
        logger.info("Exported CSV results to: %s", csv_path)

    # 3. Generate Publication Plots
    if not skip_plots:
        generated_plots = generate_plots(results, plots_dir)
        logger.info("Generated %d publication figures in: %s", len(generated_plots), plots_dir)

    print("=" * 80)
    print("Benchmark Suite Execution Complete!")
    print(f"  JSON Results: {json_path}")
    print(f"  CSV Results:  {csv_path}")
    print(f"  Plots:        {plots_dir}")
    print("=" * 80)

    return results


def main() -> None:
    """CLI entrypoint."""
    cfg = load_benchmark_config()

    default_scenarios = cfg.get(
        "scenarios",
        [
            "multi_object_dense",
            "vehicle_crossing",
            "pedestrian_at_distance",
            "empty_road",
            "object_disappearing",
            "low_confidence_obstacle",
            "resolution_boundary_stress",
        ],
    )
    default_modes = cfg.get(
        "modes",
        [
            "mode_1_uniform",
            "mode_2_distance_only",
            "mode_3_distance_semantic",
            "mode_4_distance_semantic_risk",
            "mode_5_full",
        ],
    )
    default_frames = int(cfg.get("num_frames", 10))

    parser = argparse.ArgumentParser(
        description="Fovea-LiDAR Automated Benchmark & Ablation Study Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=default_scenarios,
        help="Synthetic scenarios to evaluate",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=default_modes,
        help="Ablation study modes to benchmark",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=default_frames,
        help="Number of LiDAR frames per run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Target output directory for benchmark results",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Target output directory for generated plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )

    args = parser.parse_args()

    run_benchmarks(
        scenarios=args.scenarios,
        modes=args.modes,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        skip_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
