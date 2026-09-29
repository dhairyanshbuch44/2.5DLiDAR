"""CLI script to generate synthetic LiDAR scenario datasets and ground-truth metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fovea_lidar.io.loaders import save_pcd
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.utils.logging_setup import get_logger, setup_logging


SCENARIOS = [
    "empty_road",
    "pedestrian_at_distance",
    "vehicle_crossing",
    "low_confidence_obstacle",
    "resolution_boundary_stress",
    "multi_object_dense",
    "object_disappearing",
    "flagship_demo",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic LiDAR datasets for Fovea-LiDAR.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "samples" / "synthetic"),
        help="Target directory for synthetic point cloud files.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames per scenario (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42).",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("fovea_lidar.generate_synthetic")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    generator = SyntheticSceneGenerator(seed=args.seed)
    total_files_written = 0

    logger.info("Generating %d synthetic scenarios (frames=%d, seed=%d)...", len(SCENARIOS), args.num_frames, args.seed)

    for scenario_name in SCENARIOS:
        scenario_dir = output_root / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        n_frames = max(args.num_frames, 16) if scenario_name == "flagship_demo" else args.num_frames
        frames, ground_truth = generator.generate(scenario_name, num_frames=n_frames, seed=args.seed)

        # Write each frame as a PCD file
        for idx, frame in enumerate(frames):
            pcd_path = scenario_dir / f"frame_{idx:04d}.pcd"
            save_pcd(frame, pcd_path)
            total_files_written += 1

        # Write ground_truth.json
        gt_path = scenario_dir / "ground_truth.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)
        total_files_written += 1

        logger.info("  [OK] %s: %d frames + ground_truth.json written to %s", scenario_name, len(frames), scenario_dir)

    logger.info("Synthetic dataset generation complete! Total files generated: %d", total_files_written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
