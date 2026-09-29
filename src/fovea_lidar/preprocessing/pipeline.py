"""Configurable, instrumented LiDAR preprocessing pipeline for Fovea-LiDAR."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np

from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.contracts.preprocessing import PreprocessingMetrics
from fovea_lidar.preprocessing import stages
from fovea_lidar.utils.config import Config, load_config
from fovea_lidar.utils.logging_setup import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Configurable pipeline that applies sequential preprocessing stages to PointCloudFrames.

    Reads configuration from configs/default.yaml under 'preprocessing' (or a provided
    Config instance) to dynamically enable, tune, and order preprocessing steps.

    Attributes
    ----------
    config : Union[Config, Dict[str, Any]]
        Active configuration mapping.
    stage_order : List[str]
        Ordered sequence of stage identifiers to execute.
    """

    def __init__(self, config: Optional[Union[Config, Dict[str, Any], Path, str]] = None) -> None:
        if config is None:
            full_config = load_config()
            self.config = full_config.get("preprocessing", {})
        elif isinstance(config, (str, Path)):
            full_config = load_config(config)
            self.config = full_config.get("preprocessing", {})
        elif isinstance(config, Config):
            self.config = config.get("preprocessing", config)
        elif isinstance(config, dict):
            self.config = config.get("preprocessing", config)
        else:
            raise TypeError(f"Invalid config type for Preprocessor: {type(config).__name__}")

        # Extract stage execution order from config
        self.stage_order: List[str] = list(
            self.config.get(
                "stage_order",
                [
                    "coordinate_transform",
                    "range_filter",
                    "height_filter",
                    "voxel_downsample",
                    "statistical_outlier_removal",
                    "radius_outlier_removal",
                    "ground_candidate_extraction",
                ],
            )
        )

    def process(self, frame: PointCloudFrame) -> Tuple[PointCloudFrame, PreprocessingMetrics]:
        """Execute configured preprocessing stages and measure quantitative performance.

        Parameters
        ----------
        frame : PointCloudFrame
            Raw input point cloud frame.

        Returns
        -------
        Tuple[PointCloudFrame, PreprocessingMetrics]
            (cleaned_frame, measured_metrics)
        """
        start_total = time.perf_counter()
        input_count = int(frame.points.shape[0])
        stage_timings: Dict[str, float] = {}

        # Edge case: Empty input point cloud (brief section 45)
        if input_count == 0:
            logger.warning("Preprocessor received empty input frame (0 points). Bypassing filtering.")
            elapsed_ms = (time.perf_counter() - start_total) * 1000.0
            metrics = PreprocessingMetrics(
                input_point_count=0,
                output_point_count=0,
                filtering_ratio=0.0,
                processing_time_ms=elapsed_ms,
                stage_timings_ms={},
            )
            empty_frame = PointCloudFrame(
                points=np.empty((0, frame.points.shape[1] if frame.points.ndim == 2 else 3), dtype=np.float32),
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                source=frame.source,
                metadata=dict(frame.metadata),
            )
            empty_frame.metadata["preprocessing_metrics"] = metrics.to_dict()
            return empty_frame, metrics

        current_points = frame.points
        metadata = dict(frame.metadata)

        for stage_name in self.stage_order:
            stage_cfg = self.config.get(stage_name, {})
            if isinstance(stage_cfg, Config):
                stage_dict = stage_cfg.to_dict()
            elif isinstance(stage_cfg, dict):
                stage_dict = stage_cfg
            else:
                stage_dict = {}

            # Skip if explicitly disabled
            if not stage_dict.get("enabled", True):
                continue

            # If points became empty after a previous filter, halt stage execution gracefully
            if current_points.shape[0] == 0:
                logger.warning(
                    "Point cloud became empty during stage '%s'. Halting downstream stages.",
                    stage_name,
                )
                break

            t0 = time.perf_counter()

            if stage_name == "coordinate_transform":
                transform_matrix = stage_dict.get("matrix")
                current_points = stages.coordinate_transform(current_points, transform_matrix)

            elif stage_name == "range_filter":
                min_r = stage_dict.get("min_range", 0.5)
                max_r = stage_dict.get("max_range", 50.0)
                current_points = stages.range_filter(current_points, min_range=min_r, max_range=max_r)

            elif stage_name == "height_filter":
                min_z = stage_dict.get("min_z", -3.0)
                max_z = stage_dict.get("max_z", 4.0)
                current_points = stages.height_filter(current_points, min_z=min_z, max_z=max_z)

            elif stage_name == "voxel_downsample":
                voxel_size = stage_dict.get("voxel_size", 0.10)
                current_points = stages.voxel_downsample(current_points, voxel_size=voxel_size)

            elif stage_name == "statistical_outlier_removal":
                nb_neighbors = stage_dict.get("nb_neighbors", 20)
                std_ratio = stage_dict.get("std_ratio", 2.0)
                current_points = stages.statistical_outlier_removal(
                    current_points, nb_neighbors=nb_neighbors, std_ratio=std_ratio
                )

            elif stage_name == "radius_outlier_removal":
                radius = stage_dict.get("radius", 0.5)
                min_neighbors = stage_dict.get("min_neighbors", 4)
                current_points = stages.radius_outlier_removal(
                    current_points, radius=radius, min_neighbors=min_neighbors
                )

            elif stage_name == "ground_candidate_extraction":
                dist_thresh = stage_dict.get("distance_threshold", 0.15)
                max_iter = stage_dict.get("max_iterations", 100)
                normal_thresh = stage_dict.get("upright_normal_threshold", 0.8)
                current_points, ground_mask = stages.ground_candidate_extraction(
                    current_points,
                    distance_threshold=dist_thresh,
                    max_iterations=max_iter,
                    upright_normal_threshold=normal_thresh,
                )
                metadata["ground_mask"] = ground_mask.tolist()
                metadata["ground_point_count"] = int(np.count_nonzero(ground_mask))

            stage_timings[stage_name] = (time.perf_counter() - t0) * 1000.0

        output_count = int(current_points.shape[0])
        total_elapsed_ms = (time.perf_counter() - start_total) * 1000.0
        filtering_ratio = float((input_count - output_count) / input_count) if input_count > 0 else 0.0

        metrics = PreprocessingMetrics(
            input_point_count=input_count,
            output_point_count=output_count,
            filtering_ratio=filtering_ratio,
            processing_time_ms=total_elapsed_ms,
            stage_timings_ms=stage_timings,
        )

        metadata["preprocessing_metrics"] = metrics.to_dict()

        cleaned_frame = PointCloudFrame(
            points=current_points,
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            source=frame.source,
            metadata=metadata,
        )

        return cleaned_frame, metrics
