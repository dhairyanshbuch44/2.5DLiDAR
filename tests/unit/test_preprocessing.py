"""Unit tests for point cloud preprocessing stages and pipeline."""

import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame
from fovea_lidar.contracts.preprocessing import PreprocessingMetrics
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.preprocessing.pipeline import Preprocessor
from fovea_lidar.preprocessing.stages import (
    coordinate_transform,
    ground_candidate_extraction,
    height_filter,
    radius_outlier_removal,
    range_filter,
    statistical_outlier_removal,
    voxel_downsample,
)


# ==============================================================================
# Individual Stage Tests
# ==============================================================================

def test_coordinate_transform():
    """Verify coordinate_transform correctly rotates/translates coordinates and preserves intensity."""
    points = np.array(
        [
            [1.0, 0.0, 0.0, 0.8],
            [0.0, 2.0, 0.0, 0.4],
        ],
        dtype=np.float32,
    )
    # Translation matrix: Tx=5, Ty=10, Tz=-2
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = 5.0
    matrix[1, 3] = 10.0
    matrix[2, 3] = -2.0

    transformed = coordinate_transform(points, matrix)
    assert transformed.shape == (2, 4)
    expected_p0 = np.array([6.0, 10.0, -2.0, 0.8], dtype=np.float32)
    expected_p1 = np.array([5.0, 12.0, -2.0, 0.4], dtype=np.float32)
    assert np.allclose(transformed[0], expected_p0)
    assert np.allclose(transformed[1], expected_p1)


def test_range_filter():
    """Verify range_filter retains only points within Euclidean distance [min_r, max_r]."""
    points = np.array(
        [
            [0.1, 0.0, 0.0],   # dist = 0.1 (too close)
            [0.3, 0.4, 0.0],   # dist = 0.5 (on lower bound)
            [3.0, 4.0, 0.0],   # dist = 5.0 (inside)
            [30.0, 40.0, 0.0], # dist = 50.0 (on upper bound)
            [60.0, 0.0, 0.0],  # dist = 60.0 (too far)
        ],
        dtype=np.float32,
    )

    filtered = range_filter(points, min_range=0.5, max_range=50.0)
    assert filtered.shape[0] == 3
    assert np.allclose(filtered[0], [0.3, 0.4, 0.0])
    assert np.allclose(filtered[1], [3.0, 4.0, 0.0])
    assert np.allclose(filtered[2], [30.0, 40.0, 0.0])


def test_height_filter():
    """Verify height_filter bounds the z-axis coordinates."""
    points = np.array(
        [
            [0.0, 0.0, -5.0], # too low
            [1.0, 0.0, -1.0], # inside
            [2.0, 0.0, 0.0],  # inside
            [3.0, 0.0, 2.5],  # inside
            [4.0, 0.0, 10.0], # too high
        ],
        dtype=np.float32,
    )

    filtered = height_filter(points, min_z=-2.0, max_z=3.0)
    assert filtered.shape[0] == 3
    assert np.allclose(filtered[:, 2], [-1.0, 0.0, 2.5])


def test_voxel_downsample():
    """Verify voxel_downsample aggregates tightly clustered points into centroids."""
    # 50 points closely packed strictly within a single 0.2m voxel [2.01, 2.19]
    rng = np.random.RandomState(42)
    dense_cluster = rng.uniform(low=2.01, high=2.19, size=(50, 3)).astype(np.float32)
    # One separate point in a distant voxel
    distant_point = np.array([[10.0, 10.0, 1.0]], dtype=np.float32)
    points = np.vstack([dense_cluster, distant_point])

    downsampled = voxel_downsample(points, voxel_size=0.2)
    # Exactly 2 occupied voxels
    assert downsampled.shape[0] == 2
    # One near (2.1, 2.1, 2.1) and one near (10, 10, 1)
    dists_to_cluster = np.linalg.norm(downsampled - np.array([2.1, 2.1, 2.1]), axis=1)
    assert np.min(dists_to_cluster) < 0.1


def test_statistical_outlier_removal():
    """Verify statistical_outlier_removal eliminates isolated noise points."""
    rng = np.random.RandomState(42)
    # 30 tight points
    cluster = rng.normal(loc=0.0, scale=0.1, size=(30, 3)).astype(np.float32)
    # 1 distant outlier
    outlier = np.array([[50.0, 50.0, 50.0]], dtype=np.float32)
    points = np.vstack([cluster, outlier])

    cleaned = statistical_outlier_removal(points, nb_neighbors=15, std_ratio=1.5)
    assert cleaned.shape[0] == 30
    assert not np.any(cleaned[:, 0] > 10.0)


def test_radius_outlier_removal():
    """Verify radius_outlier_removal eliminates points with too few spatial neighbors."""
    rng = np.random.RandomState(42)
    cluster = rng.normal(loc=0.0, scale=0.05, size=(10, 3)).astype(np.float32)
    isolated = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    points = np.vstack([cluster, isolated])

    cleaned = radius_outlier_removal(points, radius=0.5, min_neighbors=3)
    assert cleaned.shape[0] == 10
    assert not np.any(cleaned[:, 0] > 5.0)


def test_ground_candidate_extraction():
    """Verify ground plane extraction classifies horizontal plane without dropping points."""
    rng = np.random.RandomState(42)
    # Flat horizontal ground points (z ≈ 0.0)
    ground_x = rng.uniform(0.0, 10.0, size=100)
    ground_y = rng.uniform(-5.0, 5.0, size=100)
    ground_z = rng.normal(0.0, 0.02, size=100)
    ground = np.column_stack([ground_x, ground_y, ground_z]).astype(np.float32)

    # Elevated obstacle points (z ≈ 2.0)
    obs = np.array([[5.0, 0.0, 2.0], [5.1, 0.1, 2.1], [5.2, -0.1, 2.0]], dtype=np.float32)
    pts = np.vstack([ground, obs])

    ret_points, ground_mask = ground_candidate_extraction(
        pts, distance_threshold=0.10, max_iterations=50, upright_normal_threshold=0.8
    )

    # Must preserve ALL input points (do not silently drop non-ground points)
    assert ret_points.shape[0] == pts.shape[0]
    assert ground_mask.shape[0] == pts.shape[0]

    # Ground points should have high inlier rate
    assert np.count_nonzero(ground_mask[:100]) >= 90
    # Elevated obstacle points must not be classified as ground
    assert not np.any(ground_mask[100:])


# ==============================================================================
# Full Pipeline and Metrics Tests
# ==============================================================================

def test_pipeline_on_empty_road_scenario():
    """Verify full Preprocessor pipeline execution on synthetic empty_road scenario."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("empty_road", num_frames=2)

    preprocessor = Preprocessor()
    cleaned_frame, metrics = preprocessor.process(frames[0])

    assert isinstance(cleaned_frame, PointCloudFrame)
    assert isinstance(metrics, PreprocessingMetrics)
    assert metrics.input_point_count == len(frames[0].points)
    assert metrics.output_point_count == len(cleaned_frame.points)
    assert metrics.output_point_count > 0
    assert metrics.processing_time_ms > 0.0
    assert 0.0 <= metrics.filtering_ratio < 1.0

    # Verify metadata attachment
    assert "preprocessing_metrics" in cleaned_frame.metadata
    assert cleaned_frame.metadata["preprocessing_metrics"]["output_point_count"] == metrics.output_point_count


def test_pipeline_on_multi_object_dense_scenario():
    """Verify full pipeline execution on multi_object_dense scenario."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("multi_object_dense", num_frames=2)

    preprocessor = Preprocessor()
    cleaned_frame, metrics = preprocessor.process(frames[0])

    assert metrics.input_point_count > 500
    assert metrics.output_point_count > 0
    assert metrics.output_point_count == len(cleaned_frame.points)
    assert "ground_point_count" in cleaned_frame.metadata


def test_pipeline_empty_input_handling():
    """Verify that an empty input frame (0 points) logs a warning and returns cleanly."""
    empty_frame = PointCloudFrame(
        points=np.empty((0, 3), dtype=np.float32),
        timestamp=0.0,
        frame_id="empty_sensor",
    )

    preprocessor = Preprocessor()
    cleaned, metrics = preprocessor.process(empty_frame)

    assert cleaned.points.shape[0] == 0
    assert metrics.input_point_count == 0
    assert metrics.output_point_count == 0
    assert metrics.filtering_ratio == 0.0
    assert metrics.processing_time_ms >= 0.0


def test_pipeline_empty_after_filtering_handling():
    """Verify that a point cloud completely filtered out produces clean empty output without raising."""
    # Points at 5 meters
    pts = np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0]], dtype=np.float32)
    frame = PointCloudFrame(points=pts, timestamp=1.0, frame_id="far_filter")

    # Custom config with impossible range filter: min_range=100.0m
    custom_cfg = {
        "stage_order": ["range_filter"],
        "range_filter": {"enabled": True, "min_range": 100.0, "max_range": 200.0},
    }
    preprocessor = Preprocessor(config=custom_cfg)
    cleaned, metrics = preprocessor.process(frame)

    assert cleaned.points.shape[0] == 0
    assert metrics.input_point_count == 2
    assert metrics.output_point_count == 0
    assert metrics.filtering_ratio == 1.0
