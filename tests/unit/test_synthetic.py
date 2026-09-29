"""Unit tests for synthetic point cloud scene generator."""

import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame
from fovea_lidar.io.synthetic import SyntheticSceneGenerator


@pytest.fixture
def generator() -> SyntheticSceneGenerator:
    return SyntheticSceneGenerator(seed=123, dt=0.1)


def test_empty_road_scenario(generator: SyntheticSceneGenerator):
    """Verify empty_road produces expected frames, non-empty clouds, and zero obstacles."""
    frames, gt = generator.generate("empty_road", num_frames=5)
    assert len(frames) == 5
    assert gt["scenario"] == "empty_road"
    assert len(gt["frames"]) == 5

    for idx, frame in enumerate(frames):
        assert isinstance(frame, PointCloudFrame)
        assert frame.source == FrameSource.MOCK
        assert frame.points.shape[0] > 100
        assert frame.timestamp == pytest.approx(idx * 0.1)
        assert gt["frames"][idx]["objects"] == []


def test_pedestrian_at_distance_scenario(generator: SyntheticSceneGenerator):
    """Verify pedestrian_at_distance produces moving pedestrian annotations.

    ego_speed=0.0 is passed explicitly so that sensor-frame position equals
    world-frame position, giving a static X == initial_distance throughout.
    """
    frames, gt = generator.generate(
        "pedestrian_at_distance", num_frames=6, initial_distance=12.0, ego_speed=0.0
    )
    assert len(frames) == 6

    for idx, frame in enumerate(frames):
        assert frame.points.shape[0] > 100
        objects = gt["frames"][idx]["objects"]
        assert len(objects) == 1
        obj = objects[0]
        assert obj["semantic_class"] == "PEDESTRIAN"
        # With ego_speed=0, sensor-frame X == world-frame initial_distance (static ped)
        assert obj["position"][0] == pytest.approx(12.0)


def test_vehicle_crossing_scenario(generator: SyntheticSceneGenerator):
    """Verify vehicle_crossing produces lateral motion ground truth."""
    frames, gt = generator.generate("vehicle_crossing", num_frames=8)
    assert len(frames) == 8

    prev_y = None
    for idx, frame in enumerate(frames):
        assert frame.points.shape[0] > 100
        objects = gt["frames"][idx]["objects"]
        assert len(objects) == 1
        obj = objects[0]
        assert obj["semantic_class"] == "VEHICLE"
        curr_y = obj["position"][1]
        if prev_y is not None:
            assert curr_y > prev_y  # Crossing laterally
        prev_y = curr_y


def test_low_confidence_obstacle_scenario(generator: SyntheticSceneGenerator):
    """Verify low_confidence_obstacle produces ambiguous obstacle annotations."""
    frames, gt = generator.generate("low_confidence_obstacle", num_frames=4)
    assert len(frames) == 4

    for idx, frame in enumerate(frames):
        assert frame.points.shape[0] > 50
        objects = gt["frames"][idx]["objects"]
        assert len(objects) == 1
        assert objects[0]["semantic_class"] == "UNKNOWN_OBSTACLE"
        assert objects[0]["is_ambiguous"] is True


def test_resolution_boundary_stress_scenario(generator: SyntheticSceneGenerator):
    """Verify boundary stress scenario embeds points around configured distances."""
    frames, gt = generator.generate("resolution_boundary_stress", num_frames=5)
    assert len(frames) == 5

    for frame in frames:
        points = frame.points
        # Check that there are points near 10.0m boundary
        x_pts = points[:, 0]
        near_10 = np.any(np.abs(x_pts - 10.0) < 0.05)
        near_25 = np.any(np.abs(x_pts - 25.0) < 0.05)
        assert near_10
        assert near_25


def test_multi_object_dense_scenario(generator: SyntheticSceneGenerator):
    """Verify multi_object_dense generates 4 concurrent annotated objects."""
    frames, gt = generator.generate("multi_object_dense", num_frames=6)
    assert len(frames) == 6

    for idx, frame in enumerate(frames):
        assert frame.points.shape[0] > 200
        objects = gt["frames"][idx]["objects"]
        assert len(objects) == 4
        classes = [o["semantic_class"] for o in objects]
        assert "PEDESTRIAN" in classes
        assert "VEHICLE" in classes
        assert "TRUCK" in classes


def test_object_disappearing_scenario(generator: SyntheticSceneGenerator):
    """Verify object_disappearing removes object after specified frame count."""
    frames, gt = generator.generate("object_disappearing", num_frames=8, disappear_after_frame=4)
    assert len(frames) == 8

    for idx, frame in enumerate(frames):
        assert frame.points.shape[0] > 100
        objects = gt["frames"][idx]["objects"]
        if idx < 4:
            assert len(objects) == 1
            assert objects[0]["semantic_class"] == "PEDESTRIAN"
        else:
            assert len(objects) == 0


def test_scenario_determinism(generator: SyntheticSceneGenerator):
    """Verify that same seed produces byte-identical point clouds."""
    frames1, gt1 = generator.generate("vehicle_crossing", num_frames=5, seed=999)
    frames2, gt2 = generator.generate("vehicle_crossing", num_frames=5, seed=999)

    for f1, f2 in zip(frames1, frames2):
        assert np.array_equal(f1.points, f2.points)
        assert f1.timestamp == f2.timestamp
    assert gt1 == gt2


def test_unknown_scenario_raises(generator: SyntheticSceneGenerator):
    """Verify unknown scenario name raises ValueError."""
    with pytest.raises(ValueError) as exc:
        generator.generate("non_existent_scenario")
    assert "unknown scenario" in str(exc.value).lower()
