"""Unit tests for pluggable mock semantic perception."""

import sys
import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame
from fovea_lidar.contracts.semantic import SemanticPoint
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.perception import (
    MockModel,
    SegmentationModel,
    build_segmentation_model,
)


@pytest.fixture
def mock_model() -> MockModel:
    return MockModel()


@pytest.fixture
def generator() -> SyntheticSceneGenerator:
    return SyntheticSceneGenerator(seed=42)


def test_no_torch_imported():
    """Verify that no torch dependency is loaded in memory."""
    assert "torch" not in sys.modules, "torch was unexpectedly imported!"


def test_mock_model_on_all_synthetic_scenarios(mock_model: MockModel, generator: SyntheticSceneGenerator):
    """Run MockModel.predict() across all 7 synthetic scenarios and verify contract adherence."""
    scenario_names = [
        "empty_road",
        "pedestrian_at_distance",
        "vehicle_crossing",
        "low_confidence_obstacle",
        "resolution_boundary_stress",
        "multi_object_dense",
        "object_disappearing",
    ]

    for scenario in scenario_names:
        frames, gt = generator.generate(scenario, num_frames=2)
        frame = frames[0]
        semantic_points = mock_model.predict(frame)

        assert len(semantic_points) == frame.points.shape[0]
        for sp in semantic_points:
            assert isinstance(sp, SemanticPoint)
            assert 0.0 <= sp.confidence <= 1.0
            assert len(sp.coordinates) == 3


def test_pedestrian_classification(mock_model: MockModel, generator: SyntheticSceneGenerator):
    """Verify pedestrian_at_distance yields PEDESTRIAN points near expected location."""
    frames, gt = generator.generate("pedestrian_at_distance", num_frames=1, initial_distance=15.0)
    sem_points = mock_model.predict(frames[0])

    ped_points = [p for p in sem_points if p.semantic_class == "PEDESTRIAN"]
    assert len(ped_points) > 0, "No PEDESTRIAN points detected"

    # Verify pedestrian coordinates are near 15.0m
    ped_x = [p.coordinates[0] for p in ped_points]
    assert np.all(np.abs(np.array(ped_x) - 15.0) < 1.0)


def test_vehicle_classification(mock_model: MockModel, generator: SyntheticSceneGenerator):
    """Verify vehicle_crossing yields VEHICLE points."""
    frames, gt = generator.generate("vehicle_crossing", num_frames=1)
    sem_points = mock_model.predict(frames[0])

    veh_points = [p for p in sem_points if p.semantic_class == "VEHICLE"]
    assert len(veh_points) > 0, "No VEHICLE points detected"


def test_multi_object_dense_classes(mock_model: MockModel, generator: SyntheticSceneGenerator):
    """Verify multi_object_dense scenario detects concurrent vehicle, pedestrian, and truck classes."""
    frames, _ = generator.generate("multi_object_dense", num_frames=1)
    sem_points = mock_model.predict(frames[0])

    classes_detected = {p.semantic_class for p in sem_points}
    assert "ROAD" in classes_detected
    assert "PEDESTRIAN" in classes_detected
    assert "VEHICLE" in classes_detected
    assert "TRUCK" in classes_detected


def test_low_confidence_obstacle_relative_confidence(mock_model: MockModel, generator: SyntheticSceneGenerator):
    """Verify low_confidence_obstacle produces strictly lower mean confidence than pedestrian scenario."""
    # 1. Pedestrian scenario
    ped_frames, _ = generator.generate("pedestrian_at_distance", num_frames=1)
    ped_sem = mock_model.predict(ped_frames[0])
    ped_cluster_confs = [p.confidence for p in ped_sem if p.semantic_class == "PEDESTRIAN"]
    assert len(ped_cluster_confs) > 0
    mean_ped_conf = float(np.mean(ped_cluster_confs))

    # 2. Ambiguous low-confidence obstacle scenario
    low_frames, _ = generator.generate("low_confidence_obstacle", num_frames=1)
    low_sem = mock_model.predict(low_frames[0])
    # Points not classified as ROAD
    obs_cluster_confs = [p.confidence for p in low_sem if p.semantic_class != "ROAD"]
    assert len(obs_cluster_confs) > 0
    mean_obs_conf = float(np.mean(obs_cluster_confs))

    # Assert relative relationship
    assert mean_obs_conf < mean_ped_conf, (
        f"Expected ambiguous obstacle confidence ({mean_obs_conf}) to be lower than "
        f"pedestrian confidence ({mean_ped_conf})"
    )


def test_registry_factory_behavior():
    """Verify factory builds MockModel or raises clear NotImplementedError for other backends."""
    # 1. Default mock backend
    model = build_segmentation_model()
    assert isinstance(model, MockModel)
    assert isinstance(model, SegmentationModel)

    # 2. Explicit pretrained backend raises NotImplementedError
    with pytest.raises(NotImplementedError) as exc_pre:
        build_segmentation_model(config={"perception": {"backend": "pretrained"}})
    assert "not yet implemented" in str(exc_pre.value).lower()

    # 3. Explicit custom backend raises NotImplementedError
    with pytest.raises(NotImplementedError) as exc_cust:
        build_segmentation_model(config={"perception": {"backend": "custom"}})
    assert "not yet configured" in str(exc_cust.value).lower()

    # 4. Unknown backend raises ValueError
    with pytest.raises(ValueError) as exc_unknown:
        build_segmentation_model(config={"perception": {"backend": "invalid_backend_xyz"}})
    assert "unknown perception backend" in str(exc_unknown.value).lower()
