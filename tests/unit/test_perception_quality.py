"""Unit tests verifying semantic perception quality and bias-free object extraction.

Covers:
- Robust multi-class geometric classification (PEDESTRIAN, CYCLIST, VEHICLE, TRUCK, POLE, WALL, CURB, POTHOLE)
- Range-adaptive density scaling (near, mid, far-field pedestrian detection)
- Cyclist detection and extraction in urban scenes
- Curb and pothole ground-level classification
- Unbiased tie-breaking and dynamic confidence weighting in ObjectExtractor
"""

import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame
from fovea_lidar.contracts.semantic import SemanticPoint
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.objects.extraction import ObjectExtractor
from fovea_lidar.perception.mock_model import MockModel


@pytest.fixture
def mock_model() -> MockModel:
    return MockModel()


@pytest.fixture
def generator() -> SyntheticSceneGenerator:
    return SyntheticSceneGenerator(seed=42)


@pytest.fixture
def extractor() -> ObjectExtractor:
    return ObjectExtractor.load()


# ==============================================================================
# 1. Multi-Class Geometric Distinction Tests
# ==============================================================================

def test_pole_vs_pedestrian_no_shadowing(mock_model: MockModel, generator: SyntheticSceneGenerator):
    """Verify that slender vertical poles are classified as POLE and not shadowed by PEDESTRIAN."""
    rng = np.random.RandomState(42)
    # Pole: height 2.0m, width 0.25m x 0.25m at X=12m
    pole_pts = generator._generate_box_cluster(rng, (12.0, 0.0, 1.0), (0.25, 0.25, 2.0), 50, 0.8)
    frame = PointCloudFrame(points=pole_pts, timestamp=0.0, frame_id="lidar_top", source=FrameSource.MOCK)

    sem_pts = mock_model.predict(frame)
    pole_count = sum(1 for p in sem_pts if p.semantic_class == "POLE")
    ped_count = sum(1 for p in sem_pts if p.semantic_class == "PEDESTRIAN")

    assert pole_count > 0, "Pole was not recognized as POLE"
    assert pole_count > ped_count, f"Pole was incorrectly shadowed by PEDESTRIAN: pole={pole_count}, ped={ped_count}"


def test_cyclist_vs_vehicle_and_pedestrian(mock_model: MockModel, generator: SyntheticSceneGenerator):
    """Verify that elongated narrow human-height clusters are classified as CYCLIST."""
    rng = np.random.RandomState(42)
    # Cyclist: length 1.7m, width 0.6m, height 1.5m at X=10m, Y=2m
    cyc_pts = generator._generate_box_cluster(rng, (10.0, 2.0, 0.85), (1.7, 0.6, 1.5), 70, 0.78)
    frame = PointCloudFrame(points=cyc_pts, timestamp=0.0, frame_id="lidar_top", source=FrameSource.MOCK)

    sem_pts = mock_model.predict(frame)
    classes = [p.semantic_class for p in sem_pts]
    assert "CYCLIST" in classes, f"Cyclist was not recognized as CYCLIST, got: {set(classes)}"
    assert classes.count("CYCLIST") >= len(classes) * 0.8


def test_truck_vs_vehicle_distinction(mock_model: MockModel, generator: SyntheticSceneGenerator):
    """Verify large commercial vehicle vs passenger car distinction."""
    rng = np.random.RandomState(42)
    # Car: 4.2m x 1.8m x 1.5m
    car_pts = generator._generate_box_cluster(rng, (15.0, -2.0, 0.75), (4.2, 1.8, 1.5), 120, 0.85)
    # Truck: 7.5m x 2.5m x 3.0m
    truck_pts = generator._generate_box_cluster(rng, (25.0, 4.0, 1.5), (7.5, 2.5, 3.0), 350, 0.90)

    f_car = PointCloudFrame(points=car_pts, timestamp=0.0, frame_id="lidar_top", source=FrameSource.MOCK)
    f_truck = PointCloudFrame(points=truck_pts, timestamp=0.0, frame_id="lidar_top", source=FrameSource.MOCK)

    sem_car = mock_model.predict(f_car)
    sem_truck = mock_model.predict(f_truck)

    car_classes = [p.semantic_class for p in sem_car]
    truck_classes = [p.semantic_class for p in sem_truck]

    assert car_classes.count("VEHICLE") > len(car_classes) * 0.8
    assert truck_classes.count("TRUCK") > len(truck_classes) * 0.8


def test_curb_and_pothole_detection(mock_model: MockModel):
    """Verify that elevated curbs and negative road depressions (potholes) are correctly classified."""
    # Synthetic scan: 100 flat road points, 20 curb points (+0.12m), 20 pothole points (-0.15m)
    road_pts = np.column_stack([
        np.linspace(2, 20, 100), np.zeros(100), np.random.normal(0, 0.01, 100), np.full(100, 0.25)
    ])
    curb_pts = np.column_stack([
        np.linspace(2, 20, 20), np.full(20, 4.8), np.full(20, 0.12), np.full(20, 0.45)
    ])
    pothole_pts = np.column_stack([
        np.linspace(8, 10, 20), np.zeros(20), np.full(20, -0.15), np.full(20, 0.20)
    ])

    all_pts = np.vstack([road_pts, curb_pts, pothole_pts]).astype(np.float32)
    frame = PointCloudFrame(points=all_pts, timestamp=0.0, frame_id="lidar_top", source=FrameSource.MOCK)

    sem_pts = mock_model.predict(frame)
    classes = [p.semantic_class for p in sem_pts]

    assert "CURB" in classes, "Curb points were not detected"
    assert "POTHOLE" in classes, "Pothole points were not detected"
    assert classes.count("CURB") >= 15
    assert classes.count("POTHOLE") >= 15


# ==============================================================================
# 2. Range-Adaptive Pedestrian Detection Tests
# ==============================================================================

@pytest.mark.parametrize("range_x", [10.0, 25.0, 40.0])
def test_pedestrian_at_various_ranges(mock_model: MockModel, generator: SyntheticSceneGenerator, range_x: float):
    """Verify pedestrian classification remains high-confidence across near, mid, and far ranges."""
    frames, gt = generator.generate("pedestrian_at_distance", num_frames=1, initial_distance=range_x)
    sem_pts = mock_model.predict(frames[0])

    ped_pts = [p for p in sem_pts if p.semantic_class == "PEDESTRIAN"]
    assert len(ped_pts) > 0, f"Pedestrian was not detected at range {range_x}m"
    # Verify confidence remains high even at distance
    mean_conf = float(np.mean([p.confidence for p in ped_pts]))
    assert mean_conf >= 0.80, f"Pedestrian confidence degraded too low at {range_x}m: {mean_conf}"


def test_low_confidence_obstacle_remains_uncertain(mock_model: MockModel, generator: SyntheticSceneGenerator):
    """Verify that sparse ambiguous clusters produce UNKNOWN_OBSTACLE with calibrated low confidence."""
    frames, _ = generator.generate("low_confidence_obstacle", num_frames=1)
    sem_pts = mock_model.predict(frames[0])

    obs_pts = [p for p in sem_pts if p.semantic_class != "ROAD" and p.semantic_class != "CURB"]
    assert len(obs_pts) > 0, "No obstacle points detected in low_confidence_obstacle"
    for p in obs_pts:
        assert p.semantic_class == "UNKNOWN_OBSTACLE"
        assert p.confidence <= 0.60, f"Low confidence obstacle had unexpectedly high confidence: {p.confidence}"


# ==============================================================================
# 3. Object Extractor Cyclist and Bias-Free Tie-Breaking Tests
# ==============================================================================

def test_surround_360_cyclist_extraction(mock_model: MockModel, extractor: ObjectExtractor, generator: SyntheticSceneGenerator):
    """Verify that surround_360_city correctly extracts CYCLIST object without dropping or misclassifying."""
    frames, gt = generator.generate("surround_360_city", num_frames=1)
    sem_pts = mock_model.predict(frames[0])
    detected_objs = extractor.extract(sem_pts)

    det_classes = [o.semantic_class for o in detected_objs]
    assert "CYCLIST" in det_classes, f"CYCLIST was not extracted from surround_360_city, got: {det_classes}"

    # Verify cyclist object position matches ground truth cyclist position (~ [5.0, 6.5, 0.85])
    cyc_obj = next(o for o in detected_objs if o.semantic_class == "CYCLIST")
    np.testing.assert_allclose(cyc_obj.position[:2], [5.0, 6.5], atol=0.6)
    assert cyc_obj.dynamic is True


def test_extractor_unbiased_tie_breaking(extractor: ObjectExtractor):
    """Verify tie-breaking uses confidence weights and geometric priors rather than fixed class bias."""
    # Test A: Ambiguous cluster with pedestrian geometry (0.6m x 0.6m x 1.7m)
    # Equal point counts: 5 PEDESTRIAN (0.85) vs 5 VEHICLE (0.85)
    ped_cluster = [
        SemanticPoint((10.0 + i * 0.1, 0.0, 1.0 + i * 0.1), "PEDESTRIAN", 0.85, 0.0)
        for i in range(5)
    ] + [
        SemanticPoint((10.0 + i * 0.1, 0.0, 1.0 + i * 0.1), "VEHICLE", 0.85, 0.0)
        for i in range(5)
    ]

    det_ped = extractor.extract(ped_cluster)
    assert len(det_ped) == 1
    # Bounding box is (0.4m x 0.0m x 0.4m), matching pedestrian scale, NOT 4.2m car scale
    assert det_ped[0].semantic_class == "PEDESTRIAN", "Tie-breaking failed to choose geometrically consistent PEDESTRIAN"

    # Test B: Ambiguous cluster with vehicle geometry (2.8m x 0.6m x 0.6m)
    # Equal point counts: 5 PEDESTRIAN (0.85) vs 5 VEHICLE (0.85)
    veh_cluster = [
        SemanticPoint((20.0 + i * 0.7, 0.0, 0.5), "PEDESTRIAN", 0.85, 0.0)
        for i in range(5)
    ] + [
        SemanticPoint((20.0 + i * 0.7, 0.6, 1.1), "VEHICLE", 0.85, 0.0)
        for i in range(5)
    ]

    det_veh = extractor.extract(veh_cluster)
    assert len(det_veh) == 1
    assert det_veh[0].semantic_class == "VEHICLE", "Tie-breaking failed to choose geometrically consistent VEHICLE"
