"""Unit tests for ObjectExtractor."""

import numpy as np
import pytest

from fovea_lidar.contracts.objects import DetectedObject
from fovea_lidar.contracts.semantic import SemanticPoint
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.objects.extraction import ObjectExtractor
from fovea_lidar.perception.mock_model import MockModel


# ==============================================================================
# Synthetic Scenario End-to-End Tests
# ==============================================================================

def test_pedestrian_at_distance_extraction():
    """Verify single pedestrian extraction matching ground truth from pedestrian_at_distance."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("pedestrian_at_distance", num_frames=1)
    gt_objs = gt["frames"][0]["objects"]

    model = MockModel()
    sem_points = model.predict(frames[0])

    extractor = ObjectExtractor.load()
    detected = extractor.extract(sem_points)

    # 1. Correct number of objects matching ground truth
    assert len(detected) == len(gt_objs) == 1

    det = detected[0]
    gt_obj = gt_objs[0]

    # 2. Extracted class matches ground truth
    assert det.semantic_class == gt_obj["semantic_class"] == "PEDESTRIAN"

    # 3. Dynamic initial classification
    assert det.dynamic is True

    # 4. Position within close proximity of ground truth position
    np.testing.assert_allclose(det.position, gt_obj["position"], atol=0.5)

    # 5. Bounding box contains all member source points
    assert len(extractor.last_extracted_clusters) == 1
    member_pts = extractor.last_extracted_clusters[0]
    min_x, min_y, min_z, max_x, max_y, max_z = det.bounding_region
    for p in member_pts:
        x, y, z = p.coordinates
        assert min_x <= x <= max_x
        assert min_y <= y <= max_y
        assert min_z <= z <= max_z


def test_vehicle_crossing_extraction():
    """Verify single vehicle extraction matching ground truth from vehicle_crossing."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("vehicle_crossing", num_frames=1)
    gt_objs = gt["frames"][0]["objects"]

    model = MockModel()
    sem_points = model.predict(frames[0])

    extractor = ObjectExtractor.load()
    detected = extractor.extract(sem_points)

    # 1. Correct number of objects matching ground truth
    assert len(detected) == len(gt_objs) == 1

    det = detected[0]
    gt_obj = gt_objs[0]

    # 2. Extracted class matches ground truth
    assert det.semantic_class == gt_obj["semantic_class"] == "VEHICLE"

    # 3. Dynamic initial classification
    assert det.dynamic is True

    # 4. Position close to ground truth
    np.testing.assert_allclose(det.position, gt_obj["position"], atol=0.5)

    # 5. Bounding box contains all member points
    assert len(extractor.last_extracted_clusters) == 1
    member_pts = extractor.last_extracted_clusters[0]
    min_x, min_y, min_z, max_x, max_y, max_z = det.bounding_region
    for p in member_pts:
        x, y, z = p.coordinates
        assert min_x <= x <= max_x
        assert min_y <= y <= max_y
        assert min_z <= z <= max_z


def test_multi_object_dense_extraction():
    """Verify multi-object extraction and spatial separation on multi_object_dense."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("multi_object_dense", num_frames=1)
    gt_objs = gt["frames"][0]["objects"]

    model = MockModel()
    sem_points = model.predict(frames[0])

    extractor = ObjectExtractor.load()
    detected = extractor.extract(sem_points)

    # 1. Correct number of objects matching ground truth
    assert len(detected) == len(gt_objs) == 4

    # 2. Classes match ground truth set (2 PEDESTRIANS, 1 VEHICLE, 1 TRUCK)
    gt_classes = sorted([o["semantic_class"] for o in gt_objs])
    det_classes = sorted([d.semantic_class for d in detected])
    assert det_classes == gt_classes == ["PEDESTRIAN", "PEDESTRIAN", "TRUCK", "VEHICLE"]

    # 3. Objects positioned close together are correctly separated (not merged)
    positions = [np.array(d.position) for d in detected]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            assert dist > 4.0, f"Objects {i} and {j} were incorrectly merged: dist={dist}"

    # 4. Bounding box contains all member source points for each object
    assert len(extractor.last_extracted_clusters) == 4
    for det, member_pts in zip(detected, extractor.last_extracted_clusters):
        min_x, min_y, min_z, max_x, max_y, max_z = det.bounding_region
        for p in member_pts:
            x, y, z = p.coordinates
            assert min_x <= x <= max_x
            assert min_y <= y <= max_y
            assert min_z <= z <= max_z


# ==============================================================================
# Noise, Degenerate Cluster, and Attribute Tests
# ==============================================================================

def test_noise_cluster_discarded():
    """Verify that clusters with fewer than min_cluster_points are discarded as noise."""
    extractor = ObjectExtractor(min_cluster_points=10)

    # Cluster of 4 pedestrian points (below threshold of 10)
    noise_points = [
        SemanticPoint((10.0 + i * 0.1, 0.0, 1.0), "PEDESTRIAN", 0.9, 0.0)
        for i in range(4)
    ]
    detected = extractor.extract(noise_points)
    assert len(detected) == 0

    # Cluster of 10 pedestrian points (meets threshold of 10)
    valid_points = [
        SemanticPoint((10.0 + i * 0.05, 0.0, 1.0), "PEDESTRIAN", 0.9, 0.0)
        for i in range(10)
    ]
    detected_valid = extractor.extract(valid_points)
    assert len(detected_valid) == 1
    assert detected_valid[0].semantic_class == "PEDESTRIAN"


def test_static_vs_dynamic_initial_guess():
    """Verify initial dynamic classification defaults based on semantic class."""
    extractor = ObjectExtractor(min_cluster_points=3)

    # Dynamic objects
    veh_pts = [SemanticPoint((5.0 + i * 0.1, 0.0, 0.5), "VEHICLE", 0.9, 0.0) for i in range(5)]
    ped_pts = [SemanticPoint((15.0 + i * 0.1, 0.0, 0.5), "PEDESTRIAN", 0.9, 0.0) for i in range(5)]
    truck_pts = [SemanticPoint((25.0 + i * 0.1, 0.0, 0.5), "TRUCK", 0.9, 0.0) for i in range(5)]

    # Static objects
    wall_pts = [SemanticPoint((35.0 + i * 0.1, 0.0, 0.5), "WALL", 0.9, 0.0) for i in range(5)]
    pole_pts = [SemanticPoint((45.0 + i * 0.1, 0.0, 0.5), "POLE", 0.9, 0.0) for i in range(5)]

    all_pts = veh_pts + ped_pts + truck_pts + wall_pts + pole_pts
    detected = extractor.extract(all_pts)
    assert len(detected) == 5

    cls_to_dyn = {d.semantic_class: d.dynamic for d in detected}
    assert cls_to_dyn["VEHICLE"] is True
    assert cls_to_dyn["PEDESTRIAN"] is True
    assert cls_to_dyn["TRUCK"] is True
    assert cls_to_dyn["WALL"] is False
    assert cls_to_dyn["POLE"] is False


def test_tie_breaking_dominant_class():
    """Verify majority voting and confidence-based tie breaking."""
    extractor = ObjectExtractor(min_cluster_points=4)

    # 4 points: 2 VEHICLE (conf=0.95), 2 PEDESTRIAN (conf=0.75)
    # Tied count (2 vs 2), VEHICLE has higher confidence -> VEHICLE wins
    points = [
        SemanticPoint((1.0, 0.0, 0.0), "VEHICLE", 0.95, 0.0),
        SemanticPoint((1.1, 0.0, 0.0), "VEHICLE", 0.95, 0.0),
        SemanticPoint((1.0, 0.1, 0.0), "PEDESTRIAN", 0.75, 0.0),
        SemanticPoint((1.1, 0.1, 0.0), "PEDESTRIAN", 0.75, 0.0),
    ]
    detected = extractor.extract(points)
    assert len(detected) == 1
    assert detected[0].semantic_class == "VEHICLE"
    assert detected[0].confidence == pytest.approx(0.85)


def test_empty_and_terrain_only_inputs():
    """Verify that empty inputs or terrain-only points produce no detected objects."""
    extractor = ObjectExtractor()

    # Empty list
    assert extractor.extract([]) == []

    # Road and sidewalk points only
    terrain_pts = [
        SemanticPoint((float(i), float(j), 0.0), "ROAD", 0.9, 0.0)
        for i in range(5)
        for j in range(5)
    ] + [
        SemanticPoint((float(i), 10.0, 0.1), "SIDEWALK", 0.85, 0.0)
        for i in range(5)
    ]
    assert extractor.extract(terrain_pts) == []


def test_provisional_object_id_uniqueness():
    """Verify provisional IDs are generated and unique per detection."""
    extractor = ObjectExtractor(min_cluster_points=3)
    pts1 = [SemanticPoint((1.0 + i * 0.1, 0.0, 0.0), "PEDESTRIAN", 0.9, 0.0) for i in range(4)]
    pts2 = [SemanticPoint((10.0 + i * 0.1, 0.0, 0.0), "PEDESTRIAN", 0.9, 0.0) for i in range(4)]

    detected = extractor.extract(pts1 + pts2)
    assert len(detected) == 2
    assert detected[0].object_id != detected[1].object_id
    assert detected[0].object_id.startswith("det_")
    assert detected[1].object_id.startswith("det_")
