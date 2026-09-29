"""Unit tests for Fovea-LiDAR data contracts and serialization."""

import math
import numpy as np
import pytest

from fovea_lidar.contracts import (
    AdaptiveCell,
    AdaptiveMap,
    DetectedObject,
    FrameSource,
    InvalidSemanticClassError,
    PointCloudFrame,
    RiskLevel,
    RiskRegion,
    SemanticClassRegistry,
    SemanticPoint,
    TrackedObject,
    TraversabilityMap,
    register_class,
    validate_class,
)


# ==============================================================================
# PointCloudFrame Tests
# ==============================================================================

def test_point_cloud_frame_roundtrip():
    """Test construction, validation, and serialization round-trip of PointCloudFrame."""
    points = np.array(
        [
            [1.0, 2.0, 3.0, 0.8],
            [4.0, 5.0, 6.0, 0.4],
            [7.0, 8.0, 9.0, 0.95],
        ],
        dtype=np.float32,
    )
    frame = PointCloudFrame(
        points=points,
        timestamp=123.456,
        frame_id="lidar_sensor_link",
        source=FrameSource.LIVE,
        metadata={"seq": 42, "temperature_c": 28.5},
    )

    d = frame.to_dict()
    assert d["source"] == "LIVE"
    assert d["frame_id"] == "lidar_sensor_link"
    assert len(d["points"]) == 3

    restored = PointCloudFrame.from_dict(d)
    assert restored == frame
    assert restored.source == FrameSource.LIVE
    assert restored.metadata["seq"] == 42
    assert np.allclose(restored.points, points)


def test_point_cloud_frame_invalid_shape():
    """Test that invalid point array shapes raise ValueError."""
    with pytest.raises(ValueError):
        PointCloudFrame(
            points=np.array([[1.0, 2.0]]),  # Nx2 is invalid (must be Nx3 or Nx4)
            timestamp=0.0,
            frame_id="test",
        )


# ==============================================================================
# SemanticPoint Tests
# ==============================================================================

def test_semantic_point_roundtrip():
    """Test valid SemanticPoint construction and serialization round-trip."""
    point = SemanticPoint(
        coordinates=(1.5, -2.5, 0.75),
        semantic_class="VEHICLE",
        confidence=0.92,
        timestamp=10.0,
    )

    d = point.to_dict()
    assert d["semantic_class"] == "VEHICLE"
    assert d["confidence"] == 0.92
    assert d["coordinates"] == [1.5, -2.5, 0.75]

    restored = SemanticPoint.from_dict(d)
    assert restored == point
    assert restored.coordinates == (1.5, -2.5, 0.75)
    assert restored.semantic_class == "VEHICLE"


def test_semantic_point_invalid_class_raises():
    """Test that constructing a SemanticPoint with an unregistered class raises InvalidSemanticClassError."""
    with pytest.raises(InvalidSemanticClassError) as exc_info:
        SemanticPoint(
            coordinates=(0.0, 0.0, 0.0),
            semantic_class="NON_EXISTENT_FLYING_OBJECT",
            confidence=0.5,
            timestamp=1.0,
        )
    assert "invalid semantic class" in str(exc_info.value).lower()


def test_semantic_point_invalid_confidence():
    """Test that confidence outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        SemanticPoint(
            coordinates=(0.0, 0.0, 0.0),
            semantic_class="ROAD",
            confidence=1.5,
            timestamp=1.0,
        )

    with pytest.raises(ValueError):
        SemanticPoint(
            coordinates=(0.0, 0.0, 0.0),
            semantic_class="ROAD",
            confidence=-0.1,
            timestamp=1.0,
        )


# ==============================================================================
# SemanticClassRegistry Tests
# ==============================================================================

def test_semantic_class_registry():
    """Test dynamic registration and validation behavior of the registry."""
    reg = SemanticClassRegistry(initial_classes=["TERRAIN", "OBSTACLE"])
    assert reg.is_valid("TERRAIN")
    assert reg.is_valid("terrain")  # case-insensitive check
    assert not reg.is_valid("ALIEN")

    reg.register("ALIEN")
    assert reg.is_valid("ALIEN")
    assert reg.validate("alien") == "ALIEN"

    reg.unregister("ALIEN")
    assert not reg.is_valid("ALIEN")


# ==============================================================================
# DetectedObject Tests
# ==============================================================================

def test_detected_object_roundtrip():
    """Test construction and serialization of DetectedObject."""
    obj = DetectedObject(
        object_id=101,
        semantic_class="PEDESTRIAN",
        position=(10.0, 5.0, 0.5),
        bounding_region=(9.5, 4.5, 0.0, 10.5, 5.5, 1.8),
        confidence=0.88,
        velocity=(0.0, 1.2, 0.0),
        dynamic=True,
        timestamp=5.0,
    )

    d = obj.to_dict()
    assert d["object_id"] == 101
    assert d["dynamic"] is True
    assert d["velocity"] == [0.0, 1.2, 0.0]

    restored = DetectedObject.from_dict(d)
    assert restored == obj
    assert restored.position == (10.0, 5.0, 0.5)
    assert restored.velocity == (0.0, 1.2, 0.0)


# ==============================================================================
# TrackedObject Tests
# ==============================================================================

def test_tracked_object_roundtrip_and_behavior():
    """Test TrackedObject velocity requirement, history bounding, and confidence decay."""
    tracked = TrackedObject(
        object_id="obj_42",
        track_id="trk_7",
        semantic_class="VEHICLE",
        position=(20.0, 0.0, 1.0),
        bounding_region=(18.0, -1.0, 0.0, 22.0, 1.0, 2.0),
        confidence=0.95,
        velocity=(15.0, 0.0, 0.0),  # Required velocity
        direction=(1.0, 0.0, 0.0),
        position_history=[(15.0, 0.0, 1.0), (18.0, 0.0, 1.0)],
        last_seen=10.0,
        track_confidence=1.0,
        max_history_len=3,
    )

    # Verify history bounding
    tracked.add_position((21.0, 0.0, 1.0))
    tracked.add_position((22.0, 0.0, 1.0))
    assert len(tracked.position_history) == 3
    assert tracked.position_history[-1] == (22.0, 0.0, 1.0)

    # Test confidence decay
    decayed = tracked.decay_confidence(current_time=12.0, decay_rate=0.5)
    assert 0.0 < decayed < 1.0
    assert math.isclose(decayed, 1.0 * math.exp(-0.5 * 2.0), rel_tol=1e-4)

    # Verify serialization round-trip
    d = tracked.to_dict()
    restored = TrackedObject.from_dict(d)
    assert restored.track_id == "trk_7"
    assert restored.velocity == (15.0, 0.0, 0.0)
    assert restored.position_history == tracked.position_history
    assert math.isclose(restored.track_confidence, tracked.track_confidence)


def test_tracked_object_requires_velocity():
    """Test that constructing a TrackedObject with None velocity raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        TrackedObject(
            object_id="fail_obj",
            track_id="fail_track",
            semantic_class="VEHICLE",
            position=(0.0, 0.0, 0.0),
            bounding_region=(-1.0, -1.0, 0.0, 1.0, 1.0, 1.0),
            confidence=0.8,
            velocity=None,  # Forbidden
        )
    assert "velocity is required" in str(exc_info.value).lower()


# ==============================================================================
# AdaptiveCell Tests
# ==============================================================================

def test_adaptive_cell_roundtrip():
    """Test AdaptiveCell initialization, slots representation, and serialization."""
    cell = AdaptiveCell(
        spatial_bounds=(0.0, 0.0, 0.5, 0.5),
        resolution=0.5,
        elevation=1.25,
        elevation_variance=0.04,
        roughness=0.1,
        occupancy_probability=0.85,
        semantic_class="ROAD",
        semantic_confidence=0.9,
        dynamic_state="STATIC",
        velocity=None,
        risk_score=0.15,
        traversability_score=0.88,
        last_updated=100.0,
    )

    d = cell.to_dict()
    assert d["resolution"] == 0.5
    assert d["traversability_score"] == 0.88

    restored = AdaptiveCell.from_dict(d)
    assert restored == cell
    assert restored.spatial_bounds == (0.0, 0.0, 0.5, 0.5)
    assert restored.resolution == 0.5


def test_adaptive_cell_rejects_negative_and_zero_resolution():
    """Verify that AdaptiveCell strictly rejects negative and zero resolutions."""
    with pytest.raises(ValueError) as exc_neg:
        AdaptiveCell(
            spatial_bounds=(0.0, 0.0, 1.0, 1.0),
            resolution=-0.2,
        )
    assert "resolution must be strictly positive" in str(exc_neg.value).lower()

    with pytest.raises(ValueError) as exc_zero:
        AdaptiveCell(
            spatial_bounds=(0.0, 0.0, 1.0, 1.0),
            resolution=0.0,
        )
    assert "resolution must be strictly positive" in str(exc_zero.value).lower()


# ==============================================================================
# AdaptiveMap and TraversabilityMap Tests
# ==============================================================================

def test_adaptive_map_and_traversability_map_roundtrip():
    """Test AdaptiveMap and wrapped TraversabilityMap serialization."""
    c1 = AdaptiveCell(
        spatial_bounds=(0.0, 0.0, 1.0, 1.0),
        resolution=1.0,
        traversability_score=0.9,
    )
    c2 = AdaptiveCell(
        spatial_bounds=(1.0, 0.0, 2.0, 1.0),
        resolution=1.0,
        traversability_score=0.2,
    )

    adaptive_map = AdaptiveMap(
        cells={"cell_0": c1, "cell_1": c2},
        resolution_zones=[{"name": "fovea", "target_resolution": 0.05}],
        fovea_overlays=[{"center": [0.0, 0.0], "radius": 5.0}],
        metadata={"frame_count": 12, "last_update_time": 105.2},
    )

    # Verify AdaptiveMap serialization
    map_dict = adaptive_map.to_dict()
    restored_map = AdaptiveMap.from_dict(map_dict)
    assert len(restored_map.cells) == 2
    assert restored_map.cells["cell_0"].traversability_score == 0.9
    assert restored_map.cells["cell_1"].traversability_score == 0.2
    assert restored_map.metadata["frame_count"] == 12

    # Verify TraversabilityMap wrapping
    trav_map = TraversabilityMap(adaptive_map=adaptive_map, default_threshold=0.5)
    assert trav_map.is_traversable("cell_0") is True
    assert trav_map.is_traversable("cell_1") is False
    assert trav_map.is_traversable("unknown_cell") is False

    trav_dict = trav_map.to_dict()
    restored_trav = TraversabilityMap.from_dict(trav_dict)
    assert restored_trav.default_threshold == 0.5
    assert restored_trav.is_traversable("cell_0") is True
    assert restored_trav.is_traversable("cell_1") is False


# ==============================================================================
# RiskRegion Tests
# ==============================================================================

def test_risk_region_roundtrip():
    """Test RiskRegion construction, validation, and serialization."""
    risk = RiskRegion(
        region_bounds=(-5.0, 0.0, 0.0, 5.0),
        risk_score=0.75,
        risk_level=RiskLevel.HIGH,
        associated_object_id="trk_7",
        timestamp=50.0,
    )

    d = risk.to_dict()
    assert d["risk_level"] == "HIGH"
    assert d["risk_score"] == 0.75
    assert d["associated_object_id"] == "trk_7"

    restored = RiskRegion.from_dict(d)
    assert restored == risk
    assert restored.risk_level == RiskLevel.HIGH


def test_risk_region_invalid_score():
    """Test that risk score outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        RiskRegion(
            region_bounds=(0.0, 0.0, 1.0, 1.0),
            risk_score=1.5,
            risk_level=RiskLevel.LOW,
        )
