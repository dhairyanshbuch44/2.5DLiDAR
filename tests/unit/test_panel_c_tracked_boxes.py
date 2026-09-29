"""Unit tests for Panel C tracked object bounding box rendering, stability, and velocity state."""

import numpy as np
import pytest

from fovea_lidar.contracts.objects import DetectedObject
from fovea_lidar.contracts.risk import RiskLevel, RiskRegion
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.pipeline import FoveaPipeline
from fovea_lidar.tracking.tracker import MultiObjectTracker, TrackState
from dashboard.server import serialize_frame_message


def test_tracked_object_box_position_kalman_synchronization():
    """Verify box position and bounds come directly from the Kalman state, even during coasting."""
    tracker = MultiObjectTracker(gating_distance=3.0, max_missed_frames=3)

    # Frame 0: Initial detection with 2m x 2m x 1m box
    det0 = DetectedObject(
        object_id="det_1",
        semantic_class="VEHICLE",
        position=(10.0, 5.0, 0.5),
        bounding_region=(9.0, 4.0, 0.0, 11.0, 6.0, 1.0),
        confidence=0.95,
        dynamic=True,
    )
    tracked_0 = tracker.update([det0], timestamp=0.0)
    assert len(tracked_0) == 1
    t0 = tracked_0[0]
    assert t0.is_coasting is False
    assert t0.position == pytest.approx((10.0, 5.0, 0.5), abs=1e-3)
    # Center of bounding box matches Kalman position
    b0 = t0.bounding_region
    cx0 = (b0[0] + b0[3]) / 2.0
    cy0 = (b0[1] + b0[4]) / 2.0
    cz0 = (b0[2] + b0[5]) / 2.0
    assert cx0 == pytest.approx(t0.position[0], abs=1e-3)
    assert cy0 == pytest.approx(t0.position[1], abs=1e-3)
    assert cz0 == pytest.approx(t0.position[2], abs=1e-3)

    # Frame 1: Detection moves forward along X at 10 m/s (dx = 1.0m in dt = 0.1s)
    det1 = DetectedObject(
        object_id="det_1",
        semantic_class="VEHICLE",
        position=(11.0, 5.0, 0.5),
        bounding_region=(10.0, 4.0, 0.0, 12.0, 6.0, 1.0),
        confidence=0.95,
        dynamic=True,
    )
    tracked_1 = tracker.update([det1], timestamp=0.1)
    assert len(tracked_1) == 1
    t1 = tracked_1[0]
    assert t1.is_coasting is False
    assert t1.track_id == t0.track_id

    # Frame 2: Object is temporarily occluded / undetected (coasting frame)
    tracked_2 = tracker.update([], timestamp=0.2)
    assert len(tracked_2) == 1
    t2 = tracked_2[0]
    assert t2.is_coasting is True
    assert t2.track_id == t0.track_id

    # Crucial Requirement: Box position/size must come directly from current frame's
    # TrackedObject state, translating with Kalman filter position, NOT frozen at old bounds!
    b2 = t2.bounding_region
    cx2 = (b2[0] + b2[3]) / 2.0
    cy2 = (b2[1] + b2[4]) / 2.0
    cz2 = (b2[2] + b2[5]) / 2.0

    # Center must strictly match current Kalman predicted position
    assert cx2 == pytest.approx(t2.position[0], abs=1e-3)
    assert cy2 == pytest.approx(t2.position[1], abs=1e-3)
    assert cz2 == pytest.approx(t2.position[2], abs=1e-3)
    # The predicted position must have advanced past 11.0m under constant velocity dynamics
    assert cx2 > 11.0, f"Predicted box center {cx2} should have advanced forward"
    # Half-extents must match physical object dimensions (2m x 2m x 1m)
    assert (b2[3] - b2[0]) == pytest.approx(2.0, abs=1e-3)
    assert (b2[4] - b2[1]) == pytest.approx(2.0, abs=1e-3)
    assert (b2[5] - b2[2]) == pytest.approx(1.0, abs=1e-3)


def test_velocity_units_and_serialization():
    """Verify velocity vectors match Kalman internal m/s state without arbitrary scaling."""
    tracker = MultiObjectTracker()
    # Feed detection moving at 5.0 m/s
    det0 = DetectedObject(
        object_id="det_1",
        semantic_class="VEHICLE",
        position=(10.0, 0.0, 0.0),
        bounding_region=(9.0, -1.0, -0.5, 11.0, 1.0, 0.5),
        confidence=0.9,
    )
    det1 = DetectedObject(
        object_id="det_1",
        semantic_class="VEHICLE",
        position=(10.5, 0.0, 0.0),  # 0.5m in 0.1s -> 5.0 m/s
        bounding_region=(9.5, -1.0, -0.5, 11.5, 1.0, 0.5),
        confidence=0.9,
    )
    det2 = DetectedObject(
        object_id="det_1",
        semantic_class="VEHICLE",
        position=(11.0, 0.0, 0.0),  # another 0.5m in 0.1s
        bounding_region=(10.0, -1.0, -0.5, 12.0, 1.0, 0.5),
        confidence=0.9,
    )
    tracker.update([det0], timestamp=0.0)
    tracker.update([det1], timestamp=0.1)
    tracked = tracker.update([det2], timestamp=0.2)

    assert len(tracked) == 1
    t = tracked[0]
    vx, vy, vz = t.velocity
    # Velocity units must be m/s and converging towards 5.0 m/s
    assert vx > 3.0, f"Expected positive velocity along X in m/s, got {vx}"

    # Verify serialization in dashboard server
    from unittest.mock import MagicMock
    mock_result = MagicMock()
    mock_result.raw_frame.points = np.zeros((10, 4), dtype=np.float32)
    mock_result.cleaned_frame.timestamp = 0.1
    mock_result.timestamp = 0.1
    mock_result.semantic_points = []
    mock_result.tracked_objects = tracked
    mock_result.active_fovea_regions = []
    mock_result.adaptive_map.cells = {}
    mock_result.traversability_map = None
    mock_result.metrics = {}
    mock_result.raw_frame.metadata = {}
    mock_result.risk_regions = []

    payload = serialize_frame_message(mock_result, scenario="test", total_frames=10)
    trk_payload = payload["tracked_objects"][0]

    assert "speed_ms" in trk_payload
    assert "is_coasting" in trk_payload
    assert trk_payload["is_coasting"] is False
    assert trk_payload["speed_ms"] == pytest.approx(float(np.linalg.norm(t.velocity)), abs=0.01)


def test_track_id_stability_dense_close_proximity_and_classes():
    """Verify Hungarian matching and class compatibility prevent ID swapping when objects are close."""
    tracker = MultiObjectTracker(gating_distance=2.5)

    # Frame 0: Pedestrian and Vehicle near each other
    ped0 = DetectedObject(
        object_id="p1",
        semantic_class="PEDESTRIAN",
        position=(15.0, 2.0, 0.0),
        bounding_region=(14.7, 1.7, -0.8, 15.3, 2.3, 0.8),
        confidence=0.9,
    )
    veh0 = DetectedObject(
        object_id="v1",
        semantic_class="VEHICLE",
        position=(15.8, 2.6, 0.0),  # Only 1.0m away from pedestrian!
        bounding_region=(14.8, 1.6, -0.7, 16.8, 3.6, 0.7),
        confidence=0.95,
    )
    tracks0 = tracker.update([ped0, veh0], timestamp=0.0)
    assert len(tracks0) == 2
    ped_track_id = [t.track_id for t in tracks0 if t.semantic_class == "PEDESTRIAN"][0]
    veh_track_id = [t.track_id for t in tracks0 if t.semantic_class == "VEHICLE"][0]
    assert ped_track_id != veh_track_id

    # Frame 1: Vehicle moves faster, pedestrian moves slower
    ped1 = DetectedObject(
        object_id="p1",
        semantic_class="PEDESTRIAN",
        position=(15.1, 2.0, 0.0),
        bounding_region=(14.8, 1.7, -0.8, 15.4, 2.3, 0.8),
        confidence=0.9,
    )
    veh1 = DetectedObject(
        object_id="v1",
        semantic_class="VEHICLE",
        position=(16.8, 2.6, 0.0),
        bounding_region=(15.8, 1.6, -0.7, 17.8, 3.6, 0.7),
        confidence=0.95,
    )
    tracks1 = tracker.update([ped1, veh1], timestamp=0.1)
    assert len(tracks1) == 2

    # Zero ID swaps: pedestrian keeps ped_track_id, vehicle keeps veh_track_id
    t_ped = [t for t in tracks1 if t.semantic_class == "PEDESTRIAN"][0]
    t_veh = [t for t in tracks1 if t.semantic_class == "VEHICLE"][0]
    assert t_ped.track_id == ped_track_id
    assert t_veh.track_id == veh_track_id


def test_track_reacquisition_after_occlusion():
    """Verify that an object occluded for multiple frames re-acquires its existing track ID."""
    tracker = MultiObjectTracker(gating_distance=2.5, max_missed_frames=4)

    # Frame 0 to 2: Object moving at 5 m/s (0.5m per 0.1s)
    track_id = None
    for f in range(3):
        x = 10.0 + f * 0.5
        det = DetectedObject(
            object_id="veh",
            semantic_class="VEHICLE",
            position=(x, 0.0, 0.0),
            bounding_region=(x - 1.0, -1.0, -0.5, x + 1.0, 1.0, 0.5),
            confidence=0.95,
        )
        tracks = tracker.update([det], timestamp=f * 0.1)
        assert len(tracks) == 1
        track_id = tracks[0].track_id
        assert tracks[0].is_coasting is False

    # Frames 3 and 4: Object occluded (coasting for 2 frames)
    for f in (3, 4):
        tracks = tracker.update([], timestamp=f * 0.1)
        assert len(tracks) == 1
        assert tracks[0].track_id == track_id
        assert tracks[0].is_coasting is True

    # Frame 5: Object re-appears at x = 12.6m (slightly deviated from pure constant velocity 12.5m)
    det_reappear = DetectedObject(
        object_id="veh",
        semantic_class="VEHICLE",
        position=(12.6, 0.1, 0.0),
        bounding_region=(11.6, -0.9, -0.5, 13.6, 1.1, 0.5),
        confidence=0.95,
    )
    tracks = tracker.update([det_reappear], timestamp=0.5)
    assert len(tracks) == 1
    # Track re-acquired: SAME track_id, no new track spawned, coasting is cleared
    assert tracks[0].track_id == track_id
    assert tracks[0].is_coasting is False


def test_pipeline_scenarios_panel_c_boxes_and_risk_validation():
    """Validate Panel C bounding box contract across vehicle_crossing, multi_object_dense, and object_disappearing."""
    gen = SyntheticSceneGenerator(seed=42)

    # 1. vehicle_crossing: crossing vehicle maintains stable ID and computes valid risk levels
    vc_frames, _ = gen.generate("vehicle_crossing", num_frames=8)
    pipeline_vc = FoveaPipeline()
    seen_ids = set()
    for f in vc_frames:
        res = pipeline_vc.process_frame(f)
        assert len(res.tracked_objects) >= 1
        for trk in res.tracked_objects:
            seen_ids.add(trk.track_id)
            # Box center matches position
            b = trk.bounding_region
            cx = (b[0] + b[3]) / 2.0
            cy = (b[1] + b[4]) / 2.0
            cz = (b[2] + b[5]) / 2.0
            assert cx == pytest.approx(trk.position[0], abs=1e-3)
            assert cy == pytest.approx(trk.position[1], abs=1e-3)

        payload = serialize_frame_message(res, scenario="vehicle_crossing", total_frames=8)
        for t_dict in payload["tracked_objects"]:
            assert t_dict["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            assert t_dict["is_coasting"] is False
            assert t_dict["speed_ms"] >= 0.0
    # Exactly one track ID across all 8 frames
    assert len(seen_ids) == 1

    # 2. multi_object_dense: concurrent tracks remain distinct with zero ID swaps
    dense_frames, _ = gen.generate("multi_object_dense", num_frames=5)
    pipeline_dense = FoveaPipeline()
    dense_seen = set()
    initial_track_count = None
    for f in dense_frames:
        res = pipeline_dense.process_frame(f)
        assert len(res.tracked_objects) >= 3
        if initial_track_count is None:
            initial_track_count = len(res.tracked_objects)
        else:
            assert len(res.tracked_objects) == initial_track_count
        for trk in res.tracked_objects:
            dense_seen.add(trk.track_id)
        payload = serialize_frame_message(res, scenario="multi_object_dense", total_frames=5)
        assert len(payload["tracked_objects"]) == initial_track_count
    # Zero ID swaps: exactly initial_track_count track IDs across all frames
    assert len(dense_seen) == initial_track_count

    # 3. object_disappearing: verifies coasting state transition and visual distinction flag
    disappear_frames, _ = gen.generate("object_disappearing", num_frames=9, disappear_after_frame=5)
    pipeline_disappear = FoveaPipeline()
    coasting_states = []
    for f_idx, f in enumerate(disappear_frames):
        res = pipeline_disappear.process_frame(f)
        payload = serialize_frame_message(res, scenario="object_disappearing", total_frames=9)
        if len(res.tracked_objects) > 0:
            trk = res.tracked_objects[0]
            coasting_states.append((f_idx, trk.is_coasting, payload["tracked_objects"][0]["is_coasting"]))

    # Frames 0..4: actively observed (is_coasting == False)
    for f_idx, trk_coast, payload_coast in coasting_states[:5]:
        assert trk_coast is False
        assert payload_coast is False

    # Frames 5..7: coasting (is_coasting == True in TrackedObject AND WebSocket payload)
    coasting_frames = [c for c in coasting_states if c[0] in (5, 6, 7)]
    assert len(coasting_frames) == 3
    for f_idx, trk_coast, payload_coast in coasting_frames:
        assert trk_coast is True, f"Frame {f_idx} expected is_coasting True"
        assert payload_coast is True, f"Frame {f_idx} payload expected is_coasting True"

