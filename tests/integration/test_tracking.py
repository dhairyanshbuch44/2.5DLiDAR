"""Integration tests for MultiObjectTracker across synthetic scenarios."""

import numpy as np
import pytest

from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.objects.extraction import ObjectExtractor
from fovea_lidar.perception.mock_model import MockModel
from fovea_lidar.tracking.tracker import MultiObjectTracker


def test_pedestrian_at_distance_tracking_id_stability():
    """Verify that moving pedestrian maintains the exact SAME track_id across all frames."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("pedestrian_at_distance", num_frames=10)

    model = MockModel()
    extractor = ObjectExtractor.load()
    tracker = MultiObjectTracker.load()

    observed_track_ids = []

    for f_idx, frame in enumerate(frames):
        sem_points = model.predict(frame)
        detections = extractor.extract(sem_points)
        tracked = tracker.update(detections, timestamp=frame.timestamp)

        assert len(tracked) == 1, f"Frame {f_idx} expected 1 tracked object, got {len(tracked)}"
        observed_track_ids.append(tracked[0].track_id)

    print(f"\nObserved track_id sequence for pedestrian_at_distance: {observed_track_ids}")

    # Assert track ID stability: exactly 1 unique ID used for all 10 frames
    assert len(set(observed_track_ids)) == 1
    assert observed_track_ids[0] == "track_1"
    assert observed_track_ids == ["track_1"] * 10


def test_vehicle_crossing_tracking_id_stability():
    """Verify that fast crossing vehicle maintains stable track_id and estimated velocity."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("vehicle_crossing", num_frames=10)

    model = MockModel()
    extractor = ObjectExtractor.load()
    tracker = MultiObjectTracker.load()

    observed_track_ids = []
    velocities = []

    for f_idx, frame in enumerate(frames):
        sem_points = model.predict(frame)
        detections = extractor.extract(sem_points)
        tracked = tracker.update(detections, timestamp=frame.timestamp)

        assert len(tracked) == 1, f"Frame {f_idx} expected 1 tracked object, got {len(tracked)}"
        observed_track_ids.append(tracked[0].track_id)
        velocities.append(tracked[0].velocity)

    # Stable track ID across all 10 frames
    assert len(set(observed_track_ids)) == 1
    assert observed_track_ids == ["track_1"] * 10

    # Ground truth vehicle velocity is (0.0, 3.5, 0.0) m/s
    # In later frames after filter converges, vy should be positive and close to 3.5 m/s
    final_vy = velocities[-1][1]
    assert final_vy > 2.0, f"Expected positive lateral velocity, got {final_vy}"


def test_object_disappearing_stale_decay_and_removal():
    """Verify track survives missed frames, confidence decays monotonically, and track is removed."""
    gen = SyntheticSceneGenerator(seed=42)
    # Object is present for frames 0..4 (5 frames), then disappears at frame 5
    frames, gt = gen.generate("object_disappearing", num_frames=10, disappear_after_frame=5)

    model = MockModel()
    extractor = ObjectExtractor.load()
    # Configure with max_missed_frames = 3, confidence_decay_rate = 0.20
    tracker = MultiObjectTracker(
        max_missed_frames=3,
        confidence_decay_rate=0.20,
        min_track_confidence=0.25,
    )

    confidences_while_unmatched = []

    for f_idx, frame in enumerate(frames):
        sem_points = model.predict(frame)
        detections = extractor.extract(sem_points)
        tracked = tracker.update(detections, timestamp=frame.timestamp)

        if f_idx < 5:
            # Object actively present
            assert len(tracked) == 1
            assert tracked[0].track_confidence >= 0.90
        elif f_idx in (5, 6, 7):
            # Object absent but within 3-frame coasting tolerance -> track survives
            assert len(tracked) == 1, f"Frame {f_idx} track should survive within coasting tolerance"
            confidences_while_unmatched.append(tracked[0].track_confidence)
        else:
            # Frame 8 and 9: missed_frames > 3 -> track must be removed
            assert len(tracked) == 0, f"Frame {f_idx} track should have been removed, got {len(tracked)}"

    # Verify track_confidence decreased strictly monotonically while unmatched
    assert len(confidences_while_unmatched) == 3
    for i in range(len(confidences_while_unmatched) - 1):
        assert confidences_while_unmatched[i] > confidences_while_unmatched[i + 1], (
            f"Confidence did not decrease monotonically: {confidences_while_unmatched}"
        )


def test_multi_object_dense_tracking_zero_id_swaps():
    """Verify multiple concurrent objects maintain distinct stable track IDs with zero ID swaps."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("multi_object_dense", num_frames=10)

    model = MockModel()
    extractor = ObjectExtractor.load()
    tracker = MultiObjectTracker.load()

    # Track ID to ground truth object association mapping
    # 4 ground truth objects: ped_near (x~8), veh_mid (x~20), truck_far (x~40), ped_roadside (x~15)
    track_to_gt_map = {}

    for f_idx, frame in enumerate(frames):
        sem_points = model.predict(frame)
        detections = extractor.extract(sem_points)
        tracked = tracker.update(detections, timestamp=frame.timestamp)

        assert len(tracked) == 4, f"Frame {f_idx} expected 4 tracked objects, got {len(tracked)}"

        gt_objs = gt["frames"][f_idx]["objects"]
        gt_positions = {obj["object_id"]: np.array(obj["position"]) for obj in gt_objs}

        for t in tracked:
            t_pos = np.array(t.position)
            # Find closest ground truth object
            closest_gt_id = min(gt_positions.keys(), key=lambda gid: np.linalg.norm(t_pos - gt_positions[gid]))
            min_dist = np.linalg.norm(t_pos - gt_positions[closest_gt_id])
            assert min_dist < 2.0, f"Track {t.track_id} position {t_pos} too far from GT {closest_gt_id} {gt_positions[closest_gt_id]}"

            if t.track_id not in track_to_gt_map:
                track_to_gt_map[t.track_id] = closest_gt_id
            else:
                # Assert ZERO ID SWAPS: this track_id must still match the same ground truth object!
                assert track_to_gt_map[t.track_id] == closest_gt_id, (
                    f"ID SWAP detected! Track {t.track_id} previously matched {track_to_gt_map[t.track_id]} "
                    f"but in frame {f_idx} matched {closest_gt_id}"
                )

    # Exactly 4 distinct tracks mapped 1:1 with the 4 ground truth objects
    assert len(track_to_gt_map) == 4
    assert set(track_to_gt_map.values()) == {"ped_near", "veh_mid", "truck_far", "ped_roadside"}
