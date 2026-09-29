"""Integration tests for dynamic semantic foveation pipeline tracking moving objects."""

import numpy as np
import pytest

from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.mapping.adaptive_map import AdaptiveMap
from fovea_lidar.objects.extraction import ObjectExtractor
from fovea_lidar.perception.mock_model import MockModel
from fovea_lidar.preprocessing.pipeline import Preprocessor
from fovea_lidar.resolution.fovea_manager import FoveaManager
from fovea_lidar.tracking.tracker import MultiObjectTracker



def test_dynamic_fovea_tracks_moving_pedestrian_across_frames():
    """Verify dynamic fovea patch moves frame-to-frame tracking the pedestrian's ground truth."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("pedestrian_at_distance", num_frames=10)

    preprocessor = Preprocessor()
    model = MockModel()
    extractor = ObjectExtractor.load()
    tracker = MultiObjectTracker.load()
    fovea_manager = FoveaManager.load()
    adaptive_map = AdaptiveMap()

    fovea_centers = []
    gt_positions = []

    for f_idx, raw_frame in enumerate(frames):
        # 1. Pipeline processing stages
        cleaned_frame, _ = preprocessor.process(raw_frame)
        sem_points = model.predict(cleaned_frame)
        detections = extractor.extract(sem_points)
        tracked_objects = tracker.update(detections, timestamp=cleaned_frame.timestamp)

        # 2. Semantic foveation decision and overlay management
        fovea_manager.update(
            tracked_objects=tracked_objects,
            adaptive_map=adaptive_map,
        )

        # 3. Map update (points within fovea route into fine overlay sub-grid)
        adaptive_map.update(cleaned_frame, semantic_points=sem_points)

        # 4. Verify active fovea region
        active_foveas = fovea_manager.get_active_fovea_regions()
        assert len(active_foveas) == 1, f"Frame {f_idx} expected 1 active fovea, got {len(active_foveas)}"

        fovea = active_foveas[0]
        fovea_center = fovea["center"]
        fovea_res = fovea["resolution"]
        fovea_centers.append((round(fovea_center[0], 3), round(fovea_center[1], 3)))

        gt_pos = gt["frames"][f_idx]["objects"][0]["position"]
        gt_positions.append((round(gt_pos[0], 3), round(gt_pos[1], 3)))

        # Fovea resolution must be significantly finer than base band resolution (0.10m)
        assert fovea_res < 0.10
        assert fovea_res <= 0.08


        # Fovea center must track the pedestrian's ground truth position within 0.5m
        dist_to_gt = float(np.hypot(fovea_center[0] - gt_pos[0], fovea_center[1] - gt_pos[1]))
        assert dist_to_gt < 0.5, (
            f"Frame {f_idx}: Fovea center {fovea_center} deviated from ground truth {gt_pos} by {dist_to_gt:.3f}m"
        )

        # 5. Isolation verification: far empty road cells (x > 25m) must remain coarse (res=0.25m)
        far_cells = [
            c for k, c in adaptive_map.cells.items()
            if isinstance(k[0], int) and k[0] == 2  # base band 2 (far field 25-50m)
        ]
        assert len(far_cells) > 0, "Expected far road base cells to be populated"
        for fc in far_cells:
            assert fc.resolution == 0.25, f"Far road cell had unexpected resolution {fc.resolution}"

    print("\n" + "=" * 70)
    print("Observed Dynamic Fovea Movement vs Ground Truth:")
    for f_idx, (fc, gtp) in enumerate(zip(fovea_centers, gt_positions)):
        print(f"  Frame {f_idx}: Fovea Center = ({fc[0]:.3f}, {fc[1]:.3f}) | Ground Truth = ({gtp[0]:.3f}, {gtp[1]:.3f})")
    print("=" * 70)

    # Verify frame-to-frame displacement: fovea center must NOT be static
    x_displacements = [fovea_centers[i][0] - fovea_centers[i + 1][0] for i in range(len(fovea_centers) - 1)]
    assert any(abs(dx) > 0.01 for dx in x_displacements), "Fovea center was unexpectedly stationary"


def test_fovea_overlay_clean_lifecycle_removal():
    """Verify fovea overlay is cleanly removed with zero orphaned cells when object is lost."""
    adaptive_map = AdaptiveMap()
    fovea_manager = FoveaManager.load()

    # Frame 1: Object present
    track = TrackedObject(
        object_id="test_track",
        track_id="test_track",
        semantic_class="PEDESTRIAN",
        position=(15.0, 0.0, 0.9),
        bounding_region=(14.5, -0.5, 0.0, 15.5, 0.5, 1.8),
        confidence=0.90,
        velocity=(0.0, 0.0, 0.0),
        dynamic=True,
    )
    fovea_manager.update([track], adaptive_map)
    assert len(fovea_manager.get_active_fovea_regions()) == 1
    assert len(adaptive_map.grid_engine.fovea_overlays) == 1

    # Populate an overlay cell
    overlay_key = ("fovea_test_track", 300, 0)
    cell = adaptive_map.grid_engine.get_or_create_cell(overlay_key)
    assert overlay_key in adaptive_map.cells

    # Frame 2: Object disappears / no tracks
    fovea_manager.update([], adaptive_map)
    assert len(fovea_manager.get_active_fovea_regions()) == 0
    assert len(adaptive_map.grid_engine.fovea_overlays) == 0

    # Overlay cell must be cleanly purged (no orphaned cells)
    assert overlay_key not in adaptive_map.cells
