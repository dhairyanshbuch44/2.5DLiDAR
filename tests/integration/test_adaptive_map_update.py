"""Integration tests for the AdaptiveMap multi-frame update cycle."""

import json
import numpy as np
import pytest

from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.mapping.adaptive_map import AdaptiveMap
from fovea_lidar.preprocessing.pipeline import Preprocessor


def test_empty_road_multi_frame_update():
    """Verify multi-frame integration of empty_road scenario through preprocessor and map.

    Checks:
    - Cell count does not explode unboundedly across static frames.
    - Elevation values are stable and close to zero for the flat ground.
    - Timestamps update sequentially.
    """
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("empty_road", num_frames=5)

    preprocessor = Preprocessor()
    adaptive_map = AdaptiveMap()

    cell_counts: list[int] = []

    for frame in frames:
        cleaned_frame, _ = preprocessor.process(frame)
        adaptive_map.update(cleaned_frame)
        cell_counts.append(adaptive_map.total_cells)

    assert adaptive_map.frame_count == 5
    assert adaptive_map.last_update_time == pytest.approx(frames[-1].timestamp)

    # In a static empty road, cell count grows sublinearly and is bounded by total points
    assert 0 < cell_counts[-1] <= 5 * 600
    assert cell_counts[-1] < 3500

    # Verify elevation values of ground cells are near 0.0m
    elevations = [c.elevation for c in adaptive_map.cells.values()]
    assert len(elevations) > 0
    mean_ground_elevation = float(np.mean(elevations))
    assert abs(mean_ground_elevation) < 0.1, f"Ground elevation drifted: {mean_ground_elevation}"


def test_multi_object_dense_multi_frame_update():
    """Verify multi_object_dense scenario integration and observe cell count stability."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("multi_object_dense", num_frames=5)

    preprocessor = Preprocessor()
    adaptive_map = AdaptiveMap()

    for frame in frames:
        cleaned_frame, _ = preprocessor.process(frame)
        adaptive_map.update(cleaned_frame)

    assert adaptive_map.frame_count == 5
    assert adaptive_map.total_cells > 0
    print(f"\nObserved total_cells in multi_object_dense after 5 frames: {adaptive_map.total_cells}")


def test_update_idempotency_on_identical_frame():
    """Verify that updating with the exact same frame twice does not duplicate cells."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("empty_road", num_frames=1)

    preprocessor = Preprocessor()
    cleaned_frame, _ = preprocessor.process(frames[0])

    adaptive_map = AdaptiveMap()
    adaptive_map.update(cleaned_frame)
    initial_cell_count = adaptive_map.total_cells

    # Second update on identical frame
    adaptive_map.update(cleaned_frame)
    assert adaptive_map.total_cells == initial_cell_count
    assert adaptive_map.frame_count == 2


def test_stale_cell_detection():
    """Verify get_stale_cells identifies cells that haven't been updated recently."""
    adaptive_map = AdaptiveMap()

    # Create dummy frame at t=0.0
    pts = np.array([[2.0, 2.0, 0.0]], dtype=np.float32)
    from fovea_lidar.contracts.point_cloud import PointCloudFrame

    f1 = PointCloudFrame(points=pts, timestamp=0.0, frame_id="lidar")
    adaptive_map.update(f1)

    # At t=10.0 with max_age=5.0, cell updated at t=0.0 must be stale
    stale_keys = adaptive_map.get_stale_cells(current_time=10.0, max_age=5.0)
    assert len(stale_keys) == 1

    # At t=2.0 with max_age=5.0, cell is fresh (not stale)
    fresh_keys = adaptive_map.get_stale_cells(current_time=2.0, max_age=5.0)
    assert len(fresh_keys) == 0


def test_to_dict_json_serialization():
    """Verify to_dict() produces valid, dumpable JSON with no custom types leaking."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("empty_road", num_frames=2)

    preprocessor = Preprocessor()
    adaptive_map = AdaptiveMap()

    for frame in frames:
        cleaned, _ = preprocessor.process(frame)
        adaptive_map.update(cleaned)

    map_dict = adaptive_map.to_dict()
    assert "cells" in map_dict
    assert "metadata" in map_dict
    assert map_dict["metadata"]["frame_count"] == 2

    # Verify JSON serialization works without error
    json_str = json.dumps(map_dict)
    assert len(json_str) > 0

    reloaded = json.loads(json_str)
    assert len(reloaded["cells"]) == adaptive_map.total_cells
