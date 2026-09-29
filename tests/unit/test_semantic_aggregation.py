"""Unit tests for semantic aggregation and highest-confidence-wins rules."""

import pytest

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame
from fovea_lidar.contracts.semantic import SemanticPoint
from fovea_lidar.mapping.adaptive_map import AdaptiveMap
from fovea_lidar.mapping.aggregation import aggregate_semantics
import numpy as np


def test_highest_confidence_wins_multi_candidates():
    """Verify that among multiple conflicting class predictions, highest confidence wins."""
    candidates = [
        ("ROAD", 0.30),
        ("VEHICLE", 0.88),
        ("PEDESTRIAN", 0.65),
    ]
    win_class, win_conf = aggregate_semantics(candidates)
    assert win_class == "VEHICLE"
    assert win_conf == pytest.approx(0.88)


def test_higher_confidence_overrides_prior_cell_belief():
    """Verify that a high-confidence new observation updates prior lower-confidence belief."""
    # Prior cell has ("ROAD", 0.50)
    candidates = [("PEDESTRIAN", 0.85)]
    win_class, win_conf = aggregate_semantics(candidates, prev_class="ROAD", prev_confidence=0.50)
    assert win_class == "PEDESTRIAN"
    assert win_conf == pytest.approx(0.85)


def test_lower_confidence_does_not_override_prior_belief():
    """Verify that a lower-confidence observation does not degrade a high-confidence belief."""
    # Prior cell has ("VEHICLE", 0.92)
    candidates = [("ROAD", 0.40)]
    win_class, win_conf = aggregate_semantics(candidates, prev_class="VEHICLE", prev_confidence=0.92)
    assert win_class == "VEHICLE"
    assert win_conf == pytest.approx(0.92)


def test_deterministic_tie_breaking():
    """Verify deterministic tie-breaking when two classes share identical confidence."""
    candidates = [
        ("VEHICLE", 0.75),
        ("PEDESTRIAN", 0.75),
    ]
    # Alphabetical sorting deterministically picks PEDESTRIAN before VEHICLE
    win_class, win_conf = aggregate_semantics(candidates)
    assert win_class == "PEDESTRIAN"
    assert win_conf == pytest.approx(0.75)


def test_empty_candidates_preserves_prior():
    """Verify empty candidate list retains existing cell class and confidence."""
    win_class, win_conf = aggregate_semantics([], prev_class="GRASS", prev_confidence=0.67)
    assert win_class == "GRASS"
    assert win_conf == pytest.approx(0.67)


def test_adaptive_map_semantic_points_integration():
    """Verify that AdaptiveMap integrates SemanticPoints into corresponding cells."""
    map_engine = AdaptiveMap()

    # Point at (2.0, 2.0, 0.5)
    pts = np.array([[2.0, 2.0, 0.5]], dtype=np.float32)
    frame = PointCloudFrame(points=pts, timestamp=1.0, frame_id="lidar")

    sp = SemanticPoint(
        coordinates=(2.0, 2.0, 0.5),
        semantic_class="VEHICLE",
        confidence=0.91,
        timestamp=1.0,
    )

    map_engine.update(frame, semantic_points=[sp])
    assert map_engine.total_cells == 1

    cell = list(map_engine.cells.values())[0]
    assert cell.semantic_class == "VEHICLE"
    assert cell.semantic_confidence == pytest.approx(0.91)
