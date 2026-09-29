"""Unit tests for AdaptiveGridEngine functionality and aggregation."""

import numpy as np
import pytest

from fovea_lidar.contracts.mapping import AdaptiveCell, AdaptiveMap
from fovea_lidar.mapping.aggregation import (
    aggregate_elevation,
    aggregate_occupancy,
    update_cell_with_points,
)
from fovea_lidar.mapping.grid_engine import AdaptiveGridEngine


def test_multiple_points_in_cell_aggregate_correctly():
    """Verify that multiple points falling into one cell produce median elevation and updated occupancy."""
    engine = AdaptiveGridEngine()

    # Create 3 points in the exact same cell (col=40, row=0, band=0): x=2.01, 2.02, 2.03, y=0.01
    # with z values: 1.0, 1.2, 5.0 (outlier 5.0) -> median must be 1.2
    pts = np.array(
        [
            [2.01, 0.01, 1.0],
            [2.02, 0.01, 1.2],
            [2.03, 0.01, 5.0],
        ],
        dtype=np.float32,
    )

    proj = engine.project_points(pts)
    assert len(proj) == 1
    key = list(proj.keys())[0]

    cell = engine.get_or_create_cell(key)
    assert cell.occupancy_probability == 0.5  # default prior

    update_cell_with_points(cell, pts, proj[key], timestamp=10.5, hit_increment=0.1)

    # Median elevation must be 1.2 (ignoring outlier 5.0)
    assert np.isclose(cell.elevation, 1.2)
    # Variance of [1.0, 1.2, 5.0]
    expected_var = float(np.var([1.0, 1.2, 5.0]))
    assert np.isclose(cell.elevation_variance, expected_var)
    # Occupancy: 0.5 + 3 * 0.1 = 0.8
    assert np.isclose(cell.occupancy_probability, 0.8)
    assert cell.last_updated == 10.5


def test_empty_cells_are_not_materialized():
    """Verify that empty, unobserved cells are never stored in memory."""
    engine = AdaptiveGridEngine()
    assert engine.total_cells == 0

    pts = np.array([[5.0, 2.0, 0.0]], dtype=np.float32)
    proj = engine.project_points(pts)

    # project_points alone does not create cells until get_or_create_cell is requested
    assert engine.total_cells == 0

    for key in proj:
        _ = engine.get_or_create_cell(key)

    assert engine.total_cells == 1


def test_get_or_create_cell_is_idempotent():
    """Verify that requesting the same CellKey returns the identical AdaptiveCell instance."""
    engine = AdaptiveGridEngine()
    key = (0, 10, -5)

    cell_a = engine.get_or_create_cell(key)
    cell_b = engine.get_or_create_cell(key)

    assert cell_a is cell_b
    assert engine.total_cells == 1


def test_to_adaptive_map_conversion():
    """Verify conversion of grid engine state to the AdaptiveMap contract."""
    engine = AdaptiveGridEngine()
    c1 = engine.get_or_create_cell((0, 2, 3))
    c1.occupancy_probability = 0.85
    c2 = engine.get_or_create_cell((1, 15, 20))
    c2.elevation = 2.4

    adaptive_map = engine.to_adaptive_map(metadata={"frame_seq": 99})
    assert isinstance(adaptive_map, AdaptiveMap)
    assert len(adaptive_map.cells) == 2
    assert len(adaptive_map.resolution_zones) == len(engine.zones)
    assert adaptive_map.metadata["frame_seq"] == 99

    # Verify serialization round-trip of the exported map
    map_dict = adaptive_map.to_dict()
    restored = AdaptiveMap.from_dict(map_dict)
    assert len(restored.cells) == 2
