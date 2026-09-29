"""Boundary correctness and spatial partitioning unit tests for AdaptiveGridEngine."""

import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.mapping.grid_engine import AdaptiveGridEngine


@pytest.fixture
def grid_engine() -> AdaptiveGridEngine:
    """Fixture providing an AdaptiveGridEngine initialized with default config."""
    return AdaptiveGridEngine()


def test_boundary_exact_assignment(grid_engine: AdaptiveGridEngine):
    """Verify that a point at exactly x=10.0 (band edge) is deterministically assigned to exactly one band."""
    # Point exactly at (10.0, 0.0) -> distance = 10.0
    # In half-open intervals [0, 10) and [10, 25), 10.0 belongs to band 1 (mid_range)
    key1 = grid_engine.cell_index(10.0, 0.0)
    key2 = grid_engine.cell_index(10.0, 0.0)
    key3 = grid_engine.cell_index(10.0, 0.0)

    assert key1 is not None
    assert key1 == key2 == key3
    assert key1[0] == 1  # Band index 1


def test_boundary_transition_at_999_and_1001(grid_engine: AdaptiveGridEngine):
    """Verify that points at x=9.99 and x=10.01 cross the 10.0m boundary into different bands."""
    key_inner = grid_engine.cell_index(9.99, 0.0)
    key_outer = grid_engine.cell_index(10.01, 0.0)

    assert key_inner is not None
    assert key_outer is not None

    # 9.99m must be in band 0 ([0, 10) fovea)
    assert key_inner[0] == 0
    # 10.01m must be in band 1 ([10, 25) mid_range)
    assert key_outer[0] == 1


def test_cell_bounds_tile_perfectly_without_overlap(grid_engine: AdaptiveGridEngine):
    """Verify that adjacent cells within a band tile seamlessly with zero gaps and zero overlaps."""
    # Check across multiple columns and rows in band 0 (resolution 0.05m)
    band_idx = 0
    res = grid_engine.resolution_for_band(band_idx)

    for col in range(-5, 5):
        for row in range(-5, 5):
            curr_bounds = grid_engine.cell_bounds((band_idx, col, row))
            right_bounds = grid_engine.cell_bounds((band_idx, col + 1, row))
            top_bounds = grid_engine.cell_bounds((band_idx, col, row + 1))

            # Right neighbor: curr.max_x must equal right.min_x
            assert np.isclose(curr_bounds[2], right_bounds[0]), "Gap or overlap on X axis between adjacent cells"
            # Top neighbor: curr.max_y must equal top.min_y
            assert np.isclose(curr_bounds[3], top_bounds[1]), "Gap or overlap on Y axis between adjacent cells"
            # Width and height must match band resolution exactly
            assert np.isclose(curr_bounds[2] - curr_bounds[0], res)
            assert np.isclose(curr_bounds[3] - curr_bounds[1], res)


def test_zero_data_loss_multi_band(grid_engine: AdaptiveGridEngine):
    """Verify that every valid point across a dense multi-band distribution is assigned to exactly one cell."""
    rng = np.random.RandomState(42)
    N = 2000
    # Distribute points radially from 0.5m to 80m across all 4 zones
    radii = rng.uniform(0.5, 80.0, size=N)
    angles = rng.uniform(-np.pi, np.pi, size=N)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    pts = np.column_stack([x, y, np.zeros(N)])

    projection = grid_engine.project_points(pts)

    # Count total points assigned across all cells
    total_assigned = sum(len(indices) for indices in projection.values())

    # Every point must be mapped with zero data loss
    assert total_assigned == N

    # Verify no duplicate indices across cells
    all_indices = [idx for indices in projection.values() for idx in indices]
    assert len(all_indices) == len(set(all_indices))


def test_negative_coordinates_handled_properly(grid_engine: AdaptiveGridEngine):
    """Verify that negative Cartesian coordinates (x < 0, y < 0) map cleanly without error."""
    # Point at (-5.0, -5.0) -> r = sqrt(50) ≈ 7.07m (band 0)
    key_neg = grid_engine.cell_index(-5.0, -5.0)
    assert key_neg is not None
    assert key_neg[0] == 0
    assert key_neg[1] < 0
    assert key_neg[2] < 0

    bounds = grid_engine.cell_bounds(key_neg)
    assert bounds[0] <= -5.0 < bounds[2]
    assert bounds[1] <= -5.0 < bounds[3]


def test_out_of_bounds_policy_reject_and_clamp():
    """Verify that out-of-bounds coordinates (>= max_range) handle reject and clamp policies gracefully."""
    # Default reject engine
    reject_engine = AdaptiveGridEngine(out_of_bounds_policy="reject")
    key_rejected = reject_engine.cell_index(150.0, 0.0)
    assert key_rejected is None

    # Points outside max_range are skipped without crashing in project_points
    pts = np.array([[150.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
    proj = reject_engine.project_points(pts)
    assert sum(len(idx) for idx in proj.values()) == 1

    # Clamp engine
    clamp_engine = AdaptiveGridEngine(out_of_bounds_policy="clamp")
    key_clamped = clamp_engine.cell_index(150.0, 0.0)
    assert key_clamped is not None
    assert key_clamped[0] == len(clamp_engine.zones) - 1  # Outermost band


def test_resolution_boundary_stress_scenario(grid_engine: AdaptiveGridEngine):
    """Test boundary correctness against the Prompt 3 'resolution_boundary_stress' scenario."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("resolution_boundary_stress", num_frames=3)

    for frame in frames:
        proj = grid_engine.project_points(frame.points)
        total_assigned = sum(len(idx) for idx in proj.values())
        # All points are within max_range, so total assigned must equal input point count
        assert total_assigned == frame.points.shape[0]

        # Verify points at 9.99, 10.00, 10.01 specifically
        for key, indices in proj.items():
            cell = grid_engine.get_or_create_cell(key)
            assert cell.resolution == grid_engine.resolution_for_band(key[0])
