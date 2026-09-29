"""Unit tests for uncertainty estimation at cell, object, and regional levels."""

import pytest

from fovea_lidar.contracts.mapping import AdaptiveCell, AdaptiveMap
from fovea_lidar.contracts.objects import DetectedObject
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.risk.uncertainty import (
    compute_uncertainty,
    estimate_region_uncertainty,
)


def test_low_confidence_semantic_predictions_produce_higher_uncertainty():
    """Verify that lower semantic confidence yields strictly higher uncertainty."""
    # Direct keyword query
    unc_low_conf = compute_uncertainty(semantic_confidence=0.20)
    unc_high_conf = compute_uncertainty(semantic_confidence=0.95)

    assert unc_low_conf > unc_high_conf
    assert unc_low_conf == pytest.approx(0.80, abs=1e-3)
    assert unc_high_conf == pytest.approx(0.05, abs=1e-3)

    # Cell-level comparison
    cell_ambiguous = AdaptiveCell(
        spatial_bounds=(0.0, 0.0, 1.0, 1.0),
        resolution=0.10,
        semantic_confidence=0.20,
        occupancy_probability=0.95,  # definitive occupancy
    )
    cell_certain = AdaptiveCell(
        spatial_bounds=(0.0, 0.0, 1.0, 1.0),
        resolution=0.10,
        semantic_confidence=0.95,
        occupancy_probability=0.95,
    )

    unc_cell_ambiguous = compute_uncertainty(cell_ambiguous)
    unc_cell_certain = compute_uncertainty(cell_certain)

    assert unc_cell_ambiguous > unc_cell_certain


def test_ambiguous_occupancy_produces_higher_uncertainty():
    """Verify that ambiguous occupancy (p close to 0.5) yields higher uncertainty than p near 0 or 1."""
    # p = 0.5 is maximum uncertainty (unobserved or unknown space)
    unc_ambiguous = compute_uncertainty(occupancy_probability=0.50)
    # p = 0.99 is definitive obstacle; p = 0.01 is definitive free space
    unc_occupied = compute_uncertainty(occupancy_probability=0.99)
    unc_free = compute_uncertainty(occupancy_probability=0.01)

    assert unc_ambiguous > unc_occupied
    assert unc_ambiguous > unc_free
    assert unc_ambiguous == pytest.approx(1.0, abs=1e-3)
    assert unc_occupied == pytest.approx(0.02, abs=1e-2)
    assert unc_free == pytest.approx(0.02, abs=1e-2)

    # Cell-level comparison
    cell_p50 = AdaptiveCell(
        spatial_bounds=(0.0, 0.0, 1.0, 1.0),
        resolution=0.10,
        occupancy_probability=0.50,
        semantic_confidence=0.80,
    )
    cell_p90 = AdaptiveCell(
        spatial_bounds=(0.0, 0.0, 1.0, 1.0),
        resolution=0.10,
        occupancy_probability=0.90,
        semantic_confidence=0.80,
    )

    assert compute_uncertainty(cell_p50) > compute_uncertainty(cell_p90)


def test_object_level_uncertainty():
    """Verify object-level uncertainty is strictly 1.0 - confidence."""
    det_low = DetectedObject(
        object_id="det_1",
        semantic_class="PEDESTRIAN",
        position=(10.0, 0.0, 0.0),
        bounding_region=(9.5, -0.5, 0.0, 10.5, 0.5, 1.5),
        confidence=0.30,
    )
    det_high = DetectedObject(
        object_id="det_2",
        semantic_class="PEDESTRIAN",
        position=(10.0, 0.0, 0.0),
        bounding_region=(9.5, -0.5, 0.0, 10.5, 0.5, 1.5),
        confidence=0.90,
    )

    trk_obj = TrackedObject(
        object_id="trk_1",
        track_id="trk_1",
        semantic_class="VEHICLE",
        position=(15.0, 2.0, 0.0),
        bounding_region=(13.0, 1.0, 0.0, 17.0, 3.0, 2.0),
        confidence=0.75,
        velocity=(0.0, 0.0, 0.0),
        dynamic=True,
    )

    assert compute_uncertainty(det_low) == pytest.approx(0.70, abs=1e-3)
    assert compute_uncertainty(det_high) == pytest.approx(0.10, abs=1e-3)
    assert compute_uncertainty(trk_obj) == pytest.approx(0.25, abs=1e-3)


def test_cell_staleness_increases_uncertainty():
    """Verify that unobserved stale cells have increased uncertainty over elapsed time."""
    cell = AdaptiveCell(
        spatial_bounds=(0.0, 0.0, 1.0, 1.0),
        resolution=0.10,
        semantic_confidence=1.0,
        occupancy_probability=1.0,
        last_updated=10.0,
    )

    # Immediate observation (dt = 0s)
    unc_fresh = compute_uncertainty(cell, current_time=10.0, staleness_tau=5.0)
    # Elapsed 2.5s (half life)
    unc_half = compute_uncertainty(cell, current_time=12.5, staleness_tau=5.0)
    # Elapsed 10.0s (fully stale)
    unc_stale = compute_uncertainty(cell, current_time=20.0, staleness_tau=5.0)

    assert unc_fresh == pytest.approx(0.0, abs=1e-3)
    assert unc_fresh < unc_half < unc_stale


def test_region_uncertainty_estimation():
    """Verify spatial uncertainty estimation over local regions of an AdaptiveMap."""
    adaptive_map = AdaptiveMap()

    # Empty map
    assert estimate_region_uncertainty(adaptive_map, center=(0.0, 0.0), radius=5.0) == 0.0

    # Add ambiguous cell near (5.0, 5.0)
    cell_near = AdaptiveCell(
        spatial_bounds=(4.8, 4.8, 5.2, 5.2),
        resolution=0.4,
        semantic_confidence=0.20,
        occupancy_probability=0.50,
    )
    # Add definitive cell far at (50.0, 50.0)
    cell_far = AdaptiveCell(
        spatial_bounds=(49.8, 49.8, 50.2, 50.2),
        resolution=0.4,
        semantic_confidence=0.99,
        occupancy_probability=0.99,
    )
    adaptive_map.cells[(0, 10, 10)] = cell_near
    adaptive_map.cells[(2, 100, 100)] = cell_far

    # Query near (5.0, 5.0) should be high uncertainty
    unc_near = estimate_region_uncertainty(adaptive_map, center=(5.0, 5.0), radius=2.0)
    assert unc_near > 0.70

    # Query near (50.0, 50.0) should be low uncertainty
    unc_far = estimate_region_uncertainty(adaptive_map, center=(50.0, 50.0), radius=2.0)
    assert unc_far < 0.15
