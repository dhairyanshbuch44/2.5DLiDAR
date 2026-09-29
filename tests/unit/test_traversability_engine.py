"""Unit tests for TraversabilityEngine dynamic occupancy and risk-level downgrade logic."""

import pytest

from fovea_lidar.contracts.mapping import AdaptiveCell, AdaptiveMap, TraversabilityMap
from fovea_lidar.contracts.risk import RiskLevel, RiskRegion
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.terrain.analysis import TerrainClass
from fovea_lidar.traversability.engine import TraversabilityEngine


def _create_test_track(
    track_id: str,
    position: tuple[float, float, float],
    risk_level: RiskLevel,
    semantic_class: str = "VEHICLE",
) -> TrackedObject:
    """Helper to construct a TrackedObject with an assigned risk_level attribute."""
    x, y, z = position
    track = TrackedObject(
        object_id=track_id,
        track_id=track_id,
        semantic_class=semantic_class,
        position=(x, y, z),
        bounding_region=(x - 1.0, y - 1.0, z - 1.0, x + 1.0, y + 1.0, z + 1.0),
        confidence=0.90,
        velocity=(0.0, 2.0, 0.0),
        dynamic=True,
    )
    # Attach risk level for explicit test control
    track.risk_level = risk_level  # type: ignore[attr-defined]
    return track


def test_safe_cell_downgrades_per_risk_level():
    """Verify that a SAFE terrain cell transitions to CAUTION for LOW/MEDIUM risk,

    and to NON_TRAVERSABLE for HIGH/CRITICAL risk.
    """
    engine = TraversabilityEngine(
        dynamic_penalties={"LOW": 0.50, "MEDIUM": 0.60, "HIGH": 0.90, "CRITICAL": 1.00}
    )

    cell_key = (1, 100, 0)
    cell = AdaptiveCell(
        spatial_bounds=(9.5, -0.5, 10.5, 0.5),
        resolution=0.10,
        elevation=0.0,
        traversability_score=1.0,  # SAFE road
        semantic_class="ROAD",
        semantic_confidence=0.95,
        occupancy_probability=0.05,
    )

    # --- Case 1: LOW risk dynamic object -> CAUTION ---
    map_low = AdaptiveMap(cells={cell_key: cell})
    trk_low = _create_test_track("trk_low", (10.0, 0.0, 0.5), RiskLevel.LOW)
    tmap_low = engine.evaluate(map_low, [trk_low])

    assert tmap_low.get_label(cell_key) == TerrainClass.CAUTION
    assert tmap_low.get_score(cell_key) == pytest.approx(0.50, abs=1e-3)
    assert tmap_low.is_traversable(cell_key, threshold=0.50) is True
    assert tmap_low.is_traversable(cell_key, threshold=0.60) is False

    # --- Case 2: MEDIUM risk dynamic object -> CAUTION ---
    map_med = AdaptiveMap(cells={cell_key: cell})
    trk_med = _create_test_track("trk_med", (10.0, 0.0, 0.5), RiskLevel.MEDIUM)
    tmap_med = engine.evaluate(map_med, [trk_med])

    assert tmap_med.get_label(cell_key) == TerrainClass.CAUTION
    assert tmap_med.get_score(cell_key) == pytest.approx(0.40, abs=1e-3)

    # --- Case 3: HIGH risk dynamic object -> NON_TRAVERSABLE ---
    map_high = AdaptiveMap(cells={cell_key: cell})
    trk_high = _create_test_track("trk_high", (10.0, 0.0, 0.5), RiskLevel.HIGH)
    tmap_high = engine.evaluate(map_high, [trk_high])

    assert tmap_high.get_label(cell_key) == TerrainClass.NON_TRAVERSABLE
    assert tmap_high.get_score(cell_key) == pytest.approx(0.10, abs=1e-3)
    assert tmap_high.is_traversable(cell_key, threshold=0.50) is False

    # --- Case 4: CRITICAL risk dynamic object -> NON_TRAVERSABLE ---
    map_crit = AdaptiveMap(cells={cell_key: cell})
    trk_crit = _create_test_track("trk_crit", (10.0, 0.0, 0.5), RiskLevel.CRITICAL)
    tmap_crit = engine.evaluate(map_crit, [trk_crit])

    assert tmap_crit.get_label(cell_key) == TerrainClass.NON_TRAVERSABLE
    assert tmap_crit.get_score(cell_key) == pytest.approx(0.00, abs=1e-3)
    assert tmap_crit.is_traversable(cell_key, threshold=0.10) is False


def test_unknown_cells_remain_unknown_regardless_of_object_overlap():
    """Verify UNKNOWN terrain / high-uncertainty cells remain UNKNOWN regardless of overlapping dynamic obstacles."""
    engine = TraversabilityEngine(uncertainty_threshold=0.70)

    cell_key = (1, 100, 0)
    # Cell with low semantic confidence -> UNKNOWN
    cell_unknown = AdaptiveCell(
        spatial_bounds=(9.5, -0.5, 10.5, 0.5),
        resolution=0.10,
        elevation=0.0,
        traversability_score=0.25,
        semantic_class=None,
        semantic_confidence=0.10,  # very low confidence
        occupancy_probability=0.50,  # ambiguous occupancy
    )

    # Test overlap with all four risk levels
    for r_level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
        adaptive_map = AdaptiveMap(cells={cell_key: cell_unknown})
        track = _create_test_track(f"trk_{r_level.value}", (10.0, 0.0, 0.5), r_level)
        tmap = engine.evaluate(adaptive_map, [track])

        # Must strictly remain UNKNOWN
        assert tmap.get_label(cell_key) == TerrainClass.UNKNOWN, (
            f"UNKNOWN cell was silently reclassified to {tmap.get_label(cell_key)} by {r_level.value} track"
        )
        assert tmap.get_score(cell_key) == pytest.approx(0.25, abs=1e-3)


def test_non_overlapping_cells_preserve_base_terrain():
    """Verify cells not overlapping any dynamic object preserve their original static terrain classification and score."""
    engine = TraversabilityEngine()

    key_road = (1, 100, 0)
    cell_road = AdaptiveCell(
        spatial_bounds=(9.5, -0.5, 10.5, 0.5),
        resolution=0.10,
        traversability_score=1.0,
        semantic_class="ROAD",
        semantic_confidence=0.90,
    )

    key_curb = (1, 100, 10)
    cell_curb = AdaptiveCell(
        spatial_bounds=(9.5, 9.5, 10.5, 10.5),
        resolution=0.10,
        traversability_score=0.5,
        semantic_class="CURB",
        semantic_confidence=0.90,
    )

    adaptive_map = AdaptiveMap(cells={key_road: cell_road, key_curb: cell_curb})

    # Track far away at (30.0, 30.0)
    distant_track = _create_test_track("trk_far", (30.0, 30.0, 0.0), RiskLevel.CRITICAL)

    tmap = engine.evaluate(adaptive_map, [distant_track])

    # Both cells must retain their static terrain labels and scores unchanged
    assert tmap.get_label(key_road) == TerrainClass.SAFE
    assert tmap.get_score(key_road) == pytest.approx(1.0, abs=1e-3)

    assert tmap.get_label(key_curb) == TerrainClass.CAUTION
    assert tmap.get_score(key_curb) == pytest.approx(0.5, abs=1e-3)


def test_evaluate_with_precomputed_risk_regions():
    """Verify that TraversabilityEngine correctly associates risk levels from pre-computed RiskRegions."""
    engine = TraversabilityEngine()

    cell_key = (1, 50, 0)
    cell = AdaptiveCell(
        spatial_bounds=(4.5, -0.5, 5.5, 0.5),
        resolution=0.10,
        traversability_score=1.0,
        semantic_class="ROAD",
        semantic_confidence=0.90,
    )
    adaptive_map = AdaptiveMap(cells={cell_key: cell})

    track = TrackedObject(
        object_id="trk_fast_ped",
        track_id="trk_fast_ped",
        semantic_class="PEDESTRIAN",
        position=(5.0, 0.0, 0.8),
        bounding_region=(4.5, -0.5, 0.0, 5.5, 0.5, 1.8),
        confidence=0.95,
        velocity=(-2.5, 0.0, 0.0),
        dynamic=True,
    )

    risk_region = RiskRegion(
        region_bounds=(4.0, -1.0, 0.0, 6.0, 1.0, 2.0),
        risk_score=0.85,
        risk_level=RiskLevel.CRITICAL,
        associated_object_id="trk_fast_ped",
    )

    tmap = engine.evaluate(adaptive_map, [track], risk_regions=[risk_region])

    assert tmap.get_label(cell_key) == TerrainClass.NON_TRAVERSABLE
    assert tmap.get_score(cell_key) == 0.0
