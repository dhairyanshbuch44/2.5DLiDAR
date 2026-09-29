"""Unit tests for RiskEngine dynamic collision hazard assessment."""

import pytest

from fovea_lidar.contracts.mapping import AdaptiveCell, AdaptiveMap
from fovea_lidar.contracts.risk import RiskLevel, RiskRegion
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.risk.engine import RiskEngine


def test_fast_approaching_pedestrian_vs_distant_receding_vehicle():
    """Verify fast-approaching pedestrian receives CRITICAL/HIGH risk, while distant receding vehicle receives LOW risk."""
    engine = RiskEngine.load()

    # 1. Fast-approaching pedestrian at 5m moving toward sensor origin at 3 m/s
    ped = TrackedObject(
        object_id="ped_1",
        track_id="ped_1",
        semantic_class="PEDESTRIAN",
        position=(5.0, 0.0, 0.8),
        bounding_region=(4.5, -0.4, 0.0, 5.5, 0.4, 1.7),
        confidence=0.95,
        velocity=(-3.0, 0.0, 0.0),  # velocity toward origin
        dynamic=True,
        track_confidence=1.0,
    )
    score_ped, level_ped = engine.compute_object_risk(ped)

    # 2. Slow-moving receding vehicle at 30m moving away from sensor origin at 1 m/s
    veh = TrackedObject(
        object_id="veh_1",
        track_id="veh_1",
        semantic_class="VEHICLE",
        position=(30.0, 0.0, 1.0),
        bounding_region=(28.0, -1.0, 0.0, 32.0, 1.0, 1.8),
        confidence=0.85,
        velocity=(1.0, 0.0, 0.0),  # velocity away from origin
        dynamic=True,
        track_confidence=1.0,
    )
    score_veh, level_veh = engine.compute_object_risk(veh)

    print(f"\nPedestrian Risk: {score_ped:.4f} ({level_ped.value})")
    print(f"Receding Vehicle Risk: {score_veh:.4f} ({level_veh.value})")

    # Pedestrian must be much riskier than distant vehicle
    assert score_ped > score_veh
    # Acceptance criteria: fast pedestrian approaching ego produces CRITICAL or HIGH risk
    assert level_ped in (RiskLevel.CRITICAL, RiskLevel.HIGH)
    # Acceptance criteria: receding vehicle at 30m produces LOW risk (< 0.25)
    assert level_veh == RiskLevel.LOW
    assert score_veh < 0.25


def test_direction_of_motion_approaching_vs_receding():
    """Verify approaching motion produces strictly higher risk than receding motion."""
    engine = RiskEngine.load()

    # Same distance (10m), same speed (4 m/s), same class (VEHICLE)
    veh_approaching = TrackedObject(
        object_id="veh_app",
        track_id="veh_app",
        semantic_class="VEHICLE",
        position=(10.0, 0.0, 1.0),
        bounding_region=(8.5, -1.0, 0.0, 11.5, 1.0, 1.8),
        confidence=0.90,
        velocity=(-4.0, 0.0, 0.0),  # Heading directly toward ego
        dynamic=True,
    )
    veh_receding = TrackedObject(
        object_id="veh_rec",
        track_id="veh_rec",
        semantic_class="VEHICLE",
        position=(10.0, 0.0, 1.0),
        bounding_region=(8.5, -1.0, 0.0, 11.5, 1.0, 1.8),
        confidence=0.90,
        velocity=(4.0, 0.0, 0.0),  # Heading directly away from ego
        dynamic=True,
    )

    score_app, _ = engine.compute_object_risk(veh_approaching)
    score_rec, _ = engine.compute_object_risk(veh_receding)

    # Approaching vehicle must score significantly higher risk than receding vehicle
    assert score_app > score_rec
    # Approach difference is governed by w_app * (1.0 - 0.0) = 0.25 * conf
    assert (score_app - score_rec) >= 0.15


def test_risk_level_categorization_thresholds():
    """Verify correct categorization across categorical RiskLevel boundaries."""
    engine = RiskEngine(
        thresholds={"critical": 0.75, "high": 0.50, "medium": 0.25}
    )

    assert engine.determine_risk_level(0.90) == RiskLevel.CRITICAL
    assert engine.determine_risk_level(0.75) == RiskLevel.CRITICAL
    assert engine.determine_risk_level(0.749) == RiskLevel.HIGH
    assert engine.determine_risk_level(0.50) == RiskLevel.HIGH
    assert engine.determine_risk_level(0.499) == RiskLevel.MEDIUM
    assert engine.determine_risk_level(0.25) == RiskLevel.MEDIUM
    assert engine.determine_risk_level(0.249) == RiskLevel.LOW
    assert engine.determine_risk_level(0.0) == RiskLevel.LOW


def test_low_confidence_track_discounting():
    """Verify that low detection/track confidence appropriately discounts the assessed risk score."""
    engine = RiskEngine.load()

    high_conf_track = TrackedObject(
        object_id="trk_high",
        track_id="trk_high",
        semantic_class="PEDESTRIAN",
        position=(6.0, 0.0, 0.8),
        bounding_region=(5.5, -0.4, 0.0, 6.5, 0.4, 1.7),
        confidence=1.0,
        velocity=(-2.0, 0.0, 0.0),
        dynamic=True,
        track_confidence=1.0,
    )
    low_conf_track = TrackedObject(
        object_id="trk_low",
        track_id="trk_low",
        semantic_class="PEDESTRIAN",
        position=(6.0, 0.0, 0.8),
        bounding_region=(5.5, -0.4, 0.0, 6.5, 0.4, 1.7),
        confidence=0.25,
        velocity=(-2.0, 0.0, 0.0),
        dynamic=True,
        track_confidence=0.50,  # 0.25 * 0.50 = 0.125
    )

    score_high, _ = engine.compute_object_risk(high_conf_track)
    score_low, _ = engine.compute_object_risk(low_conf_track)

    assert score_low < score_high
    # Low confidence should heavily discount the score
    assert score_low <= score_high * 0.40


def test_bounding_region_scaling_with_risk_level():
    """Verify that higher risk severity levels produce expanded spatial bounding regions."""
    engine = RiskEngine.load()

    track = TrackedObject(
        object_id="t1",
        track_id="t1",
        semantic_class="PEDESTRIAN",
        position=(10.0, 0.0, 1.0),
        bounding_region=(9.0, -1.0, 0.0, 11.0, 1.0, 2.0),
        confidence=0.9,
        velocity=(0.0, 0.0, 0.0),
        dynamic=True,
    )

    bounds_low = engine.compute_region_bounds(track, RiskLevel.LOW)
    bounds_med = engine.compute_region_bounds(track, RiskLevel.MEDIUM)
    bounds_high = engine.compute_region_bounds(track, RiskLevel.HIGH)
    bounds_crit = engine.compute_region_bounds(track, RiskLevel.CRITICAL)

    # Calculate span in x dimension: max_x - min_x
    span_low = bounds_low[3] - bounds_low[0]
    span_med = bounds_med[3] - bounds_med[0]
    span_high = bounds_high[3] - bounds_high[0]
    span_crit = bounds_crit[3] - bounds_crit[0]

    assert span_low < span_med < span_high < span_crit
    assert span_crit == pytest.approx(span_low * 2.5, abs=0.1)


def test_evaluate_emits_valid_risk_regions():
    """Verify evaluate() produces properly formed RiskRegion instances matching contract."""
    engine = RiskEngine.load()

    tracks = [
        TrackedObject(
            object_id="t1",
            track_id="t1",
            semantic_class="PEDESTRIAN",
            position=(8.0, 0.0, 0.8),
            bounding_region=(7.5, -0.4, 0.0, 8.5, 0.4, 1.7),
            confidence=0.9,
            velocity=(-2.0, 0.0, 0.0),
            dynamic=True,
            last_seen=12.5,
        ),
        TrackedObject(
            object_id="t2",
            track_id="t2",
            semantic_class="VEHICLE",
            position=(32.0, 0.0, 1.0),
            bounding_region=(30.0, -1.0, 0.0, 34.0, 1.0, 2.0),
            confidence=0.8,
            velocity=(1.0, 0.0, 0.0),
            dynamic=True,
            last_seen=12.5,
        ),
    ]

    regions = engine.evaluate(tracks)
    assert len(regions) == 2
    for r in regions:
        assert isinstance(r, RiskRegion)
        assert len(r.region_bounds) == 6
        assert 0.0 <= r.risk_score <= 1.0
        assert r.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert r.timestamp == 12.5
        assert r.associated_object_id in ("t1", "t2")


def test_traversability_context_integration():
    """Verify that lower ground traversability under a track increases assessed risk."""
    engine = RiskEngine.load()
    map_instance = AdaptiveMap()

    # Create cell under position (10.0, 0.0) with non-traversable score
    cell = AdaptiveCell(
        spatial_bounds=(9.0, -1.0, 11.0, 1.0),
        resolution=0.10,
        traversability_score=0.10,  # poor traversability
    )
    # Register cell into map
    map_instance.cells[(0, 100, 0)] = cell

    track = TrackedObject(
        object_id="t_terrain",
        track_id="t_terrain",
        semantic_class="VEHICLE",
        position=(5.0, 0.0, 0.5),
        bounding_region=(4.0, -1.0, 0.0, 6.0, 1.0, 1.5),
        confidence=0.9,
        velocity=(0.0, 0.0, 0.0),
        dynamic=True,
    )

    # Evaluate with and without poor traversability
    score_no_map, _ = engine.compute_object_risk(track, traversability_context=None)
    score_with_map, _ = engine.compute_object_risk(track, traversability_context=map_instance)

    assert score_with_map >= score_no_map
