"""Integration tests for dynamic traversability evaluation within the closed-loop perception pipeline."""

import pytest

from fovea_lidar.contracts.mapping import AdaptiveCell, AdaptiveMap, TraversabilityMap
from fovea_lidar.contracts.pipeline_result import PipelineResult
from fovea_lidar.contracts.risk import RiskLevel
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.pipeline import FoveaPipeline
from fovea_lidar.terrain.analysis import TerrainClass
from fovea_lidar.traversability.engine import TraversabilityEngine


def test_vehicle_crossing_dynamic_occupancy_revert_10_frames():
    """Run the standard vehicle_crossing scenario across 10 frames.

    Verify that cells along the path start/end as SAFE, transition to CAUTION
    when overlapping the moving vehicle, and naturally revert to SAFE when the vehicle passes.
    ego_speed=0.0 and original initial_position are passed so cell keys remain
    at the pre-ego-motion positions these assertions were calibrated against.
    """
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate(
        "vehicle_crossing",
        num_frames=10,
        initial_position=(20.0, -8.0, 0.75),
        ego_speed=0.0,
    )

    pipeline = FoveaPipeline()

    # Track two complementary cells along the vehicle trajectory:
    # 1. key_departed: vehicle begins over it (y=-8.5) and moves away
    key_departed = (1, 180, -85)  # bounds ~ (18.0, -8.5, 18.1, -8.4)
    # 2. key_approached: vehicle approaches and arrives at (y=-4.3)
    key_approached = (1, 200, -43)  # bounds ~ (20.0, -4.3, 20.1, -4.2)

    departed_labels = []
    departed_scores = []
    approached_labels = []
    approached_scores = []

    for f_idx, frame in enumerate(frames):
        result = pipeline.process_frame(frame)
        assert isinstance(result, PipelineResult)
        tmap = result.traversability_map
        assert isinstance(tmap, TraversabilityMap)

        # 1. Query departed cell
        lbl_dep = tmap.get_label(key_departed)
        sc_dep = tmap.get_score(key_departed)
        departed_labels.append(lbl_dep.value if lbl_dep else None)
        departed_scores.append(sc_dep)

        # 2. Query approached cell
        lbl_app = tmap.get_label(key_approached)
        sc_app = tmap.get_score(key_approached)
        approached_labels.append(lbl_app.value if lbl_app else None)
        approached_scores.append(sc_app)

    print(f"\n[10 Frames] Departed Cell {key_departed} Labels: {departed_labels}")
    print(f"[10 Frames] Departed Cell {key_departed} Scores: {departed_scores}")
    print(f"[10 Frames] Approached Cell {key_approached} Labels: {approached_labels}")
    print(f"[10 Frames] Approached Cell {key_approached} Scores: {approached_scores}")

    # Assert departed cell: starts CAUTION in frames 0..1, reverts to SAFE in frames 2..9
    assert departed_labels[0] == "CAUTION"
    assert departed_labels[1] == "CAUTION"
    for f in range(2, 10):
        assert departed_labels[f] == "SAFE", (
            f"Frame {f}: expected reverted SAFE, got {departed_labels[f]}"
        )
        assert departed_scores[f] == pytest.approx(1.0, abs=1e-3)

    # Assert approached cell: starts SAFE in frames 0..7, becomes CAUTION in frames 8..9
    for f in range(0, 8):
        assert approached_labels[f] == "SAFE", (
            f"Frame {f}: expected SAFE before arrival, got {approached_labels[f]}"
        )
        assert approached_scores[f] == pytest.approx(1.0, abs=1e-3)
    assert approached_labels[8] == "CAUTION"
    assert approached_labels[9] == "CAUTION"


def test_vehicle_crossing_full_cycle_safe_caution_safe_16_frames():
    """Run 16 frames of vehicle_crossing to verify the complete 3-stage lifecycle for a single cell.

    SAFE (before arrival) -> CAUTION / NON_TRAVERSABLE (during overlap) -> SAFE (after departure).
    ego_speed=0.0 and original initial_position are passed so cell keys remain
    at the pre-ego-motion positions these assertions were calibrated against.
    """
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate(
        "vehicle_crossing",
        num_frames=16,
        initial_position=(20.0, -8.0, 0.75),
        ego_speed=0.0,
    )

    pipeline = FoveaPipeline()
    target_key = (1, 200, -43)

    observed_labels = []
    observed_scores = []

    for f_idx, frame in enumerate(frames):
        result = pipeline.process_frame(frame)
        tmap = result.traversability_map
        assert tmap is not None

        lbl = tmap.get_label(target_key)
        sc = tmap.get_score(target_key)
        observed_labels.append(lbl.value if lbl else None)
        observed_scores.append(sc)

    print(f"\n[16 Frames] Target Cell {target_key} Lifecycle:")
    for f_idx, (l, s) in enumerate(zip(observed_labels, observed_scores)):
        print(f"  Frame {f_idx:02d}: label={l:<15} score={s:.2f}")

    # Stage 1: Before arrival (Frames 0..7) -> strictly SAFE
    for f in range(0, 8):
        assert observed_labels[f] == "SAFE"
        assert observed_scores[f] == pytest.approx(1.0, abs=1e-3)

    # Stage 2: Vehicle overlapping (Frames 8..13) -> CAUTION or NON_TRAVERSABLE
    for f in range(8, 14):
        assert observed_labels[f] in ("CAUTION", "NON_TRAVERSABLE")
        assert observed_scores[f] <= 0.50

    # Stage 3: Vehicle passed (Frames 14..15) -> naturally reverted to SAFE
    for f in range(14, 16):
        assert observed_labels[f] == "SAFE"
        assert observed_scores[f] == pytest.approx(1.0, abs=1e-3)



def test_traversability_overlay_zero_persistent_mutation():
    """Verify that dynamic evaluation never mutates the underlying persistent AdaptiveCell.traversability_score."""
    engine = TraversabilityEngine()

    cell_key = (1, 100, 0)
    cell = AdaptiveCell(
        spatial_bounds=(9.5, -0.5, 10.5, 0.5),
        resolution=0.10,
        traversability_score=1.0,
        semantic_class="ROAD",
        semantic_confidence=0.90,
    )
    adaptive_map = AdaptiveMap(cells={cell_key: cell})

    # Initially static cell score is 1.0
    assert cell.traversability_score == 1.0

    # Frame 1: dynamic track overlaps cell
    track_crit = TrackedObject(
        object_id="trk_obs",
        track_id="trk_obs",
        semantic_class="VEHICLE",
        position=(10.0, 0.0, 0.5),
        bounding_region=(9.0, -1.0, 0.0, 11.0, 1.0, 1.5),
        confidence=0.90,
        velocity=(0.0, 3.5, 0.0),
        dynamic=True,
    )
    track_crit.risk_level = RiskLevel.CRITICAL  # type: ignore[attr-defined]

    tmap_f1 = engine.evaluate(adaptive_map, [track_crit])

    # Overlay reports NON_TRAVERSABLE, but persistent cell remains untouched
    assert tmap_f1.get_label(cell_key) == TerrainClass.NON_TRAVERSABLE
    assert tmap_f1.get_score(cell_key) == 0.0
    assert cell.traversability_score == 1.0  # MUST remain 1.0

    # Frame 2: track moves away to (50.0, 50.0)
    track_away = TrackedObject(
        object_id="trk_obs",
        track_id="trk_obs",
        semantic_class="VEHICLE",
        position=(50.0, 50.0, 0.5),
        bounding_region=(49.0, 49.0, 0.0, 51.0, 51.0, 1.5),
        confidence=0.90,
        velocity=(0.0, 3.5, 0.0),
        dynamic=True,
    )
    tmap_f2 = engine.evaluate(adaptive_map, [track_away])

    # Next frame's TraversabilityMap naturally reverts to SAFE without cleanup
    assert tmap_f2.get_label(cell_key) == TerrainClass.SAFE
    assert tmap_f2.get_score(cell_key) == pytest.approx(1.0, abs=1e-3)
    assert cell.traversability_score == 1.0
