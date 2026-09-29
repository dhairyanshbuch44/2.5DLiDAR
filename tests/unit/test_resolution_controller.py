"""Unit tests for ResolutionController and FoveaOverlay."""

import numpy as np
import pytest

from fovea_lidar.contracts.mapping import AdaptiveCell, AdaptiveMap
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.mapping.fovea_overlay import FoveaOverlay
from fovea_lidar.mapping.grid_engine import AdaptiveGridEngine
from fovea_lidar.resolution.controller import FoveationContext, ResolutionController


# ==============================================================================
# ResolutionController Mathematical Formula & Logic Tests
# ==============================================================================

def test_far_low_complexity_road_produces_coarse_resolution():
    """Verify that far, low-complexity ROAD region yields coarse resolution close to R_max."""
    controller = ResolutionController.load()

    # Distance = 45m (far field near max_range=50m), ROAD semantic class, low complexity
    road_importance = controller.semantic_importance_table["ROAD"]
    res = controller.calculate_resolution(
        distance=45.0,
        scene_complexity=0.0,
        semantic_importance=road_importance,
        dynamic_state=False,
    )

    # With R_min=0.05, R_max=0.40, alpha=0.007, d=45m gives R_dist=0.365m
    assert res > 0.30
    assert res <= controller.max_resolution
    assert pytest.approx(0.35, abs=0.05) == res


def test_pedestrian_produces_measurably_finer_resolution_than_empty_region():
    """Verify core Example 2 behavior: tracked pedestrian yields significantly finer resolution."""
    controller = ResolutionController.load()

    test_distance = 20.0  # 20m from sensor

    # 1. Equivalent empty / baseline region at 20m
    res_empty = controller.calculate_resolution(
        distance=test_distance,
        scene_complexity=0.0,
        semantic_importance=0.0,
        dynamic_state=False,
    )

    # 2. Region containing tracked moving pedestrian at 20m
    ped_importance = controller.semantic_importance_table["PEDESTRIAN"]
    res_pedestrian = controller.calculate_resolution(
        distance=test_distance,
        scene_complexity=0.0,
        semantic_importance=ped_importance,
        dynamic_state=True,
    )

    print(f"\nResolution at d=20m: Empty={res_empty:.4f}m, Pedestrian={res_pedestrian:.4f}m")

    # Assert that pedestrian resolution is strictly and measurably finer (smaller cell size)
    assert res_pedestrian < res_empty
    improvement = res_empty - res_pedestrian
    assert improvement >= 0.08, f"Expected at least 0.08m finer resolution, got {improvement:.4f}m"
    assert res_pedestrian == pytest.approx(controller.min_resolution, abs=0.03)


def test_dynamic_state_boosts_refinement():
    """Verify dynamic=True produces finer resolution than dynamic=False for the same class and distance."""
    controller = ResolutionController.load()

    res_static_veh = controller.calculate_resolution(
        distance=25.0,
        semantic_importance=0.90,
        dynamic_state=False,
    )
    res_dynamic_veh = controller.calculate_resolution(
        distance=25.0,
        semantic_importance=0.90,
        dynamic_state=True,
    )

    assert res_dynamic_veh <= res_static_veh


def test_scene_complexity_boosts_refinement():
    """Verify that elevated scene complexity refines grid resolution."""
    controller = ResolutionController.load()

    res_flat = controller.calculate_resolution(
        distance=30.0,
        scene_complexity=0.0,
        semantic_importance=0.10,
    )
    res_complex = controller.calculate_resolution(
        distance=30.0,
        scene_complexity=0.80,
        semantic_importance=0.10,
    )

    assert res_complex < res_flat


def test_controller_clamping_bounds():
    """Verify resolution is strictly clamped within [R_min, R_max]."""
    controller = ResolutionController(min_resolution=0.05, max_resolution=0.40)

    # Extremely far distance
    res_far = controller.calculate_resolution(distance=500.0)
    assert res_far == controller.max_resolution

    # Zero distance with maximal stimulus
    res_near = controller.calculate_resolution(
        distance=0.0,
        scene_complexity=1.0,
        semantic_importance=1.0,
        dynamic_state=True,
    )
    assert res_near == controller.min_resolution


def test_compute_context_with_tracked_objects():
    """Verify compute() generates per-track region resolutions."""
    controller = ResolutionController.load()

    track_ped = TrackedObject(
        object_id="track_1",
        track_id="track_1",
        semantic_class="PEDESTRIAN",
        position=(15.0, 0.0, 0.9),
        bounding_region=(14.5, -0.5, 0.0, 15.5, 0.5, 1.8),
        confidence=0.90,
        velocity=(0.0, 0.0, 0.0),
        dynamic=True,
    )
    track_veh = TrackedObject(
        object_id="track_2",
        track_id="track_2",
        semantic_class="VEHICLE",
        position=(35.0, 5.0, 1.0),
        bounding_region=(33.0, 4.0, 0.0, 37.0, 6.0, 2.0),
        confidence=0.95,
        velocity=(0.0, 3.0, 0.0),
        dynamic=True,
    )

    context = FoveationContext(tracked_objects=[track_ped, track_veh])
    resolutions = controller.compute(context)

    assert "fovea_track_1" in resolutions
    assert "fovea_track_2" in resolutions
    assert resolutions["fovea_track_1"] < 0.10
    assert resolutions["fovea_track_2"] < 0.20


# ==============================================================================
# FoveaOverlay Sub-Grid and GridEngine Isolation Tests
# ==============================================================================

def test_fovea_overlay_subgrid_mechanics():
    """Verify FoveaOverlay point containment and local sub-grid cell creation."""
    overlay = FoveaOverlay(
        overlay_id="fovea_test",
        center=(15.0, 2.0),
        radius=2.0,
        resolution=0.05,
    )
    assert overlay.contains_point(15.0, 2.0) is True
    assert overlay.contains_point(16.5, 2.0) is True
    assert overlay.contains_point(20.0, 2.0) is False  # outside radius

    cell = overlay.get_or_create_cell(col=300, row=40)
    assert cell.resolution == 0.05
    assert cell.spatial_bounds == (15.0, 2.0, 15.05, 2.05)


def test_grid_engine_overlay_precedence_and_isolation():
    """Verify fovea overlay takes precedence during projection, and base cells are isolated."""
    engine = AdaptiveGridEngine()

    # 1. Without overlay: points at (15.0, 0.0) land in mid_range band 1 (res=0.10m)
    pts = np.array([[15.02, 0.02, 0.5]], dtype=np.float32)
    proj_base = engine.project_points(pts)
    key_base = list(proj_base.keys())[0]
    assert key_base[0] == 1  # band 1
    assert engine.resolution_for_band(1) == 0.10

    # 2. Add fine fovea overlay at (15.0, 0.0) with resolution 0.05m
    overlay = FoveaOverlay(
        overlay_id="fovea_1",
        center=(15.0, 0.0),
        radius=2.0,
        resolution=0.05,
    )
    engine.add_fovea_overlay(overlay)

    # 3. Project same point: must now route into fovea overlay cell
    proj_fovea = engine.project_points(pts)
    key_fovea = list(proj_fovea.keys())[0]
    assert key_fovea[0] == "fovea_1"  # overlay took precedence
    cell_fovea = engine.get_or_create_cell(key_fovea)
    assert cell_fovea.resolution == 0.05

    # 4. Far road point at (35.0, 0.0) remains in coarse base band 2 (res=0.25m) unaffected
    far_pt = np.array([[35.0, 0.0, 0.0]], dtype=np.float32)
    proj_far = engine.project_points(far_pt)
    key_far = list(proj_far.keys())[0]
    assert key_far[0] == 2  # base band 2
    assert engine.resolution_for_band(2) == 0.25

    # 5. Remove overlay: points at (15.0, 0.0) cleanly resume base band projection with zero orphaned state
    engine.remove_fovea_overlay("fovea_1")
    assert "fovea_1" not in engine._fovea_overlays
    assert all(k[0] != "fovea_1" for k in engine._cells)

    proj_resumed = engine.project_points(pts)
    key_resumed = list(proj_resumed.keys())[0]
    assert key_resumed[0] == 1  # resumed base band 1
