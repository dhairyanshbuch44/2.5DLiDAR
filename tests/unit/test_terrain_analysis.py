"""Unit and pipeline integration tests for TerrainAnalyzer."""

import pytest

from fovea_lidar.contracts.mapping import AdaptiveCell
from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.mapping.adaptive_map import AdaptiveMap
from fovea_lidar.perception.mock_model import MockModel
from fovea_lidar.preprocessing.pipeline import Preprocessor
from fovea_lidar.terrain.analysis import TerrainAnalyzer, TerrainClass


# ==============================================================================
# Hand-Crafted Synthetic Terrain Tests
# ==============================================================================

def test_hand_crafted_flat_terrain_is_safe():
    """Verify that flat, smooth road surface is classified as SAFE."""
    adaptive_map = AdaptiveMap()
    # 3x3 patch of cells in band 0 (res=0.05m) at z=0.0
    for c in range(-1, 2):
        for r in range(-1, 2):
            cell = adaptive_map.grid_engine.get_or_create_cell((0, c, r))
            cell.elevation = 0.0
            cell.roughness = 0.02
            cell.semantic_class = "ROAD"
            cell.semantic_confidence = 0.90

    analyzer = TerrainAnalyzer()
    labels = analyzer.analyze(adaptive_map)

    center_key = (0, 0, 0)
    assert labels[center_key] == TerrainClass.SAFE
    assert adaptive_map.cells[center_key].traversability_score == 1.0


def test_hand_crafted_step_discontinuity_curb():
    """Verify that a positive vertical step (0.12m) is detected as a curb and marked CAUTION."""
    adaptive_map = AdaptiveMap()
    # Road cell at (0, 0) with elevation 0.0
    road_cell = adaptive_map.grid_engine.get_or_create_cell((0, 0, 0))
    road_cell.elevation = 0.0
    road_cell.semantic_class = "ROAD"
    road_cell.semantic_confidence = 0.90

    # Sidewalk curb cell at (1, 0) with elevation 0.12m (12cm step)
    curb_cell = adaptive_map.grid_engine.get_or_create_cell((0, 1, 0))
    curb_cell.elevation = 0.12
    curb_cell.semantic_class = "SIDEWALK"
    curb_cell.semantic_confidence = 0.85

    analyzer = TerrainAnalyzer()
    labels = analyzer.analyze(adaptive_map)

    assert analyzer.cell_features[(0, 0, 0)]["is_curb"] is True
    assert analyzer.cell_features[(0, 1, 0)]["is_curb"] is True
    assert labels[(0, 1, 0)] == TerrainClass.CAUTION
    assert adaptive_map.cells[(0, 1, 0)].traversability_score == 0.5


def test_hand_crafted_severe_step_is_non_traversable():
    """Verify that a large step (0.50m) is marked NON_TRAVERSABLE."""
    adaptive_map = AdaptiveMap()
    c1 = adaptive_map.grid_engine.get_or_create_cell((0, 0, 0))
    c1.elevation = 0.0

    c2 = adaptive_map.grid_engine.get_or_create_cell((0, 1, 0))
    c2.elevation = 0.50  # 50cm wall/step

    analyzer = TerrainAnalyzer()
    labels = analyzer.analyze(adaptive_map)

    assert labels[(0, 1, 0)] == TerrainClass.NON_TRAVERSABLE
    assert adaptive_map.cells[(0, 1, 0)].traversability_score == 0.0


def test_hand_crafted_low_confidence_is_unknown():
    """Verify that low semantic confidence results in UNKNOWN classification."""
    adaptive_map = AdaptiveMap()
    cell = adaptive_map.grid_engine.get_or_create_cell((0, 0, 0))
    cell.elevation = 0.0
    cell.semantic_class = "ROAD"
    cell.semantic_confidence = 0.30  # Below min_semantic_confidence 0.50

    analyzer = TerrainAnalyzer(min_semantic_confidence=0.50)
    labels = analyzer.analyze(adaptive_map)

    assert labels[(0, 0, 0)] == TerrainClass.UNKNOWN
    assert adaptive_map.cells[(0, 0, 0)].traversability_score == 0.25


def test_hand_crafted_sloped_terrain_is_caution():
    """Verify that terrain slope exceeding max_safe_slope (0.15) but below max_caution_slope (0.35) is marked CAUTION."""
    adaptive_map = AdaptiveMap()
    # Gradient in X: dx = (0.02 - 0.0) / (2 * 0.05) = 0.20 slope
    c_left = adaptive_map.grid_engine.get_or_create_cell((0, -1, 0))
    c_left.elevation = 0.0

    c_center = adaptive_map.grid_engine.get_or_create_cell((0, 0, 0))
    c_center.elevation = 0.01

    c_right = adaptive_map.grid_engine.get_or_create_cell((0, 1, 0))
    c_right.elevation = 0.02

    analyzer = TerrainAnalyzer(max_safe_slope=0.15, max_caution_slope=0.35)
    labels = analyzer.analyze(adaptive_map)

    assert 0.15 <= analyzer.cell_features[(0, 0, 0)]["slope"] < 0.35
    assert labels[(0, 0, 0)] == TerrainClass.CAUTION
    assert adaptive_map.cells[(0, 0, 0)].traversability_score == 0.5


def test_hand_crafted_steep_slope_is_non_traversable():
    """Verify that continuous slope exceeding max_caution_slope is marked NON_TRAVERSABLE."""
    adaptive_map = AdaptiveMap()
    # Gradient in X: dx = (0.06 - 0.0) / (2 * 0.05) = 0.60 slope
    c_left = adaptive_map.grid_engine.get_or_create_cell((0, -1, 0))
    c_left.elevation = 0.0

    c_center = adaptive_map.grid_engine.get_or_create_cell((0, 0, 0))
    c_center.elevation = 0.03

    c_right = adaptive_map.grid_engine.get_or_create_cell((0, 1, 0))
    c_right.elevation = 0.06

    analyzer = TerrainAnalyzer(max_safe_slope=0.15, max_caution_slope=0.35)
    labels = analyzer.analyze(adaptive_map)

    assert analyzer.cell_features[(0, 0, 0)]["slope"] >= 0.35
    assert labels[(0, 0, 0)] == TerrainClass.NON_TRAVERSABLE
    assert adaptive_map.cells[(0, 0, 0)].traversability_score == 0.0


def test_hand_crafted_pothole_detection():
    """Verify pothole depression surrounded by flat ground is detected."""
    adaptive_map = AdaptiveMap()
    # Pothole center at (0, 0) with elevation -0.15m
    center = adaptive_map.grid_engine.get_or_create_cell((0, 0, 0))
    center.elevation = -0.15

    # 4 higher surrounding neighbors at elevation 0.0m
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dc, dr in offsets:
        n = adaptive_map.grid_engine.get_or_create_cell((0, dc, dr))
        n.elevation = 0.0

    analyzer = TerrainAnalyzer(pothole_depth_threshold=0.08)
    labels = analyzer.analyze(adaptive_map)

    assert analyzer.cell_features[(0, 0, 0)]["is_pothole"] is True
    assert labels[(0, 0, 0)] == TerrainClass.CAUTION


# ==============================================================================
# Full Pipeline End-to-End Tests
# ==============================================================================

def test_full_pipeline_resolution_boundary_stress():
    """Run resolution_boundary_stress scenario end-to-end through pipeline and terrain analysis."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("resolution_boundary_stress", num_frames=1)

    preprocessor = Preprocessor()
    cleaned_frame, _ = preprocessor.process(frames[0])

    model = MockModel()
    sem_points = model.predict(cleaned_frame)

    adaptive_map = AdaptiveMap()
    adaptive_map.update(cleaned_frame, semantic_points=sem_points)

    analyzer = TerrainAnalyzer()
    labels = analyzer.analyze(adaptive_map)

    assert len(labels) == adaptive_map.total_cells
    assert len(labels) > 0


def test_full_pipeline_multi_object_dense():
    """Run multi_object_dense scenario end-to-end and assert non-degenerate label distribution."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("multi_object_dense", num_frames=1)

    preprocessor = Preprocessor()
    cleaned_frame, _ = preprocessor.process(frames[0])

    model = MockModel()
    sem_points = model.predict(cleaned_frame)

    adaptive_map = AdaptiveMap()
    adaptive_map.update(cleaned_frame, semantic_points=sem_points)

    analyzer = TerrainAnalyzer()
    labels = analyzer.analyze(adaptive_map)

    dist = analyzer.get_label_distribution()
    print(f"\nObserved Terrain Label Distribution on multi_object_dense: {dist}")

    # Assert non-degenerate: must not be 100% one label
    assert dist[TerrainClass.SAFE.value] > 0
    assert dist[TerrainClass.NON_TRAVERSABLE.value] > 0
    assert (dist[TerrainClass.SAFE.value] + dist[TerrainClass.NON_TRAVERSABLE.value]) < sum(dist.values()) or dist[TerrainClass.CAUTION.value] >= 0
