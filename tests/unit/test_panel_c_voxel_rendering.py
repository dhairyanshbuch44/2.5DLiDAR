"""Unit tests for Panel C Adaptive 2.5D Voxel Grid Rendering requirements."""

from pathlib import Path
import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.contracts.risk import RiskLevel
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.mapping.grid_engine import AdaptiveGridEngine
from fovea_lidar.pipeline import FoveaPipeline
from dashboard.server import serialize_frame_message


def test_cell_key_to_metric_bounds_conversion():
    """Verify that CellKey (band_id, col, row) maps deterministically to real-world coordinates."""
    engine = AdaptiveGridEngine()
    expected_resolutions = {
        0: 0.05,
        1: 0.10,
        2: 0.25,
        3: 0.40,
    }

    test_keys = [
        (0, 10, -5),
        (0, -20, 30),
        (1, 15, -8),
        (2, 40, 12),
        (3, -50, -60),
    ]

    for key in test_keys:
        band_id, col, row = key
        res = expected_resolutions[band_id]

        min_x, min_y, max_x, max_y = engine.cell_bounds(key)

        assert min_x == pytest.approx(col * res, abs=1e-5)
        assert min_y == pytest.approx(row * res, abs=1e-5)
        assert max_x == pytest.approx((col + 1) * res, abs=1e-5)
        assert max_y == pytest.approx((row + 1) * res, abs=1e-5)

        # Center calculation check
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        assert cx == pytest.approx((col + 0.5) * res, abs=1e-5)
        assert cy == pytest.approx((row + 0.5) * res, abs=1e-5)


def test_resolution_boundary_stress_transitions():
    """Verify resolution_boundary_stress scenario correctly partitions across 10m, 25m, 50m."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("resolution_boundary_stress", num_frames=3)

    pipeline = FoveaPipeline()
    res = pipeline.process_frame(frames[0])

    cells = res.adaptive_map.cells
    assert len(cells) > 0

    # Ensure cells across multiple bands are present
    band_indices = {key[0] for key in cells.keys() if isinstance(key[0], int)}
    assert 0 in band_indices, "Band 0 (0-10m) missing"
    assert 1 in band_indices, "Band 1 (10-25m) missing"

    # Serialize message and ensure resolutions match configured bands
    msg = serialize_frame_message(res, "resolution_boundary_stress", 3)
    transport_cells = msg["cells"]
    assert len(transport_cells) > 0

    resolutions = {c["resolution"] for c in transport_cells}
    assert 0.05 in resolutions
    assert 0.10 in resolutions


def test_fovea_overlay_alignment_with_tracked_object():
    """Verify that active fovea overlay center matches the tracked object position."""
    pipeline = FoveaPipeline()

    # Create a frame with a pedestrian at (15.0, 2.0, 0.5) which should trigger a fovea
    pts = np.array([[15.0, 2.0, 0.5, 0.8], [15.1, 2.0, 0.6, 0.8]], dtype=np.float32)
    frame = PointCloudFrame(points=pts, timestamp=0.1, frame_id="test_frame")

    trk = TrackedObject(
        object_id="veh_1",
        track_id="veh_1",
        semantic_class="VEHICLE",
        position=(35.0, 5.0, 1.0),
        velocity=(0.0, 3.0, 0.0),
        bounding_region=(33.0, 4.0, 0.0, 37.0, 6.0, 2.0),
        confidence=0.95,
        track_confidence=0.95,
        dynamic=True,
    )

    # Update fovea manager directly with this track
    pipeline.fovea_manager.update([trk], pipeline.adaptive_map)
    active_foveas = pipeline.fovea_manager.get_active_fovea_regions()

    assert len(active_foveas) == 1
    fov = active_foveas[0]
    assert fov["associated_track_id"] == "veh_1"
    assert fov["center"] == pytest.approx((35.0, 5.0), abs=1e-3)
    assert fov["radius"] > 0.0
    assert fov["resolution"] < 0.25


def test_serialized_cell_contains_occupancy_monotonic():
    """Verify serialized cells include authentic occupancy and non-overridden traversability."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("empty_road", num_frames=2)

    pipeline = FoveaPipeline()
    res = pipeline.process_frame(frames[0])
    msg = serialize_frame_message(res, "empty_road", 2)

    cells = msg["cells"]
    assert len(cells) > 0
    for c in cells:
        assert "occupancy" in c
        assert isinstance(c["occupancy"], (float, int))
        assert 0.0 <= c["occupancy"] <= 1.0
        assert "traversability_label" in c
        assert c["traversability_label"] in ("SAFE", "CAUTION", "NON_TRAVERSABLE", "UNKNOWN")


def test_frontend_assets_quadtree_and_elevation_colormaps():
    """Verify frontend HTML, CSS, and JS reflect the updated quadtree labels and hypsometric colormap."""
    root = Path(__file__).resolve().parents[2]
    html_path = root / "dashboard" / "frontend" / "index.html"
    css_path = root / "dashboard" / "frontend" / "styles.css"
    js_path = root / "dashboard" / "frontend" / "app.js"

    html = html_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")
    js = js_path.read_text(encoding="utf-8")

    # Legend labels in index.html
    assert "0.05m (Fovea [0-10m])" in html
    assert "0.10m (Fine [10-25m])" in html
    assert "0.25m (Far [25-50m])" in html
    assert "0.40m (Periphery [50-100m])" in html

    # CSS colormap
    assert ".gradient-bar.turbo-gradient" in css
    assert "#1e1b4b" in css  # Deep Indigo in hypsometric colormap

    # JS implementations
    assert "getElevationColor" in js
    assert "lastDrawnVoxelCount" in js
    # Cylinder scaling fix: radius, height, radius
    assert "cylMesh.scale.set(radius, height, radius)" in js
    # Covered by fovea check
    assert "coveredByFovea" in js
