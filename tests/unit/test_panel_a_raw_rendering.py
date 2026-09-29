"""Unit tests for Panel A Raw Point Cloud Rendering requirements."""

from pathlib import Path
import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.pipeline import FoveaPipeline
from dashboard.server import serialize_frame_message


def test_empty_road_static_scene_no_jitter():
    """Verify that points in empty_road static scene do not jitter or drift frame-to-frame."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("empty_road", num_frames=5)

    assert len(frames) == 5
    assert len(gt["frames"]) == 5

    # Check frame 0 vs all subsequent frames: coordinates must be identical
    pts0 = frames[0].points
    assert pts0.shape[0] > 0

    for idx in range(1, len(frames)):
        pts_i = frames[idx].points
        assert pts0.shape == pts_i.shape, f"Frame {idx} shape mismatch"
        # Spatial coordinates (X, Y, Z) must be exactly identical
        np.testing.assert_array_equal(
            pts0[:, :3],
            pts_i[:, :3],
            err_msg=f"Frame {idx} point coordinates drifted from Frame 0 (jitter bug)",
        )


def test_empty_road_serialization_intensity_range():
    """Verify that raw_intensities are normalized cleanly across [0.0, 1.0] without wash-out."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("empty_road", num_frames=3)

    pipeline = FoveaPipeline()
    result = pipeline.process_frame(frames[0])

    payload = serialize_frame_message(result, "empty_road", 3)
    assert payload["type"] == "frame"
    assert "raw_points" in payload
    assert "raw_intensities" in payload

    raw_pts = payload["raw_points"]
    raw_ints = payload["raw_intensities"]

    assert len(raw_pts) > 0
    assert len(raw_pts) == len(raw_ints)

    # Check intensities strictly in [0.0, 1.0]
    for val in raw_ints:
        assert isinstance(val, (float, int))
        assert 0.0 <= val <= 1.0, f"Intensity {val} out of bounds [0.0, 1.0]"

    # Check that intensities span a non-trivial range (not all 0 or all 1)
    min_int = min(raw_ints)
    max_int = max(raw_ints)
    assert max_int > min_int, "Intensities are completely flat"


def test_surround_360_city_coverage_and_orientation():
    """Verify surround_360_city scenario produces omnidirectional returns with proper sensor frame axes."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("surround_360_city", num_frames=3)

    assert len(frames) == 3
    pts = frames[0].points
    assert pts.shape[0] > 1000

    # Sensor frame axes check:
    # +X is forward (positive x), -X is rear (negative x)
    # +Y is left (positive y), -Y is right (negative y)
    has_fwd = np.any(pts[:, 0] > 10.0)
    has_rear = np.any(pts[:, 0] < -10.0)
    has_left = np.any(pts[:, 1] > 5.0)
    has_right = np.any(pts[:, 1] < -5.0)

    assert has_fwd, "Missing forward returns (+X)"
    assert has_rear, "Missing rear returns (-X)"
    assert has_left, "Missing left returns (+Y)"
    assert has_right, "Missing right returns (-Y)"

    # Process through pipeline and serialize
    pipeline = FoveaPipeline()
    res = pipeline.process_frame(frames[0])
    payload = serialize_frame_message(res, "surround_360_city", 3)

    raw_pts = payload["raw_points"]
    raw_ints = payload["raw_intensities"]
    assert len(raw_pts) > 0
    assert len(raw_pts) == len(raw_ints)

    for val in raw_ints:
        assert 0.0 <= val <= 1.0


def test_frontend_panel_a_assets_and_scale_axis_elements():
    """Verify frontend HTML, CSS, and JS include required Panel A scale and axis indicator elements."""
    root = Path(__file__).resolve().parents[2]

    html_path = root / "dashboard" / "frontend" / "index.html"
    css_path = root / "dashboard" / "frontend" / "styles.css"
    js_path = root / "dashboard" / "frontend" / "app.js"

    assert html_path.exists()
    assert css_path.exists()
    assert js_path.exists()

    html_content = html_path.read_text(encoding="utf-8")
    css_content = css_path.read_text(encoding="utf-8")
    js_content = js_path.read_text(encoding="utf-8")

    # HTML requirements
    assert 'id="canvas-raw"' in html_content
    assert 'id="panel-a-container"' in html_content
    assert 'id="btn-toggle-panel-a"' in html_content
    assert '+X (Fwd)' in html_content
    assert '+Y (Left)' in html_content
    assert '10m' in html_content

    # CSS requirements
    assert '.panel-a-container' in css_content
    assert '#canvas-raw' in css_content
    assert '.axis-badge.x-axis' in css_content
    assert '.axis-badge.y-axis' in css_content
    assert '.scale-bar' in css_content

    # JS requirements
    assert 'drawPanelARawPointCloud' in js_content
    # Orthographic BEV projection formula: v = v0 - x * scale, u = u0 - y * scale
    assert 'u0 - y * scale' in js_content
    assert 'v0 - x * scale' in js_content
    # Axis and scale canvas drawings
    assert '"+X"' in js_content
    assert '"+Y"' in js_content
    assert '"10m"' in js_content
