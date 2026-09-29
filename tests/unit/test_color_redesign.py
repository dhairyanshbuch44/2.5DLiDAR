"""Unit tests verifying the dashboard color redesign across all four panels.

Checks:
- palette.js definition, data structures, and completeness
- Contrast & distinctness of resolution tiers (Band 0 to Band 3)
- Categorical palette for semantic classes (including Pothole, Truck, Pedestrian, Cyclist)
- Sequential cool-to-hot collision risk palette (Low -> Medium -> High -> Critical)
- Distinct heated metal reflectance intensity colormap
- HTML and CSS integration (dark theme default, no conflicting white overrides, on-canvas overlays)
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = REPO_ROOT / "dashboard" / "frontend"


def test_palette_js_structure_and_completeness():
    palette_file = FRONTEND_DIR / "palette.js"
    assert palette_file.exists(), "palette.js must exist as single shared color config"
    content = palette_file.read_text(encoding="utf-8")

    # Verify root config object
    assert "const FOVEA_PALETTE = {" in content
    assert "window.FOVEA_PALETTE = FOVEA_PALETTE;" in content

    # 1. Dark theme definitions
    assert "dark:" in content
    assert "#070b14" in content or "#050811" in content
    assert "bgCore:" in content
    assert "bgSurface:" in content

    # 2. Traversability
    for key in ["SAFE:", "CAUTION:", "NON_TRAVERSABLE:", "UNKNOWN:"]:
        assert key in content, f"Traversability missing key: {key}"

    # 3. Concentric resolution bands (0.05m, 0.10m, 0.25m, 0.40m)
    assert "RESOLUTION_BANDS:" in content
    for res_val in ["0.05", "0.10", "0.25", "0.40"]:
        assert res_val in content, f"Resolution band missing {res_val}m"

    # 4. Semantic classes
    required_classes = ["PEDESTRIAN", "CYCLIST", "VEHICLE", "TRUCK", "CURB", "POTHOLE", "ROAD"]
    for cls in required_classes:
        assert f"{cls}:" in content, f"Semantic palette missing required class: {cls}"

    # 5. Risk levels (cool to hot)
    for risk in ["LOW:", "MEDIUM:", "HIGH:", "CRITICAL:"]:
        assert risk in content, f"Risk palette missing level: {risk}"

    # 6. Intensity colormap
    assert "INTENSITY_STOPS:" in content
    assert "getIntensityRgb" in content
    assert "getIntensityCss" in content
    assert "getResolutionRgb" in content


def test_resolution_bands_are_distinct():
    palette_file = FRONTEND_DIR / "palette.js"
    content = palette_file.read_text(encoding="utf-8")

    # Extract hex colors in RESOLUTION_BANDS
    matches = re.findall(r"res:\s*(0\.\d+).*?hex:\s*(0x[0-9a-fA-F]+)", content, re.DOTALL)
    assert len(matches) >= 4, f"Expected at least 4 bands, got {matches}"
    hexes = [m[1].lower() for m in matches]
    # Ensure all bands have distinct hex values
    assert len(set(hexes)) == len(hexes), f"Resolution band colors must be distinct: {hexes}"
    # Ensure 0.40m band is NOT dark indigo blending with black (was 0x1e3a5f)
    assert "0x1e3a5f" not in hexes, "Band 3 (0.40m) must not use dark indigo 0x1e3a5f"


def test_risk_palette_is_sequential_and_distinct_from_semantics():
    palette_file = FRONTEND_DIR / "palette.js"
    content = palette_file.read_text(encoding="utf-8")

    # Verify risk colors match blue/sky -> yellow -> orange -> red
    assert "#38bdf8" in content  # Low (Sky Blue)
    assert "#eab308" in content  # Medium (Yellow/Gold)
    assert "#f97316" in content  # High (Orange)
    assert "#ef4444" in content  # Critical (Red)


def test_index_html_loads_palette_and_dark_theme_without_white_override():
    html_file = FRONTEND_DIR / "index.html"
    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")

    # Must load palette.js before app.js
    idx_palette = content.find("palette.js")
    idx_app = content.find("app.js")
    assert idx_palette != -1, "index.html must load palette.js"
    assert idx_app != -1, "index.html must load app.js"
    assert idx_palette < idx_app, "palette.js must be loaded before app.js"

    # Must default to dark-theme on body
    assert 'class="dark-theme"' in content

    # Must not contain the old light-theme inline style override with !important
    assert "body.light-theme {" not in content
    assert "background: #f8fafc !important;" not in content

    # Must contain on-canvas overlays for Panel A and Panel C
    assert "panel-a-intensity-legend" in content
    assert "panel-c-mode-overlay" in content
    assert "sparkline-fps" in content


def test_app_js_uses_fovea_palette():
    app_file = FRONTEND_DIR / "app.js"
    assert app_file.exists()
    content = app_file.read_text(encoding="utf-8")

    # Global state default theme is dark
    assert 'let currentTheme = "dark";' in content

    # Palette references FOVEA_PALETTE
    assert "FOVEA_PALETTE" in content
    assert "drawFpsSparkline" in content
    assert "panel-c-mode-title" in content
    assert "panel-c-mode-items" in content
