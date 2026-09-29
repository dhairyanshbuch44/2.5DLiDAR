"""End-to-end integration tests for Fovea-LiDAR Flagship Demo and Evaluation."""

from __future__ import annotations

import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from dashboard.server import app
from fovea_lidar.contracts.risk import RiskLevel
from fovea_lidar.terrain.analysis import TerrainClass
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.pipeline import FoveaPipeline
from fovea_lidar.utils.config import load_config
from scripts.demo import run_benchmark_comparison

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_demo_pipeline_composition_with_demo_config():
    """Verify full pipeline initializes with configs/demo.yaml and processes flagship_demo scenario."""
    demo_cfg_path = REPO_ROOT / "configs" / "demo.yaml"
    assert demo_cfg_path.exists(), "configs/demo.yaml must exist"

    cfg = load_config(demo_cfg_path)
    assert cfg.get("semantic_foveation.fovea_radius") == 3.0
    assert cfg.get("semantic_foveation.enabled") is True
    assert cfg.get("risk_refinement.enabled") is True
    assert cfg.get("uncertainty_refinement.enabled") is True

    pipeline = FoveaPipeline(config=cfg)
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt_data = gen.generate("flagship_demo", num_frames=16)

    assert len(frames) == 16
    assert gt_data["scenario"] == "flagship_demo"

    for idx, frame in enumerate(frames):
        result = pipeline.process_frame(frame)
        assert result.frame_index == idx
        assert result.timestamp == frame.timestamp
        assert result.adaptive_map is not None
        assert len(result.adaptive_map.cells) > 0
        assert result.traversability_map is not None
        assert result.metrics is not None
        assert result.metrics["total_ms"] > 0.0


def test_flagship_demo_narrative_beats():
    """Verify all narrative beats from Brief Section 90 unfold in precise sequence.

    ego_speed=0.0 and original params are passed so the narrative beat timing
    (ped_enter_frame=3, ped_exit_frame=11, num_frames=16) matches this test's
    calibrated assertions exactly.
    """
    demo_cfg_path = REPO_ROOT / "configs" / "demo.yaml"
    pipeline = FoveaPipeline(config=demo_cfg_path)
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate(
        "flagship_demo",
        num_frames=16,
        ped_start_pos=(45.0, 2.5, 0.9),
        ped_velocity=(-15.0, -3.0, 0.0),
        ped_enter_frame=3,
        ped_exit_frame=11,
        ego_speed=0.0,
    )

    results = [pipeline.process_frame(f) for f in frames]

    # Beat 1: Initial frames (0..2) empty road, 0 foveas, coarse far baseline (0.25m)
    for f_idx in (0, 1, 2):
        res = results[f_idx]
        assert len(res.active_fovea_regions) == 0, f"Frame {f_idx} should have 0 foveas"
        # Check far field near (45m, 2.5m)
        far_cells = [
            c for c in res.adaptive_map.cells.values()
            if 40.0 <= c.spatial_bounds[0] <= 50.0 and 0.0 <= c.spatial_bounds[1] <= 5.0
        ]
        if far_cells:
            # Baseline far field band target resolution is 0.25m
            assert all(c.resolution >= 0.20 for c in far_cells)

    # Beat 2: Frame 3 pedestrian enters at ~45m, active fovea overlay tracks them
    res3 = results[3]
    assert len(res3.tracked_objects) == 1
    assert len(res3.active_fovea_regions) == 1
    fovea3 = res3.active_fovea_regions[0]
    center_dist = math.hypot(fovea3["center"][0] - 45.0, fovea3["center"][1] - 2.5)
    assert center_dist < 3.0, f"Fovea should be centered near pedestrian, got {fovea3['center']}"
    assert fovea3["radius"] == 3.0
    assert fovea3["resolution"] < 0.20  # Refined resolution

    # Beat 3: Pedestrian crosses caution zone / risk escalated
    elevated_risk_observed = False
    caution_cells_observed = False
    for f_idx in range(4, 11):
        res = results[f_idx]
        if any(r.risk_level.value in ("HIGH", "CRITICAL", "MEDIUM") for r in res.risk_regions):
            elevated_risk_observed = True
        caution_count = sum(
            1 for lbl in res.traversability_map.cell_labels.values()
            if lbl.value in ("CAUTION", "NON_TRAVERSABLE")
        )
        if caution_count > 0:
            caution_cells_observed = True

    assert elevated_risk_observed, "Collision risk must be elevated as pedestrian approaches ego corridor"
    assert caution_cells_observed, "Traversability map must register caution/non-traversable cells"

    # Beat 4: Frames 14..15 pedestrian has cleared roadway, fovea overlay relaxes to 0
    res_final = results[15]
    assert len(res_final.active_fovea_regions) == 0, "Fovea count must relax back to 0 after exit and decay"

    # Beat 5: Near-field road cells remain fine (0.05m) throughout ALL frames
    for f_idx, res in enumerate(results):
        near_cells = [
            c for c in res.adaptive_map.cells.values()
            if 1.0 <= c.spatial_bounds[0] <= 8.0 and -2.0 <= c.spatial_bounds[1] <= 2.0
        ]
        assert len(near_cells) > 0, f"Near-field cells must exist in frame {f_idx}"
        assert all(c.resolution == 0.05 for c in near_cells), f"Near-field cells must be 0.05m in frame {f_idx}"



def test_demo_cli_benchmark_execution():
    """Verify scripts/demo.py --benchmark runs to completion and exits code 0."""
    status = run_benchmark_comparison(scenario="flagship_demo", num_frames=5)
    assert status == 0


def test_demo_dashboard_api_endpoints():
    """Verify FastAPI dashboard server serves index, scenarios list, and WebSocket stream."""
    client = TestClient(app)

    # 1. GET / serves HTML
    resp_index = client.get("/")
    assert resp_index.status_code == 200
    assert "Fovea-LiDAR" in resp_index.text

    # 2. GET /api/scenarios includes flagship_demo
    resp_scenarios = client.get("/api/scenarios")
    assert resp_scenarios.status_code == 200
    scenarios = resp_scenarios.json()
    assert isinstance(scenarios, list)
    assert "flagship_demo" in scenarios

    # 3. WebSocket /ws/stream with flagship_demo streams valid frames
    with client.websocket_connect("/ws/stream?scenario=flagship_demo") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "frame"
        assert msg["scenario"] == "flagship_demo"
        assert msg["frame_index"] == 0
        assert "raw_points" in msg
        assert "semantic_points" in msg
        assert "cells" in msg
        assert "active_fovea_regions" in msg
        assert "tracked_objects" in msg
        assert "metrics" in msg
