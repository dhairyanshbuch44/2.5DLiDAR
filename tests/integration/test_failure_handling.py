"""Integration tests for end-to-end failure handling, error containment, and resilience.

Covers all failure modes in Project Brief Section 45:
- Dropped/missing frames in sequence replay (temporal gap resilience)
- Corrupt / malformed point cloud file rejection at loader level
- Missing point cloud dataset file handling
- Empty point cloud frame (0 points) processing
- Unimplemented backend ("pretrained") rejection at construction time
- Malformed config / missing required section / invalid numeric field types
- WebSocket client mid-stream disconnect resilience
- CPU-only execution verification (no torch / cupy / GPU imports or dependencies)
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Any, Dict
from fastapi.testclient import TestClient
import numpy as np
import pytest

from dashboard.server import app
from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame
from fovea_lidar.io.loaders import CorruptFileError, PointCloudFileNotFoundError, load_pcd
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.pipeline import FoveaPipeline
from fovea_lidar.utils.config import Config, ConfigError, load_config, validate_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_dropped_or_missing_frames_sequence_resilience():
    """Verify dropped frames with temporal gaps (e.g., dt=0.4s) do not crash pipeline or corrupt state."""
    pipeline = FoveaPipeline()
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("pedestrian_at_distance", num_frames=6)

    # Frame 0 at t=0.0
    res0 = pipeline.process_frame(frames[0])
    assert res0.frame_index == 0

    # Frame 1 at t=0.1
    res1 = pipeline.process_frame(frames[1])
    assert res1.frame_index == 1

    # Simulate dropped frames 2 and 3: next frame is frame 4 at t=0.4s (gap of 0.3s)
    res4 = pipeline.process_frame(frames[4])
    assert res4.frame_index == 2
    assert res4.timestamp == frames[4].timestamp
    assert res4.adaptive_map is not None
    assert res4.metrics["total_ms"] > 0.0


def test_corrupt_point_cloud_file_rejected_at_loader_level():
    """Verify corrupt/malformed PCD files are rejected with CorruptFileError before pipeline invocation."""
    with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as f:
        f.write(b"# .PCD v.7 - Corrupted PCD file\n")
        f.write(b"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n")
        f.write(b"POINTS 100\nDATA ascii\n")
        f.write(b"1.0 2.0\n")  # Incomplete tokens, only 2 values instead of 3
        temp_path = Path(f.name)

    try:
        with pytest.raises(CorruptFileError):
            load_pcd(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_missing_point_cloud_file_raises_not_found():
    """Verify attempting to load a non-existent point cloud raises PointCloudFileNotFoundError."""
    missing_path = REPO_ROOT / "data" / "non_existent_scan_12345.pcd"
    with pytest.raises(PointCloudFileNotFoundError):
        load_pcd(missing_path)


def test_empty_point_cloud_frame_handled_gracefully():
    """Verify an empty point cloud frame (0 points) flows through the entire pipeline without crash."""
    pipeline = FoveaPipeline()
    empty_pts = np.zeros((0, 4), dtype=np.float32)
    empty_frame = PointCloudFrame(
        points=empty_pts,
        timestamp=0.5,
        frame_id="lidar_top",
        source=FrameSource.MOCK,
        metadata={"test": "empty_frame"},
    )

    result = pipeline.process_frame(empty_frame)

    assert result is not None
    assert result.cleaned_frame.points.shape[0] == 0
    assert len(result.semantic_points) == 0
    assert len(result.tracked_objects) == 0
    assert len(result.active_fovea_regions) == 0
    assert result.adaptive_map is not None
    assert result.metrics["total_ms"] > 0.0


def test_pretrained_backend_request_raises_not_implemented():
    """Verify requesting perception.backend='pretrained' raises NotImplementedError at pipeline init."""
    base_cfg = load_config().to_dict()
    base_cfg["perception"]["backend"] = "pretrained"

    with pytest.raises(NotImplementedError) as exc_info:
        FoveaPipeline(config=base_cfg)

    assert "backend 'pretrained' is not yet implemented" in str(exc_info.value)


def test_invalid_config_missing_required_section_raises_config_error():
    """Verify config missing a required top-level section raises ConfigError before component construction."""
    invalid_cfg = {
        "mapping": {"min_resolution": 0.05, "max_resolution": 0.40, "max_range": 50.0, "min_range": 0.5},
        # Missing 'resolution_zones', 'semantic_foveation', 'risk_refinement', 'uncertainty_refinement'
    }

    with pytest.raises(ConfigError) as exc_info:
        FoveaPipeline(config=invalid_cfg)

    assert "missing required top-level section" in str(exc_info.value)


def test_invalid_config_wrong_numeric_type_raises_config_error():
    """Verify config with non-numeric value for numeric parameter raises ConfigError."""
    base_cfg = load_config().to_dict()
    base_cfg["mapping"]["min_resolution"] = "ultra_fine"  # String instead of float/int

    with pytest.raises(ConfigError) as exc_info:
        FoveaPipeline(config=base_cfg)

    assert "must be a numeric value" in str(exc_info.value)


def test_dashboard_websocket_client_disconnect_resilience():
    """Verify WebSocket client disconnecting mid-stream is handled gracefully without server crash."""
    client = TestClient(app)

    # Connect client 1, receive 1 frame, then disconnect
    with client.websocket_connect("/ws/stream?scenario=flagship_demo") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "frame"
        assert msg["scenario"] == "flagship_demo"

    # Immediately connect client 2 to verify server remains healthy and accepts new streams
    with client.websocket_connect("/ws/stream?scenario=flagship_demo") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "frame"
        assert msg["scenario"] == "flagship_demo"
        assert msg["frame_index"] == 0


@pytest.mark.cpu_only
def test_cpu_only_execution_no_gpu_or_torch_loaded():
    """Verify the entire pipeline and demo path execute with zero GPU or deep learning library dependencies."""
    # Ensure neither torch nor GPU-accelerated libraries are imported
    assert "torch" not in sys.modules, "torch was unexpectedly imported!"
    assert "cupy" not in sys.modules, "cupy was unexpectedly imported!"
    assert "tensorflow" not in sys.modules, "tensorflow was unexpectedly imported!"

    pipeline = FoveaPipeline(config=REPO_ROOT / "configs" / "demo.yaml")
    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate("flagship_demo", num_frames=3)

    for frame in frames:
        res = pipeline.process_frame(frame)
        assert res.adaptive_map is not None

    # Re-verify no deep learning runtime was pulled into memory
    assert "torch" not in sys.modules
    assert "cupy" not in sys.modules
