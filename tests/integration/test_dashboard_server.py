"""Integration tests for FastAPI dashboard WebSocket streaming and REST endpoints."""

import pytest
from fastapi.testclient import TestClient

from dashboard.server import app

client = TestClient(app)


def test_get_scenarios_endpoint():
    """Verify GET /api/scenarios discovers and returns non-empty synthetic scenarios list."""
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    scenarios = response.json()

    assert isinstance(scenarios, list)
    assert len(scenarios) > 0
    assert "multi_object_dense" in scenarios
    assert "empty_road" in scenarios
    assert "vehicle_crossing" in scenarios


def test_get_index_html():
    """Verify GET / serves dashboard index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Fovea-LiDAR" in response.text
    assert "canvas-raw" in response.text
    assert "canvas-map" in response.text


def test_websocket_stream_receives_valid_frames():
    """Verify WebSocket /ws/stream emits schema-valid frames with non-empty, authentic pipeline outputs."""
    with client.websocket_connect("/ws/stream?scenario=multi_object_dense") as ws:
        received_frames = []

        # Read 3 consecutive frames
        for _ in range(3):
            msg = ws.receive_json()
            received_frames.append(msg)

        assert len(received_frames) == 3

        for f_idx, msg in enumerate(received_frames):
            assert msg["type"] == "frame"
            assert msg["scenario"] == "multi_object_dense"
            assert msg["frame_index"] == f_idx
            assert msg["timestamp"] >= 0.0

            # 1. Raw points transport verification
            raw_pts = msg["raw_points"]
            assert isinstance(raw_pts, list)
            assert len(raw_pts) > 0
            assert len(raw_pts[0]) == 3  # [x, y, z]

            # 2. Semantic points transport verification
            sem_pts = msg["semantic_points"]
            assert isinstance(sem_pts, list)
            assert len(sem_pts) > 0
            assert "class" in sem_pts[0]
            assert "confidence" in sem_pts[0]

            # 3. Tracked objects
            tracks = msg["tracked_objects"]
            assert isinstance(tracks, list)
            assert len(tracks) >= 1
            for trk in tracks:
                assert "track_id" in trk
                assert "position" in trk
                assert "velocity" in trk
                assert "bounding_region" in trk
                assert len(trk["bounding_region"]) >= 6

            # 4. Active fovea regions
            foveas = msg["active_fovea_regions"]
            assert isinstance(foveas, list)

            # 5. Adaptive cells and traversability view
            cells = msg["cells"]
            assert isinstance(cells, list)
            assert len(cells) > 0
            for c in cells[:10]:
                assert "bounds" in c
                assert "traversability_label" in c
                assert c["traversability_label"] in (
                    "SAFE",
                    "CAUTION",
                    "NON_TRAVERSABLE",
                    "UNKNOWN",
                )
                assert 0.0 <= c["traversability_score"] <= 1.0

            # 6. Performance Telemetry metrics
            metrics = msg["metrics"]
            assert metrics["total_ms"] > 0.0
            assert metrics["fps"] > 0.0
            assert "preprocessing_ms" in metrics
            assert "perception_ms" in metrics
            assert "terrain_ms" in metrics
            assert "mapping_ms" in metrics

            # 7. Entity counts
            counts = msg["counts"]
            assert counts["raw_points"] > 0
            assert counts["total_cells"] > 0


def test_websocket_playback_controls():
    """Verify WebSocket control messages (pause, seek, speed, resume, select_layer) alter stream state."""
    with client.websocket_connect("/ws/stream?scenario=multi_object_dense") as ws:
        # 1. Initial frame 0
        msg0 = ws.receive_json()
        assert msg0["frame_index"] == 0

        # 2. Pause stream
        ws.send_json({"type": "pause"})

        # 3. Seek to frame 5
        ws.send_json({"type": "seek", "frame": 5})

        # 4. Resume stream
        ws.send_json({"type": "resume"})

        # Next streamed frame must be frame 5
        msg_seek = ws.receive_json()
        assert msg_seek["frame_index"] == 5

        # 5. Change speed to 2.0x
        ws.send_json({"type": "speed", "value": 2.0})

        # Next frame is frame 6
        msg_next = ws.receive_json()
        assert msg_next["frame_index"] == 6

        # 6. Layer selection
        ws.send_json({"type": "select_layer", "layers": ["cells", "foveas"]})


def test_websocket_resilience_and_disconnect():
    """Verify that corrupt control messages and unexpected scenarios do not crash the server."""
    with client.websocket_connect("/ws/stream?scenario=vehicle_crossing") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "frame"

        # Send malformed JSON message
        ws.send_text("MALFORMED_NON_JSON_STRING")

        # Stream must survive and continue emitting subsequent frames
        msg_next = ws.receive_json()
        assert msg_next["type"] == "frame"
