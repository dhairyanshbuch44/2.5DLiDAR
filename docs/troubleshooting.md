# Fovea-LiDAR Troubleshooting & Failure Handling Guide

This guide documents the common failure modes, error signals, root causes, and recommended recovery procedures across the Fovea-LiDAR perception pipeline, adhering to Project Brief Section 45.

---

## Quick Diagnostic Index

| Failure Mode / Exception | Component | Root Cause | Immediate Fix |
|---|---|---|---|
| [`CorruptFileError`](#1-corrupt-or-malformed-point-cloud-files) | `io.loaders` | Truncated file or malformed header | Regenerate or re-export PCD/PLY scan |
| [`PointCloudFileNotFoundError`](#2-missing-dataset-or-scenario-files) | `io.loaders` | Missing point cloud file path | Run synthetic generation script |
| [`EmptyPointCloudError`](#3-empty-point-cloud-frames) | `io.loaders` | 0-point file loaded | Inspect sensor FOV or occlusion |
| [`ConfigError: missing required section`](#4-missing-or-invalid-configuration-sections) | `utils.config` | Incomplete YAML configuration | Inherit from `configs/default.yaml` |
| [`ConfigError: must be a numeric value`](#5-numeric-parameter-type-mismatches) | `utils.config` | Non-numeric or invalid parameter | Check numeric types in YAML |
| [`NotImplementedError: backend 'pretrained'`](#6-unsupported-perception-backend-requested) | `perception.registry` | DL backend requested in MVP | Use `backend: "mock"` |
| [`WebSocketDisconnect`](#7-dashboard-websocket-disconnections) | `dashboard.server` | Browser closed or reloaded | Normal lifecycle; reconnect browser |
| [Temporal Jumps / Dropped Frames](#8-dropped-or-missing-frames-in-sequences) | `tracking.tracker` | Sensor packet loss / frame skip | Automatic Kalman coasting handles gaps |

---

## 1. Corrupt or Malformed Point Cloud Files

### Symptom
```text
fovea_lidar.io.loaders.CorruptFileError: Truncated header in PCD file: data/samples/synthetic/...
fovea_lidar.io.loaders.CorruptFileError: Data size mismatch in ASCII PCD: expected 500 lines, got 212
```

### Root Cause
A point cloud file (`.pcd` or `.ply`) was truncated, suffered an interrupted write, or contains non-standard tokens violating the PCD/PLY specification.

### Error Containment
The error is caught and raised immediately inside `load_pcd()` or `load_ply()`. The pipeline is **never** invoked on partial or corrupted memory buffers.

### Resolution
Regenerate synthetic test datasets:
```bash
python scripts/generate_synthetic_scenarios.py
```
For hardware recordings, verify sensor capture packet streams and ensure recording threads flush before termination.

---

## 2. Missing Dataset or Scenario Files

### Symptom
```text
fovea_lidar.io.loaders.PointCloudFileNotFoundError: PCD file not found: data/samples/synthetic/flagship_demo/frame_0000.pcd
```

### Root Cause
The requested scenario directory has not been populated with pre-generated scan frames.

### Resolution
Run the dataset synthesizer to populate all 8 canonical scenarios:
```bash
python scripts/generate_synthetic_scenarios.py
```

---

## 3. Empty Point Cloud Frames

### Symptom
A point cloud scan contains zero points (`points.shape == (0, 4)`), caused by sensor blindout, occlusion, or aggressive range gating.

### Pipeline Behavior
- **Preprocessor**: Detects `input_count == 0`, logs a warning (`Preprocessor received empty input frame (0 points)`), and bypasses filtering without raising an exception.
- **Perception & Object Extraction**: Returns empty candidate lists (`[]`).
- **Adaptive Map**: Detects empty point cloud, updates timestamps, and leaves existing grid cells unchanged.
- **Result**: `FoveaPipeline.process_frame()` returns an empty-but-valid `PipelineResult` with all metrics measured.

---

## 4. Missing or Invalid Configuration Sections

### Symptom
```text
fovea_lidar.utils.config.ConfigError: Configuration is missing required top-level section(s): mapping, resolution_zones
```

### Root Cause
A custom configuration YAML or dictionary omitted one of the five mandatory architectural sections:
1. `mapping`
2. `resolution_zones`
3. `semantic_foveation`
4. `risk_refinement`
5. `uncertainty_refinement`

### Error Containment
Validated by `validate_config()` at pipeline initialization time before any submodules (grid engine, tracker, risk engine) are constructed.

### Resolution
Ensure your configuration file inherits from or includes all top-level keys present in `configs/default.yaml`.

---

## 5. Numeric Parameter Type Mismatches

### Symptom
```text
fovea_lidar.utils.config.ConfigError: Configuration field 'mapping.min_resolution' must be a numeric value, got str ('ultra_fine')
fovea_lidar.utils.config.ConfigError: Configuration field 'semantic_foveation.fovea_radius' must be positive, got -1.0
```

### Root Cause
A configuration field expecting a positive floating-point number received a string, boolean, or non-positive value.

### Resolution
Check `configs/default.yaml` or `configs/demo.yaml` for syntax errors:
```yaml
mapping:
  min_resolution: 0.05       # Valid float
  # min_resolution: "0.05m"  # INVALID: String with unit
```

---

## 6. Unsupported Perception Backend Requested

### Symptom
```text
NotImplementedError: Segmentation backend 'pretrained' is not yet implemented — deep learning inference models will be integrated in Advanced Prompt A1.
```

### Root Cause
The active configuration specified `perception.backend: "pretrained"` or `perception.backend: "custom"`.

### Design Rationale
Per project brief specification, Prompts 1-18 comprise the complete CPU-based MVP pipeline featuring heuristic rule-based semantic segmentation (`MockModel`). Deep learning networks (PointNet++, SqueezeSeg) are scheduled for Advanced Prompts.

### Resolution
Set the perception backend to `"mock"` in your config:
```yaml
perception:
  backend: "mock"
```

---

## 7. Dashboard WebSocket Disconnections

### Symptom
In server console logs:
```text
2026-09-04 22:58:30 [INFO] [fovea_dashboard]: WebSocket client disconnected.
```

### Server Behavior
The FastAPI server handles `WebSocketDisconnect` cleanly:
- Closes the active frame streaming loop.
- Cancels pending asynchronous coroutines.
- Invokes `session.pipeline.reset()` in the `finally` block to release occupied grid cells and tracker state from memory.
- The server remains operational and ready for subsequent client connections.

---

## 8. Dropped or Missing Frames in Sequences

### Symptom
LiDAR replay skips frames (e.g. frame 0, frame 1, then frame 5 with timestamp gap $\Delta t = 0.4\,	ext{s}$).

### Pipeline Behavior
- `MultiObjectTracker` computes $\Delta t = \max(10^{-4}, t - t_{	ext{last}})$.
- The constant-velocity Kalman filter advances state predictions forward by the actual elapsed duration $\Delta t$.
- Tracks coast across missed observations and update covariance matrices appropriately.
- If frames continue to be missed beyond `tracking.max_missed_frames` (default: 3 frames), the track transitions cleanly to `TrackState.REMOVED`.

---

## 9. CPU vs. GPU Runtime Verification

### Design Principle
The entire Fovea-LiDAR MVP pipeline is strictly CPU-executable:
- Point cloud filtering (spatial bounds, voxel grids) uses vectorised NumPy operations.
- Obstacle clustering uses SciPy KD-Tree neighbor graphs (`cKDTree.query_pairs`).
- Multi-object tracking runs analytic discrete Kalman filters.
- Real-time dashboard renders via standard HTML5 2D Canvas.

### Verification
Run the CPU-only test suite:
```bash
pytest -v -m cpu_only tests/integration/test_failure_handling.py
```
Asserts that `torch`, `cupy`, and `tensorflow` are **not** loaded in memory and no GPU device context is initialized.
