# Fovea-LiDAR Datasets Guide

This document outlines the **synthetic-data-first** methodology employed in the Fovea-LiDAR MVP and provides comprehensive guidance for ingesting real-world autonomous vehicle LiDAR datasets in future development phases.

---

## 1. Why Synthetic Data First?

Per Project Brief Section 36, developing an adaptive perception system using a synthetic-data-first architecture provides critical engineering advantages:

1. **Zero External Licensing & Compliance Hurdles**:
   Real-world automotive datasets (e.g., KITTI, nuScenes, Waymo Open) carry restrictive non-commercial licenses or demand multi-gigabyte credentialed registration. Synthetic scenarios can be cloned and run instantly by any jury or evaluator.
2. **Deterministic Mathematical Repeatability**:
   Every point return, obstacle trajectory, and terrain geometry is synthetically seeded (`seed=42`). Unit tests and benchmarks yield 100% deterministic results across different operating systems and CPU architectures.
3. **Controlled Edge-Case Synthesis**:
   Real-world datasets rarely contain specific critical failure scenarios in compact sequences. Our generator (`src/fovea_lidar/io/synthetic.py`) synthesizes targeted stress conditions:
   - *Flagship Demo*: Pedestrian entering at 45 m and crossing the ego path.
   - *Resolution Boundary Stress*: Obstacles oscillating precisely along concentric zone borders ($r = 10.0	ext{ m}, 25.0	ext{ m}$).
   - *Low-Confidence Ambiguity*: Obstacles with marginal sensor reflectivity triggering uncertainty foveation.
   - *Object Disappearing*: Transient dynamic obstacles testing Kalman track coasting and stale decay.
4. **Lightweight Distribution**:
   Full test suites and benchmarks execute locally in seconds without requiring 50 GB to 300 GB dataset downloads.

---

## 2. Canonical Synthetic Scenarios

The synthesizer generates 8 standardized scenarios (`src/fovea_lidar/io/synthetic.py`):

| Scenario Name | Description | Key Evaluation Target |
|---|---|---|
| **`flagship_demo`** | Sparse road $	o$ entering pedestrian $	o$ crossing trajectory $	o$ stop | End-to-end jury demonstration and 6-signal foveation |
| **`empty_road`** | Planar asphalt with nominal slope and boundaries | Baseline peripheral coarse resolution efficiency |
| **`pedestrian_at_distance`** | Pedestrian walking at 40–45 m distance | High-resolution fovea overlay triggering and tracking |
| **`vehicle_crossing`** | High-velocity vehicle traversing perpendicular to ego | Fast Doppler velocity estimation and approach factor |
| **`multi_object_dense`** | 4 to 8 concurrent obstacles (pedestrians, vehicles, obstacles) | Multi-target clustering, tracking, and overlay isolation |
| **`low_confidence_obstacle`** | Low-reflectivity obstacle with ambiguous classification | Uncertainty refinement and caution traversability |
| **`resolution_boundary_stress`**| Targets positioned exactly on $r=10.0	ext{ m}$ and $25.0	ext{ m}$ boundaries | Zero boundary artifact verification and cell stability |
| **`object_disappearing`** | Obstacle observed for 4 frames then occluded/removed | Track coasting, missed frames, and stale cell decay |

---

## 3. Integrating Real-World Datasets (Advanced Phase A1)

When extending Fovea-LiDAR to real sensor logs, the following datasets are recommended per Project Brief Section 36:

### A. SemanticKITTI (Recommended for 3D Semantic Segmentation)
- **Primary Use**: Training and evaluating 3D point-wise semantic segmentation backends.
- **Sensor**: 64-beam Velodyne HDL-64E ($10	ext{ Hz}$).
- **Annotations**: 28 semantic classes point-wise annotated on urban driving sequences.
- **Conversion Guide**:
  1. Read binary points: `.bin` file formatted as `float32` arrays $[x, y, z, 	ext{remission}]$.
  2. Read labels: `.label` file formatted as `uint32` containing upper 16-bit instance IDs and lower 16-bit class labels.
  3. Map semantic class IDs to Fovea-LiDAR `SemanticClass` enum in `src/fovea_lidar/contracts/semantic_classes.py`.
  4. Construct `PointCloudFrame(points, timestamp, frame_id)`.

### B. KITTI Tracking & 3D Object Detection
- **Primary Use**: Validating multi-object tracking (MOT) and dynamic collision risk estimation.
- **Annotations**: 3D bounding boxes $(x, y, z, l, w, h, 	heta)$ with persistent track IDs.
- **Calibration Note**: Requires transforming velodyne coordinate frame coordinates to rectified camera frame coordinates using $T_{	ext{velo}}^{	ext{cam}} = R_{	ext{rect}} \cdot [R | T]$.

### C. nuScenes Dataset
- **Primary Use**: Multi-modal sensor fusion and long-duration trajectory forecasting.
- **Sensor**: 32-beam roof LiDAR with 360-degree coverage.
- **Setup Burden**: Significant download footprint (~50 GB for mini, ~350 GB for full). Requires the `nuscenes-devkit` Python library.

---

## 4. Data Ingestion Architecture

The loader subsystem (`src/fovea_lidar/io/loaders.py`) supports generic file formats:
- **PCD Files**: Standard ASCII and binary Point Cloud Data format (`.pcd`).
- **PLY Files**: Stanford polygon format (`.ply`).
- **Custom Sequences**: Replay controller (`src/fovea_lidar/io/replay.py`) streams sequences from disk or memory at configurable playback speeds (`0.5x`, `1.0x`, `2.0x`).
