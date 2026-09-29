# Fovea-LiDAR Public API & Interface Reference

This document provides a comprehensive technical reference for all public data contracts, algorithmic modules, pipeline interfaces, network protocols, and REST endpoints across the Fovea-LiDAR perception system.

---

## Table of Contents
1. [Core Data Contracts (`fovea_lidar.contracts`)](#1-core-data-contracts)
2. [Point Cloud Preprocessing (`fovea_lidar.preprocessing`)](#2-point-cloud-preprocessing)
3. [Semantic Perception & Registry (`fovea_lidar.perception`)](#3-semantic-perception--registry)
4. [Object Extraction (`fovea_lidar.objects`)](#4-object-extraction)
5. [Multi-Object Tracking (`fovea_lidar.tracking`)](#5-multi-object-tracking)
6. [Dynamic Risk & Uncertainty (`fovea_lidar.risk`)](#6-dynamic-risk--uncertainty)
7. [Resolution Controller & Foveation (`fovea_lidar.resolution`)](#7-resolution-controller--foveation)
8. [Adaptive Grid & 2.5D Mapping (`fovea_lidar.mapping`)](#8-adaptive-grid--25d-mapping)
9. [Terrain & Traversability Engines (`fovea_lidar.traversability`)](#9-terrain--traversability-engines)
10. [Closed-Loop Pipeline (`fovea_lidar.pipeline`)](#10-closed-loop-pipeline)
11. [Real-Time Dashboard API & WebSocket Protocol](#11-real-time-dashboard-api--websocket-protocol)

---

## 1. Core Data Contracts

All data contracts are structured using standard Python `@dataclass` definitions and provide `.to_dict()` serialization methods.

### `PointCloudFrame` (`contracts.point_cloud`)
Represents an unorganized or organized 3D LiDAR point cloud scan.
```python
@dataclass
class PointCloudFrame:
    points: np.ndarray          # Shape (N, 3) or (N, 4) [X, Y, Z, Intensity], float32
    timestamp: float            # Capture time in seconds (s)
    frame_id: int               # Monotonically increasing scan sequence number
    sensor_id: str = "lidar_0"  # Identifier of the originating sensor
```

### `SemanticPoint` & `SemanticClass` (`contracts.semantic`, `contracts.semantic_classes`)
Represents a classified 3D LiDAR point return.
```python
class SemanticClass(str, Enum):
    UNLABELED = "UNLABELED"
    OUTLIER = "OUTLIER"
    ROAD = "ROAD"
    SIDEWALK = "SIDEWALK"
    BUILDING = "BUILDING"
    WALL = "WALL"
    FENCE = "FENCE"
    POLE = "POLE"
    TRAFFIC_SIGN = "TRAFFIC_SIGN"
    VEGETATION = "VEGETATION"
    TERRAIN = "TERRAIN"
    PEDESTRIAN = "PEDESTRIAN"
    CYCLIST = "CYCLIST"
    VEHICLE = "VEHICLE"
    TRUCK = "TRUCK"
    CURB = "CURB"
    POTHOLE = "POTHOLE"
    UNKNOWN_OBSTACLE = "UNKNOWN_OBSTACLE"

@dataclass(frozen=True)
class SemanticPoint:
    coordinates: Tuple[float, float, float]  # (x, y, z) in sensor frame
    intensity: float                         # Reflectivity return
    semantic_class: SemanticClass            # Class enum label
    confidence: float                        # Prediction probability [0.0, 1.0]
    original_index: int                      # Index in original PointCloudFrame
```

### `DetectedObject` (`contracts.objects`)
Object-level candidate extracted from point clustering prior to temporal association.
```python
@dataclass
class DetectedObject:
    object_id: str                              # Provisional per-frame UUID
    label: SemanticClass                        # Dominant semantic classification
    confidence: float                           # Aggregate confidence in [0.0, 1.0]
    bounding_box: Tuple[float, float, float,    # Axis-aligned box:
                        float, float, float]    # (min_x, min_y, min_z, max_x, max_y, max_z)
    centroid: Tuple[float, float, float]        # 3D spatial center (x, y, z)
    point_count: int                            # Number of constituent LiDAR points
    is_dynamic: bool = False                    # Initial static vs dynamic classification
    velocity: Optional[Tuple[float, float, float]] = None  # Provisional frame velocity
```

### `TrackedObject` (`contracts.tracking`)
Temporally stable obstacle track maintaining Kalman-filtered kinematics.
```python
@dataclass
class TrackedObject:
    track_id: int                               # Persistent integer track identifier
    label: SemanticClass                        # Track semantic classification
    confidence: float                           # Track confidence score
    bounding_box: Tuple[float, float, float,    # Filtered 3D bounding box
                        float, float, float]
    centroid: Tuple[float, float, float]        # Filtered centroid position (x, y, z)
    point_count: int                            # Observation point count
    velocity: Tuple[float, float, float]        # Estimated 3D velocity vector (vx, vy, vz) in m/s
    age: int                                    # Total frames tracked
    missed_frames: int                          # Consecutive frames coasted without association
    state: str = "confirmed"                    # Lifecycle: "tentative", "confirmed", "deleted"
```

### `AdaptiveCell` (`contracts.mapping`)
Multi-resolution 2.5D grid cell materialization.
```python
@dataclass
class AdaptiveCell:
    cell_key: Tuple[Union[int, str], int, int]  # (band_idx_or_overlay_id, col, row)
    band_index: Union[int, str]                 # Annular zone index or overlay identifier
    col: int                                    # Grid column coordinate floor(x / R)
    row: int                                    # Grid row coordinate floor(y / R)
    resolution: float                           # Cell edge length in meters (m)
    elevation: float                            # Robust median elevation (z) in meters
    elevation_variance: float                   # Surface variance (m^2)
    roughness: float                            # Surface roughness std dev (m)
    occupancy_probability: float                # Occupancy belief [0.0, 0.99]
    point_count: int                            # Cumulative point observations
    semantic_class: Optional[str]               # Dominant semantic label
    semantic_confidence: float                  # Winning confidence score [0.0, 1.0]
    last_updated: float                         # Timestamp of most recent observation
    is_fovea: bool = False                      # True if belonging to high-res fovea overlay
```

### `RiskRegion` & `RiskLevel` (`contracts.risk`)
Spatial hazard envelope generated for collision avoidance.
```python
class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskRegion:
    region_id: str                              # Identifier ("risk_trk_{id}")
    track_id: int                               # Target TrackedObject identifier
    center: Tuple[float, float, float]          # Projected center (x, y, z)
    bounds: Tuple[float, float, float, float]   # Expanded 2D bounds (min_x, min_y, max_x, max_y)
    risk_score: float                           # Composite collision risk [0.0, 1.0]
    risk_level: RiskLevel                       # Discrete threat level
    approaching: bool                           # True if moving towards ego sensor
    heading: float                              # Velocity heading in radians
    time_to_closest_approach: float             # Estimated TTC in seconds
```

### `PipelineResult` & `PipelineMetrics` (`contracts.pipeline_result`, `contracts.metrics`)
Unified per-frame output returned by `FoveaPipeline.process_frame()`.
```python
@dataclass
class PipelineResult:
    frame_id: int
    timestamp: float
    raw_frame: PointCloudFrame
    cleaned_frame: PointCloudFrame
    semantic_points: List[SemanticPoint]
    detected_objects: List[DetectedObject]
    tracked_objects: List[TrackedObject]
    risk_regions: List[RiskRegion]
    adaptive_map: Any                           # Instance of AdaptiveMap
    traversability_map: Optional[Any]           # Instance of TraversabilityMap
    metrics: PipelineMetrics
```

---

## 2. Point Cloud Preprocessing

**Module**: `fovea_lidar.preprocessing.pipeline.Preprocessor`

### Class: `Preprocessor`
Executes spatial filtering, downsampling, and ground candidate separation.

```python
class Preprocessor:
    def __init__(self, config: Optional[Union[Config, dict, Path, str]] = None) -> None: ...
    def process(self, frame: PointCloudFrame) -> PreprocessingOutput: ...
```

#### Output Container: `PreprocessingOutput`
- `cleaned_frame: PointCloudFrame`: Point cloud after voxel downsampling and outlier removal.
- `ground_candidates: PointCloudFrame`: Extracted ground-level returns for terrain analysis.
- `non_ground_points: PointCloudFrame`: Elevated obstacle returns for perception and clustering.
- `metrics: Dict[str, float]`: Preprocessing execution timings and filter reduction counts.

#### Individual Filter Stages (`preprocessing.stages`)
- `passthrough_range(points: np.ndarray, min_range: float, max_range: float) -> np.ndarray`
- `passthrough_height(points: np.ndarray, min_z: float, max_z: float) -> np.ndarray`
- `voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray`
- `statistical_outlier_removal(points: np.ndarray, k_neighbors: int, std_ratio: float) -> np.ndarray`
- `radius_outlier_removal(points: np.ndarray, min_neighbors: int, radius: float) -> np.ndarray`
- `extract_ground_candidates(points: np.ndarray, height_threshold: float) -> Tuple[np.ndarray, np.ndarray]`

---

## 3. Semantic Perception & Registry

**Module**: `fovea_lidar.perception`

### Base Class: `SegmentationModel` (`perception.base`)
```python
class SegmentationModel(ABC):
    @abstractmethod
    def predict(self, frame: PointCloudFrame) -> List[SemanticPoint]:
        # Perform 3D point-wise semantic segmentation
        ...
```

### Factory: `build_segmentation_model` (`perception.registry`)
```python
def build_segmentation_model(
    backend: str = "mock",
    scenario_name: Optional[str] = None,
    seed: int = 42,
    **kwargs: Any
) -> SegmentationModel:
    # Supported MVP backends: 'mock'.
    # Raises NotImplementedError if 'pretrained' is requested.
    ...
```

---

## 4. Object Extraction

**Module**: `fovea_lidar.objects.extraction.ObjectExtractor`

### Class: `ObjectExtractor`
Converts classified semantic points into discrete `DetectedObject` instances via Euclidean distance clustering.

```python
class ObjectExtractor:
    def __init__(
        self,
        cluster_distance: float = 0.8,
        min_cluster_size: int = 5,
        max_cluster_size: int = 2000,
        target_classes: Optional[Sequence[str]] = None,
    ) -> None: ...

    def extract(self, semantic_points: List[SemanticPoint]) -> List[DetectedObject]: ...
```

---

## 5. Multi-Object Tracking

**Module**: `fovea_lidar.tracking`

### Class: `ConstantVelocityKalmanFilter` (`tracking.kalman`)
Linear Kalman filter maintaining 6-DOF kinematic state $(\mathbf{x}, \mathbf{v}) = (x, y, z, v_x, v_y, v_z)$.
```python
class ConstantVelocityKalmanFilter:
    def __init__(
        self,
        initial_pos: Tuple[float, float, float],
        initial_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        q_pos: float = 0.05,
        q_vel: float = 0.20,
        r_pos: float = 0.15,
    ) -> None: ...

    def predict(self, dt: float) -> Tuple[float, float, float]: ...
    def update(self, measurement: Tuple[float, float, float]) -> None: ...
    @property
    def position(self) -> Tuple[float, float, float]: ...
    @property
    def velocity(self) -> Tuple[float, float, float]: ...
```

### Class: `MultiObjectTracker` (`tracking.tracker`)
Executes nearest-neighbor tracking with coasting and stale object decay.
```python
class MultiObjectTracker:
    def __init__(
        self,
        max_coasting_frames: int = 3,
        distance_threshold: float = 2.0,
    ) -> None: ...

    def update(
        self,
        detections: List[DetectedObject],
        timestamp: float
    ) -> List[TrackedObject]: ...

    def reset(self) -> None: ...
```

---

## 6. Dynamic Risk & Uncertainty

**Module**: `fovea_lidar.risk`

### Class: `RiskEngine` (`risk.engine`)
Evaluates dynamic collision risk $K \in [0.0, 1.0]$ based on distance, class hazard, speed, and approach angle.
```python
class RiskEngine:
    def __init__(
        self,
        config: Optional[Union[Config, dict, Path, str]] = None,
        weights: Optional[Dict[str, float]] = None,
        max_distance: float = 40.0,
        max_speed: float = 10.0,
    ) -> None: ...

    def evaluate(
        self,
        tracked_objects: List[TrackedObject],
        ego_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[RiskRegion]: ...
```

---

## 7. Resolution Controller & Foveation

**Module**: `fovea_lidar.resolution`

### Class: `ResolutionController` (`resolution.controller`)
Computes continuous target resolution from the six-signal foveation formula.
```python
class ResolutionController:
    def __init__(
        self,
        min_resolution: float = 0.05,
        max_resolution: float = 0.40,
        alpha: float = 0.007,
        gamma: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
        risk_refinement_enabled: bool = True,
        uncertainty_refinement_enabled: bool = True,
        risk_engine: Optional[RiskEngine] = None,
    ) -> None: ...

    def calculate_resolution(
        self,
        distance: float,
        scene_complexity: float = 0.0,
        semantic_importance: float = 0.0,
        dynamic_state: bool = False,
        uncertainty: float = 0.0,
        risk: float = 0.0,
    ) -> float: ...
```

### Class: `FoveaManager` (`resolution.fovea_manager`)
Orchestrates active high-resolution subgrid overlay life-cycles.
```python
class FoveaManager:
    def update_overlays(
        self,
        tracked_objects: List[TrackedObject],
        risk_regions: List[RiskRegion],
        resolution_controller: ResolutionController,
    ) -> List[FoveaOverlay]: ...
```

---

## 8. Adaptive Grid & 2.5D Mapping

**Module**: `fovea_lidar.mapping`

### Class: `AdaptiveGridEngine` (`mapping.grid_engine`)
Concentric distance band spatial index with overlay subgrids.
```python
class AdaptiveGridEngine:
    def __init__(
        self,
        config: Optional[Union[Config, dict, Path, str]] = None,
        out_of_bounds_policy: str = "reject",
    ) -> None: ...

    def project_points(self, points: np.ndarray) -> Dict[CellKey, List[int]]: ...
    def cell_index(self, x: float, y: float) -> Optional[CellKey]: ...
    def get_or_create_cell(self, key: CellKey) -> AdaptiveCell: ...
    def add_fovea_overlay(self, overlay: FoveaOverlay) -> None: ...
    def remove_fovea_overlay(self, overlay_id: str) -> Optional[FoveaOverlay]: ...
    def clear_fovea_overlays(self) -> None: ...
    def reset(self) -> None: ...
```

### Class: `AdaptiveMap` (`mapping.adaptive_map`)
Operational 2.5D elevation and occupancy map.
```python
class AdaptiveMap:
    def __init__(
        self,
        config: Optional[Union[Config, dict, Path, str]] = None,
        grid_engine: Optional[AdaptiveGridEngine] = None,
        hit_increment: float = 0.05,
    ) -> None: ...

    def update(
        self,
        frame: PointCloudFrame,
        semantic_points: Optional[List[SemanticPoint]] = None
    ) -> None: ...

    def get_stale_cells(self, current_time: float, max_age: float) -> List[CellKey]: ...
    def reset(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
```

---

## 9. Terrain & Traversability Engines

**Module**: `fovea_lidar.traversability`, `fovea_lidar.terrain`

### Class: `TerrainAnalyzer` (`terrain.analysis`)
Detects curbs, potholes, excessive slope angles, and surface roughness.
```python
class TerrainAnalyzer:
    def evaluate(self, adaptive_map: AdaptiveMap) -> Dict[CellKey, Dict[str, Any]]: ...
```

### Class: `TraversabilityEngine` (`traversability.engine`)
Combines geometric analysis, semantics, and dynamic obstacles into navigation costmaps.
```python
class TraversabilityEngine:
    def evaluate(
        self,
        adaptive_map: AdaptiveMap,
        tracked_objects: List[TrackedObject],
        risk_regions: Optional[List[RiskRegion]] = None,
    ) -> TraversabilityMap: ...
```

---

## 10. Closed-Loop Pipeline

**Module**: `fovea_lidar.pipeline.FoveaPipeline`

### Class: `FoveaPipeline`
Unified closed-loop execution engine connecting all stages.

```python
class FoveaPipeline:
    def __init__(
        self,
        config: Optional[Union[Config, dict, Path, str]] = None,
        *,
        preprocessor: Optional[Preprocessor] = None,
        segmentation_model: Optional[SegmentationModel] = None,
        terrain_analyzer: Optional[TerrainAnalyzer] = None,
        object_extractor: Optional[ObjectExtractor] = None,
        tracker: Optional[MultiObjectTracker] = None,
        risk_engine: Optional[RiskEngine] = None,
        resolution_controller: Optional[ResolutionController] = None,
        fovea_manager: Optional[FoveaManager] = None,
        adaptive_map: Optional[AdaptiveMap] = None,
        traversability_engine: Optional[TraversabilityEngine] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> None: ...

    def process_frame(self, frame: PointCloudFrame) -> PipelineResult:
        # Execute one complete perception iteration and return verified results
        ...

    def reset(self) -> None:
        # Reset internal states across tracking, mapping, and foveation
        ...
```

---

## 11. Real-Time Dashboard API & WebSocket Protocol

**Module**: `dashboard.server`

### REST Endpoints
- **`GET /`**: Serves the 4-panel HTML5 Canvas / Three.js dashboard web application.
- **`GET /api/scenarios`**: Returns a JSON array of available synthetic scenarios (e.g., `["flagship_demo", "multi_object_dense", ...]`).

### WebSocket Endpoint: `/ws/stream?scenario={name}`
Streams real-time frame telemetry and accepts playback control directives.

#### Client-to-Server Control Schema (JSON)
```json
// Pause streaming
{"type": "pause"}

// Resume streaming
{"type": "resume"}

// Seek to frame index
{"type": "seek", "frame": 12}

// Adjust playback speed
{"type": "speed", "value": 0.5}

// Select active rendering layers
{"type": "select_layer", "layers": ["raw", "semantics", "cells", "foveas", "tracks"]}

// Switch scenario sequence
{"type": "change_scenario", "scenario": "vehicle_crossing"}
```

#### Server-to-Client Telemetry Schema (JSON)
```json
{
  "frame_id": 4,
  "timestamp": 0.40,
  "scenario": "flagship_demo",
  "total_frames": 16,
  "raw_points": [[10.5, 2.1, -0.4]],
  "semantic_points": [{"x": 12.0, "y": 1.5, "z": -0.2, "class": "PEDESTRIAN", "confidence": 0.95}],
  "tracked_objects": [
    {
      "track_id": 1,
      "label": "PEDESTRIAN",
      "position": [42.1, 3.2, -0.1],
      "velocity": [-1.2, 0.0, 0.0],
      "confidence": 0.94,
      "bounding_box": [41.5, 2.7, -0.8, 42.7, 3.7, 1.0],
      "state": "confirmed"
    }
  ],
  "cells": [
    {
      "cell_key": [0, 12, -4],
      "band_index": 0,
      "col": 12,
      "row": -4,
      "resolution": 0.05,
      "elevation": -0.02,
      "roughness": 0.01,
      "occupancy": 0.95,
      "semantic_class": "ROAD",
      "traversability_label": "SAFE",
      "traversability_score": 0.95,
      "is_fovea": false
    }
  ],
  "foveas": [
    {
      "overlay_id": "fovea_trk_1",
      "bounds": [39.5, 0.7, 44.7, 5.7],
      "resolution": 0.05,
      "target_class": "PEDESTRIAN",
      "risk_score": 0.82
    }
  ],
  "metrics": {
    "total_latency_ms": 32.4,
    "preprocessing_ms": 4.1,
    "perception_ms": 6.2,
    "tracking_ms": 0.8,
    "mapping_ms": 12.1,
    "traversability_ms": 5.2,
    "raw_points": 1420,
    "cleaned_points": 1105,
    "active_cells": 1540,
    "fovea_cells": 380,
    "compression_ratio": 0.28
  }
}
```
