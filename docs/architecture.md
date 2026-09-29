# Fovea-LiDAR System Architecture Specification

## 1. Executive Summary & Problem Formulation

Standard autonomous robotic perception pipelines uniformly discretize 3D LiDAR environments using fixed-resolution voxel grids or 2.5D elevation maps (typically 0.05 m to 0.10 m per cell). In outdoor operations spanning 100 m x 100 m, uniform fine resolution triggers explosive computational and memory footprints:
- A 100 m x 100 m grid at 0.05 m uniform resolution demands **4,000,000 active cell evaluations**, exceeding edge-compute memory ceilings and real-time loop deadlines.
- More critically, >85% of this compute is expended evaluating empty air, uniform distant asphalt, or static background structures with zero navigational relevance.
- Conversely, coarse uniform grids (0.40 m) sacrifice fine feature detection, missing small obstacles (curbs, potholes, debris) and vulnerable road users (pedestrians) until hazardous proximity.

**Fovea-LiDAR** solves this fundamental bottleneck through **biologically inspired semantic and risk-driven foveation**:
1. Concentric radial distance bands provide an efficient base spatial representation: fine resolution near the ego-robot (0.05 m), tapering to coarse resolution in the periphery (0.40 m).
2. Dynamic high-resolution **fovea overlays** actively track and focus high-density sampling (0.05 m) onto critical regions driven by semantic importance, dynamic obstacle velocity, collision hazard, and mapping uncertainty.
3. The closed-loop architecture feeds tracking and risk estimates from frame N-1 directly into the resolution controller of frame N, ensuring the grid engine refines spatial resolution *before* sensor points are projected.

---

## 2. End-to-End System Architecture & Data Flow

The following diagram illustrates the frame-to-frame pipeline execution within `FoveaPipeline`:

```
+-------------------------------------------------------------------------------------------------+
|                                 FOVEA-LiDAR PIPELINE DATA FLOW                                  |
+-------------------------------------------------------------------------------------------------+
                                      |
                           [Raw PointCloudFrame]
                                      |
                                      v
                          [PreprocessorPipeline]
                       /                                     (Ground Candidates)                 (Cleaned Frame)
                  |                                  |
                  v                                  v
          [TerrainAnalyzer]                 [SegmentationModel]
                  |                                  |
                  |                           (SemanticPoints)
                  |                          /                                  |                         v                  v
                  |               [ObjectExtractor]     [AdaptiveMap Integration]
                  |                         |                  ^
                  |                 (DetectedObjects)          |
                  |                         |                  |
                  |                         v                  |
                  |                [MultiObjectTracker]        |
                  |                         |                  |
                  |                  (TrackedObjects)          |
                  |                         |                  |
                  |                         v                  |
                  |                    [RiskEngine]            |
                  |                         |                  |
                  |                   (RiskRegions)            |
                  |                         |                  |
                  |       +-----------------+                  |
                  |       |                                    |
                  v       v                                    |
            [ResolutionController & FoveaManager]              |
                          |                                    |
                   (FoveaOverlays)                             |
                          |                                    |
                          v                                    |
                 [AdaptiveGridEngine] -------------------------+
                          |
                          v
                    [AdaptiveMap]
                          |
                          v
                [TraversabilityEngine]
                          |
                          v
                 [TraversabilityMap]
                          |
                          v
           [MetricsCollector & Telemetry]
           /                                      v                              v
[FastAPI WebSocket Dashboard]     [Benchmark Harness & CLI]
```

### Closed-Loop Timing Sequence
1. **Frame Ingestion**: Raw scan arrives with timestamp $t$.
2. **Preprocessing**: Filtering (voxel downsample, outlier removal, coordinate transform, ground separation).
3. **Perception**: Semantic labeling assigns class and confidence to points.
4. **Object Extraction & Tracking**: Euclidean clustering groups points into bounding regions; constant-velocity Kalman filtering updates track states ($\mathbf{x}, \mathbf{v}$).
5. **Risk Assessment**: Collision trajectories, approaching velocities, and class vulnerabilities compute risk scores $K \in [0, 1]$.
6. **Foveation Overlay Setup (Closed-Loop Step)**: The `ResolutionController` computes required local resolutions and the `FoveaManager` attaches high-resolution overlays to the `AdaptiveGridEngine` *prior to point insertion*.
7. **Spatial Integration**: Preprocessed points and semantic labels are projected into the multi-resolution cells via robust aggregation.
8. **Traversability Evaluation**: Terrain geometry (slope, roughness, curb, pothole) is merged with dynamic obstacle footprints to yield safe, caution, and non-traversable costmaps.

---

## 3. Spatial Indexing & Multi-Resolution Grid Engine

The spatial indexing subsystem (`AdaptiveGridEngine`) delivers deterministic cell addressing with zero boundary artifacts and sparse memory scaling.

### Radial / Annular Distance Bands
The 2D ground plane is partitioned into concentric distance zones centered at ego sensor origin $(0, 0)$:

| Zone Index | Zone Name | Radial Range $[d_{\min}, d_{\max})$ | Base Resolution $R_b$ | Purpose |
|---|---|---|---|---|
| **0** | `fovea` | $[0.0	ext{ m}, 10.0	ext{ m})$ | $0.05	ext{ m}$ ($5	ext{ cm}$) | Immediate robot footprint, curb/step navigation |
| **1** | `mid_range` | $[10.0	ext{ m}, 25.0	ext{ m})$ | $0.10	ext{ m}$ ($10	ext{ cm}$) | Stopping distance, pedestrian interaction |
| **2** | `far_field` | $[25.0	ext{ m}, 50.0	ext{ m})$ | $0.25	ext{ m}$ ($25	ext{ cm}$) | Vehicle approach detection, coarse geometry |
| **3** | `periphery` | $[50.0	ext{ m}, 100.0	ext{ m})$ | $0.40	ext{ m}$ ($40	ext{ cm}$) | Distant highway corridor, horizon boundary |

### Boundary Guarantees & Coordinate Transforms
1. **Half-Open Intervals**: Concentric bands strictly adhere to $[d_{\min}, d_{\max})$. A point exactly at $r = 10.0	ext{ m}$ belongs deterministically to Zone 1, eliminating ambiguity or overlapping band assignments.
2. **Integer Cell Coordinate Projection**:
   Given Cartesian point $(x, y)$ and band $b$ with resolution $R_b$:
   $$	ext{col} = \left\lfloor rac{x}{R_b} ightfloor, \quad 	ext{row} = \left\lfloor rac{y}{R_b} ightfloor$$
   $$	ext{Cell Bounds}: [x_{\min}, x_{\max}) = [	ext{col} \cdot R_b, (	ext{col} + 1) \cdot R_b), \quad [y_{\min}, y_{\max}) = [	ext{row} \cdot R_b, (	ext{row} + 1) \cdot R_b)$$
3. **Negative Coordinate Handling**: Floor division naturally supports negative coordinates $(x < 0, y < 0)$, ensuring consistent cell geometry across all four Cartesian quadrants.
4. **Sparse Hash Storage**: Cells are indexed by `CellKey = (band_index_or_overlay_id, col, row)`. Unobserved regions consume zero memory.
5. **Out-of-Bounds Policy**: Coordinates beyond $d_{\max} = 100.0	ext{ m}$ are rejected (`None`) or clamped based on configuration.

### Local High-Resolution Fovea Overlays
When an obstacle or hazardous zone requires fine resolution outside of Zone 0, the `FoveaManager` injects a `FoveaOverlay` bounding box:
- Points falling within the overlay $[x_{\min}, x_{\max}] 	imes [y_{\min}, y_{\max}]$ are indexed using the overlay's subgrid resolution ($R_{	ext{overlay}} < R_b$).
- Priority resolution: If multiple overlays cover a point, the overlay with the finest resolution takes precedence.
- Overlay cell keys are prefixed with `overlay_id` (e.g., `("fovea_trk_1", col, row)`).
- When a track is terminated, its overlay and associated cells are cleanly purged to prevent orphaned memory retention.

---

## 4. Map Aggregation Rules & Uncertainty Modeling

Within each scan frame, points falling into cell $c$ are aggregated deterministically via `mapping.aggregation`:

### Elevation & Roughness Estimation
1. **Median Elevation**:
   $$\hat{z} = 	ext{median}(\{z_1, z_2, \dots, z_K\})$$
   The sample median provides a 50% breakdown point, rejecting vertical outlier returns caused by dust, spray, multipath reflections, or foliage penetration.
2. **Elevation Variance & Roughness**:
   $$\sigma_z^2 = rac{1}{K} \sum_{k=1}^K (z_k - ar{z})^2, \quad 	ext{roughness} = \sqrt{\max(0, \sigma_z^2)}$$
   Directly captures micro-terrain surface roughness in meters without requiring cross-cell neighborhood stencil operations.

### Occupancy Probability Update
Cell occupancy probability accumulates monotonically with point hit density:
$$P_{	ext{occ}}^{(t)} = \min\left(P_{\max}, P_{	ext{occ}}^{(t-1)} + K \cdot \Delta_{	ext{hit}}ight)$$
where default parameters are $P_{\max} = 0.99$ and $\Delta_{	ext{hit}} = 0.05$. (Full Bayesian log-odds line-of-sight ray tracing is slated for downstream upgrade).

### Semantic Classification Aggregation
Observations are resolved via the **Highest-Confidence-Wins** rule:
1. Among candidate class observations $[(C_1, c_1), (C_2, c_2), \dots]$, the pair with $\max(c_i)$ is selected.
2. If $\max(c_i) > c_{	ext{prev}}$, the cell adopts class $C_i$ with confidence $c_i$.
3. **Deterministic Tie-Breaking**: If candidate confidences are equal, the winning class is chosen via alphabetical sort order (`sorted(candidates, key=lambda x: (-x[1], x[0]))`), eliminating non-deterministic race conditions.

### Multi-Factor Uncertainty Modeling
Cell-level uncertainty $U_{	ext{cell}} \in [0.0, 1.0]$ combines semantic ambiguity, occupancy entropy, and observation staleness:
$$U_{	ext{sem}} = 1.0 - c_{	ext{semantic}}$$
$$U_{	ext{occ}} = 1.0 - 2.0 \cdot |P_{	ext{occ}} - 0.5|$$
$$U_{	ext{stale}} = \min\left(1.0, rac{t_{	ext{current}} - t_{	ext{last\_updated}}}{	au}ight), \quad 	au = 5.0	ext{ s}$$
$$U_{	ext{cell}} = rac{w_{	ext{sem}} U_{	ext{sem}} + w_{	ext{occ}} U_{	ext{occ}} + w_{	ext{stale}} U_{	ext{stale}}}{w_{	ext{sem}} + w_{	ext{occ}} + w_{	ext{stale}}}$$

---

## 5. Dynamic Collision Risk Engine

The `RiskEngine` computes collision risk $K \in [0.0, 1.0]$ for each tracked obstacle:

### Kinematic & Semantic Components
1. **Distance Proximity Factor ($f_{	ext{dist}}$)**:
   $$f_{	ext{dist}} = \max\left(0.0, 1.0 - rac{d}{d_{\max}}ight), \quad d_{\max} = 40.0	ext{ m}$$
2. **Semantic Vulnerability Factor ($f_{	ext{class}}$)**:
   - `PEDESTRIAN`: $1.00$ (vulnerable road user)
   - `CYCLIST`: $0.90$
   - `TRUCK`: $0.70$
   - `VEHICLE`: $0.60$
   - `UNKNOWN_OBSTACLE`: $0.50$
   - Static Obstacles (`POLE`, `WALL`): $0.20 - 0.30$
   - Ground Surfaces (`ROAD`, `SIDEWALK`): $0.05$
3. **Kinematic Velocity Factor ($f_{	ext{speed}}$)**:
   $$f_{	ext{speed}} = \min\left(1.0, rac{\|\mathbf{v}\|}{v_{\max}}ight), \quad v_{\max} = 10.0	ext{ m/s}$$
4. **Directional Approach Factor ($f_{	ext{approach}}$)**:
   Vector dot product between velocity unit vector $\mathbf{u}_v = \mathbf{v} / \|\mathbf{v}\|$ and vector to ego $\mathbf{u}_{	ext{ego}} = -\mathbf{p} / \|\mathbf{p}\|$:
   $$\cos	heta = rac{-(x v_x + y v_y)}{d \cdot \|\mathbf{v}\|}$$
   $$f_{	ext{approach}} = egin{cases} 0.5 + 0.5 \cos	heta & 	ext{if } \|\mathbf{v}\| \ge 0.1	ext{ m/s} \ 0.5 & 	ext{otherwise} \end{cases}$$
5. **Confidence-Discounted Composite Score**:
   $$K_{	ext{raw}} = w_d f_{	ext{dist}} + w_s f_{	ext{class}} + w_v f_{	ext{speed}} + w_a f_{	ext{approach}}$$
   $$K = 	ext{clip}\left(K_{	ext{raw}} \cdot \max(	ext{floor}, c_{	ext{det}} \cdot c_{	ext{track}}), 0.0, 1.0ight)$$
   Weights: $w_d = 0.35, w_s = 0.20, w_v = 0.20, w_a = 0.25$.

### Risk Categorization & Spatial Expansion
| Risk Level | Score Range | Spatial Bounding Region Scale | Action |
|---|---|---|---|
| **CRITICAL** | $K \ge 0.75$ | $	imes 2.5$ Expansion | Maximum priority foveation ($0.05	ext{ m}$), immediate trajectory safety warning |
| **HIGH** | $0.50 \le K < 0.75$ | $	imes 2.0$ Expansion | High-resolution fovea overlay, non-traversable costmap downgrade |
| **MEDIUM** | $0.25 \le K < 0.50$ | $	imes 1.5$ Expansion | Standard fovea overlay, caution traversability cost |
| **LOW** | $K < 0.25$ | $	imes 1.0$ Nominal | Base distance-governed resolution |

---

## 6. Six-Signal Resolution Controller Formulation

The `ResolutionController` synthesizes six environmental and perceptual signals into the target resolution $R_{	ext{target}}$:

$$R_{	ext{dist}}(d) = 	ext{clip}\left(R_{\min} + lpha d^\gamma, R_{\min}, R_{\max}ight)$$
$$F = 	ext{clip}\left(w_{	ext{obj}} S + w_{	ext{dyn}} (D \cdot eta_{	ext{dyn}}) + w_{	ext{comp}} C + w_{	ext{risk}} K_{	ext{eff}} + w_{	ext{unc}} U_{	ext{eff}}, 0.0, 1.0ight)$$
$$R_{	ext{target}} = 	ext{clip}\left(R_{	ext{dist}}(d) - F \cdot (R_{	ext{dist}}(d) - R_{\min}), R_{\min}, R_{\max}ight)$$

### Parameter Nomenclature
- $d$: Euclidean distance to sensor origin (m).
- $R_{\min}, R_{\max}$: Absolute resolution boundaries ($0.05	ext{ m}$ and $0.40	ext{ m}$).
- $lpha, \gamma$: Distance scaling parameters ($lpha = 0.007, \gamma = 1.0$).
- $S$: Semantic importance score ($[0, 1]$, e.g., Pedestrian $= 1.0$, Road $= 0.05$).
- $D$: Dynamic indicator ($1$ if moving, $0$ if static) scaled by boost $eta_{	ext{dyn}} = 1.0$.
- $C$: Structural complexity ($[0, 1]$, normalized from elevation variance).
- $K_{	ext{eff}}$: Collision risk score ($K_{	ext{eff}} = K$ if `risk_refinement.enabled` else $0.0$).
- $U_{	ext{eff}}$: Regional uncertainty ($U_{	ext{eff}} = U$ if `uncertainty_refinement.enabled` else $0.0$).

---

## 7. Traversability Analysis & Costmap Generation

The `TraversabilityEngine` merges geometric ground features and dynamic obstacle footprints into navigation costmaps:

### Geometric Terrain Rules (`TerrainAnalyzer`)
- **Slope Angle**: $	heta_{	ext{slope}} = rccos(n_z) / \pi 	imes 180^\circ$. Slopes $> 20^\circ$ are marked `CAUTION`; $> 35^\circ$ are `NON_TRAVERSABLE`.
- **Step / Curb Discontinuity**: Step height $\Delta z = |z_{	ext{cell}} - z_{	ext{neighbor}}|$. $\Delta z \in [0.08	ext{ m}, 0.25	ext{ m}]$ is flagged as `CURB` (`CAUTION`); $\Delta z > 0.25	ext{ m}$ is `NON_TRAVERSABLE`.
- **Pothole Detection**: Negative depression relative to neighbor plane $\Delta z < -0.08	ext{ m}$ flagged as `POTHOLE`.
- **Roughness**: Surface height standard deviation $\sigma_z > 0.06	ext{ m} \implies$ `CAUTION`; $\sigma_z > 0.18	ext{ m} \implies$ `NON_TRAVERSABLE`.

### Dynamic Occupancy Overlay Rules
1. **Base Terrain**: Derived from static geometry and surface semantics.
2. **Dynamic Downgrades**:
   - Overlap with `LOW` / `MEDIUM` risk obstacle $\implies$ downgraded to at least `CAUTION`.
   - Overlap with `HIGH` / `CRITICAL` risk obstacle $\implies$ downgraded to `NON_TRAVERSABLE`.
3. **Uncertainty Gate**: Cells with uncertainty $U_{	ext{cell}} \ge 0.70$ strictly remain `UNKNOWN` to prevent optimistic trajectory planning through unverified space.
4. **Non-Destructive Overlay**: Dynamic downgrades are computed per frame without modifying persistent terrain cells; leaving obstacles immediately restore base traversability.

---

## 8. MVP Scope vs Advanced Extensions Roadmap

To ensure 100% transparency and alignment with project guidelines, the boundaries between the fully implemented MVP (Prompts 1?18) and future Advanced Extensions (Prompts A1?A5) are delineated below:

| Architectural Component | Current MVP Implementation (Prompts 1?18) | Advanced Extension Roadmap (Prompts A1?A5) |
|---|---|---|
| **LiDAR Input & Dataset** | Synthetic scene generator (8 canonical scenarios) + PCD/PLY loaders | SemanticKITTI, nuScenes, and Velodyne/Ouster ROS 2 driver streams |
| **Semantic Perception** | Pluggable `MockPerceptionModel` with synthetic ground truth injection | Deep 3D PointNet++ / RangeNet++ / Cylinder3D GPU neural networks |
| **Object Extraction** | Euclidean spatial clustering + AABB bounding boxes | 3D Oriented Bounding Box (OBB) clustering with PCA / convex hulls |
| **Multi-Object Tracking** | 3D Constant-Velocity Kalman Filter with greedy / Hungarian assignment | Extended / Unscented Kalman Filter (UKF) with interacting multiple models (IMM) |
| **Occupancy Updating** | Point-count density accumulation ($P_{	ext{occ}} \in [0, 0.99]$) | Full 3D/2.5D Bayesian log-odds with Bresenham line-of-sight ray casting |
| **Compute Architecture** | Multi-threaded CPU pipeline, 0 GPU dependencies, zero PyTorch lock | TensorRT / CUDA-accelerated GPU inference pipeline |
| **SLAM & Odometry** | Ego-centric sensor-frame mapping with kinematic timestamp offsets | Online LiDAR Odometry and Mapping (LOAM / Fast-LIO) with loop closure |
| **Dashboard & UI** | FastAPI WebSocket server with 4-panel HTML5 Canvas / Three.js UI | Full ROS 2 RViz2 plugin and WebRTC high-definition streaming |
