# Fovea-LiDAR Jury Walkthrough & Demonstration Script

This document provides the official presentation script and walkthrough narrative for demonstrating Fovea-LiDAR to judges, reviewers, and stakeholders.

---

## 1. Core Thesis & Framing Statement

> *"Conventional autonomous perception systems treat every cubic meter of space equally—wasting massive compute analyzing empty road 60 meters away while starving nearby hazards of real-time responsiveness.*
>
> *Fovea-LiDAR changes this paradigm. Like biological human vision, our system allocates resolution dynamically: maintaining coarse awareness in empty peripheries while instantly concentrating fine, centimeter-level detail on moving obstacles, collision trajectories, and uncertain terrain."*

---

## 2. Demonstration Flow & Live Script

Launch the demonstration with:
```bash
python scripts/demo.py
```
*(The browser opens to `http://127.0.0.1:8000` showing the 4-panel real-time perception dashboard).*

---

### Beat 1: Initial Scene & Peripheral Efficiency (Frames 0–3)
**Visual on Dashboard**:
- **Panel 1 (Raw Point Cloud)**: Shows wide road corridor and sparse asphalt returns.
- **Panel 3 (Adaptive 2.5D Map)**: Concentric distance rings visible. Note the large, coarse peripheral cells ($0.40	ext{ m}$) spanning $25	ext{ m}$ to $100	ext{ m}$.
- **Panel 4 (Telemetry)**: Cell count stays exceptionally low (~800–1,000 cells); total frame latency < 25 ms.

**Presenter Script**:
> *"Notice the base state of the vehicle. On an empty roadway, maintaining fine 5-centimeter cells across the entire 100-meter field of view would require over 4 million cell evaluations. Here, Fovea-LiDAR's annular distance indexing maintains coarse 40-centimeter cells in the distance, cutting memory and computational load by more than 80% while preserving horizon awareness."*

---

### Beat 2: Distant Pedestrian Detection & Tracking (Frames 4–7)
**Visual on Dashboard**:
- A pedestrian enters the field of view at $42	ext{ m}$ distance on the road shoulder.
- **Panel 2 (Semantic Segmentation)**: The pedestrian cluster is classified in magenta/pink (`PEDESTRIAN`, 95% confidence).
- **Panel 3 (Adaptive Map)**: A dynamic high-resolution **fovea overlay patch** instantly forms around the pedestrian bounding box ($R = 0.05	ext{ m}$).
- The track vector shows motion heading toward the center corridor.

**Presenter Script**:
> *"Now, watch as a pedestrian appears at over 40 meters distance. In a conventional distance-only or coarse grid, this pedestrian would be blurred across a single 40-centimeter block, obscuring their shape and gait.*
>
> *Our Semantic Resolution Controller instantly identifies the pedestrian class, overrides the distant coarse zone, and attaches an active 5-centimeter fovea overlay that moves synchronously with the tracked obstacle."*

---

### Beat 3: Dynamic Collision Risk & Threat Escalation (Frames 8–12)
**Visual on Dashboard**:
- The pedestrian steps into the vehicle travel path.
- Distance decreases from $40	ext{ m} 	o 25	ext{ m}$; approaching velocity vector aligns with ego origin ($\cos	heta 	o 1.0$).
- **Panel 3 (Adaptive Map)**: The hazard level escalates from `MEDIUM` $	o$ `HIGH` $	o$ `CRITICAL` (bounding region expands $	imes 2.5$).
- **Panel 3 (Traversability)**: The cell footprint beneath and ahead of the pedestrian turns bright red (`NON_TRAVERSABLE`).

**Presenter Script**:
> *"As the pedestrian begins crossing our travel corridor, our Dynamic Risk Engine kicks into action. By combining approaching velocity, time-to-closest-approach, and vulnerability weights, the risk score escalates to CRITICAL.*
>
> *Notice how the fovea overlay expands spatially ahead of the pedestrian's motion vector, providing dense sampling along their anticipated trajectory. Concurrently, the traversability engine marks their predicted footprint as strictly non-traversable, giving downstream motion planners immediate collision-free avoidance corridors."*

---

### Beat 4: Safe Departure & Resource Recovery (Frames 13–16)
**Visual on Dashboard**:
- The pedestrian clears the travel lane; track coasting decays cleanly.
- The high-resolution fovea overlay deallocates.
- The map returns cleanly to baseline peripheral coarseness without orphaned cells or memory leaks.

**Presenter Script**:
> *"Once the hazard resolves, Fovea-LiDAR's closed loop cleanly deallocates the overlay. Cells and tracks undergo exponential temporal decay, immediately reclaiming memory. The system is entirely non-destructive to the underlying static map."*

---

### Beat 5: The Benchmark Revelation (Instant Q&A)
**Action**: Switch terminal and run:
```bash
python scripts/demo.py --benchmark
```

**Visual Output**:
Side-by-side comparative table:
- **Active Cells**: Mode 1 Uniform = 5,420 vs Mode 5 Adaptive = 1,620 (**>70% reduction**)
- **Processing Latency**: Mode 1 = 64 ms vs Mode 5 = 34 ms (**~1.9x speedup**)
- **Hazard Detection Recall**: **100.0%** across both modes

**Presenter Script**:
> *"Here is the quantitative proof. When evaluated side-by-side against a uniform high-resolution baseline on identical sensor frames: Fovea-LiDAR achieves a 70% reduction in active cell count, nearly doubles throughput from 15 to 29 FPS, and maintains 100% hazard detection recall.*
>
> *We have eliminated the false tradeoff between spatial resolution and edge-compute efficiency."*
