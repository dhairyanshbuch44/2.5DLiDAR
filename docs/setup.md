# Fovea-LiDAR Setup & Installation Guide

This guide details the step-by-step procedure for configuring, installing, verifying, and running the Fovea-LiDAR perception repository in a clean environment.

---

## 1. System Requirements & Prerequisites

### Platform Compatibility
- **Operating Systems**: Windows 10/11, Ubuntu Linux (20.04/22.04 LTS), macOS (12+)
- **Python Runtime**: Python 3.10, 3.11, 3.12, or 3.13 (64-bit)
- **Hardware Architecture**: Pure CPU execution supported.
  > **Note**: Fovea-LiDAR MVP is built with **zero GPU / CUDA / PyTorch lock**. It runs 100% locally on commodity CPUs without specialized hardware accelerators.
- **Memory**: 4 GB RAM minimum (8 GB recommended for multi-run benchmarks).
- **Disk Space**: ~200 MB for repository, synthetic samples, and dependencies.

---

## 2. Installation Steps

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/fovea-lidar.git
cd fovea-lidar
```

### Step 2: Create and Activate Virtual Environment
#### Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Core Dependencies
Install the package in editable mode (`-e`) along with testing and dashboard extras:
```bash
pip install --upgrade pip
pip install -e ".[dashboard,test]"
```

*Core dependencies installed:*
- `numpy`: Fast vectorized numerical operations and point cloud manipulations.
- `scipy`: Spatial KD-Tree acceleration and statistical filtering.
- `scikit-learn`: DBSCAN Euclidean spatial clustering.
- `pyyaml`: Configuration parsing and validation.
- `fastapi`, `uvicorn`, `websockets`: Real-time telemetry streaming and REST API.
- `pytest`, `pytest-cov`: Comprehensive verification and code coverage reporting.
- `matplotlib`: Benchmark plot generation and ablation visualization.

---

## 3. Immediate Verification

Run the full automated test suite to confirm complete mathematical, algorithmic, and interface correctness:

```bash
pytest tests/ -v --tb=short
```

Expected output:
```text
======================= 167 passed, 1 warning in 38.50s =======================
```

To run only the CPU-only integrity verification:
```bash
pytest -m cpu_only -v
```

---

## 4. Running the System

### Option A: Interactive 4-Panel Dashboard (Recommended)
Launch the FastAPI WebSocket server and auto-open the web browser interface:
```bash
python scripts/demo.py
```
This serves the dashboard at `http://127.0.0.1:8000`. If running in a headless environment, pass `--no-browser`:
```bash
python scripts/demo.py --no-browser --port 8000
```

### Option B: Instant Q&A Benchmark Comparison
Compare Mode 1 (Uniform High-Resolution baseline) against Mode 5 (Fovea-LiDAR Closed-Loop Adaptive) on the flagship demonstration scenario:
```bash
python scripts/demo.py --benchmark
```

### Option C: Full Ablation Benchmark Suite
Run all 5 ablation modes across synthetic scenarios and generate comparison plots:
```bash
python scripts/benchmark.py --num-frames 10
```
Benchmark reports and plots are exported to `experiments/` and `experiments/plots/`.

---

## 5. Configuration Files Overview

All behavioral constants are governed by declarative YAML files located in `configs/`:

- **`configs/default.yaml`**: The primary system configuration defining distance zones, foveation weights, risk parameters, and terrain thresholds.
- **`configs/demo.yaml`**: Optimized tuning for real-time dashboard streaming and interactive jury demonstrations.
- **`configs/benchmark.yaml`**: Matrix specifications and ablation definitions for automated evaluation runs.
