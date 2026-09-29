"""Automated benchmarking harness for ablation studies and uniform baseline comparison."""

from __future__ import annotations

import copy
import logging
import math
import os
from pathlib import Path
import platform
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import psutil
import yaml

from fovea_lidar.contracts.benchmark_result import BenchmarkResult
from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.pipeline import FoveaPipeline
from fovea_lidar.utils.config import Config, load_config

logger = logging.getLogger("fovea_benchmarking")


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overlay dictionary into base dictionary.

    Lists in overlay (such as resolution_zones) completely replace lists in base.
    """
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class BenchmarkRunner:
    """Automated benchmark harness evaluating Fovea-LiDAR performance across ablation modes."""

    def __init__(
        self,
        base_config: Optional[Union[Config, Dict[str, Any], str, Path]] = None,
        ablation_config: Optional[Union[Config, Dict[str, Any], str, Path]] = None,
    ) -> None:
        cfg_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs"

        if base_config is None:
            default_path = cfg_dir / "default.yaml"
            self.base_config = load_config(default_path).to_dict()
        elif isinstance(base_config, Config):
            self.base_config = base_config.to_dict()
        elif isinstance(base_config, (str, Path)):
            self.base_config = load_config(base_config).to_dict()
        else:
            self.base_config = copy.deepcopy(base_config)

        if ablation_config is None:
            ablation_path = cfg_dir / "ablation.yaml"
            if ablation_path.exists():
                with open(ablation_path, "r", encoding="utf-8") as f:
                    self.ablation_config = yaml.safe_load(f) or {}
            else:
                self.ablation_config = {}
        elif isinstance(ablation_config, Config):
            self.ablation_config = ablation_config.to_dict()
        elif isinstance(ablation_config, (str, Path)):
            with open(ablation_config, "r", encoding="utf-8") as f:
                self.ablation_config = yaml.safe_load(f) or {}
        else:
            self.ablation_config = copy.deepcopy(ablation_config)

    def get_mode_config(
        self, mode_name: str, extra_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Construct full pipeline config for a specific named ablation mode."""
        overlay = self.ablation_config.get(mode_name, {})
        merged = deep_merge(self.base_config, overlay)
        if extra_overrides:
            merged = deep_merge(merged, extra_overrides)
        return merged

    def _query_hardware_info(self) -> Dict[str, Any]:
        """Query real, non-fabricated runtime platform and hardware specifications."""
        mem = psutil.virtual_memory()
        return {
            "platform": platform.platform(),
            "os": platform.system(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count() or 1,
            "python_version": platform.python_version(),
            "ram_total_gb": round(float(mem.total) / (1024.0**3), 2),
        }

    def _evaluate_detection_accuracy(
        self,
        detected_objects_per_frame: List[List[Any]],
        gt_frames: List[Dict[str, Any]],
        distance_threshold: float = 2.0,
    ) -> Tuple[float, float, int]:
        """Compute object detection precision, recall, and tracking ID switches against ground truth."""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        id_switches = 0

        # Mapping of gt_object_id -> previous track_id for switch detection
        prev_gt_to_track: Dict[str, str] = {}

        for f_idx, detections in enumerate(detected_objects_per_frame):
            gt_objects = (
                gt_frames[f_idx].get("objects", []) if f_idx < len(gt_frames) else []
            )

            matched_gt_indices: Set[int] = set()
            matched_det_indices: Set[int] = set()
            current_gt_to_track: Dict[str, str] = {}

            for d_idx, det in enumerate(detections):
                d_pos = np.array(det.position[:2], dtype=np.float32)
                best_dist = float("inf")
                best_g_idx = -1

                for g_idx, gt in enumerate(gt_objects):
                    if g_idx in matched_gt_indices:
                        continue
                    g_pos = np.array(gt["position"][:2], dtype=np.float32)
                    dist = float(np.linalg.norm(d_pos - g_pos))
                    if dist < best_dist and dist <= distance_threshold:
                        best_dist = dist
                        best_g_idx = g_idx

                if best_g_idx >= 0:
                    matched_gt_indices.add(best_g_idx)
                    matched_det_indices.add(d_idx)
                    gt_id = str(gt_objects[best_g_idx].get("object_id", best_g_idx))
                    track_id = getattr(det, "track_id", getattr(det, "object_id", None))
                    if track_id is not None:
                        current_gt_to_track[gt_id] = str(track_id)

            tp = len(matched_det_indices)
            fp = len(detections) - tp
            fn = len(gt_objects) - len(matched_gt_indices)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Check ID switches for persistent GT tracks
            for gt_id, trk_id in current_gt_to_track.items():
                if gt_id in prev_gt_to_track and prev_gt_to_track[gt_id] != trk_id:
                    id_switches += 1
            prev_gt_to_track.update(current_gt_to_track)

        if total_tp + total_fp == 0:
            precision = 1.0 if total_fn == 0 else 0.0
        else:
            precision = float(total_tp) / float(total_tp + total_fp)

        if total_tp + total_fn == 0:
            recall = 1.0
        else:
            recall = float(total_tp) / float(total_tp + total_fn)

        return precision, recall, id_switches

    def evaluate_per_class_accuracy(
        self,
        detected_objects_per_frame: List[List[Any]],
        gt_frames: List[Dict[str, Any]],
        distance_threshold: float = 2.0,
    ) -> Dict[str, Dict[str, float]]:
        """Compute semantic class-wise precision, recall, and F1 score against ground truth."""
        class_stats: Dict[str, Dict[str, int]] = {}

        for f_idx, detections in enumerate(detected_objects_per_frame):
            gt_objects = gt_frames[f_idx].get("objects", []) if f_idx < len(gt_frames) else []

            matched_gt: Set[int] = set()
            matched_det: Set[int] = set()

            # For each detection, find best matching GT object with matching semantic class
            for d_idx, det in enumerate(detections):
                d_pos = np.array(det.position[:2], dtype=np.float32)
                d_cls = str(getattr(det, "semantic_class", "UNKNOWN_OBSTACLE"))
                best_dist = float("inf")
                best_g_idx = -1

                for g_idx, gt in enumerate(gt_objects):
                    if g_idx in matched_gt:
                        continue
                    g_cls = str(gt.get("semantic_class", "UNKNOWN_OBSTACLE"))
                    if g_cls != d_cls:
                        continue
                    g_pos = np.array(gt["position"][:2], dtype=np.float32)
                    dist = float(np.linalg.norm(d_pos - g_pos))
                    if dist < best_dist and dist <= distance_threshold:
                        best_dist = dist
                        best_g_idx = g_idx

                if best_g_idx >= 0:
                    matched_gt.add(best_g_idx)
                    matched_det.add(d_idx)
                    stats = class_stats.setdefault(d_cls, {"tp": 0, "fp": 0, "fn": 0})
                    stats["tp"] += 1
                else:
                    stats = class_stats.setdefault(d_cls, {"tp": 0, "fp": 0, "fn": 0})
                    stats["fp"] += 1

            # Any unmatched GT objects count as false negatives for their respective class
            for g_idx, gt in enumerate(gt_objects):
                if g_idx not in matched_gt:
                    g_cls = str(gt.get("semantic_class", "UNKNOWN_OBSTACLE"))
                    stats = class_stats.setdefault(g_cls, {"tp": 0, "fp": 0, "fn": 0})
                    stats["fn"] += 1

        results: Dict[str, Dict[str, float]] = {}
        for cls_name, s in class_stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
            rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            results[cls_name] = {
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        return results

    def _compute_distance_density(self, cells: Dict[Any, Any]) -> Dict[str, int]:
        """Categorize cells into standard distance bands per brief section 33."""
        density = {"0-10m": 0, "10-30m": 0, "30-60m": 0, "60-100m": 0}

        for cell in cells.values():
            b = cell.spatial_bounds
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            r = math.hypot(cx, cy)

            if r < 10.0:
                density["0-10m"] += 1
            elif r < 30.0:
                density["10-30m"] += 1
            elif r < 60.0:
                density["30-60m"] += 1
            else:
                density["60-100m"] += 1

        return density

    def run(
        self,
        pipeline_config: Optional[Union[Config, Dict[str, Any]]] = None,
        scenario_name: str = "multi_object_dense",
        mode_name: str = "mode_5_full",
        num_frames: int = 10,
    ) -> BenchmarkResult:
        """Run an automated benchmark evaluation for a scenario and mode."""
        # 1. Resolve configuration
        if pipeline_config is not None:
            cfg_dict = (
                pipeline_config.to_dict()
                if isinstance(pipeline_config, Config)
                else copy.deepcopy(pipeline_config)
            )
        else:
            cfg_dict = self.get_mode_config(mode_name)

        # 2. Generate scenario frames & ground truth
        gen = SyntheticSceneGenerator(seed=42)
        frames, gt_data = gen.generate(scenario_name, num_frames=num_frames)

        # 3. Setup hardware & memory tracing
        hw_info = self._query_hardware_info()
        tracemalloc.start()
        process = psutil.Process()
        rss_start = process.memory_info().rss / (1024.0 * 1024.0)

        # 4. Instantiate pipeline with target config
        pipeline = FoveaPipeline(config=cfg_dict)

        latencies_ms: List[float] = []
        active_cells_counts: List[int] = []
        detected_objects_per_frame: List[List[Any]] = []
        frame_metrics_list: List[Dict[str, Any]] = []
        total_raw_points = 0

        t_wall_start = time.perf_counter()

        for frame in frames:
            total_raw_points += int(frame.points.shape[0])
            result = pipeline.process_frame(frame)

            lat_ms = float(result.metrics.get("total_ms", 0.0))
            latencies_ms.append(lat_ms)
            active_cells_counts.append(int(result.adaptive_map.total_cells))
            detected_objects_per_frame.append(list(result.tracked_objects))
            frame_metrics_list.append(dict(result.metrics))

        total_wall_sec = time.perf_counter() - t_wall_start

        # 5. Measure Peak Memory Usage
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_traced_mb = peak_bytes / (1024.0 * 1024.0)
        rss_end = process.memory_info().rss / (1024.0 * 1024.0)
        measured_memory_mb = max(peak_traced_mb, max(0.0, rss_end - rss_start), 1.0)

        # 6. Aggregate Metrics
        mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 1.0
        p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 1.0
        mean_fps = (
            float(len(frames)) / total_wall_sec if total_wall_sec > 0 else (1000.0 / mean_latency)
        )
        mean_cells = float(np.mean(active_cells_counts)) if active_cells_counts else 0.0
        throughput_pps = (
            float(total_raw_points) / total_wall_sec if total_wall_sec > 0 else 0.0
        )

        # 7. Compute Distance Density on Final Map
        distance_density = self._compute_distance_density(pipeline.adaptive_map.cells)

        # 8. Accuracy & Tracking Evaluation
        precision, recall, id_switches = self._evaluate_detection_accuracy(
            detected_objects_per_frame,
            gt_data.get("frames", []),
        )
        per_class_accuracy = self.evaluate_per_class_accuracy(
            detected_objects_per_frame,
            gt_data.get("frames", []),
        )

        # 9. Resolution Profile Summary
        zones = cfg_dict.get("resolution_zones", [])
        if len(zones) == 1 and zones[0].get("target_resolution") == 0.05:
            res_profile = "uniform_fine_0.05m"
        else:
            res_profile = f"adaptive_{len(zones)}_zones"

        # Construct and validate contract
        benchmark_result = BenchmarkResult(
            mode_name=mode_name,
            scenario_name=scenario_name,
            num_frames=len(frames),
            hardware_info=hw_info,
            config_snapshot=cfg_dict,
            resolution_profile=res_profile,
            mean_fps=mean_fps,
            mean_latency_ms=mean_latency,
            p95_latency_ms=p95_latency,
            mean_active_cells=mean_cells,
            peak_memory_mb=measured_memory_mb,
            point_throughput_pps=throughput_pps,
            detection_precision=precision,
            detection_recall=recall,
            tracking_id_switches=id_switches,
            distance_density=distance_density,
            frame_metrics=frame_metrics_list,
            timestamp=time.time(),
        )
        # Attach detailed per-class classification metrics
        setattr(benchmark_result, "per_class_metrics", per_class_accuracy)

        return benchmark_result
