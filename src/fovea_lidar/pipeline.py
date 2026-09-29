"""Closed-loop adaptive perception pipeline for Fovea-LiDAR."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from fovea_lidar.contracts.pipeline_result import PipelineResult
from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.mapping.adaptive_map import AdaptiveMap
from fovea_lidar.metrics.collector import MetricsCollector
from fovea_lidar.objects.extraction import ObjectExtractor
from fovea_lidar.perception.base import SegmentationModel
from fovea_lidar.perception.registry import build_segmentation_model
from fovea_lidar.preprocessing.pipeline import Preprocessor
from fovea_lidar.resolution.controller import ResolutionController
from fovea_lidar.resolution.fovea_manager import FoveaManager
from fovea_lidar.risk.engine import RiskEngine
from fovea_lidar.terrain.analysis import TerrainAnalyzer
from fovea_lidar.tracking.tracker import MultiObjectTracker
from fovea_lidar.traversability.engine import TraversabilityEngine
from fovea_lidar.utils.config import Config, load_config, validate_config
from fovea_lidar.utils.logging_setup import get_logger

logger = get_logger(__name__)


class FoveaPipeline:
    """End-to-end closed-loop adaptive LiDAR perception pipeline.

    Wires together input preprocessing, semantic segmentation, terrain hazard
    analysis, object extraction, multi-object tracking, collision risk/uncertainty
    evaluation, adaptive resolution foveation, and multi-resolution grid mapping.

    Closed-Loop Principle:
    Fovea overlay patches are updated based on active tracks and risk state
    *before* the current frame's points are integrated into the AdaptiveMap.
    This guarantees that the point cloud projects into high-resolution foveation
    cells immediately upon risk detection.
    """

    def __init__(
        self,
        config: Optional[Union[Config, Dict[str, Any], Path, str]] = None,
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
    ) -> None:
        if config is None:
            self.config = load_config()
        elif isinstance(config, (str, Path)):
            self.config = load_config(config)
        elif isinstance(config, (Config, dict)):
            self.config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

        validate_config(self.config)

        # 1. Preprocessing stage
        self.preprocessor: Preprocessor = (
            preprocessor or Preprocessor(config=self.config)
        )

        # 2. Semantic segmentation model (pluggable backend)
        self.segmentation_model: SegmentationModel = (
            segmentation_model or build_segmentation_model(config=self.config)
        )

        # 3. Terrain analysis module
        self.terrain_analyzer: TerrainAnalyzer = (
            terrain_analyzer or TerrainAnalyzer(config=self.config)
        )

        # 4. Object extraction & clustering
        self.object_extractor: ObjectExtractor = (
            object_extractor or ObjectExtractor.load(config=self.config)
        )

        # 5. Multi-object tracking filter
        self.tracker: MultiObjectTracker = (
            tracker or MultiObjectTracker.load(config=self.config)
        )

        # 6. Dynamic collision risk engine
        self.risk_engine: RiskEngine = (
            risk_engine or RiskEngine.load(config=self.config)
        )

        # 7. Adaptive resolution controller
        self.resolution_controller: ResolutionController = (
            resolution_controller
            or ResolutionController.load(config=self.config)
        )

        # 8. Fovea manager coordinating overlays
        if fovea_manager is not None:
            self.fovea_manager: FoveaManager = fovea_manager
        else:
            fovea_cfg = self.config.get("semantic_foveation", {})
            f_dict = (
                fovea_cfg.to_dict()
                if isinstance(fovea_cfg, Config)
                else (fovea_cfg if isinstance(fovea_cfg, dict) else {})
            )
            self.fovea_manager = FoveaManager(
                controller=self.resolution_controller,
                fovea_radius=float(f_dict.get("fovea_radius", 2.0)),
                fovea_activation_threshold=float(
                    f_dict.get("fovea_activation_threshold", 0.02)
                ),
            )

        # 9. 2.5D Adaptive multi-resolution map
        self.adaptive_map: AdaptiveMap = (
            adaptive_map or AdaptiveMap(config=self.config)
        )

        # 10. Traversability engine (consolidating terrain and dynamic obstacles)
        self.traversability_engine: TraversabilityEngine = (
            traversability_engine or TraversabilityEngine.load(config=self.config)
        )

        # 11. Performance & telemetry collector
        self.metrics_collector: MetricsCollector = (
            metrics_collector or MetricsCollector()
        )

        # Stateful sequence index
        self.frame_index: int = 0

    def reset(self) -> None:
        """Reset all internal state across tracking, mapping, and foveation."""
        self.tracker.reset()
        self.adaptive_map.reset()
        self.fovea_manager.reset()
        self.metrics_collector.reset()
        self.frame_index = 0
        logger.info("FoveaPipeline state has been reset.")

    def process_frame(self, frame: PointCloudFrame) -> PipelineResult:
        """Execute end-to-end perception, foveation, and mapping for one LiDAR scan.

        Parameters
        ----------
        frame : PointCloudFrame
            Raw input point cloud frame.

        Returns
        -------
        PipelineResult
            Complete bundle of intermediate and final outputs, fovea overlays,
            updated map reference, and measured performance metrics.
        """
        t_start = time.perf_counter()

        # 1. Preprocessing stage (filtering, voxel downsampling, ground candidate extraction)
        t0 = time.perf_counter()
        cleaned_frame, _ = self.preprocessor.process(frame)
        t_preproc = (time.perf_counter() - t0) * 1000.0

        # 2. Semantic segmentation stage
        t0 = time.perf_counter()
        semantic_points = self.segmentation_model.predict(cleaned_frame)
        t_percep = (time.perf_counter() - t0) * 1000.0

        # 3. Terrain analysis stage (evaluates existing map surface features and hazards)
        t0 = time.perf_counter()
        terrain_data = self.terrain_analyzer.analyze(self.adaptive_map)
        t_terrain = (time.perf_counter() - t0) * 1000.0

        # 4. Object extraction stage (clustering obstacle points into candidate objects)
        t0 = time.perf_counter()
        detected_objects = self.object_extractor.extract(semantic_points)
        t_extract = (time.perf_counter() - t0) * 1000.0

        # 5. Multi-object tracking stage (Kalman state estimation, track persistence)
        t0 = time.perf_counter()
        tracked_objects = self.tracker.update(
            detected_objects, timestamp=cleaned_frame.timestamp
        )
        t_track = (time.perf_counter() - t0) * 1000.0

        # 6. Dynamic collision risk & uncertainty evaluation stage
        t0 = time.perf_counter()
        risk_regions = self.risk_engine.evaluate(
            tracked_objects, self.adaptive_map
        )
        t_risk = (time.perf_counter() - t0) * 1000.0

        # 7. Resolution computation & fovea overlay allocation (Closed Loop)
        t0 = time.perf_counter()
        self.fovea_manager.update(
            tracked_objects=tracked_objects,
            adaptive_map=self.adaptive_map,
            terrain=terrain_data,
        )
        active_foveas = self.fovea_manager.get_active_fovea_regions()
        t_resolution = (time.perf_counter() - t0) * 1000.0

        # 8. Adaptive map update (MUST occur after fovea allocation so points route into active overlays)
        t0 = time.perf_counter()
        self.adaptive_map.update(cleaned_frame, semantic_points=semantic_points)
        if hasattr(self.adaptive_map, "prune_stale_cells"):
            self.adaptive_map.prune_stale_cells(max_age=0.15)
        t_map = (time.perf_counter() - t0) * 1000.0

        # 9. Dynamic traversability evaluation stage
        t0 = time.perf_counter()
        traversability_map = None
        if self.traversability_engine is not None:
            traversability_map = self.traversability_engine.evaluate(
                self.adaptive_map, tracked_objects, risk_regions
            )
        t_trav = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_start) * 1000.0

        # Record telemetry
        stage_timings = {
            "preprocessing_ms": round(t_preproc, 3),
            "perception_ms": round(t_percep, 3),
            "terrain_ms": round(t_terrain, 3),
            "extraction_ms": round(t_extract, 3),
            "tracking_ms": round(t_track, 3),
            "risk_ms": round(t_risk, 3),
            "resolution_ms": round(t_resolution, 3),
            "mapping_ms": round(t_map, 3),
            "traversability_ms": round(t_trav, 3),
            "total_ms": round(t_total, 3),
        }
        counts = {
            "raw_points": int(frame.points.shape[0]),
            "cleaned_points": int(cleaned_frame.points.shape[0]),
            "semantic_points": len(semantic_points),
            "detected_objects": len(detected_objects),
            "tracked_objects": len(tracked_objects),
            "risk_regions": len(risk_regions),
            "active_fovea_regions": len(active_foveas),
            "total_cells": self.adaptive_map.total_cells,
        }
        self.metrics_collector.record_frame(stage_timings, counts)

        # Build pipeline result bundle
        current_metrics = self.metrics_collector.get_current()
        result = PipelineResult(
            frame_index=self.frame_index,
            timestamp=float(cleaned_frame.timestamp),
            raw_frame=frame,
            cleaned_frame=cleaned_frame,
            semantic_points=semantic_points,
            detected_objects=detected_objects,
            tracked_objects=tracked_objects,
            risk_regions=risk_regions,
            active_fovea_regions=active_foveas,
            adaptive_map=self.adaptive_map,
            terrain_data=terrain_data,
            traversability_map=traversability_map,
            metrics=current_metrics,
        )

        self.frame_index += 1
        return result
