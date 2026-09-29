"""Consolidated traversability engine integrating terrain analysis, semantics, and dynamic obstacles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from fovea_lidar.contracts.mapping import AdaptiveCell, TraversabilityMap
from fovea_lidar.contracts.risk import RiskLevel, RiskRegion
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.risk.engine import RiskEngine
from fovea_lidar.risk.uncertainty import compute_uncertainty
from fovea_lidar.terrain.analysis import TerrainAnalyzer, TerrainClass
from fovea_lidar.utils.config import Config, load_config
from fovea_lidar.utils.logging_setup import get_logger

logger = get_logger(__name__)


class TraversabilityEngine:
    """Consolidates geometric terrain analysis, semantic classification, and dynamic object occupancy.

    Produces per-cell TraversabilityMap instances with final SAFE / CAUTION /
    NON_TRAVERSABLE / UNKNOWN labels and continuous [0.0, 1.0] scores.

    Dynamic Occupancy Overlay Rules:
    1. Base Terrain: Starts from each cell's geometric and semantic classification.
    2. Dynamic Downgrade:
       - Overlap with LOW/MEDIUM risk dynamic obstacle -> downgraded to at least CAUTION.
       - Overlap with HIGH/CRITICAL risk dynamic obstacle -> downgraded to NON_TRAVERSABLE.
    3. Uncertainty Preservation:
       - Cells classified as UNKNOWN or exhibiting uncertainty above threshold
         strictly remain UNKNOWN, preventing false navigable assertions.
    4. Transient Overlay:
       - Evaluated per frame without permanently mutating the underlying static
         map cells, ensuring full recovery after moving obstacles depart.

    Attributes
    ----------
    default_threshold : float
        Score threshold in [0.0, 1.0] defining navigability (scores >= threshold are traversable).
    uncertainty_threshold : float
        Cell uncertainty threshold in [0.0, 1.0] above which cells remain UNKNOWN.
    dynamic_penalties : Dict[str, float]
        Score penalty factors for dynamic obstacles keyed by RiskLevel.
    """

    def __init__(
        self,
        config: Optional[Union[Config, Dict[str, Any], Path, str]] = None,
        *,
        terrain_analyzer: Optional[TerrainAnalyzer] = None,
        risk_engine: Optional[RiskEngine] = None,
        default_threshold: float = 0.5,
        uncertainty_threshold: float = 0.70,
        dynamic_penalties: Optional[Dict[str, float]] = None,
    ) -> None:
        self.terrain_analyzer: Optional[TerrainAnalyzer] = terrain_analyzer
        self.risk_engine: Optional[RiskEngine] = risk_engine
        self.default_threshold = float(default_threshold)
        self.uncertainty_threshold = float(uncertainty_threshold)

        default_penalties = {
            "LOW": 0.50,
            "MEDIUM": 0.60,
            "HIGH": 0.90,
            "CRITICAL": 1.00,
        }
        self.dynamic_penalties = dict(dynamic_penalties or default_penalties)

    @classmethod
    def load(
        cls, config: Optional[Union[Config, Dict[str, Any], Path, str]] = None
    ) -> TraversabilityEngine:
        """Instantiate TraversabilityEngine from system configuration."""
        if config is None:
            full_cfg = load_config()
        elif isinstance(config, (str, Path)):
            full_cfg = load_config(config)
        elif isinstance(config, (Config, dict)):
            full_cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

        trav_cfg = full_cfg.get("traversability", {})
        if isinstance(trav_cfg, Config):
            t_dict = trav_cfg.to_dict()
        elif isinstance(trav_cfg, dict):
            t_dict = trav_cfg
        else:
            t_dict = {}

        terrain_analyzer = TerrainAnalyzer(config=full_cfg)
        risk_engine = RiskEngine.load(config=full_cfg)

        return cls(
            config=full_cfg,
            terrain_analyzer=terrain_analyzer,
            risk_engine=risk_engine,
            default_threshold=float(t_dict.get("default_threshold", 0.5)),
            uncertainty_threshold=float(t_dict.get("uncertainty_threshold", 0.70)),
            dynamic_penalties=t_dict.get("dynamic_penalties"),
        )

    def check_cell_overlap(
        self,
        cell_bounds: Tuple[float, float, float, float],
        track: TrackedObject,
    ) -> bool:
        """Check 2D axis-aligned bounding box intersection between cell and track."""
        cx0, cy0, cx1, cy1 = (
            float(cell_bounds[0]),
            float(cell_bounds[1]),
            float(cell_bounds[2]),
            float(cell_bounds[3]),
        )

        bbox = track.bounding_region
        if len(bbox) == 6:
            ox0, oy0, ox1, oy1 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[3]),
                float(bbox[4]),
            )
        else:
            ox0, oy0, ox1, oy1 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )

        # 2D bounding box intersection check
        return cx0 <= ox1 and cx1 >= ox0 and cy0 <= oy1 and cy1 >= oy0

    def evaluate(
        self,
        adaptive_map: Any,
        tracked_objects: Sequence[TrackedObject],
        risk_regions: Optional[Sequence[RiskRegion]] = None,
    ) -> TraversabilityMap:
        """Consolidate terrain, semantics, and dynamic obstacles into a TraversabilityMap.

        Parameters
        ----------
        adaptive_map : AdaptiveMap
            Stateful multi-resolution map containing cells.
        tracked_objects : Sequence[TrackedObject]
            Active tracks in the current frame.
        risk_regions : Optional[Sequence[RiskRegion]]
            Pre-computed risk regions (optional; evaluated via RiskEngine if omitted).

        Returns
        -------
        TraversabilityMap
            Per-cell traversability classification and scores reflecting static
            terrain combined with dynamic obstacles.
        """
        # 1. Map pre-computed risk regions to track IDs if available
        risk_map: Dict[str, RiskLevel] = {}
        if risk_regions:
            for rr in risk_regions:
                if rr.associated_object_id is not None:
                    risk_map[str(rr.associated_object_id)] = rr.risk_level

        # 2. Recompute or refresh static terrain analysis labels
        terrain_labels: Dict[Any, TerrainClass] = {}
        if self.terrain_analyzer is not None:
            try:
                terrain_labels = self.terrain_analyzer.analyze(adaptive_map)
            except Exception as e:
                logger.warning("Terrain analysis pass raised exception: %s", e)
                terrain_labels = {}

        # 3. Filter dynamic tracked objects
        active_dynamic_tracks = [
            t for t in tracked_objects if getattr(t, "dynamic", True)
        ]

        rank_order = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }

        cell_labels: Dict[Any, TerrainClass] = {}
        cell_scores: Dict[Any, float] = {}

        cells_dict = (
            adaptive_map.cells if hasattr(adaptive_map, "cells") else {}
        )

        for key, cell in cells_dict.items():
            # Determine base static terrain label
            if hasattr(cell, "_terrain_class"):
                base_label = cell._terrain_class
            elif key in terrain_labels:
                base_label = terrain_labels[key]
            else:
                score = float(getattr(cell, "traversability_score", 0.5))
                if score >= 0.80:
                    base_label = TerrainClass.SAFE
                elif score >= 0.40:
                    base_label = TerrainClass.CAUTION
                elif score == 0.25:
                    base_label = TerrainClass.UNKNOWN
                else:
                    base_label = TerrainClass.NON_TRAVERSABLE

            # 4. Uncertainty & UNKNOWN Preservation Rule:
            # Cells with UNKNOWN terrain label or high uncertainty remain strictly UNKNOWN
            cell_unc = compute_uncertainty(cell)
            if (
                base_label == TerrainClass.UNKNOWN
                or cell_unc >= self.uncertainty_threshold
            ):
                cell_labels[key] = TerrainClass.UNKNOWN
                cell_scores[key] = 0.25
                continue

            # 5. Check dynamic obstacle overlap
            cb = cell.spatial_bounds
            overlappers = [
                t
                for t in active_dynamic_tracks
                if self.check_cell_overlap(cb, t)
            ]

            if not overlappers:
                # No dynamic obstacle: retain static terrain label and score
                cell_labels[key] = base_label
                cell_scores[key] = round(float(cell.traversability_score), 3)
                continue

            # 6. Determine maximum risk level among overlapping dynamic tracks
            max_risk = RiskLevel.LOW
            max_rank = 0

            for trk in overlappers:
                lvl = None
                trk_id = getattr(trk, "track_id", getattr(trk, "object_id", None))
                if trk_id is not None and str(trk_id) in risk_map:
                    lvl = risk_map[str(trk_id)]
                elif hasattr(trk, "risk_level") and isinstance(
                    trk.risk_level, RiskLevel
                ):
                    lvl = trk.risk_level
                elif self.risk_engine is not None:
                    _, lvl = self.risk_engine.compute_object_risk(
                        trk, adaptive_map
                    )

                if lvl is None:
                    lvl = RiskLevel.MEDIUM

                rank = rank_order.get(lvl, 1)
                if rank > max_rank:
                    max_rank = rank
                    max_risk = lvl

            # 7. Apply dynamic downgrade rules
            base_score = float(cell.traversability_score)
            penalty = float(self.dynamic_penalties.get(max_risk.value, 0.50))

            if max_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                # Overlap with HIGH or CRITICAL risk obstacle forces NON_TRAVERSABLE
                final_label = TerrainClass.NON_TRAVERSABLE
                final_score = max(0.0, base_score * (1.0 - penalty))
            else:
                # Overlap with moving object forces at least CAUTION
                if base_label == TerrainClass.SAFE:
                    final_label = TerrainClass.CAUTION
                else:
                    final_label = base_label
                final_score = max(0.0, base_score * (1.0 - penalty))

            cell_labels[key] = final_label
            cell_scores[key] = round(final_score, 3)

        return TraversabilityMap(
            adaptive_map=adaptive_map,
            default_threshold=self.default_threshold,
            cell_labels=cell_labels,
            cell_scores=cell_scores,
        )
