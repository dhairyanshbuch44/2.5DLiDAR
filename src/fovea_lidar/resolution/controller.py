"""Semantic resolution controller implementing distance, complexity, and importance foveation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from fovea_lidar.contracts.mapping import AdaptiveMap
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.risk.engine import RiskEngine
from fovea_lidar.risk.uncertainty import compute_uncertainty, estimate_region_uncertainty
from fovea_lidar.utils.config import Config, load_config


@dataclass
class FoveationContext:
    """Bundles environmental and perceptual inputs for resolution computation."""

    adaptive_map: Optional[AdaptiveMap] = None
    tracked_objects: Sequence[TrackedObject] = ()
    terrain_data: Optional[Dict[str, Any]] = None


class ResolutionController:
    """Adaptive resolution controller computing desired grid resolutions.

    Implements the core foveation formula from section 21:
    R = f(distance, scene_complexity, semantic_importance, dynamic_state, uncertainty, risk)

    Attributes
    ----------
    min_resolution : float
        Finest allowed cell edge length in meters (R_min, default 0.05m).
    max_resolution : float
        Coarsest allowed cell edge length in meters (R_max, default 0.40m).
    alpha : float
        Baseline distance scaling coefficient (default 0.007).
    gamma : float
        Baseline distance scaling exponent (default 1.0).
    weights : Dict[str, float]
        Weighting terms balancing distance, semantic objects, dynamics, complexity,
        risk, and uncertainty.
    semantic_importance_table : Dict[str, float]
        Per-class semantic importance weights in [0.0, 1.0].
    dynamic_boost : float
        Multiplier boost for dynamic moving obstacles.
    complexity_reference_variance : float
        Elevation variance (m^2) scaling scene complexity to 1.0.
    """

    def __init__(
        self,
        min_resolution: float = 0.05,
        max_resolution: float = 0.40,
        alpha: float = 0.007,
        gamma: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
        semantic_importance_table: Optional[Dict[str, float]] = None,
        dynamic_boost: float = 1.0,
        complexity_reference_variance: float = 0.05,
        risk_refinement_enabled: bool = True,
        uncertainty_refinement_enabled: bool = True,
        risk_engine: Optional[RiskEngine] = None,
    ) -> None:
        self.min_resolution = float(min_resolution)
        self.max_resolution = float(max_resolution)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        default_weights = {
            "distance_weight": 0.30,
            "object_weight": 0.50,
            "dynamic_weight": 0.30,
            "complexity_weight": 0.20,
            "risk_weight": 0.40,
            "uncertainty_weight": 0.20,
        }
        self.weights: Dict[str, float] = dict(weights or default_weights)

        default_importance = {
            "PEDESTRIAN": 1.0,
            "CYCLIST": 0.95,
            "VEHICLE": 0.90,
            "TRUCK": 0.85,
            "UNKNOWN_OBSTACLE": 0.80,
            "POTHOLE": 0.60,
            "CURB": 0.50,
            "POLE": 0.40,
            "WALL": 0.30,
            "ROAD": 0.05,
            "SIDEWALK": 0.10,
            "GRASS": 0.10,
            "OTHER_TERRAIN": 0.10,
        }
        self.semantic_importance_table: Dict[str, float] = dict(
            semantic_importance_table or default_importance
        )

        self.dynamic_boost = float(dynamic_boost)
        self.complexity_reference_variance = float(complexity_reference_variance)

        self.risk_refinement_enabled = bool(risk_refinement_enabled)
        self.uncertainty_refinement_enabled = bool(uncertainty_refinement_enabled)
        self.risk_engine = risk_engine or RiskEngine()

    @classmethod
    def load(
        cls, config: Optional[Union[Config, Dict[str, Any], Path, str]] = None
    ) -> ResolutionController:
        """Instantiate ResolutionController from system configuration."""
        if config is None:
            full_cfg = load_config()
        elif isinstance(config, (str, Path)):
            full_cfg = load_config(config)
        elif isinstance(config, (Config, dict)):
            full_cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

        fovea_cfg = full_cfg.get("semantic_foveation", {})
        if isinstance(fovea_cfg, Config):
            fovea_dict = fovea_cfg.to_dict()
        elif isinstance(fovea_cfg, dict):
            fovea_dict = fovea_cfg
        else:
            fovea_dict = {}

        risk_cfg = full_cfg.get("risk_refinement", {})
        risk_enabled = (
            risk_cfg.get("enabled", True)
            if isinstance(risk_cfg, (dict, Config))
            else True
        )

        unc_cfg = full_cfg.get("uncertainty_refinement", {})
        unc_enabled = (
            unc_cfg.get("enabled", True)
            if isinstance(unc_cfg, (dict, Config))
            else True
        )

        risk_engine = RiskEngine.load(full_cfg)

        return cls(
            min_resolution=float(fovea_dict.get("min_resolution", 0.05)),
            max_resolution=float(fovea_dict.get("max_resolution", 0.40)),
            alpha=float(fovea_dict.get("alpha", 0.007)),
            gamma=float(fovea_dict.get("gamma", 1.0)),
            weights=fovea_dict.get("weights"),
            semantic_importance_table=fovea_dict.get("semantic_importance"),
            dynamic_boost=float(fovea_dict.get("dynamic_boost", 1.0)),
            complexity_reference_variance=float(
                fovea_dict.get("complexity_reference_variance", 0.05)
            ),
            risk_refinement_enabled=risk_enabled,
            uncertainty_refinement_enabled=unc_enabled,
            risk_engine=risk_engine,
        )

    def calculate_resolution(
        self,
        distance: float,
        scene_complexity: float = 0.0,
        semantic_importance: float = 0.0,
        dynamic_state: bool = False,
        uncertainty: float = 0.0,
        risk: float = 0.0,
    ) -> float:
        """Compute the continuous target resolution for a region.

        Mathematical Formulation:
        1. Distance Baseline:
           R_dist(d) = clamp(R_min + alpha * d^gamma, R_min, R_max)
        2. Refinement Stimulus:
           F = min(1.0, w_obj * S + w_comp * C + w_dyn * (D * dynamic_boost) + w_risk * K + w_unc * U)
        3. Desired Resolution:
           R = clamp(R_dist(d) - F * (R_dist(d) - R_min), R_min, R_max)

        Parameters
        ----------
        distance : float
            Euclidean distance to the region from sensor origin in meters.
        scene_complexity : float
            Normalized terrain/structural complexity in [0.0, 1.0].
        semantic_importance : float
            Normalized semantic class importance in [0.0, 1.0].
        dynamic_state : bool
            Whether an obstacle in the region is dynamic/moving.
        uncertainty : float
            Occupancy uncertainty / entropy in [0.0, 1.0] (defaults to 0.0).
        risk : float
            Collision hazard score in [0.0, 1.0] (defaults to 0.0).

        Returns
        -------
        float
            Target cell resolution in meters within [R_min, R_max].
        """
        d = max(0.0, float(distance))
        r_dist = self.min_resolution + self.alpha * (d**self.gamma)
        r_dist = float(np.clip(r_dist, self.min_resolution, self.max_resolution))

        w_obj = self.weights.get("object_weight", 0.50)
        w_dyn = self.weights.get("dynamic_weight", 0.30)
        w_comp = self.weights.get("complexity_weight", 0.20)
        w_risk = self.weights.get("risk_weight", 0.40)
        w_unc = self.weights.get("uncertainty_weight", 0.20)

        dyn_term = (self.dynamic_boost if dynamic_state else 0.0)

        effective_risk = float(risk) if self.risk_refinement_enabled else 0.0
        effective_uncertainty = (
            float(uncertainty) if self.uncertainty_refinement_enabled else 0.0
        )

        # Refinement pull toward R_min
        stimulus = (
            w_obj * float(semantic_importance)
            + w_dyn * dyn_term
            + w_comp * float(scene_complexity)
            + w_risk * effective_risk
            + w_unc * effective_uncertainty
        )
        refinement_factor = float(np.clip(stimulus, 0.0, 1.0))

        # Resolution reduction from baseline toward R_min
        target_res = r_dist - refinement_factor * (r_dist - self.min_resolution)
        return float(np.clip(target_res, self.min_resolution, self.max_resolution))

    def _estimate_region_complexity(
        self,
        adaptive_map: Optional[AdaptiveMap],
        pos: Tuple[float, float, float],
        radius: float = 2.0,
    ) -> float:
        """Estimate normalized scene complexity from local elevation variance."""
        if adaptive_map is None or not adaptive_map.cells:
            return 0.0

        cx, cy = pos[0], pos[1]
        r2 = radius * radius
        variances: List[float] = []

        for cell in adaptive_map.cells.values():
            bx = 0.5 * (cell.spatial_bounds[0] + cell.spatial_bounds[2])
            by = 0.5 * (cell.spatial_bounds[1] + cell.spatial_bounds[3])
            if (bx - cx) ** 2 + (by - cy) ** 2 <= r2:
                variances.append(cell.elevation_variance)

        if not variances:
            return 0.0

        mean_var = float(np.mean(variances))
        return float(np.clip(mean_var / self.complexity_reference_variance, 0.0, 1.0))

    def _estimate_region_uncertainty(
        self,
        adaptive_map: Optional[AdaptiveMap],
        pos: Tuple[float, float, float],
        radius: float = 2.0,
    ) -> float:
        """Estimate normalized local uncertainty from map cells near pos."""
        if adaptive_map is None or not adaptive_map.cells:
            return 0.0
        return estimate_region_uncertainty(adaptive_map, (pos[0], pos[1]), radius)

    def compute(
        self, context: Union[FoveationContext, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute desired resolutions across all active regions of interest.

        Parameters
        ----------
        context : FoveationContext or dict
            Context containing 'tracked_objects', 'adaptive_map', and optional 'terrain'.

        Returns
        -------
        Dict[str, float]
            Mapping of region_id to desired_resolution in meters.
        """
        if isinstance(context, dict):
            adaptive_map = context.get("adaptive_map")
            tracked_objects = context.get("tracked_objects", [])
        elif isinstance(context, FoveationContext):
            adaptive_map = context.adaptive_map
            tracked_objects = context.tracked_objects
        else:
            raise TypeError(f"Unsupported context type: {type(context).__name__}")

        resolutions: Dict[str, float] = {}

        for track in tracked_objects:
            region_id = f"fovea_{track.track_id}"
            pos = track.position
            dist = float(np.hypot(pos[0], pos[1]))

            sem_class = track.semantic_class
            importance = self.semantic_importance_table.get(sem_class, 0.10)
            complexity = self._estimate_region_complexity(adaptive_map, pos)

            # Evaluate active dynamic risk if enabled
            if self.risk_refinement_enabled and self.risk_engine is not None:
                risk_score, _ = self.risk_engine.compute_object_risk(
                    track, adaptive_map
                )
            else:
                risk_score = 0.0

            # Evaluate active uncertainty if enabled
            if self.uncertainty_refinement_enabled:
                track_unc = compute_uncertainty(track)
                cell_unc = self._estimate_region_uncertainty(adaptive_map, pos)
                uncertainty = (
                    max(track_unc, cell_unc) if cell_unc > 0.0 else track_unc
                )
            else:
                uncertainty = 0.0

            res = self.calculate_resolution(
                distance=dist,
                scene_complexity=complexity,
                semantic_importance=importance,
                dynamic_state=track.dynamic,
                uncertainty=uncertainty,
                risk=risk_score,
            )
            resolutions[region_id] = round(res, 4)

        return resolutions
