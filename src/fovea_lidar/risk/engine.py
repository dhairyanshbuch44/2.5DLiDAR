"""Dynamic collision risk engine evaluating object hazards, approach kinematics, and spatial bounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from fovea_lidar.contracts.mapping import AdaptiveMap
from fovea_lidar.contracts.risk import RiskLevel, RiskRegion
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.utils.config import Config, load_config


class RiskEngine:
    """Dynamic collision risk evaluation engine.

    Computes composite collision risk from distance to sensor origin, semantic
    vulnerability, absolute speed, and relative approach direction (dot product).
    Categorizes hazards into categorical RiskLevels and emits spatially scaled
    RiskRegions.

    Attributes
    ----------
    max_distance : float
        Proximity limit in meters (m) beyond which distance risk drops to zero.
    max_velocity : float
        Speed saturation threshold in meters per second (m/s).
    approach_speed_threshold : float
        Minimum velocity (m/s) required to compute directional approach dot product.
    confidence_discount : bool
        Whether track confidence scales raw risk score.
    confidence_discount_floor : float
        Minimum multiplier floor for low-confidence tracks.
    weights : Dict[str, float]
        Relative weights balancing distance, semantic class, velocity, and approach.
    class_hazard_table : Dict[str, float]
        Normalized hazard/vulnerability scores per semantic class in [0.0, 1.0].
    thresholds : Dict[str, float]
        Score thresholds defining RiskLevel transitions.
    bounds_scaling : Dict[str, float]
        Spatial scaling multipliers expanding region bounds by RiskLevel.
    """

    def __init__(
        self,
        max_distance: float = 40.0,
        max_velocity: float = 10.0,
        approach_speed_threshold: float = 0.1,
        confidence_discount: bool = True,
        confidence_discount_floor: float = 0.2,
        weights: Optional[Dict[str, float]] = None,
        class_hazard_table: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        bounds_scaling: Optional[Dict[str, float]] = None,
    ) -> None:
        self.max_distance = float(max_distance)
        self.max_velocity = float(max_velocity)
        self.approach_speed_threshold = float(approach_speed_threshold)
        self.confidence_discount = bool(confidence_discount)
        self.confidence_discount_floor = float(confidence_discount_floor)

        default_weights = {
            "distance_weight": 0.35,
            "class_weight": 0.20,
            "velocity_weight": 0.20,
            "approach_weight": 0.25,
        }
        self.weights = dict(weights or default_weights)

        default_class_hazards = {
            "PEDESTRIAN": 1.00,
            "CYCLIST": 0.90,
            "VEHICLE": 0.60,
            "TRUCK": 0.70,
            "UNKNOWN_OBSTACLE": 0.50,
            "POLE": 0.30,
            "WALL": 0.20,
            "POTHOLE": 0.40,
            "CURB": 0.30,
            "ROAD": 0.05,
            "SIDEWALK": 0.05,
            "GRASS": 0.05,
            "OTHER_TERRAIN": 0.05,
        }
        self.class_hazard_table = dict(class_hazard_table or default_class_hazards)

        default_thresholds = {
            "critical": 0.75,
            "high": 0.50,
            "medium": 0.25,
        }
        self.thresholds = dict(thresholds or default_thresholds)

        default_bounds_scaling = {
            "LOW": 1.0,
            "MEDIUM": 1.5,
            "HIGH": 2.0,
            "CRITICAL": 2.5,
        }
        self.bounds_scaling = dict(bounds_scaling or default_bounds_scaling)

    @classmethod
    def load(
        cls, config: Optional[Union[Config, Dict[str, Any], Path, str]] = None
    ) -> RiskEngine:
        """Instantiate RiskEngine from system configuration."""
        if config is None:
            full_cfg = load_config()
        elif isinstance(config, (str, Path)):
            full_cfg = load_config(config)
        elif isinstance(config, (Config, dict)):
            full_cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

        risk_cfg = full_cfg.get("risk_refinement", {})
        if isinstance(risk_cfg, Config):
            r_dict = risk_cfg.to_dict()
        elif isinstance(risk_cfg, dict):
            r_dict = risk_cfg
        else:
            r_dict = {}

        return cls(
            max_distance=float(r_dict.get("max_distance", 40.0)),
            max_velocity=float(r_dict.get("max_velocity", 10.0)),
            approach_speed_threshold=float(
                r_dict.get("approach_speed_threshold", 0.1)
            ),
            confidence_discount=bool(r_dict.get("confidence_discount", True)),
            confidence_discount_floor=float(
                r_dict.get("confidence_discount_floor", 0.2)
            ),
            weights=r_dict.get("weights"),
            class_hazard_table=r_dict.get("class_hazard_table"),
            thresholds=r_dict.get("thresholds"),
            bounds_scaling=r_dict.get("bounds_scaling"),
        )

    def compute_distance_factor(self, pos: Sequence[float]) -> float:
        """Compute proximity hazard factor in [0.0, 1.0] (closer = higher)."""
        dist = float(np.hypot(pos[0], pos[1]))
        if dist >= self.max_distance:
            return 0.0
        return float(np.clip(1.0 - (dist / self.max_distance), 0.0, 1.0))

    def compute_class_hazard(self, semantic_class: str) -> float:
        """Retrieve normalized vulnerability/hazard for the semantic class."""
        return float(self.class_hazard_table.get(str(semantic_class).upper(), 0.10))

    def compute_velocity_factor(self, vel: Sequence[float]) -> float:
        """Compute kinematic speed hazard factor in [0.0, 1.0] (faster = higher)."""
        speed = float(np.hypot(vel[0], vel[1]))
        if self.max_velocity <= 0.0:
            return 0.0
        return float(np.clip(speed / self.max_velocity, 0.0, 1.0))

    def compute_approach_factor(
        self, pos: Sequence[float], vel: Sequence[float]
    ) -> float:
        """Compute relative approach factor via dot product with ego-directed vector.

        Returns
        -------
        float
            Factor in [0.0, 1.0]: 1.0 for direct approach toward ego, 0.5 for neutral/static/crossing,
            0.0 for direct recession away from ego.
        """
        speed = float(np.hypot(vel[0], vel[1]))
        if speed < self.approach_speed_threshold:
            return 0.5  # Neutral direction when stationary or sub-threshold

        dist = float(np.hypot(pos[0], pos[1]))
        if dist < 1e-4:
            return 1.0  # Already at sensor origin

        # Vector pointing from object toward ego origin (0, 0): (-x, -y)
        # Unit vector to ego: (-x / dist, -y / dist)
        # Unit velocity: (vx / speed, vy / speed)
        # Dot product cos(theta) = (-x * vx - y * vy) / (dist * speed)
        dot_product = (-pos[0] * vel[0] - pos[1] * vel[1]) / (dist * speed)
        cos_theta = float(np.clip(dot_product, -1.0, 1.0))

        # Map [-1.0, 1.0] to [0.0, 1.0]
        return float(np.clip(0.5 + 0.5 * cos_theta, 0.0, 1.0))

    def compute_confidence_discount(self, track: TrackedObject) -> float:
        """Compute multiplier discounting raw risk score for low-confidence tracks."""
        if not self.confidence_discount:
            return 1.0

        det_conf = getattr(track, "confidence", 1.0)
        trk_conf = getattr(track, "track_confidence", 1.0)
        combined_conf = float(det_conf) * float(trk_conf)
        return float(
            np.clip(
                max(self.confidence_discount_floor, combined_conf),
                0.0,
                1.0,
            )
        )

    def determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Categorize continuous risk score into categorical RiskLevel."""
        score = float(risk_score)
        if score >= self.thresholds.get("critical", 0.75):
            return RiskLevel.CRITICAL
        if score >= self.thresholds.get("high", 0.50):
            return RiskLevel.HIGH
        if score >= self.thresholds.get("medium", 0.25):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def compute_region_bounds(
        self, track: TrackedObject, level: RiskLevel
    ) -> Tuple[float, float, float, float, float, float]:
        """Compute spatial bounding region scaled proportionally to risk severity."""
        bbox = track.bounding_region
        if len(bbox) == 6:
            min_x, min_y, min_z, max_x, max_y, max_z = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                float(bbox[4]),
                float(bbox[5]),
            )
        else:
            min_x, min_y, max_x, max_y = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
            min_z, max_z = -1.0, 1.0

        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        cz = 0.5 * (min_z + max_z)

        # Baseline half-extents with minimum padding
        hx = max(0.5, 0.5 * (max_x - min_x))
        hy = max(0.5, 0.5 * (max_y - min_y))
        hz = max(0.5, 0.5 * (max_z - min_z))

        scale = float(self.bounds_scaling.get(level.value, 1.0))
        return (
            round(cx - hx * scale, 3),
            round(cy - hy * scale, 3),
            round(cz - hz * scale, 3),
            round(cx + hx * scale, 3),
            round(cy + hy * scale, 3),
            round(cz + hz * scale, 3),
        )

    def compute_object_risk(
        self, track: TrackedObject, traversability_context: Any = None
    ) -> Tuple[float, RiskLevel]:
        """Evaluate continuous risk score and categorical severity for a tracked object."""
        w_dist = self.weights.get("distance_weight", 0.35)
        w_class = self.weights.get("class_weight", 0.20)
        w_vel = self.weights.get("velocity_weight", 0.20)
        w_app = self.weights.get("approach_weight", 0.25)

        f_dist = self.compute_distance_factor(track.position)
        f_class = self.compute_class_hazard(track.semantic_class)
        vel = track.velocity if track.velocity is not None else (0.0, 0.0, 0.0)
        f_vel = self.compute_velocity_factor(vel)
        f_app = self.compute_approach_factor(track.position, vel)

        raw_score = (
            w_dist * f_dist
            + w_class * f_class
            + w_vel * f_vel
            + w_app * f_app
        )

        conf_discount = self.compute_confidence_discount(track)
        risk_score = raw_score * conf_discount

        # Optional traversability hazard boost
        cells_dict = None
        if hasattr(traversability_context, "cells"):
            cells_dict = traversability_context.cells
        elif isinstance(traversability_context, dict):
            cells_dict = traversability_context

        if cells_dict is not None:
            tx, ty = float(track.position[0]), float(track.position[1])
            for cell in cells_dict.values():
                if hasattr(cell, "spatial_bounds"):
                    b = cell.spatial_bounds
                    if b[0] <= tx <= b[2] and b[1] <= ty <= b[3]:
                        trav_score = getattr(cell, "traversability_score", 0.5)
                        if trav_score < 0.4:
                            risk_score += 0.05 * (1.0 - trav_score)
                        break
                elif isinstance(cell, dict):
                    bounds = cell.get("spatial_bounds")
                    if bounds and bounds[0] <= tx <= bounds[2] and bounds[1] <= ty <= bounds[3]:
                        trav_score = cell.get("traversability_score", 0.5)
                        if trav_score < 0.4:
                            risk_score += 0.05 * (1.0 - trav_score)
                        break

        risk_score = float(np.clip(risk_score, 0.0, 1.0))
        level = self.determine_risk_level(risk_score)
        return risk_score, level

    def evaluate(
        self,
        tracked_objects: Sequence[TrackedObject],
        traversability_context: Any = None,
    ) -> List[RiskRegion]:
        """Evaluate dynamic collision hazards across all active tracks and emit RiskRegions.

        Parameters
        ----------
        tracked_objects : Sequence[TrackedObject]
            Active tracks to assess.
        traversability_context : Any, optional
            Optional environment or traversability map providing ground context.

        Returns
        -------
        List[RiskRegion]
            List of assessed spatial risk regions with bounds, scores, and severity levels.
        """
        regions: List[RiskRegion] = []
        for track in tracked_objects:
            score, level = self.compute_object_risk(track, traversability_context)
            bounds = self.compute_region_bounds(track, level)
            obj_id = getattr(track, "track_id", getattr(track, "object_id", ""))
            ts = getattr(track, "last_seen", getattr(track, "timestamp", 0.0))

            region = RiskRegion(
                region_bounds=bounds,
                risk_score=round(score, 4),
                risk_level=level,
                associated_object_id=obj_id,
                timestamp=float(ts),
            )
            regions.append(region)

        return regions
