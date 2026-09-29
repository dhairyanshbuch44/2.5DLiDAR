"""Mapping contracts for Fovea-LiDAR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


@dataclass(slots=True)
class AdaptiveCell:
    """Lightweight 2.5D grid cell for adaptive multi-resolution mapping.

    Uses slots for compact memory representation as millions of cells may exist.

    Attributes
    ----------
    spatial_bounds : Tuple[float, float, float, float]
        2D bounding box (min_x, min_y, max_x, max_y) in meters (m).
    resolution : float
        Cell grid resolution (grid edge length) in meters (m). Must be strictly positive (> 0).
    elevation : float
        Mean ground/surface height in meters (m).
    elevation_variance : float
        Variance in surface height estimation in square meters (m^2).
    roughness : float
        Estimated terrain surface roughness / normal dispersion (dimensionless >= 0.0).
    occupancy_probability : float
        Estimated occupancy probability in range [0.0, 1.0] (dimensionless).
    semantic_class : Optional[str]
        Dominant semantic class in cell (e.g., 'ROAD', 'GRASS', 'OBSTACLE').
    semantic_confidence : float
        Confidence of the semantic assignment in range [0.0, 1.0] (dimensionless).
    dynamic_state : str
        Dynamic classification ('STATIC', 'DYNAMIC', 'UNKNOWN').
    velocity : Optional[Tuple[float, float, float]]
        Estimated dynamic velocity vector (vx, vy, vz) in meters per second (m/s).
        None for static terrain.
    risk_score : float
        Composite collision hazard score in range [0.0, 1.0] (dimensionless).
    traversability_score : float
        Traversability cost/score in range [0.0, 1.0] (dimensionless, 1.0 = fully traversable).
    last_updated : float
        Timestamp of last cell observation or Bayes update in seconds (s).
    """

    spatial_bounds: Tuple[float, float, float, float]
    resolution: float
    elevation: float = 0.0
    elevation_variance: float = 0.0
    roughness: float = 0.0
    occupancy_probability: float = 0.5
    semantic_class: Optional[str] = None
    semantic_confidence: float = 0.0
    dynamic_state: str = "UNKNOWN"
    velocity: Optional[Tuple[float, float, float]] = None
    risk_score: float = 0.0
    traversability_score: float = 0.5
    last_updated: float = 0.0

    def __post_init__(self) -> None:
        if len(self.spatial_bounds) != 4:
            raise ValueError(
                f"spatial_bounds must be (min_x, min_y, max_x, max_y), got length {len(self.spatial_bounds)}"
            )
        self.spatial_bounds = (
            float(self.spatial_bounds[0]),
            float(self.spatial_bounds[1]),
            float(self.spatial_bounds[2]),
            float(self.spatial_bounds[3]),
        )

        self.resolution = float(self.resolution)
        if self.resolution <= 0.0:
            raise ValueError(
                f"Cell resolution must be strictly positive (> 0), got {self.resolution}"
            )

        self.elevation = float(self.elevation)
        self.elevation_variance = float(self.elevation_variance)
        self.roughness = float(self.roughness)

        self.occupancy_probability = float(self.occupancy_probability)
        if not (0.0 <= self.occupancy_probability <= 1.0):
            raise ValueError(
                f"occupancy_probability must be in [0.0, 1.0], got {self.occupancy_probability}"
            )

        self.semantic_confidence = float(self.semantic_confidence)
        if not (0.0 <= self.semantic_confidence <= 1.0):
            raise ValueError(
                f"semantic_confidence must be in [0.0, 1.0], got {self.semantic_confidence}"
            )

        if self.velocity is not None:
            if len(self.velocity) != 3:
                raise ValueError(
                    f"Velocity must have 3 components (vx, vy, vz), got length {len(self.velocity)}"
                )
            self.velocity = (
                float(self.velocity[0]),
                float(self.velocity[1]),
                float(self.velocity[2]),
            )

        self.risk_score = float(self.risk_score)
        self.traversability_score = float(self.traversability_score)
        self.last_updated = float(self.last_updated)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize AdaptiveCell to JSON-serializable dictionary."""
        return {
            "spatial_bounds": list(self.spatial_bounds),
            "resolution": self.resolution,
            "elevation": self.elevation,
            "elevation_variance": self.elevation_variance,
            "roughness": self.roughness,
            "occupancy_probability": self.occupancy_probability,
            "semantic_class": self.semantic_class,
            "semantic_confidence": self.semantic_confidence,
            "dynamic_state": self.dynamic_state,
            "velocity": list(self.velocity) if self.velocity is not None else None,
            "risk_score": self.risk_score,
            "traversability_score": self.traversability_score,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveCell:
        """Construct AdaptiveCell from dictionary representation."""
        bounds = data["spatial_bounds"]
        vel = data.get("velocity")
        return cls(
            spatial_bounds=(
                float(bounds[0]),
                float(bounds[1]),
                float(bounds[2]),
                float(bounds[3]),
            ),
            resolution=float(data["resolution"]),
            elevation=float(data.get("elevation", 0.0)),
            elevation_variance=float(data.get("elevation_variance", 0.0)),
            roughness=float(data.get("roughness", 0.0)),
            occupancy_probability=float(data.get("occupancy_probability", 0.5)),
            semantic_class=data.get("semantic_class"),
            semantic_confidence=float(data.get("semantic_confidence", 0.0)),
            dynamic_state=str(data.get("dynamic_state", "UNKNOWN")),
            velocity=(float(vel[0]), float(vel[1]), float(vel[2])) if vel is not None else None,
            risk_score=float(data.get("risk_score", 0.0)),
            traversability_score=float(data.get("traversability_score", 0.5)),
            last_updated=float(data.get("last_updated", 0.0)),
        )


@dataclass
class AdaptiveMap:
    """Represents a global or local multi-resolution adaptive map.

    NOTE: Grid insertion, indexing, ray-casting, and quadtree/octree maintenance logic
    belongs to Prompt 5's AdaptiveGridEngine. This class is purely the shared data contract.

    Attributes
    ----------
    cells : Dict[str, AdaptiveCell]
        Mapping from cell coordinate key (e.g., 'x_y_res') to AdaptiveCell instances.
    resolution_zones : List[Dict[str, Any]]
        Configuration parameters defining concentric or directional resolution zones.
    fovea_overlays : List[Dict[str, Any]]
        Active local high-resolution foveation patches overlaid on the base grid.
    metadata : Dict[str, Any]
        Auxiliary metadata including frame_count (int), last_update_time (seconds), etc.
    """

    cells: Dict[str, AdaptiveCell] = field(default_factory=dict)
    resolution_zones: List[Dict[str, Any]] = field(default_factory=list)
    fovea_overlays: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize AdaptiveMap to JSON-serializable dictionary."""
        return {
            "cells": {key: cell.to_dict() for key, cell in self.cells.items()},
            "resolution_zones": self.resolution_zones,
            "fovea_overlays": self.fovea_overlays,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveMap:
        """Construct AdaptiveMap from dictionary representation."""
        raw_cells = data.get("cells", {})
        cells: Dict[str, AdaptiveCell] = {}
        if isinstance(raw_cells, dict):
            for key, c in raw_cells.items():
                cells[str(key)] = AdaptiveCell.from_dict(c)
        elif isinstance(raw_cells, list):
            for i, c in enumerate(raw_cells):
                cells[f"cell_{i}"] = AdaptiveCell.from_dict(c)

        return cls(
            cells=cells,
            resolution_zones=list(data.get("resolution_zones", [])),
            fovea_overlays=list(data.get("fovea_overlays", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TraversabilityMap:
    """Traversability view wrapping an AdaptiveMap without duplicating base cell storage.

    Attributes
    ----------
    adaptive_map : Any
        Underlying multi-resolution map containing terrain and occupancy cells.
    default_threshold : float
        Traversability score threshold in range [0.0, 1.0] (dimensionless).
        Scores >= threshold are considered navigable/traversable.
    cell_labels : Dict[Any, Any]
        Per-cell final traversability classification labels (SAFE, CAUTION, NON_TRAVERSABLE, UNKNOWN).
    cell_scores : Dict[Any, float]
        Per-cell combined numeric traversability scores in range [0.0, 1.0] (1.0 = safest).
    """

    adaptive_map: Any
    default_threshold: float = 0.5
    cell_labels: Dict[Any, Any] = field(default_factory=dict)
    cell_scores: Dict[Any, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.default_threshold = float(self.default_threshold)
        if not (0.0 <= self.default_threshold <= 1.0):
            raise ValueError(
                f"default_threshold must be in range [0.0, 1.0], got {self.default_threshold}"
            )

    @property
    def cells(self) -> Dict[Any, Any]:
        """Expose underlying map cells dictionary."""
        return getattr(self.adaptive_map, "cells", {})

    def is_traversable(self, cell_key: Any, threshold: Optional[float] = None) -> bool:
        """Check if a specific cell meets traversability threshold."""
        thresh = self.default_threshold if threshold is None else float(threshold)
        if cell_key in self.cell_scores:
            return self.cell_scores[cell_key] >= thresh
        cells = self.cells
        cell = cells.get(cell_key)
        if cell is None:
            return False
        return cell.traversability_score >= thresh

    def get_label(self, cell_key: Any) -> Any:
        """Retrieve traversability label for cell_key."""
        return self.cell_labels.get(cell_key)

    def get_score(self, cell_key: Any) -> float:
        """Retrieve numeric traversability score for cell_key."""
        if cell_key in self.cell_scores:
            return self.cell_scores[cell_key]
        cells = self.cells
        cell = cells.get(cell_key)
        return float(cell.traversability_score) if cell is not None else 0.0

    def get_label_at(self, x: float, y: float) -> Optional[Any]:
        """Find cell containing (x, y) coordinates and return its traversability label."""
        for key, cell in self.cells.items():
            b = cell.spatial_bounds
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                return self.cell_labels.get(key)
        return None

    def get_score_at(self, x: float, y: float) -> float:
        """Find cell containing (x, y) coordinates and return its traversability score."""
        for key, cell in self.cells.items():
            b = cell.spatial_bounds
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                return self.cell_scores.get(key, cell.traversability_score)
        return 0.0

    def find_cell_key_at(self, x: float, y: float) -> Optional[Any]:
        """Find the cell key containing coordinates (x, y)."""
        for key, cell in self.cells.items():
            b = cell.spatial_bounds
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                return key
        return None

    def __getitem__(self, cell_key: Any) -> Any:
        """Convenience dictionary access for cell traversability label."""
        return self.cell_labels.get(cell_key)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TraversabilityMap to JSON-serializable dictionary."""
        map_dict = (
            self.adaptive_map.to_dict()
            if hasattr(self.adaptive_map, "to_dict")
            else {}
        )
        labels_serialized = {
            str(k): (v.value if hasattr(v, "value") else str(v))
            for k, v in self.cell_labels.items()
        }
        scores_serialized = {str(k): float(s) for k, s in self.cell_scores.items()}
        return {
            "adaptive_map": map_dict,
            "default_threshold": self.default_threshold,
            "cell_labels": labels_serialized,
            "cell_scores": scores_serialized,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraversabilityMap:
        """Construct TraversabilityMap from dictionary representation."""
        map_instance = AdaptiveMap.from_dict(data["adaptive_map"])
        return cls(
            adaptive_map=map_instance,
            default_threshold=float(data.get("default_threshold", 0.5)),
            cell_labels=data.get("cell_labels", {}),
            cell_scores=data.get("cell_scores", {}),
        )
