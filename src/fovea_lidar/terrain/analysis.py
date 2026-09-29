"""Terrain analysis, slope estimation, hazard classification, and feature detection."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from fovea_lidar.contracts.mapping import AdaptiveCell
from fovea_lidar.mapping.adaptive_map import AdaptiveMap
from fovea_lidar.mapping.grid_engine import CellKey
from fovea_lidar.utils.config import Config, load_config


class TerrainClass(str, Enum):
    """Terrain traversability classification categories."""

    SAFE = "SAFE"
    CAUTION = "CAUTION"
    NON_TRAVERSABLE = "NON_TRAVERSABLE"
    UNKNOWN = "UNKNOWN"


class TerrainAnalyzer:
    """Combines elevation, roughness, slope, height discontinuities, and semantic labels

    to evaluate surface traversability and detect micro-terrain features (curbs, potholes).
    """

    def __init__(
        self,
        config: Optional[Union[Config, Dict[str, Any], Path, str]] = None,
        min_semantic_confidence: float = 0.50,
        max_safe_slope: float = 0.15,
        max_caution_slope: float = 0.35,
        max_safe_roughness: float = 0.06,
        max_caution_roughness: float = 0.18,
        curb_height_threshold: float = 0.08,
        non_traversable_height_threshold: float = 0.25,
        pothole_depth_threshold: float = 0.08,
    ) -> None:
        if config is not None:
            if isinstance(config, (str, Path)):
                cfg = load_config(config)
            elif isinstance(config, (Config, dict)):
                cfg = config
            else:
                raise TypeError(f"Unsupported config type: {type(config).__name__}")

            terrain_cfg = cfg.get("terrain", {})
            self.min_semantic_confidence = float(terrain_cfg.get("min_semantic_confidence", min_semantic_confidence))
            self.max_safe_slope = float(terrain_cfg.get("max_safe_slope", max_safe_slope))
            self.max_caution_slope = float(terrain_cfg.get("max_caution_slope", max_caution_slope))
            self.max_safe_roughness = float(terrain_cfg.get("max_safe_roughness", max_safe_roughness))
            self.max_caution_roughness = float(terrain_cfg.get("max_caution_roughness", max_caution_roughness))
            self.curb_height_threshold = float(terrain_cfg.get("curb_height_threshold", curb_height_threshold))
            self.non_traversable_height_threshold = float(
                terrain_cfg.get("non_traversable_height_threshold", non_traversable_height_threshold)
            )
            self.pothole_depth_threshold = float(terrain_cfg.get("pothole_depth_threshold", pothole_depth_threshold))
        else:
            self.min_semantic_confidence = float(min_semantic_confidence)
            self.max_safe_slope = float(max_safe_slope)
            self.max_caution_slope = float(max_caution_slope)
            self.max_safe_roughness = float(max_safe_roughness)
            self.max_caution_roughness = float(max_caution_roughness)
            self.curb_height_threshold = float(curb_height_threshold)
            self.non_traversable_height_threshold = float(non_traversable_height_threshold)
            self.pothole_depth_threshold = float(pothole_depth_threshold)

        self.cell_labels: Dict[CellKey, TerrainClass] = {}
        self.cell_features: Dict[CellKey, Dict[str, Any]] = {}

    def get_neighbors(
        self, key: CellKey, cell_dict: Dict[CellKey, AdaptiveCell]
    ) -> List[AdaptiveCell]:
        """Find immediate 4-connected spatial neighbors in the same resolution band."""
        band_idx, col, row = key
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors: List[AdaptiveCell] = []

        for dc, dr in neighbor_offsets:
            n_key: CellKey = (band_idx, col + dc, row + dr)
            if n_key in cell_dict:
                neighbors.append(cell_dict[n_key])

        return neighbors

    def compute_slope(
        self, cell: AdaptiveCell, key: CellKey, cell_dict: Dict[CellKey, AdaptiveCell]
    ) -> float:
        """Estimate terrain surface slope using finite differences against adjacent neighbors.

        Handles missing edge neighbors gracefully.
        """
        band_idx, col, row = key
        res = cell.resolution

        left = cell_dict.get((band_idx, col - 1, row))
        right = cell_dict.get((band_idx, col + 1, row))
        bottom = cell_dict.get((band_idx, col, row - 1))
        top = cell_dict.get((band_idx, col, row + 1))

        # Central difference along X
        if left is not None and right is not None:
            dx = (right.elevation - left.elevation) / (2.0 * res)
        elif right is not None:
            dx = (right.elevation - cell.elevation) / res
        elif left is not None:
            dx = (cell.elevation - left.elevation) / res
        else:
            dx = 0.0

        # Central difference along Y
        if bottom is not None and top is not None:
            dy = (top.elevation - bottom.elevation) / (2.0 * res)
        elif top is not None:
            dy = (top.elevation - cell.elevation) / res
        elif bottom is not None:
            dy = (cell.elevation - bottom.elevation) / res
        else:
            dy = 0.0

        return float(np.hypot(dx, dy))

    def compute_height_discontinuity(
        self, cell: AdaptiveCell, neighbors: Sequence[AdaptiveCell]
    ) -> float:
        """Compute the maximum absolute elevation step between this cell and any neighbor."""
        if not neighbors:
            return 0.0
        return float(max(abs(cell.elevation - n.elevation) for n in neighbors))

    def detect_curb(
        self, cell: AdaptiveCell, neighbors: Sequence[AdaptiveCell]
    ) -> bool:
        """Detect a curb feature (positive vertical step discontinuity between road and sidewalk/raised surface)."""
        if not neighbors:
            return False

        max_diff = max(abs(cell.elevation - n.elevation) for n in neighbors)
        # Curb height bounds: distinct step but not a tall wall or vehicle
        is_step = self.curb_height_threshold <= max_diff < self.non_traversable_height_threshold

        # Optional semantic reinforcement if labels exist
        neighbor_classes = {n.semantic_class for n in neighbors if n.semantic_class is not None}
        semantic_curb = "SIDEWALK" in neighbor_classes or "CURB" in neighbor_classes or cell.semantic_class in ("CURB", "SIDEWALK")

        return bool(is_step and (semantic_curb or cell.semantic_class == "ROAD" or cell.semantic_class is None))

    def detect_pothole(
        self, cell: AdaptiveCell, neighbors: Sequence[AdaptiveCell]
    ) -> bool:
        """Detect a localized negative elevation drop (pothole depression) surrounded by higher surface cells."""
        if len(neighbors) < 2:
            return False

        # All neighbors must be significantly higher than the candidate cell
        diffs = [n.elevation - cell.elevation for n in neighbors]
        min_drop = min(diffs)
        mean_drop = float(np.mean(diffs))

        return bool(mean_drop >= self.pothole_depth_threshold and min_drop >= (self.pothole_depth_threshold * 0.5))

    def classify_cell(
        self,
        cell: AdaptiveCell,
        slope: float,
        height_disc: float,
        is_curb: bool,
        is_pothole: bool,
    ) -> TerrainClass:
        """Evaluate the multi-factor rule table to assign a TerrainClass to a cell.

        Explicit Rule Priority:
        1. Low semantic confidence -> UNKNOWN regardless of geometry.
        2. Severe geometric or semantic obstacles -> NON_TRAVERSABLE:
           - Hard obstacle semantics (WALL, VEHICLE, PEDESTRIAN, etc.)
           - Severe height step (>= non_traversable_height_threshold)
           - Severe continuous slope (>= max_caution_slope, unless classified as curb/pothole step)
           - Severe surface roughness (>= max_caution_roughness)
        3. Moderate irregularities / caution zones -> CAUTION:
           - Caution semantic classes (GRASS, OTHER_TERRAIN, CURB, POTHOLE)
           - Curb or pothole feature detected (with step < non_traversable_height_threshold)
           - Moderate step discontinuity (>= curb_height_threshold)
           - Moderate slope (>= max_safe_slope)
           - Moderate roughness (>= max_safe_roughness)
        4. Flat navigable surface -> SAFE:
           - ROAD or unclassified flat ground with slope < max_safe_slope,
             roughness < max_safe_roughness, and height_disc < curb_height_threshold.
        """
        # 1. Low semantic confidence -> UNKNOWN regardless of geometry
        if cell.semantic_class is not None and cell.semantic_confidence < self.min_semantic_confidence:
            return TerrainClass.UNKNOWN

        # 2. Severe geometric or semantic obstacles -> NON_TRAVERSABLE
        if cell.semantic_class in ("WALL", "VEHICLE", "PEDESTRIAN", "TRUCK", "POLE"):
            return TerrainClass.NON_TRAVERSABLE
        if height_disc >= self.non_traversable_height_threshold:
            return TerrainClass.NON_TRAVERSABLE
        if not (is_curb or is_pothole) and slope >= self.max_caution_slope:
            return TerrainClass.NON_TRAVERSABLE
        if cell.roughness >= self.max_caution_roughness:
            return TerrainClass.NON_TRAVERSABLE

        # 3. Moderate geometric irregularities or off-road terrain -> CAUTION
        if cell.semantic_class in ("GRASS", "OTHER_TERRAIN", "CURB", "POTHOLE"):
            return TerrainClass.CAUTION
        if is_curb or is_pothole:
            return TerrainClass.CAUTION
        if height_disc >= self.curb_height_threshold:
            return TerrainClass.CAUTION
        if slope >= self.max_safe_slope:
            return TerrainClass.CAUTION
        if cell.roughness >= self.max_safe_roughness:
            return TerrainClass.CAUTION

        # 4. Confirmed smooth, flat navigable surface -> SAFE
        if cell.semantic_class == "ROAD" or cell.semantic_class is None:
            if (
                slope < self.max_safe_slope
                and cell.roughness < self.max_safe_roughness
                and height_disc < self.curb_height_threshold
            ):
                return TerrainClass.SAFE

        # Fallback
        if (
            slope < self.max_safe_slope
            and cell.roughness < self.max_safe_roughness
            and height_disc < self.curb_height_threshold
        ):
            return TerrainClass.SAFE
        return TerrainClass.CAUTION

    def analyze(self, adaptive_map: AdaptiveMap) -> Dict[CellKey, TerrainClass]:
        """Analyze all cells in the adaptive map and update traversability scores in place.

        Parameters
        ----------
        adaptive_map : AdaptiveMap
            Map instance containing populated cells.

        Returns
        -------
        Dict[CellKey, TerrainClass]
            Mapping of cell keys to determined terrain categories.
        """
        cell_dict = adaptive_map.cells
        self.cell_labels.clear()
        self.cell_features.clear()

        # Score mapping for AdaptiveCell.traversability_score
        score_map = {
            TerrainClass.SAFE: 1.0,
            TerrainClass.CAUTION: 0.5,
            TerrainClass.NON_TRAVERSABLE: 0.0,
            TerrainClass.UNKNOWN: 0.25,
        }

        for key, cell in cell_dict.items():
            neighbors = self.get_neighbors(key, cell_dict)
            slope = self.compute_slope(cell, key, cell_dict)
            height_disc = self.compute_height_discontinuity(cell, neighbors)
            is_curb = self.detect_curb(cell, neighbors)
            is_pothole = self.detect_pothole(cell, neighbors)

            terrain_cls = self.classify_cell(
                cell=cell,
                slope=slope,
                height_disc=height_disc,
                is_curb=is_curb,
                is_pothole=is_pothole,
            )

            # Mutate cell in place per contract
            cell.traversability_score = score_map[terrain_cls]

            self.cell_labels[key] = terrain_cls
            self.cell_features[key] = {
                "slope": slope,
                "height_discontinuity": height_disc,
                "is_curb": is_curb,
                "is_pothole": is_pothole,
            }

        return dict(self.cell_labels)

    def get_label_distribution(self) -> Dict[str, int]:
        """Return counts of classified cells by category."""
        dist: Dict[str, int] = {
            TerrainClass.SAFE.value: 0,
            TerrainClass.CAUTION.value: 0,
            TerrainClass.NON_TRAVERSABLE.value: 0,
            TerrainClass.UNKNOWN.value: 0,
        }
        for label in self.cell_labels.values():
            dist[label.value] += 1
        return dist
