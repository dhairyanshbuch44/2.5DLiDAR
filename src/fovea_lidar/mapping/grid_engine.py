"""Multi-resolution radial/annular adaptive grid engine for Fovea-LiDAR."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from fovea_lidar.contracts.mapping import AdaptiveCell, AdaptiveMap
from fovea_lidar.mapping.fovea_overlay import FoveaOverlay
from fovea_lidar.utils.config import Config, load_config

# CellKey: (band_index or overlay_id, col, row)
CellKey = Tuple[Union[int, str], int, int]



@dataclass(frozen=True)
class ResolutionZone:
    """Represents a concentric distance resolution zone."""

    index: int
    name: str
    distance_min: float
    distance_max: float
    target_resolution: float


class AdaptiveGridEngine:
    """Multi-resolution spatial indexing engine using concentric radial/annular distance bands.

    SPATIAL INDEXING DESIGN & BOUNDARY GUARANTEES:
    - The 2D observation plane is partitioned into concentric distance bands [d_min, d_max)
      centered at the sensor origin (0, 0).
    - Within each band b, space is tessellated with regular square cells of edge size R_b.
    - Cell indexing is given by integer grid coordinates:
        col = floor(x / R_b), row = floor(y / R_b)
    - Half-open radial intervals [d_min, d_max) ensure that any point on a zone boundary
      (e.g., r = 10.0m) deterministically belongs to exactly one band (the outer band [10, 25)).
    - Within each band, cells tile R^2 perfectly with zero gaps and zero overlaps:
        min_x = col * R_b, max_x = (col + 1) * R_b
        min_y = row * R_b, max_y = (row + 1) * R_b
    - Negative Cartesian coordinates (x < 0, y < 0) are supported naturally by floor division.
    - Coordinates beyond max_range are rejected by default (returning None for cell_index,
      and skipped in project_points), with an optional 'clamp' policy.
    - Sparse storage: cells are only instantiated on demand when points are projected into them.
    """

    def __init__(
        self,
        config: Optional[Union[Config, Dict[str, Any], Path, str]] = None,
        out_of_bounds_policy: str = "reject",
    ) -> None:
        """Initialize the grid engine from configuration.

        Parameters
        ----------
        config : Config, dict, Path, or str, optional
            Configuration source containing 'resolution_zones' and 'mapping'.
        out_of_bounds_policy : str
            Policy for points with r >= max_range: 'reject' (default) or 'clamp'.
        """
        if config is None:
            raw_cfg = load_config()
        elif isinstance(config, (str, Path)):
            raw_cfg = load_config(config)
        elif isinstance(config, (Config, dict)):
            raw_cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

        zones_cfg = raw_cfg.get("resolution_zones", [])
        if not zones_cfg:
            # Fallback default zones
            zones_cfg = [
                {"name": "fovea", "distance_min": 0.0, "distance_max": 10.0, "target_resolution": 0.05},
                {"name": "mid_range", "distance_min": 10.0, "distance_max": 25.0, "target_resolution": 0.10},
                {"name": "far_field", "distance_min": 25.0, "distance_max": 50.0, "target_resolution": 0.25},
                {"name": "periphery", "distance_min": 50.0, "distance_max": 100.0, "target_resolution": 0.40},
            ]

        self.zones: List[ResolutionZone] = []
        for i, z in enumerate(zones_cfg):
            zone = ResolutionZone(
                index=i,
                name=str(z.get("name", f"zone_{i}")),
                distance_min=float(z["distance_min"]),
                distance_max=float(z["distance_max"]),
                target_resolution=float(z["target_resolution"]),
            )
            self.zones.append(zone)

        self.min_range: float = self.zones[0].distance_min
        self.max_range: float = self.zones[-1].distance_max

        if out_of_bounds_policy not in ("reject", "clamp"):
            raise ValueError(f"out_of_bounds_policy must be 'reject' or 'clamp', got {out_of_bounds_policy}")
        self.out_of_bounds_policy: str = out_of_bounds_policy

        # Sparse hash storage for instantiated AdaptiveCells
        self._cells: Dict[CellKey, AdaptiveCell] = {}

        # Active local high-resolution fovea overlay sub-grids
        self._fovea_overlays: Dict[str, FoveaOverlay] = {}

    @property
    def total_cells(self) -> int:
        """Return the count of currently allocated/observed cells."""
        return len(self._cells)

    @property
    def fovea_overlays(self) -> List[FoveaOverlay]:
        """List of active local high-resolution fovea overlays."""
        return list(self._fovea_overlays.values())

    def add_fovea_overlay(self, overlay: FoveaOverlay) -> None:
        """Register an active fovea overlay sub-grid."""
        self._fovea_overlays[overlay.overlay_id] = overlay

    def remove_fovea_overlay(self, overlay_id: str) -> Optional[FoveaOverlay]:
        """Remove an active fovea overlay and purge its cells from active storage."""
        overlay = self._fovea_overlays.pop(overlay_id, None)
        if overlay is not None:
            # Purge any cells belonging to this overlay from _cells to avoid orphaned state
            keys_to_remove = [k for k in self._cells if k[0] == overlay_id]
            for k in keys_to_remove:
                del self._cells[k]
        return overlay

    def clear_fovea_overlays(self) -> None:
        """Clear all active fovea overlays and their associated cells."""
        overlay_ids = list(self._fovea_overlays.keys())
        for oid in overlay_ids:
            self.remove_fovea_overlay(oid)

    def reset(self) -> None:
        """Reset all allocated cells and active fovea overlays."""
        self.clear_fovea_overlays()
        self._cells.clear()

    def find_overlay_for_point(self, x: float, y: float) -> Optional[FoveaOverlay]:
        """Check active fovea overlays for point containment, prioritizing finest resolution."""
        best_overlay: Optional[FoveaOverlay] = None
        for overlay in self._fovea_overlays.values():
            if overlay.contains_point(x, y):
                if best_overlay is None or overlay.resolution < best_overlay.resolution:
                    best_overlay = overlay
        return best_overlay

    def band_for_distance(self, distance: float) -> Optional[int]:
        """Determine the resolution zone band index for a given Euclidean distance.

        Uses explicit half-open intervals [d_min, d_max). Points on boundary d=d_max
        of band i belong to band i+1.

        Parameters
        ----------
        distance : float
            Euclidean radial distance from sensor origin in meters (m).

        Returns
        -------
        Optional[int]
            Zone index (0 to len(zones) - 1), or None if out of bounds (when policy is 'reject').
        """
        d = float(distance)
        if d < self.min_range:
            d = self.min_range

        if d >= self.max_range:
            if self.out_of_bounds_policy == "clamp":
                return len(self.zones) - 1
            return None

        for zone in self.zones:
            if zone.distance_min <= d < zone.distance_max:
                return zone.index

        # Edge case at exact upper bound if clamped
        if self.out_of_bounds_policy == "clamp" and d == self.max_range:
            return len(self.zones) - 1

        return None

    def resolution_for_band(self, band_index: int) -> float:
        """Return the cell resolution in meters for the given band index."""
        if not (0 <= band_index < len(self.zones)):
            raise IndexError(f"Band index {band_index} out of range [0, {len(self.zones) - 1}]")
        return self.zones[band_index].target_resolution

    def cell_index(self, x: float, y: float) -> Optional[CellKey]:
        """Compute the unique CellKey for a 2D position (x, y).

        Checks active fovea overlays first; if point falls outside any overlay,
        projects to the base concentric distance bands.

        Parameters
        ----------
        x : float
            X coordinate in meters (m). Supports negative values.
        y : float
            Y coordinate in meters (m). Supports negative values.

        Returns
        -------
        Optional[CellKey]
            Tuple of (band_index or overlay_id, col, row), or None if out of range.
        """
        x_val = float(x)
        y_val = float(y)

        # 1. Fovea overlay takes precedence
        overlay = self.find_overlay_for_point(x_val, y_val)
        if overlay is not None:
            return overlay.cell_index(x_val, y_val)

        # 2. Base concentric distance bands
        dist = np.hypot(x_val, y_val)
        band_idx = self.band_for_distance(dist)
        if band_idx is None:
            return None

        res = self.resolution_for_band(band_idx)
        col = int(np.floor(x_val / res))
        row = int(np.floor(y_val / res))

        return (band_idx, col, row)

    def cell_bounds(self, key: CellKey) -> Tuple[float, float, float, float]:
        """Compute exact spatial bounds (min_x, min_y, max_x, max_y) in meters for a CellKey.

        Guarantees zero gaps and zero overlaps with neighboring cells in the same band/overlay.
        """
        band_idx, col, row = key
        if isinstance(band_idx, str):
            overlay = self._fovea_overlays.get(band_idx)
            res = overlay.resolution if overlay is not None else 0.05
        else:
            res = self.resolution_for_band(band_idx)
        min_x = col * res
        max_x = min_x + res
        min_y = row * res
        max_y = min_y + res
        return (min_x, min_y, max_x, max_y)

    def get_or_create_cell(self, key: CellKey) -> AdaptiveCell:
        """Retrieve existing AdaptiveCell or materialize a new sparse cell instance.

        Idempotent: repeatedly calling with the same key returns the identical instance.

        Parameters
        ----------
        key : CellKey
            Unique tuple of (band_index or overlay_id, col, row).

        Returns
        -------
        AdaptiveCell
            Lightweight multi-resolution cell.
        """
        if key in self._cells:
            return self._cells[key]

        band_idx, col, row = key
        if isinstance(band_idx, str):
            overlay = self._fovea_overlays.get(band_idx)
            res = overlay.resolution if overlay is not None else 0.05
            bounds = self.cell_bounds(key)
            cell = AdaptiveCell(spatial_bounds=bounds, resolution=res)
            if overlay is not None:
                overlay.cells[(col, row)] = cell
            self._cells[key] = cell
            return cell

        bounds = self.cell_bounds(key)
        res = self.resolution_for_band(band_idx)
        cell = AdaptiveCell(spatial_bounds=bounds, resolution=res)
        self._cells[key] = cell
        return cell

    def project_points(self, points: np.ndarray) -> Dict[CellKey, List[int]]:
        """Project a point cloud into multi-resolution cells.

        Fovea overlay sub-grids take precedence: points falling within active fovea
        regions are projected into finer overlay cells. All remaining points project
        into base concentric bands.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N, 2), (N, 3), or (N, 4). Coordinates in meters.

        Returns
        -------
        Dict[CellKey, List[int]]
            Dictionary mapping each occupied CellKey to the list of point indices
            landing in that cell.
        """
        if points.shape[0] == 0:
            return {}

        x = points[:, 0]
        y = points[:, 1]
        distances = np.hypot(x, y)

        projection_map: Dict[CellKey, List[int]] = defaultdict(list)

        for idx in range(points.shape[0]):
            px = x[idx]
            py = y[idx]

            # 1. Fovea overlay check takes precedence
            overlay = self.find_overlay_for_point(px, py)
            if overlay is not None:
                key: CellKey = overlay.cell_index(px, py)
                projection_map[key].append(idx)
                continue

            # 2. Base concentric band
            d = distances[idx]
            band_idx = self.band_for_distance(d)
            if band_idx is None:
                continue

            res = self.resolution_for_band(band_idx)
            col = int(np.floor(px / res))
            row = int(np.floor(py / res))
            key = (band_idx, col, row)
            projection_map[key].append(idx)

        return dict(projection_map)

    def clear(self) -> None:
        """Reset the sparse grid storage and active fovea overlays."""
        self._cells.clear()
        self.clear_fovea_overlays()

    def to_adaptive_map(self, metadata: Optional[Dict[str, Any]] = None) -> AdaptiveMap:
        """Convert the current engine state into an AdaptiveMap data contract."""
        cells_dict = {f"{k[0]}_{k[1]}_{k[2]}": cell for k, cell in self._cells.items()}
        zones_dict = [
            {
                "index": z.index,
                "name": z.name,
                "distance_min": z.distance_min,
                "distance_max": z.distance_max,
                "target_resolution": z.target_resolution,
            }
            for z in self.zones
        ]
        return AdaptiveMap(
            cells=cells_dict,
            resolution_zones=zones_dict,
            fovea_overlays=[o.to_dict() for o in self._fovea_overlays.values()],
            metadata=dict(metadata or {}),
        )

