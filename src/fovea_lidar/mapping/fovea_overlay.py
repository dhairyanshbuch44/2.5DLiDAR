"""Local high-resolution fovea overlay sub-grid for AdaptiveGridEngine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np

from fovea_lidar.contracts.mapping import AdaptiveCell


@dataclass
class FoveaOverlay:
    """Represents a localized high-resolution foveation sub-grid overlay.

    Superimposes a fine resolution mesh over a local region of interest (e.g., around
    a dynamic obstacle, high-risk region, or complex terrain feature) without mutating
    the surrounding base concentric grid indexing.

    Attributes
    ----------
    overlay_id : str
        Unique identifier for the overlay (e.g., 'fovea_track_1').
    center : Tuple[float, float]
        Center coordinate (x, y) in meters (m).
    radius : float
        Radius in meters (m) defining the spatial extent of the fovea patch.
    resolution : float
        Target refined cell size in meters (m) (e.g., 0.05m).
    associated_track_id : Optional[str]
        Track ID triggering this foveation patch (if applicable).
    """

    overlay_id: str
    center: Tuple[float, float]
    radius: float
    resolution: float
    associated_track_id: Optional[str] = None
    cells: Dict[Tuple[int, int], AdaptiveCell] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.center = (float(self.center[0]), float(self.center[1]))
        self.radius = float(self.radius)
        self.resolution = float(self.resolution)
        if self.radius <= 0.0:
            raise ValueError(f"Fovea overlay radius must be positive, got {self.radius}")
        if self.resolution <= 0.0:
            raise ValueError(f"Fovea overlay resolution must be positive, got {self.resolution}")

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box (min_x, min_y, max_x, max_y) in meters."""
        cx, cy = self.center
        r = self.radius
        return (cx - r, cy - r, cx + r, cy + r)

    def contains_point(self, x: float, y: float) -> bool:
        """Test if a 2D coordinate falls within the fovea overlay circular footprint."""
        dx = x - self.center[0]
        dy = y - self.center[1]
        return bool((dx * dx + dy * dy) <= (self.radius * self.radius))

    def cell_index(self, x: float, y: float) -> Tuple[str, int, int]:
        """Compute the unique (overlay_id, col, row) key for a point within this overlay."""
        col = int(np.floor(x / self.resolution))
        row = int(np.floor(y / self.resolution))
        return (self.overlay_id, col, row)

    def cell_bounds(self, col: int, row: int) -> Tuple[float, float, float, float]:
        """Compute exact spatial bounds (min_x, min_y, max_x, max_y) in meters for (col, row)."""
        min_x = col * self.resolution
        max_x = min_x + self.resolution
        min_y = row * self.resolution
        max_y = min_y + self.resolution
        return (min_x, min_y, max_x, max_y)

    def get_or_create_cell(self, col: int, row: int) -> AdaptiveCell:
        """Retrieve or instantiate an AdaptiveCell at sub-grid coordinates (col, row)."""
        key = (col, row)
        if key in self.cells:
            return self.cells[key]

        bounds = self.cell_bounds(col, row)
        cell = AdaptiveCell(spatial_bounds=bounds, resolution=self.resolution)
        self.cells[key] = cell
        return cell

    def to_dict(self) -> Dict[str, Any]:
        """Serialize overlay configuration to JSON-serializable dictionary."""
        return {
            "overlay_id": self.overlay_id,
            "center": list(self.center),
            "radius": self.radius,
            "resolution": self.resolution,
            "associated_track_id": self.associated_track_id,
            "bounds": list(self.bounds),
            "num_cells": len(self.cells),
        }
