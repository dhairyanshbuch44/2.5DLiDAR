"""Operational 2.5D Adaptive Map wrapping the multi-resolution grid engine."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from fovea_lidar.contracts.mapping import AdaptiveCell
from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.contracts.semantic import SemanticPoint
from fovea_lidar.mapping.aggregation import update_cell_with_points
from fovea_lidar.mapping.grid_engine import AdaptiveGridEngine, CellKey
from fovea_lidar.utils.config import Config


class AdaptiveMap:
    """Stateful 2.5D multi-resolution adaptive map.

    Wraps AdaptiveGridEngine and executes the frame-to-frame point cloud integration
    cycle. Merges geometric (elevation, variance, roughness, occupancy) and semantic
    (class, confidence) observations in place.

    Attributes
    ----------
    grid_engine : AdaptiveGridEngine
        Underlying concentric multi-resolution grid spatial index.
    frame_count : int
        Total number of LiDAR scans integrated into the map.
    last_update_time : float
        Timestamp of the most recently processed frame in seconds (s).
    hit_increment : float
        Per-point occupancy accumulation step.
    """

    def __init__(
        self,
        config: Optional[Union[Config, Dict[str, Any], Path, str]] = None,
        grid_engine: Optional[AdaptiveGridEngine] = None,
        hit_increment: float = 0.05,
    ) -> None:
        self.grid_engine: AdaptiveGridEngine = grid_engine or AdaptiveGridEngine(config=config)
        self.frame_count: int = 0
        self.last_update_time: float = 0.0
        self.hit_increment: float = float(hit_increment)

    @property
    def cells(self) -> Dict[CellKey, AdaptiveCell]:
        """Dictionary of currently materialized cells keyed by CellKey (band, col, row)."""
        return self.grid_engine._cells

    @property
    def total_cells(self) -> int:
        """Count of allocated cells in the sparse grid."""
        return len(self.grid_engine._cells)

    def reset(self) -> None:
        """Reset map cells, fovea overlays, and frame counters."""
        self.grid_engine.reset()
        self.frame_count = 0
        self.last_update_time = 0.0

    def update(
        self,
        frame: PointCloudFrame,
        semantic_points: Optional[List[SemanticPoint]] = None,
    ) -> None:
        """Integrate a point cloud frame and optional semantic predictions into the map.

        When semantic_points is None (Prompt 4-6 standalone operation), only
        geometric properties (elevation, variance, roughness, occupancy) are updated.

        Parameters
        ----------
        frame : PointCloudFrame
            Raw or preprocessed point cloud frame.
        semantic_points : Optional[List[SemanticPoint]]
            Optional list of classified 3D points matching or within this frame.
        """
        self.frame_count += 1
        self.last_update_time = float(frame.timestamp)

        if frame.points.shape[0] == 0:
            return

        # 1. Project points into multi-resolution cells
        projection = self.grid_engine.project_points(frame.points)

        # 2. Map semantic predictions to cells if provided
        cell_semantics: Dict[CellKey, List[Tuple[str, float]]] = defaultdict(list)
        if semantic_points:
            for sp in semantic_points:
                key = self.grid_engine.cell_index(sp.coordinates[0], sp.coordinates[1])
                if key is not None:
                    cell_semantics[key].append((sp.semantic_class, sp.confidence))

        # 3. Update each cell with points and semantic candidates
        for cell_key, point_indices in projection.items():
            cell = self.grid_engine.get_or_create_cell(cell_key)
            candidates = cell_semantics.get(cell_key)
            update_cell_with_points(
                cell=cell,
                points=frame.points,
                point_indices=point_indices,
                timestamp=frame.timestamp,
                hit_increment=self.hit_increment,
                semantic_candidates=candidates,
            )

    def get_stale_cells(self, current_time: float, max_age: float) -> List[CellKey]:
        """Find all cells that have not been updated within max_age seconds.

        Used for track coasting, obstacle decay, and dashboard cache eviction.

        Parameters
        ----------
        current_time : float
            Current reference timestamp in seconds (s).
        max_age : float
            Age threshold in seconds (s) beyond which a cell is deemed stale.

        Returns
        -------
        List[CellKey]
            Keys of cells satisfying: (current_time - cell.last_updated) > max_age.
        """
        stale_keys: List[CellKey] = []
        curr_t = float(current_time)
        age_thresh = float(max_age)

        for key, cell in self.cells.items():
            if (curr_t - cell.last_updated) > age_thresh:
                stale_keys.append(key)

        return stale_keys

    def prune_stale_cells(self, max_age: float = 2.0) -> int:
        """Prune cells that have not been updated within max_age seconds.

        Parameters
        ----------
        max_age : float
            Age threshold in seconds (s).

        Returns
        -------
        int
            Count of pruned stale cells.
        """
        stale_keys = self.get_stale_cells(self.last_update_time, max_age)
        for k in stale_keys:
            self.grid_engine._cells.pop(k, None)
        return len(stale_keys)

    def clear(self) -> None:
        """Reset the map state and wipe all cells."""
        self.grid_engine.clear()
        self.frame_count = 0
        self.last_update_time = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export map state to a clean JSON-serializable dictionary.

        Formats cell list and metadata suitable for JSON export and dashboard streaming.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'cells' list and 'metadata'.
        """
        cell_list: List[Dict[str, Any]] = []
        for key, cell in self.cells.items():
            d = cell.to_dict()
            d["cell_key"] = list(key)
            cell_list.append(d)

        zones_data = [
            {
                "index": z.index,
                "name": z.name,
                "distance_min": z.distance_min,
                "distance_max": z.distance_max,
                "target_resolution": z.target_resolution,
            }
            for z in self.grid_engine.zones
        ]

        return {
            "cells": cell_list,
            "metadata": {
                "frame_count": self.frame_count,
                "last_update_time": self.last_update_time,
                "total_cells": len(cell_list),
                "resolution_zones": zones_data,
            },
        }
