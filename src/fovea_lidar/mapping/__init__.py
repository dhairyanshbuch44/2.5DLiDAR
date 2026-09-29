"""Adaptive multi-resolution mapping engine and aggregation module for Fovea-LiDAR."""

from fovea_lidar.contracts.mapping import (
    AdaptiveCell,
    AdaptiveMap as AdaptiveMapContract,
    TraversabilityMap,
)
from fovea_lidar.mapping.adaptive_map import AdaptiveMap
from fovea_lidar.mapping.aggregation import (
    aggregate_elevation,
    aggregate_occupancy,
    aggregate_roughness,
    aggregate_semantics,
    update_cell_with_points,
)
from fovea_lidar.mapping.grid_engine import (
    AdaptiveGridEngine,
    CellKey,
    ResolutionZone,
)

__all__ = [
    # Engine & Map
    "AdaptiveGridEngine",
    "AdaptiveMap",
    "AdaptiveMapContract",
    "CellKey",
    "ResolutionZone",
    # Aggregation
    "aggregate_elevation",
    "aggregate_occupancy",
    "aggregate_roughness",
    "aggregate_semantics",
    "update_cell_with_points",
    # Contracts
    "AdaptiveCell",
    "TraversabilityMap",
]
