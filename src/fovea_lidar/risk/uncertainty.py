"""Uncertainty estimation module for Fovea-LiDAR."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from fovea_lidar.contracts.mapping import AdaptiveCell, AdaptiveMap
from fovea_lidar.contracts.objects import DetectedObject
from fovea_lidar.contracts.tracking import TrackedObject


def compute_uncertainty(
    entity: Optional[Any] = None,
    *,
    semantic_confidence: Optional[float] = None,
    occupancy_probability: Optional[float] = None,
    staleness: Optional[float] = None,
    last_updated: Optional[float] = None,
    current_time: Optional[float] = None,
    confidence: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    staleness_tau: float = 5.0,
) -> float:
    """Compute normalized uncertainty score in range [0.0, 1.0].

    Supports object-level uncertainty (1.0 - confidence), cell-level uncertainty
    combining semantic ambiguity, occupancy entropy, and observation staleness,
    or direct scalar evaluation.

    Parameters
    ----------
    entity : Optional[Any]
        Input entity to assess (AdaptiveCell, TrackedObject, DetectedObject, or float confidence).
    semantic_confidence : Optional[float]
        Explicit semantic classification confidence in [0.0, 1.0].
    occupancy_probability : Optional[float]
        Explicit occupancy probability in [0.0, 1.0]. Ambiguity peaks at 0.5.
    staleness : Optional[float]
        Explicit normalized observation staleness in [0.0, 1.0].
    last_updated : Optional[float]
        Timestamp of last observation in seconds (s).
    current_time : Optional[float]
        Current reference timestamp in seconds (s).
    confidence : Optional[float]
        Generic detection or tracking confidence score in [0.0, 1.0].
    weights : Optional[Dict[str, float]]
        Weights balancing semantic, occupancy, and staleness uncertainty.
    staleness_tau : float
        Characteristic decay time constant in seconds (s), default 5.0s.

    Returns
    -------
    float
        Normalized composite uncertainty in range [0.0, 1.0].
    """
    default_weights = {
        "semantic_weight": 0.50,
        "occupancy_weight": 0.30,
        "staleness_weight": 0.20,
    }
    w = dict(weights or default_weights)
    w_sem = w.get("semantic_weight", 0.50)
    w_occ = w.get("occupancy_weight", 0.30)
    w_stale = w.get("staleness_weight", 0.20)

    # 1. Object-level evaluation
    if isinstance(entity, (TrackedObject, DetectedObject)):
        obj_conf = float(entity.confidence)
        return float(np.clip(1.0 - obj_conf, 0.0, 1.0))

    if confidence is not None:
        return float(np.clip(1.0 - float(confidence), 0.0, 1.0))

    # Direct scalar float / int confidence
    if isinstance(entity, (int, float)):
        return float(np.clip(1.0 - float(entity), 0.0, 1.0))

    # 2. AdaptiveCell evaluation
    if isinstance(entity, AdaptiveCell):
        u_sem = 1.0 - float(entity.semantic_confidence)
        # Occupancy ambiguity: maximum (1.0) when p = 0.5, minimum (0.0) at p = 0.0 or 1.0
        p_occ = float(entity.occupancy_probability)
        u_occ = 1.0 - 2.0 * abs(p_occ - 0.5)

        u_stale = 0.0
        if current_time is not None and entity.last_updated > 0.0:
            dt = max(0.0, float(current_time) - float(entity.last_updated))
            u_stale = min(1.0, dt / max(1e-3, float(staleness_tau)))

        total_weight = w_sem + w_occ + (w_stale if current_time is not None else 0.0)
        if total_weight <= 0.0:
            return 0.0

        u_composite = (
            w_sem * u_sem
            + w_occ * u_occ
            + (w_stale * u_stale if current_time is not None else 0.0)
        ) / total_weight
        return float(np.clip(u_composite, 0.0, 1.0))

    # 3. Explicit keyword arguments evaluation
    terms: list[Tuple[float, float]] = []  # (uncertainty_value, weight)

    if semantic_confidence is not None:
        terms.append((1.0 - float(semantic_confidence), w_sem))

    if occupancy_probability is not None:
        p_occ = float(occupancy_probability)
        u_occ = 1.0 - 2.0 * abs(p_occ - 0.5)
        terms.append((u_occ, w_occ))

    if staleness is not None:
        terms.append((float(staleness), w_stale))
    elif last_updated is not None and current_time is not None:
        dt = max(0.0, float(current_time) - float(last_updated))
        u_stale = min(1.0, dt / max(1e-3, float(staleness_tau)))
        terms.append((u_stale, w_stale))

    if not terms:
        return 0.0

    sum_weights = sum(t[1] for t in terms)
    if sum_weights <= 0.0:
        return 0.0

    score = sum(val * weight for val, weight in terms) / sum_weights
    return float(np.clip(score, 0.0, 1.0))


def estimate_region_uncertainty(
    adaptive_map: Optional[AdaptiveMap],
    center: Tuple[float, float],
    radius: float = 2.0,
    current_time: Optional[float] = None,
    staleness_tau: float = 5.0,
) -> float:
    """Estimate spatial uncertainty within a local radius of an AdaptiveMap.

    Parameters
    ----------
    adaptive_map : Optional[AdaptiveMap]
        Adaptive map containing cells.
    center : Tuple[float, float]
        (x, y) coordinates of the query center in meters.
    radius : float
        Radius in meters defining the neighborhood.
    current_time : Optional[float]
        Current reference timestamp in seconds (s).
    staleness_tau : float
        Characteristic staleness time constant in seconds (s).

    Returns
    -------
    float
        Mean local cell uncertainty in range [0.0, 1.0], or 0.0 if empty.
    """
    if adaptive_map is None or not adaptive_map.cells:
        return 0.0

    cx, cy = float(center[0]), float(center[1])
    r2 = radius * radius
    cell_uncertainties: list[float] = []

    for cell in adaptive_map.cells.values():
        bx = 0.5 * (cell.spatial_bounds[0] + cell.spatial_bounds[2])
        by = 0.5 * (cell.spatial_bounds[1] + cell.spatial_bounds[3])
        if (bx - cx) ** 2 + (by - cy) ** 2 <= r2:
            unc = compute_uncertainty(
                cell,
                current_time=current_time,
                staleness_tau=staleness_tau,
            )
            cell_uncertainties.append(unc)

    if not cell_uncertainties:
        return 0.0

    return float(np.clip(np.mean(cell_uncertainties), 0.0, 1.0))
