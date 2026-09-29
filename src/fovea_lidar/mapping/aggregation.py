"""Deterministic cell aggregation logic for Fovea-LiDAR multi-resolution mapping."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union
import numpy as np

from fovea_lidar.contracts.mapping import AdaptiveCell


def aggregate_elevation(z_values: Union[np.ndarray, Sequence[float]]) -> Tuple[float, float]:
    """Compute robust elevation and variance from a collection of vertical point samples.

    Per project brief section 23: uses the median of vertical samples to ensure
    robustness against outlier returns (foliage, dust, multipath reflections).

    Parameters
    ----------
    z_values : array-like
        Array or sequence of Z-coordinates in meters (m).

    Returns
    -------
    Tuple[float, float]
        (median_elevation, elevation_variance) in meters (m) and square meters (m^2).
    """
    arr = np.asarray(z_values, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0

    if arr.size == 1:
        return float(arr[0]), 0.0

    median_val = float(np.median(arr))
    var_val = float(np.var(arr))
    return median_val, var_val


def aggregate_roughness(elevation_variance: float) -> float:
    """Compute terrain roughness from intra-cell elevation variance.

    ROUGHNESS STRATEGY (Prompt 6):
    Roughness is computed as the standard deviation of intra-cell elevation samples:
        roughness = sqrt(max(0.0, elevation_variance))
    This provides a localized, micro-terrain roughness measurement in meters (m)
    that operates on sparse single-cell observations without requiring spatial stencil
    queries across unobserved neighboring cells.

    Parameters
    ----------
    elevation_variance : float
        Variance of elevation within the cell in square meters (m^2).

    Returns
    -------
    float
        Roughness score (standard deviation of elevation in meters).
    """
    if elevation_variance <= 0.0:
        return 0.0
    return float(math.sqrt(elevation_variance))


def aggregate_occupancy(
    prev_occupancy: float,
    point_count: int,
    hit_increment: float = 0.05,
    max_occupancy: float = 0.99,
) -> float:
    """Update cell occupancy probability based on observed point hits.

    MVP OCCUPANCY RULE:
    Direct point-count accumulation capped at max_occupancy:
        P_occ = min(max_occupancy, P_prev + point_count * hit_increment)

    NOTE ON FUTURE UPGRADE:
    As documented in project brief section 23, full Bayesian log-odds updating with
    inverse sensor ray models (free space ray-casting along line of sight) will be
    integrated in downstream mapping enhancements.

    Parameters
    ----------
    prev_occupancy : float
        Current occupancy probability in range [0.0, 1.0].
    point_count : int
        Number of LiDAR points landing in this cell during the current frame.
    hit_increment : float
        Probability increase per point observation, default 0.05.
    max_occupancy : float
        Upper saturation bound for occupancy probability, default 0.99.

    Returns
    -------
    float
        Updated occupancy probability in range [0.0, max_occupancy].
    """
    if point_count <= 0:
        return float(prev_occupancy)

    new_occ = float(prev_occupancy) + int(point_count) * float(hit_increment)
    return float(min(max_occupancy, max(0.0, new_occ)))


def aggregate_semantics(
    candidates: Sequence[Tuple[str, float]],
    prev_class: Optional[str] = None,
    prev_confidence: float = 0.0,
) -> Tuple[Optional[str], float]:
    """Aggregate semantic classifications using the Highest-Confidence-Wins rule.

    Per project brief section 23:
    When a cell receives semantic observations from multiple classes (intra-frame
    or inter-frame), the class with the highest confidence score wins.
    Ties between differing classes with identical confidence are broken deterministically
    by alphabetical ordering.

    NOTE ON FUTURE UPGRADE:
    A Bayesian categorical Dirichlet-Multinomial belief update over class distributions
    is planned as a future upgrade.

    Parameters
    ----------
    candidates : Sequence[Tuple[str, float]]
        Sequence of (semantic_class, confidence) observations in current update.
    prev_class : Optional[str]
        Previously assigned semantic class of the cell.
    prev_confidence : float
        Confidence of previously assigned semantic class.

    Returns
    -------
    Tuple[Optional[str], float]
        (winning_semantic_class, winning_confidence)
    """
    if not candidates:
        return prev_class, float(prev_confidence)

    # Filter valid candidates
    valid_candidates = [(str(cls_name), float(conf)) for cls_name, conf in candidates if 0.0 <= conf <= 1.0]
    if not valid_candidates:
        return prev_class, float(prev_confidence)

    # Sort by confidence descending, then by class name ascending for deterministic tie-breaking
    sorted_candidates = sorted(valid_candidates, key=lambda x: (-x[1], x[0]))
    best_cand_class, best_cand_conf = sorted_candidates[0]

    if prev_class is None:
        return best_cand_class, best_cand_conf

    # Compare with prior cell belief
    if best_cand_conf > prev_confidence:
        return best_cand_class, best_cand_conf
    elif math.isclose(best_cand_conf, prev_confidence, rel_tol=1e-5, abs_tol=1e-5):
        # Deterministic tie-breaker between prior and new candidate
        winner = min(best_cand_class, prev_class)
        return winner, max(best_cand_conf, prev_confidence)
    else:
        return prev_class, float(prev_confidence)


def update_cell_with_points(
    cell: AdaptiveCell,
    points: np.ndarray,
    point_indices: Sequence[int],
    timestamp: float = 0.0,
    hit_increment: float = 0.05,
    semantic_candidates: Optional[Sequence[Tuple[str, float]]] = None,
) -> AdaptiveCell:
    """Aggregate a batch of newly projected points into an existing AdaptiveCell.

    Parameters
    ----------
    cell : AdaptiveCell
        Target cell to update.
    points : np.ndarray
        Point array of shape (N, 3) or (N, 4).
    point_indices : Sequence[int]
        Indices of points that fell within this cell's bounds.
    timestamp : float
        Current scan timestamp in seconds (s).
    hit_increment : float
        Occupancy step increment.
    semantic_candidates : Optional[Sequence[Tuple[str, float]]]
        Optional semantic class and confidence predictions for points in this cell.

    Returns
    -------
    AdaptiveCell
        Updated cell reference.
    """
    if not point_indices:
        return cell

    z_samples = points[point_indices, 2]
    elevation, variance = aggregate_elevation(z_samples)
    occupancy = aggregate_occupancy(cell.occupancy_probability, len(point_indices), hit_increment=hit_increment)
    roughness = aggregate_roughness(variance)

    cell.elevation = elevation
    cell.elevation_variance = variance
    cell.roughness = roughness
    cell.occupancy_probability = occupancy
    cell.last_updated = float(timestamp)

    if semantic_candidates:
        win_class, win_conf = aggregate_semantics(
            semantic_candidates,
            prev_class=cell.semantic_class,
            prev_confidence=cell.semantic_confidence,
        )
        cell.semantic_class = win_class
        cell.semantic_confidence = win_conf

    return cell
