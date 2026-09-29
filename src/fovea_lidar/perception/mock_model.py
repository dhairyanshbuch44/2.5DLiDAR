"""Deterministic, geometry-based mock semantic segmentation model for Fovea-LiDAR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree

from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.contracts.semantic import SemanticPoint
from fovea_lidar.perception.base import SegmentationModel
from fovea_lidar.utils.config import Config, load_config


class MockModel(SegmentationModel):
    """Deterministic, rule-based heuristic classifier for point cloud segmentation.

    Requires NO deep learning or GPU dependencies. Identifies geometric primitives:
    - Points near the ground plane -> ROAD (0.90 confidence)
    - Small vertical human-scale clusters -> PEDESTRIAN (0.88 confidence)
    - Medium box-scale clusters -> VEHICLE (0.90 confidence)
    - Large elongated/tall clusters -> TRUCK (0.88 confidence)
    - Narrow vertical structures -> POLE (0.80 confidence)
    - Sparse/irregular clusters -> UNKNOWN_OBSTACLE (0.40 confidence)

    Attributes
    ----------
    ground_threshold : float
        Points with |z| <= ground_threshold are classified as ROAD (meters).
    cluster_radius : float
        Spatial grouping radius for non-ground obstacle clustering (meters).
    min_cluster_points : int
        Minimum points required to consider a cluster an obstacle.
    sparse_point_threshold : int
        Point count below which a cluster is deemed ambiguous/sparse with low confidence.
    """

    def __init__(
        self,
        ground_threshold: float = 0.20,
        cluster_radius: float = 0.75,
        min_cluster_points: int = 3,
        sparse_point_threshold: int = 18,
    ) -> None:
        self.ground_threshold = float(ground_threshold)
        self.cluster_radius = float(cluster_radius)
        self.min_cluster_points = int(min_cluster_points)
        self.sparse_point_threshold = int(sparse_point_threshold)

    @classmethod
    def load(cls, config: Optional[Union[Config, Dict[str, Any], Path, str]] = None) -> MockModel:
        """Instantiate MockModel from system configuration."""
        if config is None:
            full_cfg = load_config()
        elif isinstance(config, (str, Path)):
            full_cfg = load_config(config)
        elif isinstance(config, (Config, dict)):
            full_cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

        percep_cfg = full_cfg.get("perception", {})
        if isinstance(percep_cfg, Config):
            percep_dict = percep_cfg.to_dict()
        elif isinstance(percep_cfg, dict):
            percep_dict = percep_cfg
        else:
            percep_dict = {}

        mock_params = percep_dict.get("mock_model", {})
        return cls(
            ground_threshold=float(mock_params.get("ground_threshold", 0.20)),
            cluster_radius=float(mock_params.get("cluster_radius", 0.75)),
            min_cluster_points=int(mock_params.get("min_cluster_points", 3)),
            sparse_point_threshold=int(mock_params.get("sparse_point_threshold", 18)),
        )

    def _cluster_points(self, points: np.ndarray) -> List[List[int]]:
        """Group 3D points into connected components using KD-tree neighbor graph."""
        if points.shape[0] == 0:
            return []

        tree = cKDTree(points)
        pairs = tree.query_pairs(r=self.cluster_radius)

        # Build adjacency list
        adj: Dict[int, List[int]] = {i: [] for i in range(points.shape[0])}
        for u, v in pairs:
            adj[u].append(v)
            adj[v].append(u)

        # Connected components via BFS
        visited: Set[int] = set()
        clusters: List[List[int]] = []

        for i in range(points.shape[0]):
            if i not in visited:
                comp: List[int] = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters.append(comp)

        return clusters

    def _classify_cluster(self, cluster_pts: np.ndarray) -> Tuple[str, float]:
        """Heuristically assign semantic class and confidence based on cluster geometry.

        Evaluates 3D bounding dimensions, range, aspect ratios, and spatial distribution
        with non-shadowed hierarchical rules:
        1. POLE: Narrow footprint, high verticality (tested before pedestrian).
        2. WALL: Elongated, narrow thickness, high planarity.
        3. TRUCK: Large volumetric bounding box (length > 5.5m or height > 2.2m).
        4. VEHICLE: Standard passenger car bounding volume.
        5. CYCLIST: Narrow elongated horizontal profile (aspect ratio >= 1.7, width <= 0.95m).
        6. PEDESTRIAN: Compact standing human proportions (horiz <= 1.15m, height 0.75-2.15m).
        7. UNKNOWN_OBSTACLE: Irregular, unclassified, or sparse returns.
        """
        count = cluster_pts.shape[0]
        centroid = np.mean(cluster_pts[:, :3], axis=0)
        range_dist = float(np.linalg.norm(centroid[:2]))

        # Range-adaptive sparse threshold (beam divergence scales point density ~ 1/R^2)
        range_factor = max(0.25, min(1.0, (12.0 / max(10.0, range_dist)) ** 1.5))
        eff_sparse_threshold = max(5, int(self.sparse_point_threshold * range_factor))

        if count < eff_sparse_threshold:
            conf = 0.35 if count < 4 else 0.45
            return "UNKNOWN_OBSTACLE", conf

        min_coords = np.min(cluster_pts[:, :3], axis=0)
        max_coords = np.max(cluster_pts[:, :3], axis=0)
        dimensions = max_coords - min_coords
        dx, dy, dz = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
        horiz_max = max(dx, dy)
        horiz_min = min(dx, dy)

        # Aspect ratios
        aspect_h = horiz_max / max(0.05, horiz_min)
        aspect_v = dz / max(0.05, horiz_max)

        # 3D PCA / Covariance eigenvalues (if enough points)
        planarity = 0.0
        linearity = 0.0
        if count >= 6:
            centered = cluster_pts[:, :3] - centroid
            cov = np.cov(centered, rowvar=False)
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            eigvals = np.maximum(0.0, eigvals)
            sum_e = float(np.sum(eigvals)) + 1e-6
            e1, e2, e3 = float(eigvals[0] / sum_e), float(eigvals[1] / sum_e), float(eigvals[2] / sum_e)
            planarity = (e2 - e3) / max(1e-6, e1)
            linearity = (e1 - e2) / max(1e-6, e1)

        # Rule 1: POLE (Narrow vertical structure, tested FIRST to prevent pedestrian shadowing)
        if horiz_max <= 0.35 and horiz_min <= 0.35 and dz >= 1.3:
            if aspect_v >= 2.2 or linearity >= 0.55:
                conf = min(0.92, 0.82 + 0.04 * min(2.5, aspect_v - 2.0))
                return "POLE", float(conf)

        # Rule 2: WALL (Planar elongated barrier)
        if horiz_max >= 2.8 and horiz_min <= 0.65 and dz >= 0.9:
            if aspect_h >= 3.8 or planarity >= 0.50:
                return "WALL", 0.88

        # Rule 3: TRUCK (Large commercial vehicle, bus, heavy freight)
        if horiz_max > 5.5 or (dz > 2.2 and horiz_min >= 1.5):
            conf = 0.90 if horiz_max > 6.0 else 0.88
            return "TRUCK", conf

        # Rule 4: VEHICLE (Passenger car, SUV, van)
        if 2.3 <= horiz_max <= 5.5 and 1.35 <= horiz_min <= 2.6 and 0.8 <= dz <= 2.2:
            return "VEHICLE", 0.91

        # Rule 5: CYCLIST (Bicycle / e-bike with rider: narrow width, elongated horizontal aspect)
        if 1.2 <= horiz_max <= 2.3 and 0.35 <= horiz_min <= 0.95 and 0.85 <= dz <= 1.9:
            if aspect_h >= 1.6:
                return "CYCLIST", 0.89

        # Rule 6: PEDESTRIAN (Compact standing human proportions)
        if horiz_max <= 1.15 and 0.70 <= dz <= 2.15:
            if aspect_h <= 1.85 and aspect_v >= 0.90:
                conf = 0.90 if count >= 10 else 0.84
                return "PEDESTRIAN", conf

        # Rule 7: Fallback general obstacle
        fallback_conf = 0.60 if count >= 15 else 0.48
        return "UNKNOWN_OBSTACLE", fallback_conf

    def predict(self, frame: PointCloudFrame) -> List[SemanticPoint]:
        """Classify each point in the frame using deterministic geometric heuristics.

        Parameters
        ----------
        frame : PointCloudFrame
            Raw or preprocessed LiDAR scan.

        Returns
        -------
        List[SemanticPoint]
            Point-by-point semantic annotations.
        """
        pts = frame.points
        num_points = pts.shape[0]
        if num_points == 0:
            return []

        coords = pts[:, :3]
        has_intensity = pts.shape[1] >= 4
        intensities = pts[:, 3] if has_intensity else np.full(num_points, 0.25, dtype=np.float32)
        z_vals = coords[:, 2]
        y_vals = coords[:, 1]

        labels: List[str] = ["ROAD"] * num_points
        confidences: List[float] = [0.90] * num_points

        # 1. Ground & Terrain feature classification
        # Pothole: distinct depression below road surface (-0.35m <= z <= -0.06m)
        pothole_mask = (z_vals >= -0.35) & (z_vals <= -0.06)
        # Curb: step height (+0.06m <= z <= 0.22m) with road edge characteristic (|y| >= 4.2m and intensity >= 0.35)
        curb_mask = (z_vals >= 0.06) & (z_vals <= 0.22) & (np.abs(y_vals) >= 4.2) & (intensities >= 0.35)

        for idx in np.where(pothole_mask)[0]:
            labels[idx] = "POTHOLE"
            confidences[idx] = 0.85

        for idx in np.where(curb_mask)[0]:
            labels[idx] = "CURB"
            confidences[idx] = 0.86

        # 2. Elevated Obstacle candidate points (z > ground_threshold, excluding curbs)
        non_ground_mask = (z_vals > self.ground_threshold) & (~curb_mask)
        non_ground_indices = np.where(non_ground_mask)[0]

        if len(non_ground_indices) > 0:
            non_ground_coords = coords[non_ground_indices]
            if has_intensity:
                non_ground_features = pts[non_ground_indices]
            else:
                non_ground_features = np.column_stack(
                    [non_ground_coords, np.full(len(non_ground_indices), 0.5, dtype=np.float32)]
                )

            clusters = self._cluster_points(non_ground_coords)

            for comp_indices in clusters:
                if len(comp_indices) < self.min_cluster_points:
                    for c_idx in comp_indices:
                        orig_idx = non_ground_indices[c_idx]
                        labels[orig_idx] = "UNKNOWN_OBSTACLE"
                        confidences[orig_idx] = 0.35
                    continue

                cluster_pts = non_ground_features[comp_indices]
                sem_class, conf = self._classify_cluster(cluster_pts)

                for c_idx in comp_indices:
                    orig_idx = non_ground_indices[c_idx]
                    labels[orig_idx] = sem_class
                    confidences[orig_idx] = conf

        # Construct validated SemanticPoint instances
        semantic_points: List[SemanticPoint] = []
        for i in range(num_points):
            sp = SemanticPoint(
                coordinates=(float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])),
                semantic_class=labels[i],
                confidence=confidences[i],
                timestamp=frame.timestamp,
            )
            semantic_points.append(sp)

        return semantic_points
