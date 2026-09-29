"""Object extraction module for Fovea-LiDAR.

Converts point-level semantic predictions (SemanticPoint) into object-level
DetectedObject instances via Euclidean distance clustering, dominant class
determination, centroid and axis-aligned bounding box (AABB) computation,
and initial static/dynamic guesses.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree

from fovea_lidar.contracts.objects import DetectedObject
from fovea_lidar.contracts.semantic import SemanticPoint
from fovea_lidar.utils.config import Config, load_config


class ObjectExtractor:
    """Extracts object-level DetectedObject instances from semantic point clouds.

    Attributes
    ----------
    cluster_distance_threshold : float
        Maximum Euclidean distance in meters between points to consider them
        part of the same spatial cluster.
    min_cluster_points : int
        Minimum number of points required for a cluster to form a valid detected
        object. Degenerate clusters with fewer points are discarded as noise.
    object_classes : Set[str]
        Semantic classes eligible for object extraction. Ground/terrain classes
        (ROAD, SIDEWALK, GRASS, OTHER_TERRAIN, CURB) are ignored.
    dynamic_classes : Set[str]
        Semantic classes assigned an initial dynamic guess of True (e.g. VEHICLE,
        PEDESTRIAN, TRUCK). Static obstacles (WALL, POLE) default to False.
    """

    def __init__(
        self,
        cluster_distance_threshold: float = 1.0,
        min_cluster_points: int = 10,
        object_classes: Optional[Sequence[str]] = None,
        dynamic_classes: Optional[Sequence[str]] = None,
    ) -> None:
        self.cluster_distance_threshold = float(cluster_distance_threshold)
        self.min_cluster_points = int(min_cluster_points)

        default_object_classes = {
            "VEHICLE",
            "PEDESTRIAN",
            "CYCLIST",
            "TRUCK",
            "WALL",
            "POLE",
            "UNKNOWN_OBSTACLE",
        }
        self.object_classes: Set[str] = (
            set(object_classes) if object_classes is not None else default_object_classes
        )

        default_dynamic_classes = {"VEHICLE", "PEDESTRIAN", "TRUCK", "CYCLIST"}
        self.dynamic_classes: Set[str] = (
            set(dynamic_classes) if dynamic_classes is not None else default_dynamic_classes
        )

        self._counter: int = 0
        self.last_extracted_clusters: List[List[SemanticPoint]] = []

    def _next_id(self) -> str:
        """Generate a provisional object ID for a detection.

        NOTE ON PROVISIONAL OBJECT IDS:
        Provisional IDs (e.g. 'det_1', 'det_2') are valid strictly within the scope
        of the single extraction frame. They are NOT stable across frames.
        Cross-frame persistent ID association, trajectory maintenance, and track
        lifecycle management is handled downstream by the multi-object tracker (Prompt 10).
        """
        self._counter += 1
        return f"det_{self._counter}"

    @classmethod
    def load(
        cls, config: Optional[Union[Config, Dict[str, Any], Path, str]] = None
    ) -> ObjectExtractor:
        """Instantiate ObjectExtractor from system configuration."""
        if config is None:
            full_cfg = load_config()
        elif isinstance(config, (str, Path)):
            full_cfg = load_config(config)
        elif isinstance(config, (Config, dict)):
            full_cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

        obj_cfg = full_cfg.get("objects", {})
        if isinstance(obj_cfg, Config):
            obj_dict = obj_cfg.to_dict()
        elif isinstance(obj_cfg, dict):
            obj_dict = obj_cfg
        else:
            obj_dict = {}

        return cls(
            cluster_distance_threshold=float(obj_dict.get("cluster_distance_threshold", 1.0)),
            min_cluster_points=int(obj_dict.get("min_cluster_points", 10)),
            object_classes=obj_dict.get("object_classes"),
            dynamic_classes=obj_dict.get("dynamic_classes"),
        )

    def _cluster_points(self, coords: np.ndarray) -> List[List[int]]:
        """Group points into spatial clusters using cKDTree query pairs and BFS."""
        if coords.shape[0] == 0:
            return []

        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=self.cluster_distance_threshold)

        adj: Dict[int, List[int]] = {i: [] for i in range(coords.shape[0])}
        for u, v in pairs:
            adj[u].append(v)
            adj[v].append(u)

        visited: Set[int] = set()
        clusters: List[List[int]] = []

        for i in range(coords.shape[0]):
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

    def _determine_dominant_class(
        self, cluster_points: Sequence[SemanticPoint]
    ) -> Tuple[str, float]:
        """Determine the dominant semantic class and aggregated confidence for a cluster.

        Methodology:
        Dominant semantic class is determined via confidence-weighted probability mass:
            score(c) = sum_{p in cluster: class(p)==c} p.confidence
        In the event of an ambiguous score tie within 1e-4, the tie is broken by comparing
        the cluster's geometric bounding dimensions against canonical class dimensional
        priors, avoiding systematic bias towards any single class.
        Aggregated confidence is the mean confidence of the dominant class's member points.
        """
        class_scores: Dict[str, float] = {}
        class_counts: Dict[str, int] = {}
        class_confidences: Dict[str, List[float]] = {}
        for p in cluster_points:
            c = p.semantic_class
            class_scores[c] = class_scores.get(c, 0.0) + float(p.confidence)
            class_counts[c] = class_counts.get(c, 0) + 1
            class_confidences.setdefault(c, []).append(float(p.confidence))

        # Geometric bounding box of the cluster for tie-breaking
        pts_coords = np.array([p.coordinates for p in cluster_points], dtype=np.float32)
        dims = np.max(pts_coords, axis=0) - np.min(pts_coords, axis=0)
        dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
        horiz_extent = max(dx, dy)
        horiz_width = min(dx, dy)

        # Canonical characteristic dimensions: (horiz_extent, horiz_width, dz)
        canonical_dims = {
            "PEDESTRIAN": (0.6, 0.6, 1.7),
            "CYCLIST": (1.7, 0.6, 1.5),
            "POLE": (0.3, 0.3, 2.5),
            "VEHICLE": (4.2, 1.9, 1.5),
            "TRUCK": (7.5, 2.5, 3.0),
            "WALL": (5.0, 0.4, 2.0),
            "UNKNOWN_OBSTACLE": (1.0, 1.0, 1.0),
        }

        def geometric_fit_score(cls_name: str) -> float:
            target = canonical_dims.get(cls_name, (1.0, 1.0, 1.0))
            err = (
                abs(np.log(max(0.1, horiz_extent) / target[0]))
                + abs(np.log(max(0.1, horiz_width) / target[1]))
                + abs(np.log(max(0.1, dz) / target[2]))
            )
            return -float(err)

        def sort_key(cls_name: str) -> Tuple[float, int, float]:
            score = round(class_scores[cls_name], 4)
            count = class_counts[cls_name]
            fit = geometric_fit_score(cls_name)
            return (score, count, fit)

        dominant_class = max(class_scores.keys(), key=sort_key)
        aggregated_confidence = float(np.mean([p.confidence for p in cluster_points]))

        return dominant_class, aggregated_confidence

    def extract(self, semantic_points: Sequence[SemanticPoint]) -> List[DetectedObject]:
        """Extract DetectedObject instances from labeled semantic points.

        Parameters
        ----------
        semantic_points : Sequence[SemanticPoint]
            Input point-level semantic annotations.

        Returns
        -------
        List[DetectedObject]
            Extracted object detections with AABB, centroid, confidence, and dynamic flag.
        """
        self.last_extracted_clusters.clear()
        if not semantic_points:
            return []

        # Step 1: Filter for points belonging to object-like classes
        candidate_points = [
            p for p in semantic_points if p.semantic_class in self.object_classes
        ]
        if not candidate_points:
            return []

        coords = np.array([p.coordinates for p in candidate_points], dtype=np.float64)

        # Step 2: Spatial Euclidean distance clustering
        clusters_indices = self._cluster_points(coords)

        # Step 3: Compute object attributes and filter degenerate clusters
        detected_objects: List[DetectedObject] = []

        for comp in clusters_indices:
            # Discard degenerate noise clusters
            if len(comp) < self.min_cluster_points:
                continue

            member_pts = [candidate_points[i] for i in comp]
            self.last_extracted_clusters.append(member_pts)

            member_coords = coords[comp]
            centroid = np.mean(member_coords, axis=0)
            min_bounds = np.min(member_coords, axis=0)
            max_bounds = np.max(member_coords, axis=0)

            dominant_class, agg_conf = self._determine_dominant_class(member_pts)

            # Static vs. dynamic initial guess
            is_dynamic = dominant_class in self.dynamic_classes

            # Observation timestamp
            timestamp = member_pts[0].timestamp

            det_obj = DetectedObject(
                object_id=self._next_id(),
                semantic_class=dominant_class,
                position=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                bounding_region=(
                    float(min_bounds[0]),
                    float(min_bounds[1]),
                    float(min_bounds[2]),
                    float(max_bounds[0]),
                    float(max_bounds[1]),
                    float(max_bounds[2]),
                ),
                confidence=agg_conf,
                velocity=None,  # Estimated downstream by tracker
                dynamic=is_dynamic,
                timestamp=timestamp,
            )
            detected_objects.append(det_obj)

        return detected_objects
