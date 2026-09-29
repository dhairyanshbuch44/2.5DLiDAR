"""Individual, testable point cloud preprocessing stages for Fovea-LiDAR."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree

try:
    import open3d as o3d
except ImportError:
    o3d = None


def coordinate_transform(
    points: np.ndarray,
    transform_matrix: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
) -> np.ndarray:
    """Apply a 4x4 homogeneous transformation matrix to 3D point coordinates.

    Parameters
    ----------
    points : np.ndarray
        Point array of shape (N, 3) or (N, 4).
    transform_matrix : array-like of shape (4, 4), optional
        Homogeneous affine transformation matrix. If None, returns unmodified points.

    Returns
    -------
    np.ndarray
        Transformed points array (same shape and dtype as input).
    """
    if points.shape[0] == 0 or transform_matrix is None:
        return points

    matrix = np.asarray(transform_matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Transform matrix must be 4x4, got shape {matrix.shape}")

    # Check for identity
    if np.allclose(matrix, np.eye(4, dtype=np.float32)):
        return points

    coords = points[:, :3]
    # Homogeneous coordinates (N, 4)
    ones = np.ones((coords.shape[0], 1), dtype=np.float32)
    homo = np.hstack([coords, ones])

    # Apply transform: (matrix @ homo.T).T
    transformed_xyz = (homo @ matrix.T)[:, :3]

    if points.shape[1] > 3:
        # Preserve intensity / extra channels untouched
        return np.column_stack([transformed_xyz, points[:, 3:]]).astype(points.dtype)
    return transformed_xyz.astype(points.dtype)


def range_filter(
    points: np.ndarray,
    min_range: float = 0.5,
    max_range: float = 50.0,
) -> np.ndarray:
    """Filter points by Euclidean distance from the sensor origin.

    Parameters
    ----------
    points : np.ndarray
        Point array of shape (N, 3) or (N, 4) with (x, y, z) in meters.
    min_range : float
        Minimum radial distance in meters.
    max_range : float
        Maximum radial distance in meters.

    Returns
    -------
    np.ndarray
        Filtered point array within [min_range, max_range].
    """
    if points.shape[0] == 0:
        return points

    distances = np.linalg.norm(points[:, :3], axis=1)
    mask = (distances >= float(min_range)) & (distances <= float(max_range))
    return points[mask]


def height_filter(
    points: np.ndarray,
    min_z: float = -3.0,
    max_z: float = 4.0,
) -> np.ndarray:
    """Filter points by vertical elevation bounds.

    Parameters
    ----------
    points : np.ndarray
        Point array with z-coordinate in column index 2 (meters).
    min_z : float
        Lower height threshold in meters.
    max_z : float
        Upper height threshold in meters.

    Returns
    -------
    np.ndarray
        Filtered point array with min_z <= z <= max_z.
    """
    if points.shape[0] == 0:
        return points

    z_vals = points[:, 2]
    mask = (z_vals >= float(min_z)) & (z_vals <= float(max_z))
    return points[mask]


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float = 0.10,
) -> np.ndarray:
    """Downsample point cloud via regular 3D voxel grid.

    Computes the geometric centroid of all points falling inside each voxel cell.

    Parameters
    ----------
    points : np.ndarray
        Point array of shape (N, 3) or (N, 4) in meters.
    voxel_size : float
        Edge length of voxel grid cube in meters. Must be > 0.

    Returns
    -------
    np.ndarray
        Downsampled point array containing one centroid per occupied voxel.
    """
    if points.shape[0] == 0:
        return points

    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be strictly positive, got {voxel_size}")

    # Use Open3D if installed and available
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        down_pts = np.asarray(down.points, dtype=points.dtype)
        if points.shape[1] > 3:
            # Assign dummy intensity (mean) or 1.0 for decimated points
            mean_intensity = float(np.mean(points[:, 3])) if points.shape[0] > 0 else 0.5
            return np.column_stack([down_pts, np.full((down_pts.shape[0], 1), mean_intensity, dtype=points.dtype)])
        return down_pts

    # Pure NumPy vectorized voxel downsampling (computing exact voxel centroids)
    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    counts = np.bincount(inverse_indices)
    num_voxels = len(counts)
    sums = np.zeros((num_voxels, points.shape[1]), dtype=np.float64)

    for col in range(points.shape[1]):
        sums[:, col] = np.bincount(inverse_indices, weights=points[:, col].astype(np.float64))

    downsampled = (sums / counts[:, None]).astype(points.dtype)
    return downsampled


def statistical_outlier_removal(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Filter out statistical outliers based on mean k-nearest neighbor distance distribution.

    Parameters
    ----------
    points : np.ndarray
        Point array of shape (N, 3) or (N, 4) in meters.
    nb_neighbors : int
        Number of nearest neighbors to evaluate.
    std_ratio : float
        Standard deviation multiplier threshold.

    Returns
    -------
    np.ndarray
        Cleaned point array with statistical outliers removed.
    """
    if points.shape[0] <= nb_neighbors:
        return points

    # Use Open3D if available
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        _, inliers = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
        return points[inliers]

    # Vectorized SciPy cKDTree implementation
    tree = cKDTree(points[:, :3])
    k = min(int(nb_neighbors) + 1, len(points))
    distances, _ = tree.query(points[:, :3], k=k)

    # Exclude distance to self (column 0 is 0.0)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    global_mean = float(np.mean(mean_dist))
    global_std = float(np.std(mean_dist))
    threshold = global_mean + float(std_ratio) * global_std

    inlier_mask = mean_dist <= threshold
    return points[inlier_mask]


def radius_outlier_removal(
    points: np.ndarray,
    radius: float = 0.5,
    min_neighbors: int = 4,
) -> np.ndarray:
    """Filter out points that have fewer than min_neighbors within a given search radius.

    Parameters
    ----------
    points : np.ndarray
        Point array of shape (N, 3) or (N, 4) in meters.
    radius : float
        Search ball radius in meters.
    min_neighbors : int
        Minimum neighbor count required to retain a point.

    Returns
    -------
    np.ndarray
        Cleaned point array with isolated points removed.
    """
    if points.shape[0] <= min_neighbors:
        return points

    # Use Open3D if available
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        _, inliers = pcd.remove_radius_outlier(nb_points=int(min_neighbors), radius=float(radius))
        return points[inliers]

    # Vectorized SciPy cKDTree implementation
    tree = cKDTree(points[:, :3])
    # query_ball_point includes self
    counts = tree.query_ball_point(points[:, :3], r=float(radius), return_sorted=False)
    inlier_mask = np.array([len(c) >= (int(min_neighbors) + 1) for c in counts], dtype=bool)
    return points[inlier_mask]


def ground_candidate_extraction(
    points: np.ndarray,
    distance_threshold: float = 0.15,
    max_iterations: int = 100,
    upright_normal_threshold: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify ground-plane inlier candidate points using RANSAC plane fitting.

    NOTE: Does NOT drop non-ground points; returns the original points alongside
    a boolean mask indicating ground candidates for downstream metadata attachment.

    Parameters
    ----------
    points : np.ndarray
        Point array of shape (N, 3) or (N, 4).
    distance_threshold : float
        Maximum distance in meters to the fitted plane to be classified as ground.
    max_iterations : int
        Maximum RANSAC iterations.
    upright_normal_threshold : float
        Minimum absolute value of normal z-component to accept plane as horizontal ground.
    seed : int
        Random seed for reproducible RANSAC.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (points, ground_mask) where ground_mask is a boolean array of length N.
    """
    num_pts = points.shape[0]
    if num_pts < 3:
        return points, np.zeros(num_pts, dtype=bool)

    # Use Open3D if available
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=float(distance_threshold),
                ransac_n=3,
                num_iterations=int(max_iterations),
            )
            a, b, c, d = plane_model
            normal = np.array([a, b, c], dtype=np.float32)
            normal /= (np.linalg.norm(normal) + 1e-8)
            if abs(normal[2]) >= upright_normal_threshold:
                mask = np.zeros(num_pts, dtype=bool)
                mask[inliers] = True
                return points, mask
        except Exception:
            pass

    # Pure NumPy RANSAC plane fitting
    rng = np.random.RandomState(seed)
    best_inlier_count = 0
    best_mask = np.zeros(num_pts, dtype=bool)
    coords = points[:, :3]

    for _ in range(max_iterations):
        sample_indices = rng.choice(num_pts, size=3, replace=False)
        p1, p2, p3 = coords[sample_indices]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue

        normal /= norm_len
        # Ensure plane normal is primarily vertical (ground orientation)
        if abs(normal[2]) < upright_normal_threshold:
            continue

        d = -np.dot(normal, p1)
        # Distance of all points to plane: |n . p + d|
        distances = np.abs(np.dot(coords, normal) + d)
        inliers = distances <= distance_threshold
        inlier_count = np.count_nonzero(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_mask = inliers

    return points, best_mask
