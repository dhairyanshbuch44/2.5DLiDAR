"""I/O loaders, synthetic scene generators, and playback controllers for Fovea-LiDAR."""

from fovea_lidar.io.loaders import (
    CorruptFileError,
    EmptyPointCloudError,
    PointCloudFileNotFoundError,
    PointCloudIOError,
    load,
    load_pcd,
    load_ply,
    save_pcd,
    save_ply,
)
from fovea_lidar.io.replay import ReplayController
from fovea_lidar.io.synthetic import SyntheticSceneGenerator

__all__ = [
    # Loaders
    "load",
    "load_pcd",
    "load_ply",
    "save_pcd",
    "save_ply",
    "PointCloudIOError",
    "PointCloudFileNotFoundError",
    "CorruptFileError",
    "EmptyPointCloudError",
    # Synthetic Generator
    "SyntheticSceneGenerator",
    # Replay Controller
    "ReplayController",
]
