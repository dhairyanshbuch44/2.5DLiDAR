"""Stable data contracts for the Fovea-LiDAR architecture."""

from fovea_lidar.contracts.mapping import (
    AdaptiveCell,
    AdaptiveMap,
    TraversabilityMap,
)
from fovea_lidar.contracts.objects import (
    DetectedObject,
)
from fovea_lidar.contracts.point_cloud import (
    FrameSource,
    PointCloudFrame,
)
from fovea_lidar.contracts.preprocessing import (
    PreprocessingMetrics,
)
from fovea_lidar.contracts.risk import (
    RiskLevel,
    RiskRegion,
)
from fovea_lidar.contracts.semantic import (
    SemanticPoint,
)
from fovea_lidar.contracts.semantic_classes import (
    DEFAULT_REGISTRY,
    FALLBACK_SEMANTIC_CLASSES,
    InvalidSemanticClassError,
    SemanticClassRegistry,
    get_default_registry,
    register_class,
    validate_class,
)
from fovea_lidar.contracts.benchmark_result import BenchmarkResult
from fovea_lidar.contracts.pipeline_result import PipelineResult
from fovea_lidar.contracts.tracking import (
    TrackedObject,
)

__all__ = [
    # Pipeline & Benchmarking
    "PipelineResult",
    "BenchmarkResult",
    # Point Cloud & Preprocessing
    "PointCloudFrame",
    "FrameSource",
    "PreprocessingMetrics",
    # Semantic
    "SemanticPoint",
    "SemanticClassRegistry",
    "validate_class",
    "register_class",
    "get_default_registry",
    "DEFAULT_REGISTRY",
    "FALLBACK_SEMANTIC_CLASSES",
    "InvalidSemanticClassError",
    # Objects & Tracking
    "DetectedObject",
    "TrackedObject",
    # Mapping & Traversability
    "AdaptiveCell",
    "AdaptiveMap",
    "TraversabilityMap",
    # Risk
    "RiskRegion",
    "RiskLevel",
]
