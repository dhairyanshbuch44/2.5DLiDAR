"""Configurable semantic class registry for Fovea-LiDAR."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Set
import logging

from fovea_lidar.utils.config import ConfigError, load_config

logger = logging.getLogger(__name__)

# Fallback classes from project brief section 9
FALLBACK_SEMANTIC_CLASSES: tuple[str, ...] = (
    "ROAD",
    "SIDEWALK",
    "GRASS",
    "OTHER_TERRAIN",
    "CURB",
    "POTHOLE",
    "VEHICLE",
    "PEDESTRIAN",
    "TRUCK",
    "WALL",
    "POLE",
    "VEGETATION",
    "UNKNOWN_OBSTACLE",
)


class InvalidSemanticClassError(ValueError):
    """Raised when a semantic class name is not recognized in the registry."""


class SemanticClassRegistry:
    """Dynamic registry for valid semantic classes loaded from configuration."""

    def __init__(
        self,
        initial_classes: Optional[Iterable[str]] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self._classes: Set[str] = set()

        if initial_classes is not None:
            for cls_name in initial_classes:
                self.register(cls_name)
        else:
            self._load_from_config_or_fallback(config_path)

    def _load_from_config_or_fallback(self, config_path: Optional[str | Path] = None) -> None:
        """Attempt loading allowed semantic classes from default config; fallback to defaults."""
        try:
            cfg = load_config(config_path)
            raw_classes = cfg.get("semantic_classes")
            if raw_classes and isinstance(raw_classes, list):
                for cls_name in raw_classes:
                    self.register(str(cls_name))
                return
        except ConfigError as exc:
            logger.debug("Failed loading semantic classes from config (%s); using fallbacks.", exc)
        except Exception as exc:
            logger.debug("Unexpected error loading config (%s); using fallbacks.", exc)

        for cls_name in FALLBACK_SEMANTIC_CLASSES:
            self.register(cls_name)

    def register(self, class_name: str) -> None:
        """Register a new valid semantic class name."""
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError(f"Class name must be a non-empty string, got {class_name!r}")
        self._classes.add(class_name.strip().upper())

    def unregister(self, class_name: str) -> None:
        """Unregister a semantic class name."""
        canonical = class_name.strip().upper()
        self._classes.discard(canonical)

    def is_valid(self, class_name: str) -> bool:
        """Check if a semantic class name is valid."""
        if not isinstance(class_name, str):
            return False
        return class_name.strip().upper() in self._classes

    def validate(self, class_name: str) -> str:
        """Validate and return canonical class name, or raise InvalidSemanticClassError."""
        if not isinstance(class_name, str):
            raise InvalidSemanticClassError(
                f"Class name must be a string, got {type(class_name).__name__}: {class_name!r}"
            )
        canonical = class_name.strip().upper()
        if canonical not in self._classes:
            allowed_sorted = sorted(self._classes)
            raise InvalidSemanticClassError(
                f"Invalid semantic class '{class_name}'. Allowed classes are: {allowed_sorted}"
            )
        return canonical

    def get_classes(self) -> Set[str]:
        """Return a copy of all registered class names."""
        return set(self._classes)

    def reset(self, new_classes: Optional[Iterable[str]] = None) -> None:
        """Reset the registry to new classes or default fallback classes."""
        self._classes.clear()
        if new_classes is not None:
            for cls_name in new_classes:
                self.register(cls_name)
        else:
            self._load_from_config_or_fallback()


# Global default registry instance
DEFAULT_REGISTRY: SemanticClassRegistry = SemanticClassRegistry()


def get_default_registry() -> SemanticClassRegistry:
    """Return the global default semantic class registry."""
    return DEFAULT_REGISTRY


def validate_class(name: str, registry: Optional[SemanticClassRegistry] = None) -> str:
    """Validate a semantic class against the active registry.

    Parameters
    ----------
    name : str
        The semantic class name to validate.
    registry : SemanticClassRegistry, optional
        Custom registry instance, or DEFAULT_REGISTRY if None.

    Returns
    -------
    str
        Canonicalized uppercase class name.

    Raises
    ------
    InvalidSemanticClassError
        If the class name is not recognized.
    """
    active_reg = registry if registry is not None else DEFAULT_REGISTRY
    return active_reg.validate(name)


def register_class(name: str, registry: Optional[SemanticClassRegistry] = None) -> None:
    """Register a semantic class into the active registry."""
    active_reg = registry if registry is not None else DEFAULT_REGISTRY
    active_reg.register(name)
