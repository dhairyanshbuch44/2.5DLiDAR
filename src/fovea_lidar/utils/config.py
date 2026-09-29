"""Configuration loader and schema validator for Fovea-LiDAR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import yaml


REQUIRED_TOP_LEVEL_SECTIONS = (
    "mapping",
    "resolution_zones",
    "semantic_foveation",
    "risk_refinement",
    "uncertainty_refinement",
)


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or contains invalid keys."""


class Config:
    """Hierarchical configuration object supporting attribute and subscript access."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise ConfigError(f"Config data must be a mapping, got {type(data).__name__}")

        self._data: Dict[str, Any] = {}
        for key, value in data.items():
            self._data[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str):
            raise ConfigError(f"Config key must be a string, got {type(key).__name__}")

        if "." in key:
            parts = key.split(".")
            curr: Any = self
            for part in parts:
                if isinstance(curr, Config):
                    curr = curr[part]
                elif isinstance(curr, Mapping):
                    if part not in curr:
                        raise ConfigError(f"Nested key '{part}' not found in key path '{key}'")
                    curr = curr[part]
                else:
                    raise ConfigError(f"Cannot traverse into non-mapping value at '{part}' in '{key}'")
            return curr

        if key not in self._data:
            raise ConfigError(f"Configuration key '{key}' not found")
        return self._data[key]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._data:
            raise ConfigError(f"Configuration key '{name}' not found")
        return self._data[name]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if "." in key:
            parts = key.split(".")
            curr: Any = self
            for part in parts:
                if isinstance(curr, Config) and part in curr._data:
                    curr = curr._data[part]
                elif isinstance(curr, Mapping) and part in curr:
                    curr = curr[part]
                else:
                    return False
            return True
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key or dot-delimited path, returning default if not found."""
        try:
            return self[key]
        except ConfigError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a plain Python dictionary."""
        result: Dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, Config) else item for item in value]
            else:
                result[key] = value
        return result

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def values(self) -> Iterable[Any]:
        return self._data.values()

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._data.items()

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self.to_dict() == dict(other)
        return False


def _find_default_config_path() -> Path:
    """Find the default config file path relative to repo or current working directory."""
    cwd_candidate = Path.cwd() / "configs" / "default.yaml"
    if cwd_candidate.is_file():
        return cwd_candidate

    repo_candidate = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
    if repo_candidate.is_file():
        return repo_candidate

    return cwd_candidate


def validate_config(config_data: Union[Config, Mapping[str, Any]]) -> None:
    """Validate configuration schema, required top-level sections, and numeric types.

    Parameters
    ----------
    config_data : Config or Mapping[str, Any]
        Configuration mapping to validate.

    Raises
    ------
    ConfigError
        If required sections are missing or numeric fields have invalid types or values.
    """
    if not isinstance(config_data, (Config, Mapping)):
        raise ConfigError(f"Configuration must be a mapping, got {type(config_data).__name__}")

    raw = config_data.to_dict() if isinstance(config_data, Config) else dict(config_data)

    # 1. Required top-level sections
    missing_sections = [s for s in REQUIRED_TOP_LEVEL_SECTIONS if s not in raw]
    if missing_sections:
        raise ConfigError(
            f"Configuration is missing required top-level section(s): {', '.join(missing_sections)}"
        )

    # 2. Validate mapping section types
    mapping_sec = raw.get("mapping")
    if not isinstance(mapping_sec, (Config, Mapping)):
        raise ConfigError(f"'mapping' section must be a mapping, got {type(mapping_sec).__name__}")

    for field in ("min_resolution", "max_resolution", "max_range", "min_range"):
        if field in mapping_sec:
            val = mapping_sec[field]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ConfigError(
                    f"Configuration field 'mapping.{field}' must be a numeric value, got {type(val).__name__} ({val!r})"
                )
            if float(val) <= 0:
                raise ConfigError(
                    f"Configuration field 'mapping.{field}' must be positive, got {val}"
                )

    # 3. Validate semantic_foveation section types
    fovea_sec = raw.get("semantic_foveation")
    if isinstance(fovea_sec, (Config, Mapping)):
        for field in ("min_resolution", "max_resolution", "alpha", "gamma", "fovea_radius"):
            if field in fovea_sec:
                val = fovea_sec[field]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ConfigError(
                        f"Configuration field 'semantic_foveation.{field}' must be a numeric value, got {type(val).__name__} ({val!r})"
                    )
                if float(val) <= 0 and field not in ("alpha", "gamma"):
                    raise ConfigError(
                        f"Configuration field 'semantic_foveation.{field}' must be positive, got {val}"
                    )


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """Load, parse, and validate a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the configuration YAML file. If not provided,
        defaults to 'configs/default.yaml'.

    Returns
    -------
    Config
        Validated configuration instance.

    Raises
    ------
    ConfigError
        If the file does not exist, cannot be parsed as YAML,
        is missing required top-level sections, or contains invalid types.
    """
    if config_path is None:
        resolved_path = _find_default_config_path()
    else:
        candidate = Path(config_path)
        if candidate.is_file():
            resolved_path = candidate
        else:
            # Check relative to configs directory
            configs_dir_candidate = Path("configs") / candidate
            if configs_dir_candidate.is_file():
                resolved_path = configs_dir_candidate
            else:
                repo_configs_candidate = Path(__file__).resolve().parents[3] / "configs" / candidate
                if repo_configs_candidate.is_file():
                    resolved_path = repo_configs_candidate
                else:
                    resolved_path = candidate

    if not resolved_path.is_file():
        raise ConfigError(f"Configuration file not found: {resolved_path}")

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML from '{resolved_path}': {exc}") from exc
    except Exception as exc:
        raise ConfigError(f"Error reading configuration file '{resolved_path}': {exc}") from exc

    if not isinstance(raw_data, Mapping):
        raise ConfigError(f"Configuration file '{resolved_path}' must contain a YAML mapping at root.")

    validate_config(raw_data)
    return Config(raw_data)

