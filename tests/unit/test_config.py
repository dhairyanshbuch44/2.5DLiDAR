"""Unit tests for the Fovea-LiDAR configuration loader and validator."""

from pathlib import Path
import pytest
import yaml

from fovea_lidar.utils.config import Config, ConfigError, load_config


def test_valid_config_loads():
    """Test that the default configuration loads properly with all required sections."""
    config = load_config()
    assert isinstance(config, Config)
    assert "mapping" in config
    assert "resolution_zones" in config
    assert "semantic_foveation" in config
    assert "risk_refinement" in config
    assert "uncertainty_refinement" in config

    # Check representative default values
    assert config.mapping.min_resolution == 0.05
    assert config.mapping.max_resolution == 0.40
    assert config.mapping.max_range == 50.0
    assert isinstance(config.resolution_zones, list)
    assert len(config.resolution_zones) > 0
    assert config.semantic_foveation.enabled is True
    assert config.risk_refinement.enabled is True
    assert config.uncertainty_refinement.enabled is True


def test_missing_file_raises_config_error(tmp_path: Path):
    """Test that requesting a non-existent configuration file raises ConfigError."""
    missing_file = tmp_path / "non_existent_config.yaml"
    with pytest.raises(ConfigError) as exc_info:
        load_config(missing_file)
    assert "not found" in str(exc_info.value).lower()


def test_missing_required_section_raises_config_error(tmp_path: Path):
    """Test that omitting a required top-level section raises ConfigError."""
    partial_config = {
        "mapping": {"min_resolution": 0.05, "max_resolution": 0.40, "max_range": 50.0},
        "resolution_zones": [{"name": "fovea", "target_resolution": 0.05}],
        "semantic_foveation": {"enabled": True},
        # "risk_refinement" is omitted intentionally
        "uncertainty_refinement": {"enabled": True},
    }
    config_file = tmp_path / "incomplete_config.yaml"
    config_file.write_text(yaml.dump(partial_config), encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file)
    assert "missing required" in str(exc_info.value).lower()
    assert "risk_refinement" in str(exc_info.value)


def test_nested_key_access():
    """Test nested key access via attribute, dictionary subscript, and dot-notation path."""
    raw_data = {
        "mapping": {
            "min_resolution": 0.05,
            "max_resolution": 0.40,
            "max_range": 50.0,
            "nested": {
                "leaf_value": 42,
            },
        },
        "resolution_zones": [
            {"name": "fovea", "target_resolution": 0.05},
            {"name": "mid_range", "target_resolution": 0.10},
        ],
        "semantic_foveation": {"enabled": True},
        "risk_refinement": {"enabled": True},
        "uncertainty_refinement": {"enabled": False},
    }
    cfg = Config(raw_data)

    # 1. Dictionary subscript access
    assert cfg["mapping"]["min_resolution"] == 0.05
    assert cfg["mapping"]["nested"]["leaf_value"] == 42

    # 2. Dot-notation key access via __getitem__ and get()
    assert cfg["mapping.min_resolution"] == 0.05
    assert cfg["mapping.nested.leaf_value"] == 42
    assert cfg.get("mapping.nested.leaf_value") == 42
    assert cfg.get("mapping.unknown_key") is None
    assert cfg.get("mapping.unknown_key", default=999) == 999

    # 3. Attribute-style access
    assert cfg.mapping.min_resolution == 0.05
    assert cfg.mapping.nested.leaf_value == 42

    # 4. List elements wrapping
    assert cfg.resolution_zones[0].name == "fovea"
    assert cfg.resolution_zones[0]["target_resolution"] == 0.05

    # 5. Missing keys raise typed ConfigError rather than bare KeyError/AttributeError
    with pytest.raises(ConfigError) as exc_info:
        _ = cfg["non_existent_key"]
    assert "not found" in str(exc_info.value).lower()

    with pytest.raises(ConfigError) as exc_info:
        _ = cfg["mapping.non_existent_sub_key"]
    assert "not found" in str(exc_info.value).lower()

    with pytest.raises(ConfigError) as exc_info:
        _ = cfg.non_existent_attribute
    assert "not found" in str(exc_info.value).lower()
