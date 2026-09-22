"""YAML configuration loader with schema validation (Stages 0–4).

Parses and validates configuration documents across all simulation stages:
Stage 0 (timing benchmarks), Stage 1 (pilots), Stage 2 (extensions),
Stage 3 (adaptive learning), and Stage 4 (analytics).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from src.config.schema import ExecutionConfig, PhysicsConfig, RootConfig, SimulationConfig


def load_config_from_yaml_string(yaml_string: str) -> RootConfig:
    """Parse YAML string and validate against configuration schemas (Stages 0–4).

    Args:
        yaml_string: Raw YAML string content.

    Returns:
        Validated RootConfig instance.

    Raises:
        ValueError: If YAML is empty, invalid, or violates schema constraints.
    """
    if not yaml_string or not yaml_string.strip():
        raise ValueError("Empty YAML document cannot be parsed.")

    try:
        data = yaml.safe_load(yaml_string)
    except Exception as exc:
        raise ValueError(f"Failed to parse YAML string: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary at root of YAML document, got {type(data).__name__}")

    if not data or not any(v is not None for v in data.values()):
        raise ValueError("YAML document resolved to an empty configuration.")

    # Support flat simulation configs by placing them under 'experiment'
    if "experiment" not in data and "physics" not in data and "execution" not in data:
        cfg = RootConfig(experiment=SimulationConfig(**data))
    else:
        cfg = RootConfig.model_validate(data)

    if cfg.is_empty():
        raise ValueError("Configuration must contain at least one valid section: experiment, physics, or execution.")

    return cfg


def load_config_from_yaml(file_path: Union[str, Path]) -> RootConfig:
    """Load YAML file from filesystem and validate against schema.

    Args:
        file_path: Path to YAML configuration file.

    Returns:
        Validated RootConfig instance.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected file, got directory: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return load_config_from_yaml_string(content)


# Convenient aliases
load_yaml = load_config_from_yaml_string
load_config = load_config_from_yaml
