"""YAML configuration loading and merging."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from urban_greening_classifier.paths import resolve_project_path


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for an empty document."""
    config_path = resolve_project_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings without mutating either input."""
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested YAML sections into argparse-compatible keys."""
    flattened: dict[str, Any] = {}
    for value in config.values():
        if isinstance(value, Mapping):
            flattened.update(value)
    return flattened
