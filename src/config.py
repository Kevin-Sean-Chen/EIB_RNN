"""Load and save YAML configuration files."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


ConfigType = TypeVar("ConfigType")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML file and require a mapping at its root."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            content = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration file does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Configuration file has invalid YAML: {path}") from error

    if not isinstance(content, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return content


def load_dataclass_sections(
    path: Path,
    config_type: type[ConfigType],
    section_names: tuple[str, ...],
) -> tuple[ConfigType, dict[str, Any]]:
    """Load selected YAML sections into one dataclass."""
    document = load_yaml(path)
    allowed = {field.name for field in fields(config_type)}
    values: dict[str, Any] = {}

    for section_name in section_names:
        section = document.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Configuration section must be a mapping: {section_name}")
        for name, value in section.items():
            if name not in allowed:
                raise ValueError(
                    f"Unknown configuration field in {section_name}: {name}"
                )
            if name in values:
                raise ValueError(f"Configuration field occurs more than once: {name}")
            values[name] = value

    return config_type(**values), document


def save_yaml(path: Path, content: dict[str, Any]) -> None:
    """Save a mapping as YAML."""
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_plain_value(content), stream, sort_keys=False)


def _plain_value(value: Any) -> Any:
    """Convert configuration values to standard Python values."""
    if isinstance(value, Mapping):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except ValueError:
            pass
    return value


def dataclass_to_sections(
    config: Any,
    section_fields: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, Any]]:
    """Convert selected dataclass fields to grouped mappings."""
    return {
        section: {name: getattr(config, name) for name in names}
        for section, names in section_fields.items()
    }
