"""Path helpers for repository-local files."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root, independent of the current working directory."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a relative path against the repository root."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else project_root() / value
