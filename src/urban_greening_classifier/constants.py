"""Project-wide constants."""

from __future__ import annotations

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
