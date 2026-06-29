"""Dataset helpers for image classification."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from urban_greening_classifier.constants import IMAGE_EXTENSIONS


class ImagePathDataset(Dataset):
    """A dataset backed by image paths and integer labels."""

    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose | None = None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load and transform one image."""
        path, label = self.samples[index]
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
        return rgb_image, label


def load_samples(data_dir: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    """Load class-folder image paths and return samples plus class names."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    class_dirs = sorted([path for path in data_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
    class_names = [path.name for path in class_dirs]
    samples: list[tuple[Path, int]] = []
    for label_id, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label_id))
    if not samples:
        raise RuntimeError(f"No images found under {data_dir}")
    return samples, class_names
