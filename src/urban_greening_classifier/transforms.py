"""Torchvision transform builders."""

from __future__ import annotations

from torchvision import transforms

from urban_greening_classifier.constants import DEFAULT_MEAN, DEFAULT_STD


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Build the deterministic preprocessing transform used for validation and inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )


def build_train_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Build training and validation transforms without changing the original augmentation policy."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=12)], p=0.5),
            transforms.RandomResizedCrop(image_size, scale=(0.86, 1.0), ratio=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
        ]
    )
    return train_transform, build_eval_transform(image_size)
