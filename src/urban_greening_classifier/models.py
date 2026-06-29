"""Model factory shared by training and inference."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS: tuple[str, ...] = ("small_cnn", "resnet18", "mobilenet_v3_small", "efficientnet_b0")


class SmallCNN(nn.Module):
    """Small convolutional baseline used for quick experiments."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(192, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def create_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Create a classifier backbone with an output layer sized for the dataset."""
    if model_name == "small_cnn":
        return SmallCNN(num_classes)

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")
