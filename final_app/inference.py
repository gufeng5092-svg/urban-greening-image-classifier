from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = APP_DIR / "checkpoints"
CLASS_NAMES = json.loads((APP_DIR / "class_names.json").read_text(encoding="utf-8"))
MODEL_WEIGHTS = {
    "efficientnet_b0": 0.25,
    "mobilenet_v3_small": 0.35,
    "resnet18": 0.40,
}


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class EnsembleClassifier:
    def __init__(self) -> None:
        self.device = choose_device()
        self.transform = build_transform()
        self.models: dict[str, list[nn.Module]] = {}
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        for model_name in MODEL_WEIGHTS:
            model_dir = CHECKPOINT_DIR / model_name
            checkpoint_paths = sorted(model_dir.glob("fold_*.pth"))
            if not checkpoint_paths:
                raise FileNotFoundError(f"No checkpoints found for {model_name}: {model_dir}")
            self.models[model_name] = []
            for checkpoint_path in checkpoint_paths:
                model = create_model(model_name, len(CLASS_NAMES))
                state = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(state)
                model.to(self.device)
                model.eval()
                self.models[model_name].append(model)
        self.loaded = True

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        self.load()
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        final_probs = torch.zeros(len(CLASS_NAMES), device=self.device)
        model_details = {}

        for model_name, weight in MODEL_WEIGHTS.items():
            fold_probs = []
            for model in self.models[model_name]:
                logits = model(tensor)
                fold_probs.append(F.softmax(logits, dim=1).squeeze(0))
            model_probs = torch.stack(fold_probs, dim=0).mean(dim=0)
            final_probs += weight * model_probs
            model_details[model_name] = [float(x) for x in model_probs.detach().cpu()]

        probs = final_probs.detach().cpu()
        pred_id = int(torch.argmax(probs).item())
        return {
            "label": CLASS_NAMES[pred_id],
            "class_id": pred_id,
            "confidence": float(probs[pred_id].item()),
            "probabilities": [
                {"label": label, "probability": float(probs[idx].item())}
                for idx, label in enumerate(CLASS_NAMES)
            ],
            "model_weights": MODEL_WEIGHTS,
            "model_details": model_details,
            "device": str(self.device),
        }


classifier = EnsembleClassifier()
