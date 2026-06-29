"""Reusable inference implementation for CLI, Flask and FastAPI surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from urban_greening_classifier.device import choose_device
from urban_greening_classifier.io import read_json
from urban_greening_classifier.models import create_model
from urban_greening_classifier.transforms import build_eval_transform


class EnsembleClassifier:
    """Lazy-loaded weighted ensemble classifier."""

    def __init__(
        self,
        checkpoint_dir: Path,
        class_names_path: Path,
        model_weights: dict[str, float],
        image_size: int = 224,
        device: str = "auto",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.class_names: list[str] = list(read_json(class_names_path))
        self.model_weights = model_weights
        self.device = choose_device(device)
        self.transform = build_eval_transform(image_size)
        self.models: dict[str, list[torch.nn.Module]] = {}
        self.loaded = False

    def load(self) -> None:
        """Load checkpoints once and keep models in evaluation mode."""
        if self.loaded:
            return
        for model_name in self.model_weights:
            model_dir = self.checkpoint_dir / model_name
            checkpoint_paths = sorted(model_dir.glob("fold_*.pth"))
            if not checkpoint_paths:
                raise FileNotFoundError(f"No checkpoints found for {model_name}: {model_dir}")
            self.models[model_name] = []
            for checkpoint_path in checkpoint_paths:
                model = create_model(model_name, len(self.class_names), pretrained=False)
                state = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(state)
                model.to(self.device)
                model.eval()
                self.models[model_name].append(model)
        self.loaded = True

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Predict a class for one PIL image."""
        self.load()
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        final_probs = torch.zeros(len(self.class_names), device=self.device)
        model_details: dict[str, list[float]] = {}

        for model_name, weight in self.model_weights.items():
            fold_probs = []
            for model in self.models[model_name]:
                logits = model(tensor)
                fold_probs.append(F.softmax(logits, dim=1).squeeze(0))
            model_probs = torch.stack(fold_probs, dim=0).mean(dim=0)
            final_probs += weight * model_probs
            model_details[model_name] = [float(value) for value in model_probs.detach().cpu()]

        probs = final_probs.detach().cpu()
        pred_id = int(torch.argmax(probs).item())
        return {
            "label": self.class_names[pred_id],
            "class": self.class_names[pred_id],
            "class_id": pred_id,
            "confidence": float(probs[pred_id].item()),
            "probabilities": [
                {"label": label, "probability": float(probs[idx].item())} for idx, label in enumerate(self.class_names)
            ],
            "model_weights": self.model_weights,
            "model_details": model_details,
            "device": str(self.device),
        }
