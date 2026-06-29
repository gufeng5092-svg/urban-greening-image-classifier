from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from urban_greening_classifier.inference import EnsembleClassifier
from urban_greening_classifier.io import write_json
from urban_greening_classifier.models import create_model


def test_ensemble_classifier_predicts_with_local_checkpoint(tmp_path: Path) -> None:
    class_names_path = tmp_path / "class_names.json"
    checkpoint_dir = tmp_path / "checkpoints"
    model_dir = checkpoint_dir / "small_cnn"
    model_dir.mkdir(parents=True)
    write_json(class_names_path, ["healthy", "issue"])

    model = create_model("small_cnn", num_classes=2, pretrained=False)
    torch.save(model.state_dict(), model_dir / "fold_1.pth")

    classifier = EnsembleClassifier(
        checkpoint_dir=checkpoint_dir,
        class_names_path=class_names_path,
        model_weights={"small_cnn": 1.0},
        image_size=32,
        device="cpu",
    )
    result = classifier.predict(Image.new("RGB", (32, 32), color=(80, 120, 60)))

    assert result["class"] in {"healthy", "issue"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == 2
