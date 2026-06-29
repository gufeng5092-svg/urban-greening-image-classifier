from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from urban_greening_classifier.config import load_yaml_config
from urban_greening_classifier.inference import EnsembleClassifier
from urban_greening_classifier.io import read_json
from urban_greening_classifier.models import create_model  # noqa: F401
from urban_greening_classifier.paths import resolve_project_path
from urban_greening_classifier.transforms import build_eval_transform as build_transform  # noqa: F401

_CONFIG = load_yaml_config("configs/inference.yaml")["inference"]
CHECKPOINT_DIR = resolve_project_path(_CONFIG["checkpoint_dir"])
CLASS_NAMES = list(read_json(resolve_project_path(_CONFIG["class_names_path"])))
MODEL_WEIGHTS = dict(_CONFIG["model_weights"])


def build_classifier() -> EnsembleClassifier:
    """Build the trained ensemble classifier from repository configuration."""
    return EnsembleClassifier(
        checkpoint_dir=CHECKPOINT_DIR,
        class_names_path=resolve_project_path(_CONFIG["class_names_path"]),
        model_weights=MODEL_WEIGHTS,
        image_size=int(_CONFIG["image_size"]),
        device=str(_CONFIG["device"]),
    )


classifier = build_classifier()
