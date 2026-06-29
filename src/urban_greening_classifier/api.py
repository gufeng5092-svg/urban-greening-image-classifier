"""FastAPI inference service."""

from __future__ import annotations

import logging
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from urban_greening_classifier.config import load_yaml_config
from urban_greening_classifier.inference import EnsembleClassifier
from urban_greening_classifier.paths import resolve_project_path

LOGGER = logging.getLogger(__name__)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


class PredictionResponse(BaseModel):
    """Minimal prediction response required by the HTTP API contract."""

    predicted_class: str = Field(alias="class")
    confidence: float


def create_classifier() -> EnsembleClassifier:
    """Create the configured ensemble classifier."""
    config = load_yaml_config("configs/inference.yaml")["inference"]
    return EnsembleClassifier(
        checkpoint_dir=resolve_project_path(config["checkpoint_dir"]),
        class_names_path=resolve_project_path(config["class_names_path"]),
        model_weights=dict(config["model_weights"]),
        image_size=int(config["image_size"]),
        device=str(config["device"]),
    )


app = FastAPI(title="Urban Greening Image Classifier", version="1.0.0")
classifier = create_classifier()


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> dict[str, float | str]:
    """Classify an uploaded image."""
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    try:
        content = await image.read()
        with Image.open(BytesIO(content)) as pil_image:
            result = classifier.predict(pil_image)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    return {"class": str(result["class"]), "confidence": float(result["confidence"])}
