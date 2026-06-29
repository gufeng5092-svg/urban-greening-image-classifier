from __future__ import annotations

import base64
import io
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flask import Flask, jsonify, render_template, request
from inference import classifier
from PIL import Image

from urban_greening_classifier.config import load_yaml_config
from urban_greening_classifier.logging_utils import configure_logging

app = Flask(__name__)
APP_CONFIG = load_yaml_config("configs/inference.yaml")["app"]
app.config["MAX_CONTENT_LENGTH"] = int(APP_CONFIG["max_upload_mb"]) * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOGGER = logging.getLogger(__name__)


def image_to_data_url(image: Image.Image) -> str:
    """Create a browser-friendly preview image data URL."""
    preview = image.convert("RGB").copy()
    preview.thumbnail((900, 640))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/")
def index() -> str:
    """Render the HTML inference demo."""
    return render_template("index.html")


@app.errorhandler(413)
def file_too_large(_error: Exception):
    """Return a JSON error when upload size exceeds the configured limit."""
    return (
        jsonify({"error": f"The image is too large. Please use an image under {APP_CONFIG['max_upload_mb']} MB."}),
        413,
    )


@app.post("/predict")
def predict():
    """Classify an uploaded image and return JSON predictions."""
    if "image" not in request.files:
        return jsonify({"error": "No image file was received."}), 400
    file = request.files["image"]
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Supported image formats: jpg, jpeg, png, webp, bmp."}), 400
    try:
        with Image.open(file.stream) as image:
            result = classifier.predict(image)
            result["preview"] = image_to_data_url(image)
        return jsonify(result)
    except Exception as exc:
        LOGGER.exception("Image classification failed")
        return jsonify({"error": f"Image classification failed: {exc}"}), 500


if __name__ == "__main__":
    configure_logging()
    app.run(host=APP_CONFIG["host"], port=int(APP_CONFIG["port"]), debug=False)
