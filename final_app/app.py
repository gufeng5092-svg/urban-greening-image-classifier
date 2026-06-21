from __future__ import annotations

import base64
import io
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from inference import classifier


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_to_data_url(image: Image.Image) -> str:
    preview = image.convert("RGB").copy()
    preview.thumbnail((900, 640))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/")
def index():
    return render_template("index.html")


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "The image is too large. Please use an image under 64 MB."}), 413


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file was received."}), 400
    file = request.files["image"]
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Supported image formats: jpg, jpeg, png, webp, bmp."}), 400
    try:
        image = Image.open(file.stream)
        result = classifier.predict(image)
        result["preview"] = image_to_data_url(image)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Image classification failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
