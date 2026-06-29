from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from urban_greening_classifier import api


class FakeClassifier:
    """Small classifier stub that avoids loading real checkpoints in API tests."""

    def predict(self, image: Image.Image) -> dict[str, float | str]:
        return {"class": "存在明显暴露垃圾", "confidence": 0.91}


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (12, 12), color=(40, 120, 70)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint() -> None:
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_with_mocked_classifier(monkeypatch) -> None:
    monkeypatch.setattr(api, "classifier", FakeClassifier())
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        files={"image": ("sample.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {"class": "存在明显暴露垃圾", "confidence": 0.91}


def test_predict_endpoint_rejects_non_image(monkeypatch) -> None:
    monkeypatch.setattr(api, "classifier", FakeClassifier())
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        files={"image": ("sample.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported image type."
