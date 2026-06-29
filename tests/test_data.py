from __future__ import annotations

from pathlib import Path

from PIL import Image

from urban_greening_classifier.data import ImagePathDataset, load_samples
from urban_greening_classifier.transforms import build_eval_transform


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(30, 120, 60)).save(path)


def test_load_samples_from_class_directories(tmp_path: Path) -> None:
    _write_image(tmp_path / "class_a" / "a.png")
    _write_image(tmp_path / "class_b" / "b.jpg")
    (tmp_path / "class_b" / "ignore.txt").write_text("x", encoding="utf-8")

    samples, class_names = load_samples(tmp_path)

    assert class_names == ["class_a", "class_b"]
    assert len(samples) == 2
    assert {label for _, label in samples} == {0, 1}


def test_image_path_dataset_returns_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "class_a" / "a.png"
    _write_image(image_path)
    dataset = ImagePathDataset([(image_path, 4)], transform=build_eval_transform(32))

    tensor, label = dataset[0]

    assert tuple(tensor.shape) == (3, 32, 32)
    assert label == 4
