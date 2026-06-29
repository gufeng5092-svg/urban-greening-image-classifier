from __future__ import annotations

from pathlib import Path

from urban_greening_classifier.config import deep_merge, flatten_config, load_yaml_config


def test_load_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  epochs: 2\n  lr: 0.001\n", encoding="utf-8")

    config = load_yaml_config(config_path)

    assert config["training"]["epochs"] == 2
    assert config["training"]["lr"] == 0.001


def test_deep_merge_and_flatten_config() -> None:
    merged = deep_merge({"a": {"x": 1}, "b": 2}, {"a": {"y": 3}})

    assert merged == {"a": {"x": 1, "y": 3}, "b": 2}
    assert flatten_config({"data": {"path": "data"}, "training": {"epochs": 3}}) == {
        "path": "data",
        "epochs": 3,
    }
