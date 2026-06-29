from __future__ import annotations

import csv
from pathlib import Path

from urban_greening_classifier.io import read_json, write_csv, write_json
from urban_greening_classifier.paths import resolve_project_path
from urban_greening_classifier.reproducibility import set_seed


def test_json_and_csv_helpers(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "data.json"
    csv_path = tmp_path / "rows.csv"

    write_json(json_path, {"name": "urban"})
    write_csv(csv_path, [{"a": 1, "b": "x"}])

    assert read_json(json_path) == {"name": "urban"}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows == [{"a": "1", "b": "x"}]


def test_seed_and_path_helpers() -> None:
    set_seed(123)
    assert resolve_project_path("configs/train.yaml").name == "train.yaml"
