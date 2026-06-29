from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from urban_greening_classifier.io import write_csv as shared_write_csv
from urban_greening_classifier.logging_utils import configure_logging, get_logger

LOGGER = get_logger(__name__)


def read_predictions(path: Path) -> dict[tuple[int, str], dict]:
    """Read exported fold predictions keyed by fold and image path."""
    rows: dict[tuple[int, str], dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[(int(row["fold"]), row["path"])] = row
    return rows


def metric_rows(y_true: list[int], y_pred: list[int]) -> list[dict]:
    """Compute standard classification metrics."""
    return [
        {"metric": "accuracy", "value": accuracy_score(y_true, y_pred)},
        {"metric": "macro_precision", "value": precision_score(y_true, y_pred, average="macro", zero_division=0)},
        {"metric": "macro_recall", "value": recall_score(y_true, y_pred, average="macro", zero_division=0)},
        {"metric": "macro_f1", "value": f1_score(y_true, y_pred, average="macro", zero_division=0)},
        {"metric": "weighted_f1", "value": f1_score(y_true, y_pred, average="weighted", zero_division=0)},
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Backward-compatible wrapper around the shared CSV writer."""
    shared_write_csv(path, rows)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Average CV prediction probabilities from multiple models.")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pred_maps = [read_predictions(path) for path in args.predictions]
    keys = set(pred_maps[0])
    for pred_map in pred_maps[1:]:
        keys &= set(pred_map)
    if not keys:
        raise RuntimeError("No aligned prediction rows found.")

    weights = args.weights or [1.0] * len(pred_maps)
    if len(weights) != len(pred_maps):
        raise ValueError("--weights length must match --predictions length")
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()

    rows: list[dict] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for fold, path in sorted(keys):
        first = pred_maps[0][(fold, path)]
        prob_cols = sorted([key for key in first if key.startswith("prob_")], key=lambda x: int(x.split("_")[1]))
        probs = np.zeros(len(prob_cols), dtype=np.float64)
        for weight, pred_map in zip(weights_arr, pred_maps):
            row = pred_map[(fold, path)]
            probs += weight * np.array([float(row[col]) for col in prob_cols], dtype=np.float64)
        pred_id = int(probs.argmax())
        true_id = int(first["true_id"])
        item = {
            "fold": fold,
            "path": path,
            "true_label": first["true_label"],
            "true_id": true_id,
            "pred_id": pred_id,
        }
        for idx, prob in enumerate(probs):
            item[f"prob_{idx}"] = float(prob)
        rows.append(item)
        y_true.append(true_id)
        y_pred.append(pred_id)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "ensemble_predictions.csv", rows)
    metrics = metric_rows(y_true, y_pred)
    write_csv(args.output / "ensemble_metrics.csv", metrics)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(prob_cols))))
    cm_rows = [
        {"true_id": idx, **{f"pred_{j}": int(value) for j, value in enumerate(row)}} for idx, row in enumerate(cm)
    ]
    write_csv(args.output / "ensemble_confusion_matrix.csv", cm_rows)

    lines = [
        "# Ensemble Results",
        "",
        f"- Number of fused models: {len(pred_maps)}",
        f"- Weights: {', '.join(f'{w:.3f}' for w in weights_arr)}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for row in metrics:
        lines.append(f"| {row['metric']} | {row['value']:.6f} |")
    (args.output / "ensemble_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("saved: %s", args.output / "ensemble_report.md")


if __name__ == "__main__":
    main()
