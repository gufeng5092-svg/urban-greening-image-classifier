from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from urban_greening_classifier.device import choose_device
from urban_greening_classifier.io import read_json
from urban_greening_classifier.io import write_csv as shared_write_csv
from urban_greening_classifier.logging_utils import configure_logging, get_logger
from urban_greening_classifier.models import SUPPORTED_MODELS, create_model
from urban_greening_classifier.transforms import build_eval_transform

LOGGER = get_logger(__name__)


def read_class_names(path: Path) -> list[str]:
    """Read class names in training output order."""
    return list(read_json(path))


def read_val_splits(path: Path) -> dict[int, list[dict]]:
    """Read validation split rows grouped by fold."""
    folds: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "val":
                continue
            folds.setdefault(int(row["fold"]), []).append(row)
    return folds


@torch.no_grad()
def predict_fold(
    model: torch.nn.Module,
    rows: list[dict],
    class_to_id: dict[str, int],
    transform: torch.nn.Module,
    device: torch.device,
) -> list[dict]:
    """Run a fold checkpoint over its validation split and return probabilities."""
    model.eval()
    predictions: list[dict] = []
    for row in rows:
        path = Path(row["path"])
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
        true_id = class_to_id[row["label"]]
        pred_id = int(probs.argmax())
        item = {
            "fold": row["fold"],
            "path": row["path"],
            "true_label": row["label"],
            "true_id": true_id,
            "pred_id": pred_id,
        }
        for idx, prob in enumerate(probs):
            item[f"prob_{idx}"] = float(prob)
        predictions.append(item)
    return predictions


def write_csv(path: Path, rows: list[dict]) -> None:
    """Backward-compatible wrapper around the shared CSV writer."""
    shared_write_csv(path, rows)


def write_metrics(path: Path, rows: list[dict]) -> None:
    """Write aggregate metrics for exported predictions."""
    y_true = [int(r["true_id"]) for r in rows]
    y_pred = [int(r["pred_id"]) for r in rows]
    metrics = [
        {"metric": "accuracy", "value": accuracy_score(y_true, y_pred)},
        {"metric": "macro_precision", "value": precision_score(y_true, y_pred, average="macro", zero_division=0)},
        {"metric": "macro_recall", "value": recall_score(y_true, y_pred, average="macro", zero_division=0)},
        {"metric": "macro_f1", "value": f1_score(y_true, y_pred, average="macro", zero_division=0)},
        {"metric": "weighted_f1", "value": f1_score(y_true, y_pred, average="weighted", zero_division=0)},
    ]
    write_csv(path, metrics)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Export fold validation probabilities from saved CV models.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    class_names = read_class_names(args.result_dir / "class_names.json")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    folds = read_val_splits(args.result_dir / "fold_splits.csv")
    transform = build_eval_transform(args.image_size)

    all_predictions: list[dict] = []
    for fold, rows in sorted(folds.items()):
        model = create_model(args.model, len(class_names), pretrained=False).to(device)
        state_path = args.result_dir / f"fold_{fold}" / "best_model.pth"
        model.load_state_dict(torch.load(state_path, map_location=device))
        fold_predictions = predict_fold(model, rows, class_to_id, transform, device)
        write_csv(args.result_dir / f"fold_{fold}" / "predictions.csv", fold_predictions)
        all_predictions.extend(fold_predictions)
        LOGGER.info("fold %s: %s predictions", fold, len(fold_predictions))

    write_csv(args.result_dir / "overall_predictions.csv", all_predictions)
    write_metrics(args.result_dir / "prediction_metrics.csv", all_predictions)
    LOGGER.info("saved: %s", args.result_dir / "overall_predictions.csv")


if __name__ == "__main__":
    main()
