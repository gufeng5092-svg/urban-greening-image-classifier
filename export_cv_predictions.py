from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision import transforms

from train_cross_validation import choose_device, create_model


def read_class_names(path: Path) -> list[str]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def read_val_splits(path: Path) -> dict[int, list[dict]]:
    folds: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "val":
                continue
            folds.setdefault(int(row["fold"]), []).append(row)
    return folds


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@torch.no_grad()
def predict_fold(
    model: torch.nn.Module,
    rows: list[dict],
    class_to_id: dict[str, int],
    transform: transforms.Compose,
    device: torch.device,
) -> list[dict]:
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
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path: Path, rows: list[dict]) -> None:
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
    parser = argparse.ArgumentParser(description="Export fold validation probabilities from saved CV models.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["small_cnn", "resnet18", "mobilenet_v3_small", "efficientnet_b0"], required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    class_names = read_class_names(args.result_dir / "class_names.json")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    folds = read_val_splits(args.result_dir / "fold_splits.csv")
    transform = build_transform(args.image_size)

    all_predictions: list[dict] = []
    for fold, rows in sorted(folds.items()):
        model = create_model(args.model, len(class_names), pretrained=False).to(device)
        state_path = args.result_dir / f"fold_{fold}" / "best_model.pth"
        model.load_state_dict(torch.load(state_path, map_location=device))
        fold_predictions = predict_fold(model, rows, class_to_id, transform, device)
        write_csv(args.result_dir / f"fold_{fold}" / "predictions.csv", fold_predictions)
        all_predictions.extend(fold_predictions)
        print(f"fold {fold}: {len(fold_predictions)} predictions")

    write_csv(args.result_dir / "overall_predictions.csv", all_predictions)
    write_metrics(args.result_dir / "prediction_metrics.csv", all_predictions)
    print(f"saved: {args.result_dir / 'overall_predictions.csv'}")


if __name__ == "__main__":
    main()
