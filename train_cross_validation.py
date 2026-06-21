from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def configure_matplotlib_font() -> None:
    font_candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            from matplotlib import font_manager

            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return


configure_matplotlib_font()


@dataclass
class FoldMetric:
    model: str
    fold: int
    epochs: int
    best_epoch: int
    best_val_macro_f1: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    train_samples: int
    val_samples: int
    train_seconds: float
    inference_ms_per_image: float
    parameters_m: float
    device: str


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose | None = None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(192, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_samples(data_dir: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    class_names = [p.name for p in class_dirs]
    samples: list[tuple[Path, int]] = []
    for label_id, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.iterdir(), key=lambda p: p.name):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label_id))
    if not samples:
        raise RuntimeError(f"No images found under {data_dir}")
    return samples, class_names


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=12)], p=0.5),
            transforms.RandomResizedCrop(image_size, scale=(0.86, 1.0), ratio=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, val_transform


def create_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name == "small_cnn":
        return SmallCNN(num_classes)

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def make_sampler(train_labels: list[int]) -> WeightedRandomSampler:
    counts = Counter(train_labels)
    weights = [1.0 / counts[label] for label in train_labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def class_weight_tensor(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    start = time.perf_counter()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    elapsed = time.perf_counter() - start
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "labels": all_labels,
        "preds": all_preds,
        "inference_ms_per_image": elapsed * 1000.0 / max(1, len(loader.dataset)),
    }


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_curve(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="val macro-F1")
    axes[1].set_title("Accuracy / Macro-F1")
    axes[1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(metrics: list[FoldMetric]) -> list[dict]:
    fields = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "train_seconds", "inference_ms_per_image"]
    rows = []
    for field in fields:
        values = np.array([getattr(m, field) for m in metrics], dtype=np.float64)
        rows.append({"metric": field, "mean": round(float(values.mean()), 6), "std": round(float(values.std(ddof=0)), 6)})
    return rows


def run_cross_validation(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    samples, class_names = load_samples(args.data)
    labels = np.array([label for _, label in samples])
    class_counts = Counter(labels.tolist())

    tags = [
        args.model,
        f"k{args.folds}",
        f"seed{args.seed}",
        "aug" if args.augment else "noaug",
        "cw" if args.class_weights else "nocw",
        "sampler" if args.weighted_sampler else "nosampler",
        "pretrained" if args.pretrained else "scratch",
    ]
    if args.run_name:
        tags.append(args.run_name)
    output_dir = args.output / "_".join(tags)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_names.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    train_transform, val_transform = build_transforms(args.image_size)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_metrics: list[FoldMetric] = []
    all_labels: list[int] = []
    all_preds: list[int] = []
    split_rows: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        train_labels = [label for _, label in train_samples]

        for idx in train_idx:
            split_rows.append({"fold": fold, "split": "train", "path": str(samples[idx][0]), "label": class_names[samples[idx][1]]})
        for idx in val_idx:
            split_rows.append({"fold": fold, "split": "val", "path": str(samples[idx][0]), "label": class_names[samples[idx][1]]})

        train_dataset = ImagePathDataset(train_samples, train_transform if args.augment else val_transform)
        val_dataset = ImagePathDataset(val_samples, val_transform)
        sampler = make_sampler(train_labels) if args.weighted_sampler else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = create_model(args.model, len(class_names), args.pretrained).to(device)
        parameters_m = sum(p.numel() for p in model.parameters()) / 1_000_000
        loss_weights = class_weight_tensor(train_labels, len(class_names), device) if args.class_weights else None
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

        best_state = None
        best_epoch = 0
        best_macro_f1 = -1.0
        history: list[dict] = []
        start = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(float(train_acc), 6),
                "val_loss": round(float(val_metrics["loss"]), 6),
                "val_acc": round(float(val_metrics["accuracy"]), 6),
                "val_macro_f1": round(float(val_metrics["macro_f1"]), 6),
            }
            history.append(row)
            print(
                f"fold {fold}/{args.folds} epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}",
                flush=True,
            )
            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = float(val_metrics["macro_f1"])
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        train_seconds = time.perf_counter() - start
        if best_state is not None:
            model.load_state_dict(best_state)
        final_metrics = evaluate(model, val_loader, criterion, device)
        fold_cm = confusion_matrix(final_metrics["labels"], final_metrics["preds"], labels=list(range(len(class_names))))
        save_confusion_matrix(fold_cm, class_names, fold_dir / "confusion_matrix.png")
        save_curve(history, fold_dir / "training_curve.png")
        write_csv(fold_dir / "history.csv", history)
        torch.save(model.state_dict(), fold_dir / "best_model.pth")

        all_labels.extend(final_metrics["labels"])
        all_preds.extend(final_metrics["preds"])
        fold_metrics.append(
            FoldMetric(
                model=args.model,
                fold=fold,
                epochs=args.epochs,
                best_epoch=best_epoch,
                best_val_macro_f1=round(best_macro_f1, 6),
                accuracy=round(float(final_metrics["accuracy"]), 6),
                macro_precision=round(float(final_metrics["macro_precision"]), 6),
                macro_recall=round(float(final_metrics["macro_recall"]), 6),
                macro_f1=round(float(final_metrics["macro_f1"]), 6),
                weighted_f1=round(float(final_metrics["weighted_f1"]), 6),
                train_samples=len(train_samples),
                val_samples=len(val_samples),
                train_seconds=round(train_seconds, 3),
                inference_ms_per_image=round(float(final_metrics["inference_ms_per_image"]), 4),
                parameters_m=round(parameters_m, 4),
                device=str(device),
            )
        )

    write_csv(output_dir / "fold_metrics.csv", [asdict(metric) for metric in fold_metrics])
    write_csv(output_dir / "summary_metrics.csv", summarize_metrics(fold_metrics))
    write_csv(output_dir / "fold_splits.csv", split_rows)
    overall_cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    save_confusion_matrix(overall_cm, class_names, output_dir / "confusion_matrix_overall.png")

    summary = summarize_metrics(fold_metrics)
    lines = [
        "# Cross-Validation Training Report",
        "",
        f"- Data directory: `{args.data}`",
        f"- Model: `{args.model}`",
        f"- Folds: {args.folds}",
        f"- Image size: {args.image_size}x{args.image_size}",
        f"- Training augmentation: {'yes' if args.augment else 'no'}",
        f"- Class weights: {'yes' if args.class_weights else 'no'}",
        f"- Weighted sampler: {'yes' if args.weighted_sampler else 'no'}",
        f"- Pretrained weights: {'yes' if args.pretrained else 'no'}",
        f"- Device: `{device}`",
        "",
        "## Class Distribution",
        "",
    ]
    for label_id, count in sorted(class_counts.items()):
        lines.append(f"- {class_names[label_id]}: {count}")
    lines.extend(["", "## Mean Metrics", "", "| Metric | Mean | Std |", "|---|---:|---:|"])
    for row in summary:
        lines.append(f"| {row['metric']} | {row['mean']} | {row['std']} |")
    (output_dir / "cross_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold cross validation for urban greening image classification.")
    parser.add_argument("--data", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--output", type=Path, default=Path("cv_results"))
    parser.add_argument("--model", choices=["small_cnn", "resnet18", "mobilenet_v3_small", "efficientnet_b0"], default="efficientnet_b0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run_cross_validation(parse_args())
