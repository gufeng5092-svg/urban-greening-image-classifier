from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SOURCE = Path("data/cleaned")
DEFAULT_OUTPUT = Path("data/augmented")
DEFAULT_REPORT = Path("augmentation_report")


@dataclass
class AugmentRecord:
    output_file: str
    source_file: str
    label: str
    operation: str
    width: int
    height: int


def list_images(source: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for class_dir in sorted([p for p in source.iterdir() if p.is_dir()]):
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        grouped[class_dir.name] = sorted(files, key=lambda p: p.name)
    return grouped


def random_resized_crop(img: Image.Image, rng: random.Random) -> Image.Image:
    width, height = img.size
    scale = rng.uniform(0.86, 1.0)
    crop_w = max(1, int(width * scale))
    crop_h = max(1, int(height * scale))
    left = rng.randint(0, width - crop_w) if width > crop_w else 0
    top = rng.randint(0, height - crop_h) if height > crop_h else 0
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def translate(img: Image.Image, rng: random.Random) -> Image.Image:
    width, height = img.size
    dx = int(width * rng.uniform(-0.08, 0.08))
    dy = int(height * rng.uniform(-0.08, 0.08))
    return ImageOps.expand(img, border=0).transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(int(np.asarray(img)[:, :, 0].mean()), int(np.asarray(img)[:, :, 1].mean()), int(np.asarray(img)[:, :, 2].mean())),
    )


def color_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    out = img
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.82, 1.18))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.82, 1.20))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.80, 1.20))
    return out


def add_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    sigma = rng.uniform(3.0, 9.0)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def cutout(img: Image.Image, rng: random.Random) -> Image.Image:
    out = img.copy()
    width, height = out.size
    box_w = int(width * rng.uniform(0.08, 0.16))
    box_h = int(height * rng.uniform(0.08, 0.16))
    left = rng.randint(0, max(0, width - box_w))
    top = rng.randint(0, max(0, height - box_h))
    arr = np.asarray(img)
    fill = tuple(int(arr[:, :, c].mean()) for c in range(3))
    patch = Image.new("RGB", (box_w, box_h), fill)
    out.paste(patch, (left, top))
    return out


def augment_image(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    out = img.convert("RGB")
    operations: list[str] = []

    if rng.random() < 0.55:
        out = ImageOps.mirror(out)
        operations.append("horizontal_flip")

    if rng.random() < 0.75:
        angle = rng.uniform(-12, 12)
        out = out.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=tuple(int(x) for x in np.asarray(out).reshape(-1, 3).mean(axis=0)))
        operations.append(f"rotate_{angle:.1f}")

    if rng.random() < 0.60:
        out = random_resized_crop(out, rng)
        operations.append("random_resized_crop")

    if rng.random() < 0.45:
        out = translate(out, rng)
        operations.append("translate")

    if rng.random() < 0.85:
        out = color_jitter(out, rng)
        operations.append("brightness_contrast_saturation")

    blur_or_sharpen = rng.random()
    if blur_or_sharpen < 0.18:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.8)))
        operations.append("slight_gaussian_blur")
    elif blur_or_sharpen < 0.36:
        out = out.filter(ImageFilter.SHARPEN)
        operations.append("sharpen")

    if rng.random() < 0.25:
        out = add_noise(out, rng)
        operations.append("gaussian_noise")

    if rng.random() < 0.18:
        out = cutout(out, rng)
        operations.append("cutout")

    if not operations:
        out = color_jitter(out, rng)
        operations.append("brightness_contrast_saturation")

    return out, "+".join(operations)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for font_path in candidates:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def compact_caption(name: str, max_chars: int = 20) -> str:
    stem = Path(name).stem
    if "_aug_" in stem:
        base, aug_id = stem.rsplit("_aug_", 1)
        number = base.rsplit("_", 1)[-1]
        return f"{number}_aug_{aug_id}"
    return stem[-max_chars:] if len(stem) > max_chars else stem


def create_contact_sheet(image_paths: list[Path], output_path: Path, title: str) -> None:
    if not image_paths:
        return
    columns = 5
    thumb_size = (180, 140)
    title_height = 42
    caption_height = 30
    gap = 14
    rows = (len(image_paths) + columns - 1) // columns
    width = columns * thumb_size[0] + (columns + 1) * gap
    height = title_height + rows * (thumb_size[1] + caption_height + gap) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(18)
    caption_font = load_font(12)
    draw.text((gap, 10), title, fill=(20, 20, 20), font=title_font)
    for idx, path in enumerate(image_paths):
        x = gap + (idx % columns) * (thumb_size[0] + gap)
        y = title_height + (idx // columns) * (thumb_size[1] + caption_height + gap)
        with Image.open(path) as img:
            thumb = ImageOps.contain(img.convert("RGB"), thumb_size)
        sheet.paste(thumb, (x + (thumb_size[0] - thumb.width) // 2, y + (thumb_size[1] - thumb.height) // 2))
        draw.text((x + 4, y + thumb_size[1] + 6), compact_caption(path.name), fill=(20, 20, 20), font=caption_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_report(report_dir: Path, before: Counter, after: Counter, records: list[AugmentRecord]) -> None:
    lines = [
        "# Data Augmentation Report",
        "",
        "## 1. Purpose",
        "",
        "The cleaned urban greening maintenance image dataset is class-imbalanced. Bare-soil classes have fewer samples than the exposed-trash class. Offline augmentation is applied to minority classes to reduce majority-class bias and improve robustness to lighting, viewpoint, scale variation, and partial occlusion in street-scene images.",
        "",
        "## 2. Strategy",
        "",
        "- Keep all cleaned original images.",
        "- Use the largest class size as the target count and generate augmented images only for minority classes.",
        "- Avoid vertical flips and large rotations because they can break the natural orientation of street scenes.",
        "- Use lightweight augmentation with controlled magnitude to keep generated samples realistic.",
        "",
        "## 3. Augmentation Methods",
        "",
        "- Horizontal flip: simulates left-right road layout variation.",
        "- Small-angle rotation: simulates camera mounting and handheld shooting angle variation.",
        "- Random resized crop: improves robustness to scale and local-region changes.",
        "- Translation: simulates object position changes in the frame.",
        "- Brightness, contrast, and saturation jitter: simulates weather, shadow, and lighting changes.",
        "- Slight Gaussian blur and sharpening: simulates camera image-quality variation.",
        "- Gaussian noise: simulates surveillance-device or compression noise.",
        "- Cutout: improves robustness to partial occlusion and complex backgrounds.",
        "",
        "## 4. Class Count Changes",
        "",
        "| Class | Before augmentation | After augmentation | New augmented images |",
        "|---|---:|---:|---:|",
    ]
    for label in sorted(after):
        lines.append(f"| {label} | {before[label]} | {after[label]} | {after[label] - before[label]} |")
    lines.extend(
        [
            "",
            "## 5. Outputs",
            "",
            f"- Generated augmented images: {len(records)}.",
            "- `data/augmented/`: original and augmented images organized by class.",
            "- `augmentation_records.csv`: source image and operation records for every augmented image.",
            "- `preview_sheets/`: augmentation preview sheets.",
        ]
    )
    (report_dir / "augmentation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment cleaned urban greening classification dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--target", type=int, default=None, help="Target count per class. Defaults to max class count.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    grouped = list_images(args.source)
    if not grouped:
        raise RuntimeError(f"No class images found under {args.source}")

    before = Counter({label: len(files) for label, files in grouped.items()})
    target = args.target or max(before.values())

    if args.output.exists():
        shutil.rmtree(args.output)
    if args.report.exists():
        shutil.rmtree(args.report)
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)

    records: list[AugmentRecord] = []
    preview_paths: dict[str, list[Path]] = defaultdict(list)

    for label, files in grouped.items():
        class_out = args.output / label
        class_out.mkdir(parents=True, exist_ok=True)

        for src in files:
            shutil.copy2(src, class_out / src.name)

        needed = max(0, target - len(files))
        for idx in range(needed):
            src = files[idx % len(files)]
            with Image.open(src) as img:
                aug, operation = augment_image(img.convert("RGB"), rng)
            out_name = f"{src.stem}_aug_{idx + 1:04d}.png"
            out_path = class_out / out_name
            aug.save(out_path, format="PNG", optimize=True)
            records.append(
                AugmentRecord(
                    output_file=str(out_path.relative_to(args.output)),
                    source_file=str(src.relative_to(args.source)),
                    label=label,
                    operation=operation,
                    width=aug.width,
                    height=aug.height,
                )
            )
            if len(preview_paths[label]) < 15:
                preview_paths[label].append(out_path)

    after = Counter()
    for label_dir in args.output.iterdir():
        if label_dir.is_dir():
            after[label_dir.name] = len([p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])

    write_csv(args.report / "augmentation_records.csv", [asdict(r) for r in records])
    write_csv(
        args.report / "class_distribution_after_augmentation.csv",
        [
            {
                "label": label,
                "before_augmentation": before[label],
                "after_augmentation": after[label],
                "generated": after[label] - before[label],
            }
            for label in sorted(after)
        ],
    )

    for label, paths in preview_paths.items():
        create_contact_sheet(paths, args.report / "preview_sheets" / f"{label}.jpg", f"Augmentation Preview: {label}")

    write_report(args.report, before, after, records)

    print(f"source: {args.source}")
    print(f"target_per_class: {target}")
    print(f"generated: {len(records)}")
    print(f"output: {args.output}")
    for label in sorted(after):
        print(f"{label}: {before[label]} -> {after[label]}")


if __name__ == "__main__":
    main()
