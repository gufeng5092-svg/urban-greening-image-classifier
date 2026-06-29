from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from urban_greening_classifier.config import load_yaml_config
from urban_greening_classifier.constants import IMAGE_EXTENSIONS
from urban_greening_classifier.io import write_csv as shared_write_csv
from urban_greening_classifier.logging_utils import configure_logging, get_logger

try:
    from scipy.fftpack import dct
except Exception:  # pragma: no cover - only used when scipy is unavailable
    dct = None


LOGGER = get_logger(__name__)


@dataclass
class ImageMetric:
    file_name: str
    label: str
    width: int
    height: int
    mode: str
    format: str
    md5: str
    phash: str
    blur_laplacian_var: float
    brightness_mean: float
    brightness_std: float
    vegetation_ratio: float
    soil_like_ratio: float
    readable: bool
    issue: str


def parse_label(path: Path) -> str:
    """Infer the class label from the filename prefix before the final underscore."""
    stem = path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def file_md5(path: Path) -> str:
    """Calculate an MD5 checksum for duplicate detection."""
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def laplacian_variance(gray: np.ndarray) -> float:
    """Compute a lightweight Laplacian variance blur heuristic."""
    arr = gray.astype(np.float32)
    lap = -4.0 * arr
    lap[:-1, :] += arr[1:, :]
    lap[1:, :] += arr[:-1, :]
    lap[:, :-1] += arr[:, 1:]
    lap[:, 1:] += arr[:, :-1]
    return float(lap.var())


def perceptual_hash(image: Image.Image) -> str:
    """Calculate a simple perceptual hash for near-duplicate review."""
    gray = ImageOps.grayscale(image).resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    if dct is not None:
        coeff = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
        values = coeff.flatten()
        median = np.median(values[1:])
    else:
        small = gray.resize((8, 8), Image.Resampling.LANCZOS)
        values = np.asarray(small, dtype=np.float32).flatten()
        median = np.median(values)
    bits = (values > median).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_hex(a: str, b: str) -> int:
    """Return the Hamming distance between two hexadecimal hashes."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def color_ratios(rgb_image: Image.Image) -> tuple[float, float]:
    """Estimate vegetation and soil-like color ratios for manual review."""
    rgb = np.asarray(rgb_image.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    hsv = np.asarray(rgb_image.convert("HSV").resize((256, 256), Image.Resampling.BILINEAR))

    h = hsv[:, :, 0].astype(np.float32) * 360.0 / 255.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    green = (h >= 45) & (h <= 150) & (s >= 0.18) & (v >= 0.15)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    soil_like = (r > 65) & (g > 45) & (b > 25) & (r >= g * 0.9) & (g >= b * 0.85) & (r > b * 1.15) & (v >= 0.12)
    total = float(green.size)
    return float(green.sum() / total), float(soil_like.sum() / total)


def collect_metrics(files: Iterable[Path]) -> list[ImageMetric]:
    """Collect readability, duplicate and image-quality metrics."""
    metrics: list[ImageMetric] = []
    for path in sorted(files, key=lambda p: p.name):
        label = parse_label(path)
        try:
            with Image.open(path) as img:
                img.load()
                rgb = img.convert("RGB")
                gray = np.asarray(ImageOps.grayscale(rgb).resize((256, 256), Image.Resampling.BILINEAR))
                vegetation_ratio, soil_like_ratio = color_ratios(rgb)
                metrics.append(
                    ImageMetric(
                        file_name=path.name,
                        label=label,
                        width=img.width,
                        height=img.height,
                        mode=img.mode,
                        format=img.format or path.suffix.lstrip(".").upper(),
                        md5=file_md5(path),
                        phash=perceptual_hash(rgb),
                        blur_laplacian_var=round(laplacian_variance(gray), 4),
                        brightness_mean=round(float(gray.mean()), 4),
                        brightness_std=round(float(gray.std()), 4),
                        vegetation_ratio=round(vegetation_ratio, 6),
                        soil_like_ratio=round(soil_like_ratio, 6),
                        readable=True,
                        issue="",
                    )
                )
        except Exception as exc:
            metrics.append(
                ImageMetric(
                    file_name=path.name,
                    label=label,
                    width=0,
                    height=0,
                    mode="",
                    format=path.suffix.lstrip(".").upper(),
                    md5=file_md5(path),
                    phash="",
                    blur_laplacian_var=0.0,
                    brightness_mean=0.0,
                    brightness_std=0.0,
                    vegetation_ratio=0.0,
                    soil_like_ratio=0.0,
                    readable=False,
                    issue=str(exc),
                )
            )
    return metrics


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Backward-compatible wrapper around the shared CSV writer."""
    shared_write_csv(path, rows, fieldnames)


def exact_duplicate_rows(metrics: list[ImageMetric]) -> tuple[list[dict], set[str]]:
    """Build exact duplicate records and return filenames to remove from the cleaned set."""
    grouped: dict[str, list[ImageMetric]] = defaultdict(list)
    for item in metrics:
        if item.readable:
            grouped[item.md5].append(item)

    rows: list[dict] = []
    remove_names: set[str] = set()
    group_id = 1
    for _, group in grouped.items():
        if len(group) <= 1:
            continue
        ordered = sorted(group, key=lambda x: x.file_name)
        keep = ordered[0].file_name
        for item in ordered:
            action = "keep" if item.file_name == keep else "remove_duplicate"
            if action == "remove_duplicate":
                remove_names.add(item.file_name)
            rows.append(
                {
                    "group_id": group_id,
                    "file_name": item.file_name,
                    "label": item.label,
                    "md5": item.md5,
                    "action": action,
                    "kept_file": keep,
                }
            )
        group_id += 1
    return rows, remove_names


def near_duplicate_rows(metrics: list[ImageMetric], threshold: int) -> list[dict]:
    """Build candidate near-duplicate pairs using perceptual-hash distance."""
    readable = [m for m in metrics if m.readable and m.phash]
    rows: list[dict] = []
    for i, left in enumerate(readable):
        for right in readable[i + 1 :]:
            distance = hamming_hex(left.phash, right.phash)
            if distance <= threshold:
                rows.append(
                    {
                        "file_a": left.file_name,
                        "label_a": left.label,
                        "file_b": right.file_name,
                        "label_b": right.label,
                        "phash_distance": distance,
                        "review_suggestion": "manual_review",
                    }
                )
    return sorted(rows, key=lambda r: (r["phash_distance"], r["file_a"], r["file_b"]))


def percentile_suspects(metrics: list[ImageMetric], key: str, fraction: float, low: bool) -> list[ImageMetric]:
    """Select suspicious images from the low or high end of a numeric metric."""
    readable = [m for m in metrics if m.readable]
    count = max(1, math.ceil(len(readable) * fraction))
    return sorted(readable, key=lambda m: getattr(m, key), reverse=not low)[:count]


def copy_clean_dataset(source_dir: Path, clean_dir: Path, metrics: list[ImageMetric], remove_names: set[str]) -> None:
    """Copy readable, non-duplicate images into class directories as RGB PNG files."""
    clean_dir.mkdir(parents=True, exist_ok=True)
    for item in metrics:
        if not item.readable or item.file_name in remove_names:
            continue
        src = source_dir / item.file_name
        label_dir = clean_dir / item.label
        label_dir.mkdir(parents=True, exist_ok=True)
        dst = label_dir / item.file_name
        with Image.open(src) as img:
            img.convert("RGB").save(dst, format="PNG", optimize=True)


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a CJK-capable font when available and fall back to the PIL default."""
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


def compact_caption(name: str) -> str:
    """Create a short caption for review contact sheets."""
    stem = Path(name).stem
    label = stem.rsplit("_", 1)[0] if "_" in stem else stem
    number = stem.rsplit("_", 1)[-1] if "_" in stem else stem
    if "明显暴露垃圾" in label:
        short_label = "trash"
    elif "大于5m" in label or "大于5m²" in label:
        short_label = "large_bare_soil"
    elif "大于1m" in label or "大于1m²" in label:
        short_label = "small_bare_soil"
    else:
        short_label = label[:6]
    return f"{short_label}_{number}"


def create_contact_sheet(
    source_dir: Path,
    output_path: Path,
    file_names: list[str],
    title: str,
    columns: int = 5,
    thumb_size: tuple[int, int] = (180, 140),
) -> None:
    """Create a thumbnail contact sheet for manual review."""
    if not file_names:
        return
    rows = math.ceil(len(file_names) / columns)
    title_height = 42
    caption_height = 30
    gap = 14
    width = columns * thumb_size[0] + (columns + 1) * gap
    height = title_height + rows * (thumb_size[1] + caption_height + gap) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(18)
    caption_font = load_font(12)
    draw.text((gap, 10), title, fill=(20, 20, 20), font=title_font)

    for idx, name in enumerate(file_names):
        x = gap + (idx % columns) * (thumb_size[0] + gap)
        y = title_height + (idx // columns) * (thumb_size[1] + caption_height + gap)
        try:
            with Image.open(source_dir / name) as img:
                thumb = ImageOps.contain(img.convert("RGB"), (thumb_size[0], thumb_size[1]))
        except Exception:
            thumb = Image.new("RGB", thumb_size, (230, 230, 230))
        px = x + (thumb_size[0] - thumb.width) // 2
        py = y + (thumb_size[1] - thumb.height) // 2
        sheet.paste(thumb, (px, py))
        draw.text(
            (x + 4, y + thumb_size[1] + 6), f"{idx + 1}. {compact_caption(name)}", fill=(20, 20, 20), font=caption_font
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_report(
    report_dir: Path,
    metrics: list[ImageMetric],
    duplicate_rows: list[dict],
    near_rows: list[dict],
    remove_names: set[str],
    blur_rows: list[dict],
    dark_rows: list[dict],
    bright_rows: list[dict],
    low_green_rows: list[dict],
) -> None:
    """Write the markdown data-cleaning summary report."""
    readable = [m for m in metrics if m.readable]
    class_before = Counter(m.label for m in readable)
    class_after = Counter(m.label for m in readable if m.file_name not in remove_names)
    mode_counter = Counter(m.mode for m in readable)
    dim_counter = Counter((m.width, m.height) for m in readable)

    lines = [
        "# Urban Greening Maintenance Image Dataset Cleaning Report",
        "",
        "## 1. Dataset Overview",
        "",
        f"- Raw files: {len(metrics)}",
        f"- Readable images: {len(readable)}",
        f"- Corrupt or unreadable images: {len(metrics) - len(readable)}",
        f"- Exact duplicate copies removed automatically: {len(remove_names)}",
        f"- Images after cleaning: {len(readable) - len(remove_names)}",
        f"- Image mode distribution: {dict(mode_counter)}",
        f"- Main resolutions: {', '.join([f'{w}x{h}: {c}' for (w, h), c in dim_counter.most_common(5)])}",
        "",
        "## 2. Class Distribution",
        "",
        "| Class | Before cleaning | After cleaning |",
        "|---|---:|---:|",
    ]
    for label in sorted(class_before):
        lines.append(f"| {label} | {class_before[label]} | {class_after[label]} |")

    lines.extend(
        [
            "",
            "## 3. Cleaning Method",
            "",
            "- PIL is used to verify image readability, and cleaned images are saved as RGB PNG files.",
            (
                "- MD5 is used to detect exact duplicates. The first file in each sorted duplicate group is kept; "
                "the rest are removed from the cleaned set."
            ),
            (
                "- Perceptual hash is used to detect near duplicates with Hamming distance <= 5. These records are "
                "only used for manual review."
            ),
            "- Grayscale Laplacian variance is used as a blur heuristic. Lower values are more likely to be blurred.",
            "- Grayscale mean brightness is used to list underexposed and overexposed candidates.",
            (
                "- HSV vegetation ratio and soil-like color ratio are used as weak relevance checks for manual "
                "review only."
            ),
            "",
            "## 4. Automatic Processing Results",
            "",
            (
                "- Exact duplicate groups: "
                f"{len({row['group_id'] for row in duplicate_rows}) if duplicate_rows else 0}"
            ),
            f"- Images involved in exact duplicate groups: {len(duplicate_rows)}",
            f"- Duplicate copies excluded from the cleaned dataset: {len(remove_names)}",
            "- The raw data directory is not modified.",
            "",
            "## 5. Manual Review Suggestions",
            "",
            f"- Near-duplicate pairs: {len(near_rows)}, see `near_duplicates.csv`.",
            f"- Blur candidates: {len(blur_rows)}, see `suspicious_blur.csv`.",
            f"- Dark-image candidates: {len(dark_rows)}, see `suspicious_dark.csv`.",
            f"- Bright-image candidates: {len(bright_rows)}, see `suspicious_bright.csv`.",
            f"- Low-vegetation-ratio candidates: {len(low_green_rows)}, see `suspicious_low_green.csv`.",
            "",
            "## 6. Outputs",
            "",
            "- `all_images_metrics.csv`: cleaning metrics for every image.",
            "- `class_distribution.csv`: class distribution before and after cleaning.",
            "- `exact_duplicates.csv`: exact duplicate handling records.",
            "- `near_duplicates.csv`: near-duplicate candidates.",
            "- `review_sheets/`: thumbnail sheets for manual review.",
            "- `../data/cleaned/`: cleaned dataset organized by class.",
        ]
    )
    (report_dir / "cleaning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_logging()
    defaults = load_yaml_config("configs/data.yaml")["cleaning"]
    parser = argparse.ArgumentParser(description="Clean urban greening classification image dataset.")
    parser.add_argument("--source", type=Path, default=Path(defaults["source"]))
    parser.add_argument("--report", type=Path, default=Path(defaults["report"]))
    parser.add_argument("--clean", type=Path, default=Path(defaults["clean"]))
    parser.add_argument("--near-threshold", type=int, default=defaults["near_threshold"])
    parser.add_argument("--suspect-fraction", type=float, default=defaults["suspect_fraction"])
    parser.add_argument("--no-copy", action="store_true", help="Only generate reports; do not create cleaned dataset.")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    files = [p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    args.report.mkdir(parents=True, exist_ok=True)

    metrics = collect_metrics(files)
    duplicate_rows, remove_names = exact_duplicate_rows(metrics)
    near_rows = near_duplicate_rows(metrics, args.near_threshold)

    blur_items = percentile_suspects(metrics, "blur_laplacian_var", args.suspect_fraction, low=True)
    dark_items = percentile_suspects(metrics, "brightness_mean", args.suspect_fraction / 2, low=True)
    bright_items = percentile_suspects(metrics, "brightness_mean", args.suspect_fraction / 2, low=False)
    low_green_items = percentile_suspects(metrics, "vegetation_ratio", args.suspect_fraction, low=True)

    metric_rows = [asdict(m) for m in metrics]
    write_csv(args.report / "all_images_metrics.csv", metric_rows)
    write_csv(args.report / "exact_duplicates.csv", duplicate_rows)
    write_csv(args.report / "near_duplicates.csv", near_rows)

    blur_rows = [asdict(m) for m in blur_items]
    dark_rows = [asdict(m) for m in dark_items]
    bright_rows = [asdict(m) for m in bright_items]
    low_green_rows = [asdict(m) for m in low_green_items]
    write_csv(args.report / "suspicious_blur.csv", blur_rows)
    write_csv(args.report / "suspicious_dark.csv", dark_rows)
    write_csv(args.report / "suspicious_bright.csv", bright_rows)
    write_csv(args.report / "suspicious_low_green.csv", low_green_rows)

    before = Counter(m.label for m in metrics if m.readable)
    after = Counter(m.label for m in metrics if m.readable and m.file_name not in remove_names)
    dist_rows = [
        {"label": label, "before_cleaning": before[label], "after_cleaning": after[label]} for label in sorted(before)
    ]
    write_csv(args.report / "class_distribution.csv", dist_rows)

    sheet_dir = args.report / "review_sheets"
    create_contact_sheet(
        args.source, sheet_dir / "suspicious_blur.jpg", [m.file_name for m in blur_items[:25]], "Blur Candidate Review"
    )
    create_contact_sheet(
        args.source,
        sheet_dir / "suspicious_dark.jpg",
        [m.file_name for m in dark_items[:25]],
        "Dark Image Candidate Review",
    )
    create_contact_sheet(
        args.source,
        sheet_dir / "suspicious_bright.jpg",
        [m.file_name for m in bright_items[:25]],
        "Bright Image Candidate Review",
    )
    create_contact_sheet(
        args.source,
        sheet_dir / "suspicious_low_green.jpg",
        [m.file_name for m in low_green_items[:25]],
        "Low Vegetation Ratio Review",
    )

    if duplicate_rows:
        duplicate_names = [row["file_name"] for row in duplicate_rows[:25]]
        create_contact_sheet(args.source, sheet_dir / "exact_duplicates.jpg", duplicate_names, "Exact Duplicate Review")

    if not args.no_copy:
        if args.clean.exists():
            shutil.rmtree(args.clean)
        copy_clean_dataset(args.source, args.clean, metrics, remove_names)

    write_report(
        args.report,
        metrics,
        duplicate_rows,
        near_rows,
        remove_names,
        blur_rows,
        dark_rows,
        bright_rows,
        low_green_rows,
    )

    LOGGER.info("Raw images: %s", len(metrics))
    LOGGER.info("Readable images: %s", sum(1 for m in metrics if m.readable))
    LOGGER.info("Exact duplicate copies removed: %s", len(remove_names))
    LOGGER.info("Cleaned images: %s", sum(1 for m in metrics if m.readable and m.file_name not in remove_names))
    LOGGER.info("Report directory: %s", args.report)
    if not args.no_copy:
        LOGGER.info("Cleaned data directory: %s", args.clean)


if __name__ == "__main__":
    main()
