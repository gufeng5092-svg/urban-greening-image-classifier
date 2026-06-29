from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from inference import CLASS_NAMES, build_transform, create_model
from matplotlib import font_manager
from PIL import Image

CHECKPOINT_DIR = APP_DIR / "checkpoints"
DEFAULT_IMAGE = APP_DIR / "examples" / "存在明显暴露垃圾_790.png"
DEFAULT_OUTPUT = APP_DIR / "model_stage_comparison.png"
MODEL_NAMES = ["efficientnet_b0", "mobilenet_v3_small", "resnet18"]
DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "resnet18": "ResNet18",
}
SHORT_CLASS_NAMES = ["Small bare soil", "Large bare soil", "Exposed trash"]
LOGGER = logging.getLogger(__name__)


def setup_plot_font() -> None:
    font_candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_fold1_model(model_name: str, device: torch.device) -> torch.nn.Module:
    model = create_model(model_name, len(CLASS_NAMES))
    checkpoint_path = CHECKPOINT_DIR / model_name / "fold_1.pth"
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    if model_name in {"efficientnet_b0", "mobilenet_v3_small"}:
        return model.features[-1]
    if model_name == "resnet18":
        return model.layer4[-1]
    raise ValueError(f"Unsupported model: {model_name}")


def grad_cam(
    model: torch.nn.Module,
    model_name: str,
    tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    layer = target_layer(model, model_name)
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze(0)
    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].detach().cpu().item())
    logits[0, pred_id].backward()

    forward_handle.remove()
    backward_handle.remove()

    if activations is None or gradients is None:
        raise RuntimeError(f"Failed to collect Grad-CAM tensors for {model_name}")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1).squeeze(0)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam[None, None, :, :],
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    cam = cam.detach().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, probs.detach().cpu().numpy(), pred_id, confidence


def make_heatmap_overlay(image_224: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[..., :3]
    overlay = 0.58 * image_224 + 0.42 * heatmap
    return np.clip(overlay, 0, 1)


def probability_text(probs: np.ndarray) -> str:
    lines = []
    for label, prob in zip(SHORT_CLASS_NAMES, probs):
        lines.append(f"{label}: {prob * 100:.2f}%")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate three-model stage comparison figure.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    args = parser.parse_args()

    setup_plot_font()
    device = torch.device("cpu")
    transform = build_transform()
    original = Image.open(args.image).convert("RGB")
    resized = original.resize((224, 224), Image.Resampling.BILINEAR)
    image_224 = np.asarray(resized).astype(np.float32) / 255.0
    tensor = transform(original).unsqueeze(0).to(device)

    fig = plt.figure(figsize=(15.8, 8.2), dpi=180)
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.15, 1.15, 1.15, 1.22],
        height_ratios=[1, 1],
        wspace=0.12,
        hspace=0.18,
    )

    ax_original = fig.add_subplot(grid[0, 0])
    ax_original.imshow(original)
    ax_original.set_title("Original Input Image", fontsize=14, pad=8)
    ax_original.axis("off")

    ax_pre = fig.add_subplot(grid[1, 0])
    ax_pre.imshow(image_224)
    ax_pre.set_title("Unified Preprocessing\nResize 224x224 + Normalize", fontsize=13, pad=8)
    ax_pre.axis("off")

    for col, model_name in enumerate(MODEL_NAMES, start=1):
        model = load_fold1_model(model_name, device)
        cam, probs, pred_id, confidence = grad_cam(model, model_name, tensor.clone())
        overlay = make_heatmap_overlay(image_224, cam)

        ax_heat = fig.add_subplot(grid[0, col])
        ax_heat.imshow(overlay)
        ax_heat.set_title(f"{DISPLAY_NAMES[model_name]}\nFeature Attention Heatmap", fontsize=13, pad=8)
        ax_heat.axis("off")

        ax_text = fig.add_subplot(grid[1, col])
        ax_text.axis("off")
        ax_text.text(
            0.02,
            0.96,
            (
                f"Predicted class: {SHORT_CLASS_NAMES[pred_id]}\n\n"
                f"Confidence: {confidence * 100:.2f}%\n\n"
                f"Class probabilities:\n{probability_text(probs)}"
            ),
            va="top",
            ha="left",
            fontsize=11.5,
            linespacing=1.6,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#f7faf8",
                "edgecolor": "#d9e5dd",
                "linewidth": 1.0,
            },
        )

    fig.suptitle("Three-Model Processing Stages and Attention Regions", fontsize=19, y=0.985)
    fig.text(
        0.5,
        0.025,
        (
            "Note: redder heatmap areas indicate stronger model attention for the current classification. "
            "Each model visualization uses fold 1."
        ),
        ha="center",
        fontsize=11.5,
        color="#4f5f57",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("saved: %s", args.output)


if __name__ == "__main__":
    main()
