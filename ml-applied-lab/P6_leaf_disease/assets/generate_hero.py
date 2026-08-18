"""
generate_hero
=============

Hero image for the P6 Leaf Disease README.

Composes a 2×2 panel:
    - top-left   : ResNet50 Grad-CAM heatmap overlaid on a synthetic leaf.
    - top-right  : EfficientNetV2 Grad-CAM heatmap.
    - bottom-left: ViT-B/16 Grad-CAM heatmap.
    - bottom-right: ONNX vs PyTorch logit parity scatter plot.

Re-run after any model change to refresh ``assets/hero.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import logging  # noqa: E402
logging.getLogger("torch.onnx").setLevel(logging.CRITICAL)
logging.getLogger("onnxscript").setLevel(logging.CRITICAL)

from shared import apply_style  # noqa: E402
apply_style()

import torch  # noqa: E402

from dataset import (  # noqa: E402
    DEFAULT_CLASSES, DEFAULT_CONFIG, _generate_synthetic_leaf_image,
    make_augmentations,
)
from model import (  # noqa: E402
    BackboneKind, LeafClassifier, GradCAM,
    export_to_onnx, load_onnx_session, predict_with_onnx,
)


def _denormalize(t: torch.Tensor) -> torch.Tensor:
    """ImageNet denormalization: x * std + mean.

    Works on both (C, H, W) and (B, C, H, W) tensors.
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    # Reshape mean/std to broadcast against the input.
    if t.ndim == 3:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif t.ndim == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    return t * std + mean


def main() -> None:
    import tempfile

    # Generate a synthetic leaf image of a "Tomato_Late_blight" class.
    leaf_img = _generate_synthetic_leaf_image(class_idx=2, size=224, seed=42)
    # Apply the val augmentation pipeline (Resize+CenterCrop+Normalize+ToTensor).
    augs = make_augmentations(224)
    augmented = augs["val"](image=leaf_img)
    input_tensor = augmented["image"].unsqueeze(0)  # (1, 3, 224, 224)

    # For display: denormalize + clip to [0, 1]. Albumentations returns
    # (C, H, W) tensors (no batch dim); we denormalize the (3, H, W) tensor.
    img_disp = _denormalize(input_tensor[0]).permute(1, 2, 0).numpy()
    img_disp = np.clip(img_disp, 0, 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    # Train each backbone, compute Grad-CAM, render the overlay.
    backbones = ["resnet50", "efficientnet_v2", "vit_b_16"]
    titles = ["ResNet50 (23.5M params)", "EfficientNetV2-S (20.2M params)",
              "ViT-B/16 (85.8M params)"]
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0]]

    import gc
    for i, (backbone_name, title) in enumerate(zip(backbones, titles)):
        ax = axes_flat[i]
        model = LeafClassifier(BackboneKind(backbone_name), num_classes=10, pretrained=False)
        model.eval()
        cam = GradCAM(model, target_layer=model.target_layer_name)
        # Use the synthetic input (with requires_grad=True for backward).
        x_in = input_tensor.clone().detach().requires_grad_(True)
        with torch.no_grad():
            pred = int(model(x_in).argmax(dim=1).item())
        heatmap = cam(x_in, class_idx=pred)
        cam.remove_hooks()

        ax.imshow(img_disp)
        ax.imshow(heatmap, cmap="jet", alpha=0.5)
        ax.set_title(f"{title}\nGrad-CAM → predicted class: {DEFAULT_CLASSES[pred][:25]}", loc="left", fontsize=10)
        ax.axis("off")

        # Free memory between backbones (ViT-B/16 is 350MB).
        del model, cam, x_in, heatmap
        gc.collect()

    # Bottom-right: ONNX vs PyTorch parity scatter.
    ax = axes[1, 1]
    model = LeafClassifier(BackboneKind.RESNET50, num_classes=10, pretrained=False)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "leaf.onnx"
        export_to_onnx(model, onnx_path, image_size=224)
        session = load_onnx_session(onnx_path)

        # Generate 8 random inputs, run through both PyTorch and ONNX,
        # scatter-plot the max-logit per sample.
        n_samples = 16
        pt_logits = []
        onnx_logits = []
        for _ in range(n_samples):
            x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
            with torch.no_grad():
                pt_out = model(x).numpy()[0]
            _, onnx_probas = predict_with_onnx(session, x.numpy())
            # Compute ONNX logits from probabilities.
            onnx_out = np.log(onnx_probas[0] + 1e-12)
            pt_logits.append(pt_out.max())
            onnx_logits.append(onnx_out.max())

    ax.scatter(pt_logits, onnx_logits, s=80, color="#0072B2", alpha=0.7, edgecolor="white")
    # Diagonal line.
    lim = [min(min(pt_logits), min(onnx_logits)) - 0.5, max(max(pt_logits), max(onnx_logits)) + 0.5]
    ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.6, label="parity (y=x)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("PyTorch max logit")
    ax.set_ylabel("ONNX max logit")
    ax.set_title("ONNX runtime parity vs PyTorch\n(ResNet50, 16 random inputs)", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.suptitle("Leaf Disease — Comparative Vision Architectures + Grad-CAM + ONNX Export",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
