#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P6_leaf_disease — the comparative vision architecture
benchmark with Grad-CAM attribution + ONNX/TFLite export.

Usage
-----
::

    # 1. Default: synthetic dataset, ResNet50 backbone, 3 epochs
    python train.py

    # 2. Switch backbone
    python train.py --backbone efficientnet_v2
    python train.py --backbone vit_b_16

    # 3. Real Plant Village dataset (download first)
    python train.py --data-dir data/plantvillage --backbone resnet50 --epochs 20

    # 4. Mixed-precision + cosine annealing + ONNX/TFLite export
    python train.py --mixed-precision --cosine-annealing \\
        --onnx-out models/leaf_resnet50.onnx \\
        --tflite-out models/leaf_resnet50.tflite

Exit codes
----------
* 0  : training completed + ONNX exported.
* 1  : usage error.
* 2  : data loading failed.
* 3  : training failed.
* 4  : ONNX export failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
try:
    from shared import apply_style
    apply_style()
except Exception:  # pragma: no cover
    pass

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from dataset import (  # noqa: E402
    DEFAULT_CONFIG, DEFAULT_CLASSES, LeafDiseaseConfig,
    build_dataloaders, make_synthetic_dataset,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, BackboneKind, LeafClassifier, GradCAM,
    export_to_onnx, export_to_tflite,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("leaf_train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leaf_train",
        description="P6 Leaf Disease — train ResNet50/EfficientNetV2/ViT on Plant Village.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # Default: synthetic data, ResNet50, 3 epochs
  python train.py

  # Switch backbone
  python train.py --backbone vit_b_16

  # Real Plant Village dataset
  python train.py --data-dir data/plantvillage --backbone resnet50 --epochs 20

  # Mixed-precision + cosine annealing + ONNX export
  python train.py --mixed-precision --cosine-annealing \\
      --onnx-out models/leaf_resnet50.onnx
""",
    )
    parser.add_argument(
        "--backbone", "-b",
        choices=list(CANDIDATE_MODELS.keys()),
        default="resnet50",
        help="Vision backbone (default: resnet50).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to Plant Village dataset (default: synthetic).",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=50,
        help="Synthetic dataset images per class (default: 50).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3 — synthetic test).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="L2 weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0 — sync).",
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="Enable AMP autocast + GradScaler (default: off — CPU-only torch).",
    )
    parser.add_argument(
        "--cosine-annealing", action="store_true",
        help="Use CosineAnnealingLR instead of StepLR (default: off).",
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Disable ImageNet pretraining (faster startup, much worse accuracy).",
    )
    parser.add_argument(
        "--onnx-out", default=None,
        help="Optional path to save the best model as ONNX.",
    )
    parser.add_argument(
        "--tflite-out", default=None,
        help="Optional path to save the best model as TFLite (requires onnx2tf).",
    )
    parser.add_argument(
        "--gradcam-out", default=None,
        help="Optional path to save a Grad-CAM visualization PNG.",
    )
    parser.add_argument(
        "--metrics-json", default=None,
        help="Optional path to dump final metrics as JSON.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Compute accuracy + top-3 accuracy on a dataloader."""
    model.eval()
    correct = 0
    top3_correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss_sum += float(criterion(out, y).item())
            _, pred = out.max(dim=1)
            correct += (pred == y).sum().item()
            top3 = out.topk(3, dim=1).indices
            top3_correct += sum(int(y[i].item() in top3[i].tolist()) for i in range(len(y)))
            total += len(y)
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
    }


def _plot_gradcam(model: LeafClassifier, loader: DataLoader, output_path: Path,
                  class_names: List[str], device: torch.device, n_examples: int = 8) -> None:
    """Render a grid of (image, Grad-CAM overlay, true label, pred) tuples."""
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    n = min(n_examples, len(images))

    cam = GradCAM(model, target_layer=model.target_layer_name)

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    # ImageNet denormalization for display.
    mean = torch.tensor([0.485, 0.456,0.406]).view(1,3,1,1).to(device)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)

    with torch.no_grad():
        preds = model(images[:n]).argmax(dim=1).cpu().tolist()

    for i in range(n):
        img_disp = images[i:i+1]
        # Display image (denormalized).
        img_show = (img_disp * std + mean).squeeze(0).cpu().numpy()
        img_show = np.clip(np.transpose(img_show, (1, 2, 0)), 0, 1)

        axes[0, i].imshow(img_show)
        axes[0, i].set_title(f"true: {class_names[int(labels[i].item())][:12]}\npred: {class_names[preds[i]][:12]}", fontsize=7)
        axes[0, i].axis("off")

        # Grad-CAM.
        heatmap = cam(img_disp.clone().detach().requires_grad_(True),
                       class_idx=preds[i])
        axes[1, i].imshow(img_show)
        axes[1, i].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[1, i].axis("off")

    cam.remove_hooks()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)
    elif args.verbose == 1:
        log.setLevel(logging.DEBUG)

    # Device setup.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Device: %s (mixed_precision=%s, cuda_available=%s)",
             device, args.mixed_precision, torch.cuda.is_available())

    # On CPU, mixed-precision is a no-op — warn but proceed.
    if args.mixed_precision and device.type == "cpu":
        log.warning("--mixed-precision is set but no CUDA device available; AMP is a no-op on CPU.")
        args.mixed_precision = False

    # Step 1 — dataloaders.
    try:
        log.info("Building dataloaders ...")
        train_loader, val_loader, test_loader, manifest = build_dataloaders(
            data_dir=args.data_dir,
            config=DEFAULT_CONFIG,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_per_class=args.n_per_class,
            seed=args.seed,
        )
        log.info("  manifest: %d rows, train=%d batches, val=%d, test=%d",
                 len(manifest), len(train_loader), len(val_loader), len(test_loader))
    except Exception as exc:
        log.error("Dataloader setup failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 2

    # Step 2 — model.
    try:
        log.info("Building %s backbone (pretrained=%s, num_classes=%d) ...",
                 args.backbone, not args.no_pretrained, DEFAULT_CONFIG.num_classes)
        model = LeafClassifier(
            BackboneKind(args.backbone),
            num_classes=DEFAULT_CONFIG.num_classes,
            pretrained=not args.no_pretrained,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log.info("  %s: %.1fM params, target_layer=%s",
                 args.backbone, n_params / 1e6, model.target_layer_name)
    except Exception as exc:
        log.error("Model build failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 3

    # Step 3 — optimizer + scheduler + criterion.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    if args.cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        log.info("  Scheduler: CosineAnnealingLR(T_max=%d)", args.epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.5)

    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None

    # Step 4 — training loop.
    best_val_acc = -1.0
    best_epoch = -1
    history: List[Dict] = []

    log.info("Training for %d epoch(s) ...", args.epochs)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            if args.mixed_precision:
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * len(y)
            running_correct += int((out.argmax(dim=1) == y).sum().item())
            running_total += len(y)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_metrics = _evaluate(model, val_loader, device)
        scheduler.step()

        log.info(
            "  Epoch %d/%d — train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_top3=%.4f",
            epoch + 1, args.epochs, train_loss, train_acc,
            val_metrics["loss"], val_metrics["accuracy"], val_metrics["top3_accuracy"],
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
            "val_top3_acc": val_metrics["top3_accuracy"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch + 1

    # Step 5 — final test eval.
    log.info("Evaluating on test set ...")
    test_metrics = _evaluate(model, test_loader, device)
    log.info("  Test: loss=%.4f acc=%.4f top3=%.4f",
             test_metrics["loss"], test_metrics["accuracy"], test_metrics["top3_accuracy"])

    # Step 6 — ONNX export.
    onnx_path_str = args.onnx_out or f"models/leaf_{args.backbone}.onnx"
    try:
        onnx_path = export_to_onnx(model, onnx_path_str, image_size=DEFAULT_CONFIG.image_size)
        log.info("✓ Exported best model to ONNX → %s (%.1f KB)",
                 onnx_path.resolve(), onnx_path.stat().st_size / 1024)
    except Exception as exc:
        log.error("ONNX export failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 4

    # Step 7 — optional TFLite export.
    if args.tflite_out:
        try:
            tflite_path = export_to_tflite(model, args.tflite_out,
                                           image_size=DEFAULT_CONFIG.image_size)
            log.info("✓ Exported to TFLite → %s", tflite_path.resolve())
        except Exception as exc:
            log.warning("TFLite export failed (continuing): %s", exc)

    # Step 8 — optional Grad-CAM plot.
    if args.gradcam_out:
        try:
            _plot_gradcam(model, test_loader, Path(args.gradcam_out),
                          list(DEFAULT_CLASSES), device)
            log.info("Saved Grad-CAM visualization → %s", args.gradcam_out)
        except Exception as exc:
            log.warning("Grad-CAM plot failed: %s", exc)

    # Step 9 — optional metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {
                "backbone": args.backbone,
                "data_dir": args.data_dir or "synthetic",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "mixed_precision": args.mixed_precision,
                "cosine_annealing": args.cosine_annealing,
                "pretrained": not args.no_pretrained,
                "n_per_class": args.n_per_class,
                "seed": args.seed,
                "num_classes": DEFAULT_CONFIG.num_classes,
                "image_size": DEFAULT_CONFIG.image_size,
                "classes": list(DEFAULT_CLASSES),
            },
            "metrics": {
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["accuracy"],
                "test_top3_acc": test_metrics["top3_accuracy"],
                "n_params": int(sum(p.numel() for p in model.parameters())),
                "onnx_path": str(Path(onnx_path_str).resolve()),
            },
            "history": history,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Summary line.
    print(f"BACKBONE={args.backbone}")
    print(f"BEST_VAL_ACC={best_val_acc:.4f}")
    print(f"BEST_EPOCH={best_epoch}")
    print(f"TEST_ACC={test_metrics['accuracy']:.4f}")
    print(f"TEST_TOP3_ACC={test_metrics['top3_accuracy']:.4f}")
    print(f"ONNX_PATH={Path(onnx_path_str).resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
