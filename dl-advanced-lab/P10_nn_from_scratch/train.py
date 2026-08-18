#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P10_nn_from_scratch — trains an MLP or CNN on MNIST/CIFAR-10
using the custom autograd engine, verifies gradient correctness against PyTorch,
and logs training loss/accuracy curves.

Usage
-----
::

    # 1. Train MLP on synthetic MNIST (default)
    python train.py

    # 2. Train CNN on synthetic CIFAR-10
    python train.py --dataset cifar10 --model cnn

    # 3. Use real MNIST (requires torchvision)
    python train.py --use-real

    # 4. Run PyTorch gradient parity check only
    python train.py --grad-check-only

    # 5. Custom hyperparameters
    python train.py --epochs 10 --batch-size 32 --lr 0.01 --optimizer adam

Exit codes
----------
* 0  : training completed.
* 1  : usage error.
* 2  : data loading failed.
* 3  : training failed.
* 4  : gradient check failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from autograd import Tensor  # noqa: E402
import nn  # noqa: E402
from dataset import (  # noqa: E402
    MNIST_CONFIG, CIFAR_CONFIG,
    load_mnist, load_cifar10, batch_generator,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nn_train")


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def build_mlp(config, hidden_dim: int = 64) -> nn.Module:
    """Build a simple MLP: Flatten → Linear → ReLU → Linear → ReLU → Linear."""
    input_dim = config.channels * config.image_size * config.image_size
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, config.num_classes),
    )


def build_cnn(config) -> nn.Module:
    """Build a small CNN: Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Linear → Linear."""
    img = config.image_size
    # After conv1 (k=3): img-2; after pool1 (k=2): (img-2)//2.
    # After conv2 (k=3): ...; after pool2 (k=2): ...
    out1 = img - 2  # conv1
    out1 = out1 // 2  # pool1
    out2 = out1 - 2  # conv2
    out2 = out2 // 2  # pool2
    flat_dim = 8 * out2 * out2  # 8 out channels after conv2
    return nn.Sequential(
        nn.Conv2D(config.channels, 4, 3),
        nn.ReLU(),
        nn.MaxPool2D(2),
        nn.Conv2D(4, 8, 3),
        nn.ReLU(),
        nn.MaxPool2D(2),
        nn.Flatten(),
        nn.Linear(flat_dim, 32),
        nn.ReLU(),
        nn.Linear(32, config.num_classes),
    )


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------
def gradient_check_against_pytorch() -> Dict[str, float]:
    """Verify our autograd matches PyTorch for a sequence of ops.

    Returns a dict of {op_name: max_grad_diff}.
    """
    import torch
    results: Dict[str, float] = {}
    np.random.seed(42)

    # --- Test 1: add + mul + pow ---
    x_np = np.random.randn(3, 4)
    x_ours = Tensor(x_np, requires_grad=True)
    y_ours = ((x_ours * 2.0 + x_ours ** 2).sum())
    y_ours.backward()

    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = ((x_pt * 2.0 + x_pt ** 2).sum())
    y_pt.backward()
    results["add_mul_pow"] = float(np.abs(x_ours.grad - x_pt.grad.numpy()).max())

    # --- Test 2: matmul + broadcast ---
    A_np = np.random.randn(3, 4)
    b_np = np.random.randn(4)
    A_ours = Tensor(A_np, requires_grad=True)
    b_ours = Tensor(b_np, requires_grad=True)
    y_ours = (A_ours @ b_ours).sum()
    y_ours.backward()

    A_pt = torch.tensor(A_np, requires_grad=True)
    b_pt = torch.tensor(b_np, requires_grad=True)
    y_pt = (A_pt @ b_pt).sum()
    y_pt.backward()
    results["matmul_broadcast_A"] = float(np.abs(A_ours.grad - A_pt.grad.numpy()).max())
    results["matmul_broadcast_b"] = float(np.abs(b_ours.grad - b_pt.grad.numpy()).max())

    # --- Test 3: relu + sigmoid ---
    x_np = np.random.randn(5)
    x_ours = Tensor(x_np, requires_grad=True)
    y_ours = (x_ours.relu() + x_ours.sigmoid()).sum()
    y_ours.backward()

    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = (torch.relu(x_pt) + torch.sigmoid(x_pt)).sum()
    y_pt.backward()
    results["relu_sigmoid"] = float(np.abs(x_ours.grad - x_pt.grad.numpy()).max())

    # --- Test 4: cross_entropy ---
    logits_np = np.random.randn(4, 3)
    targets = np.array([0, 1, 2, 1])
    logits_ours = Tensor(logits_np, requires_grad=True)
    loss_ours = logits_ours.cross_entropy(targets)
    loss_ours.backward()

    logits_pt = torch.tensor(logits_np, requires_grad=True)
    loss_pt = torch.nn.functional.cross_entropy(logits_pt, torch.tensor(targets))
    loss_pt.backward()
    results["cross_entropy_loss"] = float(abs(loss_ours.data - loss_pt.item()))
    results["cross_entropy_grad"] = float(np.abs(logits_ours.grad - logits_pt.grad.numpy()).max())

    # --- Test 5: mse ---
    pred_np = np.random.randn(3, 2)
    target_np = np.random.randn(3, 2)
    pred_ours = Tensor(pred_np, requires_grad=True)
    loss_ours = pred_ours.mse(target_np)
    loss_ours.backward()

    pred_pt = torch.tensor(pred_np, requires_grad=True)
    loss_pt = torch.nn.functional.mse_loss(pred_pt, torch.tensor(target_np))
    loss_pt.backward()
    results["mse_loss"] = float(abs(loss_ours.data - loss_pt.item()))
    results["mse_grad"] = float(np.abs(pred_ours.grad - pred_pt.grad.numpy()).max())

    # --- Test 6: Conv2D + MaxPool2D ---
    x_np = np.random.randn(2, 3, 8, 8)
    conv = nn.Conv2D(3, 4, 3)
    pool = nn.MaxPool2D(2)
    x_ours = Tensor(x_np, requires_grad=True)
    out_ours = pool(conv(x_ours))
    loss_ours = out_ours.sum()
    loss_ours.backward()

    x_pt = torch.tensor(x_np, requires_grad=True)
    conv_pt = torch.nn.Conv2d(3, 4, 3, bias=True)
    conv_pt.weight.data = torch.tensor(conv.weight.data)
    conv_pt.bias.data = torch.tensor(conv.bias.data)
    pool_pt = torch.nn.MaxPool2d(2)
    out_pt = pool_pt(conv_pt(x_pt))
    loss_pt = out_pt.sum()
    loss_pt.backward()
    results["conv_x_grad"] = float(np.abs(x_ours.grad - x_pt.grad.numpy()).max())
    results["conv_w_grad"] = float(np.abs(conv.weight.grad - conv_pt.weight.grad.numpy()).max())

    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module, optimizer: nn.Optimizer, X: np.ndarray, y: np.ndarray,
    batch_size: int, seed: int,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    for X_batch, y_batch in batch_generator(X, y, batch_size, shuffle=True, seed=seed):
        # Forward.
        x_tensor = Tensor(X_batch, requires_grad=False)
        logits = model(x_tensor)
        loss = nn.cross_entropy(logits, y_batch)

        # Backward.
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats.
        total_loss += float(loss.data) * len(y_batch)
        preds = logits.data.argmax(axis=1)
        n_correct += int((preds == y_batch).sum())
        n_total += len(y_batch)

    return total_loss / n_total, n_correct / n_total


def evaluate(
    model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int = 64,
) -> Tuple[float, float]:
    """Evaluate on a dataset. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    for X_batch, y_batch in batch_generator(X, y, batch_size, shuffle=False):
        x_tensor = Tensor(X_batch, requires_grad=False)
        logits = model(x_tensor)
        loss = nn.cross_entropy(logits, y_batch)
        total_loss += float(loss.data) * len(y_batch)
        preds = logits.data.argmax(axis=1)
        n_correct += int((preds == y_batch).sum())
        n_total += len(y_batch)
    model.train()
    return total_loss / n_total, n_correct / n_total


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nn_train",
        description="P10 NN From Scratch — train MLP/CNN on MNIST/CIFAR-10.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # Default: synthetic MNIST, MLP, 3 epochs
  python train.py

  # CNN on synthetic CIFAR-10
  python train.py --dataset cifar10 --model cnn

  # Gradient check only (no training)
  python train.py --grad-check-only
""",
    )
    parser.add_argument(
        "--dataset", choices=["mnist", "cifar10"], default="mnist",
        help="Dataset (default: mnist).",
    )
    parser.add_argument(
        "--model", choices=["mlp", "cnn"], default="mlp",
        help="Model architecture (default: mlp).",
    )
    parser.add_argument(
        "--use-real", action="store_true",
        help="Download real MNIST/CIFAR-10 via torchvision (default: synthetic).",
    )
    parser.add_argument(
        "--n-train", type=int, default=1000,
        help="Synthetic train set size (default: 1000).",
    )
    parser.add_argument(
        "--n-test", type=int, default=200,
        help="Synthetic test set size (default: 200).",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--optimizer", choices=["sgd", "adam"], default="sgd",
        help="Optimizer (default: sgd).",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64,
        help="MLP hidden dimension (default: 64).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--grad-check-only", action="store_true",
        help="Run PyTorch gradient parity check only (no training).",
    )
    parser.add_argument(
        "--metrics-json", default=None,
        help="Optional path to dump training metrics as JSON.",
    )
    parser.add_argument(
        "--training-plot", default=None,
        help="Optional path to save training loss/accuracy curves PNG.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity.",
    )
    return parser


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_training_curves(history: List[Dict], output_path: Path) -> None:
    """Plot training loss + accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    axes[0].plot(epochs, train_loss, "o-", color="#0072B2", label="train loss")
    axes[0].plot(epochs, val_loss, "s-", color="#D55E00", label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / validation loss", loc="left")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, "o-", color="#0072B2", label="train acc")
    axes[1].plot(epochs, val_acc, "s-", color="#D55E00", label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training / validation accuracy", loc="left")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
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

    # Step 1 — gradient check (always run).
    log.info("Running PyTorch gradient parity check ...")
    try:
        grad_results = gradient_check_against_pytorch()
    except Exception as exc:
        log.error("Gradient check failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 4

    log.info("  Gradient check results (max diff vs PyTorch):")
    all_passed = True
    for op, diff in grad_results.items():
        status = "OK" if diff < 1e-5 else "FAIL"
        if diff >= 1e-5:
            all_passed = False
        log.info("    %-30s %s (max diff = %.2e)", op, status, diff)

    if not all_passed:
        log.error("Gradient check FAILED — some ops differ from PyTorch by > 1e-5.")
        return 4
    log.info("  ✓ All gradients match PyTorch to 1e-5 precision.")

    # Early exit if --grad-check-only.
    if args.grad_check_only:
        print(f"GRAD_CHECK_PASSED={'true' if all_passed else 'false'}")
        return 0

    # Step 2 — load dataset.
    try:
        log.info("Loading %s dataset ...", args.dataset)
        if args.dataset == "mnist":
            ds = load_mnist(
                n_train_synthetic=args.n_train, n_test_synthetic=args.n_test,
                seed=args.seed, use_real=args.use_real,
            )
        else:
            ds = load_cifar10(
                n_train_synthetic=args.n_train, n_test_synthetic=args.n_test,
                seed=args.seed, use_real=args.use_real,
            )
        log.info("  Loaded train=%d, test=%d (source=%s)", ds.n_train, ds.n_test, ds.source)
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 2

    # Step 3 — build model.
    try:
        np.random.seed(args.seed)
        if args.model == "mlp":
            model = build_mlp(ds.config, hidden_dim=args.hidden_dim)
        else:
            model = build_cnn(ds.config)
        n_params = sum(p.data.size for p in model.parameters())
        log.info("  Built %s with %d parameters", args.model, n_params)
    except Exception as exc:
        log.error("Model build failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 3

    # Step 4 — optimizer.
    if args.optimizer == "sgd":
        optimizer = nn.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = nn.Adam(model.parameters(), lr=args.lr)
    log.info("  Optimizer: %s (lr=%.4f)", args.optimizer, args.lr)

    # Step 5 — training loop.
    history: List[Dict] = []
    log.info("Training for %d epoch(s) ...", args.epochs)
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, optimizer, ds.X_train, ds.y_train,
            batch_size=args.batch_size, seed=args.seed + epoch,
        )
        val_loss, val_acc = evaluate(model, ds.X_test, ds.y_test, batch_size=64)
        elapsed = time.perf_counter() - t0
        log.info(
            "  Epoch %d/%d — train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f (%.1fs)",
            epoch + 1, args.epochs, train_loss, train_acc, val_loss, val_acc, elapsed,
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

    # Step 6 — final metrics.
    final_train_loss, final_train_acc = evaluate(model, ds.X_train, ds.y_train, batch_size=64)
    final_val_loss, final_val_acc = evaluate(model, ds.X_test, ds.y_test, batch_size=64)
    log.info("Final: train_acc=%.4f, val_acc=%.4f", final_train_acc, final_val_acc)

    # Optional metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {
                "dataset": args.dataset, "model": args.model,
                "use_real": args.use_real, "epochs": args.epochs,
                "batch_size": args.batch_size, "lr": args.lr,
                "optimizer": args.optimizer, "hidden_dim": args.hidden_dim,
                "seed": args.seed,
            },
            "gradient_check": grad_results,
            "final_metrics": {
                "train_loss": final_train_loss, "train_acc": final_train_acc,
                "val_loss": final_val_loss, "val_acc": final_val_acc,
                "n_params": n_params,
            },
            "history": history,
            "dataset_source": ds.source,
            "n_train": ds.n_train, "n_test": ds.n_test,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Optional training curves plot.
    if args.training_plot:
        try:
            _plot_training_curves(history, Path(args.training_plot))
            log.info("Saved training curves → %s", args.training_plot)
        except Exception as exc:
            log.warning("Failed to render training plot: %s", exc)

    # Summary line.
    print(f"GRAD_CHECK_PASSED={'true' if all_passed else 'false'}")
    print(f"FINAL_TRAIN_ACC={final_train_acc:.4f}")
    print(f"FINAL_VAL_ACC={final_val_acc:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
