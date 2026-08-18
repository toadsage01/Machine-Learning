"""
generate_hero
=============

Hero image for the P10 NN From Scratch README.

Composes a 2×2 panel:
    - top-left   : gradient check results bar chart (max diff vs PyTorch per op).
    - top-right  : training loss/accuracy curves on synthetic MNIST.
    - bottom-left: Conv2D + MaxPool2D forward shapes diagram.
    - bottom-right: DAG visualization for a simple computation.

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

from shared import apply_style  # noqa: E402
apply_style()

from autograd import Tensor  # noqa: E402
import nn  # noqa: E402
from dataset import load_mnist, batch_generator  # noqa: E402
from train import gradient_check_against_pytorch, build_mlp, train_one_epoch, evaluate  # noqa: E402


def main() -> None:
    # Run gradient check.
    grad_results = gradient_check_against_pytorch()

    # Train MLP on synthetic MNIST.
    ds = load_mnist(n_train_synthetic=500, n_test_synthetic=100, seed=42)
    np.random.seed(42)
    model = build_mlp(ds.config, hidden_dim=64)
    optimizer = nn.SGD(model.parameters(), lr=0.1, momentum=0.9)

    history = []
    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, ds.X_train, ds.y_train, batch_size=32, seed=42 + epoch,
        )
        val_loss, val_acc = evaluate(model, ds.X_test, ds.y_test, batch_size=64)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

    # Plot.
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # --- Top-left: gradient check results --------------------------------
    ax = axes[0, 0]
    ops = list(grad_results.keys())
    diffs = list(grad_results.values())
    colors = ["#009E73" if d < 1e-5 else "#D55E00" for d in diffs]
    bars = ax.barh(ops, diffs, color=colors)
    ax.axvline(1e-5, color="#2b2b2b", linestyle="--", linewidth=0.8,
               label="1e-5 threshold")
    ax.set_xlabel("Max gradient diff vs PyTorch (log scale)")
    ax.set_xscale("log")
    ax.set_title("Gradient parity check (all ops < 1e-5)", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # --- Top-right: training curves --------------------------------------
    ax = axes[0, 1]
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history], "o-", color="#0072B2",
            label="train loss", linewidth=2)
    ax.plot(epochs, [h["val_loss"] for h in history], "s-", color="#D55E00",
            label="val loss", linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(epochs, [h["train_acc"] for h in history], "o--", color="#009E73",
             label="train acc", linewidth=1.5, alpha=0.7)
    ax2.plot(epochs, [h["val_acc"] for h in history], "s--", color="#CC79A7",
             label="val acc", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color="#0072B2")
    ax2.set_ylabel("Accuracy", color="#009E73")
    ax.set_title("MLP training on synthetic MNIST (500 samples)", loc="left", fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: Conv2D + MaxPool2D shape flow ----------------------
    ax = axes[1, 0]
    ax.set_axis_off()
    ax.set_title("Conv2D + MaxPool2D forward shape flow", loc="left", fontsize=11)
    shapes = [
        ("Input", "(B, 3, 8, 8)"),
        ("Conv2D(3→4, k=3)", "(B, 4, 6, 6)"),
        ("MaxPool2D(2)", "(B, 4, 3, 3)"),
        ("Conv2D(4→8, k=3)", "(B, 8, 1, 1)"),
        ("Flatten", "(B, 8)"),
        ("Linear(8→10)", "(B, 10)"),
    ]
    for i, (name, shape) in enumerate(shapes):
        y = len(shapes) - i - 1
        ax.text(0.1, y, name, fontsize=10, fontweight="bold", va="center")
        ax.text(0.6, y, shape, fontsize=9, va="center", color="#0072B2",
                family="monospace")
        if i < len(shapes) - 1:
            ax.annotate("", xy=(0.35, y - 0.4), xytext=(0.35, y - 0.6),
                        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(shapes) - 0.5)

    # --- Bottom-right: autograd DAG visualization ------------------------
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("Autograd DAG: y = ((x * W + b).relu()).sum()", loc="left", fontsize=11)
    # Draw nodes.
    nodes = [
        ("x", 0.1, 0.8, "#0072B2"),
        ("W", 0.1, 0.5, "#0072B2"),
        ("b", 0.1, 0.2, "#0072B2"),
        ("x * W", 0.4, 0.65, "#D55E00"),
        ("+ b", 0.6, 0.5, "#D55E00"),
        ("relu", 0.75, 0.5, "#009E73"),
        ("sum → y", 0.9, 0.5, "#2b2b2b"),
    ]
    for name, x, y, color in nodes:
        ax.scatter(x, y, s=300, color=color, zorder=5, edgecolors="white", linewidth=1.5)
        ax.text(x, y, name, fontsize=8, ha="center", va="center", color="white",
                fontweight="bold", zorder=6)
    # Draw edges.
    edges = [(0, 3), (1, 3), (3, 4), (2, 4), (4, 5), (5, 6)]
    for i, j in edges:
        x0, y0 = nodes[i][1], nodes[i][2]
        x1, y1 = nodes[j][1], nodes[j][2]
        ax.annotate("", xy=(x1 - 0.03, y1), xytext=(x0 + 0.03, y0),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=1.0))
    # Backward arrow.
    ax.annotate("backward()", xy=(0.1, 0.05), xytext=(0.9, 0.05),
                arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=2.0,
                                connectionstyle="arc3,rad=-0.3"),
                fontsize=9, color="#CC79A7", fontweight="bold",
                ha="center", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("NN From Scratch — Reverse-Mode Autograd Engine in NumPy",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
