"""
generate_hero
=============

Generate the hero image for the P3 minigrad README.

Composes a 2×2 panel:
    - top-left   : side-by-side contour trajectories on Rosenbrock.
    - top-right  : loss-vs-iteration curves on the ill-conditioned quadratic.
    - bottom-left: 3-D loss landscape of Beale.
    - bottom-right: trajectory grid (4 surfaces × 6 optimizers, small multiples).

Re-run after any optimizer change to refresh ``assets/hero.png``.
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

from shared import apply_style  # noqa: E402
apply_style()

from dataset import ALL_SURFACES, LossSurface  # noqa: E402
from model import run_optimization, OptimizationResult  # noqa: E402
from visualize import (  # noqa: E402
    plot_contour_trajectory, plot_loss_curves, plot_3d_loss_landscape,
    plot_optimizer_grid, plot_side_by_side_comparison,
)


def main() -> None:
    # Run all 6 optimizers on all 4 canonical 2-D surfaces.
    surfaces_2d = [
        ALL_SURFACES["rosenbrock"],
        ALL_SURFACES["rastrigin"],
        ALL_SURFACES["ill_conditioned_quadratic"],
        ALL_SURFACES["beale"],
    ]
    results_per_surface = {}
    for surface in surfaces_2d:
        rs = [run_optimization(name, surface, max_iters=2000) for name in
              ["batch_gd", "momentum", "adagrad", "rmsprop", "adam"]]
        results_per_surface[surface.name] = rs

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)

    # --- Top-left: Rosenbrock contour trajectories --------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    surface = ALL_SURFACES["rosenbrock"]
    (x_lo, x_hi), (y_lo, y_hi) = surface.bounds
    n_grid = 200
    xs = np.linspace(x_lo, x_hi, n_grid)
    ys = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = surface.f(np.array([X[i, j], Y[i, j]]))
    levels = np.logspace(np.log10(max(np.nanmin(Z), 1e-8)), np.log10(np.nanmax(Z)), 25)
    ax1.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.85)
    ax1.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.3, alpha=0.3)

    colors = ["#D55E00", "#CC79A7", "#009E73", "#56B4E9", "#E69F00", "#0072B2"]
    for i, r in enumerate(results_per_surface[surface.name]):
        traj = r.history_x
        color = colors[i % len(colors)]
        ax1.plot(traj[:, 0], traj[:, 1], "-", color=color, linewidth=1.4, alpha=0.9, label=r.name)
        ax1.scatter(traj[0, 0], traj[0, 1], s=50, facecolors="none", edgecolors=color, linewidths=1.2, zorder=5)
        ax1.scatter(traj[-1, 0], traj[-1, 1], s=30, color=color, marker="s", zorder=5)
    ax1.scatter(surface.minimum_x[0], surface.minimum_x[1], s=200, marker="*",
                color="white", edgecolors="black", linewidths=1.4, zorder=6)
    ax1.set_title("Rosenbrock — optimizer trajectories", loc="left", fontsize=12)
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylim(y_lo, y_hi)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # --- Top-right: loss curves on ill-conditioned quadratic -----------------
    ax2 = fig.add_subplot(2, 2, 2)
    surface = ALL_SURFACES["ill_conditioned_quadratic"]
    with np.errstate(over="ignore", invalid="ignore"):
        for i, r in enumerate(results_per_surface[surface.name]):
            f = r.history_f.copy()
            f = np.where(np.isfinite(f) & (f > 0), f, np.nan)
            color = colors[i % len(colors)]
            ax2.plot(np.arange(len(f)), f, "-", linewidth=1.6, color=color,
                     label=f"{r.name} (f={r.f_final:.1e})")
        ax2.set_yscale("log")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss f(x)  (log scale)")
    ax2.set_title("Ill-conditioned quadratic (κ=2500) — convergence", loc="left", fontsize=12)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax2.grid(True, which="both", linewidth=0.3, alpha=0.5)

    # --- Bottom-left: 3-D loss landscape of Beale ----------------------------
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    surface = ALL_SURFACES["beale"]
    (x_lo, x_hi), (y_lo, y_hi) = surface.bounds
    n_grid = 60
    xs = np.linspace(x_lo, x_hi, n_grid)
    ys = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = surface.f(np.array([X[i, j], Y[i, j]]))
    Z_log = np.log10(np.maximum(Z, 1e-8))
    surf = ax3.plot_surface(X, Y, Z_log, cmap="viridis", edgecolor="none", alpha=0.92,
                            rstride=1, cstride=1, antialiased=True)
    ax3.set_xlabel("x₀")
    ax3.set_ylabel("x₁")
    ax3.set_zlabel("log₁₀(f)")
    ax3.set_title("Beale — 3-D loss landscape", loc="left", fontsize=12)
    ax3.scatter([surface.minimum_x[0]], [surface.minimum_x[1]],
                [np.log10(max(surface.minimum_f, 1e-8))],
                s=80, color="red", marker="*", label="global min")
    fig.colorbar(surf, ax=ax3, shrink=0.55, pad=0.1, label="log₁₀(f)")

    # --- Bottom-right: summary table of final losses ------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_axis_off()
    # Build a small comparison table.
    rows = []
    surfaces_in_table = [
        ALL_SURFACES["rosenbrock"].name,
        ALL_SURFACES["rastrigin"].name,
        ALL_SURFACES["ill_conditioned_quadratic"].name,
        ALL_SURFACES["beale"].name,
    ]
    optimizers_in_table = ["batch_gd", "momentum", "adagrad", "rmsprop", "adam"]
    for s_name in surfaces_in_table:
        row = [s_name[:18]]
        for opt_name in optimizers_in_table:
            r = next((r for r in results_per_surface[s_name] if r.name == opt_name), None)
            if r is None:
                row.append("—")
            elif not np.isfinite(r.f_final):
                row.append("diverged")
            else:
                row.append(f"{r.f_final:.1e}")
        rows.append(row)
    # Truncate surface names for display.
    rows = [[r[0].replace("_κ2500", "")] + r[1:] for r in rows]
    headers = ["surface"] + optimizers_in_table
    table = ax4.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.15, 1.6)
    # Colour the header row.
    for j, _ in enumerate(headers):
        cell = table[0, j]
        cell.set_facecolor("#0072B2")
        cell.set_text_props(color="white", fontweight="bold")
    ax4.set_title("Final loss per (optimizer, surface)", loc="left", fontsize=12)

    fig.suptitle("MiniGrad — From-Scratch NumPy Optimization Benchmarks",
                 fontsize=16, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
