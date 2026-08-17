"""
visualize
=========

Loss-landscape + trajectory visualizations for the P3 minigrad benchmark.

Public surface
--------------
- ``plot_contour_trajectory``    : 2-D contour + optimizer paths for one surface.
- ``plot_loss_curves``           : log-scale loss-vs-iteration lines for one surface.
- ``plot_side_by_side_comparison`` : 2×N grid of contour plots (one per surface).
- ``plot_3d_loss_landscape``    : 3-D surface plot of a 2-D loss surface.
- ``plot_optimizer_grid``       : per-surface, per-optimizer 2-D trajectory grid.

All figures use the project-wide matplotlib style via ``shared.apply_style()``,
so they share the same color cycle, font stack, and spine treatment as
every other chart in the monorepo.

Design notes
------------
1. **Log-scale loss axis** — optimizer trajectories span many orders of
   magnitude (1e0 → 1e-13 on well-conditioned problems). Linear plots
   hide the late-stage convergence behaviour, so we always use ``ax.set_yscale('log')``.

2. **Contourf with log-spaced levels** — for the contour plots we use
   ``np.logspace`` for the levels so the banana valley of Rosenbrock and
   the flat plateau of Beale are both legible. Linear spacing would either
   saturate the centre or wash out the periphery.

3. **Subsampled trajectories** — for Adam/RMSProp runs of 5000 iters we
   plot every Nth point (computed to keep ≤200 markers per optimizer) so
   the line stays visible without becoming an opaque blob.

4. **3-D surface plot** — uses ``mpl_toolkits.mplot3d`` with a viridis
   colormap and a vertical log-scale so the narrow valley is visible
   alongside the plateau.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared import apply_style  # noqa: E402
apply_style()

from dataset import LossSurface  # noqa: E402
from model import OptimizationResult  # noqa: E402


# ---------------------------------------------------------------------------
# 2-D contour + optimizer trajectories
# ---------------------------------------------------------------------------
def plot_contour_trajectory(
    surface: LossSurface,
    results: Sequence[OptimizationResult],
    output_path: Optional[Path | str] = None,
    title: Optional[str] = None,
    max_markers: int = 200,
    show_minima: bool = True,
) -> Optional[plt.Figure]:
    """Plot a 2-D contour of ``surface`` with optimizer trajectories overlaid.

    Parameters
    ----------
    surface : LossSurface
        2-D loss surface (uses ``surface.bounds`` for the contour extent).
    results : sequence of OptimizationResult
        Trajectories to overlay. Each ``result.history_x`` is plotted as
        a coloured line + markers.
    output_path : str or Path, optional
        If provided, save the figure here. Otherwise the figure is returned
        without saving.
    title : str, optional
        Override the default title.
    max_markers : int
        Maximum number of markers to draw per trajectory (subsampled
        uniformly). Lines are always full-resolution.
    show_minima : bool
        If True and ``surface.minimum_x`` is not None, draw a star marker.
    """
    (x_lo, x_hi), (y_lo, y_hi) = surface.bounds
    n_grid = 200
    xs = np.linspace(x_lo, x_hi, n_grid)
    ys = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = surface.f(np.array([X[i, j], Y[i, j]]))

    # Log-spaced contour levels (clipped to a small floor so log doesn't blow up).
    z_min = max(np.nanmin(Z), 1e-8)
    z_max = np.nanmax(Z)
    levels = np.logspace(np.log10(z_min), np.log10(z_max), 25)

    fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.85)
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.4, alpha=0.4)
    fig.colorbar(contour, ax=ax, label="f(x)", fraction=0.046, pad=0.04)

    # Optimizer colour cycle — matches the Okabe-Ito-inspired project style.
    colors = ["#D55E00", "#CC79A7", "#009E73", "#56B4E9", "#E69F00", "#0072B2", "#000000"]

    for i, result in enumerate(results):
        traj = result.history_x
        n = traj.shape[0]
        # Subsample markers, but always plot the full line.
        step = max(1, n // max_markers)
        color = colors[i % len(colors)]
        ax.plot(traj[:, 0], traj[:, 1], "-", color=color, linewidth=1.4, alpha=0.85, label=result.name)
        ax.scatter(traj[::step, 0], traj[::step, 1], s=12, color=color, alpha=0.6, edgecolors="none")
        # Start marker (open circle) and end marker (filled square).
        ax.scatter(traj[0, 0], traj[0, 1], s=80, facecolors="none", edgecolors=color, linewidths=1.5, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], s=50, color=color, marker="s", zorder=5)

    if show_minima and surface.minimum_x is not None:
        ax.scatter(surface.minimum_x[0], surface.minimum_x[1], s=200, marker="*",
                   color="white", edgecolors="black", linewidths=1.4, zorder=6, label="global min")

    ax.set_xlabel(surface.name.split("_")[0] + " x₀")
    ax.set_ylabel(surface.name.split("_")[0] + " x₁")
    ax.set_title(title or f"{surface.name} — optimizer trajectories", loc="left")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Loss curves (log-scale)
# ---------------------------------------------------------------------------
def plot_loss_curves(
    surface: LossSurface,
    results: Sequence[OptimizationResult],
    output_path: Optional[Path | str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot ``f(x_t)`` vs ``t`` for every optimizer on a log-y axis.

    Each optimizer's loss trajectory is plotted from iteration 0 (initial
    loss) to iteration ``n_iters``. Curves are clipped to ``max(1e-12, f_final)``
    to avoid log(0) artifacts.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = ["#D55E00", "#CC79A7", "#009E73", "#56B4E9", "#E69F00", "#0072B2", "#000000"]

    # Suppress matplotlib's "overflow encountered in power" warning when
    # log-scaling NaN/inf values (which are produced by diverged runs).
    # The NaNs are masked out below so they don't actually render.
    with np.errstate(over="ignore", invalid="ignore"):
        for i, result in enumerate(results):
            f = result.history_f.copy()
            # Replace non-finite / non-positive values with NaN so matplotlib
            # breaks the line instead of attempting log(0) or log(negative).
            f = np.where(np.isfinite(f) & (f > 0), f, np.nan)
            ax.plot(np.arange(len(f)), f, "-", linewidth=1.6, color=colors[i % len(colors)],
                    label=f"{result.name} (final={result.f_final:.2e})")

        if surface.minimum_f is not None and surface.minimum_f > 0:
            ax.axhline(surface.minimum_f, color="#2b2b2b", linestyle=":", linewidth=1.0,
                       label=f"global min ({surface.minimum_f:.2e})")

        ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss f(x)")
    ax.set_title(title or f"{surface.name} — convergence curves", loc="left")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Side-by-side grid: one contour per surface, all optimizers overlaid
# ---------------------------------------------------------------------------
def plot_side_by_side_comparison(
    surfaces: Sequence[LossSurface],
    results_per_surface: Dict[str, Sequence[OptimizationResult]],
    output_path: Path | str,
    n_cols: int = 2,
) -> None:
    """Render a grid of contour plots, one per surface.

    Parameters
    ----------
    surfaces : sequence of LossSurface
        Surfaces to plot (in display order).
    results_per_surface : dict[str, sequence[OptimizationResult]]
        ``{surface.name: [OptimizationResult, ...]}``. Missing entries
        render an empty contour.
    output_path : Path
        Destination PNG.
    n_cols : int
        Number of columns in the grid.
    """
    n = len(surfaces)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5.5 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    colors = ["#D55E00", "#CC79A7", "#009E73", "#56B4E9", "#E69F00", "#0072B2", "#000000"]

    for i, surface in enumerate(surfaces):
        ax = axes[i]
        (x_lo, x_hi), (y_lo, y_hi) = surface.bounds
        n_grid = 120
        xs = np.linspace(x_lo, x_hi, n_grid)
        ys = np.linspace(y_lo, y_hi, n_grid)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)
        for r in range(n_grid):
            for c in range(n_grid):
                Z[r, c] = surface.f(np.array([X[r, c], Y[r, c]]))
        z_min = max(np.nanmin(Z), 1e-8)
        z_max = np.nanmax(Z)
        levels = np.logspace(np.log10(z_min), np.log10(z_max), 20)
        ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.85)

        results = results_per_surface.get(surface.name, [])
        for j, result in enumerate(results):
            traj = result.history_x
            color = colors[j % len(colors)]
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color, linewidth=1.2, alpha=0.9, label=result.name)
            ax.scatter(traj[0, 0], traj[0, 1], s=40, facecolors="none", edgecolors=color, linewidths=1.2, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], s=30, color=color, marker="s", zorder=5)
        if surface.minimum_x is not None:
            ax.scatter(surface.minimum_x[0], surface.minimum_x[1], s=120, marker="*",
                       color="white", edgecolors="black", linewidths=1.0, zorder=6)
        ax.set_title(surface.name, loc="left", fontsize=11)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    # Hide unused subplots.
    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3-D loss landscape
# ---------------------------------------------------------------------------
def plot_3d_loss_landscape(
    surface: LossSurface,
    output_path: Optional[Path | str] = None,
    title: Optional[str] = None,
    n_grid: int = 80,
) -> Optional[plt.Figure]:
    """Render a 3-D surface plot of a 2-D loss surface.

    Uses a vertical log-scale (``np.log10(f + 1e-8)``) so the narrow
    valley of Rosenbrock and the plateau of Beale are both legible.
    """
    (x_lo, x_hi), (y_lo, y_hi) = surface.bounds
    xs = np.linspace(x_lo, x_hi, n_grid)
    ys = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for r in range(n_grid):
        for c in range(n_grid):
            Z[r, c] = surface.f(np.array([X[r, c], Y[r, c]]))
    # Log-transform for visualization (add small floor so log(0) is finite).
    Z_log = np.log10(np.maximum(Z, 1e-8))

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z_log, cmap="viridis", edgecolor="none", alpha=0.92,
                           rstride=1, cstride=1, antialiased=True)
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_zlabel("log₁₀(f)")
    ax.set_title(title or f"{surface.name} — 3-D loss landscape", loc="left")
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1, label="log₁₀(f)")

    if surface.minimum_x is not None:
        z_at_min = np.log10(max(surface.minimum_f, 1e-8))
        ax.scatter([surface.minimum_x[0]], [surface.minimum_x[1]], [z_at_min],
                   s=80, color="red", marker="*", label="global min")
        ax.legend(loc="upper right", fontsize=9)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Per-surface, per-optimizer grid
# ---------------------------------------------------------------------------
def plot_optimizer_grid(
    surfaces: Sequence[LossSurface],
    results_per_surface: Dict[str, Sequence[OptimizationResult]],
    output_path: Path | str,
) -> None:
    """Render an N_surfaces × N_optimizers grid of small trajectory plots.

    Useful for spotting which optimizer wins on which surface.
    """
    n_surfaces = len(surfaces)
    # Determine n_optimizers from the first non-empty results list.
    n_optimizers = max((len(rs) for rs in results_per_surface.values()), default=0)
    if n_optimizers == 0:
        return

    fig, axes = plt.subplots(n_surfaces, n_optimizers,
                             figsize=(3.5 * n_optimizers, 3.0 * n_surfaces),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    colors = ["#D55E00", "#CC79A7", "#009E73", "#56B4E9", "#E69F00", "#0072B2", "#000000"]

    for i, surface in enumerate(surfaces):
        results = results_per_surface.get(surface.name, [])
        (x_lo, x_hi), (y_lo, y_hi) = surface.bounds
        n_grid = 80
        xs = np.linspace(x_lo, x_hi, n_grid)
        ys = np.linspace(y_lo, y_hi, n_grid)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)
        for r in range(n_grid):
            for c in range(n_grid):
                Z[r, c] = surface.f(np.array([X[r, c], Y[r, c]]))
        z_min = max(np.nanmin(Z), 1e-8)
        z_max = np.nanmax(Z)
        levels = np.logspace(np.log10(z_min), np.log10(z_max), 15)

        for j in range(n_optimizers):
            ax = axes[i, j] if n_surfaces > 1 else axes[j]
            ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.85)
            if j < len(results):
                result = results[j]
                traj = result.history_x
                ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[j % len(colors)],
                        linewidth=1.2, alpha=0.9)
                ax.scatter(traj[0, 0], traj[0, 1], s=25, facecolors="none",
                           edgecolors=colors[j % len(colors)], linewidths=1.0)
                ax.scatter(traj[-1, 0], traj[-1, 1], s=18, color=colors[j % len(colors)],
                           marker="s")
            if surface.minimum_x is not None:
                ax.scatter(surface.minimum_x[0], surface.minimum_x[1], s=60, marker="*",
                           color="white", edgecolors="black", linewidths=0.8)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(results[j].name if j < len(results) else f"opt{j}", fontsize=10)
            if j == 0:
                ax.set_ylabel(surface.name, fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


__all__ = [
    "plot_contour_trajectory",
    "plot_loss_curves",
    "plot_side_by_side_comparison",
    "plot_3d_loss_landscape",
    "plot_optimizer_grid",
]
