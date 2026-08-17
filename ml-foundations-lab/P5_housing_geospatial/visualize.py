"""
visualize
=========

Geospatial + quantile-interval visualizations for the P5 housing benchmark.

Public surface
--------------
- ``plot_spatial_price_heatmap``   : lat/lon scatter coloured by price.
- ``plot_proximity_features``       : small-multiples of the 7 OSM proximity columns.
- ``plot_quantile_intervals``       : per-row p10/p50/p90 bands over the test set.
- ``plot_calibration_curve``         : observed vs. nominal quantile coverage.

All figures use the project-wide matplotlib style via ``shared.apply_style()``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from shared import apply_style
    apply_style()
except Exception:  # pragma: no cover
    pass

from dataset import HousingDataset, METRO_BBOXES, PROXIMITY_POIS  # noqa: E402
from model import QuantileModel, QuantileMetrics  # noqa: E402


# ---------------------------------------------------------------------------
# Spatial price heatmap
# ---------------------------------------------------------------------------
def plot_spatial_price_heatmap(
    ds: HousingDataset,
    output_path: Optional[Path | str] = None,
    title: Optional[str] = None,
    show_locality_labels: bool = False,
) -> Optional[plt.Figure]:
    """Scatter-plot every property's lat/lon, coloured by price.

    Uses a log-scale colourmap because Mumbai prices span 2 orders of
    magnitude (5 lakh → 1000+ lakh). The heatmap is the canonical
    "isochrone of wealth" view of Indian metro housing.
    """
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    # Custom sequential colormap: blue → orange → red (cool to hot).
    cmap = LinearSegmentedColormap.from_list("price", ["#0072B2", "#009E73", "#E69F00", "#D55E00"])

    lat = ds.df["latitude"].values
    lon = ds.df["longitude"].values
    price = ds.df["price_lakh"].values
    # Log-scale the colour to handle the long tail.
    sc = ax.scatter(lon, lat, c=price, cmap=cmap, s=10, alpha=0.7,
                    norm=LogNorm(vmin=max(price.min(), 1), vmax=price.max()))
    fig.colorbar(sc, ax=ax, label="Price (lakh INR, log scale)")

    if show_locality_labels:
        # Label each unique locality at its centroid.
        for loc, grp in ds.df.groupby("locality"):
            ax.text(grp["longitude"].mean(), grp["latitude"].mean(), loc,
                    fontsize=7, alpha=0.85, color="#2b2b2b",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

    # Set axis limits to the metro's bbox.
    bbox = METRO_BBOXES[ds.metro.value]
    lat_min, lat_max, lon_min, lon_max = bbox
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"{ds.metro.value.title()} — spatial price heatmap", loc="left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Proximity features small-multiples
# ---------------------------------------------------------------------------
def plot_proximity_features(
    ds: HousingDataset,
    output_path: Optional[Path | str] = None,
    max_per_panel: int = 1500,
) -> Optional[plt.Figure]:
    """One scatter-plot per OSM proximity feature, coloured by distance.

    Visualizes which neighbourhoods are well-served by which amenities.
    """
    proximity_cols = [c for c, _ in PROXIMITY_POIS]
    n_panels = len(proximity_cols)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    df = ds.df
    if len(df) > max_per_panel:
        df = df.sample(max_per_panel, random_state=42)

    for i, col in enumerate(proximity_cols):
        ax = axes[i]
        sc = ax.scatter(df["longitude"], df["latitude"],
                        c=df[col], cmap="viridis_r", s=8, alpha=0.7)
        fig.colorbar(sc, ax=ax, label=f"{col} (km)", fraction=0.046, pad=0.04)
        ax.set_title(col.replace("dist_", "").replace("_km", ""), loc="left", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots.
    for j in range(n_panels, len(axes)):
        axes[j].set_axis_off()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Quantile prediction intervals over the test set
# ---------------------------------------------------------------------------
def plot_quantile_intervals(
    model: QuantileModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Optional[Path | str] = None,
    n_samples_to_plot: int = 60,
    sort_by_prediction: bool = True,
) -> Optional[plt.Figure]:
    """Plot p10/p50/p90 bands over the first ``n_samples_to_plot`` test rows.

    If ``sort_by_prediction``, rows are sorted by p50 (so the bands form
    a smooth fan shape); otherwise they're plotted in dataset order.
    """
    preds = model.predict(X_test)
    y_true = np.asarray(y_test, dtype=float)

    # Subsample for legibility.
    if len(preds) > n_samples_to_plot:
        idx = np.linspace(0, len(preds) - 1, n_samples_to_plot).astype(int)
        preds = preds.iloc[idx].reset_index(drop=True)
        y_true = y_true[idx]

    if sort_by_prediction:
        order = np.argsort(preds["p50"].values)
        preds = preds.iloc[order].reset_index(drop=True)
        y_true = y_true[order]

    p10 = preds["p10"].values
    p50 = preds["p50"].values
    p90 = preds["p90"].values
    x = np.arange(len(p50))

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    # 80% interval band (p10 to p90).
    ax.fill_between(x, p10, p90, color="#0072B2", alpha=0.18, label="80% interval [p10, p90]")
    # Median line.
    ax.plot(x, p50, "-", color="#0072B2", linewidth=1.8, label="Median prediction (p50)")
    # Actuals — colour-coded by whether they fall inside the interval.
    inside = (y_true >= p10) & (y_true <= p90)
    ax.scatter(x[inside], y_true[inside], s=24, color="#009E73", alpha=0.8,
               label=f"Actual (covered: {inside.sum()}/{len(inside)})")
    ax.scatter(x[~inside], y_true[~inside], s=24, color="#D55E00", alpha=0.9, marker="x",
               label=f"Actual (out of band: {(~inside).sum()})")
    ax.set_xlabel("Test instance (sorted by predicted median price)")
    ax.set_ylabel("Price (lakh INR)")
    ax.set_title("Quantile prediction intervals — p10 / p50 / p90 over test set", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Calibration curve (coverage vs. nominal quantile)
# ---------------------------------------------------------------------------
def plot_calibration_curve(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Optional[Path | str] = None,
) -> Optional[plt.Figure]:
    """Per-model coverage-vs-nominal-quantile bar chart.

    For each model, we report the empirical coverage of the [p10, p90]
    interval — the target is 80%. A model with coverage < 0.80 is
    over-confident (intervals too narrow); > 0.85 is under-confident.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    names = list(models.keys())
    coverages = []
    for name, m in models.items():
        preds = m.predict(X_test)
        y_true = np.asarray(y_test, dtype=float)
        inside = (y_true >= preds["p10"].values) & (y_true <= preds["p90"].values)
        coverages.append(float(np.mean(inside)))

    x = np.arange(len(names))
    bars = ax.bar(x, coverages, color=["#0072B2", "#D55E00", "#009E73"][:len(names)])
    ax.axhline(0.80, color="#2b2b2b", linestyle="--", linewidth=1.0, label="Target (80%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Empirical coverage of [p10, p90] interval")
    ax.set_title("Quantile calibration — empirical vs. nominal coverage", loc="left")
    # Annotate bars with the coverage value.
    for bar, c in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 0.02,
                f"{c:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return None
    return fig


__all__ = [
    "plot_spatial_price_heatmap",
    "plot_proximity_features",
    "plot_quantile_intervals",
    "plot_calibration_curve",
]
