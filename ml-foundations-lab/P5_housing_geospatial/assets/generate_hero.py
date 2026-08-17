"""
generate_hero
=============

Hero image for the P5 Housing Geospatial README.

Composes a 2×2 panel:
    - top-left   : spatial price heatmap (log-scale lat/lon scatter).
    - top-right  : quantile prediction intervals over 60 test instances.
    - bottom-left: coverage calibration bar chart (LightGBM vs GBR).
    - bottom-right: OSM proximity feature (metro station) scatter.

Re-run after any model or data change to refresh ``assets/hero.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from shared import apply_style  # noqa: E402
apply_style()

from dataset import Metro, load_housing, METRO_BBOXES, PROXIMITY_POIS  # noqa: E402
from model import (  # noqa: E402
    CANDIDATE_MODELS, QuantileKind, DEFAULT_QUANTILES,
    train_quantile_model, evaluate_quantile_model,
)


def main() -> None:
    ds = load_housing(Metro.MUMBAI, n_samples=1200, seed=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=0.2, random_state=42)

    # Train both quantile models.
    models = {}
    metrics = {}
    for name in CANDIDATE_MODELS:
        qm = train_quantile_model(QuantileKind(name), X_tr, y_tr, random_state=42)
        models[name] = qm
        metrics[name] = evaluate_quantile_model(qm, X_te, y_te)

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)

    # --- Top-left: spatial price heatmap ---------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    cmap = LinearSegmentedColormap.from_list("price", ["#0072B2", "#009E73", "#E69F00", "#D55E00"])
    lat = ds.df["latitude"].values
    lon = ds.df["longitude"].values
    price = ds.df["price_lakh"].values
    sc = ax1.scatter(lon, lat, c=price, cmap=cmap, s=12, alpha=0.75,
                     norm=LogNorm(vmin=max(price.min(), 1), vmax=price.max()))
    fig.colorbar(sc, ax=ax1, label="Price (lakh INR, log)", fraction=0.046, pad=0.04)
    bbox = METRO_BBOXES["mumbai"]
    ax1.set_xlim(bbox[2], bbox[3])
    ax1.set_ylim(bbox[0], bbox[1])
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("Mumbai — spatial price heatmap", loc="left", fontsize=12)
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)

    # --- Top-right: quantile prediction intervals (use GBR) ---------------
    ax2 = fig.add_subplot(2, 2, 2)
    qm = models["gradient_boosting"]
    preds = qm.predict(X_te)
    y_true = np.asarray(y_te, dtype=float)
    # Subsample 60 rows sorted by p50.
    idx = np.argsort(preds["p50"].values)
    n_show = 60
    idx = idx[np.linspace(0, len(idx) - 1, n_show).astype(int)]
    p10 = preds["p10"].values[idx]
    p50 = preds["p50"].values[idx]
    p90 = preds["p90"].values[idx]
    y_show = y_true[idx]
    x = np.arange(n_show)
    ax2.fill_between(x, p10, p90, color="#0072B2", alpha=0.18, label="80% interval [p10, p90]")
    ax2.plot(x, p50, "-", color="#0072B2", linewidth=1.8, label="Median prediction (p50)")
    inside = (y_show >= p10) & (y_show <= p90)
    ax2.scatter(x[inside], y_show[inside], s=24, color="#009E73", alpha=0.8,
                label=f"Actual (covered: {inside.sum()}/{len(inside)})")
    ax2.scatter(x[~inside], y_show[~inside], s=24, color="#D55E00", alpha=0.9, marker="x",
                label=f"Actual (out of band: {(~inside).sum()})")
    ax2.set_xlabel("Test instance (sorted by predicted median price)")
    ax2.set_ylabel("Price (lakh INR)")
    ax2.set_title("Quantile prediction intervals — GradientBoosting", loc="left", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Bottom-left: coverage calibration bar chart ---------------------
    ax3 = fig.add_subplot(2, 2, 3)
    names = list(models.keys())
    coverages = [metrics[n].coverage_p10_p90 for n in names]
    mean_widths = [metrics[n].mean_interval_width for n in names]
    x = np.arange(len(names))
    width = 0.4
    bars1 = ax3.bar(x - width/2, coverages, width, color=["#0072B2", "#D55E00"], label="Coverage")
    ax3.axhline(0.80, color="#2b2b2b", linestyle="--", linewidth=1.0, label="Target (80%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_ylabel("Coverage of [p10, p90]")
    ax3.set_title("Coverage calibration — empirical vs. nominal 80%", loc="left", fontsize=12)
    for bar, c in zip(bars1, coverages):
        ax3.text(bar.get_x() + bar.get_width() / 2, c + 0.02,
                 f"{c:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, axis="y", alpha=0.3)

    # --- Bottom-right: pinball loss per quantile, grouped bar ------------
    ax4 = fig.add_subplot(2, 2, 4)
    qs = ["q=0.10", "q=0.50", "q=0.90"]
    lgbm_pbs = metrics["lightgbm"].pinball_per_quantile
    gbr_pbs = metrics["gradient_boosting"].pinball_per_quantile
    x = np.arange(len(qs))
    width = 0.4
    ax4.bar(x - width/2, lgbm_pbs, width, label="LightGBM", color="#0072B2")
    ax4.bar(x + width/2, gbr_pbs, width, label="GradientBoosting", color="#D55E00")
    ax4.set_xticks(x)
    ax4.set_xticklabels(qs)
    ax4.set_ylabel("Pinball loss (lakh INR)")
    ax4.set_title("Pinball loss per quantile — lower is better", loc="left", fontsize=12)
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Housing Geospatial — Quantile Regression with OSM Proximity Features",
                 fontsize=16, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
