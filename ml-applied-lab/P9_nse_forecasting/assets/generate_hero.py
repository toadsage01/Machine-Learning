"""
generate_hero
=============

Hero image for the P9 NSE Forecasting README.

Composes a 2×2 panel:
    - top-left   : walk-forward train/test windows visualization.
    - top-right  : stationarity comparison (Close vs returns) ADF p-values.
    - bottom-left: forecast-vs-actual for the best model.
    - bottom-right: model comparison bar chart (MAE).

Re-run after any model change to refresh ``assets/hero.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from shared import apply_style  # noqa: E402
apply_style()

from dataset import (  # noqa: E402
    load_equity_dataset, build_walk_forward_splits,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, build_forecaster, walk_forward_evaluate,
)


def main() -> None:
    # Use a moderate-size synthetic dataset for hero generation speed.
    ds = load_equity_dataset(
        symbol="^NSEI", n_days_synthetic=500, seed=42,
        lags=(1, 5, 10), rolling_windows=(5, 10, 20),
        run_stationarity=True,
    )

    # Train all three forecasters via walk-forward.
    reports = {}
    predictions = {}
    for name in ["naive", "lightgbm", "transformer_fallback"]:
        try:
            kind = CANDIDATE_MODELS[name]
            fc = build_forecaster(
                kind, n_estimators=100, n_epochs=10, seq_len=20,
            )
            report = walk_forward_evaluate(
                fc, ds.features_df,
                train_window=200, test_window=30, step=50,
            )
            reports[name] = report
            # Re-collect predictions for the forecast-vs-actual plot.
            preds = []
            for train_df, test_df in build_walk_forward_splits(
                ds.features_df, train_window=200, test_window=30, step=50,
            ):
                fold_fc = build_forecaster(kind, n_estimators=100, n_epochs=10, seq_len=20)
                fold_fc.fit(train_df, target_column=ds.target_column)
                y_pred, _, _ = fold_fc.predict(test_df)
                preds.extend(y_pred.tolist())
            predictions[name] = np.array(preds)
        except Exception as e:
            print(f"Skipping {name} in hero: {e}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # --- Top-left: walk-forward windows ---------------------------------
    ax = axes[0, 0]
    folds = list(build_walk_forward_splits(
        ds.features_df, train_window=200, test_window=30, step=50,
    ))
    colors_train = ["#0072B2", "#56B4E9", "#009E73"]
    colors_test = ["#D55E00", "#E69F00", "#CC79A7"]
    for i, (train_df, test_df) in enumerate(folds[:3]):
        ax.axvspan(train_df.index[0], train_df.index[-1],
                   alpha=0.3, color=colors_train[i % 3],
                   label=f"Train fold {i+1}")
        ax.axvspan(test_df.index[0], test_df.index[-1],
                   alpha=0.5, color=colors_test[i % 3],
                   label=f"Test fold {i+1}")
    ax.set_title(f"Walk-forward windows (train=200d, test=30d, step=50d, n_folds={len(folds)})",
                 loc="left", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Fold")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Top-right: stationarity comparison -----------------------------
    ax = axes[0, 1]
    if ds.stationarity:
        cols = list(ds.stationarity.keys())
        p_values = [ds.stationarity[c].p_value for c in cols]
        is_stat = [ds.stationarity[c].is_stationary for c in cols]
        colors = ["#009E73" if s else "#D55E00" for s in is_stat]
        bars = ax.bar(cols, p_values, color=colors)
        ax.axhline(0.05, color="#2b2b2b", linestyle="--", linewidth=0.8,
                   label="α = 0.05 significance")
        ax.set_ylabel("ADF p-value (log scale)")
        ax.set_yscale("log")
        ax.set_title("Augmented Dickey-Fuller stationarity check", loc="left", fontsize=11)
        for bar, p_val, stat in zip(bars, p_values, is_stat):
            label = "stationary" if stat else "non-stationary"
            ax.text(bar.get_x() + bar.get_width() / 2, p_val * 1.5,
                    f"p={p_val:.4f}\n{label}", ha="center", fontsize=8)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    # --- Bottom-left: forecast vs actual (best model) -------------------
    ax = axes[1, 0]
    if reports:
        best_name = min(reports.keys(), key=lambda k: reports[k].mae)
        preds = predictions.get(best_name, np.array([]))
        # Plot actual close + the best model's predictions.
        ax.plot(ds.features_df.index, ds.features_df["Close"].values,
                "-", color="#2b2b2b", linewidth=1.0, alpha=0.7, label="Actual Close")
        if len(preds) > 0:
            # Align predictions with the last len(preds) rows of features_df.
            n_pred = len(preds)
            if n_pred > len(ds.features_df):
                n_pred = len(ds.features_df)
                preds = preds[-n_pred:]
            x_idx = ds.features_df.index[-n_pred:]
            ax.plot(x_idx, preds, "-", color="#0072B2", linewidth=1.5,
                    alpha=0.85, label=f"{best_name} (MAE={reports[best_name].mae:.2f})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close price")
        ax.set_title(f"Forecast vs actual — best model: {best_name}", loc="left", fontsize=11)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.4)

    # --- Bottom-right: model MAE comparison bar chart -------------------
    ax = axes[1, 1]
    if reports:
        names = list(reports.keys())
        maes = [reports[n].mae for n in names]
        colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"][:len(names)]
        bars = ax.bar(names, maes, color=colors)
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, mae + max(maes) * 0.02,
                    f"{mae:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("MAE (lower is better)")
        ax.set_title("Walk-forward MAE comparison", loc="left", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("NSE Forecasting — Walk-Forward Benchmark (Naive / LightGBM / Transformer)",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
