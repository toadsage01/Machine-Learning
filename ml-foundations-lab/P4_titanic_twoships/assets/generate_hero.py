"""
generate_hero
=============

Generate the hero image for the P4 Titanic Two-Ships README.

Composes a 2×2 panel:
    - top-left   : model comparison bar chart on classic Titanic.
    - top-right  : calibration curves for all 3 boosters on classic.
    - bottom-left: fairness selection-rate bar chart (sex × pclass).
    - bottom-right: ROC curves for all 3 boosters on classic.

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, brier_score_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from shared import apply_style  # noqa: E402
apply_style()

from dataset import load_classic_titanic  # noqa: E402
from model import (  # noqa: E402
    CANDIDATE_MODELS, ModelKind, CalibrationKind,
    train_model, calibrate, evaluate_calibration, compute_fairness,
)


def main() -> None:
    ds = load_classic_titanic()
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=0.2, stratify=ds.y, random_state=42)

    pipes = {}
    metrics = {}
    cal_results = {}
    cal_pipes = {}
    for name in CANDIDATE_MODELS:
        pipe, m = train_model(ModelKind(name), X_tr, y_tr, X_te, y_te, cv_folds=5)
        pipes[name] = pipe
        metrics[name] = m
        cal_pipe = calibrate(pipe, X_tr, y_tr, CalibrationKind.ISOTONIC, cv_folds=5)
        cal_pipes[name] = cal_pipe
        cal_results[name] = evaluate_calibration(
            cal_pipe, X_te, y_te, name, "isotonic", n_bins=10)

    # Use the best model for the fairness plot.
    best_name = max(metrics.keys(), key=lambda k: metrics[k].accuracy)
    fairness = compute_fairness(pipes[best_name], X_te, y_te, best_name)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # --- Top-left: model comparison bar chart ---------------------------
    ax = axes[0, 0]
    names = list(metrics.keys())
    accs = [metrics[n].accuracy for n in names]
    aucs = [metrics[n].roc_auc for n in names]
    briers = [metrics[n].brier_score for n in names]
    cv_means = [metrics[n].cv_accuracy_mean for n in names]
    cv_stds = [metrics[n].cv_accuracy_std for n in names]
    x = np.arange(len(names))
    width = 0.22
    ax.bar(x - width, accs, width, label="Test accuracy", color="#0072B2")
    ax.bar(x, aucs, width, label="ROC-AUC", color="#009E73")
    ax.bar(x + width, cv_means, width, yerr=cv_stds, label="CV accuracy (5-fold)",
           color="#E69F00", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Model comparison — classic Titanic (n={ds.n_samples})", loc="left")
    ax.legend(loc="lower right", fontsize=9)

    # --- Top-right: calibration curves ----------------------------------
    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="perfect")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for i, (name, cr) in enumerate(cal_results.items()):
        color = colors[i % len(colors)]
        ax.plot(cr.mean_predicted_value, cr.fraction_of_positives,
                "o-", color=color, linewidth=1.8, markersize=7,
                label=f"{name} (Brier={cr.brier_score:.4f})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (in bin)")
    ax.set_title("Calibration curves — isotonic (classic Titanic)", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.4)

    # --- Bottom-left: fairness selection-rate bar chart -----------------
    ax = axes[1, 0]
    # Show only sex × pclass slices for legibility.
    slices_to_plot = [s for s in fairness.slices if s.feature in ("sex", "pclass")]
    slice_labels = [f"{s.feature}={s.value}" for s in slices_to_plot]
    selection_rates = [s.selection_rate for s in slices_to_plot]
    base_rates = [s.base_rate for s in slices_to_plot]
    x = np.arange(len(slice_labels))
    width = 0.4
    ax.bar(x - width/2, selection_rates, width, label="Selection rate", color="#CC79A7")
    ax.bar(x + width/2, base_rates, width, label="Base rate (true + rate)", color="#56B4E9")
    ax.set_xticks(x)
    ax.set_xticklabels(slice_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_title(f"Fairness audit — {best_name} (classic)", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)

    # --- Bottom-right: ROC curves --------------------------------------
    ax = axes[1, 1]
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="random")
    for i, name in enumerate(metrics.keys()):
        proba = pipes[name].predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        ax.plot(fpr, tpr, "-", color=colors[i % len(colors)], linewidth=2.0,
                label=f"{name} (AUC={metrics[name].roc_auc:.4f})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — classic Titanic (uncalibrated)", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.4)

    fig.suptitle("Titanic Two-Ships — Boosting + Calibration + Fairness",
                 fontsize=16, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
