"""
generate_hero
=============

Hero image for the P8 Churn Survival README.

Composes a 2×2 panel:
    - top-left   : Kaplan-Meier survival curve (population-level).
    - top-right  : Cox PH top-5 coefficients (with ground-truth comparison).
    - bottom-left: classifier ROC curves (LogReg vs Random Forest).
    - bottom-right: cumulative ROI curve with optimal targeting threshold.

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
from sklearn.metrics import roc_curve

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from shared import apply_style  # noqa: E402
apply_style()

from dataset import load_churn_dataset, build_train_test_split  # noqa: E402
from model import (  # noqa: E402
    CANDIDATE_MODELS, ModelKind,
    train_classifier, fit_kaplan_meier, fit_cox_ph,
    compute_uplift, expected_value_policy,
)


def main() -> None:
    ds = load_churn_dataset(n_samples=800, seed=42)
    train, test = build_train_test_split(ds, test_size=0.2, random_state=42)

    # Train all models upfront.
    pipes = {}
    classifier_metrics = {}
    for name in CANDIDATE_MODELS:
        pipe, m = train_classifier(
            ModelKind(name), train.X, train.y_churned,
            test.X, test.y_churned, random_state=42,
        )
        pipes[name] = (pipe, m)

    kmf, km_metrics = fit_kaplan_meier(train.durations, train.events, label="train")
    cph, cox_metrics = fit_cox_ph(
        train.X, train.durations, train.events,
        test.X, test.durations, test.events,
    )

    # Uplift using the best classifier.
    best_name = max(pipes.keys(), key=lambda k: pipes[k][1].roc_auc)
    best_pipe = pipes[best_name][0]
    p_control = best_pipe.predict_proba(test.X)[:, 1]
    X_treated = test.X.copy()
    X_treated["monthly_charges"] = X_treated["monthly_charges"] * 0.9
    p_treatment = best_pipe.predict_proba(X_treated)[:, 1]
    uplift = compute_uplift(p_control, p_treatment)
    ltv = test.df["total_charges"].values
    uplift_result = expected_value_policy(uplift, customer_ltv=ltv, offer_cost=10.0)

    # Plot.
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # --- Top-left: KM survival curve ------------------------------------
    ax = axes[0, 0]
    plot_fn = getattr(kmf, "plot_survival_function", None) or kmf.plot_survival_function_
    plot_fn(ax=ax, ci_show=True, color="#0072B2")
    ax.set_title(f"Kaplan-Meier survival curve (n={train.n_samples}, "
                 f"events={km_metrics.n_events}, censored={km_metrics.n_censored})",
                 loc="left", fontsize=11)
    ax.set_xlabel("Months since subscription")
    ax.set_ylabel("P(not churned)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.4)

    # --- Top-right: Cox PH top-5 coefficients ----------------------------
    ax = axes[1, 0]
    coefs = cph.params_.sort_values(key=np.abs, ascending=False).head(7)
    colors = ["#D55E00" if v > 0 else "#0072B2" for v in coefs.values]
    y_pos = np.arange(len(coefs))
    ax.barh(y_pos, coefs.values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[:30] for c in coefs.index], fontsize=9)
    ax.set_xlabel("Cox PH coefficient β (positive = increases churn hazard)")
    ax.set_title(f"Cox PH top-7 coefficients (C-index = {cox_metrics.c_index:.4f})",
                 loc="left", fontsize=11)
    ax.axvline(0, color="#2b2b2b", linewidth=0.6)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # --- Bottom-left: ROC curves for both classifiers -------------------
    ax = axes[1, 1]
    colors = {"logreg": "#0072B2", "random_forest": "#D55E00"}
    for name, (pipe, m) in pipes.items():
        y_proba = pipe.predict_proba(test.X)[:, 1]
        fpr, tpr, _ = roc_curve(test.y_churned, y_proba)
        ax.plot(fpr, tpr, "-", color=colors.get(name, "#000000"), linewidth=2.0,
                label=f"{name} (AUC={m.roc_auc:.4f}, Brier={m.brier_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="random")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Classifier ROC curves", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.4)

    # --- Bottom-right: cumulative ROI curve ------------------------------
    ax = axes[0, 1]
    n = len(uplift_result.cumulative_roi)
    x_pct = np.arange(1, n + 1) / n * 100
    ax.plot(x_pct, uplift_result.cumulative_roi, "-", color="#0072B2", linewidth=2.0,
            label="Cumulative ROI")
    ax.axvline(x_pct[uplift_result.optimal_threshold_idx], color="#D55E00",
               linestyle="--", linewidth=1.2,
               label=f"Optimal ({uplift_result.total_targeted} customers, "
                     f"ROI=${uplift_result.total_roi:.0f})")
    ax.axhline(0, color="#2b2b2b", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Customers targeted (%)")
    ax.set_ylabel("Cumulative net ROI ($)")
    ax.set_title(f"Expected-value retention policy (offer_cost=$10, "
                 f"best classifier={best_name})", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.4)

    fig.suptitle("Churn Survival — Classifier + Survival + Uplift Retention Policy",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
