"""
generate_hero
=============

Generate the hero image for the P2 Iris Production README.

Composes a 2×2 panel showing:
    - top-left   : candidate model comparison bar chart (acc / f1 / cv_acc).
    - top-right  : confusion matrix of the best model.
    - bottom-left: SHAP feature-importance summary for the best tree-based model.
    - bottom-right: ROC-style per-class probability strip for 3 sample rows.

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

from shared import apply_style  # noqa: E402
apply_style()

from dataset import load_iris_split, FEATURE_NAMES, TARGET_NAMES  # noqa: E402
from model import (  # noqa: E402
    CANDIDATE_MODELS, build_pipeline, evaluate_pipeline,
    explain_with_shap, HAVE_SHAP,
)


def main() -> None:
    ds = load_iris_split(random_state=42)

    # Train all 4 candidates.
    results = {}
    for kind_str, kind in CANDIDATE_MODELS.items():
        pipe = build_pipeline(kind)
        m = evaluate_pipeline(pipe, ds.X_train, ds.y_train, ds.X_test, ds.y_test, cv_folds=5)
        results[kind_str] = (pipe, m)

    # Pick best by accuracy.
    best_kind = max(results.keys(), key=lambda k: results[k][1].accuracy)
    best_pipe, best_metrics = results[best_kind]
    print(f"Best model: {best_kind}  acc={best_metrics.accuracy:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # --- Top-left: candidate comparison ---------------------------------------
    ax = axes[0, 0]
    kinds = list(results.keys())
    acc = [results[k][1].accuracy for k in kinds]
    f1 = [results[k][1].f1_macro for k in kinds]
    cv = [results[k][1].cv_accuracy_mean for k in kinds]
    cv_std = [results[k][1].cv_accuracy_std for k in kinds]
    x = np.arange(len(kinds))
    width = 0.26
    ax.bar(x - width, acc, width, label="Test accuracy", color="#0072B2")
    ax.bar(x, f1, width, label="Test F1 (macro)", color="#009E73")
    ax.bar(x + width, cv, width, yerr=cv_std, label="CV accuracy (5-fold)",
           color="#E69F00", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, rotation=10)
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Candidate Model Comparison", loc="left")
    ax.legend(loc="lower right", fontsize=9)

    # --- Top-right: confusion matrix of best model ----------------------------
    ax = axes[0, 1]
    cm = np.array(best_metrics.confusion_matrix)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(list(TARGET_NAMES), rotation=20, ha="right")
    ax.set_yticklabels(list(TARGET_NAMES))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — best model: {best_kind}", loc="left")
    # Annotate cells.
    n_total = cm.sum()
    for i in range(3):
        for j in range(3):
            v = cm[i, j]
            color = "white" if v > n_total / 6 else "#2b2b2b"
            ax.text(j, i, f"{v}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Bottom-left: SHAP summary (try tree-based, fall back to LogReg) ------
    ax = axes[1, 0]
    shap_kind = best_kind
    if shap_kind not in ("random_forest", "lightgbm"):
        # Use RF for the SHAP panel (best_kind is svm/logreg which lack native TreeExplainer).
        shap_kind = "random_forest"
        shap_pipe = build_pipeline("random_forest")
        shap_pipe.fit(ds.X_train, ds.y_train)
    else:
        shap_pipe = best_pipe

    if HAVE_SHAP:
        explanation = explain_with_shap(
            shap_pipe,
            X_background=ds.X_train,
            X_explain=ds.X_test[:30],
            feature_names=list(FEATURE_NAMES),
            target_names=list(TARGET_NAMES),
        )
        if explanation is not None:
            # Build a grouped bar chart: feature × class mean(|SHAP|).
            means_per_class = []
            for ci in range(3):
                summary = explanation.summary_for_class(ci)
                means_per_class.append([summary[f] for f in FEATURE_NAMES])
            means_per_class = np.array(means_per_class)  # (3, 4)
            x = np.arange(len(FEATURE_NAMES))
            width = 0.26
            colors = ["#0072B2", "#009E73", "#CC79A7"]
            for ci, color in enumerate(colors):
                ax.bar(x + (ci - 1) * width, means_per_class[ci], width,
                       label=TARGET_NAMES[ci], color=color)
            ax.set_xticks(x)
            ax.set_xticklabels([f.replace("_", "\n") for f in FEATURE_NAMES], fontsize=8)
            ax.set_ylabel("mean |SHAP value|")
            ax.set_title(f"SHAP Feature Importance ({shap_kind})", loc="left")
            ax.legend(loc="upper right", fontsize=9)
        else:
            ax.text(0.5, 0.5, "SHAP unavailable", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "SHAP not installed", ha="center", va="center")
        ax.set_axis_off()

    # --- Bottom-right: per-class probability strip for 3 canonical rows -------
    ax = axes[1, 1]
    samples = np.array([
        [5.1, 3.5, 1.4, 0.2],  # setosa
        [6.2, 2.9, 4.3, 1.3],  # versicolor
        [7.7, 3.8, 6.7, 2.2],  # virginica
    ])
    probas = best_pipe.predict_proba(samples)
    im = ax.imshow(probas, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(list(TARGET_NAMES))
    ax.set_yticklabels(["setosa\nsample", "versicolor\nsample", "virginica\nsample"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Input sample")
    ax.set_title("Predicted probabilities (3 canonical rows)", loc="left")
    for i in range(3):
        for j in range(3):
            v = probas[i, j]
            color = "white" if v > 0.5 else "#2b2b2b"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Iris Production — End-to-End ML Pipeline (best model: {best_kind}, acc={best_metrics.accuracy:.2%})",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
