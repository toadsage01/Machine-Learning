"""
generate_hero
=============

Hero image for the P13 AutoML Pipeline README.
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
import os  # noqa: E402
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from shared import apply_style  # noqa: E402
apply_style()

from dataset import load_automl_dataset, ColumnType  # noqa: E402
from model import run_automl_pipeline  # noqa: E402


def main() -> None:
    df, profile = load_automl_dataset(task="classification", n_samples=300,
                                        n_features=6, seed=42)
    report, pipeline, _ = run_automl_pipeline(df, profile, n_trials=8,
                                                cv_folds=3, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # Top-left: schema type distribution.
    ax = axes[0, 0]
    types = ["numeric", "categorical", "datetime", "binary", "text", "id"]
    counts = [profile.n_numeric, profile.n_categorical, profile.n_datetime,
              profile.n_binary, profile.n_text, profile.n_id]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    bars = ax.bar(types, counts, color=colors[:len(types)])
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, c + 0.1, str(c),
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Column count")
    ax.set_title("Schema type inference", loc="left", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    # Top-right: HPO trial scores.
    ax = axes[0, 1]
    trial_nums = [t.trial_number for t in report.all_trials]
    scores = [t.score for t in report.all_trials]
    models = [t.model_kind for t in report.all_trials]
    model_colors = {"lightgbm": "#0072B2", "xgboost": "#D55E00", "catboost": "#009E73"}
    colors = [model_colors.get(m, "#999") for m in models]
    ax.bar(trial_nums, scores, color=colors, width=0.6)
    ax.axhline(report.best_score, color="#D55E00", linestyle="--",
               linewidth=1.0, label=f"Best score = {report.best_score:.4f}")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Score")
    ax.set_title("Optuna HPO trial scores", loc="left", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Bottom-left: pipeline architecture.
    ax = axes[1, 0]
    ax.set_axis_off()
    ax.set_title("AutoML pipeline architecture", loc="left", fontsize=12)
    steps = [
        ("Input CSV", 0.05, 0.85),
        ("Schema Inference", 0.05, 0.65),
        ("AutoFeatureTransformer", 0.05, 0.45),
        ("Optuna HPO", 0.05, 0.25),
        ("MLflow Tracking", 0.05, 0.05),
        ("Numeric: impute+scale", 0.4, 0.65),
        ("Categorical: one-hot", 0.4, 0.45),
        ("DateTime: decompose", 0.4, 0.25),
        ("LightGBM", 0.7, 0.35),
        ("XGBoost", 0.7, 0.2),
        ("CatBoost", 0.7, 0.05),
        ("Best Model", 0.9, 0.15),
    ]
    for name, x, y in steps:
        color = "#0072B2" if x < 0.35 else "#D55E00" if x < 0.65 else "#009E73"
        ax.scatter(x, y, s=200, color=color, zorder=5, edgecolors="white", linewidth=1.0)
        ax.text(x, y, name, fontsize=6, ha="center", va="center", color="white",
                fontweight="bold", zorder=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Bottom-right: feature importance (if available).
    ax = axes[1, 1]
    if pipeline is not None:
        try:
            model = pipeline.named_steps.get("model", None)
            if model is not None and hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]
                ax.barh(range(len(indices)), importances[indices][::-1],
                        color="#0072B2")
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([f"feature_{i}" for i in indices[::-1]], fontsize=8)
                ax.set_xlabel("Importance")
                ax.set_title("Top-10 feature importances (best model)", loc="left", fontsize=12)
                ax.grid(True, axis="x", alpha=0.3)
            else:
                ax.text(0.5, 0.5, "Feature importances\nnot available", ha="center", va="center")
                ax.set_axis_off()
        except Exception:
            ax.text(0.5, 0.5, "Feature importances\nnot available", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Pipeline not available", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle("AutoML Pipeline — Schema Inference + Optuna HPO + MLflow Tracking",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
