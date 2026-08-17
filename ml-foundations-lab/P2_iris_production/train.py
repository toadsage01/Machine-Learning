#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P2_iris_production.

Trains four candidate pipelines (Logistic Regression, Random Forest, SVM,
LightGBM) on the Iris dataset, evaluates each with 5-fold CV + holdout test
metrics, and exports the best-performing model to ONNX for serving.

Usage
-----
::

    # Default: evaluate all four candidates, save best to models/best.onnx
    python train.py

    # Restrict to a subset of models
    python train.py --models logreg lightgbm

    # Use a local CSV instead of sklearn's bundled copy
    python train.py --csv /path/to/iris.csv

    # Custom test size / seed / output
    python train.py --test-size 0.25 --seed 0 --out models/winner.onnx

    # Save evaluation plots + per-model metrics JSON
    python train.py --plot assets/eval.png --metrics-json models/metrics.json

Exit codes
----------
* 0  : best model trained + ONNX exported.
* 1  : usage error.
* 2  : data loading failed.
* 3  : training / evaluation failed.
* 4  : ONNX export failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Bootstrap repo & project roots onto sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Apply the project-wide matplotlib style before any figure is created.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
try:
    from shared import apply_style
    apply_style()
except Exception:  # pragma: no cover
    pass

from dataset import load_iris_split, FEATURE_NAMES, TARGET_NAMES  # noqa: E402
from model import (  # noqa: E402
    CANDIDATE_MODELS, ModelKind, build_pipeline, evaluate_pipeline,
    export_to_onnx, explain_with_shap, HAVE_SHAP, HAVE_SKL2ONNX,
    save_pipeline,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("iris_production")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iris_train",
        description="Train 4 candidate models on Iris, export best to ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # Evaluate everything, save best ONNX
  python train.py

  # Just two models
  python train.py --models logreg lightgbm

  # With SHAP + evaluation plot
  python train.py --plot assets/eval.png --shap
""",
    )

    parser.add_argument(
        "--models", "-m", nargs="+",
        choices=list(CANDIDATE_MODELS.keys()),
        default=list(CANDIDATE_MODELS.keys()),
        help="Subset of candidate models to evaluate (default: all four).",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to a local Iris CSV (default: use sklearn.datasets.load_iris).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction held out for test (default: 0.20 = 30 of 150).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random state for train/test split & model fitting (default: 42).",
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="Number of cross-validation folds on the training set (default: 5).",
    )
    parser.add_argument(
        "--out", "-o", default="models/best.onnx",
        help="Output ONNX path (default: models/best.onnx).",
    )
    parser.add_argument(
        "--joblib-out", default="models/best.joblib",
        help="Optional joblib-persisted fallback path (default: models/best.joblib).",
    )
    parser.add_argument(
        "--metrics-json", default=None,
        help="Optional path to dump all per-model metrics as JSON.",
    )
    parser.add_argument(
        "--plot", default=None,
        help="Optional path to save an evaluation comparison PNG.",
    )
    parser.add_argument(
        "--shap", action="store_true",
        help="Compute SHAP values for the best model (tree/linear only).",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_candidate(
    kind: ModelKind,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    cv_folds: int,
    seed: int,
):
    """Train + evaluate one candidate pipeline."""
    log.info("Training %s ...", kind.value)
    pipeline = build_pipeline(kind=kind, random_state=seed)
    metrics = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test, cv_folds=cv_folds)
    log.info(
        "  %s — acc=%.4f, f1=%.4f, cv_acc=%.4f ± %.4f, fit=%.1fs",
        kind.value, metrics.accuracy, metrics.f1_macro,
        metrics.cv_accuracy_mean, metrics.cv_accuracy_std, metrics.fit_time_seconds,
    )
    return pipeline, metrics


def _pick_best(per_model: Dict[str, Tuple]) -> str:
    """Pick the best model by holdout accuracy (tiebreak: f1_macro)."""
    def score(item):
        kind, (pipeline, metrics) = item
        return (metrics.accuracy, metrics.f1_macro)
    return max(per_model.items(), key=score)[0]


def _plot_comparison(per_model: Dict[str, Tuple], out_path: Path) -> None:
    """Side-by-side bar chart of accuracy / f1 / cv_accuracy for all candidates."""
    kinds = list(per_model.keys())
    acc = [per_model[k][1].accuracy for k in kinds]
    f1 = [per_model[k][1].f1_macro for k in kinds]
    cv = [per_model[k][1].cv_accuracy_mean for k in kinds]
    cv_std = [per_model[k][1].cv_accuracy_std for k in kinds]
    x = np.arange(len(kinds))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, acc, width, label="Test accuracy", color="#0072B2")
    ax.bar(x, f1, width, label="Test F1 (macro)", color="#009E73")
    ax.bar(x + width, cv, width, yerr=cv_std, label="CV accuracy (5-fold)", color="#E69F00", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, rotation=10)
    ax.set_ylim(0.8, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Iris Production — Candidate Model Comparison", loc="left")
    ax.legend(loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Saved evaluation plot → %s", out_path)


def _plot_shap(explanation, out_path: Path) -> None:
    """Save a SHAP summary bar chart (mean |SHAP| per feature per class)."""
    if explanation is None:
        return
    n_classes = len(explanation.target_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4.5), constrained_layout=True)
    if n_classes == 1:
        axes = [axes]
    for ci, ax in enumerate(axes):
        means = explanation.summary_for_class(ci)
        names = list(means.keys())
        vals = list(means.values())
        order = np.argsort(vals)[::-1]
        ax.barh([names[i] for i in order][::-1], [vals[i] for i in order][::-1], color="#CC79A7")
        ax.set_title(f"Class {ci}: {explanation.target_names[ci]}", loc="left")
        ax.set_xlabel("mean |SHAP value|")
    fig.suptitle("SHAP Feature Importance by Class", fontsize=14, fontweight="bold", x=0.01, ha="left", y=1.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Saved SHAP summary plot → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)
    elif args.verbose == 1:
        log.setLevel(logging.DEBUG)

    # Step 1 — load + split.
    try:
        log.info("Loading Iris dataset (source=%s)", args.csv or "sklearn")
        ds = load_iris_split(csv_path=args.csv, test_size=args.test_size, random_state=args.seed)
        log.info("  train: %d rows, test: %d rows, features=%s",
                 ds.n_train, ds.n_test, list(ds.feature_names))
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 2

    # Step 2 — train + evaluate each candidate.
    per_model: Dict[str, Tuple] = {}
    try:
        for kind_str in args.models:
            kind = CANDIDATE_MODELS[kind_str]
            pipeline, metrics = _run_candidate(
                kind, ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                cv_folds=args.cv, seed=args.seed,
            )
            per_model[kind_str] = (pipeline, metrics)
    except Exception as exc:
        log.error("Training failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 3

    # Step 3 — pick the best.
    best_kind = _pick_best(per_model)
    best_pipeline, best_metrics = per_model[best_kind]
    log.info("🏆 Best model: %s (accuracy=%.4f, f1=%.4f, cv=%.4f ± %.4f)",
             best_kind, best_metrics.accuracy, best_metrics.f1_macro,
             best_metrics.cv_accuracy_mean, best_metrics.cv_accuracy_std)

    # Step 4 — save the best pipeline to joblib (always; serves as a fallback).
    try:
        joblib_path = save_pipeline(best_pipeline, Path(args.joblib_out))
        log.info("Saved joblib fallback → %s", joblib_path)
    except Exception as exc:
        log.warning("Joblib save failed (continuing): %s", exc)

    # Step 5 — export to ONNX.
    try:
        if not HAVE_SKL2ONNX:
            raise RuntimeError("skl2onnx is not installed; cannot export to ONNX.")
        onnx_path = export_to_onnx(best_pipeline, Path(args.out))
        log.info("✓ Exported best model to ONNX → %s (%.1f KB)",
                 onnx_path.resolve(), onnx_path.stat().st_size / 1024)
    except Exception as exc:
        log.error("ONNX export failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 4

    # Step 6 — optional metrics JSON.
    if args.metrics_json:
        try:
            payload = {
                "best_model": best_kind,
                "best_metrics": best_metrics.to_dict(),
                "all_models": {
                    k: m.to_dict() for k, (_, m) in per_model.items()
                },
                "config": {
                    "test_size": args.test_size,
                    "seed": args.seed,
                    "cv_folds": args.cv,
                    "feature_names": list(FEATURE_NAMES),
                    "target_names": list(TARGET_NAMES),
                },
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            metrics_path = Path(args.metrics_json)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info("Saved metrics JSON → %s", metrics_path)
        except Exception as exc:
            log.warning("Failed to write metrics JSON: %s", exc)

    # Step 7 — optional eval plot.
    if args.plot:
        try:
            _plot_comparison(per_model, Path(args.plot))
        except Exception as exc:
            log.warning("Failed to render eval plot: %s", exc)

    # Step 8 — optional SHAP.
    if args.shap and HAVE_SHAP:
        try:
            log.info("Computing SHAP values for the best model (%s) ...", best_kind)
            explanation = explain_with_shap(
                best_pipeline,
                X_background=ds.X_train,
                X_explain=ds.X_test[: min(50, len(ds.X_test))],
                feature_names=list(FEATURE_NAMES),
                target_names=list(TARGET_NAMES),
            )
            if explanation is None:
                log.warning("SHAP explanation returned None (model kind may be unsupported).")
            else:
                shap_path = Path("assets/shap_summary.png")
                _plot_shap(explanation, shap_path)
        except Exception as exc:
            log.warning("SHAP computation failed (continuing): %s", exc)
            if args.verbose:
                traceback.print_exc()

    # Single-line summary for shell scripting.
    print(f"BEST_MODEL={best_kind}")
    print(f"ONNX_PATH={Path(args.out).resolve()}")
    print(f"JOBLIB_PATH={Path(args.joblib_out).resolve()}")
    print(f"ACCURACY={best_metrics.accuracy:.4f}")
    print(f"F1_MACRO={best_metrics.f1_macro:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
