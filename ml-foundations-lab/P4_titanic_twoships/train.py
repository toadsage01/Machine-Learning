#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P4_titanic_twoships — the dual-dataset Titanic
benchmark with probability calibration and fairness auditing.

Usage
-----
::

    # 1. Full benchmark: all 3 boosters × both datasets
    python train.py

    # 2. Restrict to one dataset / model
    python train.py -d classic -m catboost

    # 3. Apply Isotonic calibration + save calibration curves
    python train.py --calibration isotonic --plot assets/calibration_classic.png

    # 4. Save fairness audit + metrics JSON
    python train.py --fairness-plot assets/fairness.png --metrics-json metrics.json

Exit codes
----------
* 0  : benchmark completed.
* 1  : usage error.
* 2  : data loading failed.
* 3  : training failed.
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
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Apply project-wide matplotlib style before pyplot import.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
try:
    from shared import apply_style
    apply_style()
except Exception:  # pragma: no cover
    pass

from dataset import (  # noqa: E402
    DatasetKind, load_classic_titanic, load_spaceship_titanic, load_unified,
    UnifiedDataset, SCHEMA,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, ModelKind, CalibrationKind,
    ModelMetrics, CalibrationResult, FairnessReport,
    train_model, calibrate, evaluate_calibration, compute_fairness,
    DEFAULT_FAIRNESS_FEATURES,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("titanic_train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="titanic_train",
        description="P4 Titanic Two-Ships benchmark: train + calibrate + fairness audit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # Full benchmark
  python train.py

  # Just CatBoost on the classic dataset, isotonic calibration
  python train.py -d classic -m catboost --calibration isotonic

  # Save calibration + fairness plots
  python train.py --calibration-plot assets/calib.png --fairness-plot assets/fair.png
""",
    )
    parser.add_argument(
        "--datasets", "-d", nargs="+",
        choices=["classic", "spaceship"],
        default=["classic", "spaceship"],
        help="Subset of datasets to benchmark (default: both).",
    )
    parser.add_argument(
        "--models", "-m", nargs="+",
        choices=list(CANDIDATE_MODELS.keys()),
        default=list(CANDIDATE_MODELS.keys()),
        help="Subset of boosters to evaluate (default: all three).",
    )
    parser.add_argument(
        "--calibration", "-c",
        choices=["isotonic", "sigmoid", "none"],
        default="isotonic",
        help="Calibration method applied to every model (default: isotonic).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction held out for test (default: 0.20).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random state for split + model fitting (default: 42).",
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="CV folds on the training set (default: 5).",
    )
    parser.add_argument(
        "--metrics-json", default=None,
        help="Optional path to dump all per-model metrics as JSON.",
    )
    parser.add_argument(
        "--calibration-plot", default=None,
        help="Optional path to save calibration curve PNG (one per dataset).",
    )
    parser.add_argument(
        "--fairness-plot", default=None,
        help="Optional path to save fairness disparity bar chart PNG.",
    )
    parser.add_argument(
        "--roc-plot", default=None,
        help="Optional path to save ROC curves PNG.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_dataset(name: str) -> UnifiedDataset:
    """Dispatch by name."""
    return load_unified(name)


def _train_one(
    model_name: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    cv_folds: int,
    seed: int,
) -> Tuple[object, ModelMetrics]:
    """Train + evaluate one booster."""
    log.info("  Training %s ...", model_name)
    pipe, metrics = train_model(
        ModelKind(model_name), X_train, y_train, X_test, y_test,
        cv_folds=cv_folds, random_state=seed,
    )
    log.info("    %s — acc=%.4f, auc=%.4f, brier=%.4f, cv=%.4f ± %.4f",
             model_name, metrics.accuracy, metrics.roc_auc,
             metrics.brier_score, metrics.cv_accuracy_mean, metrics.cv_accuracy_std)
    return pipe, metrics


def _format_table(rows: List[Dict]) -> str:
    headers = ["dataset", "model", "acc", "auc", "brier", "logloss", "cv_acc", "f1", "fit_ms"]
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r.get(h, ""))))
    sep = "  ".join("-" * widths[h] for h in headers)
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    out = [header_line, sep]
    for r in rows:
        out.append("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers))
    return "\n".join(out)


def _plot_calibration(
    results: List[CalibrationResult],
    output_path: Path,
    title: str,
) -> None:
    """Plot calibration curves for multiple models on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 7.5), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    # Perfect-calibration diagonal.
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="perfect")

    for i, cr in enumerate(results):
        color = colors[i % len(colors)]
        ax.plot(
            cr.mean_predicted_value, cr.fraction_of_positives,
            "o-", color=color, linewidth=1.8, markersize=7,
            label=f"{cr.model_name} ({cr.calibration_kind}) — Brier={cr.brier_score:.4f}",
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (in bin)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_fairness(
    fairness_reports: Dict[str, FairnessReport],
    output_path: Path,
) -> None:
    """Grouped bar chart of selection-rate per (model, slice)."""
    # Gather all unique (feature, value) slices across models.
    all_slices: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    for model_name, rep in fairness_reports.items():
        for s in rep.slices:
            key = (s.feature, s.value)
            all_slices.setdefault(key, []).append((model_name, s.selection_rate))

    slice_keys = sorted(all_slices.keys())
    n_slices = len(slice_keys)
    n_models = len(fairness_reports)
    if n_slices == 0 or n_models == 0:
        return

    fig, ax = plt.subplots(figsize=(max(12, n_slices * 0.7), 6.5), constrained_layout=True)
    x = np.arange(n_slices)
    width = 0.85 / max(n_models, 1)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    for i, model_name in enumerate(fairness_reports.keys()):
        vals = []
        for k in slice_keys:
            v = next((m[1] for m in all_slices[k] if m[0] == model_name), 0.0)
            vals.append(v)
        ax.bar(x + i * width - width * (n_models - 1) / 2, vals, width,
               label=model_name, color=colors[i % len(colors)])

    slice_labels = [f"{f}={v}" for (f, v) in slice_keys]
    ax.set_xticks(x)
    ax.set_xticklabels(slice_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Selection rate (fraction predicted positive)")
    ax.set_title("Fairness audit — selection rate per subgroup", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_roc(
    models: Dict[str, Tuple[object, pd.DataFrame, pd.Series]],
    output_path: Path,
    title: str,
) -> None:
    """ROC curves for multiple models on the same axes."""
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="random")
    for i, (name, (model, X_test, y_test)) in enumerate(models.items()):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2:
            proba = proba[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, "-", color=color, linewidth=2.0, label=f"{name}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.4)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)
    elif args.verbose == 1:
        log.setLevel(logging.DEBUG)

    table_rows: List[Dict] = []
    all_metrics: Dict[str, Dict] = {}
    all_calibration: Dict[str, List[CalibrationResult]] = {}
    all_fairness: Dict[str, FairnessReport] = {}
    all_calibrated_models: Dict[str, Tuple[object, pd.DataFrame, pd.Series]] = {}

    # Step 1 — for each dataset.
    for ds_name in args.datasets:
        log.info("=== Dataset: %s ===", ds_name)
        try:
            ds = _load_dataset(ds_name)
            log.info("  Loaded %d samples, target rate=%.3f, source=%s",
                     ds.n_samples, float(ds.y.mean()), ds.source_url)
        except Exception as exc:
            log.error("Failed to load %s: %s", ds_name, exc)
            if args.verbose:
                traceback.print_exc()
            return 2

        # Train/test split — stratified on the target.
        X_tr, X_te, y_tr, y_te = train_test_split(
            ds.X, ds.y, test_size=args.test_size,
            stratify=ds.y, random_state=args.seed,
        )
        log.info("  Split: train=%d, test=%d", len(X_tr), len(X_te))

        # Step 2 — train + evaluate every candidate.
        dataset_metrics: Dict[str, Tuple[ModelMetrics, object]] = {}
        dataset_cal: List[CalibrationResult] = []
        dataset_models: Dict[str, Tuple[object, pd.DataFrame, pd.Series]] = {}
        for model_name in args.models:
            try:
                pipe, metrics = _train_one(
                    model_name, X_tr, y_tr, X_te, y_te, args.cv, args.seed,
                )
                dataset_metrics[model_name] = (metrics, pipe)
                table_rows.append({
                    "dataset": ds_name,
                    "model": model_name,
                    "acc": f"{metrics.accuracy:.4f}",
                    "auc": f"{metrics.roc_auc:.4f}",
                    "brier": f"{metrics.brier_score:.4f}",
                    "logloss": f"{metrics.log_loss:.4f}",
                    "cv_acc": f"{metrics.cv_accuracy_mean:.4f}±{metrics.cv_accuracy_std:.4f}",
                    "f1": f"{metrics.f1_macro:.4f}",
                    "fit_ms": f"{metrics.fit_time_seconds * 1000:.0f}",
                })
            except Exception as exc:
                log.error("  %s failed: %s", model_name, exc)
                if args.verbose:
                    traceback.print_exc()
                return 3

        # Step 3 — apply calibration to every model.
        cal_kind = CalibrationKind(args.calibration)
        log.info("  Applying %s calibration to every model ...", cal_kind.value)
        for model_name, (metrics, pipe) in dataset_metrics.items():
            try:
                cal_pipe = calibrate(pipe, X_tr, y_tr, cal_kind, cv_folds=args.cv)
                cal_result = evaluate_calibration(
                    cal_pipe, X_te, y_te, model_name, cal_kind.value, n_bins=10,
                )
                dataset_cal.append(cal_result)
                dataset_models[f"{model_name} ({cal_kind.value})"] = (cal_pipe, X_te, y_te)
                log.info("    %-10s brier=%s skill=%.4f logloss=%s",
                         model_name, f"{cal_result.brier_score:.4f}",
                         cal_result.brier_skill_score, f"{cal_result.log_loss:.4f}")
            except Exception as exc:
                log.warning("  Calibration failed for %s: %s", model_name, exc)

        all_calibration[ds_name] = dataset_cal
        all_calibrated_models[ds_name] = list(dataset_models.values())[0] if dataset_models else None
        # Convert dataset_models to dataset_models_dict (named)
        all_calibrated_models[ds_name] = dataset_models

        # Step 4 — fairness audit on the BEST model (highest accuracy).
        best_name = max(dataset_metrics.keys(),
                        key=lambda k: dataset_metrics[k][0].accuracy)
        best_pipe = dataset_metrics[best_name][1]
        log.info("  Fairness audit on best model: %s ...", best_name)
        fairness = compute_fairness(best_pipe, X_te, y_te, best_name)
        all_fairness[ds_name] = fairness
        log.info("    acc_disp_ratio=%.4f, sel_disp_ratio=%.4f, n_slices=%d",
                 fairness.accuracy_disparity_ratio,
                 fairness.selection_disparity_ratio,
                 len(fairness.slices))
        log.info("    Worst slice (by selection_rate gap):")
        # Identify the slice with the most extreme selection rate.
        sorted_slices = sorted(fairness.slices, key=lambda s: abs(s.selection_rate - s.base_rate), reverse=True)
        for s in sorted_slices[:3]:
            log.info("      %s=%s  n=%d  acc=%.3f  sel=%.3f  base=%.3f  fpr=%.3f  fnr=%.3f",
                     s.feature, s.value, s.n_samples, s.accuracy,
                     s.selection_rate, s.base_rate, s.false_positive_rate, s.false_negative_rate)

        # Save metrics per dataset.
        all_metrics[ds_name] = {
            "models": {k: v[0].to_dict() for k, v in dataset_metrics.items()},
            "calibration": [c.to_dict() for c in dataset_cal],
            "fairness": fairness.to_dict(),
        }

    # Print final results table.
    print()
    print(_format_table(table_rows))
    print()

    # Optional metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {
                "datasets": args.datasets,
                "models": args.models,
                "calibration": args.calibration,
                "test_size": args.test_size,
                "seed": args.seed,
                "cv_folds": args.cv,
            },
            "results": all_metrics,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Optional per-dataset calibration plots.
    if args.calibration_plot:
        for ds_name, cal_results in all_calibration.items():
            if not cal_results:
                continue
            plot_path = Path(args.calibration_plot).with_name(
                Path(args.calibration_plot).stem + f"_{ds_name}" + Path(args.calibration_plot).suffix
            )
            try:
                _plot_calibration(
                    cal_results, plot_path,
                    title=f"Calibration curves — {ds_name} ({args.calibration})",
                )
                log.info("Saved calibration plot for %s → %s", ds_name, plot_path)
            except Exception as exc:
                log.warning("Failed to render calibration plot: %s", exc)

    # Optional fairness plot (single PNG with all datasets side-by-side).
    if args.fairness_plot:
        try:
            # Flatten fairness reports across datasets.
            flattened: Dict[str, FairnessReport] = {}
            for ds_name, rep in all_fairness.items():
                flattened[f"{ds_name}/{rep.model_name}"] = rep
            _plot_fairness(flattened, Path(args.fairness_plot))
            log.info("Saved fairness plot → %s", args.fairness_plot)
        except Exception as exc:
            log.warning("Failed to render fairness plot: %s", exc)

    # Optional ROC plots.
    if args.roc_plot:
        for ds_name, models_dict in all_calibrated_models.items():
            if not models_dict:
                continue
            plot_path = Path(args.roc_plot).with_name(
                Path(args.roc_plot).stem + f"_{ds_name}" + Path(args.roc_plot).suffix
            )
            try:
                # Use the calibrated models for ROC.
                roc_models: Dict[str, Tuple[object, pd.DataFrame, pd.Series]] = {}
                for name, (model, X_te, y_te) in models_dict.items():
                    roc_models[name] = (model, X_te, y_te)
                _plot_roc(roc_models, plot_path,
                          title=f"ROC curves — {ds_name} ({args.calibration})")
                log.info("Saved ROC plot for %s → %s", ds_name, plot_path)
            except Exception as exc:
                log.warning("Failed to render ROC plot: %s", exc)

    # Summary line.
    best_row = max(
        (r for r in table_rows),
        key=lambda r: float(r["acc"]),
        default=None,
    )
    if best_row:
        print(f"BEST_DATASET={best_row['dataset']}")
        print(f"BEST_MODEL={best_row['model']}")
        print(f"BEST_ACCURACY={best_row['acc']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
