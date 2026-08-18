#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P8_churn_survival — dual-modeling churn benchmark
(classifier + survival) with uplift/expected-value retention targeting.

Usage
-----
::

    # 1. Default: synthetic data, both classifiers + Cox PH
    python train.py

    # 2. Real Telco dataset (downloads on first run)
    python train.py --use-real

    # 3. Custom CSV (must have the unified schema columns)
    python train.py --csv /path/to/churn.csv

    # 4. Save artifacts
    python train.py \\
        --metrics-json metrics.json \\
        --survival-plot assets/survival.png \\
        --uplift-plot assets/uplift.png

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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
try:
    from shared import apply_style
    apply_style()
except Exception:  # pragma: no cover
    pass

from dataset import (  # noqa: E402
    SCHEMA, load_churn_dataset, build_train_test_split, ChurnDataset,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, ModelKind,
    ClassifierMetrics, SurvivalMetrics, UpliftResult,
    train_classifier, fit_kaplan_meier, fit_cox_ph,
    compute_uplift, expected_value_policy,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("churn_train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churn_train",
        description="P8 Churn Survival — classifier + survival + uplift.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # Default: synthetic data, all models
  python train.py

  # Real Telco dataset (downloads on first run)
  python train.py --use-real

  # Save artifacts
  python train.py --metrics-json metrics.json \\
      --survival-plot assets/survival.png --uplift-plot assets/uplift.png
""",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to a churn CSV (must contain the unified schema columns).",
    )
    parser.add_argument(
        "--use-real", action="store_true",
        help="Download the real IBM Telco churn dataset (default: synthetic).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=2000,
        help="Synthetic dataset size (default: 2000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test fraction (default: 0.20).",
    )
    parser.add_argument(
        "--skip-classifier", action="store_true",
        help="Skip classifier training (LogReg + Random Forest).",
    )
    parser.add_argument(
        "--skip-survival", action="store_true",
        help="Skip survival analysis (Kaplan-Meier + Cox PH).",
    )
    parser.add_argument(
        "--skip-uplift", action="store_true",
        help="Skip uplift / expected-value computation.",
    )
    parser.add_argument(
        "--offer-cost", type=float, default=10.0,
        help="Retention offer cost in $ (default: 10).",
    )
    parser.add_argument(
        "--metrics-json", default=None,
        help="Optional path to dump all metrics as JSON.",
    )
    parser.add_argument(
        "--survival-plot", default=None,
        help="Optional path to save a survival curves PNG.",
    )
    parser.add_argument(
        "--uplift-plot", default=None,
        help="Optional path to save a cumulative ROI / uplift PNG.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_classifier_table(rows: List[Dict]) -> str:
    headers = ["model", "accuracy", "f1", "auc", "brier", "logloss", "fit_s"]
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r.get(h, ""))))
    sep = "  ".join("-" * widths[h] for h in headers)
    out = ["  ".join(h.ljust(widths[h]) for h in headers), sep]
    for r in rows:
        out.append("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers))
    return "\n".join(out)


def _plot_survival_curves(kmf, cph, test_X: pd.DataFrame, output_path: Path) -> None:
    """Plot KM survival curve + Cox PH individual survival curves for a few customers."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # Left: KM curve with confidence interval.
    ax = axes[0]
    # lifelines 0.30+ renamed plot_survival_function_ to plot_survival_function.
    plot_fn = getattr(kmf, "plot_survival_function", None) or kmf.plot_survival_function_
    plot_fn(ax=ax, ci_show=True, color="#0072B2")
    ax.set_title("Kaplan-Meier survival curve (with 95% CI)", loc="left")
    ax.set_xlabel("Months")
    ax.set_ylabel("P(not churned)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.4)

    # Right: Cox PH individual survival curves for 5 sample customers.
    ax = axes[1]
    sample_X = test_X.sample(n=min(5, len(test_X)), random_state=42)
    try:
        for idx, (_, row) in enumerate(sample_X.iterrows()):
            # Predict survival function for this customer.
            row_df = pd.DataFrame([row])
            # Apply the same one-hot encoding the Cox PH fitter expects.
            from model import _prepare_cox_input
            cox_input = _prepare_cox_input(row_df, pd.Series([0]), pd.Series([0]))
            cox_features = cox_input.drop(columns=["duration", "event"])
            sf = cph.predict_survival_function(cox_features)
            color = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"][idx % 5]
            ax.plot(sf.index, sf.values.ravel(), "-", color=color, linewidth=1.5,
                    label=f"Customer #{idx+1}")
        ax.set_title("Cox PH predicted survival curves (5 sample customers)", loc="left")
        ax.set_xlabel("Months")
        ax.set_ylabel("P(not churned)")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.4)
    except Exception as exc:
        ax.text(0.5, 0.5, f"Cox PH prediction failed:\n{exc}",
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_uplift_curve(result: UpliftResult, output_path: Path, offer_cost: float) -> None:
    """Plot cumulative ROI curve vs. number of customers targeted."""
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    n = len(result.cumulative_roi)
    x_pct = np.arange(1, n + 1) / n * 100  # percentage of customers targeted

    ax.plot(x_pct, result.cumulative_roi, "-", color="#0072B2", linewidth=2.0,
            label="Cumulative ROI")
    ax.axvline(x_pct[result.optimal_threshold_idx], color="#D55E00",
               linestyle="--", linewidth=1.2,
               label=f"Optimal threshold ({result.total_targeted} customers, "
                     f"ROI=${result.total_roi:.0f})")
    ax.axhline(0, color="#2b2b2b", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Customers targeted (%)")
    ax.set_ylabel("Cumulative net ROI ($)")
    ax.set_title(f"Expected-value retention policy (offer_cost=${offer_cost:.2f})", loc="left")
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

    # Step 1 — load dataset.
    try:
        log.info("Loading churn dataset ...")
        ds = load_churn_dataset(
            csv_path=args.csv, use_real=args.use_real,
            n_samples=args.n_samples, seed=args.seed,
        )
        log.info("  Loaded %d samples (source=%s)", ds.n_samples, ds.source)
        log.info("  churn rate: %.3f, mean tenure: %.1f months",
                 ds.y_churned.mean(), ds.durations.mean())
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 2

    # Step 2 — stratified split.
    train, test = build_train_test_split(ds, test_size=args.test_size,
                                          random_state=args.seed)
    log.info("  Splits: train=%d, test=%d (test churn rate=%.3f)",
             train.n_samples, test.n_samples, test.y_churned.mean())

    classifier_rows: List[Dict] = []
    classifier_metrics: Dict[str, ClassifierMetrics] = {}
    survival_metrics: Dict[str, SurvivalMetrics] = {}
    uplift_result: Optional[UpliftResult] = None
    best_pipe = None  # for uplift computation

    # Step 3 — classifier training.
    if not args.skip_classifier:
        for name in CANDIDATE_MODELS:
            try:
                log.info("Training classifier: %s", name)
                pipe, m = train_classifier(
                    ModelKind(name), train.X, train.y_churned,
                    test.X, test.y_churned, random_state=args.seed,
                )
                classifier_metrics[name] = m
                classifier_rows.append({
                    "model": name,
                    "accuracy": f"{m.accuracy:.4f}",
                    "f1": f"{m.f1_macro:.4f}",
                    "auc": f"{m.roc_auc:.4f}",
                    "brier": f"{m.brier_score:.4f}",
                    "logloss": f"{m.log_loss:.4f}",
                    "fit_s": f"{m.fit_time_seconds:.2f}",
                })
                log.info("  %s — acc=%.4f, auc=%.4f, brier=%.4f",
                         name, m.accuracy, m.roc_auc, m.brier_score)
                # Keep the best classifier for uplift computation.
                if best_pipe is None or m.roc_auc > best_pipe[1].roc_auc:
                    best_pipe = (pipe, m, name)
            except Exception as exc:
                log.error("  %s failed: %s", name, exc)
                if args.verbose:
                    traceback.print_exc()
                return 3

    # Step 4 — survival analysis.
    kmf = None
    cph = None
    if not args.skip_survival:
        # Kaplan-Meier.
        try:
            log.info("Fitting Kaplan-Meier ...")
            kmf, km_metrics = fit_kaplan_meier(train.durations, train.events, label="train")
            survival_metrics["kaplan_meier"] = km_metrics
            log.info("  KM — mean_survival=%s months, events=%d, censored=%d",
                     f"{km_metrics.mean_survival_time_months:.1f}" if km_metrics.mean_survival_time_months else "—",
                     km_metrics.n_events, km_metrics.n_censored)
            log.info("  S(12)=%.3f, S(36)=%.3f, S(72)=%.3f",
                     kmf.predict(12), kmf.predict(36), kmf.predict(72))
        except Exception as exc:
            log.error("  Kaplan-Meier failed: %s", exc)
            if args.verbose:
                traceback.print_exc()

        # Cox PH.
        try:
            log.info("Fitting Cox Proportional Hazards ...")
            cph, cox_metrics = fit_cox_ph(
                train.X, train.durations, train.events,
                test.X, test.durations, test.events,
            )
            survival_metrics["cox_ph"] = cox_metrics
            log.info("  Cox PH — c_index=%.4f, median_survival=%s months",
                     cox_metrics.c_index,
                     f"{cox_metrics.median_survival_time_months:.1f}" if cox_metrics.median_survival_time_months else "—")
            # Top coefficients by absolute magnitude.
            top_coefs = cph.params_.sort_values(key=np.abs, ascending=False).head(5)
            log.info("  Top 5 coefficients by |β|:")
            for cname, val in top_coefs.items():
                log.info("    %-40s β=%+.4f", cname, val)
        except Exception as exc:
            log.error("  Cox PH failed: %s", exc)
            if args.verbose:
                traceback.print_exc()

    # Step 5 — uplift / expected-value.
    if not args.skip_uplift and best_pipe is not None:
        try:
            log.info("Computing uplift via two-model approach (best classifier = %s) ...",
                     best_pipe[2])
            pipe = best_pipe[0]
            # Scenario A: control (no offer).
            p_control = pipe.predict_proba(test.X)[:, 1]
            # Scenario B: treatment = 10% discount on monthly_charges.
            X_treated = test.X.copy()
            X_treated["monthly_charges"] = X_treated["monthly_charges"] * 0.9
            p_treatment = pipe.predict_proba(X_treated)[:, 1]
            uplift = compute_uplift(p_control, p_treatment)
            log.info("  Uplift: mean=%.4f, std=%.4f, positive=%d/%d (%.1f%%)",
                     uplift.mean(), uplift.std(),
                     int((uplift > 0).sum()), len(uplift),
                     100 * (uplift > 0).mean())

            ltv = test.df["total_charges"].values
            uplift_result = expected_value_policy(uplift, customer_ltv=ltv,
                                                   offer_cost=args.offer_cost)
            log.info("  Optimal targeting: %d / %d customers (%.1f%%), total ROI = $%.2f",
                     uplift_result.total_targeted, len(uplift),
                     100 * uplift_result.total_targeted / len(uplift),
                     uplift_result.total_roi)
            log.info("  Optimal threshold: EV >= %.4f", uplift_result.optimal_threshold_value)
        except Exception as exc:
            log.error("  Uplift computation failed: %s", exc)
            if args.verbose:
                traceback.print_exc()

    # Print classifier table.
    if classifier_rows:
        print()
        print(_format_classifier_table(classifier_rows))
        print()

    # Optional metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {
                "csv": args.csv,
                "use_real": args.use_real,
                "n_samples": args.n_samples,
                "seed": args.seed,
                "test_size": args.test_size,
                "offer_cost": args.offer_cost,
            },
            "classifier_metrics": {k: v.to_dict() for k, v in classifier_metrics.items()},
            "survival_metrics": {k: v.to_dict() for k, v in survival_metrics.items()},
            "uplift": uplift_result.to_dict() if uplift_result else None,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Optional survival plot.
    if args.survival_plot and kmf is not None and cph is not None:
        try:
            _plot_survival_curves(kmf, cph, test.X, Path(args.survival_plot))
            log.info("Saved survival curves → %s", args.survival_plot)
        except Exception as exc:
            log.warning("Failed to render survival plot: %s", exc)

    # Optional uplift plot.
    if args.uplift_plot and uplift_result is not None:
        try:
            _plot_uplift_curve(uplift_result, Path(args.uplift_plot), args.offer_cost)
            log.info("Saved uplift curve → %s", args.uplift_plot)
        except Exception as exc:
            log.warning("Failed to render uplift plot: %s", exc)

    # Summary line.
    if classifier_metrics:
        best_name = max(classifier_metrics.keys(),
                        key=lambda k: classifier_metrics[k].roc_auc)
        best = classifier_metrics[best_name]
        print(f"BEST_CLASSIFIER={best_name}")
        print(f"BEST_ROC_AUC={best.roc_auc:.4f}")
        print(f"BEST_BRIER={best.brier_score:.4f}")
    if "cox_ph" in survival_metrics:
        print(f"COX_C_INDEX={survival_metrics['cox_ph'].c_index:.4f}")
    if uplift_result:
        print(f"OPTIMAL_TARGETED={uplift_result.total_targeted}")
        print(f"TOTAL_ROI={uplift_result.total_roi:.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
