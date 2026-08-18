#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P9_nse_forecasting — comparative equity forecasting
benchmark with walk-forward validation.

Usage
-----
::

    # 1. Default: synthetic data, all three forecasters (naive, lightgbm, transformer_fallback)
    python train.py

    # 2. Real NSE data via yfinance
    python train.py --symbol ^NSEI --use-yfinance

    # 3. Restrict to a subset of models
    python train.py --models naive lightgbm

    # 4. Try Chronos (requires `chronos-forecasting` package + network)
    python train.py --models chronos

    # 5. Save artifacts
    python train.py --metrics-json metrics.json \\
        --forecast-plot assets/forecast.png

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
    DEFAULT_TICKERS, DEFAULT_TRAIN_WINDOW, DEFAULT_TEST_WINDOW, DEFAULT_STEP,
    load_equity_dataset,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, ForecastKind, build_forecaster,
    walk_forward_evaluate,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nse_train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nse_train",
        description="P9 NSE Forecasting — Naive / LightGBM / Chronos zero-shot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # Default: synthetic data, all three forecasters
  python train.py

  # Real NSE data via yfinance
  python train.py --symbol ^NSEI --use-yfinance

  # Restrict models
  python train.py --models naive lightgbm
""",
    )
    parser.add_argument(
        "--symbol", default="^NSEI",
        help="NSE ticker symbol (default: ^NSEI = Nifty 50).",
    )
    parser.add_argument(
        "--models", "-m", nargs="+",
        choices=list(CANDIDATE_MODELS.keys()),
        default=["naive", "lightgbm", "transformer_fallback"],
        help="Subset of forecasters (default: naive, lightgbm, transformer_fallback).",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to an OHLCV CSV (must have Open/High/Low/Close/Volume columns).",
    )
    parser.add_argument(
        "--use-yfinance", action="store_true",
        help="Download real data via yfinance (default: synthetic).",
    )
    parser.add_argument(
        "--n-days-synthetic", type=int, default=1000,
        help="Synthetic dataset size (default: 1000 days).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic generator (default: 42).",
    )
    parser.add_argument(
        "--start-date", default="2020-01-01",
        help="yfinance start date (default: 2020-01-01).",
    )
    parser.add_argument(
        "--end-date", default="2024-12-31",
        help="yfinance end date (default: 2024-12-31).",
    )
    parser.add_argument(
        "--train-window", type=int, default=DEFAULT_TRAIN_WINDOW,
        help=f"Walk-forward train window in days (default: {DEFAULT_TRAIN_WINDOW}).",
    )
    parser.add_argument(
        "--test-window", type=int, default=DEFAULT_TEST_WINDOW,
        help=f"Walk-forward test window in days (default: {DEFAULT_TEST_WINDOW}).",
    )
    parser.add_argument(
        "--step", type=int, default=DEFAULT_STEP,
        help=f"Walk-forward step in days (default: {DEFAULT_STEP}).",
    )
    parser.add_argument(
        "--lgbm-n-estimators", type=int, default=200,
        help="LightGBM n_estimators (default: 200).",
    )
    parser.add_argument(
        "--lgbm-learning-rate", type=float, default=0.05,
        help="LightGBM learning rate (default: 0.05).",
    )
    parser.add_argument(
        "--transformer-epochs", type=int, default=20,
        help="Fallback transformer n_epochs (default: 20).",
    )
    parser.add_argument(
        "--transformer-seq-len", type=int, default=30,
        help="Fallback transformer sequence length (default: 30).",
    )
    parser.add_argument(
        "--chronos-prediction-length", type=int, default=21,
        help="Chronos prediction_length (default: 21).",
    )
    parser.add_argument(
        "--chronos-num-samples", type=int, default=20,
        help="Chronos num_samples (default: 20).",
    )
    parser.add_argument(
        "--skip-stationarity", action="store_true",
        help="Skip ADF stationarity checks (default: run them).",
    )
    parser.add_argument(
        "--metrics-json", default=None,
        help="Optional path to dump all metrics as JSON.",
    )
    parser.add_argument(
        "--forecast-plot", default=None,
        help="Optional path to save a forecast-vs-actual PNG.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_table(rows: List[Dict]) -> str:
    headers = ["model", "MAE", "RMSE", "MAPE%", "DirAcc", "Pb10", "Pb50", "Pb90", "cov", "width", "fit_s"]
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r.get(h, ""))))
    sep = "  ".join("-" * widths[h] for h in headers)
    out = ["  ".join(h.ljust(widths[h]) for h in headers), sep]
    for r in rows:
        out.append("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers))
    return "\n".join(out)


def _plot_forecast_vs_actual(
    features_df: pd.DataFrame,
    forecasters_results: Dict[str, "WalkForwardReport"],
    forecasters_predictions: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot actual close prices vs each forecaster's predictions."""
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)

    # The actual series is the ``Close`` column of features_df.
    # For each forecast, the predictions cover only the walk-forward test windows.
    # We plot them at their actual timestamps.
    colors = {"naive": "#0072B2", "lightgbm": "#D55E00",
              "chronos": "#009E73", "transformer_fallback": "#CC79A7"}

    ax.plot(features_df.index, features_df["Close"].values,
            "-", color="#2b2b2b", linewidth=1.2, alpha=0.7, label="Actual Close")

    for name, preds in forecasters_predictions.items():
        if preds is None or len(preds) == 0:
            continue
        # We don't have exact timestamps for each fold's test window here
        # (we'd need to re-iterate splits). Instead, plot at the tail end
        # of the dataframe, assuming walk-forward predictions cover the
        # last ``len(preds)`` rows.
        n_pred = len(preds)
        if n_pred > len(features_df):
            n_pred = len(features_df)
            preds = preds[-n_pred:]
        x_idx = features_df.index[-n_pred:]
        ax.plot(x_idx, preds, "-", color=colors.get(name, "#000000"),
                linewidth=1.5, alpha=0.85,
                label=f"{name} (MAE={forecasters_results[name].mae:.2f})")

    ax.set_xlabel("Date")
    ax.set_ylabel("Close price")
    ax.set_title("Walk-forward forecasts vs. actual close prices", loc="left")
    ax.legend(loc="best", fontsize=9)
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
        log.info("Loading equity dataset for symbol %s ...", args.symbol)
        ds = load_equity_dataset(
            symbol=args.symbol,
            csv_path=args.csv,
            use_yfinance=args.use_yfinance,
            n_days_synthetic=args.n_days_synthetic,
            seed=args.seed,
            start_date=args.start_date,
            end_date=args.end_date,
            run_stationarity=not args.skip_stationarity,
        )
        log.info("  Loaded %d samples (source=%s)", ds.n_samples, ds.source)
        log.info("  %d feature columns", len(ds.feature_columns))
        if ds.stationarity:
            log.info("  Stationarity:")
            for col, rep in ds.stationarity.items():
                log.info("    %-15s ADF p=%.4f, stationary=%s",
                         col, rep.p_value, rep.is_stationary)
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 2

    # Step 2 — train + evaluate each forecaster via walk-forward.
    table_rows: List[Dict] = []
    all_reports: Dict[str, "WalkForwardReport"] = {}
    all_predictions: Dict[str, np.ndarray] = {}

    for name in args.models:
        try:
            kind = CANDIDATE_MODELS[name]
            fc = build_forecaster(
                kind,
                n_estimators=args.lgbm_n_estimators,
                learning_rate=args.lgbm_learning_rate,
                n_epochs=args.transformer_epochs,
                seq_len=args.transformer_seq_len,
                prediction_length=args.chronos_prediction_length,
                num_samples=args.chronos_num_samples,
            )
            log.info("Training %s via walk-forward ...", name)
            report = walk_forward_evaluate(
                fc, ds.features_df,
                train_window=args.train_window,
                test_window=args.test_window,
                step=args.step,
                target_column=ds.target_column,
            )
            all_reports[name] = report
            log.info("  %s — MAE=%.4f, RMSE=%.4f, MAPE=%.4f%%, DirAcc=%.4f, fit=%.2fs",
                     name, report.mae, report.rmse, report.mape,
                     report.directional_accuracy, report.total_fit_time_seconds)
            if report.coverage_p10_p90 is not None:
                log.info("    Coverage p10-p90: %.4f, mean_width=%.4f",
                         report.coverage_p10_p90, report.mean_interval_width)

            table_rows.append({
                "model": name,
                "MAE": f"{report.mae:.4f}",
                "RMSE": f"{report.rmse:.4f}",
                "MAPE%": f"{report.mape:.4f}",
                "DirAcc": f"{report.directional_accuracy:.4f}",
                "Pb10": f"{report.pinball_p10:.4f}" if not np.isnan(report.pinball_p10) else "—",
                "Pb50": f"{report.pinball_p50:.4f}",
                "Pb90": f"{report.pinball_p90:.4f}" if not np.isnan(report.pinball_p90) else "—",
                "cov": f"{report.coverage_p10_p90:.4f}" if report.coverage_p10_p90 is not None else "—",
                "width": f"{report.mean_interval_width:.4f}" if report.mean_interval_width is not None else "—",
                "fit_s": f"{report.total_fit_time_seconds:.2f}",
            })
        except Exception as exc:
            log.error("  %s failed: %s", name, exc)
            if args.verbose:
                traceback.print_exc()
            return 3

    # Print the results table.
    print()
    print(_format_table(table_rows))
    print()

    # Optional metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {
                "symbol": args.symbol,
                "models": args.models,
                "csv": args.csv,
                "use_yfinance": args.use_yfinance,
                "n_days_synthetic": args.n_days_synthetic,
                "seed": args.seed,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "train_window": args.train_window,
                "test_window": args.test_window,
                "step": args.step,
            },
            "results": {name: r.to_dict() for name, r in all_reports.items()},
            "stationarity": {
                col: {"adf_statistic": r.adf_statistic, "p_value": r.p_value,
                       "is_stationary": r.is_stationary}
                for col, r in ds.stationarity.items()
            },
            "source": ds.source,
            "n_samples": ds.n_samples,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Optional forecast-vs-actual plot.
    if args.forecast_plot:
        try:
            # Re-run the walk-forward to collect raw predictions for plotting.
            # We only do this if the user explicitly asked for the plot.
            for name in args.models:
                try:
                    from dataset import build_walk_forward_splits
                    kind = CANDIDATE_MODELS[name]
                    fc = build_forecaster(
                        kind, n_estimators=args.lgbm_n_estimators,
                        learning_rate=args.lgbm_learning_rate,
                        n_epochs=args.transformer_epochs, seq_len=args.transformer_seq_len,
                        prediction_length=args.chronos_prediction_length,
                        num_samples=args.chronos_num_samples,
                    )
                    preds = []
                    for train_df, test_df in build_walk_forward_splits(
                        ds.features_df, train_window=args.train_window,
                        test_window=args.test_window, step=args.step,
                    ):
                        fold_fc = build_forecaster(kind)
                        fold_fc.fit(train_df, target_column=ds.target_column)
                        y_pred, _, _ = fold_fc.predict(test_df)
                        preds.extend(y_pred.tolist())
                    all_predictions[name] = np.array(preds)
                except Exception as exc:
                    log.warning("Failed to collect predictions for %s: %s", name, exc)
                    all_predictions[name] = None
            _plot_forecast_vs_actual(
                ds.features_df, all_reports, all_predictions,
                Path(args.forecast_plot),
            )
            log.info("Saved forecast plot → %s", args.forecast_plot)
        except Exception as exc:
            log.warning("Failed to render forecast plot: %s", exc)

    # Summary line.
    if all_reports:
        best_name = min(all_reports.keys(), key=lambda k: all_reports[k].mae)
        best = all_reports[best_name]
        print(f"BEST_MODEL={best_name}")
        print(f"BEST_MAE={best.mae:.4f}")
        print(f"BEST_RMSE={best.rmse:.4f}")
        print(f"BEST_DIR_ACC={best.directional_accuracy:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
