#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P5_housing_geospatial — quantile-regression benchmark
with OSMnx geospatial enrichment.

Usage
-----
::

    # 1. Default: synthetic Mumbai housing, both quantile models
    python train.py

    # 2. With real OSMnx enrichment (requires network on first run)
    python train.py --use-osm

    # 3. Delhi / Bangalore
    python train.py --metro delhi
    python train.py --metro bangalore

    # 4. Save all artifacts
    python train.py \\
        --metrics-json metrics.json \\
        --heatmap assets/heatmap.png \\
        --proximity-plot assets/proximity.png \\
        --intervals-plot assets/intervals.png \\
        --calibration-plot assets/calibration.png

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
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset import (  # noqa: E402
    Metro, load_housing, HousingDataset, SCHEMA, PROXIMITY_POIS,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, QuantileKind, DEFAULT_QUANTILES,
    QuantileModel, QuantileMetrics,
    train_quantile_model, evaluate_quantile_model, pinball_loss,
)
from visualize import (  # noqa: E402
    plot_spatial_price_heatmap, plot_proximity_features,
    plot_quantile_intervals, plot_calibration_curve,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("housing_train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="housing_train",
        description="P5 Housing Geospatial — quantile regression with OSM features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # Default: synthetic Mumbai, both quantile models, p10/p50/p90
  python train.py

  # With real OSMnx enrichment (network required on first run)
  python train.py --use-osm

  # Save all artifacts
  python train.py --metrics-json metrics.json \\
      --heatmap assets/heatmap.png \\
      --intervals-plot assets/intervals.png \\
      --calibration-plot assets/calibration.png
""",
    )
    parser.add_argument(
        "--metro", choices=["mumbai", "delhi", "bangalore"],
        default="mumbai",
        help="Indian metro to benchmark (default: mumbai).",
    )
    parser.add_argument(
        "--models", "-m", nargs="+",
        choices=list(CANDIDATE_MODELS.keys()),
        default=list(CANDIDATE_MODELS.keys()),
        help="Subset of quantile models to evaluate (default: both).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1500,
        help="Synthetic dataset size (default: 1500).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random state (default: 42).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction held out for test (default: 0.20).",
    )
    parser.add_argument(
        "--use-osm", action="store_true",
        help="Attempt OSMnx enrichment (network required on first run).",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Optional path to a real housing CSV (must contain lat/lon + price_lakh).",
    )
    parser.add_argument(
        "--metrics-json", default=None,
        help="Optional path to dump all metrics as JSON.",
    )
    parser.add_argument(
        "--heatmap", default=None,
        help="Optional path to save a spatial price heatmap PNG.",
    )
    parser.add_argument(
        "--proximity-plot", default=None,
        help="Optional path to save the OSM proximity small-multiples PNG.",
    )
    parser.add_argument(
        "--intervals-plot", default=None,
        help="Optional path to save a quantile-intervals PNG (per model).",
    )
    parser.add_argument(
        "--calibration-plot", default=None,
        help="Optional path to save a calibration bar chart PNG.",
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
    headers = ["model", "pb_p10", "pb_p50", "pb_p90", "mean_pb", "mae", "rmse",
               "coverage_80", "mean_width", "cross_rate", "fit_s"]
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
        log.info("Loading %s housing data ...", args.metro)
        ds = load_housing(
            metro=Metro(args.metro),
            csv_path=args.csv,
            n_samples=args.n_samples,
            seed=args.seed,
            use_osm=args.use_osm,
        )
        log.info("  Loaded %d samples (source=%s, proximity=%s)",
                 ds.n_samples, ds.source, ds.proximity_source)
        log.info("  Target: mean=%.1f, median=%.1f, std=%.1f lakh INR",
                 ds.y.mean(), ds.y.median(), ds.y.std())
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 2

    # Step 2 — train/test split.
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=args.test_size, random_state=args.seed,
    )
    log.info("  Split: train=%d, test=%d", len(X_tr), len(X_te))

    # Step 3 — train each candidate quantile model.
    table_rows: List[Dict] = []
    all_models: Dict[str, QuantileModel] = {}
    all_metrics: Dict[str, QuantileMetrics] = {}
    for name in args.models:
        try:
            log.info("Training %s ...", name)
            qm = train_quantile_model(
                QuantileKind(name), X_tr, y_tr,
                quantiles=DEFAULT_QUANTILES, random_state=args.seed,
            )
            metrics = evaluate_quantile_model(qm, X_te, y_te)
            all_models[name] = qm
            all_metrics[name] = metrics
            table_rows.append({
                "model": name,
                "pb_p10": f"{metrics.pinball_per_quantile[0]:.3f}",
                "pb_p50": f"{metrics.pinball_per_quantile[1]:.3f}",
                "pb_p90": f"{metrics.pinball_per_quantile[2]:.3f}",
                "mean_pb": f"{metrics.mean_pinball:.3f}",
                "mae": f"{metrics.median_mae:.2f}",
                "rmse": f"{metrics.median_rmse:.2f}",
                "coverage_80": f"{metrics.coverage_p10_p90:.3f}",
                "mean_width": f"{metrics.mean_interval_width:.2f}",
                "cross_rate": f"{metrics.crossing_rate:.3f}",
                "fit_s": f"{metrics.fit_time_seconds:.2f}",
            })
            log.info("  %s — coverage=%.3f (target=0.80), mean_pb=%.3f, mae=%.2f, cross_rate=%.3f",
                     name, metrics.coverage_p10_p90, metrics.mean_pinball,
                     metrics.median_mae, metrics.crossing_rate)
        except Exception as exc:
            log.error("  %s failed: %s", name, exc)
            if args.verbose:
                traceback.print_exc()
            return 3

    # Print results.
    print()
    print(_format_table(table_rows))
    print()

    # Optional metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {
                "metro": args.metro,
                "models": args.models,
                "n_samples": args.n_samples,
                "seed": args.seed,
                "test_size": args.test_size,
                "use_osm": args.use_osm,
                "quantiles": list(DEFAULT_QUANTILES),
                "feature_names": SCHEMA.all_features,
            },
            "results": {name: m.to_dict() for name, m in all_metrics.items()},
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Optional spatial heatmap (one per dataset).
    if args.heatmap:
        try:
            plot_spatial_price_heatmap(ds, Path(args.heatmap))
            log.info("Saved spatial heatmap → %s", args.heatmap)
        except Exception as exc:
            log.warning("Failed to render heatmap: %s", exc)

    # Optional proximity small-multiples.
    if args.proximity_plot:
        try:
            plot_proximity_features(ds, Path(args.proximity_plot))
            log.info("Saved proximity small-multiples → %s", args.proximity_plot)
        except Exception as exc:
            log.warning("Failed to render proximity plot: %s", exc)

    # Optional quantile-intervals plot (one per model).
    if args.intervals_plot:
        for name, qm in all_models.items():
            try:
                plot_path = Path(args.intervals_plot).with_name(
                    Path(args.intervals_plot).stem + f"_{name}" + Path(args.intervals_plot).suffix
                )
                plot_quantile_intervals(qm, X_te, y_te, plot_path)
                log.info("Saved quantile intervals for %s → %s", name, plot_path)
            except Exception as exc:
                log.warning("Failed to render intervals plot for %s: %s", name, exc)

    # Optional calibration bar chart (all models on one chart).
    if args.calibration_plot:
        try:
            plot_calibration_curve(all_models, X_te, y_te, Path(args.calibration_plot))
            log.info("Saved calibration chart → %s", args.calibration_plot)
        except Exception as exc:
            log.warning("Failed to render calibration plot: %s", exc)

    # Summary line — best model = lowest mean pinball loss.
    if all_metrics:
        best_name = min(all_metrics.keys(), key=lambda k: all_metrics[k].mean_pinball)
        best = all_metrics[best_name]
        print(f"BEST_MODEL={best_name}")
        print(f"BEST_MEAN_PINBALL={best.mean_pinball:.4f}")
        print(f"BEST_COVERAGE_80={best.coverage_p10_p90:.4f}")
        print(f"BEST_MAE={best.median_mae:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
