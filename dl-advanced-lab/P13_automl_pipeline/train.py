#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P13_automl_pipeline — runs the automated ML pipeline
on input data, performs Optuna HPO trials, logs to MLflow, and exports
the best model pipeline.

Usage
-----
::

    # 1. Default: synthetic classification, 10 HPO trials
    python train.py

    # 2. Regression task
    python train.py --task regression

    # 3. Real CSV
    python train.py --csv /path/to/data.csv --target price

    # 4. More trials + MLflow + export model
    python train.py --n-trials 20 --use-mlflow --model-out models/best.joblib

Exit codes
----------
* 0  : completed.
* 1  : usage error.
* 2  : data loading failed.
* 3  : pipeline error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import load_automl_dataset  # noqa: E402
from model import run_automl_pipeline, HAVE_OPTUNA, HAVE_MLFLOW  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("automl_train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automl_train",
        description="P13 AutoML Pipeline — schema inference + HPO + MLflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", default=None, help="Path to input CSV.")
    parser.add_argument("--target", default=None, help="Target column name.")
    parser.add_argument("--task", choices=["classification", "regression"],
                        default="classification", help="Task type (default: classification).")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Synthetic data size (default: 500).")
    parser.add_argument("--n-features", type=int, default=8,
                        help="Number of numeric features (synthetic only, default: 8).")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of Optuna HPO trials (default: 10).")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Cross-validation folds (default: 3).")
    parser.add_argument("--use-mlflow", action="store_true",
                        help="Enable MLflow experiment tracking.")
    parser.add_argument("--mlflow-experiment", default="automl_pipeline",
                        help="MLflow experiment name (default: automl_pipeline).")
    parser.add_argument("--model-out", default=None,
                        help="Optional path to export the best model pipeline (joblib).")
    parser.add_argument("--metrics-json", default=None,
                        help="Optional path to dump HPO report as JSON.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)

    # Step 1 — load data + infer schema.
    try:
        log.info("Loading dataset ...")
        df, profile = load_automl_dataset(
            csv_path=args.csv,
            target_column=args.target,
            task=args.task,
            n_samples=args.n_samples,
            n_features=args.n_features,
            seed=args.seed,
        )
        log.info("  %d rows × %d cols (task=%s, target=%s)",
                 profile.n_rows, profile.n_cols, profile.task_type, profile.target_column)
        log.info("  %d numeric, %d categorical, %d datetime, %d binary",
                 profile.n_numeric, profile.n_categorical, profile.n_datetime, profile.n_binary)
    except Exception as exc:
        log.error("Data loading failed: %s", exc)
        return 2

    # Step 2 — run AutoML pipeline.
    try:
        log.info("Running AutoML pipeline (%d trials, %d-fold CV, MLflow=%s) ...",
                 args.n_trials, args.cv_folds, args.use_mlflow)
        report, pipeline, tracker = run_automl_pipeline(
            df, profile,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            use_mlflow=args.use_mlflow,
            mlflow_experiment=args.mlflow_experiment,
            model_artifact_path=args.model_out,
            seed=args.seed,
        )
        log.info("  Best trial #%d: %s (score=%.4f)",
                 report.best_trial, report.best_model, report.best_score)
        log.info("  All trial scores: %s",
                 [round(t.score, 4) for t in report.all_trials])
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 3

    # Step 3 — export best model (optional).
    if args.model_out and pipeline is not None:
        try:
            import joblib
            path = Path(args.model_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, path)
            log.info("  ✓ Best model exported → %s", path.resolve())
        except Exception as exc:
            log.warning("Model export failed: %s", exc)

    # Step 4 — metrics JSON (optional).
    if args.metrics_json:
        payload = {
            "config": {
                "csv": args.csv, "target": args.target,
                "task": args.task, "n_samples": args.n_samples,
                "n_features": args.n_features, "n_trials": args.n_trials,
                "cv_folds": args.cv_folds, "use_mlflow": args.use_mlflow,
                "seed": args.seed,
            },
            "schema": profile.to_dict(),
            "hpo_report": report.to_dict(),
            "mlflow_enabled": tracker is not None,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Summary.
    print(f"BEST_MODEL={report.best_model}")
    print(f"BEST_SCORE={report.best_score:.4f}")
    print(f"BEST_TRIAL={report.best_trial}")
    print(f"N_TRIALS={report.n_trials}")
    if args.use_mlflow:
        print(f"MLFLOW_EXPERIMENT={args.mlflow_experiment}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
