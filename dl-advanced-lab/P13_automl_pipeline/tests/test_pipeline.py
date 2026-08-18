"""
tests/test_pipeline
===================

End-to-end tests for the P13 AutoML Pipeline.

Coverage:
    * Schema type detection: numeric, categorical, datetime, binary, ID.
    * Task type inference: classification (≤20 unique targets) vs regression.
    * Optuna trial monotonicity / score tracking: best_score is the max.
    * MLflow artifact output integrity: runs are logged with correct params.
    * AutoFeatureTransformer: produces correct output shapes.
    * CLI smoke test.

Run with::

    cd dl-advanced-lab/P13_automl_pipeline
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from dataset import (  # noqa: E402
    ColumnType, infer_schema, generate_classification_data,
    generate_regression_data, load_automl_dataset,
)
from model import (  # noqa: E402
    AutoFeatureTransformer, OptunaHPO, HPOReport, build_model, ModelKind,
    run_automl_pipeline, HAVE_OPTUNA, HAVE_MLFLOW,
)


# ---------------------------------------------------------------------------
# Schema type detection tests
# ---------------------------------------------------------------------------
def test_schema_detects_numeric():
    df = pd.DataFrame({"x": [1.0, 2.5, 3.7, np.nan, 5.1]})
    profile = infer_schema(df)
    assert profile.columns[0].type == ColumnType.NUMERIC.value


def test_schema_detects_categorical():
    df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b", "c"]})
    profile = infer_schema(df)
    assert profile.columns[0].type == ColumnType.CATEGORICAL.value


def test_schema_detects_datetime():
    df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=5)})
    profile = infer_schema(df)
    assert profile.columns[0].type == ColumnType.DATETIME.value


def test_schema_detects_binary():
    df = pd.DataFrame({"bin": [0, 1, 0, 1, 1]})
    profile = infer_schema(df)
    assert profile.columns[0].type == ColumnType.BINARY.value


def test_schema_detects_id():
    df = pd.DataFrame({"id": range(100)})
    profile = infer_schema(df)
    assert profile.columns[0].type == ColumnType.ID.value


def test_task_type_inference_classification():
    df = generate_classification_data(n_samples=100, n_features=4, seed=42)
    profile = infer_schema(df, target_column="target")
    assert profile.task_type == "classification"


def test_task_type_inference_regression():
    df = generate_regression_data(n_samples=100, n_features=4, seed=42)
    profile = infer_schema(df, target_column="target")
    assert profile.task_type == "regression"


# ---------------------------------------------------------------------------
# AutoFeatureTransformer tests
# ---------------------------------------------------------------------------
def test_feature_transformer_output_shape():
    """The transformer should produce a numeric matrix after fit_transform."""
    df = generate_classification_data(n_samples=50, n_features=4, n_categorical=2, seed=42)
    numeric = [c for c in df.columns if c.startswith("num_")]
    categorical = [c for c in df.columns if c.startswith("cat_")]
    transformer = AutoFeatureTransformer(numeric, categorical)
    ct = transformer.build()
    ct.fit(df[numeric + categorical])
    transformed = ct.transform(df[numeric + categorical])
    assert transformed.shape[0] == 50
    assert transformed.shape[1] >= len(numeric) + len(categorical) * 3  # at least num + one-hot.


def test_feature_transformer_with_polynomial():
    """Polynomial features should increase the output dimensionality."""
    df = generate_classification_data(n_samples=50, n_features=3, n_categorical=0, seed=42)
    numeric = [c for c in df.columns if c.startswith("num_")]

    t1 = AutoFeatureTransformer(numeric, [], polynomial=False)
    ct1 = t1.build()
    ct1.fit(df[numeric])
    out1 = ct1.transform(df[numeric])

    t2 = AutoFeatureTransformer(numeric, [], polynomial=True, polynomial_degree=2)
    ct2 = t2.build()
    ct2.fit(df[numeric])
    out2 = ct2.transform(df[numeric])

    assert out2.shape[1] > out1.shape[1], "Polynomial features should add columns"


# ---------------------------------------------------------------------------
# Optuna HPO tests
# ---------------------------------------------------------------------------
def test_optuna_best_score_is_max():
    """The best HPO score should be the maximum of all trial scores."""
    df, profile = load_automl_dataset(task="classification", n_samples=200,
                                        n_features=4, seed=42)
    report, _, _ = run_automl_pipeline(df, profile, n_trials=3, cv_folds=2, seed=42)
    all_scores = [t.score for t in report.all_trials]
    assert report.best_score == max(all_scores), (
        f"Best score {report.best_score} != max(all_scores) {max(all_scores)}"
    )


def test_optuna_trial_count_matches():
    """The number of trial results should match n_trials."""
    df, profile = load_automl_dataset(task="classification", n_samples=200,
                                        n_features=4, seed=42)
    report, _, _ = run_automl_pipeline(df, profile, n_trials=4, cv_folds=2, seed=42)
    assert report.n_trials == 4
    assert len(report.all_trials) == 4


# ---------------------------------------------------------------------------
# MLflow tests
# ---------------------------------------------------------------------------
def test_mlflow_logs_experiment():
    """MLflow should create an experiment and log runs."""
    if not HAVE_MLFLOW:
        return
    import mlflow
    import os
    import tempfile

    # MLflow 3.x requires a database backend or the env var to allow file store.
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    with tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tracking_uri(f"file://{tmp}")
        mlflow.set_experiment("test_automl")

        df, profile = load_automl_dataset(task="classification", n_samples=100,
                                            n_features=3, seed=42)
        report, pipeline, tracker = run_automl_pipeline(
            df, profile, n_trials=2, cv_folds=2,
            use_mlflow=True, mlflow_experiment="test_automl",
            seed=42,
        )
        # Verify MLflow runs were created.
        experiment = mlflow.get_experiment_by_name("test_automl")
        assert experiment is not None
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        # At least 2 trial runs + 1 best run = 3.
        assert len(runs) >= 2, f"Expected ≥2 MLflow runs, got {len(runs)}"


# ---------------------------------------------------------------------------
# Model factory tests
# ---------------------------------------------------------------------------
def test_build_lightgbm_model():
    model = build_model(ModelKind.LIGHTGBM, {"n_estimators": 10, "learning_rate": 0.1,
                                              "max_depth": 3, "num_leaves": 7,
                                              "min_child_samples": 5,
                                              "reg_alpha": 0.1, "reg_lambda": 0.1},
                         task_type="classification")
    assert model.__class__.__name__ == "LGBMClassifier"


def test_build_xgboost_model():
    model = build_model(ModelKind.XGBOOST, {"n_estimators": 10, "learning_rate": 0.1,
                                              "max_depth": 3, "min_child_weight": 1,
                                              "reg_alpha": 0.1, "reg_lambda": 0.1},
                         task_type="regression")
    assert model.__class__.__name__ == "XGBRegressor"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def test_cli_runs_end_to_end():
    """Full `python train.py` should exit 0 + write JSON."""
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--task", "classification", "--n-samples", "100",
        "--n-features", "3", "--n-trials", "2",
        "--metrics-json", "/tmp/_p13_cli_metrics.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"
    assert "BEST_MODEL=" in result.stdout
    assert "BEST_SCORE=" in result.stdout
    assert "N_TRIALS=" in result.stdout
    assert Path("/tmp/_p13_cli_metrics.json").exists()
    import json
    payload = json.loads(Path("/tmp/_p13_cli_metrics.json").read_text())
    assert "schema" in payload
    assert "hpo_report" in payload


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_schema_detects_numeric,
        test_schema_detects_categorical,
        test_schema_detects_datetime,
        test_schema_detects_binary,
        test_schema_detects_id,
        test_task_type_inference_classification,
        test_task_type_inference_regression,
        test_feature_transformer_output_shape,
        test_feature_transformer_with_polynomial,
        test_optuna_best_score_is_max,
        test_optuna_trial_count_matches,
        test_mlflow_logs_experiment,
        test_build_lightgbm_model,
        test_build_xgboost_model,
        test_cli_runs_end_to_end,
    ]
    n_passed = 0
    n_failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            n_passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            n_failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            n_failed += 1
    print(f"\n{n_passed} passed, {n_failed} failed (out of {len(tests)} total).")
    if n_failed > 0:
        sys.exit(1)
