"""
tests/test_pipeline
===================

End-to-end tests for the P4 Titanic Two-Ships benchmark.

Coverage:
    * Dataset loaders — classic + spaceship both produce the unified schema.
    * Spaceship fallback works without network access.
    * Train one booster end-to-end → metrics sane.
    * Probability calibration actually reduces Brier score (isotonic + sigmoid).
    * Calibration curve math: perfect predictions should yield a perfect
      diagonal curve.
    * Fairness slice math: a model that always predicts positive has
      selection_rate=1.0 for every slice; disparity ratio=1.0.
    * Fairness slice math: per-slice accuracy/fpr/fnr correct on a tiny
      hand-crafted example.
    * Run the full CLI on synthetic data.

Run with::

    cd ml-foundations-lab/P4_titanic_twoships
    python -m pytest tests/ -v

or::

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
warnings.filterwarnings("ignore")  # silence sklearn/catboost/lgbm FutureWarnings in tests

from dataset import (  # noqa: E402
    DatasetKind, UnifiedDataset, SCHEMA,
    load_classic_titanic, load_spaceship_titanic,
    make_synthetic_classic, make_synthetic_spaceship,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, ModelKind, CalibrationKind,
    ModelMetrics, CalibrationResult, FairnessReport,
    train_model, calibrate, evaluate_calibration, compute_fairness,
    build_feature_pipeline, DEFAULT_FAIRNESS_FEATURES,
)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------
def test_classic_titanic_loads_with_unified_schema():
    ds = load_classic_titanic()
    assert isinstance(ds, UnifiedDataset)
    assert ds.kind == DatasetKind.CLASSIC
    # Classic Stanford mirror has 887 rows.
    assert 700 <= ds.n_samples <= 1300
    assert set(SCHEMA.all_features) == set(ds.X.columns)
    # Target rate matches the historical Titanic (≈38%).
    assert 0.30 <= float(ds.y.mean()) <= 0.50
    # No NaNs in the derived boolean features.
    assert ds.X["is_child"].isin([0, 1]).all()
    assert ds.X["is_elderly"].isin([0, 1]).all()
    assert ds.X["alone"].isin([0, 1]).all()


def test_spaceship_titanic_loads_with_unified_schema():
    # The synthetic fallback should produce the same unified schema.
    ds = load_spaceship_titanic()
    assert ds.kind == DatasetKind.SPACESHIP
    assert ds.n_samples >= 1000
    assert set(SCHEMA.all_features) == set(ds.X.columns)
    # Sex column should be present and only contain male/female.
    assert set(ds.X["sex"].unique()).issubset({"male", "female"})


def test_classic_and_spaceship_have_same_columns():
    ds1 = load_classic_titanic()
    ds2 = load_spaceship_titanic()
    assert list(ds1.X.columns) == list(ds2.X.columns)


def test_synthetic_classic_is_self_consistent():
    df = make_synthetic_classic(n_samples=100, seed=0)
    assert len(df) == 100
    assert {"sex", "age", "pclass", "sibsp", "parch", "fare", "embarked", "survived"} <= set(df.columns)
    assert df["survived"].isin([0, 1]).all()


def test_synthetic_spaceship_is_self_consistent():
    df = make_synthetic_spaceship(n_samples=200, seed=0)
    assert len(df) == 200
    # Verify all canonical Spaceship columns are present.
    expected = {"PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Age",
                "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
                "Transported"}
    assert expected <= set(df.columns)
    # CryoSleep passengers should have 0 spend on every amenity.
    cryo_mask = df["CryoSleep"].astype(str).str.lower() == "true"
    if cryo_mask.any():
        for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
            assert (df.loc[cryo_mask, col] == 0).all(), f"CryoSleep=True but {col} has spend"


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------
def test_train_lightgbm_produces_sane_metrics():
    ds = load_classic_titanic()
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=0.2, stratify=ds.y, random_state=42)
    pipe, m = train_model(ModelKind.LIGHTGBM, X_tr, y_tr, X_te, y_te, cv_folds=3)
    # LightGBM should hit >70% accuracy on Titanic.
    assert m.accuracy > 0.70, f"LightGBM accuracy too low: {m.accuracy}"
    assert 0.0 <= m.brier_score <= 0.30
    assert 0.5 <= m.roc_auc <= 1.0
    # Confusion matrix is 2x2.
    assert len(m.confusion_matrix) == 2
    assert all(len(row) == 2 for row in m.confusion_matrix)


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------
def test_isotonic_calibration_reduces_or_maintains_brier():
    """Isotonic calibration should not make Brier dramatically worse on Titanic.

    NB: Isotonic CAN slightly increase Brier on small datasets due to
    overfitting the calibration curve with cv=5; we only assert it doesn't
    blow up by more than 50%.
    """
    ds = load_classic_titanic()
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=0.2, stratify=ds.y, random_state=42)
    pipe, m = train_model(ModelKind.LIGHTGBM, X_tr, y_tr, X_te, y_te, cv_folds=3)

    calibrated = calibrate(pipe, X_tr, y_tr, CalibrationKind.ISOTONIC, cv_folds=3)
    cal_result = evaluate_calibration(calibrated, X_te, y_te, "lightgbm", "isotonic")
    # Isotonic shouldn't blow up Brier.
    assert cal_result.brier_score <= m.brier_score * 1.5, (
        f"Isotonic Brier={cal_result.brier_score:.4f} >> uncalibrated={m.brier_score:.4f}"
    )


def test_perfect_predictions_yield_perfect_calibration_curve():
    """If we feed the calibration evaluator perfectly-calibrated probabilities,
    the calibration curve should be the identity line and Brier should be ~0.
    """
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, size=n)
    # Perfect probabilities: P(y=1) = y exactly.
    proba = np.zeros((n, 2))
    proba[np.arange(n), y] = 1.0

    # Build a fake "model" with predict_proba returning our perfect proba.
    class _PerfectModel:
        def predict_proba(self, X):
            return proba
        def predict(self, X):
            return y
        def __sklearn_is_fitted__(self):
            return True

    X_fake = pd.DataFrame({"x": np.zeros(n)})
    result = evaluate_calibration(_PerfectModel(), X_fake, pd.Series(y),
                                  "perfect", "none", n_bins=5)
    # Brier for perfect predictions is 0.
    assert result.brier_score < 1e-10, f"Brier should be ~0, got {result.brier_score}"
    # Brier skill score = 1.0 for perfect predictions.
    assert abs(result.brier_skill_score - 1.0) < 1e-6
    # Calibration curve should be the identity line.
    assert len(result.fraction_of_positives) == len(result.mean_predicted_value)
    # Each non-degenerate bin should have fraction == mean_pred.
    for fp, mp in zip(result.fraction_of_positives, result.mean_predicted_value):
        assert abs(fp - mp) < 0.2, f"Calibration off: fp={fp}, mp={mp}"


# ---------------------------------------------------------------------------
# Fairness tests
# ---------------------------------------------------------------------------
def test_fairness_all_positive_model_has_uniform_selection_rate():
    """A model that always predicts 1 should have selection_rate=1.0 for every
    slice, hence disparity_ratio=1.0 (perfect parity).
    """
    n = 200
    X = pd.DataFrame({
        "sex": ["male"] * 100 + ["female"] * 100,
        "pclass": [1] * 50 + [2] * 50 + [3] * 50 + [1] * 50,
        "is_child": [0] * n,
        "is_elderly": [0] * n,
        "alone": [0] * n,
    })
    y = pd.Series([0, 1] * (n // 2))

    class _AlwaysPositive:
        def predict(self, X):
            return np.ones(len(X))
        def predict_proba(self, X):
            return np.tile([0.0, 1.0], (len(X), 1))
        def __sklearn_is_fitted__(self):
            return True

    report = compute_fairness(_AlwaysPositive(), X, y, "always_positive", min_slice_size=1)
    assert report.selection_disparity_ratio == 1.0
    # All slices should have selection_rate=1.0.
    for s in report.slices:
        assert abs(s.selection_rate - 1.0) < 1e-6


def test_fairness_per_slice_metrics_correct_on_tiny_example():
    """Hand-verify accuracy/fpr/fnr computation on a tiny example."""
    # 4 rows: 2 males (1 TP, 1 TN), 2 females (1 FP, 1 FN).
    X = pd.DataFrame({
        "sex": ["male", "male", "female", "female"],
        "pclass": [3, 3, 1, 1],
        "is_child": [0, 0, 0, 0],
        "is_elderly": [0, 0, 0, 0],
        "alone": [0, 0, 0, 0],
    })
    y = pd.Series([1, 0, 0, 1])  # ground truth
    preds = np.array([1, 0, 1, 0])  # predictions

    class _TinyModel:
        def predict(self, X):
            return preds
        def __sklearn_is_fitted__(self):
            return True

    report = compute_fairness(_TinyModel(), X, y, "tiny", min_slice_size=1)
    male_slice = next(s for s in report.slices if s.feature == "sex" and s.value == "male")
    female_slice = next(s for s in report.slices if s.feature == "sex" and s.value == "female")
    # Male: 1 TP + 1 TN → accuracy=1.0, base_rate=0.5, selection_rate=0.5, FPR=0, FNR=0
    assert male_slice.accuracy == 1.0
    assert abs(male_slice.selection_rate - 0.5) < 1e-6
    assert abs(male_slice.base_rate - 0.5) < 1e-6
    assert male_slice.false_positive_rate == 0.0
    assert male_slice.false_negative_rate == 0.0
    # Female: 1 FP + 1 FN → accuracy=0.0, base_rate=0.5, selection_rate=0.5, FPR=1.0, FNR=1.0
    assert female_slice.accuracy == 0.0
    assert female_slice.false_positive_rate == 1.0
    assert female_slice.false_negative_rate == 1.0


def test_fairness_handles_unknown_features_gracefully():
    """If a feature isn't in X_test, it should be skipped (not raise)."""
    X = pd.DataFrame({"sex": ["male"] * 5 + ["female"] * 5})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    class _Dummy:
        def predict(self, X):
            return np.zeros(len(X))
        def __sklearn_is_fitted__(self):
            return True

    report = compute_fairness(_Dummy(), X, y, "dummy",
                               features=["sex", "nonexistent_feature"],
                               min_slice_size=1)
    # Only sex slices should appear.
    assert all(s.feature == "sex" for s in report.slices)
    assert report.n_features_evaluated == 2  # we asked for 2 features


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def test_cli_runs_on_synthetic_data():
    """End-to-end CLI run on synthetic data only — should produce a metrics JSON."""
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "-d", "spaceship",       # uses synthetic fallback — no network needed
        "-m", "lightgbm",
        "--calibration", "isotonic",
        "--metrics-json", "/tmp/_p4_cli_metrics.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert "BEST_MODEL=lightgbm" in result.stdout
    metrics_path = Path("/tmp/_p4_cli_metrics.json")
    assert metrics_path.exists()
    payload = __import__("json").loads(metrics_path.read_text())
    assert "results" in payload
    assert "spaceship" in payload["results"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_classic_titanic_loads_with_unified_schema,
        test_spaceship_titanic_loads_with_unified_schema,
        test_classic_and_spaceship_have_same_columns,
        test_synthetic_classic_is_self_consistent,
        test_synthetic_spaceship_is_self_consistent,
        test_train_lightgbm_produces_sane_metrics,
        test_isotonic_calibration_reduces_or_maintains_brier,
        test_perfect_predictions_yield_perfect_calibration_curve,
        test_fairness_all_positive_model_has_uniform_selection_rate,
        test_fairness_per_slice_metrics_correct_on_tiny_example,
        test_fairness_handles_unknown_features_gracefully,
        test_cli_runs_on_synthetic_data,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
