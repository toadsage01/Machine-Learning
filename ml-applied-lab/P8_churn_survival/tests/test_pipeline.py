"""
tests/test_pipeline
===================

End-to-end tests for the P8 Churn Survival pipeline.

Coverage:
    * Dataset — synthetic generator produces valid schema + ~40% churn rate.
    * Stratified split — train/test have matching churn rates.
    * Classifier — LogReg + RF produce sane metrics.
    * Kaplan-Meier — survival curve is non-increasing (mathematically guaranteed).
    * Cox PH — C-index in [0, 1], coefficients recover the ground-truth direction.
    * C-index — verified on a hand-crafted example.
    * Uplift — `compute_uplift` math verified on tiny example.
    * Expected-value policy — optimal threshold found, ROI > 0 for sensible inputs.
    * CLI smoke test — full `python train.py` invocation exits 0.

Run with::

    cd ml-applied-lab/P8_churn_survival
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
warnings.filterwarnings("ignore")

from dataset import (  # noqa: E402
    SCHEMA, load_churn_dataset, build_train_test_split, generate_synthetic_churn,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, ModelKind,
    train_classifier, fit_kaplan_meier, fit_cox_ph,
    compute_c_index, compute_uplift, expected_value_policy,
)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------
def test_synthetic_dataset_produces_valid_schema():
    df = generate_synthetic_churn(n_samples=300, seed=42)
    # All schema columns present.
    for col in SCHEMA.all_features:
        assert col in df.columns, f"Missing column: {col}"
    assert "churned" in df.columns
    assert "true_log_hazard" in df.columns  # generator diagnostic


def test_synthetic_churn_rate_is_reasonable():
    df = generate_synthetic_churn(n_samples=1000, seed=42)
    # Default parameters target ~30-45% observed churn rate.
    assert 0.25 <= df["churned"].mean() <= 0.55, (
        f"Churn rate {df['churned'].mean():.3f} outside expected range [0.25, 0.55]"
    )


def test_load_churn_dataset_returns_valid_object():
    ds = load_churn_dataset(n_samples=200, seed=0)
    assert ds.n_samples == 200
    assert "tenure_months" in ds.X.columns
    assert "churned" in ds.df.columns
    # Durations and events are integer-typed.
    assert ds.durations.dtype in (np.int64, np.int32, int)
    assert ds.events.dtype in (np.int64, np.int32, int)
    assert ds.events.isin([0, 1]).all()


def test_stratified_split_preserves_churn_rate():
    ds = load_churn_dataset(n_samples=500, seed=42)
    train, test = build_train_test_split(ds, test_size=0.2, random_state=42)
    # Churn rate should be very close between train and test.
    diff = abs(train.y_churned.mean() - test.y_churned.mean())
    assert diff < 0.05, f"Train/test churn rate gap too large: {diff:.3f}"


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------
def test_train_logreg_produces_sane_metrics():
    ds = load_churn_dataset(n_samples=500, seed=42)
    train, test = build_train_test_split(ds, test_size=0.2, random_state=42)
    pipe, m = train_classifier(
        ModelKind.LOGREG, train.X, train.y_churned, test.X, test.y_churned,
    )
    # Synthetic data is highly separable — LogReg should easily beat random.
    assert m.accuracy > 0.70, f"LogReg accuracy too low: {m.accuracy}"
    assert 0.5 <= m.roc_auc <= 1.0
    assert 0.0 <= m.brier_score <= 0.25
    assert len(m.confusion_matrix) == 2


def test_train_random_forest_produces_sane_metrics():
    ds = load_churn_dataset(n_samples=500, seed=42)
    train, test = build_train_test_split(ds, test_size=0.2, random_state=42)
    pipe, m = train_classifier(
        ModelKind.RANDOM_FOREST, train.X, train.y_churned, test.X, test.y_churned,
    )
    assert m.accuracy > 0.70
    assert 0.5 <= m.roc_auc <= 1.0


# ---------------------------------------------------------------------------
# Kaplan-Meier tests
# ---------------------------------------------------------------------------
def test_kaplan_meier_survival_is_non_increasing():
    """The KM survival function must be non-increasing — this is a
    mathematical invariant (S(t) = ∏(1 − d_i/n_i) where each factor is ≤ 1).
    """
    ds = load_churn_dataset(n_samples=500, seed=42)
    train, test = build_train_test_split(ds, test_size=0.2, random_state=42)
    kmf, m = fit_kaplan_meier(train.durations, train.events)
    survival = kmf.survival_function_.values.ravel()
    diffs = np.diff(survival)
    # All differences must be ≤ 0 (allowing a tiny float tolerance for
    # periods with no events — those have diff=0 exactly).
    assert (diffs <= 1e-9).all(), (
        f"KM survival increased at some t: max diff = {diffs.max():.6f}"
    )


def test_kaplan_meier_bounded_in_0_1():
    """Survival probabilities must be in [0, 1]."""
    ds = load_churn_dataset(n_samples=300, seed=42)
    train, _ = build_train_test_split(ds, test_size=0.2, random_state=42)
    kmf, m = fit_kaplan_meier(train.durations, train.events)
    survival = kmf.survival_function_.values.ravel()
    assert (survival >= 0.0).all() and (survival <= 1.0 + 1e-9).all()


def test_kaplan_meier_starts_at_1_and_decreases():
    """S(0) = 1 by definition; the curve must end ≤ S(0)."""
    ds = load_churn_dataset(n_samples=300, seed=42)
    train, _ = build_train_test_split(ds, test_size=0.2, random_state=42)
    kmf, m = fit_kaplan_meier(train.durations, train.events)
    survival = kmf.survival_function_.values.ravel()
    # KM curve starts at 1.0 by definition.
    assert abs(survival[0] - 1.0) < 1e-9, f"KM survival doesn't start at 1.0: {survival[0]}"
    # End is ≤ start (already covered by non-increasing test, but check explicitly).
    assert survival[-1] <= survival[0]


# ---------------------------------------------------------------------------
# Cox PH tests
# ---------------------------------------------------------------------------
def test_cox_ph_c_index_in_valid_range():
    """C-index must be in [0, 1]."""
    ds = load_churn_dataset(n_samples=500, seed=42)
    train, test = build_train_test_split(ds, test_size=0.2, random_state=42)
    cph, m = fit_cox_ph(
        train.X, train.durations, train.events,
        test.X, test.durations, test.events,
    )
    assert 0.0 <= m.c_index <= 1.0
    # On synthetic data with strong signal, Cox PH should beat random (>0.5).
    assert m.c_index > 0.55, f"Cox C-index too low: {m.c_index}"


def test_cox_ph_recovers_ground_truth_coefficient_directions():
    """The synthetic generator uses known β values — verify Cox recovers
    the DIRECTION of the largest coefficients.

    Ground-truth β (from dataset.py):
        contract_type_Two year:      -1.4   → should be NEGATIVE
        internet_service_Fiber optic: +0.6  → should be POSITIVE
        contract_type_One year:      -0.7   → should be NEGATIVE
    """
    ds = load_churn_dataset(n_samples=1500, seed=42)
    train, test = build_train_test_split(ds, test_size=0.2, random_state=42)
    cph, m = fit_cox_ph(
        train.X, train.durations, train.events,
        test.X, test.durations, test.events,
    )
    params = cph.params_
    # Two-year contract should reduce hazard (negative coefficient).
    if "contract_type_Two year" in params.index:
        assert params["contract_type_Two year"] < 0, (
            f"Expected Two year contract to reduce hazard, got β={params['contract_type_Two year']:.4f}"
        )
    # Fiber optic internet should increase hazard (positive coefficient).
    if "internet_service_Fiber optic" in params.index:
        assert params["internet_service_Fiber optic"] > 0, (
            f"Expected Fiber optic to increase hazard, got β={params['internet_service_Fiber optic']:.4f}"
        )


# ---------------------------------------------------------------------------
# C-index unit test on a hand-crafted example
# ---------------------------------------------------------------------------
def test_compute_c_index_perfect_ranking():
    """If predicted scores perfectly order durations, C-index should be 1.0."""
    durations = pd.Series([1, 2, 3, 4, 5])
    events = pd.Series([1, 1, 1, 1, 1])
    # Higher predicted score = longer survival.
    predicted_scores = np.array([1, 2, 3, 4, 5])
    c_idx = compute_c_index(predicted_scores, durations, events)
    assert abs(c_idx - 1.0) < 1e-6, f"Perfect ranking should give C=1.0, got {c_idx}"


def test_compute_c_index_random_ranking():
    """If predicted scores are constant, C-index should be 0.5 (random)."""
    durations = pd.Series([1, 2, 3, 4, 5])
    events = pd.Series([1, 1, 1, 1, 1])
    predicted_scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # all tied
    c_idx = compute_c_index(predicted_scores, durations, events)
    # lifelines returns 0.5 when all predictions are tied.
    assert abs(c_idx - 0.5) < 1e-6, f"Tied predictions should give C=0.5, got {c_idx}"


# ---------------------------------------------------------------------------
# Uplift tests
# ---------------------------------------------------------------------------
def test_compute_uplift_basic_math():
    """uplift = P(control) - P(treatment). Verify on a hand-crafted example."""
    p_control = np.array([0.6, 0.8, 0.3, 0.5])
    p_treatment = np.array([0.4, 0.9, 0.2, 0.5])
    uplift = compute_uplift(p_control, p_treatment)
    expected = np.array([0.2, -0.1, 0.1, 0.0])
    np.testing.assert_allclose(uplift, expected, atol=1e-9)


def test_expected_value_policy_finds_optimal_threshold():
    """The optimal threshold should target customers with positive EV."""
    # 5 customers: uplift, LTV per customer.
    uplift = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    ltv = np.array([100, 100, 100, 100, 100])
    offer_cost = 10.0

    # EV per customer = uplift * LTV - offer_cost = [40, 30, 20, 10, 0]
    # Cumulative ROI when targeting in descending EV order: [40, 70, 90, 100, 100]
    # Both index 3 and 4 give ROI=100. np.argmax returns the first occurrence (3).
    # The optimal targeting is 4 customers (the last one adds $0, neither
    # helping nor hurting).
    result = expected_value_policy(uplift, customer_ltv=ltv, offer_cost=offer_cost)
    assert result.optimal_threshold_idx in (3, 4)  # either is optimal
    assert abs(result.total_roi - 100.0) < 1e-6
    # Cumulative ROI should be monotonically increasing here (all EV ≥ 0).
    assert (np.diff(result.cumulative_roi) >= -1e-9).all()


def test_expected_value_policy_skips_negative_ev_customers():
    """When some customers have negative EV (offer cost > uplift*LTV),
    the optimal policy should NOT target them.
    """
    # 4 customers: uplift values such that:
    # EV = [40, -5, 30, -10] (sorted: [40, 30, -5, -10])
    uplift = np.array([0.5, 0.05, 0.4, 0.0])  # EV = [40, -5, 30, -10] for LTV=100, cost=10
    ltv = np.array([100, 100, 100, 100])
    result = expected_value_policy(uplift, ltv, offer_cost=10.0)
    # Optimal should target 2 customers (the ones with EV > 0).
    assert result.total_targeted == 2
    # Total ROI = 40 + 30 = 70
    assert abs(result.total_roi - 70.0) < 1e-6


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def test_cli_runs_end_to_end():
    """Full `python train.py` invocation should exit 0 + write JSON."""
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--n-samples", "400",
        "--metrics-json", "/tmp/_p8_cli_metrics.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"
    assert "BEST_CLASSIFIER=" in result.stdout
    assert "COX_C_INDEX=" in result.stdout
    assert "TOTAL_ROI=" in result.stdout
    assert Path("/tmp/_p8_cli_metrics.json").exists()
    import json
    payload = json.loads(Path("/tmp/_p8_cli_metrics.json").read_text())
    assert "classifier_metrics" in payload
    assert "survival_metrics" in payload
    assert "uplift" in payload


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_synthetic_dataset_produces_valid_schema,
        test_synthetic_churn_rate_is_reasonable,
        test_load_churn_dataset_returns_valid_object,
        test_stratified_split_preserves_churn_rate,
        test_train_logreg_produces_sane_metrics,
        test_train_random_forest_produces_sane_metrics,
        test_kaplan_meier_survival_is_non_increasing,
        test_kaplan_meier_bounded_in_0_1,
        test_kaplan_meier_starts_at_1_and_decreases,
        test_cox_ph_c_index_in_valid_range,
        test_cox_ph_recovers_ground_truth_coefficient_directions,
        test_compute_c_index_perfect_ranking,
        test_compute_c_index_random_ranking,
        test_compute_uplift_basic_math,
        test_expected_value_policy_finds_optimal_threshold,
        test_expected_value_policy_skips_negative_ev_customers,
        test_cli_runs_end_to_end,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
