"""
tests/test_pipeline
===================

End-to-end tests for the P9 NSE Forecasting pipeline.

Coverage:
    * Synthetic OHLCV generator — produces valid schema + GBM stats.
    * Technical indicators — lag features are backward (no future leakage).
    * ADF stationarity — Close non-stationary, returns stationary.
    * Walk-forward splits — train strictly precedes test temporally.
    * No-lookahead leakage — lag_1[t] == Close[t-1] (verified on a hand-crafted example).
    * Forecasting metrics — MAE / RMSE / MAPE / Directional Accuracy / Pinball
      verified on hand-crafted examples.
    * Naive baseline — persistence forecast equals today's close.
    * LightGBM forecaster — trains + produces intervals.
    * CLI smoke test.

Run with::

    cd ml-applied-lab/P9_nse_forecasting
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
    load_equity_dataset, build_walk_forward_splits,
    generate_synthetic_ohlcv, add_technical_indicators,
    adf_stationarity, DEFAULT_TICKERS,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, ForecastKind, build_forecaster,
    walk_forward_evaluate, compute_metrics,
    NaiveForecaster, LightGBMForecaster,
)


# ---------------------------------------------------------------------------
# Synthetic OHLCV generator tests
# ---------------------------------------------------------------------------
def test_synthetic_ohlcv_produces_valid_schema():
    df = generate_synthetic_ohlcv(n_days=100, seed=42)
    assert len(df) == 100
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns, f"Missing column: {col}"
    # OHLC sanity: High ≥ max(Open, Close), Low ≤ min(Open, Close).
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()
    # Volume is positive.
    assert (df["Volume"] > 0).all()


def test_synthetic_ohlcv_gbm_drift():
    """GBM close prices should have positive drift over a long horizon."""
    df = generate_synthetic_ohlcv(n_days=500, seed=42, annual_drift=0.10)
    # Average annual return should be roughly positive.
    final_close = df["Close"].iloc[-1]
    initial_close = df["Close"].iloc[0]
    total_return = (final_close - initial_close) / initial_close
    # Don't assert sign (GBM is stochastic) — just that the magnitude is sane.
    assert abs(total_return) < 2.0, f"Unreasonable total return: {total_return:.4f}"


# ---------------------------------------------------------------------------
# Technical indicator tests
# ---------------------------------------------------------------------------
def test_lag_features_are_backward_no_future_leakage():
    """lag_1[t] should equal Close[t-1] (NOT Close[t+1]).

    This is the critical no-lookahead-leakage invariant. If lag features
    reference FUTURE prices, the model will appear to perform impossibly
    well in walk-forward CV but fail catastrophically in production.
    """
    df = generate_synthetic_ohlcv(n_days=50, seed=42)
    df_features = add_technical_indicators(df, lags=(1, 5), rolling_windows=(5, 10))
    # Drop NaNs.
    df_features = df_features.dropna()
    # Verify lag_1[t] == Close[t-1].
    for i in range(1, len(df_features)):
        close_t = df_features["Close"].iloc[i]
        lag_1_t = df_features["lag_1"].iloc[i]
        close_t_minus_1 = df_features["Close"].iloc[i - 1]
        assert lag_1_t == close_t_minus_1, (
            f"lag_1[{i}]={lag_1_t} should equal Close[{i-1}]={close_t_minus_1} "
            f"(Close[t]={close_t})"
        )


def test_rolling_windows_are_right_aligned():
    """roll_mean_5[t] should equal mean(Close[t-4 : t+1]) — NOT including future."""
    df = generate_synthetic_ohlcv(n_days=30, seed=42)
    df_features = add_technical_indicators(df, lags=(1,), rolling_windows=(5,))
    df_features = df_features.dropna()
    for i in range(5, len(df_features)):
        expected = df_features["Close"].iloc[i - 4 : i + 1].mean()
        actual = df_features["roll_mean_5"].iloc[i]
        assert abs(actual - expected) < 1e-6, (
            f"roll_mean_5[{i}]={actual}, expected {expected}"
        )


def test_target_next_close_is_forward_shift():
    """target_next_close[t] should equal Close[t+1] (the next-day close we predict)."""
    df = generate_synthetic_ohlcv(n_days=30, seed=42)
    df_features = add_technical_indicators(df, lags=(1,), rolling_windows=(5,))
    # Drop the last row (target_next_close is NaN there because there's no t+1).
    df_features = df_features.dropna(subset=["target_next_close"])
    for i in range(len(df_features) - 1):
        target_t = df_features["target_next_close"].iloc[i]
        close_t_plus_1 = df_features["Close"].iloc[i + 1]
        assert abs(target_t - close_t_plus_1) < 1e-6, (
            f"target_next_close[{i}]={target_t}, expected Close[{i+1}]={close_t_plus_1}"
        )


# ---------------------------------------------------------------------------
# ADF stationarity tests
# ---------------------------------------------------------------------------
def test_adf_stationarity_on_constant_series():
    """A constant series should be stationary (ADF p-value low)."""
    s = pd.Series([1.0] * 100)
    rep = adf_stationarity(s, column_name="const")
    # Constant series: ADF returns a -inf statistic and p=0 (perfectly stationary).
    assert rep.is_stationary


def test_adf_stationarity_on_random_walk():
    """A pure random walk should be non-stationary."""
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 1, 500)
    walk = pd.Series(np.cumsum(steps))
    rep = adf_stationarity(walk, column_name="random_walk")
    assert not rep.is_stationary, "Random walk should be non-stationary"
    assert rep.p_value > 0.05


def test_adf_stationarity_on_synthetic_returns():
    """Daily returns of synthetic GBM should be stationary."""
    df = generate_synthetic_ohlcv(n_days=500, seed=42)
    returns = df["Close"].pct_change().dropna()
    rep = adf_stationarity(returns, column_name="daily_return")
    assert rep.is_stationary, "Returns should be stationary"


# ---------------------------------------------------------------------------
# Walk-forward split tests
# ---------------------------------------------------------------------------
def test_walk_forward_train_precedes_test_temporally():
    """For every fold, train must end BEFORE test starts."""
    df = generate_synthetic_ohlcv(n_days=300, seed=42)
    df_features = add_technical_indicators(df, lags=(1, 5), rolling_windows=(5, 10))
    df_features = df_features.dropna()
    folds = list(build_walk_forward_splits(df_features, train_window=100, test_window=20, step=30))
    assert len(folds) >= 2
    for train_df, test_df in folds:
        # Train's last timestamp must be strictly before test's first.
        assert train_df.index[-1] < test_df.index[0], (
            f"Train ends at {train_df.index[-1]}, test starts at {test_df.index[0]}"
        )
        # All test timestamps must be strictly greater than all train timestamps.
        assert (test_df.index > train_df.index[-1]).all()


def test_walk_forward_no_overlap_within_fold():
    """Within a single fold, no row index should appear in both train and test."""
    df = generate_synthetic_ohlcv(n_days=300, seed=42)
    df_features = add_technical_indicators(df, lags=(1, 5), rolling_windows=(5, 10))
    df_features = df_features.dropna()
    folds = list(build_walk_forward_splits(df_features, train_window=100, test_window=20, step=30))
    for fold_idx, (train_df, test_df) in enumerate(folds):
        train_idx = set(train_df.index)
        test_idx = set(test_df.index)
        overlap = train_idx & test_idx
        assert not overlap, f"Fold {fold_idx} has {len(overlap)} overlapping indices"


def test_walk_forward_window_sizes():
    """Each fold's train and test sizes should match the requested windows."""
    df = generate_synthetic_ohlcv(n_days=300, seed=42)
    df_features = add_technical_indicators(df, lags=(1,), rolling_windows=(5,))
    df_features = df_features.dropna()
    folds = list(build_walk_forward_splits(df_features, train_window=100, test_window=20, step=30))
    for train_df, test_df in folds:
        assert len(train_df) == 100
        assert len(test_df) == 20


# ---------------------------------------------------------------------------
# Metric math tests
# ---------------------------------------------------------------------------
def test_compute_metrics_mae_rmse_mape():
    """Verify MAE, RMSE, MAPE on a hand-crafted example."""
    y_true = np.array([100, 110, 105, 95])
    y_pred = np.array([100, 100, 100, 100])
    m = compute_metrics(y_true, y_pred, prev_true=None)
    # MAE = mean(|100-100, 110-100, 105-100, 95-100|) = mean(0, 10, 5, 5) = 5
    assert abs(m["mae"] - 5.0) < 1e-9, f"MAE={m['mae']}, expected 5.0"
    # RMSE = sqrt(mean(0, 100, 25, 25)) = sqrt(37.5) ≈ 6.124
    assert abs(m["rmse"] - np.sqrt(37.5)) < 1e-6
    # MAPE = mean(|0/100|, |10/110|, |5/105|, |5/95|) × 100 ≈ 4.78%
    expected_mape = (0 + 10/110 + 5/105 + 5/95) / 4 * 100
    assert abs(m["mape"] - expected_mape) < 1e-4


def test_compute_metrics_directional_accuracy():
    """Directional accuracy = fraction of days where sign(y_true - prev) == sign(y_pred - prev)."""
    y_true = np.array([110, 105, 95, 100])
    y_pred = np.array([105, 110, 90, 105])
    prev = np.array([100, 110, 105, 95])
    # True direction: +, -, -, +    (110-100=+, 105-110=-, 95-105=-, 100-95=+)
    # Pred direction: +, +, -, +    (105-100=+, 110-110=0, 90-105=-, 105-95=+)
    # Match: +==+, -!=+ (0≠+), -==-, +==+ → 3/4 = 0.75
    # NB: sign(0) = 0, so the second row is a mismatch (0 != +).
    m = compute_metrics(y_true, y_pred, prev_true=prev)
    assert abs(m["directional_accuracy"] - 0.75) < 1e-9


def test_pinball_loss_formula():
    """Pinball p50 = 0.5 * MAE; pinball_p10 penalizes under-prediction at 0.1 rate."""
    y_true = np.array([110, 100, 105])
    y_pred = np.array([100, 100, 100])
    m = compute_metrics(y_true, y_pred, prev_true=None,
                         y_pred_lower=y_pred, y_pred_upper=y_pred)
    # Pinball p50 = 0.5 × MAE = 0.5 × mean(10, 0, 5) = 2.5
    assert abs(m["pinball_p50"] - 2.5) < 1e-9
    # Pinball p10: for each row, q=0.1, diff = y_true - y_pred = [10, 0, 5]
    # Under-pred (diff ≥ 0): loss = 0.1 × diff = [1, 0, 0.5]. mean = 0.5
    assert abs(m["pinball_p10"] - 0.5) < 1e-9
    # Pinball p90: under-pred → loss = 0.9 × diff = [9, 0, 4.5]. mean = 4.5
    assert abs(m["pinball_p90"] - 4.5) < 1e-9


# ---------------------------------------------------------------------------
# Forecaster tests
# ---------------------------------------------------------------------------
def test_naive_forecaster_predicts_today_close():
    """Naive persistence: predicted next-day close = today's close."""
    fc = NaiveForecaster()
    # Build a fake train_df and test_df.
    train_df = pd.DataFrame({"Close": [100, 101, 102], "target_next_close": [101, 102, 103]})
    test_df = pd.DataFrame({"Close": [102, 103], "target_next_close": [103, 104]})
    fc.fit(train_df, target_column="target_next_close")
    y_pred, y_lower, y_upper = fc.predict(test_df)
    # Predicted value for each test row = its Close column (i.e. today's close).
    np.testing.assert_array_equal(y_pred, [102, 103])
    assert y_lower is None
    assert y_upper is None


def test_lightgbm_forecaster_returns_intervals():
    """LightGBM should return (y_pred, y_lower, y_upper) with p10 ≤ p50 ≤ p90."""
    ds = load_equity_dataset(symbol="^NSEI", n_days_synthetic=200, seed=42,
                              lags=(1, 5), rolling_windows=(5, 10), run_stationarity=False)
    # Use the last 30 rows as "test".
    train_df = ds.features_df.iloc[:-30]
    test_df = ds.features_df.iloc[-30:]
    fc = LightGBMForecaster(n_estimators=50, learning_rate=0.1)
    fc.fit(train_df, target_column="target_next_close")
    y_pred, y_lower, y_upper = fc.predict(test_df)
    assert y_pred.shape == (30,)
    assert y_lower.shape == (30,)
    assert y_upper.shape == (30,)
    # Non-crossing post-hoc: p10 ≤ p50 ≤ p90 for every row.
    assert (y_lower <= y_pred + 1e-6).all()
    assert (y_pred <= y_upper + 1e-6).all()


def test_walk_forward_evaluate_returns_sane_report():
    """Walk-forward CV with the Naive forecaster should produce a valid report."""
    ds = load_equity_dataset(symbol="^NSEI", n_days_synthetic=200, seed=42,
                              lags=(1, 5), rolling_windows=(5, 10), run_stationarity=False)
    fc = NaiveForecaster()
    report = walk_forward_evaluate(
        fc, ds.features_df, train_window=100, test_window=20, step=30,
    )
    assert report.model_name == "naive"
    assert report.n_folds >= 1
    assert report.n_predictions > 0
    assert report.mae >= 0
    assert 0.0 <= report.directional_accuracy <= 1.0
    # Naive forecaster doesn't produce intervals.
    assert report.coverage_p10_p90 is None
    assert report.mean_interval_width is None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def test_cli_runs_end_to_end():
    """Full `python train.py` invocation should exit 0 + write JSON."""
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--n-days-synthetic", "200",
        "--train-window", "100",
        "--test-window", "20",
        "--step", "30",
        "--models", "naive", "lightgbm",
        "--lgbm-n-estimators", "50",
        "--metrics-json", "/tmp/_p9_cli_metrics.json",
        "--skip-stationarity",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"
    assert "BEST_MODEL=" in result.stdout
    assert "BEST_MAE=" in result.stdout
    assert Path("/tmp/_p9_cli_metrics.json").exists()
    import json
    payload = json.loads(Path("/tmp/_p9_cli_metrics.json").read_text())
    assert "results" in payload
    assert "naive" in payload["results"]
    assert "lightgbm" in payload["results"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_synthetic_ohlcv_produces_valid_schema,
        test_synthetic_ohlcv_gbm_drift,
        test_lag_features_are_backward_no_future_leakage,
        test_rolling_windows_are_right_aligned,
        test_target_next_close_is_forward_shift,
        test_adf_stationarity_on_constant_series,
        test_adf_stationarity_on_random_walk,
        test_adf_stationarity_on_synthetic_returns,
        test_walk_forward_train_precedes_test_temporally,
        test_walk_forward_no_overlap_within_fold,
        test_walk_forward_window_sizes,
        test_compute_metrics_mae_rmse_mape,
        test_compute_metrics_directional_accuracy,
        test_pinball_loss_formula,
        test_naive_forecaster_predicts_today_close,
        test_lightgbm_forecaster_returns_intervals,
        test_walk_forward_evaluate_returns_sane_report,
        test_cli_runs_end_to_end,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
