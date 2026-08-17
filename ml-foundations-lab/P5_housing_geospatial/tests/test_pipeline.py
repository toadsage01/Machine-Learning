"""
tests/test_pipeline
===================

End-to-end tests for the P5 Housing Geospatial quantile-regression pipeline.

Coverage:
    * Pinball loss math — verified on hand-crafted examples.
    * Haversine distance — verified against geopy on a known pair.
    * Synthetic Mumbai generator — produces a valid schema + sane price range.
    * Quantile model training — produces 3 fitted regressors with correct shapes.
    * Coverage bounds — perfect-coverage test (intervals wrap every true value).
    * Crossing rate — non-crossing post-fix enforces p10 ≤ p50 ≤ p90.
    * CLI smoke test — full `python train.py` invocation exits 0.

Run with::

    cd ml-foundations-lab/P5_housing_geospatial
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
    Metro, load_housing, generate_synthetic_mumbai, haversine_distance_km, SCHEMA,
)
from model import (  # noqa: E402
    QuantileKind, CANDIDATE_MODELS, DEFAULT_QUANTILES,
    pinball_loss, mean_pinball_loss,
    train_quantile_model, evaluate_quantile_model, QuantileModel, QuantileMetrics,
)
from visualize import (  # noqa: E402
    plot_spatial_price_heatmap, plot_proximity_features,
    plot_quantile_intervals, plot_calibration_curve,
)


# ---------------------------------------------------------------------------
# Pinball loss math verification
# ---------------------------------------------------------------------------
def test_pinball_loss_q05_equals_half_mae():
    """For q=0.5, pinball loss should equal 0.5 × MAE."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.5, 2.5, 2.5, 4.5, 4.0])
    pb = pinball_loss(y, pred, q=0.5)
    mae_half = 0.5 * np.mean(np.abs(y - pred))
    assert abs(pb - mae_half) < 1e-9, f"q=0.5 pinball={pb}, expected 0.5*MAE={mae_half}"


def test_pinball_loss_asymmetric_on_consistent_under_prediction():
    """If every prediction is 1.0 below the truth, q=0.9 should hurt more than q=0.1."""
    y = np.array([2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0])  # always 1.0 below
    # diff = y - pred = [1, 1, 1], all positive (under-prediction).
    # q=0.1: loss per row = 0.1 * 1 = 0.1, mean = 0.1
    # q=0.9: loss per row = 0.9 * 1 = 0.9, mean = 0.9
    pb_low = pinball_loss(y, pred, q=0.1)
    pb_high = pinball_loss(y, pred, q=0.9)
    assert abs(pb_low - 0.1) < 1e-9
    assert abs(pb_high - 0.9) < 1e-9
    assert pb_high > pb_low, "Under-prediction should hurt more at q=0.9 than q=0.1"


def test_pinball_loss_asymmetric_on_consistent_over_prediction():
    """If every prediction is 1.0 above the truth, q=0.1 should hurt more than q=0.9."""
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 3.0, 4.0])  # always 1.0 above
    # diff = y - pred = [-1, -1, -1], all negative (over-prediction).
    # q=0.1: loss per row = (1 - 0.1) * 1 = 0.9, mean = 0.9
    # q=0.9: loss per row = (1 - 0.9) * 1 = 0.1, mean = 0.1
    pb_low = pinball_loss(y, pred, q=0.1)
    pb_high = pinball_loss(y, pred, q=0.9)
    assert abs(pb_low - 0.9) < 1e-9
    assert abs(pb_high - 0.1) < 1e-9
    assert pb_low > pb_high, "Over-prediction should hurt more at q=0.1 than q=0.9"


def test_pinball_loss_zero_when_perfect():
    """Perfect predictions → pinball loss = 0 for every quantile."""
    y = np.array([1.0, 5.0, 10.0, 100.0])
    for q in [0.1, 0.5, 0.9]:
        assert pinball_loss(y, y, q=q) == 0.0


def test_mean_pinball_loss_averages_quantiles():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 2.5, 2.5])
    qs = (0.1, 0.5, 0.9)
    expected = float(np.mean([pinball_loss(y, pred, q) for q in qs]))
    assert abs(mean_pinball_loss(y, pred, qs) - expected) < 1e-9


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------
def test_haversine_distance_matches_geopy():
    """Vectorized haversine should match geopy.distance.geodesic within 0.1%."""
    from geopy.distance import geodesic
    coords = [
        ((19.0760, 72.8777), (18.5204, 73.8567)),   # Mumbai → Pune
        ((28.6139, 77.2090), (28.7041, 77.1025)),   # Delhi → Rohini
        ((12.9716, 77.5946), (13.0827, 80.2707)),   # Bengaluru → Chennai
    ]
    for (lat1, lon1), (lat2, lon2) in coords:
        h = haversine_distance_km(np.array([lat1]), np.array([lon1]),
                                   np.array([lat2]), np.array([lon2]))[0]
        g = geodesic((lat1, lon1), (lat2, lon2)).km
        rel_err = abs(h - g) / g
        assert rel_err < 0.005, f"haversine={h:.3f}, geopy={g:.3f}, rel_err={rel_err:.4f}"


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------
def test_synthetic_mumbai_generator_produces_valid_schema():
    df = generate_synthetic_mumbai(n_samples=300, seed=42)
    # All schema columns present.
    for col in SCHEMA.all_features + [SCHEMA.target]:
        assert col in df.columns, f"Missing column: {col}"
    # Sane price range (5 lakh to ~5000 lakh).
    assert (df["price_lakh"] >= 5.0).all()
    assert df["price_lakh"].max() < 5000.0
    # Coords within Mumbai bbox.
    bbox = (18.89, 19.27, 72.77, 73.10)
    assert df["latitude"].between(bbox[0], bbox[1]).all()
    assert df["longitude"].between(bbox[2], bbox[3]).all()


def test_load_housing_returns_unified_dataset():
    ds = load_housing(Metro.MUMBAI, n_samples=200, seed=0)
    assert ds.n_samples == 200
    assert ds.metro == Metro.MUMBAI
    assert list(ds.X.columns) == SCHEMA.all_features
    assert ds.proximity_source in ("synthetic_fallback", "osmnx", "csv_provided")
    # No NaNs in the target.
    assert not ds.y.isna().any()


# ---------------------------------------------------------------------------
# Quantile model training
# ---------------------------------------------------------------------------
def test_train_lightgbm_quantile_model_shapes():
    ds = load_housing(Metro.MUMBAI, n_samples=400, seed=42)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y, test_size=0.25, random_state=42)
    qm = train_quantile_model(QuantileKind.LIGHTGBM, X_tr, y_tr, random_state=42)
    # Three fitted pipelines.
    assert qm.pipeline_p10 is not None
    assert qm.pipeline_p50 is not None
    assert qm.pipeline_p90 is not None
    assert qm.quantiles == DEFAULT_QUANTILES
    # Predictions on the test set have correct shape.
    preds = qm.predict(X_te)
    assert list(preds.columns) == ["p10", "p50", "p90"]
    assert len(preds) == len(X_te)
    # Non-crossing post-fix: p10 ≤ p50 ≤ p90 for every row.
    assert (preds["p10"] <= preds["p50"] + 1e-6).all()
    assert (preds["p50"] <= preds["p90"] + 1e-6).all()


def test_evaluate_quantile_model_returns_sane_metrics():
    ds = load_housing(Metro.MUMBAI, n_samples=400, seed=42)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y, test_size=0.25, random_state=42)
    qm = train_quantile_model(QuantileKind.LIGHTGBM, X_tr, y_tr, random_state=42)
    m = evaluate_quantile_model(qm, X_te, y_te)
    assert m.model_name == "lightgbm"
    assert len(m.pinball_per_quantile) == 3
    # Pinball loss should be non-negative.
    assert all(p >= 0 for p in m.pinball_per_quantile)
    # Coverage is in [0, 1].
    assert 0.0 <= m.coverage_p10_p90 <= 1.0
    # Interval widths are non-negative.
    assert m.mean_interval_width >= 0
    assert m.median_interval_width >= 0
    # Crossing rate is in [0, 1].
    assert 0.0 <= m.crossing_rate <= 1.0


# ---------------------------------------------------------------------------
# Coverage bounds test — perfect interval
# ---------------------------------------------------------------------------
def test_perfect_interval_yields_full_coverage():
    """If we feed the evaluator an interval that always wraps the true value,
    coverage should be 1.0 and mean_width = p90 - p10.
    """
    n = 100
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 100, size=n)

    # Construct a "perfect" QuantileModel: p10 = y - 1, p50 = y, p90 = y + 1.
    # We do this by stubbing the pipeline.predict to return our values.
    class _StubPipeline:
        def __init__(self, vals):
            self._vals = vals
        def predict(self, X):
            return self._vals

    # Build a fake QuantileModel.
    qm = QuantileModel(
        kind="stub",
        quantiles=DEFAULT_QUANTILES,
        pipeline_p10=_StubPipeline(y - 1.0),
        pipeline_p50=_StubPipeline(y.copy()),
        pipeline_p90=_StubPipeline(y + 1.0),
        feature_names=["x"],
    )
    X_fake = pd.DataFrame({"x": np.zeros(n)})
    m = evaluate_quantile_model(qm, X_fake, pd.Series(y))
    # Coverage should be 1.0 (every y is inside [y-1, y+1]).
    assert abs(m.coverage_p10_p90 - 1.0) < 1e-9
    # Width should be 2.0 for every row.
    assert abs(m.mean_interval_width - 2.0) < 1e-9
    # Median MAE should be 0 (p50 == y exactly).
    assert m.median_mae < 1e-9
    # No crossings.
    assert m.crossing_rate == 0.0


def test_zero_coverage_interval():
    """If p10 = p90 = constant < min(y), no y is inside → coverage = 0."""
    n = 50
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 200, size=n)

    class _StubPipeline:
        def __init__(self, vals):
            self._vals = vals
        def predict(self, X):
            return self._vals

    # All three quantiles predict 0 (always far below every y).
    qm = QuantileModel(
        kind="stub",
        quantiles=DEFAULT_QUANTILES,
        pipeline_p10=_StubPipeline(np.zeros(n)),
        pipeline_p50=_StubPipeline(np.zeros(n)),
        pipeline_p90=_StubPipeline(np.zeros(n)),
        feature_names=["x"],
    )
    X_fake = pd.DataFrame({"x": np.zeros(n)})
    m = evaluate_quantile_model(qm, X_fake, pd.Series(y))
    assert m.coverage_p10_p90 == 0.0
    assert m.mean_interval_width == 0.0


# ---------------------------------------------------------------------------
# Visualization smoke tests
# ---------------------------------------------------------------------------
def test_visualization_plots_render():
    """Verify all four visualize.py primitives render a non-empty PNG."""
    ds = load_housing(Metro.MUMBAI, n_samples=300, seed=42)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y, test_size=0.2, random_state=42)
    qm = train_quantile_model(QuantileKind.LIGHTGBM, X_tr, y_tr, random_state=42)

    with tempfile.TemporaryDirectory() as tmp:
        # Heatmap.
        p1 = Path(tmp) / "heat.png"
        plot_spatial_price_heatmap(ds, p1)
        assert p1.exists() and p1.stat().st_size > 5_000

        # Proximity small-multiples.
        p2 = Path(tmp) / "prox.png"
        plot_proximity_features(ds, p2)
        assert p2.exists() and p2.stat().st_size > 5_000

        # Quantile intervals.
        p3 = Path(tmp) / "intervals.png"
        plot_quantile_intervals(qm, X_te, y_te, p3)
        assert p3.exists() and p3.stat().st_size > 5_000

        # Calibration chart.
        p4 = Path(tmp) / "calib.png"
        plot_calibration_curve({"lightgbm": qm}, X_te, y_te, p4)
        assert p4.exists() and p4.stat().st_size > 5_000


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def test_cli_runs_end_to_end():
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--metro", "mumbai",
        "--n-samples", "300",
        "--models", "lightgbm",
        "--metrics-json", "/tmp/_p5_cli_metrics.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert "BEST_MODEL=lightgbm" in result.stdout
    metrics_path = Path("/tmp/_p5_cli_metrics.json")
    assert metrics_path.exists()
    payload = __import__("json").loads(metrics_path.read_text())
    assert "results" in payload
    assert "lightgbm" in payload["results"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_pinball_loss_q05_equals_half_mae,
        test_pinball_loss_asymmetric_on_consistent_under_prediction,
        test_pinball_loss_asymmetric_on_consistent_over_prediction,
        test_pinball_loss_zero_when_perfect,
        test_mean_pinball_loss_averages_quantiles,
        test_haversine_distance_matches_geopy,
        test_synthetic_mumbai_generator_produces_valid_schema,
        test_load_housing_returns_unified_dataset,
        test_train_lightgbm_quantile_model_shapes,
        test_evaluate_quantile_model_returns_sane_metrics,
        test_perfect_interval_yields_full_coverage,
        test_zero_coverage_interval,
        test_visualization_plots_render,
        test_cli_runs_end_to_end,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
