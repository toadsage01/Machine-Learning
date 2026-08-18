"""
tests/test_pipeline
===================

End-to-end tests for the P15 Experiment Kit.

Coverage:
    * CUPED variance reduction: σ²_cuped < σ²_raw (mathematically guaranteed).
    * mSPRT false positive control under null hypothesis (no false stops).
    * Bayesian posterior integration (Beta-Binomial conjugate correctness).
    * Frequentist test p-values in [0, 1].
    * Welch t-test detects large effects.
    * CLI smoke test.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from dataset import generate_ab_experiment, ExperimentConfig  # noqa: E402
from model import (  # noqa: E402
    FrequentistEngine, CUPADEngine, SequentialEngine, BayesianEngine,
    ExperimentKit,
)


# ---------------------------------------------------------------------------
# CUPED tests
# ---------------------------------------------------------------------------
def test_cuped_variance_is_reduced():
    """CUPED adjusted variance must be ≤ raw variance when covariate is
    correlated with the outcome."""
    rng = np.random.default_rng(42)
    n = 1000
    covariate = rng.normal(0, 1, n)
    # Outcome = 0.5 * covariate + noise.
    outcome = 0.5 * covariate + rng.normal(0, 0.5, n)

    adjusted, theta = CUPADEngine.adjust(outcome, covariate)

    raw_var = float(np.var(outcome, ddof=1))
    adj_var = float(np.var(adjusted, ddof=1))
    assert adj_var < raw_var, (
        f"CUPED did not reduce variance: raw={raw_var:.6f}, adjusted={adj_var:.6f}"
    )
    # Variance reduction should be substantial (covariate correlation ~0.7).
    reduction_pct = (1 - adj_var / raw_var) * 100
    assert reduction_pct > 10, f"Expected >10% variance reduction, got {reduction_pct:.2f}%"


def test_cuped_theta_sign_matches_correlation():
    """θ should be positive when covariate and outcome are positively correlated."""
    rng = np.random.default_rng(42)
    n = 500
    covariate = rng.normal(0, 1, n)
    outcome = 0.8 * covariate + rng.normal(0, 0.3, n)  # positive correlation.
    _, theta = CUPADEngine.adjust(outcome, covariate)
    assert theta > 0, f"Expected positive theta for positive correlation, got {theta}"


def test_cuped_preserves_mean():
    """CUPED adjustment should not change the mean of the outcome."""
    rng = np.random.default_rng(42)
    n = 1000
    covariate = rng.normal(5, 2, n)
    outcome = rng.normal(10, 3, n)
    adjusted, _ = CUPADEngine.adjust(outcome, covariate)
    raw_mean = float(np.mean(outcome))
    adj_mean = float(np.mean(adjusted))
    assert abs(raw_mean - adj_mean) < 0.01, (
        f"CUPED changed the mean: raw={raw_mean:.4f}, adjusted={adj_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# mSPRT tests
# ---------------------------------------------------------------------------
def test_msprt_no_false_positive_under_null():
    """Under H0 (no effect), mSPRT should NOT recommend stopping for effect
    at the first peek."""
    rng = np.random.default_rng(42)
    # Generate two groups from the same distribution (no effect).
    control = rng.binomial(1, 0.1, 1000).astype(float)
    treatment = rng.binomial(1, 0.1, 1000).astype(float)

    result = SequentialEngine.evaluate(control, treatment, peek_number=1, total_peeks=5)
    assert not result.should_stop, (
        f"mSPRT false positive: should_stop={result.should_stop} under H0"
    )
    assert result.recommendation == "continue", (
        f"Expected 'continue', got '{result.recommendation}'"
    )


def test_msprt_boundary_is_large_at_first_peek():
    """The O'Brien-Fleming boundary should be very large at early peeks
    (conservative — protects against false positives)."""
    rng = np.random.default_rng(42)
    control = rng.normal(0, 1, 100)
    treatment = rng.normal(0, 1, 100)
    result = SequentialEngine.evaluate(control, treatment, peek_number=1, total_peeks=5)
    assert result.boundary > 10, (
        f"Boundary at peek 1 should be >10 (conservative), got {result.boundary:.2f}"
    )


# ---------------------------------------------------------------------------
# Bayesian tests
# ---------------------------------------------------------------------------
def test_bayesian_posterior_means_match_data():
    """Posterior means should be close to the observed conversion rates."""
    rng = np.random.default_rng(42)
    control = rng.binomial(1, 0.10, 1000).astype(float)
    treatment = rng.binomial(1, 0.15, 1000).astype(float)

    result = BayesianEngine.evaluate(control, treatment, n_samples=50000)
    observed_c = float(np.mean(control))
    observed_t = float(np.mean(treatment))
    # Posterior mean (shrinkage toward prior Beta(1,1) → slight bias).
    assert abs(result.posterior_mean_control - observed_c) < 0.01, (
        f"Posterior mean control={result.posterior_mean_control:.4f} vs observed={observed_c:.4f}"
    )
    assert abs(result.posterior_mean_treatment - observed_t) < 0.01, (
        f"Posterior mean treatment={result.posterior_mean_treatment:.4f} vs observed={observed_t:.4f}"
    )


def test_bayesian_p_superiority_high_for_large_effect():
    """When treatment has a much higher conversion rate, P(superiority) should
    be close to 1.0."""
    rng = np.random.default_rng(42)
    control = rng.binomial(1, 0.05, 2000).astype(float)    # 5% conversion
    treatment = rng.binomial(1, 0.20, 2000).astype(float)   # 20% conversion

    result = BayesianEngine.evaluate(control, treatment, n_samples=50000)
    assert result.p_superiority > 0.999, (
        f"P(superiority)={result.p_superiority:.4f}, expected >0.999 for large effect"
    )


def test_bayesian_beta_binomial_conjugate_correctness():
    """Verify: posterior = Beta(α + s, β + n - s)."""
    rng = np.random.default_rng(42)
    control = np.array([1] * 50 + [0] * 950)  # 1000 trials, 50 successes.
    treatment = np.array([1] * 80 + [0] * 920)  # 1000 trials, 80 successes.

    result = BayesianEngine.evaluate(control, treatment, prior_alpha=1, prior_beta=1,
                                      n_samples=10000)
    # Expected: control posterior = Beta(1+50, 1+950) = Beta(51, 951).
    assert result.alpha_control == 51, f"Expected alpha_c=51, got {result.alpha_control}"
    assert result.beta_control == 951, f"Expected beta_c=951, got {result.beta_control}"
    assert result.alpha_treatment == 81, f"Expected alpha_t=81, got {result.alpha_treatment}"
    assert result.beta_treatment == 921, f"Expected beta_t=921, got {result.beta_treatment}"


# ---------------------------------------------------------------------------
# Frequentist tests
# ---------------------------------------------------------------------------
def test_welch_ttest_p_value_in_range():
    """p-value must be in [0, 1]."""
    rng = np.random.default_rng(42)
    control = rng.normal(0, 1, 100)
    treatment = rng.normal(0, 1, 100)
    result = FrequentistEngine.welch_ttest(control, treatment)
    assert 0 <= result.p_value <= 1


def test_welch_ttest_detects_large_effect():
    """A large effect (Cohen's d > 1) should produce p < 0.05."""
    rng = np.random.default_rng(42)
    control = rng.normal(0, 1, 200)
    treatment = rng.normal(1.0, 1, 200)  # 1 std-dev shift.
    result = FrequentistEngine.welch_ttest(control, treatment)
    assert result.p_value < 0.05, f"Expected p<0.05 for large effect, got {result.p_value}"
    assert result.significant


def test_chi_square_p_value_in_range():
    rng = np.random.default_rng(42)
    control = rng.binomial(1, 0.1, 500).astype(float)
    treatment = rng.binomial(1, 0.1, 500).astype(float)
    result = FrequentistEngine.chi_square(control, treatment)
    assert 0 <= result.p_value <= 1


# ---------------------------------------------------------------------------
# ExperimentKit integration test
# ---------------------------------------------------------------------------
def test_experiment_kit_run_all():
    """ExperimentKit.run_all() should return results from all 4 engines."""
    config = ExperimentConfig(n_users=500, true_lift=0.10, seed=42)
    data = generate_ab_experiment(config)
    kit = ExperimentKit(data)
    results = kit.run_all()
    assert "frequentist" in results
    assert "cuped" in results
    assert "sequential" in results
    assert "bayesian" in results
    assert "welch_ttest" in results["frequentist"]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def test_cli_runs_end_to_end():
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--n-users", "500", "--true-lift", "0.10",
        "--metrics-json", "/tmp/_p15_cli_metrics.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                            cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"
    assert "FREQUENTIST_P_VALUE=" in result.stdout
    assert "CUPED_VAR_REDUCTION_PCT=" in result.stdout
    assert "SEQUENTIAL_RECOMMENDATION=" in result.stdout
    assert "BAYES_P_SUPERIORITY=" in result.stdout
    assert Path("/tmp/_p15_cli_metrics.json").exists()
    import json
    payload = json.loads(Path("/tmp/_p15_cli_metrics.json").read_text())
    assert "frequentist" in payload
    assert "cuped" in payload


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_cuped_variance_is_reduced,
        test_cuped_theta_sign_matches_correlation,
        test_cuped_preserves_mean,
        test_msprt_no_false_positive_under_null,
        test_msprt_boundary_is_large_at_first_peek,
        test_bayesian_posterior_means_match_data,
        test_bayesian_p_superiority_high_for_large_effect,
        test_bayesian_beta_binomial_conjugate_correctness,
        test_welch_ttest_p_value_in_range,
        test_welch_ttest_detects_large_effect,
        test_chi_square_p_value_in_range,
        test_experiment_kit_run_all,
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
