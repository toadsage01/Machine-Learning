"""
model
=====

Statistical experimentation engine with four sub-engines:
  1. Frequentist: Welch t-test, Chi-Square, Mann-Whitney U.
  2. CUPED: variance reduction via covariate regression.
  3. Sequential: mSPRT with alpha-spending (peeking protection).
  4. Bayesian: Beta-Binomial conjugate updating + posterior probability.

Public surface
--------------
- ``FrequentistResult``       : p-value, CI, effect size.
- ``CUPEDResult``             : adjusted outcome, variance reduction %.
- ``SequentialResult``        : mSPRT statistic, stopping boundary, recommendation.
- ``BayesianResult``          : posterior Beta params, P(superiority), credible interval.
- ``FrequentistEngine``       : runs Welch t-test, Chi-Square, Mann-Whitney U.
- ``CUPADEngine``             : variance reduction via pre-period covariate.
- ``SequentialEngine``        : mSPRT with alpha-spending.
- ``BayesianEngine``          : Beta-Binomial conjugate updating.
- ``ExperimentKit``           : orchestrates all four engines.

Design notes
------------
1. **CUPED** (Controlled-Experiment Using Pre-Experiment Data) —
   adjusts the outcome by subtracting θ × (covariate - mean(covariate)),
   where θ = Cov(Y, X) / Var(X). This reduces variance without biasing
   the estimate, giving tighter CIs and higher power.

2. **mSPRT** (mixed Sequential Probability Ratio Test) — at each
   "peek" (interim look), the test computes the likelihood ratio under
   H0 vs H1 and compares it to an alpha-spending boundary. This allows
   valid early stopping without inflating the false-positive rate.

3. **Bayesian Beta-Binomial** — for binary outcomes, the conjugate
   prior is Beta(α, β). After observing s successes out of n trials,
   the posterior is Beta(α + s, β + n - s). We compute P(treatment > control)
   via Monte Carlo sampling from the two posteriors.

4. **All engines share the same input format** — two arrays of outcomes
   (control vs treatment), optionally with pre-period covariates.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass
class FrequentistResult:
    """Frequentist hypothesis test result."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float       # mean difference (treatment - control)
    ci_lower: float          # 95% CI lower
    ci_upper: float          # 95% CI upper
    significant: bool        # p < 0.05
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CUPEDResult:
    """CUPED variance reduction result."""
    theta: float                      # CUPED adjustment coefficient
    raw_variance: float                # Var(Y_raw)
    adjusted_variance: float           # Var(Y_adjusted)
    variance_reduction_pct: float      # % reduction
    raw_effect: float                  # mean(treatment) - mean(control) on raw
    adjusted_effect: float             # on CUPED-adjusted outcomes
    raw_ci_lower: float
    raw_ci_upper: float
    adjusted_ci_lower: float
    adjusted_ci_upper: float
    n_control: int
    n_treatment: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SequentialResult:
    """mSPRT sequential testing result."""
    peek_number: int
    statistic: float                  # mSPRT statistic (likelihood ratio)
    alpha_spent: float                # cumulative alpha spent
    boundary: float                   # stopping boundary
    should_stop: bool                 # True if statistic > boundary
    recommendation: str              # "continue" / "stop_for_effect" / "stop_for_futility"
    n_control: int
    n_treatment: int
    cumulative_p_value: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BayesianResult:
    """Bayesian Beta-Binomial result."""
    alpha_control: float
    beta_control: float
    alpha_treatment: float
    beta_treatment: float
    p_superiority: float              # P(treatment > control)
    posterior_mean_control: float
    posterior_mean_treatment: float
    credible_lower: float             # 95% credible interval for effect
    credible_upper: float
    n_control: int
    n_treatment: int
    rope_probability: float           # P(effect is in ROPE [-0.01, 0.01])

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Frequentist Engine
# ---------------------------------------------------------------------------
class FrequentistEngine:
    """Frequentist hypothesis testing: Welch t-test, Chi-Square, Mann-Whitney U."""

    @staticmethod
    def welch_ttest(control: np.ndarray, treatment: np.ndarray,
                    alpha: float = 0.05) -> FrequentistResult:
        """Welch's t-test (unequal variances)."""
        t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
        effect = float(np.mean(treatment) - np.mean(control))
        # 95% CI for the difference.
        se = math.sqrt(np.var(treatment, ddof=1) / len(treatment) +
                       np.var(control, ddof=1) / len(control))
        ci_lower = effect - 1.96 * se
        ci_upper = effect + 1.96 * se
        return FrequentistResult(
            test_name="welch_ttest",
            statistic=float(t_stat), p_value=float(p_val),
            effect_size=effect, ci_lower=ci_lower, ci_upper=ci_upper,
            significant=bool(p_val < alpha),
            n_control=len(control), n_treatment=len(treatment),
            mean_control=float(np.mean(control)),
            mean_treatment=float(np.mean(treatment)),
        )

    @staticmethod
    def chi_square(control: np.ndarray, treatment: np.ndarray,
                   alpha: float = 0.05) -> FrequentistResult:
        """Chi-Square test for binary outcomes (conversion rate comparison)."""
        # Build 2x2 contingency table.
        c_success = int(np.sum(control))
        c_fail = len(control) - c_success
        t_success = int(np.sum(treatment))
        t_fail = len(treatment) - t_success
        table = np.array([[c_success, c_fail], [t_success, t_fail]])
        chi2, p_val, _, _ = stats.chi2_contingency(table, correction=False)
        effect = float(np.mean(treatment) - np.mean(control))
        # CI for proportion difference.
        se = math.sqrt(np.mean(treatment) * (1 - np.mean(treatment)) / len(treatment) +
                       np.mean(control) * (1 - np.mean(control)) / len(control))
        return FrequentistResult(
            test_name="chi_square",
            statistic=float(chi2), p_value=float(p_val),
            effect_size=effect,
            ci_lower=effect - 1.96 * se, ci_upper=effect + 1.96 * se,
            significant=bool(p_val < alpha),
            n_control=len(control), n_treatment=len(treatment),
            mean_control=float(np.mean(control)),
            mean_treatment=float(np.mean(treatment)),
        )

    @staticmethod
    def mann_whitney(control: np.ndarray, treatment: np.ndarray,
                     alpha: float = 0.05) -> FrequentistResult:
        """Mann-Whitney U test (non-parametric)."""
        u_stat, p_val = stats.mannwhitneyu(treatment, control, alternative="two-sided")
        effect = float(np.mean(treatment) - np.mean(control))
        se = math.sqrt(np.var(treatment, ddof=1) / len(treatment) +
                       np.var(control, ddof=1) / len(control))
        return FrequentistResult(
            test_name="mann_whitney",
            statistic=float(u_stat), p_value=float(p_val),
            effect_size=effect,
            ci_lower=effect - 1.96 * se, ci_upper=effect + 1.96 * se,
            significant=bool(p_val < alpha),
            n_control=len(control), n_treatment=len(treatment),
            mean_control=float(np.mean(control)),
            mean_treatment=float(np.mean(treatment)),
        )


# ---------------------------------------------------------------------------
# 2. CUPED Engine
# ---------------------------------------------------------------------------
class CUPADEngine:
    """CUPED: variance reduction via pre-experiment covariate regression.

    CUPED adjusts the outcome Y:
        Y_adjusted = Y - θ * (X - mean(X))
    where θ = Cov(Y, X) / Var(X).

    This reduces Var(Y) without biasing the treatment effect estimate,
    giving tighter confidence intervals and higher statistical power.
    """

    @staticmethod
    def adjust(
        outcome: np.ndarray,
        covariate: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Apply CUPED adjustment to outcomes.

        Parameters
        ----------
        outcome : np.ndarray, shape (N,)
            Post-experiment outcomes.
        covariate : np.ndarray, shape (N,)
            Pre-experiment covariates (must be from the pre-period,
            unaffected by treatment).

        Returns
        -------
        (adjusted_outcome, theta)
            ``adjusted_outcome`` has the same shape as ``outcome``.
            ``theta`` is the CUPED coefficient.
        """
        y = np.asarray(outcome, dtype=float)
        x = np.asarray(covariate, dtype=float)
        # θ = Cov(Y, X) / Var(X)
        cov_yx = np.cov(y, x, ddof=1)[0, 1]
        var_x = np.var(x, ddof=1)
        theta = float(cov_yx / var_x) if var_x > 0 else 0.0
        # Y_adjusted = Y - θ * (X - mean(X))
        adjusted = y - theta * (x - np.mean(x))
        return adjusted, theta

    @staticmethod
    def evaluate(
        control_outcome: np.ndarray,
        treatment_outcome: np.ndarray,
        control_covariate: np.ndarray,
        treatment_covariate: np.ndarray,
    ) -> CUPEDResult:
        """Run full CUPED evaluation."""
        # Adjust both groups using a pooled theta (computed from all data).
        all_outcome = np.concatenate([control_outcome, treatment_outcome])
        all_covariate = np.concatenate([control_covariate, treatment_covariate])
        _, theta = CUPADEngine.adjust(all_outcome, all_covariate)

        # Adjust each group separately.
        c_adjusted, _ = CUPADEngine.adjust(control_outcome, control_covariate)
        t_adjusted, _ = CUPADEngine.adjust(treatment_outcome, treatment_covariate)

        # Variances.
        raw_var = np.var(all_outcome, ddof=1)
        adj_var = np.var(np.concatenate([c_adjusted, t_adjusted]), ddof=1)
        var_reduction_pct = float((1.0 - adj_var / raw_var) * 100.0) if raw_var > 0 else 0.0

        # Effects + CIs.
        raw_effect = float(np.mean(treatment_outcome) - np.mean(control_outcome))
        adj_effect = float(np.mean(t_adjusted) - np.mean(c_adjusted))

        raw_se = math.sqrt(np.var(treatment_outcome, ddof=1) / len(treatment_outcome) +
                           np.var(control_outcome, ddof=1) / len(control_outcome))
        adj_se = math.sqrt(np.var(t_adjusted, ddof=1) / len(t_adjusted) +
                           np.var(c_adjusted, ddof=1) / len(c_adjusted))

        return CUPEDResult(
            theta=theta,
            raw_variance=float(raw_var),
            adjusted_variance=float(adj_var),
            variance_reduction_pct=var_reduction_pct,
            raw_effect=raw_effect,
            adjusted_effect=adj_effect,
            raw_ci_lower=raw_effect - 1.96 * raw_se,
            raw_ci_upper=raw_effect + 1.96 * raw_se,
            adjusted_ci_lower=adj_effect - 1.96 * adj_se,
            adjusted_ci_upper=adj_effect + 1.96 * adj_se,
            n_control=len(control_outcome),
            n_treatment=len(treatment_outcome),
        )


# ---------------------------------------------------------------------------
# 3. Sequential Testing Engine (mSPRT)
# ---------------------------------------------------------------------------
class SequentialEngine:
    """mSPRT (mixed Sequential Probability Ratio Test) with alpha-spending.

    At each "peek" (interim look), the test computes the likelihood ratio
    under H0 (no effect) vs H1 (normal prior on effect). If the ratio
    exceeds the boundary (determined by the alpha-spending function),
    we stop and declare significance.

    The alpha-spending function follows the O'Brien-Fleming approach:
        alpha_spent(peek k of K) ≈ 2 * (1 - Φ(z_{α/2} * sqrt(K/k)))
    which spends very little alpha at early peeks (conservative).
    """

    @staticmethod
    def evaluate(
        control: np.ndarray,
        treatment: np.ndarray,
        peek_number: int = 1,
        total_peeks: int = 5,
        alpha: float = 0.05,
    ) -> SequentialResult:
        """Run mSPRT at a given peek.

        Parameters
        ----------
        control, treatment : np.ndarray
            Outcomes for the two groups at this peek.
        peek_number : int
            Which interim look this is (1-indexed).
        total_peeks : int
            Total planned peeks.
        alpha : float
            Family-wise error rate (default 0.05).

        Returns
        -------
        SequentialResult
        """
        n_c = len(control)
        n_t = len(treatment)
        n_total = n_c + n_t

        # Observed effect.
        mean_diff = float(np.mean(treatment) - np.mean(control))

        # Pooled standard error.
        pooled_var = (np.var(treatment, ddof=1) * n_t + np.var(control, ddof=1) * n_c) / max(n_total - 2, 1)
        se = math.sqrt(pooled_var * (1.0 / n_c + 1.0 / n_t)) if pooled_var > 0 else 1e-8

        # mSPRT statistic: likelihood ratio under H0 (effect=0) vs H1 (effect ~ N(0, τ²)).
        # For the normal mixture, the mSPRT statistic is:
        #   LR = sqrt(τ² / (τ² + se²)) * exp(0.5 * mean_diff² / (se² + se⁴/τ²))
        # We set τ² = se² (unit information prior).
        tau2 = se ** 2
        variance_sum = se ** 2 + tau2
        lr = math.sqrt(tau2 / variance_sum) * math.exp(0.5 * mean_diff ** 2 / variance_sum)
        lr = max(lr, 0.0)

        # Alpha-spending (O'Brien-Fleming).
        # Boundary: 1 / alpha_spent_at_this_peek
        # alpha_spent(k) ≈ 2 * (1 - Φ(z_{α/2} * sqrt(K/k)))
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        alpha_spent = 2.0 * (1.0 - stats.norm.cdf(z_alpha * math.sqrt(total_peeks / max(peek_number, 1))))
        alpha_spent = min(alpha_spent, alpha)

        # Boundary: the likelihood ratio must exceed 1/alpha_spent.
        boundary = 1.0 / max(alpha_spent, 1e-10)

        # Cumulative p-value (Bonferroni-adjusted).
        _, raw_p = stats.ttest_ind(treatment, control, equal_var=False)
        cumulative_p = float(raw_p)

        should_stop = bool(lr > boundary)
        if should_stop and mean_diff > 0:
            recommendation = "stop_for_effect"
        elif should_stop and mean_diff < 0:
            recommendation = "stop_for_effect"
        elif peek_number >= total_peeks:
            recommendation = "stop_for_futility" if not should_stop else "stop_for_effect"
        else:
            recommendation = "continue"

        return SequentialResult(
            peek_number=peek_number,
            statistic=float(lr),
            alpha_spent=float(alpha_spent),
            boundary=float(boundary),
            should_stop=should_stop,
            recommendation=recommendation,
            n_control=n_c,
            n_treatment=n_t,
            cumulative_p_value=cumulative_p,
        )


# ---------------------------------------------------------------------------
# 4. Bayesian Engine
# ---------------------------------------------------------------------------
class BayesianEngine:
    """Bayesian Beta-Binomial conjugate updating.

    For binary outcomes (conversion), the Beta-Binomial model is the
    conjugate prior:
        Prior: Beta(α, β)
        Likelihood: Binomial(n, p)
        Posterior: Beta(α + s, β + n - s)

    where s = successes, n = trials.

    P(treatment > control) is computed via Monte Carlo sampling from
    the two posteriors.
    """

    @staticmethod
    def evaluate(
        control: np.ndarray,
        treatment: np.ndarray,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_samples: int = 100000,
        rope: float = 0.01,
    ) -> BayesianResult:
        """Run Bayesian analysis.

        Parameters
        ----------
        control, treatment : np.ndarray
            Binary outcomes (0/1).
        prior_alpha, prior_beta : float
            Beta prior parameters (default: Beta(1,1) = Uniform).
        n_samples : int
            Monte Carlo samples for P(superiority).
        rope : float
            Region of Practical Equivalence (ROPE) half-width.
        """
        rng = np.random.default_rng(42)

        # Sufficient statistics.
        s_c = int(np.sum(control))
        n_c = len(control)
        s_t = int(np.sum(treatment))
        n_t = len(treatment)

        # Posterior parameters.
        alpha_c = prior_alpha + s_c
        beta_c = prior_beta + n_c - s_c
        alpha_t = prior_alpha + s_t
        beta_t = prior_beta + n_t - s_t

        # Monte Carlo: sample from both posteriors.
        samples_c = rng.beta(alpha_c, beta_c, size=n_samples)
        samples_t = rng.beta(alpha_t, beta_t, size=n_samples)

        # P(treatment > control).
        p_superiority = float(np.mean(samples_t > samples_c))

        # Posterior means.
        mean_c = float(alpha_c / (alpha_c + beta_c))
        mean_t = float(alpha_t / (alpha_t + beta_t))

        # Credible interval for the difference.
        diff_samples = samples_t - samples_c
        cred_lower = float(np.percentile(diff_samples, 2.5))
        cred_upper = float(np.percentile(diff_samples, 97.5))

        # ROPE probability: P(|effect| < rope).
        rope_prob = float(np.mean(np.abs(diff_samples) < rope))

        return BayesianResult(
            alpha_control=alpha_c, beta_control=beta_c,
            alpha_treatment=alpha_t, beta_treatment=beta_t,
            p_superiority=p_superiority,
            posterior_mean_control=mean_c,
            posterior_mean_treatment=mean_t,
            credible_lower=cred_lower,
            credible_upper=cred_upper,
            n_control=n_c, n_treatment=n_t,
            rope_probability=rope_prob,
        )


# ---------------------------------------------------------------------------
# 5. Experiment Kit (orchestrator)
# ---------------------------------------------------------------------------
class ExperimentKit:
    """Orchestrates all four engines on a single experiment."""

    def __init__(self, data: "ExperimentData"):
        from dataset import ExperimentData
        self.data: ExperimentData = data
        self.df = data.df
        self.control = self.df[self.df["variant"] == "control"]
        self.treatment = self.df[self.df["variant"] != "control"]

    def run_frequentist(self) -> Dict[str, FrequentistResult]:
        """Run all three frequentist tests."""
        c = self.control["outcome"].values
        t = self.treatment["outcome"].values
        return {
            "welch_ttest": FrequentistEngine.welch_ttest(c, t),
            "chi_square": FrequentistEngine.chi_square(c, t),
            "mann_whitney": FrequentistEngine.mann_whitney(c, t),
        }

    def run_cuped(self) -> CUPEDResult:
        """Run CUPED variance reduction."""
        c_out = self.control["outcome"].values
        t_out = self.treatment["outcome"].values
        c_cov = self.control["pre_period_outcome"].values
        t_cov = self.treatment["pre_period_outcome"].values
        return CUPADEngine.evaluate(c_out, t_out, c_cov, t_cov)

    def run_sequential(self, peek: int = 1, total_peeks: int = 5) -> SequentialResult:
        """Run mSPRT sequential test."""
        c = self.control["outcome"].values
        t = self.treatment["outcome"].values
        return SequentialEngine.evaluate(c, t, peek_number=peek, total_peeks=total_peeks)

    def run_bayesian(self, n_samples: int = 100000) -> BayesianResult:
        """Run Bayesian Beta-Binomial analysis."""
        c = self.control["outcome"].values
        t = self.treatment["outcome"].values
        return BayesianEngine.evaluate(c, t, n_samples=n_samples)

    def run_all(self) -> Dict[str, Any]:
        """Run all engines and return combined results."""
        return {
            "frequentist": {k: v.to_dict() for k, v in self.run_frequentist().items()},
            "cuped": self.run_cuped().to_dict(),
            "sequential": self.run_sequential().to_dict(),
            "bayesian": self.run_bayesian().to_dict(),
        }


__all__ = [
    "FrequentistResult",
    "CUPEDResult",
    "SequentialResult",
    "BayesianResult",
    "FrequentistEngine",
    "CUPADEngine",
    "SequentialEngine",
    "BayesianEngine",
    "ExperimentKit",
]
