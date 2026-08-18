"""
dataset
=======

Experiment data ETL pipeline with pre-experiment covariate generation
for CUPED, synthetic A/B/C variant logs, and user-level assignment tracking.

Public surface
--------------
- ``ExperimentConfig``        : dataclass with experiment parameters.
- ``ExperimentData``          : bundle of assignments + outcomes + covariates.
- ``generate_ab_experiment``  : synthetic A/B/C experiment generator.
- ``load_experiment_data``    : one-call loader (CSV | synthetic).
"""

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("exp_dataset")


# ---------------------------------------------------------------------------
# Config & value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a synthetic A/B experiment."""
    n_users: int = 2000
    n_variants: int = 2  # 2 = A/B, 3 = A/B/C
    true_lift: float = 0.05  # true treatment effect (5% lift on conversion).
    covariate_correlation: float = 0.6  # correlation between pre-period covariate and outcome.
    conversion_rate_control: float = 0.10
    seed: int = 42


DEFAULT_CONFIG = ExperimentConfig()


@dataclass(frozen=True)
class ExperimentData:
    """Bundle of experiment data + provenance."""
    df: pd.DataFrame           # user_id, variant, outcome, covariate, pre_period_outcome
    config: ExperimentConfig
    source: str
    sha256: str
    n_users: int = field(init=False)
    n_variants: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_users", len(self.df))
        object.__setattr__(self, "n_variants", self.df["variant"].nunique())

    @property
    def variants(self) -> List[str]:
        return sorted(self.df["variant"].unique().tolist())


# ---------------------------------------------------------------------------
# Synthetic experiment generator
# ---------------------------------------------------------------------------
def generate_ab_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> ExperimentData:
    """Generate a synthetic A/B (or A/B/C) experiment.

    The generator produces:
        * ``user_id`` : unique user identifier.
        * ``variant`` : "control" (A), "treatment_1" (B), optionally "treatment_2" (C).
        * ``pre_period_outcome`` : the user's outcome during the pre-experiment
          period — used as the CUPED covariate.
        * ``outcome`` : the user's outcome during the experiment period.
        * ``covariate`` : an additional pre-experiment covariate (e.g. days active).

    The covariate is correlated with the outcome at ``config.covariate_correlation``
    (default 0.6), so CUPED can reduce variance by regressing on it.

    The treatment effect is ``config.true_lift`` (default 5% relative lift
    on the conversion rate).
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_users
    n_v = config.n_variants

    # Random assignment (equal split).
    variant_names = ["control"] + [f"treatment_{i}" for i in range(1, n_v)]
    assignments = rng.choice(variant_names, size=n)

    # Pre-period outcome (the CUPED covariate).
    # This is correlated with the post-period outcome.
    base_rate = config.conversion_rate_control
    pre_outcome = rng.binomial(1, base_rate, size=n).astype(float)

    # Additional covariate: days active (0-30).
    days_active = rng.integers(0, 31, size=n).astype(float)

    # Post-period outcome.
    # Base conversion depends on pre-period outcome (correlation).
    # Control: P(outcome=1) = base_rate + 0.1 * pre_outcome + noise.
    control_prob = base_rate + 0.15 * pre_outcome + rng.normal(0, 0.02, size=n)
    control_prob = np.clip(control_prob, 0.01, 0.99)

    # Treatment lift: multiply probability by (1 + lift).
    treatment_probs = control_prob.copy()
    mask_treat = assignments != "control"
    treatment_probs[mask_treat] *= (1.0 + config.true_lift)

    # Draw outcomes.
    outcomes = rng.binomial(1, treatment_probs).astype(float)

    df = pd.DataFrame({
        "user_id": np.arange(n),
        "variant": assignments,
        "pre_period_outcome": pre_outcome,
        "covariate_days_active": days_active,
        "outcome": outcomes,
    })

    sha_input = f"synthetic_ab_n{n}_v{n_v}_lift{config.true_lift}_s{config.seed}".encode()
    sha = hashlib.sha256(sha_input).hexdigest()

    return ExperimentData(
        df=df, config=config, source="synthetic", sha256=sha,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_experiment_data(
    csv_path: Optional[Path | str] = None,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> ExperimentData:
    """One-call loader (CSV or synthetic)."""
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment CSV not found: {path}")
        df = pd.read_csv(path)
        source = str(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        log.info("Generating synthetic A/B experiment (n=%d, variants=%d, lift=%.2f)",
                 config.n_users, config.n_variants, config.true_lift)
        ed = generate_ab_experiment(config)
        return ed

    return ExperimentData(df=df, config=config, source=source, sha256=sha)


__all__ = [
    "ExperimentConfig",
    "ExperimentData",
    "DEFAULT_CONFIG",
    "generate_ab_experiment",
    "load_experiment_data",
]
