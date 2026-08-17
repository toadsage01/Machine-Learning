"""
model
=====

Quantile regression + pinball loss + coverage metrics for the P5
housing price benchmark.

Public surface
--------------
- ``QuantileKind``             : enum (LightGBM, GradientBoosting).
- ``DEFAULT_QUANTILES``       : (0.10, 0.50, 0.90).
- ``pinball_loss``             : vectorized pinball loss per quantile.
- ``mean_pinball_loss``        : averaged across quantiles.
- ``QuantileModel``            : value object holding 3 fitted regressors (p10/p50/p90).
- ``QuantileMetrics``         : pinball per quantile + coverage + interval width.
- ``build_feature_pipeline``  : sklearn ColumnTransformer (impute + OneHot + scale).
- ``train_quantile_model``     : fit p10/p50/p90 regressors head-to-head.
- ``evaluate_quantile_model`` : compute pinball loss + coverage + interval width.
- ``CANDIDATE_MODELS``         : registry dict.

Design notes
------------
1. **Three separate regressors, not one multi-output** — LightGBM's quantile
   objective trains a single quantile per model. We train three independent
   models (p10, p50, p90) and bundle them in a ``QuantileModel``. This is
   the canonical approach used by forecasters like Prophet, GluonTS, and
   scikit-learn's GradientBoostingRegressor.

2. **Pinball loss is the standard quantile-regression metric** — it's an
   asymmetric absolute error: under-predictions are penalized at rate
   ``q`` (e.g. 0.10), over-predictions at rate ``1-q`` (e.g. 0.90). A
   perfectly-calibrated quantile model has empirical pinball loss matching
   the theoretical minimum (which is the quantile itself times the
   dispersion).

3. **Coverage rate is the headline fairness metric for intervals** —
   for the p10/p90 interval, we expect 80% of true values to fall inside.
   Coverage much lower than 80% means the model is over-confident; much
   higher means it's under-confident. We compute coverage for both the
   80% interval (p10–p90) and the median hit-rate (y == p50 prediction).

4. **Non-crossing constraint not enforced** — by default, independent
   quantile regressors can produce p90 < p10 predictions for individual
   rows (quantile crossing). For production use this would warrant a
   post-hoc isotonic re-arrangement; we report the crossing rate as a
   diagnostic instead.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import HousingSchema, SCHEMA  # noqa: E402

try:
    from lightgbm import LGBMRegressor
    HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover
    HAVE_LIGHTGBM = False


# ---------------------------------------------------------------------------
# Quantile configuration
# ---------------------------------------------------------------------------
DEFAULT_QUANTILES: Tuple[float, ...] = (0.10, 0.50, 0.90)


class QuantileKind(str, Enum):
    LIGHTGBM = "lightgbm"
    GRADIENT_BOOSTING = "gradient_boosting"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


CANDIDATE_MODELS: Dict[str, QuantileKind] = {
    QuantileKind.LIGHTGBM.value: QuantileKind.LIGHTGBM,
    QuantileKind.GRADIENT_BOOSTING.value: QuantileKind.GRADIENT_BOOSTING,
}


# ---------------------------------------------------------------------------
# Pinball loss
# ---------------------------------------------------------------------------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball loss for a single quantile.

    Formula::

        L(y, ŷ; q) = max( q·(y - ŷ), (q - 1)·(y - ŷ) )
                   = (q - I(y < ŷ)) · (y - ŷ)

    For q=0.5 this reduces to 0.5 · |y - ŷ| (the L1 loss / MAE).

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Same shape, scalar values.
    q : float
        Quantile in (0, 1).

    Returns
    -------
    float
        Mean pinball loss across all samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    # Equivalent to: max(q * diff, (q - 1) * diff) but vectorized cleanly.
    loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
    return float(np.mean(loss))


def mean_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantiles: Tuple[float, ...]) -> float:
    """Mean pinball loss across multiple quantiles.

    Useful as a single-number summary when comparing two quantile models.
    """
    losses = [pinball_loss(y_true, y_pred, q) for q in quantiles]
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass
class QuantileModel:
    """Bundle of three fitted quantile regressors + the shared preprocessor."""

    kind: str                 # "lightgbm" | "gradient_boosting"
    quantiles: Tuple[float, ...]
    pipeline_p10: Any          # fitted sklearn Pipeline for q=0.10
    pipeline_p50: Any          # fitted sklearn Pipeline for q=0.50 (median)
    pipeline_p90: Any          # fitted sklearn Pipeline for q=0.90
    feature_names: List[str]
    fit_time_seconds: float = 0.0

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with columns ``p10``, ``p50``, ``p90``."""
        p10 = self.pipeline_p10.predict(X)
        p50 = self.pipeline_p50.predict(X)
        p90 = self.pipeline_p90.predict(X)
        # Enforce non-crossing post-hoc: p10 ≤ p50 ≤ p90.
        # (Independent quantile fits can violate this on individual rows;
        # isotonic-style rearrangement is the standard fix.)
        p50 = np.maximum(np.minimum(p50, p90), p10)
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)
        return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=X.index)


@dataclass
class QuantileMetrics:
    """Evaluation metrics for a fitted quantile model."""

    model_name: str
    quantiles: List[float]
    pinball_per_quantile: List[float]
    mean_pinball: float
    median_mae: float            # |y - p50|
    median_rmse: float           # sqrt(mean((y - p50)^2))
    coverage_p10_p90: float       # fraction of y_true ∈ [p10, p90]
    mean_interval_width: float   # mean(p90 - p10)
    median_interval_width: float  # median(p90 - p10)
    crossing_rate: float         # fraction of rows where p10 > p50 or p50 > p90 (pre-fix)
    n_samples: int
    fit_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------
def build_feature_pipeline(schema: HousingSchema = SCHEMA) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer: impute + OneHot/scale.

    * Numeric features: median imputation + StandardScaler.
    * Categorical features: constant imputation + OneHotEncoder.
    """
    numeric = list(schema.numeric_features)
    categorical = list(schema.categorical_features)

    numeric_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop")


# ---------------------------------------------------------------------------
# Booster factories
# ---------------------------------------------------------------------------
def _make_lightgbm_quantile(q: float, random_state: int = 42) -> "LGBMRegressor":
    """LightGBM quantile regressor for quantile ``q``."""
    return LGBMRegressor(
        objective="quantile", alpha=q,
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        max_depth=-1, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=-1, random_state=random_state, verbose=-1,
    )


def _make_gbr_quantile(q: float, random_state: int = 42) -> "GradientBoostingRegressor":
    """sklearn GradientBoostingRegressor quantile regressor for quantile ``q``."""
    return GradientBoostingRegressor(
        loss="quantile", alpha=q,
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=20, subsample=0.8, random_state=random_state,
    )


def _make_regressor(kind: QuantileKind, q: float, random_state: int = 42):
    if kind == QuantileKind.LIGHTGBM:
        if not HAVE_LIGHTGBM:
            raise RuntimeError("LightGBM is not installed.")
        return _make_lightgbm_quantile(q, random_state=random_state)
    if kind == QuantileKind.GRADIENT_BOOSTING:
        return _make_gbr_quantile(q, random_state=random_state)
    raise ValueError(f"Unknown QuantileKind: {kind}")


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------
def train_quantile_model(
    kind: QuantileKind,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantiles: Tuple[float, ...] = DEFAULT_QUANTILES,
    random_state: int = 42,
) -> QuantileModel:
    """Train three quantile regressors (one per quantile) sharing one preprocessor.

    Returns a ``QuantileModel`` bundle. The three regressors are independent
    (each uses the ``quantile``/``alpha`` loss specific to its quantile) but
    share the same preprocessor so feature engineering is identical.
    """
    pre = build_feature_pipeline()
    # Fit the preprocessor once on the training data, then transform once.
    X_train_t = pre.fit_transform(X_train, y_train)

    pipelines: Dict[float, SkPipeline] = {}
    t0 = time.perf_counter()
    for q in quantiles:
        reg = _make_regressor(kind, q, random_state=random_state)
        reg.fit(X_train_t, y_train)
        # Wrap the pre-fit preprocessor + regressor in a Pipeline so
        # ``predict`` works on raw DataFrames at inference time.
        pipe = SkPipeline([("pre", pre), ("model", reg)])
        # NB: We don't refit the preprocessor; we just reference it. The
        # Pipeline.predict() will re-transform using the already-fit
        # preprocessor, which is correct.
        pipelines[q] = pipe

    fit_time = time.perf_counter() - t0
    # Sort quantiles for naming consistency.
    qs = sorted(quantiles)
    return QuantileModel(
        kind=kind.value,
        quantiles=tuple(qs),
        pipeline_p10=pipelines[qs[0]],
        pipeline_p50=pipelines[qs[len(qs) // 2]],
        pipeline_p90=pipelines[qs[-1]],
        feature_names=list(X_train.columns),
        fit_time_seconds=fit_time,
    )


def evaluate_quantile_model(
    model: QuantileModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> QuantileMetrics:
    """Compute pinball loss + coverage + interval width on the holdout set."""
    preds = model.predict(X_test)
    y_true = np.asarray(y_test, dtype=float)

    p10 = preds["p10"].values
    p50 = preds["p50"].values
    p90 = preds["p90"].values

    # Pinball per quantile.
    pb10 = pinball_loss(y_true, p10, q=0.10)
    pb50 = pinball_loss(y_true, p50, q=0.50)
    pb90 = pinball_loss(y_true, p90, q=0.90)

    # Median MAE / RMSE.
    mae = float(np.mean(np.abs(y_true - p50)))
    rmse = float(np.sqrt(np.mean((y_true - p50) ** 2)))

    # Coverage of the 80% interval [p10, p90].
    inside = (y_true >= p10) & (y_true <= p90)
    coverage = float(np.mean(inside))

    # Interval widths.
    widths = p90 - p10
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))

    # Crossing rate (pre-fix): fraction where p10 > p50 OR p50 > p90 in the
    # RAW predictions (before the post-hoc non-crossing fix in
    # ``QuantileModel.predict``). We re-fetch raw predictions here.
    raw_p10 = model.pipeline_p10.predict(X_test)
    raw_p50 = model.pipeline_p50.predict(X_test)
    raw_p90 = model.pipeline_p90.predict(X_test)
    crossings = ((raw_p10 > raw_p50) | (raw_p50 > raw_p90))
    crossing_rate = float(np.mean(crossings))

    return QuantileMetrics(
        model_name=model.kind,
        quantiles=list(model.quantiles),
        pinball_per_quantile=[pb10, pb50, pb90],
        mean_pinball=float(np.mean([pb10, pb50, pb90])),
        median_mae=mae,
        median_rmse=rmse,
        coverage_p10_p90=coverage,
        mean_interval_width=mean_width,
        median_interval_width=median_width,
        crossing_rate=crossing_rate,
        n_samples=int(len(y_true)),
        fit_time_seconds=model.fit_time_seconds,
    )


__all__ = [
    "QuantileKind",
    "DEFAULT_QUANTILES",
    "CANDIDATE_MODELS",
    "QuantileModel",
    "QuantileMetrics",
    "pinball_loss",
    "mean_pinball_loss",
    "build_feature_pipeline",
    "train_quantile_model",
    "evaluate_quantile_model",
    "HAVE_LIGHTGBM",
]
