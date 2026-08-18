"""
model
=====

Dual-modeling engine for subscription churn:
  1. Standard classifiers (Logistic Regression / Random Forest) for
     binary ``churned`` prediction.
  2. Survival analysis (Kaplan-Meier estimator, Cox Proportional Hazards)
     for time-to-churn estimation.
  3. Uplift / Expected-Value decision policy for retention targeting.

Public surface
--------------
- ``ModelKind``              : enum (logreg, random_forest, kaplan_meier, cox_ph).
- ``CANDIDATE_MODELS``       : classifier registry.
- ``ClassifierMetrics``      : holdout metrics for binary classifiers.
- ``SurvivalMetrics``        : C-index + median-survival-time metrics for survival models.
- ``UpliftResult``           : per-customer uplift + expected-value metrics.
- ``build_feature_pipeline`` : sklearn ColumnTransformer for classifier input.
- ``train_classifier``        : train LogReg/RF, return pipeline + metrics.
- ``fit_kaplan_meier``       : fit a KaplanMeierFitter.
- ``fit_cox_ph``             : fit a CoxPHFitter.
- ``compute_c_index``        : concordance index for survival predictions.
- ``compute_uplift``         : compute uplift per customer (treatment - control).
- ``expected_value_policy``  : rank customers by expected ROI and return
                              the optimal targeting threshold.

Design notes
------------
1. **Dual-modeling is mandatory** — a pure classifier answers "will
   this customer churn?" but not "when?". A pure survival model answers
   "when?" but treats all currently-active customers as censored. The
   dual approach lets us:
     * Use the classifier for short-term ("next 30 days") interventions.
     * Use the survival model for lifetime-value (LTV) calculations and
       prioritizing long-term retention offers.

2. **C-index is the survival equivalent of AUC** — it measures the
   fraction of comparable pairs where the model correctly orders
   the predicted risk. A C-index of 0.5 is random; 1.0 is perfect.

3. **Uplift modeling targets the "persuadables"** — customers who would
   churn without a retention offer but stay if they receive one. We
   compute uplift as ``P(churn | no offer) − P(churn | offer)`` via
   either:
     * Two-model approach (separate classifiers for treatment/control)
     * CATE approach (single classifier with treatment as a feature)
   The two-model approach is simpler and is what we implement here.

4. **Expected-Value policy** — given uplift + per-customer LTV + offer
   cost, compute the expected ROI of targeting each customer and rank
   them. The optimal targeting threshold is where cumulative ROI
   stops increasing. We return the full ROI curve so the user can
   pick any threshold.

5. **Kaplan-Meier non-increasing survival** — the KM estimator is
   mathematically guaranteed to produce non-increasing survival
   probabilities. The test suite verifies this invariant holds on
   the synthetic data.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, log_loss, confusion_matrix,
)
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import ChurnSchema, SCHEMA  # noqa: E402


# ---------------------------------------------------------------------------
# Enums & registries
# ---------------------------------------------------------------------------
class ModelKind(str, Enum):
    LOGREG = "logreg"
    RANDOM_FOREST = "random_forest"
    # NB: KM and Cox PH are survival models, not classifiers, but we
    # expose them via the same enum so the CLI can dispatch uniformly.
    KAPLAN_MEIER = "kaplan_meier"
    COX_PH = "cox_ph"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


CANDIDATE_MODELS: Dict[str, ModelKind] = {
    ModelKind.LOGREG.value: ModelKind.LOGREG,
    ModelKind.RANDOM_FOREST.value: ModelKind.RANDOM_FOREST,
}


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass
class ClassifierMetrics:
    """Holdout metrics for a binary classifier."""

    model_name: str
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    roc_auc: float
    brier_score: float
    log_loss: float
    confusion_matrix: List[List[int]]
    fit_time_seconds: float
    predict_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SurvivalMetrics:
    """Metrics for a survival model (KM or Cox PH).

    For Kaplan-Meier (which has no covariates), ``c_index`` is None
    because there's nothing to rank — every customer gets the same
    survival curve. For Cox PH, ``c_index`` is computed via
    ``lifelines.utils.concordance_index`` against the holdout set.
    """

    model_name: str
    c_index: Optional[float]
    median_survival_time_months: Optional[float]
    mean_survival_time_months: Optional[float]
    n_samples: int
    n_events: int
    n_censored: int
    fit_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UpliftResult:
    """Per-customer uplift + expected-value metrics.

    Attributes
    ----------
    customer_ids : list
    uplift : np.ndarray
        Per-customer uplift = P(churn | control) − P(churn | treatment).
        Positive uplift means the treatment *reduces* churn.
    expected_value : np.ndarray
        Per-customer expected ROI if treated = uplift * LTV − offer_cost.
    cumulative_roi : np.ndarray
        Cumulative ROI when targeting customers in descending order
        of expected_value. ``cumulative_roi[i]`` is the total ROI if
        you target the top ``i+1`` customers.
    optimal_threshold_idx : int
        Index of the maximum cumulative ROI. Targeting this many
        customers maximizes total ROI.
    optimal_threshold_value : float
        The expected_value threshold at the optimum.
    """

    customer_ids: List[int]
    uplift: np.ndarray
    expected_value: np.ndarray
    cumulative_roi: np.ndarray
    optimal_threshold_idx: int
    optimal_threshold_value: float
    total_targeted: int
    total_roi: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_threshold_idx": self.optimal_threshold_idx,
            "optimal_threshold_value": self.optimal_threshold_value,
            "total_targeted": self.total_targeted,
            "total_roi": self.total_roi,
            "n_customers": len(self.customer_ids),
            "uplift_mean": float(np.mean(self.uplift)),
            "uplift_std": float(np.std(self.uplift)),
        }


# ---------------------------------------------------------------------------
# Feature pipeline for classifiers
# ---------------------------------------------------------------------------
def build_feature_pipeline(schema: ChurnSchema = SCHEMA) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer for classifier input."""
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
# Classifier training
# ---------------------------------------------------------------------------
def _make_classifier(kind: ModelKind, random_state: int = 42):
    if kind == ModelKind.LOGREG:
        return LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            random_state=random_state, n_jobs=-1,
        )
    if kind == ModelKind.RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=random_state,
        )
    raise ValueError(f"Unknown classifier kind: {kind}")


def train_classifier(
    kind: ModelKind,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[SkPipeline, ClassifierMetrics]:
    """Train + evaluate a binary classifier (LogReg or RandomForest)."""
    pre = build_feature_pipeline()
    clf = _make_classifier(kind, random_state=random_state)
    pipe = SkPipeline([("pre", pre), ("model", clf)])

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    predict_time = time.perf_counter() - t0

    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = y_pred.astype(float)

    metrics = ClassifierMetrics(
        model_name=kind.value,
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        precision_macro=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_proba)),
        brier_score=float(brier_score_loss(y_test, y_proba)),
        log_loss=float(log_loss(y_test, pipe.predict_proba(X_test))),
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
        fit_time_seconds=fit_time,
        predict_time_seconds=predict_time,
    )
    return pipe, metrics


# ---------------------------------------------------------------------------
# Kaplan-Meier estimator
# ---------------------------------------------------------------------------
def fit_kaplan_meier(
    durations: pd.Series,
    events: pd.Series,
    label: str = "KM",
) -> Tuple[KaplanMeierFitter, SurvivalMetrics]:
    """Fit a Kaplan-Meier estimator and compute summary metrics.

    Parameters
    ----------
    durations : pd.Series
        Time-to-event (or time-to-censoring).
    events : pd.Series
        Binary event indicator (1=event observed, 0=censored).
    label : str
        Label for the KM curve.

    Returns
    -------
    (kmf, metrics)
    """
    t0 = time.perf_counter()
    kmf = KaplanMeierFitter(label=label)
    kmf.fit(durations, event_observed=events)
    fit_time = time.perf_counter() - t0

    # Median survival time (lifelines returns np.inf if the survival
    # curve never drops below 0.5 — which happens when most customers
    # are censored at the observation window).
    median = kmf.median_survival_time_
    if pd.isna(median) or np.isinf(median):
        median_val: Optional[float] = None
    else:
        median_val = float(median)

    # Mean survival time — truncated at the longest observed duration.
    try:
        # lifelines exposes this via the survival_function_ attribute.
        # We integrate S(t) dt from 0 to the last observed time.
        s = kmf.survival_function_.values.ravel()
        timeline = kmf.survival_function_.index.values
        mean_val: Optional[float] = float(np.trapz(s, timeline))
    except Exception:
        mean_val = None

    metrics = SurvivalMetrics(
        model_name="kaplan_meier",
        c_index=None,  # KM has no covariates — no ranking.
        median_survival_time_months=median_val,
        mean_survival_time_months=mean_val,
        n_samples=int(len(durations)),
        n_events=int(events.sum()),
        n_censored=int((1 - events).sum()),
        fit_time_seconds=fit_time,
    )
    return kmf, metrics


# ---------------------------------------------------------------------------
# Cox Proportional Hazards
# ---------------------------------------------------------------------------
def _prepare_cox_input(
    X: pd.DataFrame,
    durations: pd.Series,
    events: pd.Series,
    schema: ChurnSchema = SCHEMA,
) -> pd.DataFrame:
    """Prepare a Cox-PH-ready dataframe.

    The Cox PH fitter expects numeric columns + a ``duration`` column +
    an ``event`` column. Categorical features are one-hot encoded.
    """
    df = X.copy()
    # Reset indices so concatenation aligns.
    df = df.reset_index(drop=True)
    durations = durations.reset_index(drop=True)
    events = events.reset_index(drop=True)

    # One-hot encode categoricals.
    cat_cols = list(schema.categorical_features)
    num_cols = list(schema.numeric_features)
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    # Drop any non-numeric columns left over.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df = df.drop(col, axis=1)
    # Impute NaN numerics with median.
    df = df.fillna(df.median())

    # Append duration + event.
    df["duration"] = durations.astype(float)
    df["event"] = events.astype(int)
    return df


def fit_cox_ph(
    X_train: pd.DataFrame,
    durations_train: pd.Series,
    events_train: pd.Series,
    X_test: pd.DataFrame,
    durations_test: pd.Series,
    events_test: pd.Series,
    schema: ChurnSchema = SCHEMA,
) -> Tuple[CoxPHFitter, SurvivalMetrics]:
    """Fit a Cox Proportional Hazards model and compute the C-index on the test set."""
    train_df = _prepare_cox_input(X_train, durations_train, events_train, schema)
    test_df = _prepare_cox_input(X_test, durations_test, events_test, schema)
    # Drop duration + event for prediction.
    test_features = test_df.drop(columns=["duration", "event"])

    t0 = time.perf_counter()
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df, duration_col="duration", event_col="event")
    fit_time = time.perf_counter() - t0

    # Predict partial hazard on the test set.
    try:
        partial_hazard = cph.predict_partial_hazard(test_features).values.ravel()
    except Exception:
        partial_hazard = np.zeros(len(test_features))

    # C-index: concordance between predicted partial hazard and observed
    # durations / events.
    c_idx = float(concordance_index(
        durations_test.values, -partial_hazard, events_test.values,
    ))
    # NB: lifelines' concordance_index expects "predicted scores" where
    # higher = longer survival. Our partial_hazard is exp(β·x) where
    # higher = MORE likely to churn (so SHORTER survival). We negate it
    # so higher predicted score = longer survival, matching the convention.

    # Median survival time on the test set (averaged across customers).
    try:
        predicted_medians = cph.predict_median(test_features)
        # Filter out infinities (median survival > observation window).
        finite = predicted_medians[np.isfinite(predicted_medians)]
        median_val = float(np.median(finite)) if len(finite) > 0 else None
    except Exception:
        median_val = None

    try:
        predicted_means = cph.predict_mean(test_features)
        finite_means = predicted_means[np.isfinite(predicted_means)]
        mean_val = float(np.mean(finite_means)) if len(finite_means) > 0 else None
    except Exception:
        mean_val = None

    metrics = SurvivalMetrics(
        model_name="cox_ph",
        c_index=c_idx,
        median_survival_time_months=median_val,
        mean_survival_time_months=mean_val,
        n_samples=int(len(durations_test)),
        n_events=int(events_test.sum()),
        n_censored=int((1 - events_test).sum()),
        fit_time_seconds=fit_time,
    )
    return cph, metrics


def compute_c_index(
    predicted_scores: np.ndarray,
    durations: pd.Series,
    events: pd.Series,
) -> float:
    """Compute the concordance index.

    Parameters
    ----------
    predicted_scores : np.ndarray
        Higher score = longer predicted survival.
    durations, events : pd.Series
        Observed durations and event indicators.

    Returns
    -------
    float
        Concordance index in [0, 1]. 0.5 = random; 1.0 = perfect.
    """
    return float(concordance_index(durations.values, predicted_scores, events.values))


# ---------------------------------------------------------------------------
# Uplift modeling
# ---------------------------------------------------------------------------
def compute_uplift(
    p_churn_control: np.ndarray,
    p_churn_treatment: np.ndarray,
) -> np.ndarray:
    """Compute per-customer uplift = P(churn | control) − P(churn | treatment).

    A positive uplift means the treatment *reduces* churn for that customer.

    Parameters
    ----------
    p_churn_control : np.ndarray, shape (n,)
        Predicted churn probability under the no-offer (control) condition.
    p_churn_treatment : np.ndarray, shape (n,)
        Predicted churn probability under the offer (treatment) condition.

    Returns
    -------
    np.ndarray, shape (n,)
        Per-customer uplift. Positive = persuadable (treatment helps).
        Negative = "do not disturb" (treatment backfires).
    """
    p_churn_control = np.asarray(p_churn_control, dtype=float)
    p_churn_treatment = np.asarray(p_churn_treatment, dtype=float)
    return p_churn_control - p_churn_treatment


def expected_value_policy(
    uplift: np.ndarray,
    customer_ltv: np.ndarray,
    offer_cost: float = 10.0,
) -> UpliftResult:
    """Rank customers by expected ROI and find the optimal targeting threshold.

    For each customer:
        expected_value = uplift * LTV − offer_cost

    Where:
        uplift = P(churn | control) − P(churn | treatment)
               = probability the offer *saves* the customer

    So ``uplift * LTV`` is the expected revenue saved by making the offer,
    and ``offer_cost`` is what it costs to make the offer.

    Parameters
    ----------
    uplift : np.ndarray, shape (n,)
        Per-customer uplift (from ``compute_uplift``).
    customer_ltv : np.ndarray, shape (n,)
        Per-customer lifetime value (revenue if not churned).
    offer_cost : float
        Cost of the retention offer (e.g. $10 discount).

    Returns
    -------
    UpliftResult
        Contains the cumulative-ROI curve + the optimal targeting threshold
        (where cumulative ROI is maximized).
    """
    uplift = np.asarray(uplift, dtype=float)
    customer_ltv = np.asarray(customer_ltv, dtype=float)

    # Per-customer expected value of targeting.
    ev = uplift * customer_ltv - offer_cost

    # Rank customers by expected value (descending).
    order = np.argsort(ev)[::-1]
    ev_sorted = ev[order]

    # Cumulative ROI when targeting the top-k customers.
    cumulative_roi = np.cumsum(ev_sorted)

    # Optimal threshold = index of maximum cumulative ROI.
    optimal_idx = int(np.argmax(cumulative_roi))
    optimal_value = float(ev_sorted[optimal_idx])
    total_roi = float(cumulative_roi[optimal_idx])

    return UpliftResult(
        customer_ids=order.tolist(),
        uplift=uplift,
        expected_value=ev,
        cumulative_roi=cumulative_roi,
        optimal_threshold_idx=optimal_idx,
        optimal_threshold_value=optimal_value,
        total_targeted=optimal_idx + 1,
        total_roi=total_roi,
    )


__all__ = [
    "ModelKind",
    "CANDIDATE_MODELS",
    "ClassifierMetrics",
    "SurvivalMetrics",
    "UpliftResult",
    "build_feature_pipeline",
    "train_classifier",
    "fit_kaplan_meier",
    "fit_cox_ph",
    "compute_c_index",
    "compute_uplift",
    "expected_value_policy",
]
