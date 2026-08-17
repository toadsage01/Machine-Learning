"""
model
=====

Feature engineering, model training, probability calibration, and fairness
analysis for the P4 Titanic Two-Ships benchmark.

Public surface
--------------
- ``ModelKind``              : enum (LightGBM, CatBoost, XGBoost).
- ``CalibrationKind``        : enum (Isotonic, Sigmoid).
- ``FairnessSlice``           : value object describing a (feature, value) subgroup.
- ``ModelMetrics``            : holdout + CV metrics for one trained model.
- ``CalibrationResult``       : calibration curve + Brier + log-loss per model.
- ``FairnessReport``          : per-slice disparity metrics across subgroups.
- ``build_feature_pipeline``  : ``ColumnTransformer`` with imputers + OneHotEncoder.
- ``train_model``              : train one of the three boosters + return metrics.
- ``calibrate``                : wrap a fitted model with Isotonic or Sigmoid calibration.
- ``evaluate_calibration``    : calibration curve + Brier score + log-loss.
- ``compute_fairness``        : per-slice accuracy / F1 / selection-rate disparity.
- ``CANDIDATE_MODELS``         : registry dict ``name -> ModelKind``.

Design notes
------------
1. **Three boosters, one interface** — LightGBM, CatBoost, and XGBoost each
   have their own Python API. We wrap them in a single ``train_model(name)``
   entry-point so the CLI can benchmark them head-to-head without bespoke
   per-booster code in ``train.py``.

2. **Probability calibration is mandatory** — boosting models are notoriously
   mis-calibrated (especially XGBoost on small datasets). We expose both
   Isotonic (non-parametric, flexible, data-hungry) and Sigmoid (parametric,
   robust on small data) calibrators via sklearn's ``CalibratedClassifierCV``
   with ``cv=5`` so calibration never reuses training data.

3. **Fairness slicing is first-class** — every model is evaluated not just on
   overall accuracy but on per-slice accuracy / F1 / selection-rate across
   sex, pclass, is_child, is_elderly, and alone. The ``FairnessReport``
   includes both the raw per-slice metrics AND the disparity ratios
   (max/min) which are what fairness audits actually look at.

4. **Brier score is the calibration metric** — we report both the raw Brier
   (lower = better) and the *Brier skill score* vs. a climatology
   forecaster (predicting the base rate). A skill score of 0 means "no
   better than predicting the base rate"; 1 means "perfect".
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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, log_loss, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import UnifiedDataset, UnifiedSchema, SCHEMA  # noqa: E402

# Boosters (all three are optional — degrade gracefully).
try:
    from lightgbm import LGBMClassifier
    HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover
    HAVE_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAVE_CATBOOST = True
except Exception:  # pragma: no cover
    HAVE_CATBOOST = False

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except Exception:  # pragma: no cover
    HAVE_XGBOOST = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ModelKind(str, Enum):
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    XGBOOST = "xgboost"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


class CalibrationKind(str, Enum):
    ISOTONIC = "isotonic"
    SIGMOID = "sigmoid"
    NONE = "none"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


CANDIDATE_MODELS: Dict[str, ModelKind] = {
    ModelKind.LIGHTGBM.value: ModelKind.LIGHTGBM,
    ModelKind.CATBOOST.value: ModelKind.CATBOOST,
    ModelKind.XGBOOST.value: ModelKind.XGBOOST,
}


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass
class ModelMetrics:
    """Holdout + CV metrics for a single trained model."""

    model_name: str
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    roc_auc: float
    average_precision: float
    brier_score: float
    log_loss: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    confusion_matrix: List[List[int]]
    fit_time_seconds: float
    predict_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationResult:
    """Calibration curve + metrics for a fitted model."""

    model_name: str
    calibration_kind: str
    n_bins: int
    fraction_of_positives: List[float]
    mean_predicted_value: List[float]
    brier_score: float
    brier_skill_score: float  # 1 - brier / brier_climatology
    log_loss: float
    brier_climatology: float  # baseline Brier (predicting base rate)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FairnessSlice:
    """One subgroup's metrics."""

    feature: str           # e.g. "sex"
    value: str             # e.g. "female"
    n_samples: int
    accuracy: float
    f1_macro: float
    selection_rate: float  # fraction predicted positive
    base_rate: float        # fraction actually positive
    false_positive_rate: float
    false_negative_rate: float


@dataclass
class FairnessReport:
    """Per-slice fairness report for a fitted model."""

    model_name: str
    slices: List[FairnessSlice]
    accuracy_disparity_ratio: float  # max / min accuracy across slices
    selection_disparity_ratio: float  # max / min selection_rate across slices
    n_features_evaluated: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "accuracy_disparity_ratio": self.accuracy_disparity_ratio,
            "selection_disparity_ratio": self.selection_disparity_ratio,
            "n_features_evaluated": self.n_features_evaluated,
            "slices": [asdict(s) for s in self.slices],
        }


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------
def build_feature_pipeline(schema: UnifiedSchema = SCHEMA) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer: impute + OneHot/scale.

    * Numeric features: median imputation + StandardScaler.
    * Categorical features: constant imputation ('missing') + OneHotEncoder.

    The OneHotEncoder uses ``handle_unknown='ignore'`` so unseen categories
    at inference time don't raise (critical for the unified schema, where
    e.g. Spaceship's ``deck='T'`` may not appear in the classic training set).
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
def _make_lightgbm(random_state: int = 42) -> "LGBMClassifier":
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        max_depth=-1, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        objective="binary", n_jobs=-1, random_state=random_state, verbose=-1,
    )


def _make_catboost(random_state: int = 42) -> "CatBoostClassifier":
    return CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0, loss_function="Logloss",
        random_seed=random_state, verbose=0, allow_writing_files=False,
    )


def _make_xgboost(random_state: int = 42) -> "XGBClassifier":
    return XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        min_child_weight=1, reg_alpha=0.1, reg_lambda=0.1,
        objective="binary:logistic", eval_metric="logloss",
        n_jobs=-1, random_state=random_state, verbosity=0,
        tree_method="hist",
    )


def _make_booster(kind: ModelKind, random_state: int = 42):
    if kind == ModelKind.LIGHTGBM:
        if not HAVE_LIGHTGBM:
            raise RuntimeError("LightGBM is not installed.")
        return _make_lightgbm(random_state)
    if kind == ModelKind.CATBOOST:
        if not HAVE_CATBOOST:
            raise RuntimeError("CatBoost is not installed.")
        return _make_catboost(random_state)
    if kind == ModelKind.XGBOOST:
        if not HAVE_XGBOOST:
            raise RuntimeError("XGBoost is not installed.")
        return _make_xgboost(random_state)
    raise ValueError(f"Unknown ModelKind: {kind}")


# ---------------------------------------------------------------------------
# Train + evaluate (uncalibrated baseline)
# ---------------------------------------------------------------------------
def _safe_proba(model, X) -> Optional[np.ndarray]:
    try:
        return model.predict_proba(X)
    except Exception:
        return None


def train_model(
    kind: ModelKind,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[SkPipeline, ModelMetrics]:
    """Train + evaluate one booster end-to-end.

    Returns a fitted sklearn Pipeline (ColumnTransformer → booster) and a
    ``ModelMetrics`` value object. No calibration is applied here — call
    ``calibrate`` separately on the returned pipeline.
    """
    pre = build_feature_pipeline()
    booster = _make_booster(kind, random_state=random_state)
    pipe = SkPipeline([("pre", pre), ("model", booster)])

    # CV accuracy on training set.
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    # Fit on full training fold.
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # Predict + measure.
    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    predict_time = time.perf_counter() - t0

    proba = _safe_proba(pipe, X_test)
    if proba is None:
        proba = np.zeros((len(y_pred), 2))
    y_proba_pos = proba[:, 1] if proba.ndim == 2 else proba

    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = ModelMetrics(
        model_name=kind.value,
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        precision_macro=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_proba_pos)),
        average_precision=float(average_precision_score(y_test, y_proba_pos)),
        brier_score=float(brier_score_loss(y_test, y_proba_pos)),
        log_loss=float(log_loss(y_test, proba)),
        cv_accuracy_mean=float(np.mean(cv_scores)),
        cv_accuracy_std=float(np.std(cv_scores)),
        confusion_matrix=cm,
        fit_time_seconds=fit_time,
        predict_time_seconds=predict_time,
    )
    return pipe, metrics


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(
    pipeline: SkPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kind: CalibrationKind = CalibrationKind.ISOTONIC,
    cv_folds: int = 5,
) -> SkPipeline:
    """Wrap a fitted pipeline with sklearn's ``CalibratedClassifierCV``.

    The calibration is fit using cross-validation on the *training* set,
    never reusing test data. Returns a new pipeline (the original is not
    mutated).

    For ``kind=NONE`` we return the original pipeline unchanged.
    """
    if kind == CalibrationKind.NONE:
        return pipeline
    cal = CalibratedClassifierCV(
        estimator=pipeline,  # sklearn 1.2+ API
        method=kind.value,
        cv=cv_folds,
    )
    cal.fit(X_train, y_train)
    return cal


def evaluate_calibration(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    calibration_kind: str,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute the calibration curve + Brier score for a (calibrated) model.

    Uses ``sklearn.calibration.calibration_curve`` with ``strategy='quantile'``
    so each bin contains roughly the same number of samples (this is the
    recommendation from the sklearn docs — uniform-strategy bins are
    misleading when probabilities are clustered, which is the norm for
    well-trained boosters).
    """
    proba = _safe_proba(model, X_test)
    if proba is None:
        proba = np.zeros((len(y_test), 2))
    y_proba_pos = proba[:, 1] if proba.ndim == 2 else proba

    # Calibration curve — handle degenerate cases (e.g. all-same proba).
    try:
        frac_pos, mean_pred = calibration_curve(
            y_test, y_proba_pos, n_bins=n_bins, strategy="quantile",
        )
    except Exception:
        frac_pos = np.array([0.0, 1.0])
        mean_pred = np.array([0.0, 1.0])

    brier = float(brier_score_loss(y_test, y_proba_pos))
    ll = float(log_loss(y_test, proba))
    base_rate = float(np.mean(y_test))
    # Climatology Brier = base_rate × (1 - base_rate) — the Brier you'd
    # get by always predicting the base rate.
    brier_clim = base_rate * (1.0 - base_rate)
    skill = 1.0 - brier / max(brier_clim, 1e-12)

    return CalibrationResult(
        model_name=model_name,
        calibration_kind=calibration_kind,
        n_bins=n_bins,
        fraction_of_positives=frac_pos.tolist(),
        mean_predicted_value=mean_pred.tolist(),
        brier_score=brier,
        brier_skill_score=skill,
        log_loss=ll,
        brier_climatology=brier_clim,
    )


# ---------------------------------------------------------------------------
# Fairness analysis
# ---------------------------------------------------------------------------
DEFAULT_FAIRNESS_FEATURES: List[str] = [
    "sex", "pclass", "is_child", "is_elderly", "alone",
]


def compute_fairness(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    features: Optional[List[str]] = None,
    min_slice_size: int = 10,
) -> FairnessReport:
    """Compute per-slice fairness metrics.

    For each ``feature`` in ``features`` and each unique value of that
    feature in ``X_test``, we compute:
        * accuracy
        * F1 (macro)
        * selection_rate (fraction predicted positive — measures "how often
          does the model say yes for this subgroup")
        * base_rate (actual positive rate in the subgroup)
        * false_positive_rate
        * false_negative_rate

    The ``accuracy_disparity_ratio`` is ``max(accuracy) / min(accuracy)``
    across all slices. A ratio of 1.0 means perfect parity; > 1.2 typically
    warrants investigation. The ``selection_disparity_ratio`` is the same
    but for ``selection_rate`` — this is the standard "demographic parity"
    fairness metric.

    Parameters
    ----------
    min_slice_size : int
        Slices smaller than this are skipped (no statistically meaningful
        metrics). Defaults to 10; tests that need to verify per-row math
        can pass ``min_slice_size=1``.
    """
    features = features or DEFAULT_FAIRNESS_FEATURES
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "ravel"):
        y_pred = y_pred.ravel()
    y_pred = np.asarray(y_pred).astype(int)
    y_true = np.asarray(y_test).astype(int)

    slices: List[FairnessSlice] = []
    for feat in features:
        if feat not in X_test.columns:
            continue
        col = X_test[feat]
        # Coerce to string for categorical / boolean features; bin numerics.
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > 5:
            # Bin into quartiles for fairness analysis.
            try:
                cats = pd.qcut(col, q=4, duplicates="drop").astype(str)
            except Exception:
                cats = col.astype(str)
        else:
            cats = col.astype(str)

        for val in sorted(cats.unique()):
            mask = (cats == val).values
            n = int(mask.sum())
            if n < min_slice_size:
                # Skip slices too small to be statistically meaningful.
                continue
            y_t = y_true[mask]
            y_p = y_pred[mask]

            # FPR / FNR only well-defined when both classes present.
            tn = fp = fn = tp = 0
            for t, p in zip(y_t, y_p):
                if t == 0 and p == 0: tn += 1
                elif t == 0 and p == 1: fp += 1
                elif t == 1 and p == 0: fn += 1
                else: tp += 1
            fpr = fp / max(fp + tn, 1)
            fnr = fn / max(fn + tp, 1)

            slices.append(FairnessSlice(
                feature=feat,
                value=str(val),
                n_samples=n,
                accuracy=float(accuracy_score(y_t, y_p)),
                f1_macro=float(f1_score(y_t, y_p, average="macro", zero_division=0)),
                selection_rate=float(np.mean(y_p)),
                base_rate=float(np.mean(y_t)),
                false_positive_rate=float(fpr),
                false_negative_rate=float(fnr),
            ))

    # Disparity ratios.
    if slices:
        accs = [s.accuracy for s in slices if s.accuracy > 0]
        sels = [s.selection_rate for s in slices]
        acc_disp = max(accs) / min(accs) if min(accs) > 0 else float("inf")
        sel_disp = max(sels) / min(sels) if min(sels) > 0 else float("inf")
    else:
        acc_disp = sel_disp = 1.0

    return FairnessReport(
        model_name=model_name,
        slices=slices,
        accuracy_disparity_ratio=float(acc_disp),
        selection_disparity_ratio=float(sel_disp),
        n_features_evaluated=len(features),
    )


__all__ = [
    "ModelKind",
    "CalibrationKind",
    "CANDIDATE_MODELS",
    "ModelMetrics",
    "CalibrationResult",
    "FairnessSlice",
    "FairnessReport",
    "DEFAULT_FAIRNESS_FEATURES",
    "build_feature_pipeline",
    "train_model",
    "calibrate",
    "evaluate_calibration",
    "compute_fairness",
    "HAVE_LIGHTGBM",
    "HAVE_CATBOOST",
    "HAVE_XGBOOST",
]
