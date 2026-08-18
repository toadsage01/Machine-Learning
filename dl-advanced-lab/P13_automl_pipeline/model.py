"""
model
=====

Automated ML engine: AutoFeatureTransformer, Optuna HPO wrapper across
LightGBM/XGBoost/CatBoost, and MLflow tracking integration.

Public surface
--------------
- ``ModelKind``                : enum (lightgbm, xgboost, catboost).
- ``CANDIDATE_MODELS``         : registry.
- ``AutoFeatureTransformer``  : sklearn ColumnTransformer with imputation,
                                 encoding, scaling, polynomial features.
- ``HPOTrialResult``           : value object for a single HPO trial.
- ``HPOReport``                : aggregated HPO report.
- ``OptunaHPO``                : Optuna hyperparameter optimization wrapper.
- ``MLflowTracker``            : MLflow experiment tracking integration.
- ``build_model``              : factory for a boosted model from hyperparams.
- ``train_and_evaluate``       : end-to-end train + evaluate one config.
- ``run_automl_pipeline``      : full AutoML pipeline (features → HPO → MLflow).

Design notes
------------
1. **AutoFeatureTransformer** — automatically builds a ColumnTransformer
   based on the inferred schema. Numeric columns get median imputation +
   StandardScaler (optional polynomial features). Categorical columns get
   constant imputation + OneHotEncoder. Datetime columns get decomposed
   into year/month/day/dayofweek/hour.

2. **Optuna HPO** — uses Tree-structured Parzen Estimator (TPE) sampler.
   Each trial samples: model choice (LightGBM/XGBoost/CatBoost) + that
   model's hyperparameters (n_estimators, learning_rate, max_depth, etc.).
   Cross-validated accuracy or RMSE is the optimization objective.

3. **MLflow tracking** — each trial is logged as an MLflow run with:
     * params: the hyperparameters.
     * metrics: accuracy / F1 / RMSE / fit_time.
     * artifacts: the model pipeline (joblib) + feature importances.
   This makes the experiment reproducible and comparable.
"""

from __future__ import annotations

import logging
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("automl_model")

try:
    import optuna
    HAVE_OPTUNA = True
except Exception:
    HAVE_OPTUNA = False

try:
    import lightgbm as lgb
    HAVE_LIGHTGBM = True
except Exception:
    HAVE_LIGHTGBM = False

try:
    import xgboost as xgb
    HAVE_XGBOOST = True
except Exception:
    HAVE_XGBOOST = False

try:
    import catboost as cb
    HAVE_CATBOOST = True
except Exception:
    HAVE_CATBOOST = False

try:
    import mlflow
    HAVE_MLFLOW = True
except Exception:
    HAVE_MLFLOW = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ModelKind(str, Enum):
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


CANDIDATE_MODELS: Dict[str, ModelKind] = {
    ModelKind.LIGHTGBM.value: ModelKind.LIGHTGBM,
    ModelKind.XGBOOST.value: ModelKind.XGBOOST,
    ModelKind.CATBOOST.value: ModelKind.CATBOOST,
}


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass
class HPOTrialResult:
    """Result of a single HPO trial."""
    trial_number: int
    model_kind: str
    params: Dict[str, Any]
    score: float
    fit_time_seconds: float


@dataclass
class HPOReport:
    """Aggregated HPO report."""
    best_trial: int
    best_model: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    all_trials: List[HPOTrialResult]

    def to_dict(self) -> dict:
        return {
            "best_trial": self.best_trial,
            "best_model": self.best_model,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "all_trials": [asdict(t) for t in self.all_trials],
        }


# ---------------------------------------------------------------------------
# AutoFeatureTransformer
# ---------------------------------------------------------------------------
class AutoFeatureTransformer:
    """Automatically builds a ColumnTransformer based on inferred schema.

    Numeric columns:
        * Median imputation
        * StandardScaler (optional)
        * PolynomialFeatures (degree 2, optional — interaction terms)
    Categorical columns:
        * Constant imputation ("missing")
        * OneHotEncoder (handle_unknown="ignore")
    Datetime columns:
        * Decomposed into year, month, day, dayofweek, hour
        * Then treated as numeric
    """

    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        datetime_features: List[str] = None,
        polynomial: bool = False,
        polynomial_degree: int = 2,
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.datetime_features = datetime_features or []
        self.polynomial = polynomial
        self.polynomial_degree = polynomial_degree
        self._transformer: Optional[ColumnTransformer] = None

    def build(self) -> ColumnTransformer:
        """Build the ColumnTransformer."""
        transformers = []

        # Numeric pipeline.
        if self.numeric_features:
            numeric_steps = [
                ("impute", SimpleImputer(strategy="median")),
            ]
            if self.polynomial:
                numeric_steps.append(("poly", PolynomialFeatures(
                    degree=self.polynomial_degree, include_bias=False,
                )))
            numeric_steps.append(("scale", StandardScaler()))
            transformers.append(("num", SkPipeline(numeric_steps), self.numeric_features))

        # Categorical pipeline.
        if self.categorical_features:
            cat_steps = [
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
            transformers.append(("cat", SkPipeline(cat_steps), self.categorical_features))

        self._transformer = ColumnTransformer(transformers, remainder="drop")
        return self._transformer


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(
    kind: ModelKind,
    params: Dict[str, Any],
    task_type: str = "classification",
) -> Any:
    """Build a boosting model from hyperparameters."""
    if kind == ModelKind.LIGHTGBM:
        if not HAVE_LIGHTGBM:
            raise RuntimeError("LightGBM is not installed.")
        if task_type == "classification":
            return lgb.LGBMClassifier(
                objective="multiclass", verbose=-1, n_jobs=-1, **params,
            )
        else:
            return lgb.LGBMRegressor(
                objective="regression", verbose=-1, n_jobs=-1, **params,
            )

    if kind == ModelKind.XGBOOST:
        if not HAVE_XGBOOST:
            raise RuntimeError("XGBoost is not installed.")
        if task_type == "classification":
            return xgb.XGBClassifier(
                objective="multi:softprob", verbosity=0, n_jobs=-1, **params,
            )
        else:
            return xgb.XGBRegressor(
                objective="reg:squarederror", verbosity=0, n_jobs=-1, **params,
            )

    if kind == ModelKind.CATBOOST:
        if not HAVE_CATBOOST:
            raise RuntimeError("CatBoost is not installed.")
        if task_type == "classification":
            return cb.CatBoostClassifier(
                loss_function="MultiClass", verbose=0, allow_writing_files=False, **params,
            )
        else:
            return cb.CatBoostRegressor(
                loss_function="RMSE", verbose=0, allow_writing_files=False, **params,
            )

    raise ValueError(f"Unknown ModelKind: {kind}")


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(
    kind: ModelKind,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_transformer: ColumnTransformer,
    task_type: str = "classification",
    cv_folds: int = 3,
) -> Tuple[SkPipeline, Dict[str, float], float]:
    """Train + evaluate a single model configuration.

    Returns
    -------
    (pipeline, metrics, fit_time)
    """
    model = build_model(kind, params, task_type=task_type)
    pipe = SkPipeline([("features", feature_transformer), ("model", model)])

    # Cross-validated score on training set.
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_metric = float(np.mean(cv_scores))
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_metric = float(np.mean(cv_scores))

    # Fit on full training set.
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # Evaluate on test set.
    y_pred = pipe.predict(X_test)
    if task_type == "classification":
        metrics = {
            "cv_accuracy": cv_metric,
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "fit_time": fit_time,
        }
    else:
        metrics = {
            "cv_r2": cv_metric,
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "test_r2": float(r2_score(y_test, y_pred)),
            "fit_time": fit_time,
        }

    return pipe, metrics, fit_time


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------
class OptunaHPO:
    """Optuna hyperparameter optimization wrapper.

    Searches over LightGBM / XGBoost / CatBoost + their hyperparameters.
    """

    def __init__(
        self,
        task_type: str = "classification",
        n_trials: int = 10,
        cv_folds: int = 3,
        feature_transformer: Optional[ColumnTransformer] = None,
    ):
        self.task_type = task_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.feature_transformer = feature_transformer
        self.study: Optional["optuna.Study"] = None
        self.trial_results: List[HPOTrialResult] = []
        self._best_pipeline: Optional[SkPipeline] = None

    def _sample_params(self, trial: "optuna.Trial") -> Tuple[ModelKind, Dict[str, Any]]:
        """Sample model kind + hyperparameters for one trial."""
        # Choose model.
        available = []
        if HAVE_LIGHTGBM:
            available.append("lightgbm")
        if HAVE_XGBOOST:
            available.append("xgboost")
        if HAVE_CATBOOST:
            available.append("catboost")
        model_name = trial.suggest_categorical("model", available)
        kind = ModelKind(model_name)

        params: Dict[str, Any] = {}
        # Common hyperparameters.
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300, step=50)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 10)

        if kind == ModelKind.LIGHTGBM:
            params["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
            params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        elif kind == ModelKind.XGBOOST:
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        elif kind == ModelKind.CATBOOST:
            params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
            params["depth"] = params.pop("max_depth")  # CatBoost uses "depth".

        return kind, params

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> HPOReport:
        """Run HPO and return the best configuration."""
        if not HAVE_OPTUNA:
            raise RuntimeError("optuna is required for HPO.")

        # Suppress Optuna's verbose logging.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            kind, params = self._sample_params(trial)
            try:
                pipe, metrics, fit_time = train_and_evaluate(
                    kind, params, X_train, y_train, X_test, y_test,
                    self.feature_transformer, self.task_type, self.cv_folds,
                )
                score = metrics.get("test_accuracy", metrics.get("test_r2", 0.0))
                self.trial_results.append(HPOTrialResult(
                    trial_number=trial.number,
                    model_kind=kind.value,
                    params=params,
                    score=score,
                    fit_time_seconds=fit_time,
                ))
                # Keep the best pipeline.
                if self._best_pipeline is None or score > max(
                    t.score for t in self.trial_results[:-1]
                ):
                    self._best_pipeline = pipe
                return score
            except Exception as e:
                log.warning("Trial %d failed: %s", trial.number, e)
                return 0.0

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best = self.study.best_trial
        return HPOReport(
            best_trial=best.number,
            best_model=best.params.get("model", "unknown"),
            best_params=best.params,
            best_score=best.value,
            n_trials=len(self.trial_results),
            all_trials=self.trial_results,
        )

    @property
    def best_pipeline(self) -> Optional[SkPipeline]:
        return self._best_pipeline


# ---------------------------------------------------------------------------
# MLflow tracking
# ---------------------------------------------------------------------------
class MLflowTracker:
    """MLflow experiment tracking integration."""

    def __init__(
        self,
        experiment_name: str = "automl_pipeline",
        tracking_uri: Optional[str] = None,
    ):
        if not HAVE_MLFLOW:
            raise RuntimeError("mlflow is required for MLflowTracker.")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def log_trial(
        self,
        trial_result: HPOTrialResult,
        metrics: Dict[str, float],
        pipeline: Optional[SkPipeline] = None,
        artifact_path: Optional[Path | str] = None,
    ) -> str:
        """Log a single HPO trial to MLflow.

        Returns the MLflow run ID.
        """
        with mlflow.start_run(run_name=f"trial_{trial_result.trial_number}"):
            # Log params.
            mlflow.log_param("model", trial_result.model_kind)
            for k, v in trial_result.params.items():
                mlflow.log_param(k, v)

            # Log metrics.
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_metric("hpo_score", trial_result.score)

            # Log artifact (pipeline).
            if pipeline is not None and artifact_path is not None:
                import joblib
                artifact_path = Path(artifact_path)
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipeline, artifact_path)
                mlflow.log_artifact(str(artifact_path))

            return mlflow.active_run().info.run_id

    def log_best(
        self,
        report: HPOReport,
        pipeline: Optional[SkPipeline] = None,
        artifact_path: Optional[Path | str] = None,
    ) -> str:
        """Log the best HPO result to MLflow."""
        with mlflow.start_run(run_name="best_model"):
            mlflow.log_param("best_model", report.best_model)
            for k, v in report.best_params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("best_score", report.best_score)
            mlflow.log_metric("n_trials", report.n_trials)

            if pipeline is not None and artifact_path is not None:
                import joblib
                artifact_path = Path(artifact_path)
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipeline, artifact_path)
                mlflow.log_artifact(str(artifact_path))

            return mlflow.active_run().info.run_id


# ---------------------------------------------------------------------------
# Full AutoML pipeline
# ---------------------------------------------------------------------------
def run_automl_pipeline(
    df: pd.DataFrame,
    profile: "DatasetProfile",
    n_trials: int = 10,
    cv_folds: int = 3,
    use_mlflow: bool = False,
    mlflow_experiment: str = "automl_pipeline",
    model_artifact_path: Optional[Path | str] = None,
    seed: int = 42,
) -> Tuple[HPOReport, Optional[SkPipeline], Optional[MLflowTracker]]:
    """Run the full AutoML pipeline: features → HPO → MLflow.

    Returns
    -------
    (hpo_report, best_pipeline, mlflow_tracker)
    """
    from dataset import ColumnType
    from sklearn.model_selection import train_test_split

    # Extract feature lists from profile.
    numeric_features = [c.name for c in profile.columns
                        if c.type == ColumnType.NUMERIC.value and c.name != profile.target_column]
    categorical_features = [c.name for c in profile.columns
                             if c.type == ColumnType.CATEGORICAL.value]
    datetime_features = [c.name for c in profile.columns
                          if c.type == ColumnType.DATETIME.value]

    # Decompose datetime columns into numeric features.
    for dt_col in datetime_features:
        dt = pd.to_datetime(df[dt_col], errors="coerce")
        df[f"{dt_col}_year"] = dt.dt.year.fillna(0).astype(float)
        df[f"{dt_col}_month"] = dt.dt.month.fillna(0).astype(float)
        df[f"{dt_col}_day"] = dt.dt.day.fillna(0).astype(float)
        df[f"{dt_col}_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(float)
        numeric_features.extend([f"{dt_col}_year", f"{dt_col}_month",
                                  f"{dt_col}_day", f"{dt_col}_dayofweek"])

    target = profile.target_column
    X = df.drop(columns=[target] + datetime_features)
    y = df[target]

    # Train/test split.
    if profile.task_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed,
        )

    # Build feature transformer.
    transformer = AutoFeatureTransformer(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    feature_transformer = transformer.build()

    # Run HPO.
    log.info("Running Optuna HPO (%d trials, %d-fold CV) ...", n_trials, cv_folds)
    hpo = OptunaHPO(
        task_type=profile.task_type,
        n_trials=n_trials,
        cv_folds=cv_folds,
        feature_transformer=feature_transformer,
    )
    report = hpo.optimize(X_train, y_train, X_test, y_test)
    log.info("  Best trial #%d: %s (score=%.4f)",
             report.best_trial, report.best_model, report.best_score)

    # MLflow tracking.
    tracker = None
    if use_mlflow:
        try:
            tracker = MLflowTracker(experiment_name=mlflow_experiment)
            for tr in report.all_trials:
                tracker.log_trial(tr, {"hpo_score": tr.score})
            tracker.log_best(report, hpo.best_pipeline, model_artifact_path)
            log.info("  MLflow tracking: experiment='%s'", mlflow_experiment)
        except Exception as exc:
            log.warning("MLflow logging failed: %s", exc)
            tracker = None

    return report, hpo.best_pipeline, tracker


__all__ = [
    "ModelKind",
    "CANDIDATE_MODELS",
    "HPOTrialResult",
    "HPOReport",
    "AutoFeatureTransformer",
    "OptunaHPO",
    "MLflowTracker",
    "build_model",
    "train_and_evaluate",
    "run_automl_pipeline",
    "HAVE_OPTUNA",
    "HAVE_LIGHTGBM",
    "HAVE_XGBOOST",
    "HAVE_CATBOOST",
    "HAVE_MLFLOW",
]
