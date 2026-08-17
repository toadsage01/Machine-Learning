"""
model
=====

Production ML pipeline definitions for the Iris classification task.

Public surface
--------------
- ``ModelKind``            : enum of supported model families.
- ``build_pipeline``       : construct a sklearn Pipeline with a
                             ``ColumnTransformer`` (StandardScaler for numeric
                             features, passthrough for categorical) followed
                             by a configurable classifier.
- ``CANDIDATE_MODELS``     : factory dict for the 4 candidate models
                             (LogReg, RF, SVM, LightGBM).
- ``evaluate_pipeline``    : cross_validate + holdout test metrics.
- ``explain_with_shap``    : compute SHAP values for a fitted tree-based pipeline.
- ``export_to_onnx``        : serialize a fitted pipeline to ONNX using skl2onnx.
- ``save_pipeline`` / ``load_pipeline`` : joblib persistence.

Design notes
------------
1. **Pipeline shape** — every candidate is a single ``Pipeline`` object:
       ColumnTransformer([("num", StandardScaler(), numeric_features)])
         → classifier
   This is critical for ONNX export (skl2onnx can serialize the whole
   pipeline, including the scaler, as a single ONNX graph) and for serving
   (the FastAPI app receives raw features and the ONNX runtime handles
   scaling internally).

2. **Scaling only on numeric features** — Iris has 4 numeric features and
   zero categoricals, but the ColumnTransformer is set up generically so
   that adding e.g. a "region" categorical feature later requires only a
   schema change, not a code change.

3. **SHAP integration** — for tree-based models (RF, LightGBM) we use
   ``shap.TreeExplainer`` which is exact and fast. For LogReg/SVM we
   would fall back to ``shap.LinearExplainer`` / ``shap.KernelExplainer``
   — for Iris the tree-based models dominate anyway, so SHAP is
   primarily demonstrated on those.

4. **ONNX export** — uses ``skl2onnx.to_onnx`` with explicit ``target_types``
   and ``target_names`` so the resulting ONNX graph has interpretable
   output names (``" probabilities"`` + ``"label"``). Float32 is used for
   inference efficiency on CPU.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    log_loss, confusion_matrix,
)
from sklearn.model_selection import cross_val_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import FEATURE_NAMES, TARGET_NAMES, FeatureSchema  # noqa: E402

# LightGBM is the only optional dependency; degrade gracefully if missing.
try:
    from lightgbm import LGBMClassifier
    HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover
    HAVE_LIGHTGBM = False

# SHAP is optional too (it pulls in numba which is heavy on constrained envs).
try:
    import shap
    HAVE_SHAP = True
except Exception:  # pragma: no cover
    HAVE_SHAP = False

# ONNX export is optional (skl2onnx has a stricter dep tree).
try:
    from skl2onnx import to_onnx
    HAVE_SKL2ONNX = True
except Exception:  # pragma: no cover
    HAVE_SKL2ONNX = False

try:
    import onnxruntime as ort
    HAVE_ONNXRUNTIME = True
except Exception:  # pragma: no cover
    HAVE_ONNXRUNTIME = False

try:
    import joblib
    HAVE_JOBLIB = True
except Exception:  # pragma: no cover
    HAVE_JOBLIB = False


# ---------------------------------------------------------------------------
# Model kinds
# ---------------------------------------------------------------------------
class ModelKind(str, Enum):
    LOGREG = "logreg"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    LIGHTGBM = "lightgbm"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


# ---------------------------------------------------------------------------
# Metrics value object
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    """Holdout test metrics for a single trained pipeline."""

    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    roc_auc_ovr: Optional[float]
    log_loss: Optional[float]
    cv_accuracy_mean: float
    cv_accuracy_std: float
    confusion_matrix: List[List[int]]
    fit_time_seconds: float
    predict_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            **{k: v for k, v in self.__dict__.items()},
            "confusion_matrix": self.confusion_matrix,
        }


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def _make_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    """ColumnTransformer: scale numerics, passthrough categoricals.

    Uses integer column indices (rather than string names) so the pipeline
    accepts both numpy arrays and pandas DataFrames. With string column
    names, sklearn's ColumnTransformer raises ``ValueError: Specifying the
    columns using strings is only supported for dataframes`` when given a
    numpy array — and the ONNX runtime always feeds the pipeline numpy
    arrays at inference time.
    """
    transformers: List[Tuple[str, Any, List[int]]] = []
    if schema.numeric_features:
        # All numeric features are the first N columns of the Iris matrix.
        # (FeatureSchema stores names, but their positional indices are
        # range(len(numeric_features)) because dataset.py emits them in order.)
        num_idx = list(range(len(schema.numeric_features)))
        transformers.append(("num", StandardScaler(), num_idx))
    if schema.categorical_features:
        # Categorical features start after the numeric ones.
        cat_idx = list(range(len(schema.numeric_features),
                              len(schema.numeric_features) + len(schema.categorical_features)))
        transformers.append(("cat", "passthrough", cat_idx))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_classifier(kind: ModelKind, random_state: int = 42) -> Any:
    """Instantiate a fresh classifier of the requested kind."""
    if kind == ModelKind.LOGREG:
        # `multi_class="multinomial"` is deprecated in sklearn>=1.5; the
        # default behaviour already does multinomial for solver="lbfgs".
        return LogisticRegression(
            max_iter=500, solver="lbfgs",
            random_state=random_state,
        )
    if kind == ModelKind.RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=1,
            n_jobs=-1, random_state=random_state,
        )
    if kind == ModelKind.SVM:
        # probability=True is required for roc_auc + log_loss.
        return SVC(
            C=1.0, kernel="rbf", probability=True,
            random_state=random_state,
        )
    if kind == ModelKind.LIGHTGBM:
        if not HAVE_LIGHTGBM:
            raise RuntimeError("LightGBM is not installed but was requested.")
        return LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            objective="multiclass", n_jobs=-1, random_state=random_state,
            verbose=-1,
        )
    raise ValueError(f"Unknown ModelKind: {kind}")


def build_pipeline(kind: ModelKind, schema: Optional[FeatureSchema] = None,
                   random_state: int = 42) -> Pipeline:
    """Build a sklearn Pipeline: ColumnTransformer → classifier.

    Parameters
    ----------
    kind : ModelKind
        Which classifier to attach.
    schema : FeatureSchema, optional
        Defaults to the Iris schema.
    random_state : int
        Seed propagated into every stochastic component.

    Returns
    -------
    Pipeline
        Unfitted pipeline.
    """
    schema = schema or FeatureSchema()
    preprocessor = _make_preprocessor(schema)
    classifier = _make_classifier(kind, random_state=random_state)
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# Candidate models registry
# ---------------------------------------------------------------------------
CANDIDATE_MODELS: Dict[str, ModelKind] = {
    ModelKind.LOGREG.value: ModelKind.LOGREG,
    ModelKind.RANDOM_FOREST.value: ModelKind.RANDOM_FOREST,
    ModelKind.SVM.value: ModelKind.SVM,
    ModelKind.LIGHTGBM.value: ModelKind.LIGHTGBM,
}


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------
def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> Optional[float]:
    """roc_auc_score with graceful failure for degenerate folds."""
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        return None


def _safe_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> Optional[float]:
    try:
        return float(log_loss(y_true, y_proba))
    except Exception:
        return None


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
) -> Metrics:
    """Fit ``pipeline`` on (X_train, y_train) and evaluate on (X_test, y_test).

    Returns a ``Metrics`` object with holdout metrics + CV accuracy.
    """
    # Cross-validated accuracy on the *training* set (out-of-fold estimate).
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1)

    # Fit on the full training fold.
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # Predict + measure.
    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time = time.perf_counter() - t0

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    y_proba: Optional[np.ndarray] = None
    try:
        y_proba = pipeline.predict_proba(X_test)
    except Exception:
        y_proba = None

    cm = confusion_matrix(y_test, y_pred).tolist()

    return Metrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro")),
        precision_macro=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        roc_auc_ovr=_safe_roc_auc(y_test, y_proba, n_classes) if y_proba is not None else None,
        log_loss=_safe_log_loss(y_test, y_proba) if y_proba is not None else None,
        cv_accuracy_mean=float(np.mean(cv_scores)),
        cv_accuracy_std=float(np.std(cv_scores)),
        confusion_matrix=cm,
        fit_time_seconds=fit_time,
        predict_time_seconds=predict_time,
    )


# ---------------------------------------------------------------------------
# SHAP explainability hook
# ---------------------------------------------------------------------------
@dataclass
class ShapExplanation:
    """SHAP values + base values + feature names for a single prediction set."""

    feature_names: List[str]
    target_names: List[str]
    values: Any           # shape (n_samples, n_features, n_classes) for tree
    base_values: Any      # shape (n_classes,) or (n_samples, n_classes)
    data: Any             # original input features (raw, unscaled)
    explainer_type: str   # "tree" | "linear" | "kernel"

    def summary_for_class(self, class_idx: int) -> Dict[str, float]:
        """Mean |SHAP| per feature, for the requested class index."""
        if self.values is None:
            return {}
        # values shape: (n_samples, n_features, n_classes) — pick class slice.
        class_vals = np.abs(self.values[:, :, class_idx])
        means = class_vals.mean(axis=0)
        return {name: float(v) for name, v in zip(self.feature_names, means)}


def explain_with_shap(
    pipeline: Pipeline,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
    n_background: int = 50,
) -> Optional[ShapExplanation]:
    """Compute SHAP values for a fitted pipeline.

    Uses ``shap.TreeExplainer`` for tree-based classifiers (RF, LightGBM)
    and falls back to ``shap.LinearExplainer`` for LogReg. SVM uses
    ``shap.KernelExplainer`` (slow) and is skipped by default — set
    ``pipeline.named_steps['classifier']`` to a LinearSVC + Calibrated
    if you need SVM SHAP in production.

    Parameters
    ----------
    pipeline : Pipeline
        Must be fitted.
    X_background : np.ndarray
        Background data for the explainer (typically the training fold).
        Subsampled to ``n_background`` rows.
    X_explain : np.ndarray
        Rows to compute SHAP values for.
    feature_names, target_names : list[str], optional
        Override the defaults from ``dataset.py``.

    Returns
    -------
    ShapExplanation or None
        ``None`` if SHAP is unavailable or the classifier kind is unsupported.
    """
    if not HAVE_SHAP:
        return None
    if feature_names is None:
        feature_names = list(FEATURE_NAMES)
    if target_names is None:
        target_names = list(TARGET_NAMES)

    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocess"]

    # Subsample the background set for KernelExplainer speed.
    if len(X_background) > n_background:
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X_background), size=n_background, replace=False)
        X_background = X_background[bg_idx]

    # Transform the inputs the same way the pipeline does.
    X_bg_transformed = preprocessor.transform(X_background)
    X_explain_transformed = preprocessor.transform(X_explain)

    explainer_type = "unknown"
    values = None
    base_values = None
    data = X_explain

    # Tree-based models: exact & fast TreeExplainer.
    if HAVE_LIGHTGBM and isinstance(clf, LGBMClassifier):
        explainer = shap.TreeExplainer(clf)
        shap_obj = explainer(X_explain_transformed)
        values = shap_obj.values  # (n_samples, n_features, n_classes)
        base_values = shap_obj.base_values
        explainer_type = "tree"
    elif isinstance(clf, RandomForestClassifier):
        explainer = shap.TreeExplainer(clf)
        shap_obj = explainer(X_explain_transformed)
        values = shap_obj.values
        base_values = shap_obj.base_values
        explainer_type = "tree"
    elif isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_bg_transformed)
        shap_obj = explainer(X_explain_transformed)
        # LinearExplainer returns shape (n_samples, n_features, n_classes)
        # in shap>=0.42 for multiclass.
        values = shap_obj.values
        base_values = shap_obj.base_values
        explainer_type = "linear"
    else:
        # SVM / other — fall back to KernelExplainer only on small inputs.
        if len(X_explain) <= 50:
            explainer = shap.KernelExplainer(clf.predict_proba, X_bg_transformed)
            values = explainer.shap_values(X_explain_transformed, silent=True)
            base_values = explainer.expected_value
            explainer_type = "kernel"
        else:
            return None

    return ShapExplanation(
        feature_names=feature_names,
        target_names=target_names,
        values=values,
        base_values=base_values,
        data=data,
        explainer_type=explainer_type,
    )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_to_onnx(
    pipeline: Pipeline,
    output_path: Path | str,
    feature_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
    n_samples: int = 150,
) -> Path:
    """Serialize a fitted pipeline to ONNX.

    Parameters
    ----------
    pipeline : Pipeline
        Must be fitted.
    output_path : str or Path
        Destination ``.onnx`` file.
    feature_names, target_names : list[str], optional
        Override schema defaults.
    n_samples : int
        Number of rows used to infer ONNX input shape (does not affect the
        model, just the graph signature).

    Returns
    -------
    Path
        Resolved output path.

    Raises
    ------
    RuntimeError
        If skl2onnx is unavailable or conversion fails.
    """
    if not HAVE_SKL2ONNX:
        raise RuntimeError(
            "skl2onnx is not installed; cannot export to ONNX. "
            "Run `pip install skl2onnx onnxruntime` to enable."
        )
    # skl2onnx renamed Float32TensorType -> FloatTensorType between major
    # versions. Try both for forward/backward compatibility.
    try:
        from skl2onnx.common.data_types import FloatTensorType as _FloatTensorType
    except ImportError:  # pragma: no cover
        from skl2onnx.common.data_types import Float32TensorType as _FloatTensorType  # type: ignore

    feature_names = feature_names or list(FEATURE_NAMES)
    target_names = target_names or list(TARGET_NAMES)

    # skl2onnx expects an initial_type spec in the form
    # [(name, TensorType([None, n_features]))].
    initial_type = [("input", _FloatTensorType([None, len(feature_names)]))]

    # For multiclass, ask skl2onnx to NOT emit a list-of-dicts (zipmap=False)
    # so the output is a clean (n, n_classes) tensor we can index into.
    options = {id(pipeline): {"zipmap": False}}

    onnx_model = to_onnx(
        pipeline,
        initial_types=initial_type,
        target_opset=15,
        options=options,
        name="iris_production_pipeline",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())
    return output_path


def load_onnx_session(onnx_path: Path | str) -> "ort.InferenceSession":
    """Load an ONNX model into an ``onnxruntime.InferenceSession``."""
    if not HAVE_ONNXRUNTIME:
        raise RuntimeError("onnxruntime is not installed.")
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def predict_with_onnx(
    session: "ort.InferenceSession",
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference via ONNX runtime.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
        Loaded ONNX session.
    X : np.ndarray
        Raw input features, shape (n, 4). Will be cast to float32.

    Returns
    -------
    (labels, probabilities) : tuple[np.ndarray, np.ndarray]
        ``labels`` has shape (n,) int64; ``probabilities`` has shape (n, 3) float32.
    """
    if not HAVE_ONNXRUNTIME:
        raise RuntimeError("onnxruntime is not installed.")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X})
    # skl2onnx convention: first output = label, second = probabilities.
    # But the order depends on options — find them by name.
    label_out = None
    proba_out = None
    for o in session.get_outputs():
        if "label" in o.name.lower():
            label_out = outputs[session.get_outputs().index(o)]
        elif "probabilities" in o.name.lower() or "probability" in o.name.lower():
            proba_out = outputs[session.get_outputs().index(o)]
    if label_out is None:
        label_out = outputs[0]
    if proba_out is None:
        proba_out = outputs[1] if len(outputs) > 1 else None

    labels = np.asarray(label_out).ravel().astype(np.int64)
    if proba_out is not None:
        probas = np.asarray(proba_out, dtype=np.float32)
    else:
        probas = np.zeros((len(labels), 3), dtype=np.float32)
    return labels, probas


# ---------------------------------------------------------------------------
# Joblib persistence (fallback for environments without ONNX)
# ---------------------------------------------------------------------------
def save_pipeline(pipeline: Pipeline, path: Path | str) -> Path:
    """Persist a fitted pipeline to disk via joblib."""
    if not HAVE_JOBLIB:
        raise RuntimeError("joblib is not installed.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path: Path | str) -> Pipeline:
    """Load a joblib-persisted pipeline."""
    if not HAVE_JOBLIB:
        raise RuntimeError("joblib is not installed.")
    return joblib.load(path)


__all__ = [
    "ModelKind",
    "CANDIDATE_MODELS",
    "Metrics",
    "ShapExplanation",
    "build_pipeline",
    "evaluate_pipeline",
    "explain_with_shap",
    "export_to_onnx",
    "load_onnx_session",
    "predict_with_onnx",
    "save_pipeline",
    "load_pipeline",
    "HAVE_LIGHTGBM",
    "HAVE_SHAP",
    "HAVE_SKL2ONNX",
    "HAVE_ONNXRUNTIME",
]
