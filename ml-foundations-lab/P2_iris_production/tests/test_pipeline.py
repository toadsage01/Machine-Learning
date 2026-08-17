"""
tests/test_pipeline
===================

End-to-end tests for the P2 Iris production pipeline.

Covers:
    * Dataset loading (sklearn + CSV fallback).
    * Train/test split shape & stratification.
    * Pipeline construction for all 4 candidate model kinds.
    * Training + evaluation produces sane metrics.
    * ONNX export round-trips: sklearn and onnxruntime agree.
    * SHAP hook returns non-None for tree-based models.
    * FastAPI app: /health + /predict + /predict/batch + validation 422.

Run with::

    cd ml-foundations-lab/P2_iris_production
    python -m pytest tests/ -v

or::

    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Make project & repo root importable.
REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import (  # noqa: E402
    load_iris_data, load_iris_split, build_train_test,
    FEATURE_NAMES, TARGET_NAMES, IrisDataset,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, ModelKind, build_pipeline, evaluate_pipeline,
    export_to_onnx, load_onnx_session, predict_with_onnx,
    explain_with_shap, HAVE_SHAP, HAVE_SKL2ONNX, HAVE_ONNXRUNTIME,
)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------
def test_load_from_sklearn():
    X, y = load_iris_data()
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert set(np.unique(y).tolist()) == {0, 1, 2}
    assert X.dtype == np.float64
    assert y.dtype == np.int64


def test_load_from_csv_roundtrip():
    X, y = load_iris_data()
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "iris.csv"
        import pandas as pd
        df = pd.DataFrame(X, columns=list(FEATURE_NAMES))
        df["target"] = y
        df.to_csv(csv_path, index=False)
        X2, y2 = load_iris_data(csv_path=csv_path)
        assert X2.shape == X.shape
        assert y2.shape == y.shape
        np.testing.assert_allclose(X2, X)
        np.testing.assert_array_equal(y2, y)


def test_train_test_split_stratified():
    ds = load_iris_split(test_size=0.2, random_state=42)
    assert isinstance(ds, IrisDataset)
    assert ds.X_train.shape == (120, 4)
    assert ds.X_test.shape == (30, 4)
    assert ds.y_train.shape == (120,)
    assert ds.y_test.shape == (30,)
    # Stratification: each class should be 40 train / 10 test (roughly).
    train_counts = np.bincount(ds.y_train, minlength=3)
    test_counts = np.bincount(ds.y_test, minlength=3)
    assert all(c >= 35 for c in train_counts), f"Train unbalanced: {train_counts}"
    assert all(c >= 8 for c in test_counts), f"Test unbalanced: {test_counts}"


def test_split_reproducible():
    ds1 = load_iris_split(random_state=42)
    ds2 = load_iris_split(random_state=42)
    np.testing.assert_array_equal(ds1.y_test, ds2.y_test)


# ---------------------------------------------------------------------------
# Pipeline + training tests
# ---------------------------------------------------------------------------
def test_build_pipeline_for_all_kinds():
    for kind_str, kind in CANDIDATE_MODELS.items():
        pipe = build_pipeline(kind)
        # Two named steps: preprocess + classifier.
        assert "preprocess" in pipe.named_steps
        assert "classifier" in pipe.named_steps
        assert pipe.named_steps["classifier"].__class__.__name__


def test_evaluate_logreg_produces_sane_metrics():
    ds = load_iris_split()
    pipe = build_pipeline(ModelKind.LOGREG)
    m = evaluate_pipeline(pipe, ds.X_train, ds.y_train, ds.X_test, ds.y_test, cv_folds=3)
    # Iris is easy — accuracy should be ≥ 0.85 for any sane model.
    assert m.accuracy >= 0.85, f"LogReg accuracy too low: {m.accuracy}"
    assert m.f1_macro >= 0.85
    assert m.cv_accuracy_mean >= 0.85
    assert m.cv_accuracy_std >= 0.0
    assert m.fit_time_seconds >= 0.0
    assert len(m.confusion_matrix) == 3  # 3 classes
    assert m.roc_auc_ovr is not None
    assert m.log_loss is not None


# ---------------------------------------------------------------------------
# ONNX round-trip
# ---------------------------------------------------------------------------
def test_onnx_export_and_inference_match_sklearn():
    if not (HAVE_SKL2ONNX and HAVE_ONNXRUNTIME):
        return  # skip when deps missing
    ds = load_iris_split()
    pipe = build_pipeline(ModelKind.LOGREG)
    pipe.fit(ds.X_train, ds.y_train)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        export_to_onnx(pipe, onnx_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

        session = load_onnx_session(onnx_path)
        labels, probas = predict_with_onnx(session, ds.X_test.astype(np.float32))

        # ONNX predictions should match sklearn's to within float32 tolerance.
        sklearn_pred = pipe.predict(ds.X_test)
        agreement = (labels == sklearn_pred).mean()
        assert agreement >= 0.95, f"ONNX/sklearn disagreement: {agreement:.2%}"

        # Probability sum per row should be ≈ 1.0.
        np.testing.assert_allclose(probas.sum(axis=1), np.ones(len(probas)), atol=1e-4)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def test_shap_on_random_forest():
    if not HAVE_SHAP:
        return
    ds = load_iris_split()
    pipe = build_pipeline(ModelKind.RANDOM_FOREST)
    pipe.fit(ds.X_train, ds.y_train)
    explanation = explain_with_shap(
        pipe,
        X_background=ds.X_train,
        X_explain=ds.X_test[:5],
        feature_names=list(FEATURE_NAMES),
        target_names=list(TARGET_NAMES),
    )
    assert explanation is not None
    assert explanation.explainer_type == "tree"
    # SHAP values should have shape (5, 4, 3): 5 rows × 4 features × 3 classes.
    arr = np.asarray(explanation.values)
    assert arr.shape == (5, 4, 3), f"Unexpected SHAP shape: {arr.shape}"
    # mean(|SHAP|) per class should be non-negative.
    for ci in range(3):
        summary = explanation.summary_for_class(ci)
        assert all(v >= 0 for v in summary.values())


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def test_fastapi_endpoints():
    from fastapi.testclient import TestClient
    # Ensure a model exists.
    models_dir = PROJECT_ROOT / "models"
    onnx_path = models_dir / "best.onnx"
    if not onnx_path.exists():
        # Train a quick LogReg-only model for the test.
        import subprocess
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "train.py"),
             "--models", "logreg",
             "--out", str(onnx_path),
             "--joblib-out", str(models_dir / "best.joblib")],
            check=True, capture_output=True,
        )

    # Import AFTER ensuring the ONNX file exists, so get_session() succeeds.
    from app import app
    client = TestClient(app)

    # /health
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["onnx_loaded"] is True
    assert set(body["feature_names"]) == set(FEATURE_NAMES)
    assert set(body["target_names"]) == set(TARGET_NAMES)

    # /predict — setosa-shaped row.
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200, r.text
    pred = r.json()
    assert pred["predicted_class"] == 0
    assert pred["predicted_label"] == "setosa"
    assert abs(sum(pred["probabilities"]) - 1.0) < 1e-4

    # /predict/batch — three canonical examples.
    r = client.post("/predict/batch", json={
        "rows": [
            {"features": [5.1, 3.5, 1.4, 0.2]},  # setosa
            {"features": [6.2, 2.9, 4.3, 1.3]},  # versicolor
            {"features": [7.7, 3.8, 6.7, 2.2]},  # virginica
        ]
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_rows"] == 3
    labels = [p["predicted_label"] for p in body["predictions"]]
    assert labels == ["setosa", "versicolor", "virginica"], labels

    # Validation: short feature vector → 422.
    r = client.post("/predict", json={"features": [5.1, 3.5]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_load_from_sklearn,
        test_load_from_csv_roundtrip,
        test_train_test_split_stratified,
        test_split_reproducible,
        test_build_pipeline_for_all_kinds,
        test_evaluate_logreg_produces_sane_metrics,
        test_onnx_export_and_inference_match_sklearn,
        test_shap_on_random_forest,
        test_fastapi_endpoints,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
