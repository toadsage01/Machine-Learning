"""
app
===

FastAPI inference server for the Iris production pipeline.

Loads the ONNX model exported by ``train.py`` and exposes three endpoints:

* ``GET  /``                — health + model metadata.
* ``GET  /health``           — readiness probe for k8s/ELB.
* ``POST /predict``          — single-row prediction (JSON in / JSON out).
* ``POST /predict/batch``    — batch prediction (JSON array in / JSON array out).

Usage
-----
::

    # 1. Train first (writes models/best.onnx):
    python train.py

    # 2. Start the server:
    uvicorn app:app --host 0.0.0.0 --port 8000
    # or
    python app.py --port 8000

    # 3. Call it:
    curl -X POST http://localhost:8000/predict \
         -H 'Content-Type: application/json' \
         -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

Design notes
------------
1. **ONNX runtime, not sklearn** — the server loads the ONNX graph and runs
   inference via ``onnxruntime``. This avoids loading the heavy sklearn +
   LightGBM stack in production, makes the server trivially portable (the
   only Python deps are ``fastapi`` + ``onnxruntime`` + ``pydantic`` + ``uvicorn``),
   and unlocks cross-language serving (the same .onnx file works in C++ /
   Java / Go ONNX runtimes).

2. **Input validation via pydantic** — request bodies are validated before
   they reach the ONNX runtime. Invalid requests return 422 with a
   structured error message.

3. **No global mutable state** — the ONNX session is created once at
   startup via ``lru_cache`` and reused for every request. This is the
   recommended pattern from the FastAPI docs.

4. **Float32 input** — ONNX runtime is most efficient with float32, so
   inputs are cast before inference. The pipeline's ``StandardScaler``
   was serialized into the ONNX graph at export time, so callers send
   **raw** Iris measurements (cm), not pre-scaled features.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Bootstrap project root so `from dataset import ...` works.
_PROJECT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import FEATURE_NAMES, TARGET_NAMES  # noqa: E402
from model import (  # noqa: E402
    load_onnx_session, predict_with_onnx, HAVE_ONNXRUNTIME,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("iris_server")

# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------
DEFAULT_ONNX_PATH = _PROJECT_ROOT / "models" / "best.onnx"
ONNX_PATH = Path(os.environ.get("IRIS_ONNX_PATH", DEFAULT_ONNX_PATH))


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Single-row prediction request."""

    features: List[float] = Field(
        ..., min_length=4, max_length=4,
        description=f"Ordered Iris features: {list(FEATURE_NAMES)}",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )

    @field_validator("features")
    @classmethod
    def _check_finite(cls, v: List[float]) -> List[float]:
        for x in v:
            if not isinstance(x, (int, float)):
                raise ValueError(f"Feature values must be numeric; got {type(x).__name__}")
            if not np.isfinite(x):
                raise ValueError(f"Feature values must be finite; got {x}")
        return [float(x) for x in v]


class PredictResponse(BaseModel):
    """Single-row prediction response."""

    predicted_class: int
    predicted_label: str
    probabilities: List[float]
    feature_names: List[str]
    target_names: List[str]
    inference_ms: float


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    rows: List[PredictRequest] = Field(
        ..., min_length=1, max_length=10000,
        description="List of single-row prediction requests.",
    )


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictResponse]
    total_inference_ms: float
    n_rows: int


class HealthResponse(BaseModel):
    """Readiness probe response."""

    status: str
    onnx_loaded: bool
    onnx_path: str
    feature_names: List[str]
    target_names: List[str]


# ---------------------------------------------------------------------------
# ONNX session — created once at first use, reused thereafter.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_session():
    """Lazily load the ONNX runtime session (singleton)."""
    if not HAVE_ONNXRUNTIME:
        raise RuntimeError("onnxruntime is not installed.")
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {ONNX_PATH}. "
            f"Run `python train.py` first, or set IRIS_ONNX_PATH."
        )
    log.info("Loading ONNX model from %s", ONNX_PATH)
    return load_onnx_session(ONNX_PATH)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Iris Production — Inference Server",
    description="ONNX-backed inference for the P2 Iris classification pipeline.",
    version="1.0.0",
)


@app.get("/", response_model=HealthResponse, tags=["meta"])
def root() -> HealthResponse:
    """Root endpoint — also serves as a discovery probe."""
    onnx_loaded = False
    try:
        get_session()
        onnx_loaded = True
    except Exception as exc:  # pragma: no cover
        log.warning("ONNX not loaded: %s", exc)
    return HealthResponse(
        status="ok" if onnx_loaded else "degraded",
        onnx_loaded=onnx_loaded,
        onnx_path=str(ONNX_PATH),
        feature_names=list(FEATURE_NAMES),
        target_names=list(TARGET_NAMES),
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Kubernetes/ELB readiness probe."""
    return root()


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """Single-row prediction.

    Returns the predicted class index, label, per-class probabilities, and
    inference latency in milliseconds.
    """
    try:
        session = get_session()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    X = np.asarray(req.features, dtype=np.float32).reshape(1, -1)
    t0 = time.perf_counter()
    labels, probas = predict_with_onnx(session, X)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    label_idx = int(labels[0])
    return PredictResponse(
        predicted_class=label_idx,
        predicted_label=TARGET_NAMES[label_idx],
        probabilities=[float(p) for p in probas[0]],
        feature_names=list(FEATURE_NAMES),
        target_names=list(TARGET_NAMES),
        inference_ms=round(dt_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Batch prediction — single ONNX inference call for many rows."""
    try:
        session = get_session()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    X = np.asarray([r.features for r in req.rows], dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    t0 = time.perf_counter()
    labels, probas = predict_with_onnx(session, X)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    predictions: List[PredictResponse] = []
    for i, row in enumerate(req.rows):
        label_idx = int(labels[i])
        predictions.append(PredictResponse(
            predicted_class=label_idx,
            predicted_label=TARGET_NAMES[label_idx],
            probabilities=[float(p) for p in probas[i]],
            feature_names=list(FEATURE_NAMES),
            target_names=list(TARGET_NAMES),
            inference_ms=round(dt_ms / len(req.rows), 3),
        ))

    return BatchPredictResponse(
        predictions=predictions,
        total_inference_ms=round(dt_ms, 3),
        n_rows=len(predictions),
    )


# ---------------------------------------------------------------------------
# CLI entry-point: `python app.py --port 8000`
# ---------------------------------------------------------------------------
def _cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="iris_server", description="Iris ONNX FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")
    args = parser.parse_args()

    try:
        get_session()
        log.info("ONNX session ready, serving on %s:%d", args.host, args.port)
    except Exception as exc:
        log.error("Cannot start server: %s", exc)
        return 1

    import uvicorn
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
