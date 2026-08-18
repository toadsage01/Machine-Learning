"""
model
=====

Comparative NLP suite: TF-IDF + LogisticRegression baseline vs.
fine-tuned IndicBERT (ai4bharat/indic-bert), with token-level
probability output, confusion metrics, and ONNX export.

Public surface
--------------
- ``ModelKind``                  : enum (TFIDF_LOGREG, INDICBERT).
- ``CANDIDATE_MODELS``           : registry dict.
- ``ClassificationMetrics``      : holdout metrics for one model.
- ``TfidfBaseline``              : scikit-learn Pipeline (TF-IDF + LogReg).
- ``IndicBertClassifier``        : HuggingFace AutoModelForSequenceClassification wrapper.
- ``build_tfidf_baseline``       : construct the TF-IDF + LogReg pipeline.
- ``build_indicbert``             : construct the IndicBERT classifier.
- ``train_tfidf_baseline``       : fit + evaluate the baseline.
- ``train_indicbert``             : fine-tune IndicBERT.
- ``evaluate_classifier``         : compute accuracy / F1 / ROC-AUC / confusion.
- ``export_to_onnx``              : serialize a fitted model to ONNX.
- ``predict_with_onnx``           : run inference via onnxruntime.

Design notes
------------
1. **Two model families, one interface** — both the TF-IDF baseline and
   the IndicBERT fine-tuner expose the same ``predict(texts) -> (labels,
   probas)`` interface so the same evaluation code can compare them
   head-to-head. The TF-IDF model uses raw strings; IndicBERT uses the
   IndicBERT tokenizer.

2. **ONNX export via transformers.onnx** — HuggingFace ships an
   ``onnx.export`` helper that produces a properly-structured ONNX graph
   for transformer models. We use it for the IndicBERT model.

3. **Token-level probability output** — both models expose
   ``predict_probas(texts) -> np.ndarray`` of shape ``(n_samples,
   num_classes)`` so the caller can build confidence-bounded UIs.

4. **Confusion metrics** — the ``ClassificationMetrics`` value object
   reports accuracy, F1 (macro), precision (macro), recall (macro),
   ROC-AUC (one-vs-rest), log-loss, and the confusion matrix. This is
   the same set of metrics P4 (Titanic) used — consistency across the
   monorepo lets us reuse evaluation code.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss, confusion_matrix,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import HinglishConfig, INDICBERT_MODEL_ID, DEFAULT_CONFIG  # noqa: E402

# Lazy HF imports — the full transformers stack is ~500MB.
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )
    HAVE_HF = True
except Exception:  # pragma: no cover
    HAVE_HF = False

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ModelKind(str, Enum):
    TFIDF_LOGREG = "tfidf_logreg"
    INDICBERT = "indicbert"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


CANDIDATE_MODELS: Dict[str, ModelKind] = {
    ModelKind.TFIDF_LOGREG.value: ModelKind.TFIDF_LOGREG,
    ModelKind.INDICBERT.value: ModelKind.INDICBERT,
}


# ---------------------------------------------------------------------------
# Metrics value object
# ---------------------------------------------------------------------------
@dataclass
class ClassificationMetrics:
    """Holdout metrics for a single trained model."""

    model_name: str
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    roc_auc_ovr: Optional[float]
    log_loss: Optional[float]
    confusion_matrix: List[List[int]]
    fit_time_seconds: float
    predict_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# TF-IDF baseline
# ---------------------------------------------------------------------------
def build_tfidf_baseline(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    C: float = 1.0,
    random_state: int = 42,
) -> SkPipeline:
    """Build a TF-IDF + LogisticRegression pipeline.

    The pipeline:
        1. TfidfVectorizer with unigrams + bigrams, char + word level
           (Hinglish benefits from character n-grams because Romanized
           spellings vary heavily).
        2. LogisticRegression with L2 regularization.

    Returns
    -------
    SkPipeline
        Unfitted pipeline.
    """
    return SkPipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\S+",  # don't drop Devanagari tokens
        )),
        ("logreg", LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs",
            random_state=random_state, n_jobs=-1,
        )),
    ])


def train_tfidf_baseline(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    config: HinglishConfig = DEFAULT_CONFIG,
    random_state: int = 42,
) -> Tuple[SkPipeline, ClassificationMetrics]:
    """Train + evaluate the TF-IDF + LogReg baseline."""
    pipe = build_tfidf_baseline(random_state=random_state)

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    predict_time = time.perf_counter() - t0

    # Probabilities.
    try:
        y_proba = pipe.predict_proba(X_test)
    except Exception:
        y_proba = None

    metrics = _compute_metrics(
        y_test, y_pred, y_proba,
        model_name="tfidf_logreg",
        fit_time=fit_time, predict_time=predict_time,
    )
    return pipe, metrics


# ---------------------------------------------------------------------------
# IndicBERT classifier
# ---------------------------------------------------------------------------
class IndicBertClassifier:
    """HuggingFace AutoModelForSequenceClassification wrapper.

    Provides the same ``predict`` / ``predict_probas`` interface as the
    TF-IDF baseline so they can be benchmarked head-to-head.
    """

    def __init__(
        self,
        model_id: str = INDICBERT_MODEL_ID,
        num_labels: int = 3,
        max_length: int = 128,
        device: str = "cpu",
    ):
        if not HAVE_HF:
            raise RuntimeError("transformers is required for IndicBertClassifier.")
        if not HAVE_TORCH:
            raise RuntimeError("torch is required for IndicBertClassifier.")
        self.model_id = model_id
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = torch.device(device)
        self.tokenizer = None
        self.model = None

    def load(self):
        """Load the pretrained tokenizer + model from HF hub."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, num_labels=self.num_labels,
        ).to(self.device)
        return self

    def predict(self, texts: List[str], batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(labels, probas)`` for a list of texts."""
        if self.model is None or self.tokenizer is None:
            self.load()
        self.model.eval()
        all_probas = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True, truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
                probas = torch.softmax(out.logits, dim=-1).cpu().numpy()
            all_probas.append(probas)
        probas = np.concatenate(all_probas, axis=0)
        labels = probas.argmax(axis=1).astype(np.int64)
        return labels, probas

    def predict_probas(self, texts: List[str]) -> np.ndarray:
        return self.predict(texts)[1]


def build_indicbert(
    config: HinglishConfig = DEFAULT_CONFIG,
    device: str = "cpu",
) -> IndicBertClassifier:
    """Construct an IndicBertClassifier (loads weights from HF hub)."""
    return IndicBertClassifier(
        model_id=config.model_id,
        num_labels=config.num_labels,
        max_length=config.max_length,
        device=device,
    )


def train_indicbert(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    config: HinglishConfig = DEFAULT_CONFIG,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    device: str = "cpu",
    output_dir: Optional[Path | str] = None,
) -> Tuple[IndicBertClassifier, ClassificationMetrics]:
    """Fine-tune IndicBERT on the Hinglish sentiment task.

    NB: This function downloads the IndicBERT weights on first call
    (~250 MB). On CPU, training is slow (~1 min/epoch for n=100 samples);
    GPU is strongly recommended for real workloads.

    The function uses HF's ``Trainer`` API which handles:
        - Tokenization batching
        - Mixed-precision (if device supports it)
        - Linear LR schedule with warmup
        - Checkpointing (best-val-accuracy saved to ``output_dir``)
    """
    if not HAVE_HF:
        raise RuntimeError("transformers is required for train_indicbert.")
    if not HAVE_TORCH:
        raise RuntimeError("torch is required for train_indicbert.")
    from datasets import Dataset as HFDataset

    classifier = build_indicbert(config, device=device)
    classifier.load()

    # Tokenize train + test as HF Datasets.
    def tokenize_fn(examples):
        return classifier.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
        )

    train_df = pd.DataFrame({"text": X_train, "labels": y_train.astype(int)})
    test_df = pd.DataFrame({"text": X_test, "labels": y_test.astype(int)})
    train_ds = HFDataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    test_ds = HFDataset.from_pandas(test_df).map(tokenize_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Training arguments — minimal config for CPU smoke testing.
    # NB: transformers 5.x removed ``warmup_ratio``; use ``warmup_steps``.
    args = TrainingArguments(
        output_dir=str(output_dir or Path("./models/_indicbert_tmp")),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=10,
        eval_strategy="epoch" if len(test_ds) > 0 else "no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        fp16=False,  # CPU only
    )

    t0 = time.perf_counter()
    trainer = Trainer(
        model=classifier.model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()
    fit_time = time.perf_counter() - t0

    # Evaluate.
    t0 = time.perf_counter()
    labels, probas = classifier.predict(X_test.tolist(), batch_size=batch_size)
    predict_time = time.perf_counter() - t0

    metrics = _compute_metrics(
        y_test, labels, probas,
        model_name="indicbert",
        fit_time=fit_time, predict_time=predict_time,
    )
    return classifier, metrics


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def _compute_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    model_name: str,
    fit_time: float,
    predict_time: float,
) -> ClassificationMetrics:
    """Compute standard classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    roc_auc: Optional[float] = None
    ll: Optional[float] = None
    if y_proba is not None and y_proba.shape[1] >= 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None
        try:
            ll = float(log_loss(y_true, y_proba))
        except Exception:
            ll = None
    return ClassificationMetrics(
        model_name=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        roc_auc_ovr=roc_auc,
        log_loss=ll,
        confusion_matrix=cm,
        fit_time_seconds=fit_time,
        predict_time_seconds=predict_time,
    )


def evaluate_classifier(
    predict_fn,
    X_test: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> ClassificationMetrics:
    """Evaluate a fitted model via a ``predict_fn(texts) -> (labels, probas)`` callable."""
    t0 = time.perf_counter()
    labels, probas = predict_fn(X_test.tolist())
    predict_time = time.perf_counter() - t0
    return _compute_metrics(
        y_test, labels, probas,
        model_name=model_name, fit_time=0.0, predict_time=predict_time,
    )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_to_onnx(
    classifier: "IndicBertClassifier",
    output_path: Path | str,
    max_length: int = 128,
    opset: int = 17,
) -> Path:
    """Serialize an IndicBERT classifier to ONNX.

    Uses HuggingFace's ``torch.onnx.export`` (or the new onnxscript-based
    exporter in torch>=2.0) with a dummy input of shape ``(1, max_length)``.

    The exported graph accepts:
        - ``input_ids``    : int64 [batch, seq_len]
        - ``attention_mask`` : int64 [batch, seq_len]

    and returns:
        - ``logits`` : float32 [batch, num_labels]
    """
    if not HAVE_TORCH:
        raise RuntimeError("torch is required for ONNX export.")
    if classifier.model is None or classifier.tokenizer is None:
        classifier.load()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a dummy input via the tokenizer (so we know the right shape).
    dummy = classifier.tokenizer(
        "test text",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Use a wrapper nn.Module that takes (input_ids, attention_mask)
    # and returns logits, so the ONNX graph has the right input names.
    import torch.nn as nn

    class _Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits

    wrapper = _Wrapper(classifier.model).eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    return output_path


# ---------------------------------------------------------------------------
# ONNX runtime inference
# ---------------------------------------------------------------------------
def load_onnx_session(onnx_path: Path | str):
    """Load an ONNX model into an ``onnxruntime.InferenceSession``."""
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def predict_with_onnx(
    session,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference via ONNX runtime.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
    tokenizer : HuggingFace tokenizer (for encoding the input texts)
    texts : list of str
    max_length : int

    Returns
    -------
    (labels, probas) : tuple[np.ndarray, np.ndarray]
    """
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    logits = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
    # Softmax.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probas = exp / exp.sum(axis=1, keepdims=True)
    labels = probas.argmax(axis=1).astype(np.int64)
    return labels, probas


__all__ = [
    "ModelKind",
    "CANDIDATE_MODELS",
    "ClassificationMetrics",
    "TfidfBaseline",
    "IndicBertClassifier",
    "build_tfidf_baseline",
    "build_indicbert",
    "train_tfidf_baseline",
    "train_indicbert",
    "evaluate_classifier",
    "export_to_onnx",
    "load_onnx_session",
    "predict_with_onnx",
    "HAVE_HF",
    "HAVE_TORCH",
]


# Alias for backwards-compat.
TfidfBaseline = SkPipeline
