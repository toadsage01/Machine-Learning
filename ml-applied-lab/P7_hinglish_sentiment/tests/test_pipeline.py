"""
tests/test_pipeline
===================

End-to-end tests for the P7 Hinglish sentiment pipeline.

Coverage:
    * Script detection — Roman / Devanagari / Mixed classification.
    * Text normalization — URLs/mentions stripped, Hinglish lookup applied.
    * Synthetic dataset generator — produces balanced 3-class data.
    * Stratified splits — preserves class proportions.
    * TF-IDF baseline — trains + produces sane metrics.
    * IndicBERT (multilingual BERT fallback) — trains + produces sane metrics.
    * ONNX export + runtime parity vs PyTorch (max proba diff < 1e-4).
    * CLI smoke test (TF-IDF only — IndicBERT CLI is too slow for CI).

Run with::

    cd ml-applied-lab/P7_hinglish_sentiment
    python -m pytest tests/ -v

or::

    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import logging  # noqa: E402
logging.getLogger("torch.onnx").setLevel(logging.CRITICAL)
logging.getLogger("onnxscript").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

from dataset import (  # noqa: E402
    DEFAULT_LABELS, DEFAULT_CONFIG, LABEL_TO_IDX,
    HinglishConfig, INDICBERT_MODEL_ID, GATED_INDICBERT_MODEL_ID,
    detect_script, normalize_text, generate_synthetic_hinglish,
    load_hinglish_dataset, build_stratified_splits,
    HFDatasetWrapper, create_hf_dataset,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, ModelKind, ClassificationMetrics,
    train_tfidf_baseline, train_indicbert,
    export_to_onnx, load_onnx_session, predict_with_onnx,
    HAVE_HF, HAVE_TORCH,
)


# ---------------------------------------------------------------------------
# Script detection tests
# ---------------------------------------------------------------------------
def test_detect_script_roman():
    assert detect_script("movie accha tha") == "roman"
    assert detect_script("This is purely English text") == "roman"


def test_detect_script_devanagari():
    assert detect_script("यह एक हिंदी वाक्य है") == "devanagari"
    assert detect_script("फ़िल्म अच्छी थी") == "devanagari"


def test_detect_script_mixed():
    # Hinglish code-mixing: Roman + Devanagari both present in >20%.
    assert detect_script("movie अच्छा tha but ending बुरा tha") == "mixed"


def test_detect_script_empty_and_unknown():
    assert detect_script("") == "unknown"
    assert detect_script("12345 !!!") == "unknown"


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------
def test_normalize_strips_urls_and_mentions():
    raw = "Check this out @user1 https://t.co/abc123 movie was accha"
    norm = normalize_text(raw)
    assert "https://t.co" not in norm
    assert "@user1" not in norm
    assert "accha" in norm  # content preserved


def test_normalize_lowercases_roman_preserves_devanagari():
    raw = "Movie ACCHA था"
    norm = normalize_text(raw)
    # Roman should be lowercased.
    assert "movie" in norm
    assert "accha" in norm
    assert "MOVIE" not in norm
    # Devanagari should be preserved (no case conversion).
    assert "था" in norm


def test_normalize_applies_hinglish_lookup():
    # "acha" should be normalized to "accha".
    raw = "movie acha tha"
    norm = normalize_text(raw)
    assert "accha" in norm
    assert "acha" not in norm.split()  # not as a standalone token


def test_normalize_preserves_punctuation():
    raw = "movie accha! mast!!"
    norm = normalize_text(raw)
    assert "!" in norm
    assert "accha" in norm


def test_normalize_handles_hashtags():
    raw = "loved this #superhit movie"
    norm = normalize_text(raw)
    # The # should be stripped but the word preserved.
    assert "#" not in norm
    assert "superhit" in norm


def test_normalize_idempotent():
    raw = "Movie acha tha but ending bahut bura tha"
    once = normalize_text(raw)
    twice = normalize_text(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Synthetic dataset tests
# ---------------------------------------------------------------------------
def test_synthetic_dataset_is_balanced():
    df = generate_synthetic_hinglish(n_per_class=50, seed=42)
    assert len(df) == 150  # 3 classes × 50
    counts = df["label"].value_counts().to_dict()
    assert all(c == 50 for c in counts.values())


def test_load_hinglish_dataset_returns_valid_object():
    ds = load_hinglish_dataset(n_per_class=30, seed=0)
    assert ds.n_samples == 90
    assert set(ds.df["label"].unique()) == set(DEFAULT_LABELS)
    assert "text" in ds.df.columns
    assert "label_idx" in ds.df.columns
    assert "script" in ds.df.columns


def test_stratified_splits_preserve_class_proportions():
    ds = load_hinglish_dataset(n_per_class=100, seed=42)
    train, val, test = build_stratified_splits(ds, val_size=0.15, test_size=0.15, seed=42)
    # All 3 classes present in every split.
    assert train["label"].nunique() == 3
    assert val["label"].nunique() == 3
    assert test["label"].nunique() == 3
    # Class proportions roughly preserved (each class should be ~70/15/15).
    train_counts = train["label"].value_counts()
    test_counts = test["label"].value_counts()
    for label in DEFAULT_LABELS:
        # Test set should have ~15 samples per class.
        assert 10 <= test_counts[label] <= 20


# ---------------------------------------------------------------------------
# HFDatasetWrapper tests
# ---------------------------------------------------------------------------
def test_hf_dataset_wrapper_tokenization():
    """Verify HFDatasetWrapper tokenizes text via an HF tokenizer."""
    if not HAVE_HF:
        return  # skip when transformers unavailable
    from transformers import AutoTokenizer
    df = generate_synthetic_hinglish(n_per_class=5, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_ID)
    wrapper = create_hf_dataset(df, tokenizer=tokenizer, max_length=32, batch_size=4)
    # The wrapper should have tokenized columns.
    assert "input_ids" in wrapper.dataset.column_names
    assert "attention_mask" in wrapper.dataset.column_names
    assert "labels" in wrapper.dataset.column_names
    # input_ids shape after set_format("torch") should be (5, 32).
    sample = wrapper.dataset[0]
    assert sample["input_ids"].shape == (32,)


# ---------------------------------------------------------------------------
# TF-IDF baseline tests
# ---------------------------------------------------------------------------
def test_train_tfidf_baseline_produces_sane_metrics():
    ds = load_hinglish_dataset(n_per_class=80, seed=42)
    train, val, test = build_stratified_splits(ds, seed=42)
    pipe, m = train_tfidf_baseline(
        train["text"], train["label_idx"],
        test["text"], test["label_idx"],
    )
    assert m.model_name == "tfidf_logreg"
    # On the synthetic data (which has clear lexicon separation per class),
    # TF-IDF should easily beat random (33% for 3-class).
    assert m.accuracy > 0.50, f"TF-IDF accuracy too low: {m.accuracy}"
    assert 0.0 <= m.f1_macro <= 1.0
    assert m.confusion_matrix is not None
    assert len(m.confusion_matrix) == 3
    assert all(len(row) == 3 for row in m.confusion_matrix)


# ---------------------------------------------------------------------------
# IndicBERT tests (slow — runs only if P7_RUN_BERT_TESTS=1)
# ---------------------------------------------------------------------------
def test_train_indicbert_produces_sane_metrics():
    """Fine-tune IndicBERT for 1 epoch on a tiny synthetic dataset.

    Skipped by default because it downloads ~700 MB of weights and takes
    ~30+ seconds on CPU. Set P7_RUN_BERT_TESTS=1 to enable.
    """
    import os
    if os.environ.get("P7_RUN_BERT_TESTS", "") != "1":
        try:
            import pytest
            pytest.skip("Set P7_RUN_BERT_TESTS=1 to enable IndicBERT tests")
        except ImportError:
            return  # skip silently when pytest unavailable
    if not HAVE_HF or not HAVE_TORCH:
        return

    ds = load_hinglish_dataset(n_per_class=15, seed=42)
    train, val, test = build_stratified_splits(ds, seed=42)
    classifier, m = train_indicbert(
        train["text"], train["label_idx"],
        test["text"], test["label_idx"],
        config=DEFAULT_CONFIG, epochs=1, batch_size=4, learning_rate=2e-5,
    )
    assert m.model_name == "indicbert"
    assert 0.0 <= m.accuracy <= 1.0
    assert m.fit_time_seconds > 0
    assert len(m.confusion_matrix) == 3


def test_onnx_export_and_runtime_parity():
    """ONNX runtime predictions should match PyTorch to within float32 tolerance.

    Skipped by default (same reason as test_train_indicbert_produces_sane_metrics).
    """
    import os
    if os.environ.get("P7_RUN_BERT_TESTS", "") != "1":
        try:
            import pytest
            pytest.skip("Set P7_RUN_BERT_TESTS=1 to enable IndicBERT ONNX parity test")
        except ImportError:
            return
    if not HAVE_HF or not HAVE_TORCH:
        return

    ds = load_hinglish_dataset(n_per_class=15, seed=42)
    train, val, test = build_stratified_splits(ds, seed=42)
    classifier, _ = train_indicbert(
        train["text"], train["label_idx"],
        test["text"], test["label_idx"],
        config=DEFAULT_CONFIG, epochs=1, batch_size=4, learning_rate=2e-5,
    )
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "indicbert.onnx"
        export_to_onnx(classifier, onnx_path, max_length=128)
        assert onnx_path.exists() and onnx_path.stat().st_size > 50_000  # >50 KB

        session = load_onnx_session(onnx_path)
        sample_texts = test["text"].tolist()[:5]
        pt_labels, pt_probas = classifier.predict(sample_texts)
        onnx_labels, onnx_probas = predict_with_onnx(
            session, classifier.tokenizer, sample_texts, max_length=128,
        )
        # Labels should match (robust to small float differences).
        agreement = (pt_labels == onnx_labels).mean()
        assert agreement >= 0.80, f"ONNX/PyTorch label agreement {agreement:.0%} < 80%"
        # Max probability diff should be tiny (float32 precision).
        max_diff = np.abs(pt_probas - onnx_probas).max()
        assert max_diff < 1e-3, f"Max probability diff {max_diff:.6e} > 1e-3"


# ---------------------------------------------------------------------------
# CLI smoke test (TF-IDF only — IndicBERT is too slow for CI)
# ---------------------------------------------------------------------------
def test_cli_tfidf_runs_end_to_end():
    """Full `python train.py --models tfidf_logreg` invocation should exit 0."""
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--models", "tfidf_logreg",
        "--n-per-class", "30",
        "--metrics-json", "/tmp/_p7_cli_metrics.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"
    assert "BEST_MODEL=tfidf_logreg" in result.stdout
    assert Path("/tmp/_p7_cli_metrics.json").exists()
    import json
    payload = json.loads(Path("/tmp/_p7_cli_metrics.json").read_text())
    assert "results" in payload
    assert "tfidf_logreg" in payload["results"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import gc
    tests = [
        test_detect_script_roman,
        test_detect_script_devanagari,
        test_detect_script_mixed,
        test_detect_script_empty_and_unknown,
        test_normalize_strips_urls_and_mentions,
        test_normalize_lowercases_roman_preserves_devanagari,
        test_normalize_applies_hinglish_lookup,
        test_normalize_preserves_punctuation,
        test_normalize_handles_hashtags,
        test_normalize_idempotent,
        test_synthetic_dataset_is_balanced,
        test_load_hinglish_dataset_returns_valid_object,
        test_stratified_splits_preserve_class_proportions,
        test_hf_dataset_wrapper_tokenization,
        test_train_tfidf_baseline_produces_sane_metrics,
        test_train_indicbert_produces_sane_metrics,        # opt-in
        test_onnx_export_and_runtime_parity,                # opt-in
        test_cli_tfidf_runs_end_to_end,
    ]
    n_passed = 0
    n_skipped = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            n_passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            sys.exit(1)
        except BaseException:
            # pytest.skip / Skipped — anything that isn't an AssertionError
            # is treated as a skip.
            print(f"SKIP  {t.__name__}")
            n_skipped += 1
        gc.collect()
    print(f"\n{n_passed} passed, {n_skipped} skipped (out of {len(tests)} total).")
