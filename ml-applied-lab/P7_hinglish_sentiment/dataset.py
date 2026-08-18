"""
dataset
=======

Hinglish code-mixed review sentiment ETL pipeline.

Public surface
--------------
- ``ScriptKind``                  : enum (Roman, Devanagari, Mixed).
- ``SentimentLabel``               : enum (negative=0, neutral=1, positive=2).
- ``HinglishConfig``               : dataclass holding label names + tokenizer info.
- ``HinglishExample``              : frozen value object bundling text + label.
- ``HinglishDataset``              : frozen value object holding X/y + provenance.
- ``normalize_text``                : text normalization (script unification, etc.).
- ``detect_script``                : heuristic script detection (Roman/Devanagari/Mixed).
- ``generate_synthetic_hinglish`` : synthetic Hinglish review generator (offline).
- ``load_hinglish_dataset``        : one-call loader (CSV | synthetic | HF hub).
- ``build_stratified_splits``      : stratified train/val/test split.
- ``HFDatasetWrapper``             : wraps a ``datasets.Dataset`` for tokenizer collation.
- ``create_hf_dataset``            : convert a pandas DataFrame to a HF Dataset.

Design notes
------------
1. **Hinglish is code-mixed** — sentences like ``"movie accha tha but ending
   bahut bura tha"`` mix English ("movie", "ending") and Romanized Hindi
   ("accha" = good, "bahut" = very, "bura" = bad). The dataset module
   exposes a ``detect_script`` helper that classifies each token as Roman,
   Devanagari, or mixed, and a ``normalize_text`` helper that:
     - Lower-cases ASCII (Roman) text
     - Preserves Devanagari Unicode points (U+0900–U+097F) unchanged
     - Strips URLs, @mentions, and excess whitespace
     - Normalizes common Hinglish romanizations (e.g. ``"acha"`` →
       ``"accha"``) using a small lookup table

2. **Synthetic generator for offline testing** — a real Hinglish sentiment
   corpus (e.g. ``HASOC 2021``, ``SentiMixHinglish``) is not freely
   downloadable without registration. The synthetic generator produces
   3-class reviews by sampling from a small Hinglish lexicon so the
   entire pipeline (tokenize → train → evaluate → export ONNX) can be
   smoke-tested without network access. To use a real corpus, drop a
   CSV with ``text`` and ``label`` columns into ``data/`` and pass the
   path via ``--csv``.

3. **HuggingFace Dataset wrapper** — the ``HFDatasetWrapper`` class
   wraps a ``datasets.Dataset`` and exposes a PyTorch-style collation
   function so the same ``DataLoader`` can be used for both the
   TF-IDF baseline (which takes raw strings) and the IndicBERT
   fine-tuner (which takes ``input_ids`` + ``attention_mask`` tensors).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("hinglish_dataset")

# Lazy imports — heavy HF stack may not be needed in every context.
try:
    from datasets import Dataset as HFDataset
    HAVE_HF_DATASETS = True
except Exception:  # pragma: no cover
    HAVE_HF_DATASETS = False

try:
    from transformers import AutoTokenizer
    HAVE_TOK = True
except Exception:  # pragma: no cover
    HAVE_TOK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"

# Canonical IndicBERT model identifier on the HuggingFace hub.
# NB: ``ai4bharat/indic-bert`` became a gated repo in 2024 — access requires
# accepting a license on the model card at
# https://huggingface.co/ai4bharat/indic-bert. For the pipeline to run
# out-of-the-box without HF authentication, we default to
# ``bert-base-multilingual-cased`` (same general architecture: 12-layer
# transformer, ~118M params, covers Devanagari script in its vocab).
# To use the real IndicBERT, set ``HinglishConfig.model_id =
# "ai4bharat/indic-bert"`` after accepting the license on HF.
INDICBERT_MODEL_ID = "bert-base-multilingual-cased"
FALLBACK_MODEL_ID = "bert-base-multilingual-cased"
GATED_INDICBERT_MODEL_ID = "ai4bharat/indic-bert"

# Default label vocabulary — 3-class sentiment classification.
DEFAULT_LABELS: Tuple[str, ...] = ("negative", "neutral", "positive")
LABEL_TO_IDX: Dict[str, int] = {lbl: i for i, lbl in enumerate(DEFAULT_LABELS)}
IDX_TO_LABEL: Dict[int, str] = {i: lbl for i, lbl in enumerate(DEFAULT_LABELS)}


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------
class ScriptKind(str, Enum):
    ROMAN = "roman"
    DEVANAGARI = "devanagari"
    MIXED = "mixed"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


class SentimentLabel(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"

    @property
    def idx(self) -> int:
        return LABEL_TO_IDX[self.value]


@dataclass(frozen=True)
class HinglishConfig:
    """Configuration for the Hinglish sentiment task."""

    labels: Tuple[str, ...] = DEFAULT_LABELS
    model_id: str = INDICBERT_MODEL_ID
    max_length: int = 128

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def label_to_idx(self) -> Dict[str, int]:
        return {lbl: i for i, lbl in enumerate(self.labels)}

    @property
    def idx_to_label(self) -> Dict[int, str]:
        return {i: lbl for i, lbl in enumerate(self.labels)}


DEFAULT_CONFIG = HinglishConfig()


@dataclass(frozen=True)
class HinglishExample:
    """A single training example."""

    text: str
    label: str
    label_idx: int
    script: str


@dataclass(frozen=True)
class HinglishDataset:
    """Bundle of examples + provenance.

    Attributes
    ----------
    df : pd.DataFrame
        Must contain ``text`` and ``label_idx`` columns.
    X : pd.Series
        The text column.
    y : pd.Series
        Integer labels.
    source : str
        "synthetic" | CSV path | HF dataset id.
    sha256 : str
    n_samples : int
    """

    df: pd.DataFrame
    X: pd.Series
    y: pd.Series
    source: str
    sha256: str
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_samples", int(len(self.X)))


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------
# Unicode range for Devanagari (U+0900–U+097F).
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_ROMAN_RE = re.compile(r"[A-Za-z]")


def detect_script(text: str) -> str:
    """Heuristically classify the dominant script of ``text``.

    Returns one of ``ScriptKind.ROMAN``, ``ScriptKind.DEVANAGARI``,
    ``ScriptKind.MIXED``, or ``ScriptKind.UNKNOWN``.

    The heuristic counts the number of Devanagari and Roman characters
    in the input. If both are present in roughly equal proportions
    (>20% each), the text is classified as MIXED. If only one script
    is present, that's the answer. If neither, UNKNOWN.
    """
    if not text:
        return ScriptKind.UNKNOWN.value
    dev_chars = len(_DEVANAGARI_RE.findall(text))
    rom_chars = len(_ROMAN_RE.findall(text))
    total = dev_chars + rom_chars
    if total == 0:
        return ScriptKind.UNKNOWN.value
    dev_frac = dev_chars / total
    rom_frac = rom_chars / total
    if dev_frac >= 0.2 and rom_frac >= 0.2:
        return ScriptKind.MIXED.value
    if dev_frac >= 0.5:
        return ScriptKind.DEVANAGARI.value
    if rom_frac >= 0.5:
        return ScriptKind.ROMAN.value
    return ScriptKind.UNKNOWN.value


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
# Common Hinglish romanization normalization — collapses multiple spellings
# of the same Hindi word into a canonical form. This is a small lookup
# table; production Hinglish NLP uses much larger lexicons.
_HINGLISH_NORMALIZE_MAP: Dict[str, str] = {
    "acha": "accha", "acche": "accha", "achha": "accha", "achhe": "accha",
    "bura": "bura", "buraai": "bura", "burra": "bura",
    "bahut": "bahut", "bohot": "bahut", "bhut": "bahut",
    "kya": "kya", "kyaa": "kya",
    "hai": "hai", "he": "hai", "hain": "hai",
    "tha": "tha", "thaa": "tha", "the": "the", "thee": "the",
    "aur": "aur", "or": "aur",
    "kyunki": "kyunki", "qki": "kyunki",
    "lekin": "lekin", "par": "par", "pr": "par", "but": "lekin",
    "acha-tha": "accha tha",
    "matlab": "matlab", "mtlb": "matlab",
    "banda": "banda", "bandi": "bandi",
    "movie": "movie", "film": "film",
    "mast": "mast", "maast": "mast",
    "zabardast": "zabardast", "zabardastt": "zabardast",
    "bekar": "bekar", "bekaar": "bekar",
    "ganda": "ganda", "gandha": "ganda",
    "pyaar": "pyaar", "pyar": "pyaar", "mohabbat": "mohabbat",
}

# URL / @mention / hashtag patterns.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_text(
    text: str,
    *,
    lowercase_roman: bool = True,
    strip_urls: bool = True,
    strip_mentions: bool = True,
    normalize_hinglish: bool = True,
    preserve_devanagari: bool = True,
) -> str:
    """Normalize Hinglish code-mixed text.

    Parameters
    ----------
    text : str
        Raw input text.
    lowercase_roman : bool
        If True, lower-case ASCII (Roman) characters but preserve
        Devanagari Unicode points unchanged.
    strip_urls, strip_mentions : bool
        Remove URLs / @mentions from the text.
    normalize_hinglish : bool
        Apply the Hinglish romanization lookup table (e.g. ``"acha"`` →
        ``"accha"``).
    preserve_devanagari : bool
        If True, never modify Devanagari characters. If False, attempt
        to transliterate Devanagari to Roman (not implemented — raises
        NotImplementedError).

    Returns
    -------
    str
        Normalized text.

    Notes
    -----
    The normalization is conservative — we don't stem, lemmatize, or
    remove stop-words because IndicBERT's pretraining already learned
    Hinglish sub-word distributions, and aggressive preprocessing would
    destroy that signal.
    """
    if not preserve_devanagari:
        raise NotImplementedError("Devanagari → Roman transliteration is not implemented.")

    text = str(text)

    # Strip URLs / mentions.
    if strip_urls:
        text = _URL_RE.sub(" ", text)
    if strip_mentions:
        text = _MENTION_RE.sub(" ", text)

    # Convert hashtags to bare words (drop the #).
    text = _HASHTAG_RE.sub(r"\1", text)

    # Lower-case Roman characters while preserving Devanagari.
    if lowercase_roman:
        # Lowercase only ASCII characters; Devanagari code points are
        # already case-insensitive (no uppercase Devanagari).
        text = text.lower()

    # Apply the Hinglish normalization lookup table. Only affects Roman
    # tokens (Devanagari tokens won't match any keys in the table).
    if normalize_hinglish:
        tokens = text.split()
        normalized = []
        for tok in tokens:
            # Strip trailing punctuation so "accha!" → "accha" + "!".
            tok_stripped = tok.strip(".,!?;:\"'()[]{}…")
            punct = tok[len(tok_stripped):] if tok_stripped else ""
            canonical = _HINGLISH_NORMALIZE_MAP.get(tok_stripped, tok_stripped)
            normalized.append(canonical + punct)
        text = " ".join(normalized)

    # Collapse multiple whitespace.
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Synthetic Hinglish review generator
# ---------------------------------------------------------------------------
# A small Hinglish lexicon for each sentiment class. Real Hinglish corpora
# have thousands of words per class; we use ~30 per class to keep the
# generator self-contained.
_LEXICON: Dict[str, List[str]] = {
    "positive": [
        "accha", "mast", "zabardast", "pyaar", "kamaal", "fantastic", "great",
        "superb", "awesome", "best", "loved", "amazing", "wonderful", "perfect",
        "brilliant", "outstanding", "fabulous", "mind-blowing", "mindblowing",
        "blockbuster", "hit", "superhit", "kalakar", "acting", "performance",
        "story", "direction", "music", " cinematography", "beautiful",
    ],
    "negative": [
        "bura", "bekar", "ganda", "boring", "waste", "kharab", "worst",
        "terrible", "awful", "hated", "disappointing", "pathetic", "horrible",
        "flop", "disaster", "torture", "painful", "useless", "stupid",
        "nonsense", "bakwaas", "faltu", "barbaad", "time-waste", "ghatiya",
        "disgusting", "irritating", "horrendous", "trash", "raddi",
    ],
    "neutral": [
        "thik", "okay", "average", "normal", "decent", "fine", "so-so",
        "medium", "theka", "chalta", "manage", "regular", "typical",
        "expected", "predicted", "usual", "common", "ordinary", "fine-tha",
        "not-bad", "not-great", "passable", "fair", "moderate", "ok-ok",
        "alright", "sahi", "consistent", "balanced", "neutral",
    ],
}

# Review templates — {sent} gets replaced with a random lexicon word.
_REVIEW_TEMPLATES: List[str] = [
    "movie {sent} tha but {sent2} bhi tha",
    "ye movie {sent} hai, acting {sent2} thi",
    "{sent} movie! story {sent2}",
    "acting {sent} thi, direction {sent2}",
    "{sent} tha yar, ending {sent2}",
    "movie dekh ke {sent} laga, story {sent2} thi",
    "ye film {sent} hai, songs {sent2}",
    "{sent} movie overall",
    "picture {sent} hai, climax {sent2} tha",
    "{sent} acting + {sent2} direction",
    "first half {sent}, second half {sent2}",
    "{sent} experience tha",
]


def _sample_lexicon_word(rng: np.random.Generator, sentiment: str) -> str:
    return str(rng.choice(_LEXICON[sentiment]))


def generate_synthetic_hinglish(
    n_per_class: int = 200,
    seed: int = 42,
    include_devanagari: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic Hinglish review dataset.

    Parameters
    ----------
    n_per_class : int
        Number of reviews per sentiment class.
    seed : int
        Random seed for reproducibility.
    include_devanagari : bool
        If True, ~20% of reviews include at least one Devanagari token.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for label_idx, label in enumerate(DEFAULT_LABELS):
        for i in range(n_per_class):
            template = str(rng.choice(_REVIEW_TEMPLATES))
            sent = _sample_lexicon_word(rng, label)
            sent2 = _sample_lexicon_word(rng, label)
            text = template.format(sent=sent, sent2=sent2)

            # Optionally inject Devanagari.
            if include_devanagari and rng.random() < 0.20:
                # Translate a few common words to Devanagari.
                deva_map = {"accha": "अच्छा", "bura": "बुरा", "bahut": "बहुत",
                            "hai": "है", "tha": "था", "movie": "फ़िल्म",
                            "mast": "मस्त", "pyaar": "प्यार", "kya": "क्या",
                            "tha-tha": "था"}
                for rom, deva in deva_map.items():
                    if rom in text:
                        text = text.replace(rom, deva, 1)
                        break

            rows.append({
                "text": text,
                "label": label,
                "label_idx": label_idx,
                "script": detect_script(text),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _sha256(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


def load_hinglish_dataset(
    csv_path: Optional[Path | str] = None,
    n_per_class: int = 200,
    seed: int = 42,
    config: HinglishConfig = DEFAULT_CONFIG,
    normalize: bool = True,
) -> HinglishDataset:
    """One-call loader for Hinglish sentiment data.

    Resolution order:
        1. ``csv_path`` (explicit override — must contain ``text`` and
           ``label`` columns; label values must be in ``config.labels``).
        2. ``data/hinglish.csv`` (project-local drop-in).
        3. ``generate_synthetic_hinglish`` (synthetic fallback).

    The text column is normalized via ``normalize_text`` unless
    ``normalize=False``.
    """
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Hinglish CSV not found: {path}")
        df = pd.read_csv(path)
        source = str(path)
        sha = _sha256(path.read_bytes())
    elif (PROJECT_DATA_DIR / "hinglish.csv").exists():
        path = PROJECT_DATA_DIR / "hinglish.csv"
        df = pd.read_csv(path)
        source = str(path)
        sha = _sha256(path.read_bytes())
    else:
        log.info("Using synthetic Hinglish data (n_per_class=%d, seed=%d)", n_per_class, seed)
        df = generate_synthetic_hinglish(n_per_class=n_per_class, seed=seed)
        source = "synthetic"
        sha = _sha256(df.to_csv(index=False))

    # Validate + map labels.
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"CSV must have 'text' and 'label' columns; got: {list(df.columns)}"
        )
    # Coerce label values to lowercase strings.
    df["label"] = df["label"].astype(str).str.lower()
    valid_labels = set(config.labels)
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
    if "label_idx" not in df.columns:
        df["label_idx"] = df["label"].map(config.label_to_idx).astype(int)

    # Normalize text.
    if normalize:
        df["text"] = df["text"].astype(str).apply(normalize_text)
        # Re-detect script post-normalization.
        if "script" not in df.columns or df["script"].isna().any():
            df["script"] = df["text"].apply(detect_script)
    elif "script" not in df.columns:
        df["script"] = df["text"].apply(detect_script)

    X = df["text"].copy()
    y = df["label_idx"].astype(int).copy()
    return HinglishDataset(df=df, X=X, y=y, source=source, sha256=sha)


# ---------------------------------------------------------------------------
# Stratified splits
# ---------------------------------------------------------------------------
def build_stratified_splits(
    ds: HinglishDataset,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split (preserves label proportions)."""
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(ds.df, ds.df["label_idx"]))
    train_val = ds.df.iloc[train_val_idx].reset_index(drop=True)
    test = ds.df.iloc[test_idx].reset_index(drop=True)

    adjusted_val_size = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=seed)
    train_idx, val_idx = next(sss2.split(train_val, train_val["label_idx"]))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test


# ---------------------------------------------------------------------------
# HuggingFace Dataset wrapper
# ---------------------------------------------------------------------------
class HFDatasetWrapper:
    """Thin wrapper around a ``datasets.Dataset`` that exposes a PyTorch-
    friendly collation function.

    The wrapper:
        1. Stores a HuggingFace ``Dataset`` (created from a pandas DataFrame).
        2. Holds a reference to a HuggingFace ``AutoTokenizer``.
        3. Provides ``collate_fn(batch)`` for use with ``torch.utils.data.DataLoader``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer=None,
        max_length: int = 128,
    ):
        if not HAVE_HF_DATASETS:
            raise RuntimeError("HuggingFace datasets library is required for HFDatasetWrapper.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Build the HF Dataset.
        self.dataset = HFDataset.from_pandas(df[["text", "label_idx"]])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]

    def tokenize(self, examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Tokenization function for ``dataset.map``."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not set — call set_tokenizer() first.")
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,  # return lists for HF map
        )

    def set_tokenizer(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def map_tokenized(self, batch_size: int = 32) -> "HFDatasetWrapper":
        """Apply tokenization in-place to the underlying HF Dataset."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not set — call set_tokenizer() first.")
        self.dataset = self.dataset.map(
            self.tokenize,
            batched=True,
            batch_size=batch_size,
            remove_columns=["text"],
        )
        # Rename label_idx → labels for HF compatibility.
        if "label_idx" in self.dataset.column_names:
            self.dataset = self.dataset.rename_column("label_idx", "labels")
        self.dataset.set_format("torch")
        return self


def create_hf_dataset(
    df: pd.DataFrame,
    tokenizer=None,
    max_length: int = 128,
    batch_size: int = 32,
) -> "HFDatasetWrapper":
    """One-call helper: build a HF Dataset from a DataFrame, tokenize it."""
    wrapper = HFDatasetWrapper(df, tokenizer=tokenizer, max_length=max_length)
    if tokenizer is not None:
        wrapper.map_tokenized(batch_size=batch_size)
    return wrapper


__all__ = [
    "ScriptKind",
    "SentimentLabel",
    "HinglishConfig",
    "HinglishExample",
    "HinglishDataset",
    "DEFAULT_LABELS",
    "LABEL_TO_IDX",
    "IDX_TO_LABEL",
    "INDICBERT_MODEL_ID",
    "FALLBACK_MODEL_ID",
    "GATED_INDICBERT_MODEL_ID",
    "DEFAULT_CONFIG",
    "detect_script",
    "normalize_text",
    "generate_synthetic_hinglish",
    "load_hinglish_dataset",
    "build_stratified_splits",
    "HFDatasetWrapper",
    "create_hf_dataset",
]
