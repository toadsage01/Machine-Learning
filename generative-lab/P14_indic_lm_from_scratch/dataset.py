"""
dataset
=======

Hindi and Hinglish text dataset builder with synthetic text fallback,
custom BPE/SentencePiece tokenizer trained on Indic text, sequence
chunking, and PyTorch DataLoader.

Public surface
--------------
- ``LMConfig``                    : dataclass with model + tokenizer config.
- ``TextDataset``                  : PyTorch Dataset returning token ID sequences.
- ``generate_synthetic_hindi_text`` : synthetic Hindi/Hinglish text generator.
- ``BPETokenizer``                 : SentencePiece BPE tokenizer wrapper.
- ``build_dataloaders``            : one-call helper returning train/val DataLoaders.
- ``load_text_corpus``             : one-call loader (text file | synthetic).

Design notes
------------
1. **Synthetic Hindi/Hinglish text** — real Hindi text corpora (e.g.
   HindiText, HI-CF) are gated or large. The synthetic generator produces
   realistic-shape sentences using a small Hindi lexicon + templates
   with Devanagari + Roman script mixing.

2. **SentencePiece BPE tokenizer** — we train a SentencePiece BPE model
   on the corpus. BPE is preferred over word-level tokenization for
   Hindi because it handles morphological variation (suffixes, compounds)
   and handles code-mixed Hinglish (Roman + Devanagari) gracefully.

3. **Sequence chunking** — the corpus is tokenized into a flat ID stream,
   then chunked into fixed-length sequences for training. The last chunk
   is padded with the EOS token.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("lm_dataset")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

try:
    import sentencepiece as spm
    HAVE_SPM = True
except Exception:  # pragma: no cover
    HAVE_SPM = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LMConfig:
    """Configuration for the Indic LM."""
    # Model.
    vocab_size: int = 1000
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 128
    ff_dim: int = 512
    # Tokenizer.
    tokenizer_model: Optional[str] = None  # path to .model file; None = train new.
    tokenizer_vocab_size: int = 500  # BPE vocab size for SentencePiece.
    # Data.
    seq_len: int = 64
    batch_size: int = 16
    val_size: float = 0.1
    seed: int = 42


DEFAULT_CONFIG = LMConfig()


# ---------------------------------------------------------------------------
# Synthetic Hindi/Hinglish text generator
# ---------------------------------------------------------------------------
# Small Hindi lexicon (Roman + Devanagari).
_HINDI_WORDS_ROMAN = [
    "main", "tum", "woh", "ye", "hum", "kya", "kyun", "kaisa", "accha",
    "bura", "bahut", "thoda", "zyada", "kam", "zyada", "acha", "theek",
    "hai", "tha", "hoga", "kiya", "karta", "gaya", "aaya", "gaya",
    "movie", "film", "gaana", "khana", "peena", "sona", "jaagna",
    "pyaar", "dosti", "ghar", "bahar", "school", "college", "office",
    "subah", "shaam", "raat", "din", "kal", "aaj", "parso",
    "nahi", "haan", "thik", "acha", "bilkul", "sahi", "galat",
]

_HINDI_WORDS_DEVANAGARI = [
    "मैं", "तुम", "वह", "यह", "हम", "क्या", "क्यों", "कैसा", "अच्छा",
    "बुरा", "बहुत", "थोड़ा", "ज़्यादा", "कम", "अच्छा", "ठीक",
    "है", "था", "होगा", "किया", "करता", "गया", "आया",
    "फ़िल्म", "गाना", "खाना", "पीना", "सोना", "जागना",
    "प्यार", "दोस्ती", "घर", "बाहर", "स्कूल", "कॉलेज",
    "सुबह", "शाम", "रात", "दिन", "कल", "आज",
    "नहीं", "हाँ", "ठीक", "बिल्कुल", "सही", "गलत",
]

_SENTENCE_TEMPLATES = [
    "{s1} {v} {s2} {adj} {n}.",
    "{n} {adj} {v}.",
    "{s1} {s2} {v}.",
    "{q} {s1} {v} {n}?",
    "{s1} {v} {n}, {s2} {v} {n}.",
    "{adj} {n} {v} {s1}.",
    "{s1} {n} {v}, {s2} {q} {v}?",
    "{q} {n} {adj} {v}?",
]


def generate_synthetic_hindi_text(
    n_sentences: int = 500,
    devanagari_ratio: float = 0.3,
    seed: int = 42,
) -> List[str]:
    """Generate synthetic Hindi/Hinglish sentences.

    Parameters
    ----------
    n_sentences : int
        Number of sentences to generate.
    devanagari_ratio : float
        Fraction of sentences that use Devanagari script (rest use Roman).
    seed : int
        Reproducibility seed.

    Returns
    -------
    List[str]
        List of sentences.
    """
    rng = np.random.default_rng(seed)
    sentences: List[str] = []

    for _ in range(n_sentences):
        use_devanagari = rng.random() < devanagari_ratio
        words = _HINDI_WORDS_DEVANAGARI if use_devanagari else _HINDI_WORDS_ROMAN

        # Pick a template.
        template = str(rng.choice(_SENTENCE_TEMPLATES))

        # Fill template slots.
        s1 = str(rng.choice(words[:8]))    # subjects (main, tum, woh, ye, hum, ...)
        s2 = str(rng.choice(words[:8]))
        v = str(rng.choice(words[15:25]))  # verbs (hai, tha, kiya, gaya, ...)
        adj = str(rng.choice(words[8:14])) # adjectives (accha, bura, bahut, ...)
        n = str(rng.choice(words[25:40]))  # nouns (movie, film, ghar, ...)
        q = str(rng.choice(words[5:8]))    # question words (kya, kyun, kaisa)

        sentence = template.format(s1=s1, s2=s2, v=v, adj=adj, n=n, q=q)
        sentences.append(sentence)

    return sentences


# ---------------------------------------------------------------------------
# BPE tokenizer (SentencePiece wrapper)
# ---------------------------------------------------------------------------
class BPETokenizer:
    """SentencePiece BPE tokenizer wrapper.

    Trains a BPE model on a corpus of text and provides:
        * ``encode(text) -> List[int]``: tokenize text to IDs.
        * ``decode(ids) -> str``: detokenize IDs to text.
        * ``vocab_size``: the tokenizer's vocabulary size.
    """

    def __init__(self, vocab_size: int = 500, model_type: str = "bpe"):
        if not HAVE_SPM:
            raise RuntimeError("sentencepiece is required for BPETokenizer.")
        self.vocab_size = vocab_size
        self.model_type = model_type
        self._sp: Optional[spm.SentencePieceProcessor] = None
        self._model_path: Optional[str] = None

    def train(self, sentences: List[str], cache_dir: Optional[Path] = None) -> "BPETokenizer":
        """Train the BPE tokenizer on a list of sentences.

        Writes the model to a temporary file (or ``cache_dir``).
        """
        # Write corpus to a temp file for SentencePiece training.
        if cache_dir is None:
            cache_dir = Path(tempfile.mkdtemp())
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        corpus_path = cache_dir / "corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

        model_prefix = str(cache_dir / "tokenizer")
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=0.98,  # high coverage for Devanagari.
            normalization_rule_name="identity",  # preserve Devanagari.
        )

        self._model_path = model_prefix + ".model"
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(self._model_path)
        return self

    def load(self, model_path: Path | str) -> "BPETokenizer":
        """Load a pre-trained tokenizer model."""
        self._model_path = str(model_path)
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(self._model_path)
        self.vocab_size = self._sp.get_piece_size()
        return self

    def encode(self, text: str) -> List[int]:
        """Tokenize text to a list of token IDs."""
        if self._sp is None:
            raise RuntimeError("Tokenizer not trained or loaded.")
        return self._sp.encode(text)

    def decode(self, ids: List[int]) -> str:
        """Detokenize a list of IDs to text."""
        if self._sp is None:
            raise RuntimeError("Tokenizer not trained or loaded.")
        return self._sp.decode(ids)

    @property
    def vocab_size_actual(self) -> int:
        """Actual vocab size (may differ from requested if corpus is small)."""
        if self._sp is None:
            return self.vocab_size
        return self._sp.get_piece_size()

    @property
    def eos_id(self) -> int:
        """EOS token ID (or the last token if no explicit EOS)."""
        if self._sp is None:
            return self.vocab_size - 1
        eos = self._sp.eos_id()
        return eos if eos >= 0 else self._sp.get_piece_size() - 1

    @property
    def pad_id(self) -> int:
        """PAD token ID (or 0 if not set)."""
        if self._sp is None:
            return 0
        pad = self._sp.pad_id()
        return pad if pad >= 0 else 0


# ---------------------------------------------------------------------------
# Text dataset
# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    """PyTorch Dataset returning fixed-length token ID sequences.

    The entire corpus is tokenized into a flat ID stream, then chunked
    into overlapping sequences of length ``seq_len + 1`` (the extra token
    is the target for next-token prediction).
    """

    def __init__(self, token_ids: List[int], seq_len: int = 64, pad_id: int = 0):
        self.token_ids = np.array(token_ids, dtype=np.int64)
        self.seq_len = seq_len
        self.pad_id = pad_id
        # Number of non-overlapping chunks.
        self.n_chunks = max(1, len(self.token_ids) // seq_len)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target shift.
        chunk = self.token_ids[start:end]
        # Pad if needed (last chunk).
        if len(chunk) < self.seq_len + 1:
            chunk = np.pad(chunk, (0, self.seq_len + 1 - len(chunk)),
                           mode="constant", constant_values=self.pad_id)
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # (seq_len,)
        y = torch.tensor(chunk[1:], dtype=torch.long)    # (seq_len,)
        return x, y


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders(
    sentences: List[str],
    tokenizer: BPETokenizer,
    config: LMConfig = DEFAULT_CONFIG,
) -> Tuple[DataLoader, DataLoader]:
    """Build train + val DataLoaders from a list of sentences.

    Returns
    -------
    (train_loader, val_loader)
    """
    # Tokenize all sentences into a flat ID stream.
    all_ids: List[int] = []
    for s in sentences:
        all_ids.extend(tokenizer.encode(s))
        all_ids.append(tokenizer.eos_id)  # sentence boundary.

    # Split into train / val.
    n = len(all_ids)
    n_val = max(1, int(n * config.val_size))
    train_ids = all_ids[:n - n_val]
    val_ids = all_ids[n - n_val:]

    train_ds = TextDataset(train_ids, seq_len=config.seq_len, pad_id=tokenizer.pad_id)
    val_ds = TextDataset(val_ids, seq_len=config.seq_len, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, drop_last=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_text_corpus(
    text_path: Optional[Path | str] = None,
    n_sentences: int = 500,
    seed: int = 42,
) -> Tuple[List[str], str]:
    """Load a text corpus (from file or synthetic).

    Returns
    -------
    (sentences, source)
    """
    if text_path is not None:
        path = Path(text_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        return sentences, str(path)

    log.info("Using synthetic Hindi/Hinglish text (n=%d, seed=%d)", n_sentences, seed)
    sentences = generate_synthetic_hindi_text(
        n_sentences=n_sentences, seed=seed,
    )
    return sentences, "synthetic"


__all__ = [
    "LMConfig",
    "DEFAULT_CONFIG",
    "TextDataset",
    "BPETokenizer",
    "generate_synthetic_hindi_text",
    "build_dataloaders",
    "load_text_corpus",
]
