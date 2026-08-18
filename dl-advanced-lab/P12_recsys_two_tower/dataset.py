"""
dataset
=======

E-commerce / movie recommendation ETL pipeline with synthetic user-item
interaction logs, implicit/explicit feedback, user/item metadata, and
temporal train/val/test splits.

Public surface
--------------
- ``RecConfig``                 : dataclass with embedding dim + split sizes.
- ``Interaction``                : value object for a single user-item interaction.
- ``RecDataset``                 : bundle of interactions + user/item metadata + splits.
- ``generate_synthetic_interactions`` : synthetic interaction generator with
                                          latent-factor user/item embeddings.
- ``build_temporal_splits``       : time-based train/val/test split.
- ``build_leave_one_out_splits``  : leave-one-out split per user (for eval).
- ``load_rec_dataset``            : one-call loader.
- ``InteractionBatch``           : collated batch of (user_ids, item_ids, ratings).
- ``batch_generator``            : yields batches for training.

Design notes
------------
1. **Synthetic generator with latent factors** — we sample user and item
   latent factors from a standard normal, then generate ratings via
   ``rating = sigmoid(user_factor · item_factor + noise) * 4 + 1`` (so
   ratings are in [1, 5]). This gives realistic collaborative-filtering
   structure: users who share latent factors rate similar items similarly.

2. **Temporal split** — interactions are sorted by timestamp, and the
   last ``test_size`` fraction goes to test, the preceding ``val_size``
   fraction goes to val, and the rest goes to train. This prevents
   temporal leakage (training on future data).

3. **Leave-one-out split** — for evaluation, each user's most recent
   interaction goes to the test set, the second-most-recent to val,
   and the rest to train. This is the standard MovieLens-style protocol.

4. **Implicit + explicit feedback** — the generator produces both a
   ``rating`` (1-5, explicit) and an ``interaction`` (1=clicked/purchased,
   0=not interacted, implicit). The two-tower model trains on implicit
   feedback (clicks); the ranking model uses explicit ratings as the
   ranking target.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("rec_dataset")


# ---------------------------------------------------------------------------
# Config & value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecConfig:
    """Configuration for the recommendation dataset."""

    embedding_dim: int = 64
    num_users: int = 500
    num_items: int = 200
    n_interactions: int = 5000
    rating_scale: Tuple[int, int] = (1, 5)
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42


DEFAULT_CONFIG = RecConfig()


@dataclass
class Interaction:
    """A single user-item interaction."""

    user_id: int
    item_id: int
    rating: float          # explicit feedback (1-5)
    clicked: int            # implicit feedback (0 or 1)
    timestamp: int          # Unix timestamp (or ordinal)
    user_features: Optional[np.ndarray] = None
    item_features: Optional[np.ndarray] = None


@dataclass
class InteractionBatch:
    """Collated batch of interactions."""

    user_ids: np.ndarray   # (B,) int64
    item_ids: np.ndarray   # (B,) int64
    ratings: np.ndarray    # (B,) float32
    clicked: np.ndarray    # (B,) int64


@dataclass(frozen=True)
class RecDataset:
    """Bundle of interactions + metadata + splits."""

    config: RecConfig
    interactions: pd.DataFrame
    user_features: np.ndarray   # (num_users, n_user_features)
    item_features: np.ndarray    # (num_items, n_item_features)
    user_latent: np.ndarray      # (num_users, embedding_dim) — ground truth
    item_latent: np.ndarray      # (num_items, embedding_dim) — ground truth
    source: str
    sha256: str
    n_interactions: int = field(init=False)
    n_users: int = field(init=False)
    n_items: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_interactions", len(self.interactions))
        object.__setattr__(self, "n_users", self.config.num_users)
        object.__setattr__(self, "n_items", self.config.num_items)


# ---------------------------------------------------------------------------
# Synthetic interaction generator
# ---------------------------------------------------------------------------
def generate_synthetic_interactions(
    config: RecConfig = DEFAULT_CONFIG,
) -> RecDataset:
    """Generate a synthetic recommendation dataset with latent factors.

    The generator:
        1. Samples user latent factors: U ∈ R^{n_users × d} ~ N(0, 1)
        2. Samples item latent factors: V ∈ R^{n_items × d} ~ N(0, 1)
        3. For each interaction:
             - Sample a user + item (biased toward users/items with
               higher norms = more active/popular).
             - Compute: affinity = sigmoid(U[u] · V[v] / sqrt(d))
             - Rating = 1 + affinity * 4 + noise (so ratings ∈ [1, 5])
             - Clicked = 1 if affinity > threshold (0.4)
             - Timestamp = sequential (for temporal split).

    Returns
    -------
    RecDataset
    """
    rng = np.random.default_rng(config.seed)
    d = config.embedding_dim
    n_u, n_i = config.num_users, config.num_items

    # Latent factors.
    user_latent = rng.standard_normal((n_u, d)).astype(np.float32)
    item_latent = rng.standard_normal((n_i, d)).astype(np.float32)

    # User/item activity/popularity (higher norm → more interactions).
    user_activity = np.linalg.norm(user_latent, axis=1)
    item_popularity = np.linalg.norm(item_latent, axis=1)
    # Normalize to probabilities.
    user_probs = user_activity / user_activity.sum()
    item_probs = item_popularity / item_popularity.sum()

    # Generate interactions.
    n = config.n_interactions
    user_ids = rng.choice(n_u, size=n, p=user_probs)
    item_ids = rng.choice(n_i, size=n, p=item_probs)

    # Compute affinities.
    affinities = np.array([
        float(np.dot(user_latent[u], item_latent[i]) / np.sqrt(d))
        for u, i in zip(user_ids, item_ids)
    ])
    # Sigmoid.
    affinities = 1.0 / (1.0 + np.exp(-affinities))

    # Ratings: 1 + affinity * 4 + noise → [1, 5].
    ratings = 1.0 + affinities * 4.0 + rng.normal(0, 0.3, size=n)
    ratings = np.clip(ratings, 1.0, 5.0)

    # Clicked: 1 if affinity > threshold.
    clicked = (affinities > 0.45).astype(int)

    # Timestamps: sequential.
    timestamps = np.arange(n)

    df = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "rating": ratings.astype(np.float32),
        "clicked": clicked,
        "timestamp": timestamps,
    })

    # User/item features (metadata): a few categorical + numeric features.
    user_features = np.column_stack([
        rng.integers(0, 5, size=n_u),   # age_group (0-4)
        rng.integers(0, 2, size=n_u),   # gender (0/1)
        rng.integers(0, 10, size=n_u),  # location (0-9)
    ]).astype(np.float32)

    item_features = np.column_stack([
        rng.integers(0, 5, size=n_i),  # category (0-4)
        rng.integers(0, 3, size=n_i),  # price_tier (0-2)
        rng.integers(0, 10, size=n_i), # brand (0-9)
    ]).astype(np.float32)

    sha_input = f"synthetic_rec_u{n_u}_i{n_i}_n{n}_d{d}_s{config.seed}".encode()
    sha = hashlib.sha256(sha_input).hexdigest()

    return RecDataset(
        config=config,
        interactions=df,
        user_features=user_features,
        item_features=item_features,
        user_latent=user_latent,
        item_latent=item_latent,
        source="synthetic",
        sha256=sha,
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def build_temporal_splits(
    ds: RecDataset,
    val_size: Optional[float] = None,
    test_size: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based train/val/test split.

    Interactions are sorted by timestamp; the last ``test_size`` fraction
    goes to test, the preceding ``val_size`` to val, the rest to train.
    """
    val_size = val_size if val_size is not None else ds.config.val_size
    test_size = test_size if test_size is not None else ds.config.test_size

    df = ds.interactions.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1.0 - test_size))
    val_start = int(n * (1.0 - test_size - val_size))

    train = df.iloc[:val_start].reset_index(drop=True)
    val = df.iloc[val_start:test_start].reset_index(drop=True)
    test = df.iloc[test_start:].reset_index(drop=True)
    return train, val, test


def build_leave_one_out_splits(
    ds: RecDataset,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-one-out split: each user's most recent interaction → test,
    second-most-recent → val, rest → train.
    """
    df = ds.interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    train_rows, val_rows, test_rows = [], [], []
    for user_id, group in df.groupby("user_id"):
        n = len(group)
        if n >= 3:
            train_rows.append(group.iloc[:-2])
            val_rows.append(group.iloc[-2:-1])
            test_rows.append(group.iloc[-1:])
        elif n == 2:
            train_rows.append(group.iloc[:-1])
            val_rows.append(group.iloc[-1:])
        else:
            train_rows.append(group)

    train = pd.concat(train_rows).reset_index(drop=True) if train_rows else pd.DataFrame(columns=df.columns)
    val = pd.concat(val_rows).reset_index(drop=True) if val_rows else pd.DataFrame(columns=df.columns)
    test = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame(columns=df.columns)
    return train, val, test


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------
def batch_generator(
    df: pd.DataFrame,
    batch_size: int = 256,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Iterator[InteractionBatch]:
    """Yield batches of interactions."""
    n = len(df)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        batch = df.iloc[batch_idx]
        yield InteractionBatch(
            user_ids=batch["user_id"].values.astype(np.int64),
            item_ids=batch["item_id"].values.astype(np.int64),
            ratings=batch["rating"].values.astype(np.float32),
            clicked=batch["clicked"].values.astype(np.int64),
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_rec_dataset(
    csv_path: Optional[Path | str] = None,
    config: RecConfig = DEFAULT_CONFIG,
) -> RecDataset:
    """One-call loader (CSV override or synthetic)."""
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Interactions CSV not found: {path}")
        df = pd.read_csv(path)
        source = str(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        # Extract metadata.
        n_users = df["user_id"].nunique()
        n_items = df["item_id"].nunique()
        config = RecConfig(
            embedding_dim=config.embedding_dim,
            num_users=n_users,
            num_items=n_items,
            n_interactions=len(df),
            rating_scale=config.rating_scale,
            val_size=config.val_size,
            test_size=config.test_size,
            seed=config.seed,
        )
        user_features = np.zeros((n_users, 3), dtype=np.float32)
        item_features = np.zeros((n_items, 3), dtype=np.float32)
        user_latent = np.zeros((n_users, config.embedding_dim), dtype=np.float32)
        item_latent = np.zeros((n_items, config.embedding_dim), dtype=np.float32)
        return RecDataset(
            config=config, interactions=df,
            user_features=user_features, item_features=item_features,
            user_latent=user_latent, item_latent=item_latent,
            source=source, sha256=sha,
        )
    return generate_synthetic_interactions(config)


__all__ = [
    "RecConfig",
    "Interaction",
    "InteractionBatch",
    "RecDataset",
    "DEFAULT_CONFIG",
    "generate_synthetic_interactions",
    "build_temporal_splits",
    "build_leave_one_out_splits",
    "batch_generator",
    "load_rec_dataset",
]
