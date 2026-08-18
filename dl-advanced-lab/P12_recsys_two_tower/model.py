"""
model
=====

Two-stage recommendation pipeline:
  1. Two-Tower PyTorch Retrieval Engine (User & Item DNNs with 64-D
     normalized embeddings, in-batch softmax / InfoNCE loss).
  2. LightGBM Ranking Model operating on candidate pairs.

Public surface
--------------
- ``UserTower`` / ``ItemTower``   : nn.Module DNN towers producing embeddings.
- ``TwoTowerModel``                : combined model with InfoNCE loss.
- ``infonce_loss``                  : in-batch softmax / InfoNCE loss function.
- ``LightGBMRanker``               : gradient-boosted ranking model.
- ``RecallAtK`` / ``NDCGAtK`` / ``MRR`` : evaluation metrics.
- ``RetrievalEvaluator``           : end-to-end retrieval evaluation.
- ``build_ranking_features``       : construct candidate-pair features for ranking.
- ``FAISSItemIndex``               : FAISS index for item embeddings.

Design notes
------------
1. **Two-Tower architecture** — the user tower and item tower are
   independent DNNs that map their respective input features to a
   shared embedding space. The loss is computed as the dot product of
   user and item embeddings (in-batch negatives = all other items in
   the batch are treated as negatives).

2. **InfoNCE loss** — the InfoNCE (Information Noise Contrastive
   Estimation) loss is the standard contrastive loss for retrieval:
       L = -log(exp(sim(u, i+)) / Σ_j exp(sim(u, i_j)))
   where i+ is the positive item and i_j are the in-batch negatives.
   With a batch size of B, this gives B-1 negatives per positive —
   much more efficient than sampling negatives separately.

3. **L2-normalized embeddings** — after the final linear layer, we
   apply F.normalize to make each embedding unit-length. This makes
   the dot product equivalent to cosine similarity, which is what
   FAISS's IndexFlatIP computes.

4. **LightGBM ranking** — after retrieval, we generate candidate pairs
   (user, retrieved_item) and construct features: user embedding × item
   embedding (element-wise product), user/item metadata, and similarity
   score. LightGBM then re-ranks the candidates using LambdaMART.

5. **Metrics** — Recall@K (does the relevant item appear in the top-K?),
   NDCG@K (discounted cumulative gain, position-weighted), MRR (mean
   reciprocal rank). All are standard information-retrieval metrics.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

try:
    import lightgbm as lgb
    HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover
    HAVE_LIGHTGBM = False

try:
    import faiss
    HAVE_FAISS = True
except Exception:  # pragma: no cover
    HAVE_FAISS = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_DIM = 64


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    hit_rate_at_10: float
    n_queries: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankingMetrics:
    """Ranking evaluation metrics (after LightGBM re-ranking)."""

    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    hit_rate_at_10: float
    n_queries: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Two-Tower model
# ---------------------------------------------------------------------------
class UserTower(nn.Module):
    """User DNN tower: maps user features → embedding.

    Architecture:
        Linear(in_dim, 128) → ReLU → Linear(128, 64) → L2-normalize
    """

    def __init__(self, num_users: int, embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                 user_feature_dim: int = 0):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, 128)
        # Add a small MLP on top of the embedding + user features.
        input_dim = 128 + user_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, user_ids: torch.Tensor,
                user_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.user_embedding(user_ids)  # (B, 128)
        if user_features is not None:
            emb = torch.cat([emb, user_features], dim=-1)
        out = self.mlp(emb)  # (B, embedding_dim)
        return F.normalize(out, p=2, dim=1)  # L2-normalize


class ItemTower(nn.Module):
    """Item DNN tower: maps item features → embedding.

    Architecture:
        Embedding(num_items, 128) → concat(features) → MLP → L2-normalize
    """

    def __init__(self, num_items: int, embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                 item_feature_dim: int = 0):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, 128)
        input_dim = 128 + item_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, item_ids: torch.Tensor,
                item_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.item_embedding(item_ids)
        if item_features is not None:
            emb = torch.cat([emb, item_features], dim=-1)
        out = self.mlp(emb)
        return F.normalize(out, p=2, dim=1)


class TwoTowerModel(nn.Module):
    """Combined Two-Tower model with InfoNCE loss.

    The model computes:
        user_emb = UserTower(user_ids, user_features)   # (B, d)
        item_emb = ItemTower(item_ids, item_features)     # (B, d)
        similarities = user_emb @ item_emb.T              # (B, B)
        loss = InfoNCE(similarities, diagonal=positive)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        user_feature_dim: int = 0,
        item_feature_dim: int = 0,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim, user_feature_dim)
        self.item_tower = ItemTower(num_items, embedding_dim, item_feature_dim)
        self.temperature = temperature
        self.embedding_dim = embedding_dim

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: compute user + item embeddings.

        Returns
        -------
        (user_emb, item_emb)
            Both shape (B, embedding_dim), L2-normalized.
        """
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)
        return user_emb, item_emb

    def compute_loss(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        The similarity matrix is user_emb @ item_emb.T (shape (B, B)).
        The diagonal entries are the positive pairs. The off-diagonal
        entries are the in-batch negatives.

        L = -mean(log(diagonal / row_sum))
        """
        return infonce_loss(user_emb, item_emb, self.temperature)

    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get item embeddings for indexing."""
        with torch.no_grad():
            return self.item_tower(item_ids, item_features)

    def get_user_embeddings(
        self,
        user_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get user embeddings for querying."""
        with torch.no_grad():
            return self.user_tower(user_ids, user_features)


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------
def infonce_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """In-batch softmax / InfoNCE loss.

    The loss treats the diagonal of the similarity matrix as positives
    and the off-diagonal entries as negatives:

        sim = user_emb @ item_emb.T / temperature   # (B, B)
        L = -mean(log(softmax(sim)[diag]))

    Parameters
    ----------
    user_emb, item_emb : torch.Tensor
        Shape (B, D). Must be L2-normalized for cosine similarity.
    temperature : float
        Scaling factor for the logits (lower = sharper distribution).

    Returns
    -------
    torch.Tensor (scalar)
        Mean InfoNCE loss.
    """
    # Similarity matrix: (B, B) — cosine sim since embeddings are normalized.
    sim = user_emb @ item_emb.T / temperature  # (B, B)

    # Labels: the i-th user should match the i-th item (diagonal).
    B = sim.shape[0]
    labels = torch.arange(B, device=sim.device)

    # Cross-entropy loss with the similarity matrix as logits.
    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# FAISS item index
# ---------------------------------------------------------------------------
class FAISSItemIndex:
    """FAISS index for item embeddings (inner product = cosine sim)."""

    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        if not HAVE_FAISS:
            raise RuntimeError("faiss is required for FAISSItemIndex.")
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.item_ids: List[int] = []

    def add(self, embeddings: np.ndarray, item_ids: List[int]) -> None:
        """Add item embeddings to the index."""
        vecs = np.asarray(embeddings, dtype=np.float32)
        # Re-normalize to be safe.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        self.index.add(vecs)
        self.item_ids.extend(item_ids)

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-K items.

        Returns
        -------
        (distances, item_ids)
            ``distances`` shape (k,) — inner product similarities.
            ``item_ids`` shape (k,) — original item IDs.
        """
        vec = np.asarray(query, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        k_actual = min(k, len(self.item_ids))
        distances, indices = self.index.search(vec, k_actual)
        item_ids = np.array([self.item_ids[i] for i in indices[0] if i >= 0])
        return distances[0], item_ids

    def search_batch(self, queries: np.ndarray, k: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Batch search."""
        results = []
        for i in range(len(queries)):
            results.append(self.search(queries[i], k=k))
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal


# ---------------------------------------------------------------------------
# LightGBM ranking model
# ---------------------------------------------------------------------------
class LightGBMRanker:
    """LightGBM ranking model for candidate re-ranking.

    Uses LambdaMART (``objective="lambdarank"``) to re-rank the top-K
    candidates produced by the retrieval model.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 num_leaves: int = 31):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.model: Optional["lgb.LGBMRanker"] = None

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> "LightGBMRanker":
        """Fit the ranking model.

        Parameters
        ----------
        X : np.ndarray, shape (N, F)
            Candidate-pair features.
        y : np.ndarray, shape (N,)
            Relevance scores (higher = more relevant).
        group : np.ndarray, shape (n_queries,)
            Number of candidates per query (must sum to N).
        """
        if not HAVE_LIGHTGBM:
            raise RuntimeError("lightgbm is required for LightGBMRanker.")
        self.model = lgb.LGBMRanker(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            objective="lambdarank",
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y, group=group)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict relevance scores for candidate pairs."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X)


# ---------------------------------------------------------------------------
# Feature engineering for ranking
# ---------------------------------------------------------------------------
def build_ranking_features(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    user_features: Optional[np.ndarray] = None,
    item_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build candidate-pair features for the ranking model.

    Features:
        1. Element-wise product of user_emb × item_emb (64 dims)
        2. Cosine similarity (1 dim)
        3. User features (if provided)
        4. Item features (if provided)

    Returns
    -------
    np.ndarray, shape (N, F)
    """
    features = []

    # 1. Element-wise product.
    ew_product = user_embeddings * item_embeddings
    features.append(ew_product)

    # 2. Cosine similarity (dot product since both are normalized).
    cos_sim = np.sum(user_embeddings * item_embeddings, axis=1, keepdims=True)
    features.append(cos_sim)

    # 3. User features.
    if user_features is not None:
        u_feats = user_features[user_ids]
        features.append(u_feats)

    # 4. Item features.
    if item_features is not None:
        i_feats = item_features[item_ids]
        features.append(i_feats)

    return np.concatenate(features, axis=1)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def recall_at_k(
    recommended: List[List[int]],
    relevant: List[set],
    k: int = 10,
) -> float:
    """Recall@K: fraction of queries where any relevant item is in top-K.

    Parameters
    ----------
    recommended : list of list of int
        Top-K recommended item IDs per query.
    relevant : list of set
        Ground-truth relevant item IDs per query.
    k : int
        Cutoff.

    Returns
    -------
    float
        Mean recall@K across all queries.
    """
    if not recommended:
        return 0.0
    hits = 0
    for rec, rel in zip(recommended, relevant):
        if len(rel) == 0:
            continue
        top_k = set(rec[:k])
        if top_k & rel:
            hits += 1
    return hits / len(recommended)


def ndcg_at_k(
    recommended: List[List[int]],
    relevant: List[set],
    k: int = 10,
) -> float:
    """NDCG@K: normalized discounted cumulative gain.

    DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K

    where ``rel_i = 1`` if the i-th recommended item is relevant, else 0.
    IDCG@K is the ideal DCG (all relevant items ranked first).
    """
    if not recommended:
        return 0.0
    total_ndcg = 0.0
    n_valid = 0
    for rec, rel in zip(recommended, relevant):
        if len(rel) == 0:
            continue
        n_valid += 1
        # DCG.
        dcg = 0.0
        for i, item_id in enumerate(rec[:k]):
            if item_id in rel:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0.
        # IDCG: all relevant items ranked first.
        ideal_n = min(len(rel), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        total_ndcg += ndcg
    return total_ndcg / max(n_valid, 1)


def mrr(
    recommended: List[List[int]],
    relevant: List[set],
) -> float:
    """Mean Reciprocal Rank.

    For each query, find the rank of the first relevant item.
    RR = 1 / rank. MRR = mean(RR) across all queries.
    """
    if not recommended:
        return 0.0
    total_rr = 0.0
    n_valid = 0
    for rec, rel in zip(recommended, relevant):
        if len(rel) == 0:
            continue
        n_valid += 1
        for i, item_id in enumerate(rec):
            if item_id in rel:
                total_rr += 1.0 / (i + 1)
                break
    return total_rr / max(n_valid, 1)


def hit_rate_at_k(
    recommended: List[List[int]],
    relevant: List[set],
    k: int = 10,
) -> float:
    """Hit Rate@K: fraction of queries with at least one hit in top-K.

    (Same as recall@K when each query has exactly 1 relevant item.)
    """
    if not recommended:
        return 0.0
    hits = 0
    n_valid = 0
    for rec, rel in zip(recommended, relevant):
        if len(rel) == 0:
            continue
        n_valid += 1
        if set(rec[:k]) & rel:
            hits += 1
    return hits / max(n_valid, 1)


# ---------------------------------------------------------------------------
# Retrieval evaluator
# ---------------------------------------------------------------------------
class RetrievalEvaluator:
    """End-to-end retrieval evaluation using FAISS index + metrics."""

    def __init__(self, k_values: Tuple[int, ...] = (1, 5, 10)):
        self.k_values = k_values

    def evaluate(
        self,
        index: FAISSItemIndex,
        user_embeddings: np.ndarray,
        user_ids: List[int],
        relevant_items_per_user: Dict[int, set],
        k_max: int = 10,
    ) -> RetrievalMetrics:
        """Evaluate retrieval: for each user, search the FAISS index
        and compute Recall@K / NDCG@K / MRR / HitRate@K.
        """
        recommended: List[List[int]] = []
        relevant: List[set] = []
        for i, uid in enumerate(user_ids):
            dists, item_ids = index.search(user_embeddings[i], k=k_max)
            recommended.append(item_ids.tolist())
            relevant.append(relevant_items_per_user.get(uid, set()))

        return RetrievalMetrics(
            recall_at_1=recall_at_k(recommended, relevant, k=1),
            recall_at_5=recall_at_k(recommended, relevant, k=5),
            recall_at_10=recall_at_k(recommended, relevant, k=10),
            ndcg_at_5=ndcg_at_k(recommended, relevant, k=5),
            ndcg_at_10=ndcg_at_k(recommended, relevant, k=10),
            mrr=mrr(recommended, relevant),
            hit_rate_at_10=hit_rate_at_k(recommended, relevant, k=10),
            n_queries=len(recommended),
        )


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "RetrievalMetrics",
    "RankingMetrics",
    "UserTower",
    "ItemTower",
    "TwoTowerModel",
    "infonce_loss",
    "FAISSItemIndex",
    "LightGBMRanker",
    "build_ranking_features",
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "hit_rate_at_k",
    "RetrievalEvaluator",
    "HAVE_TORCH",
    "HAVE_LIGHTGBM",
    "HAVE_FAISS",
]
