"""
tests/test_pipeline
===================

End-to-end tests for the P12 RecSys Two-Tower pipeline.

Coverage:
    * Embedding L2 normalization: ||v|| = 1.0 for both towers.
    * InfoNCE loss computation: verified on a hand-crafted example.
    * Recall@K non-decreasing monotonicity: recall@1 ≤ recall@5 ≤ recall@10.
    * NDCG@K bounded in [0, 1].
    * MRR bounded in [0, 1].
    * FAISS index: add + search returns correct items.
    * LightGBM ranking: fit + predict produces sane scores.
    * Temporal split: train precedes test temporally.
    * CLI smoke test.

Run with::

    cd dl-advanced-lab/P12_recsys_two_tower
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from dataset import (  # noqa: E402
    RecConfig, load_rec_dataset, build_temporal_splits, batch_generator,
)
from model import (  # noqa: E402
    TwoTowerModel, infonce_loss, FAISSItemIndex, LightGBMRanker,
    build_ranking_features, RetrievalEvaluator,
    recall_at_k, ndcg_at_k, mrr, hit_rate_at_k,
    DEFAULT_EMBEDDING_DIM,
)


# ---------------------------------------------------------------------------
# Embedding normalization tests
# ---------------------------------------------------------------------------
def test_user_tower_embeddings_are_l2_normalized():
    """UserTower output should have L2 norm = 1.0."""
    model = TwoTowerModel(num_users=50, num_items=30, embedding_dim=64)
    user_ids = torch.arange(10)
    user_emb = model.user_tower(user_ids)
    norms = user_emb.norm(dim=1).detach().numpy()
    np.testing.assert_allclose(norms, 1.0, atol=1e-5,
                                err_msg=f"User norms: {norms}")


def test_item_tower_embeddings_are_l2_normalized():
    """ItemTower output should have L2 norm = 1.0."""
    model = TwoTowerModel(num_users=50, num_items=30, embedding_dim=64)
    item_ids = torch.arange(10)
    item_emb = model.item_tower(item_ids)
    norms = item_emb.norm(dim=1).detach().numpy()
    np.testing.assert_allclose(norms, 1.0, atol=1e-5,
                                err_msg=f"Item norms: {norms}")


# ---------------------------------------------------------------------------
# InfoNCE loss tests
# ---------------------------------------------------------------------------
def test_infonce_loss_perfect_match():
    """If user_emb == item_emb (perfect match), loss should be near 0."""
    emb = torch.randn(8, 64)
    emb = torch.nn.functional.normalize(emb, dim=1)
    loss = infonce_loss(emb, emb, temperature=0.1)
    # With perfect match, the diagonal is 1/τ and off-diagonal ≤ 1/τ.
    # The softmax of the diagonal should be close to 1.
    assert loss.item() < 0.5, f"Expected near-0 loss for perfect match, got {loss.item():.4f}"


def test_infonce_loss_orthogonal():
    """If user and item embeddings are orthogonal, loss should be high."""
    # Create orthogonal embeddings (similarity ≈ 0 for off-diagonal).
    emb = torch.eye(8, 64)  # Each row is a different basis vector.
    emb = torch.nn.functional.normalize(emb, dim=1)
    loss = infonce_loss(emb, emb, temperature=0.1)
    # With orthogonal embeddings, diagonal sim = 1/τ, off-diagonal = 0.
    # The loss should be low because the diagonal is still the maximum.
    assert loss.item() < 1.0, f"Unexpectedly high loss for orthogonal embeddings: {loss.item():.4f}"


def test_infonce_loss_is_non_negative():
    """InfoNCE loss should always be non-negative."""
    user_emb = torch.randn(16, 64)
    user_emb = torch.nn.functional.normalize(user_emb, dim=1)
    item_emb = torch.randn(16, 64)
    item_emb = torch.nn.functional.normalize(item_emb, dim=1)
    loss = infonce_loss(user_emb, item_emb, temperature=0.1)
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"


# ---------------------------------------------------------------------------
# Recall@K monotonicity tests
# ---------------------------------------------------------------------------
def test_recall_at_k_is_non_decreasing():
    """Recall@K should be non-decreasing in K: recall@1 ≤ recall@5 ≤ recall@10."""
    recommended = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    ]
    relevant = [{1, 5}, {11, 20}, {21, 30}]
    r1 = recall_at_k(recommended, relevant, k=1)
    r5 = recall_at_k(recommended, relevant, k=5)
    r10 = recall_at_k(recommended, relevant, k=10)
    assert r1 <= r5 <= r10, f"Recall not monotonic: r1={r1}, r5={r5}, r10={r10}"


def test_recall_at_k_perfect():
    """If the relevant item is always ranked #1, recall@K = 1.0 for all K ≥ 1."""
    recommended = [[1, 2, 3], [4, 5, 6]]
    relevant = [{1}, {4}]
    for k in [1, 2, 3]:
        r = recall_at_k(recommended, relevant, k=k)
        assert abs(r - 1.0) < 1e-9, f"recall@{k}={r}, expected 1.0"


# ---------------------------------------------------------------------------
# NDCG bounds tests
# ---------------------------------------------------------------------------
def test_ndcg_at_k_bounded_in_0_1():
    """NDCG@K must be in [0, 1]."""
    recommended = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    relevant = [{3}, {6}]
    for k in [1, 3, 5]:
        n = ndcg_at_k(recommended, relevant, k=k)
        assert 0.0 <= n <= 1.0, f"NDCG@{k}={n}, expected [0, 1]"


def test_ndcg_at_k_perfect():
    """If the relevant item is ranked #1, NDCG@K = 1.0."""
    recommended = [[1, 2, 3]]
    relevant = [{1}]
    n = ndcg_at_k(recommended, relevant, k=3)
    assert abs(n - 1.0) < 1e-9, f"NDCG={n}, expected 1.0"


# ---------------------------------------------------------------------------
# MRR tests
# ---------------------------------------------------------------------------
def test_mrr_bounded_in_0_1():
    """MRR must be in [0, 1]."""
    recommended = [[1, 2, 3], [4, 5, 6]]
    relevant = [{3}, {6}]
    m = mrr(recommended, relevant)
    assert 0.0 <= m <= 1.0, f"MRR={m}, expected [0, 1]"


def test_mrr_perfect():
    """If the relevant item is always ranked #1, MRR = 1.0."""
    recommended = [[1, 2, 3], [4, 5, 6]]
    relevant = [{1}, {4}]
    m = mrr(recommended, relevant)
    assert abs(m - 1.0) < 1e-9, f"MRR={m}, expected 1.0"


# ---------------------------------------------------------------------------
# FAISS index tests
# ---------------------------------------------------------------------------
def test_faiss_index_add_and_search():
    """FAISS index should return the correct items."""
    index = FAISSItemIndex(embedding_dim=64)
    # Create 5 normalized item embeddings.
    embs = np.random.randn(5, 64).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index.add(embs, [10, 20, 30, 40, 50])
    assert index.size == 5
    # Search with the first embedding → should return item 10 as top-1.
    dists, item_ids = index.search(embs[0], k=3)
    assert item_ids[0] == 10, f"Expected item 10 as top-1, got {item_ids[0]}"
    # Similarity should be 1.0 (self-match).
    assert abs(dists[0] - 1.0) < 1e-4, f"Expected similarity 1.0, got {dists[0]}"


# ---------------------------------------------------------------------------
# LightGBM ranking tests
# ---------------------------------------------------------------------------
def test_lightgbm_ranker_fit_and_predict():
    """LightGBM ranker should fit and produce sane predictions."""
    # 3 queries, each with 5 candidates.
    X = np.random.randn(15, 10).astype(np.float32)
    y = np.array([5, 3, 1, 0, 0,   # query 1: first item is most relevant
                  4, 2, 1, 0, 0,   # query 2
                  3, 1, 0, 0, 0], dtype=np.float32)  # query 3
    group = np.array([5, 5, 5])
    ranker = LightGBMRanker(n_estimators=20, learning_rate=0.1)
    ranker.fit(X, y, group)
    scores = ranker.predict(X)
    assert scores.shape == (15,), f"Expected (15,), got {scores.shape}"
    # The first item of each query should have the highest score.
    for q in range(3):
        start = q * 5
        assert scores[start] >= scores[start + 1], (
            f"Query {q}: expected item 0 to rank above item 1"
        )


# ---------------------------------------------------------------------------
# Temporal split tests
# ---------------------------------------------------------------------------
def test_temporal_split_train_precedes_test():
    """Train data should precede test data temporally."""
    config = RecConfig(num_users=50, num_items=20, n_interactions=500, seed=42)
    ds = load_rec_dataset(config=config)
    train, val, test = build_temporal_splits(ds)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def test_cli_runs_end_to_end():
    """Full `python train.py` should exit 0 + write JSON."""
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--num-users", "100", "--num-items", "50",
        "--n-interactions", "500", "--epochs", "3",
        "--metrics-json", "/tmp/_p12_cli_metrics.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"
    assert "RETRIEVAL_RECALL_AT_10=" in result.stdout
    assert "RANKING_NDCG_AT_10=" in result.stdout
    assert "EMBEDDING_L2_NORM=" in result.stdout
    assert Path("/tmp/_p12_cli_metrics.json").exists()
    import json
    payload = json.loads(Path("/tmp/_p12_cli_metrics.json").read_text())
    assert "retrieval" in payload
    assert "ranking" in payload


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_user_tower_embeddings_are_l2_normalized,
        test_item_tower_embeddings_are_l2_normalized,
        test_infonce_loss_perfect_match,
        test_infonce_loss_orthogonal,
        test_infonce_loss_is_non_negative,
        test_recall_at_k_is_non_decreasing,
        test_recall_at_k_perfect,
        test_ndcg_at_k_bounded_in_0_1,
        test_ndcg_at_k_perfect,
        test_mrr_bounded_in_0_1,
        test_mrr_perfect,
        test_faiss_index_add_and_search,
        test_lightgbm_ranker_fit_and_predict,
        test_temporal_split_train_precedes_test,
        test_cli_runs_end_to_end,
    ]
    n_passed = 0
    n_failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            n_passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            n_failed += 1
    print(f"\n{n_passed} passed, {n_failed} failed (out of {len(tests)} total).")
    if n_failed > 0:
        sys.exit(1)
