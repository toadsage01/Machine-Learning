#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P12_recsys_two_tower — trains two-tower retrieval,
indexes items, computes candidate recall@K, trains ranking model on
top-K candidates, and reports final NDCG@K, MRR, and Hit Rate.

Usage
-----
::

    # 1. Default: synthetic data, train + evaluate
    python train.py

    # 2. Custom dataset size
    python train.py --num-users 1000 --num-items 500 --n-interactions 10000

    # 3. Save metrics + plots
    python train.py --metrics-json metrics.json --metrics-plot assets/metrics.png

Exit codes
----------
* 0  : completed.
* 1  : usage error.
* 2  : data loading failed.
* 3  : training failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
try:
    from shared import apply_style
    apply_style()
except Exception:  # pragma: no cover
    pass

import torch  # noqa: E402
from dataset import (  # noqa: E402
    RecConfig, load_rec_dataset, build_temporal_splits, batch_generator,
)
from model import (  # noqa: E402
    TwoTowerModel, FAISSItemIndex, LightGBMRanker,
    build_ranking_features, RetrievalEvaluator,
    recall_at_k, ndcg_at_k, mrr, hit_rate_at_k,
    DEFAULT_EMBEDDING_DIM,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rec_train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rec_train",
        description="P12 RecSys Two-Tower — retrieval + ranking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num-users", type=int, default=500)
    parser.add_argument("--num-items", type=int, default=200)
    parser.add_argument("--n-interactions", type=int, default=5000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--k-candidates", type=int, default=10,
                        help="Number of candidates to retrieve for ranking.")
    parser.add_argument("--lgbm-n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default=None,
                        help="Path to a real interactions CSV (user_id, item_id, rating, clicked, timestamp).")
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--metrics-plot", default=None)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)

    # Step 1 — load dataset.
    try:
        config = RecConfig(
            num_users=args.num_users, num_items=args.num_items,
            n_interactions=args.n_interactions,
            embedding_dim=args.embedding_dim, seed=args.seed,
        )
        log.info("Loading recommendation dataset ...")
        ds = load_rec_dataset(csv_path=args.csv, config=config)
        log.info("  %d interactions, %d users, %d items (source=%s)",
                 ds.n_interactions, ds.n_users, ds.n_items, ds.source)
    except Exception as exc:
        log.error("Data loading failed: %s", exc)
        return 2

    # Step 2 — temporal splits.
    train, val, test = build_temporal_splits(ds)
    log.info("  Splits: train=%d, val=%d, test=%d", len(train), len(val), len(test))

    # Step 3 — train Two-Tower retrieval.
    try:
        log.info("Building Two-Tower model (embedding_dim=%d, temperature=%.2f) ...",
                 args.embedding_dim, args.temperature)
        model = TwoTowerModel(
            num_users=ds.n_users, num_items=ds.n_items,
            embedding_dim=args.embedding_dim,
            temperature=args.temperature,
        )
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        log.info("Training Two-Tower for %d epochs ...", args.epochs)
        for epoch in range(args.epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in batch_generator(train, batch_size=args.batch_size,
                                          shuffle=True, seed=args.seed + epoch):
                u_ids = torch.from_numpy(batch.user_ids).long()
                i_ids = torch.from_numpy(batch.item_ids).long()
                user_emb, item_emb = model(u_ids, i_ids)
                loss = model.compute_loss(user_emb, item_emb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log.info("  Epoch %d/%d — InfoNCE loss=%.4f", epoch + 1, args.epochs, avg_loss)
    except Exception as exc:
        log.error("Training failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 3

    # Step 4 — build FAISS index.
    log.info("Building FAISS item index ...")
    all_item_ids = torch.arange(ds.n_items)
    item_embs = model.get_item_embeddings(all_item_ids).numpy()
    index = FAISSItemIndex(embedding_dim=args.embedding_dim)
    index.add(item_embs, list(range(ds.n_items)))
    log.info("  FAISS index size: %d", index.size)

    # Verify L2 norms.
    norms = np.linalg.norm(item_embs, axis=1)
    log.info("  Item embedding L2 norms: min=%.6f, max=%.6f (expected 1.0)",
             norms.min(), norms.max())

    # Step 5 — evaluate retrieval.
    log.info("Evaluating retrieval ...")
    all_user_ids = torch.arange(ds.n_users)
    user_embs = model.get_user_embeddings(all_user_ids).numpy()

    # Build relevant items per user from test set.
    relevant: Dict[int, set] = {uid: set() for uid in range(ds.n_users)}
    for _, row in test.iterrows():
        relevant[int(row["user_id"])].add(int(row["item_id"]))
    eval_users = [uid for uid in range(ds.n_users) if len(relevant[uid]) > 0]
    eval_user_embs = user_embs[eval_users]

    evaluator = RetrievalEvaluator()
    retrieval_metrics = evaluator.evaluate(index, eval_user_embs, eval_users, relevant)
    log.info("  Recall@1=%.4f, Recall@5=%.4f, Recall@10=%.4f",
             retrieval_metrics.recall_at_1, retrieval_metrics.recall_at_5,
             retrieval_metrics.recall_at_10)
    log.info("  NDCG@5=%.4f, NDCG@10=%.4f, MRR=%.4f, HitRate@10=%.4f",
             retrieval_metrics.ndcg_at_5, retrieval_metrics.ndcg_at_10,
             retrieval_metrics.mrr, retrieval_metrics.hit_rate_at_10)

    # Step 6 — train LightGBM ranking model.
    log.info("Training LightGBM ranking model (LambdaMART) ...")
    k = args.k_candidates
    ranking_features_list = []
    ranking_labels_list = []
    ranking_groups = []
    for uid in eval_users:
        dists, item_ids = index.search(user_embs[uid], k=k)
        if len(item_ids) == 0:
            continue
        u_emb_tiled = np.tile(user_embs[uid], (len(item_ids), 1))
        i_embs = item_embs[item_ids]
        feats = build_ranking_features(
            u_emb_tiled, i_embs,
            np.full(len(item_ids), uid), item_ids,
        )
        labels = np.array([1.0 if iid in relevant[uid] else 0.0 for iid in item_ids])
        ranking_features_list.append(feats)
        ranking_labels_list.append(labels)
        ranking_groups.append(len(item_ids))

    if ranking_features_list:
        X = np.concatenate(ranking_features_list)
        y = np.concatenate(ranking_labels_list)
        groups = np.array(ranking_groups)
        ranker = LightGBMRanker(n_estimators=args.lgbm_n_estimators, learning_rate=0.1)
        ranker.fit(X, y, groups)

        # Step 7 — evaluate ranking.
        log.info("Evaluating ranking (after re-ranking) ...")
        reranked_recs = []
        relevant_list = []
        for uid in eval_users:
            dists, item_ids = index.search(user_embs[uid], k=k)
            if len(item_ids) == 0:
                reranked_recs.append([])
                relevant_list.append(relevant[uid])
                continue
            u_emb_tiled = np.tile(user_embs[uid], (len(item_ids), 1))
            i_embs = item_embs[item_ids]
            feats = build_ranking_features(
                u_emb_tiled, i_embs,
                np.full(len(item_ids), uid), item_ids,
            )
            scores = ranker.predict(feats)
            order = np.argsort(-scores)
            reranked_recs.append(item_ids[order].tolist())
            relevant_list.append(relevant[uid])

        ranking_ndcg5 = ndcg_at_k(reranked_recs, relevant_list, k=5)
        ranking_ndcg10 = ndcg_at_k(reranked_recs, relevant_list, k=10)
        ranking_mrr = mrr(reranked_recs, relevant_list)
        ranking_hr10 = hit_rate_at_k(reranked_recs, relevant_list, k=10)
        log.info("  NDCG@5=%.4f, NDCG@10=%.4f, MRR=%.4f, HitRate@10=%.4f",
                 ranking_ndcg5, ranking_ndcg10, ranking_mrr, ranking_hr10)
    else:
        ranking_ndcg5 = ranking_ndcg10 = ranking_mrr = ranking_hr10 = 0.0
        log.warning("No ranking candidates generated.")

    # Step 8 — metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {
                "num_users": ds.n_users, "num_items": ds.n_items,
                "n_interactions": ds.n_interactions, "embedding_dim": args.embedding_dim,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "temperature": args.temperature,
                "k_candidates": k, "seed": args.seed,
            },
            "retrieval": retrieval_metrics.to_dict(),
            "ranking": {
                "ndcg_at_5": ranking_ndcg5, "ndcg_at_10": ranking_ndcg10,
                "mrr": ranking_mrr, "hit_rate_at_10": ranking_hr10,
            },
            "embedding": {
                "l2_norm_min": float(norms.min()),
                "l2_norm_max": float(norms.max()),
            },
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Step 9 — plots.
    if args.metrics_plot:
        try:
            _plot_metrics(retrieval_metrics, ranking_ndcg5, ranking_ndcg10,
                           ranking_mrr, ranking_hr10, Path(args.metrics_plot))
            log.info("Saved metrics plot → %s", args.metrics_plot)
        except Exception as exc:
            log.warning("Failed to render plot: %s", exc)

    # Summary.
    print(f"RETRIEVAL_RECALL_AT_10={retrieval_metrics.recall_at_10:.4f}")
    print(f"RETRIEVAL_NDCG_AT_10={retrieval_metrics.ndcg_at_10:.4f}")
    print(f"RANKING_NDCG_AT_10={ranking_ndcg10:.4f}")
    print(f"RANKING_MRR={ranking_mrr:.4f}")
    print(f"EMBEDDING_L2_NORM={norms.mean():.6f}")
    return 0


def _plot_metrics(retrieval, rank_ndcg5, rank_ndcg10, rank_mrr, rank_hr10, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    # Left: retrieval Recall@K.
    ks = [1, 5, 10]
    recalls = [retrieval.recall_at_1, retrieval.recall_at_5, retrieval.recall_at_10]
    axes[0].bar([f"@{k}" for k in ks], recalls, color="#0072B2")
    for i, (k, r) in enumerate(zip(ks, recalls)):
        axes[0].text(i, r + 0.01, f"{r:.1%}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, max(max(recalls) * 1.5, 0.1))
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Retrieval Recall@K", loc="left")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Right: ranking vs retrieval NDCG.
    labels = ["NDCG@5", "NDCG@10", "MRR", "HitRate@10"]
    retrieval_vals = [retrieval.ndcg_at_5, retrieval.ndcg_at_10, retrieval.mrr, retrieval.hit_rate_at_10]
    ranking_vals = [rank_ndcg5, rank_ndcg10, rank_mrr, rank_hr10]
    x = np.arange(len(labels))
    width = 0.35
    axes[1].bar(x - width/2, retrieval_vals, width, label="Retrieval", color="#0072B2")
    axes[1].bar(x + width/2, ranking_vals, width, label="Ranking", color="#D55E00")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Retrieval vs Ranking metrics", loc="left")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis="y", alpha=0.3)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
