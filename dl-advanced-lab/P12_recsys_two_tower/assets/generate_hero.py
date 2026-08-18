"""
generate_hero
=============

Hero image for the P12 RecSys Two-Tower README.

Composes a 2×2 panel:
    - top-left   : Two-Tower architecture diagram (text-based).
    - top-right  : InfoNCE loss curve over training epochs.
    - bottom-left: Retrieval vs ranking metrics comparison.
    - bottom-right: Embedding L2 norm verification histogram.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from shared import apply_style  # noqa: E402
apply_style()

from dataset import RecConfig, load_rec_dataset, build_temporal_splits, batch_generator  # noqa: E402
from model import (  # noqa: E402
    TwoTowerModel, FAISSItemIndex, LightGBMRanker,
    build_ranking_features, RetrievalEvaluator,
    recall_at_k, ndcg_at_k, mrr, hit_rate_at_k,
)


def main() -> None:
    config = RecConfig(num_users=200, num_items=100, n_interactions=2000,
                       embedding_dim=64, seed=42)
    ds = load_rec_dataset(config=config)
    train, val, test = build_temporal_splits(ds)

    model = TwoTowerModel(num_users=200, num_items=100, embedding_dim=64, temperature=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train + record loss.
    losses = []
    for epoch in range(15):
        total_loss = 0
        n_batches = 0
        for batch in batch_generator(train, batch_size=128, shuffle=True, seed=42+epoch):
            u_ids = torch.from_numpy(batch.user_ids).long()
            i_ids = torch.from_numpy(batch.item_ids).long()
            user_emb, item_emb = model(u_ids, i_ids)
            loss = model.compute_loss(user_emb, item_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        losses.append(total_loss / max(n_batches, 1))

    # Index + evaluate.
    item_ids_t = torch.arange(100)
    item_embs = model.get_item_embeddings(item_ids_t).numpy()
    index = FAISSItemIndex(embedding_dim=64)
    index.add(item_embs, list(range(100)))

    user_ids_t = torch.arange(200)
    user_embs = model.get_user_embeddings(user_ids_t).numpy()

    relevant = {uid: set() for uid in range(200)}
    for _, row in test.iterrows():
        relevant[int(row["user_id"])].add(int(row["item_id"]))
    eval_users = [uid for uid in range(200) if len(relevant[uid]) > 0]
    eval_user_embs = user_embs[eval_users]

    evaluator = RetrievalEvaluator()
    retrieval = evaluator.evaluate(index, eval_user_embs, eval_users, relevant)

    # LightGBM ranking.
    k = 10
    feats_list, labels_list, groups = [], [], []
    for uid in eval_users:
        dists, ids = index.search(user_embs[uid], k=k)
        u_emb = np.tile(user_embs[uid], (len(ids), 1))
        i_embs = item_embs[ids]
        feats = build_ranking_features(u_emb, i_embs, np.full(len(ids), uid), ids)
        labels = np.array([1.0 if iid in relevant[uid] else 0.0 for iid in ids])
        feats_list.append(feats)
        labels_list.append(labels)
        groups.append(len(ids))
    X = np.concatenate(feats_list)
    y = np.concatenate(labels_list)
    groups = np.array(groups)
    ranker = LightGBMRanker(n_estimators=100, learning_rate=0.1)
    ranker.fit(X, y, groups)

    reranked = []
    rel_list = []
    for uid in eval_users:
        dists, ids = index.search(user_embs[uid], k=k)
        u_emb = np.tile(user_embs[uid], (len(ids), 1))
        i_embs = item_embs[ids]
        feats = build_ranking_features(u_emb, i_embs, np.full(len(ids), uid), ids)
        scores = ranker.predict(feats)
        order = np.argsort(-scores)
        reranked.append(ids[order].tolist())
        rel_list.append(relevant[uid])
    rank_ndcg5 = ndcg_at_k(reranked, rel_list, k=5)
    rank_ndcg10 = ndcg_at_k(reranked, rel_list, k=10)
    rank_mrr = mrr(reranked, rel_list)
    rank_hr10 = hit_rate_at_k(reranked, rel_list, k=10)

    # Plot.
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # Top-left: architecture.
    ax = axes[0, 0]
    ax.set_axis_off()
    ax.set_title("Two-Tower architecture", loc="left", fontsize=12)
    steps = [
        ("User IDs", 0.1, 0.8, "#0072B2"),
        ("UserTower\n(Embedding+MLP)", 0.35, 0.8, "#D55E00"),
        ("User Emb\n(64-D, L2-norm)", 0.65, 0.8, "#009E73"),
        ("Item IDs", 0.1, 0.3, "#0072B2"),
        ("ItemTower\n(Embedding+MLP)", 0.35, 0.3, "#D55E00"),
        ("Item Emb\n(64-D, L2-norm)", 0.65, 0.3, "#009E73"),
        ("InfoNCE\nLoss", 0.85, 0.55, "#CC79A7"),
    ]
    for name, x, y, color in steps:
        ax.scatter(x, y, s=400, color=color, zorder=5, edgecolors="white", linewidth=1.5)
        ax.text(x, y, name, fontsize=7, ha="center", va="center", color="white",
                fontweight="bold", zorder=6)
    for (x0, y0), (x1, y1) in [
        ((0.2, 0.8), (0.25, 0.8)), ((0.5, 0.8), (0.55, 0.8)),
        ((0.2, 0.3), (0.25, 0.3)), ((0.5, 0.3), (0.55, 0.3)),
        ((0.75, 0.8), (0.8, 0.65)), ((0.75, 0.3), (0.8, 0.45)),
    ]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#999", lw=1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Top-right: loss curve.
    ax = axes[0, 1]
    ax.plot(range(1, len(losses) + 1), losses, "o-", color="#0072B2", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title("Two-Tower training loss", loc="left", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Bottom-left: retrieval vs ranking.
    ax = axes[1, 0]
    labels = ["NDCG@5", "NDCG@10", "MRR", "HitRate@10"]
    ret_vals = [retrieval.ndcg_at_5, retrieval.ndcg_at_10, retrieval.mrr, retrieval.hit_rate_at_10]
    rank_vals = [rank_ndcg5, rank_ndcg10, rank_mrr, rank_hr10]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, ret_vals, width, label="Retrieval", color="#0072B2")
    ax.bar(x + width/2, rank_vals, width, label="Ranking", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval vs Ranking metrics", loc="left", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Bottom-right: L2 norms.
    ax = axes[1, 1]
    norms = np.linalg.norm(item_embs, axis=1)
    ax.hist(norms, bins=20, color="#009E73", alpha=0.8, edgecolor="white")
    ax.axvline(1.0, color="#D55E00", linestyle="--", linewidth=2, label="||v|| = 1.0 (target)")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Count")
    ax.set_title("Item embedding L2 normalization", loc="left", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("RecSys Two-Tower — Retrieval + LightGBM Ranking",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
