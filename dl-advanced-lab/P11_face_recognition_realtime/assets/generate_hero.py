"""
generate_hero
=============

Hero image for the P11 Face Recognition README.

Composes a 2×2 panel:
    - top-left   : synthetic face grid (5 identities × 2 images).
    - top-right  : embedding L2 normalization verification (histogram of norms).
    - bottom-left: FAISS retrieval results (Top-1 + Top-5 accuracy).
    - bottom-right: ONNX vs PyTorch embedding parity scatter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from shared import apply_style  # noqa: E402
apply_style()

from dataset import generate_synthetic_face_dataset  # noqa: E402
from model import (  # noqa: E402
    ArcFaceEmbedder, FaceVectorIndex, AntiSpoofingChecker,
    export_to_onnx, load_onnx_embedder,
)
from train import evaluate_retrieval, compute_verification_roc  # noqa: E402


def main() -> None:
    ds = generate_synthetic_face_dataset(n_identities=5, n_images_per_identity=3, seed=42)
    embedder = ArcFaceEmbedder(device="cpu")
    embedder.load()
    index = FaceVectorIndex(embedding_dim=512)

    from dataset import align_face
    all_embs = []
    for img in ds.images:
        aligned = align_face(img.image, img.landmarks)
        emb = embedder.embed(aligned)
        index.add(emb, img.identity)
        all_embs.append(emb)
    all_embs = np.array(all_embs)

    retrieval = evaluate_retrieval(index, embedder, ds.images, k_values=[1, 5])
    roc_data = compute_verification_roc(embedder, ds.images)

    # ONNX parity.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "arcface.onnx"
        export_to_onnx(embedder, onnx_path)
        onnx_embedder = load_onnx_embedder(onnx_path)
        pt_emb = embedder.embed(ds.images[0].image)
        onnx_emb = onnx_embedder.embed(ds.images[0].image)

    # Plot.
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # Top-left: synthetic face grid.
    ax = axes[0, 0]
    n_show = min(5, ds.n_identities)
    grid = np.zeros((112 * 2, 112 * n_show, 3), dtype=np.uint8)
    for id_idx in range(n_show):
        for img_idx in range(2):
            img = ds.images[id_idx * 3 + img_idx]
            grid[img_idx * 112 : (img_idx + 1) * 112,
                 id_idx * 112 : (id_idx + 1) * 112] = img.image[:, :, ::-1]  # BGR→RGB
    ax.imshow(grid)
    ax.set_title(f"Synthetic faces ({n_show} identities × 2 images)", loc="left", fontsize=11)
    ax.set_xticks([112 * i + 56 for i in range(n_show)])
    ax.set_xticklabels([f"person_{i}" for i in range(n_show)], fontsize=9)
    ax.set_yticks([56, 168])
    ax.set_yticklabels(["img 1", "img 2"], fontsize=9)
    for i in range(n_show + 1):
        ax.axvline(112 * i, color="white", linewidth=0.5)
    for i in range(3):
        ax.axhline(112 * i, color="white", linewidth=0.5)

    # Top-right: embedding L2 norms.
    ax = axes[0, 1]
    norms = np.linalg.norm(all_embs, axis=1)
    ax.hist(norms, bins=20, color="#0072B2", alpha=0.8, edgecolor="white")
    ax.axvline(1.0, color="#D55E00", linestyle="--", linewidth=2, label="||v|| = 1.0 (target)")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Count")
    ax.set_title(f"Embedding L2 normalization ({len(norms)} embeddings)", loc="left", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: retrieval accuracy.
    ax = axes[1, 0]
    ks = [1, 5]
    accs = [retrieval[f"top_{k}_accuracy"] for k in ks]
    colors = ["#0072B2", "#D55E00"]
    bars = ax.bar([f"Top-{k}" for k in ks], accs, color=colors)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f"{acc:.1%}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("FAISS retrieval accuracy (IndexFlatIP)", loc="left", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Bottom-right: ONNX vs PyTorch scatter.
    ax = axes[1, 1]
    ax.scatter(pt_emb, onnx_emb, s=20, color="#0072B2", alpha=0.5)
    lim = [min(pt_emb.min(), onnx_emb.min()) - 0.01,
           max(pt_emb.max(), onnx_emb.max()) + 0.01]
    ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.6, label="parity (y=x)")
    max_diff = float(np.abs(pt_emb - onnx_emb).max())
    ax.set_xlabel("PyTorch embedding values")
    ax.set_ylabel("ONNX embedding values")
    ax.set_title(f"ONNX vs PyTorch parity (max diff = {max_diff:.2e})", loc="left", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Face Recognition — RetinaFace + ArcFace + FAISS + Anti-Spoofing",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
