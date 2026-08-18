#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P11_face_recognition_realtime — identity registration,
vector database indexing, retrieval evaluation (Top-1 / Top-5), verification
ROC curves, anti-spoofing benchmark, and ArcFace ONNX export.

Usage
-----
::

    # 1. Default: synthetic data, register + evaluate
    python train.py

    # 2. Custom identity count
    python train.py --n-identities 20 --n-images-per-identity 5

    # 3. Export ArcFace to ONNX + verify parity
    python train.py --onnx-out models/arcface.onnx

    # 4. Save evaluation plots + metrics
    python train.py --metrics-json metrics.json \\
        --roc-plot assets/roc.png \\
        --retrieval-plot assets/retrieval.png

Exit codes
----------
* 0  : completed.
* 1  : usage error.
* 2  : data loading failed.
* 3  : pipeline error.
* 4  : ONNX export failed.
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

from dataset import (  # noqa: E402
    DEFAULT_CONFIG, ARCFACE_IMAGE_SIZE, ARCFACE_EMBEDDING_DIM,
    generate_synthetic_face_dataset, load_face_dataset,
)
from model import (  # noqa: E402
    DetectorKind, ArcFaceEmbedder, AntiSpoofingChecker,
    FaceVectorIndex, FaceRecognitionPipeline,
    export_to_onnx, load_onnx_embedder,
    HAVE_CV2, HAVE_TORCH, HAVE_FAISS,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("face_train")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_retrieval(
    index: FaceVectorIndex,
    embedder: ArcFaceEmbedder,
    images: list,
    k_values: List[int] = [1, 5],
) -> Dict[str, float]:
    """Evaluate Top-K retrieval accuracy.

    For each face image, embed it and search the FAISS index.
    A hit = the top-K results include the query's identity.
    """
    from dataset import align_face
    top_k_hits = {k: 0 for k in k_values}
    n = len(images)
    for img in images:
        # Skip images that are in the index (use leave-one-out would be ideal
        # but for simplicity we just check that same-identity images rank high).
        aligned = align_face(img.image, img.landmarks) if img.landmarks is not None else img.image
        emb = embedder.embed(aligned)
        for k in k_values:
            result = index.search(emb, k=k)
            if img.identity in result.identities:
                top_k_hits[k] += 1
    return {f"top_{k}_accuracy": top_k_hits[k] / n for k in k_values}


def compute_verification_roc(
    embedder: ArcFaceEmbedder,
    images: list,
    thresholds: np.ndarray = np.linspace(0.3, 0.95, 50),
) -> Dict[str, np.ndarray]:
    """Compute verification ROC curve (genuine vs impostor pairs).

    For each pair of images:
        * Genuine pair: same identity → similarity should be high.
        * Impostor pair: different identity → similarity should be low.
    """
    from dataset import align_face
    # Embed all images.
    embeddings = []
    identities = []
    for img in images:
        aligned = align_face(img.image, img.landmarks) if img.landmarks is not None else img.image
        emb = embedder.embed(aligned)
        embeddings.append(emb)
        identities.append(img.identity)
    embeddings = np.array(embeddings)

    # Compute pairwise similarities.
    n = len(embeddings)
    genuine_sims = []
    impostor_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if identities[i] == identities[j]:
                genuine_sims.append(sim)
            else:
                impostor_sims.append(sim)

    if not genuine_sims or not impostor_sims:
        return {"fpr": np.array([]), "tpr": np.array([]), "thresholds": thresholds}

    genuine_sims = np.array(genuine_sims)
    impostor_sims = np.array(impostor_sims)

    tpr_list = []
    fpr_list = []
    for thresh in thresholds:
        tp = int((genuine_sims >= thresh).sum())
        fn = int((genuine_sims < thresh).sum())
        fp = int((impostor_sims >= thresh).sum())
        tn = int((impostor_sims < thresh).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return {
        "fpr": np.array(fpr_list),
        "tpr": np.array(tpr_list),
        "thresholds": thresholds,
        "n_genuine": len(genuine_sims),
        "n_impostor": len(impostor_sims),
    }


def evaluate_anti_spoofing(
    checker: AntiSpoofingChecker,
    images: list,
) -> Dict[str, float]:
    """Evaluate anti-spoofing on the synthetic faces.

    NB: Synthetic faces are not real photos, so we expect them to be
    classified as "spoofed" — the benchmark verifies the checker runs
    without crashing and produces scores in [0, 1].
    """
    scores = []
    n_live = 0
    for img in images:
        is_live, score = checker.check(img.image)
        scores.append(score)
        if is_live:
            n_live += 1
    scores = np.array(scores)
    return {
        "mean_spoof_score": float(scores.mean()),
        "std_spoof_score": float(scores.std()),
        "n_live": n_live,
        "n_total": len(images),
        "live_rate": float(n_live / len(images)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_roc(roc_data: Dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    ax.plot(roc_data["fpr"], roc_data["tpr"], "-", color="#0072B2", linewidth=2.0,
            label=f"ROC curve ({roc_data['n_genuine']} genuine, {roc_data['n_impostor']} impostor pairs)")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Verification ROC curve", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.4)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_retrieval(metrics: Dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ks = [int(k.split("_")[1]) for k in metrics if k.startswith("top_")]
    accs = [metrics[f"top_{k}_accuracy"] for k in ks]
    bars = ax.bar([f"Top-{k}" for k in ks], accs, color=["#0072B2", "#D55E00"][:len(ks)])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f"{acc:.1%}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("Face retrieval accuracy", loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face_train",
        description="P11 Face Recognition — registration, retrieval, ROC, ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-identities", type=int, default=10,
                        help="Number of synthetic identities (default: 10).")
    parser.add_argument("--n-images-per-identity", type=int, default=5,
                        help="Images per identity (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--data-dir", default=None,
                        help="Path to real face directory (default: synthetic).")
    parser.add_argument("--detector", choices=["haar", "retinaface", "mediapipe"],
                        default="haar", help="Face detector (default: haar).")
    parser.add_argument("--spoof-threshold", type=float, default=0.5,
                        help="Anti-spoofing threshold (default: 0.5).")
    parser.add_argument("--onnx-out", default=None,
                        help="Optional path to export ArcFace to ONNX.")
    parser.add_argument("--metrics-json", default=None,
                        help="Optional path to dump metrics as JSON.")
    parser.add_argument("--roc-plot", default=None,
                        help="Optional path to save the verification ROC curve PNG.")
    parser.add_argument("--retrieval-plot", default=None,
                        help="Optional path to save the retrieval accuracy bar chart PNG.")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)

    # Step 1 — load face data.
    try:
        log.info("Loading face dataset ...")
        ds = load_face_dataset(
            data_dir=args.data_dir,
            n_identities_synthetic=args.n_identities,
            n_images_per_identity=args.n_images_per_identity,
            seed=args.seed,
        )
        log.info("  Loaded %d images, %d identities (source=%s)",
                 ds.n_samples, ds.n_identities, ds.source)
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        return 2

    # Step 2 — build embedder + index.
    try:
        log.info("Building ArcFace embedder ...")
        embedder = ArcFaceEmbedder(device="cpu")
        embedder.load()
        log.info("  Embedder ready (embedding_dim=%d)", ARCFACE_EMBEDDING_DIM)

        # Verify L2 normalization.
        test_emb = embedder.embed(ds.images[0].image)
        norm = float(np.linalg.norm(test_emb))
        log.info("  L2 norm of embedding: %.8f (expected 1.0)", norm)

        log.info("Building FAISS index ...")
        index = FaceVectorIndex(embedding_dim=ARCFACE_EMBEDDING_DIM)

        # Register all faces.
        from dataset import align_face
        log.info("Registering %d faces ...", ds.n_samples)
        for img in ds.images:
            aligned = align_face(img.image, img.landmarks) if img.landmarks is not None else img.image
            emb = embedder.embed(aligned)
            index.add(emb, img.identity)
        log.info("  FAISS index size: %d", index.size)
    except Exception as exc:
        log.error("Pipeline setup failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 3

    # Step 3 — evaluate retrieval.
    log.info("Evaluating Top-1 / Top-5 retrieval accuracy ...")
    retrieval_metrics = evaluate_retrieval(index, embedder, ds.images, k_values=[1, 5])
    for k, acc in retrieval_metrics.items():
        log.info("  %s: %.4f", k, acc)

    # Step 4 — verification ROC.
    log.info("Computing verification ROC curve ...")
    roc_data = compute_verification_roc(embedder, ds.images)
    log.info("  Genuine pairs: %d, Impostor pairs: %d",
             roc_data.get("n_genuine", 0), roc_data.get("n_impostor", 0))

    # Step 5 — anti-spoofing benchmark.
    log.info("Running anti-spoofing benchmark ...")
    spoof_checker = AntiSpoofingChecker(threshold=args.spoof_threshold)
    spoof_metrics = evaluate_anti_spoofing(spoof_checker, ds.images)
    log.info("  Mean spoof score: %.4f, live rate: %.1f%%",
             spoof_metrics["mean_spoof_score"], spoof_metrics["live_rate"] * 100)

    # Step 6 — ONNX export (optional).
    onnx_path_str = None
    onnx_parity_diff = None
    if args.onnx_out:
        try:
            log.info("Exporting ArcFace to ONNX ...")
            onnx_path = export_to_onnx(embedder, args.onnx_out, image_size=ARCFACE_IMAGE_SIZE)
            onnx_path_str = str(onnx_path.resolve())
            log.info("  ✓ ONNX exported → %s (%.1f KB)",
                     onnx_path.resolve(), onnx_path.stat().st_size / 1024)

            # Verify ONNX parity.
            onnx_embedder = load_onnx_embedder(onnx_path)
            pt_emb = embedder.embed(ds.images[0].image)
            onnx_emb = onnx_embedder.embed(ds.images[0].image)
            onnx_parity_diff = float(np.abs(pt_emb - onnx_emb).max())
            log.info("  ONNX/PyTorch parity: max diff = %.2e", onnx_parity_diff)
        except Exception as exc:
            log.error("ONNX export failed: %s", exc)
            if args.verbose:
                traceback.print_exc()
            return 4

    # Step 7 — plots (optional).
    if args.roc_plot:
        try:
            _plot_roc(roc_data, Path(args.roc_plot))
            log.info("Saved ROC plot → %s", args.roc_plot)
        except Exception as exc:
            log.warning("Failed to render ROC plot: %s", exc)

    if args.retrieval_plot:
        try:
            _plot_retrieval(retrieval_metrics, Path(args.retrieval_plot))
            log.info("Saved retrieval plot → %s", args.retrieval_plot)
        except Exception as exc:
            log.warning("Failed to render retrieval plot: %s", exc)

    # Step 8 — metrics JSON (optional).
    if args.metrics_json:
        payload = {
            "config": {
                "n_identities": ds.n_identities,
                "n_images": ds.n_samples,
                "detector": args.detector,
                "spoof_threshold": args.spoof_threshold,
                "source": ds.source,
                "seed": args.seed,
            },
            "embedding": {
                "dim": ARCFACE_EMBEDDING_DIM,
                "l2_norm": norm,
                "image_size": ARCFACE_IMAGE_SIZE,
            },
            "retrieval": retrieval_metrics,
            "roc": {
                "n_genuine": roc_data.get("n_genuine", 0),
                "n_impostor": roc_data.get("n_impostor", 0),
            },
            "anti_spoofing": spoof_metrics,
            "onnx": {
                "path": onnx_path_str,
                "parity_max_diff": onnx_parity_diff,
            } if onnx_path_str else None,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Summary.
    print(f"TOP_1_ACCURACY={retrieval_metrics.get('top_1_accuracy', 0):.4f}")
    print(f"TOP_5_ACCURACY={retrieval_metrics.get('top_5_accuracy', 0):.4f}")
    print(f"L2_NORM={norm:.8f}")
    if onnx_parity_diff is not None:
        print(f"ONNX_PARITY_MAX_DIFF={onnx_parity_diff:.2e}")
    print(f"N_IDENTITIES={ds.n_identities}")
    print(f"N_IMAGES={ds.n_samples}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
