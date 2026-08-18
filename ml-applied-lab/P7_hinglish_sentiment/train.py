#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P7_hinglish_sentiment — TF-IDF baseline vs IndicBERT
fine-tuning on code-mixed Hinglish sentiment.

Usage
-----
::

    # 1. Default: synthetic data, TF-IDF baseline only (fast smoke)
    python train.py

    # 2. Both models
    python train.py --models tfidf_logreg indicbert

    # 3. IndicBERT fine-tuning hyperparams
    python train.py --models indicbert \\
        --epochs 3 --batch-size 16 --lr 2e-5 --max-length 128

    # 4. Real CSV dataset
    python train.py --csv /path/to/hinglish.csv --models tfidf_logreg indicbert

    # 5. Save ONNX export + metrics JSON
    python train.py --onnx-out models/indicbert.onnx --metrics-json metrics.json

Exit codes
----------
* 0  : benchmark completed.
* 1  : usage error.
* 2  : data loading failed.
* 3  : training failed.
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

from dataset import (  # noqa: E402
    DEFAULT_CONFIG, DEFAULT_LABELS, LABEL_TO_IDX, INDICBERT_MODEL_ID,
    GATED_INDICBERT_MODEL_ID, HinglishConfig,
    load_hinglish_dataset, build_stratified_splits,
    detect_script, normalize_text,
)
from model import (  # noqa: E402
    CANDIDATE_MODELS, ModelKind, ClassificationMetrics,
    train_tfidf_baseline, train_indicbert,
    export_to_onnx, load_onnx_session, predict_with_onnx,
    HAVE_HF, HAVE_TORCH,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("hinglish_train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hinglish_train",
        description="P7 Hinglish Sentiment — TF-IDF baseline vs IndicBERT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # Default: synthetic data, TF-IDF only
  python train.py

  # Both models
  python train.py --models tfidf_logreg indicbert

  # IndicBERT fine-tuning
  python train.py --models indicbert --epochs 3 --batch-size 16 --lr 2e-5

  # Real CSV
  python train.py --csv /path/to/hinglish.csv

  # Save ONNX + metrics
  python train.py --models indicbert --onnx-out models/indicbert.onnx \\
      --metrics-json metrics.json --confusion-plot assets/confusion.png
""",
    )
    parser.add_argument(
        "--models", "-m", nargs="+",
        choices=list(CANDIDATE_MODELS.keys()),
        default=["tfidf_logreg"],
        help="Subset of models to evaluate (default: tfidf_logreg).",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to a Hinglish CSV (must have 'text' and 'label' columns).",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=200,
        help="Synthetic dataset size per class (default: 200).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--val-size", type=float, default=0.15,
        help="Validation fraction (default: 0.15).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15,
        help="Test fraction (default: 0.15).",
    )
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="IndicBERT max sequence length (default: 128).",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="IndicBERT fine-tuning epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="IndicBERT batch size (default: 16).",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="IndicBERT learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="IndicBERT weight decay (default: 0.01).",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=0,
        help="IndicBERT warmup steps (default: 0).",
    )
    parser.add_argument(
        "--model-id", default=INDICBERT_MODEL_ID,
        help=f"HF model id (default: {INDICBERT_MODEL_ID}). "
             f"Use '{GATED_INDICBERT_MODEL_ID}' for the real IndicBERT (gated repo).",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Disable text normalization (default: enabled).",
    )
    parser.add_argument(
        "--onnx-out", default=None,
        help="Optional path to save the best model as ONNX.",
    )
    parser.add_argument(
        "--metrics-json", default=None,
        help="Optional path to dump all metrics as JSON.",
    )
    parser.add_argument(
        "--confusion-plot", default=None,
        help="Optional path to save a confusion-matrix PNG per model.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_table(rows: List[Dict]) -> str:
    headers = ["model", "accuracy", "f1_macro", "precision", "recall", "auc", "logloss", "fit_s"]
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r.get(h, ""))))
    sep = "  ".join("-" * widths[h] for h in headers)
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    out = [header_line, sep]
    for r in rows:
        out.append("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers))
    return "\n".join(out)


def _plot_confusion(cm: List[List[int]], labels: List[str], title: str, out_path: Path) -> None:
    """Plot a single confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left")
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "#2b2b2b"
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)
    elif args.verbose == 1:
        log.setLevel(logging.DEBUG)

    # Validate deps for IndicBERT.
    if "indicbert" in args.models:
        if not HAVE_HF or not HAVE_TORCH:
            log.error("IndicBERT requires transformers + torch. Install via `pip install -r requirements.txt`.")
            return 3

    # Step 1 — load dataset.
    try:
        log.info("Loading Hinglish dataset ...")
        config = HinglishConfig(
            labels=DEFAULT_LABELS,
            model_id=args.model_id,
            max_length=args.max_length,
        )
        ds = load_hinglish_dataset(
            csv_path=args.csv,
            n_per_class=args.n_per_class,
            seed=args.seed,
            config=config,
            normalize=not args.no_normalize,
        )
        log.info("  Loaded %d samples (source=%s)", ds.n_samples, ds.source)
        log.info("  Label distribution: %s", ds.df["label"].value_counts().to_dict())
        log.info("  Script distribution: %s", ds.df["script"].value_counts().to_dict())
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 2

    # Step 2 — stratified splits.
    train_df, val_df, test_df = build_stratified_splits(
        ds, val_size=args.val_size, test_size=args.test_size, seed=args.seed,
    )
    log.info("  Splits: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # Step 3 — train + evaluate each model.
    table_rows: List[Dict] = []
    all_metrics: Dict[str, ClassificationMetrics] = {}
    trained_models: Dict[str, object] = {}

    for model_name in args.models:
        try:
            log.info("Training %s ...", model_name)
            if model_name == "tfidf_logreg":
                pipe, metrics = train_tfidf_baseline(
                    train_df["text"], train_df["label_idx"],
                    test_df["text"], test_df["label_idx"],
                    config=config, random_state=args.seed,
                )
                trained_models[model_name] = pipe
            elif model_name == "indicbert":
                classifier, metrics = train_indicbert(
                    train_df["text"], train_df["label_idx"],
                    test_df["text"], test_df["label_idx"],
                    config=config,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    warmup_steps=args.warmup_steps,
                )
                trained_models[model_name] = classifier
            else:
                log.error("Unknown model: %s", model_name)
                return 1

            all_metrics[model_name] = metrics
            table_rows.append({
                "model": model_name,
                "accuracy": f"{metrics.accuracy:.4f}",
                "f1_macro": f"{metrics.f1_macro:.4f}",
                "precision": f"{metrics.precision_macro:.4f}",
                "recall": f"{metrics.recall_macro:.4f}",
                "auc": f"{metrics.roc_auc_ovr:.4f}" if metrics.roc_auc_ovr is not None else "—",
                "logloss": f"{metrics.log_loss:.4f}" if metrics.log_loss is not None else "—",
                "fit_s": f"{metrics.fit_time_seconds:.2f}",
            })
            log.info("  %s — acc=%.4f, f1=%.4f, fit_time=%.2fs",
                     model_name, metrics.accuracy, metrics.f1_macro, metrics.fit_time_seconds)
        except Exception as exc:
            log.error("  %s failed: %s", model_name, exc)
            if args.verbose:
                traceback.print_exc()
            return 3

    # Print the results table.
    print()
    print(_format_table(table_rows))
    print()

    # Step 4 — optional ONNX export (only IndicBERT supported).
    if args.onnx_out:
        if "indicbert" not in trained_models:
            log.warning("--onnx-out is only supported for the indicbert model; skipping.")
        else:
            try:
                classifier = trained_models["indicbert"]
                onnx_path = export_to_onnx(classifier, args.onnx_out, max_length=args.max_length)
                log.info("✓ Exported IndicBERT to ONNX → %s (%.1f KB)",
                         onnx_path.resolve(), onnx_path.stat().st_size / 1024)

                # Verify parity.
                session = load_onnx_session(onnx_path)
                sample_texts = test_df["text"].tolist()[:5]
                pt_labels, pt_probas = classifier.predict(sample_texts)
                onnx_labels, onnx_probas = predict_with_onnx(
                    session, classifier.tokenizer, sample_texts, max_length=args.max_length,
                )
                agreement = (pt_labels == onnx_labels).mean()
                max_diff = np.abs(pt_probas - onnx_probas).max()
                log.info("  ONNX/PyTorch parity: agreement=%.0f%%, max_proba_diff=%.2e",
                         agreement * 100, max_diff)
            except Exception as exc:
                log.error("ONNX export failed: %s", exc)
                if args.verbose:
                    traceback.print_exc()
                return 4

    # Step 5 — optional confusion plot.
    if args.confusion_plot:
        for name, m in all_metrics.items():
            try:
                plot_path = Path(args.confusion_plot).with_name(
                    Path(args.confusion_plot).stem + f"_{name}" + Path(args.confusion_plot).suffix
                )
                _plot_confusion(m.confusion_matrix, list(DEFAULT_LABELS),
                                f"Confusion matrix — {name}", plot_path)
                log.info("Saved confusion plot for %s → %s", name, plot_path)
            except Exception as exc:
                log.warning("Failed to render confusion plot for %s: %s", name, exc)

    # Step 6 — optional metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {
                "models": args.models,
                "csv": args.csv,
                "n_per_class": args.n_per_class,
                "seed": args.seed,
                "val_size": args.val_size,
                "test_size": args.test_size,
                "max_length": args.max_length,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "model_id": args.model_id,
                "normalize": not args.no_normalize,
                "labels": list(DEFAULT_LABELS),
            },
            "results": {name: m.to_dict() for name, m in all_metrics.items()},
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Summary line.
    if all_metrics:
        best_name = max(all_metrics.keys(), key=lambda k: all_metrics[k].accuracy)
        best = all_metrics[best_name]
        print(f"BEST_MODEL={best_name}")
        print(f"BEST_ACCURACY={best.accuracy:.4f}")
        print(f"BEST_F1_MACRO={best.f1_macro:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
