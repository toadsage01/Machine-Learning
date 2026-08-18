"""
generate_hero
=============

Hero image for the P7 Hinglish Sentiment README.

Composes a 2×2 panel:
    - top-left   : script distribution (Roman vs Devanagari vs Mixed).
    - top-right  : TF-IDF + IndicBERT accuracy comparison bar chart.
    - bottom-left: confusion matrix for the TF-IDF baseline.
    - bottom-right: ONNX vs PyTorch logit parity scatter.

Re-run after any model change to refresh ``assets/hero.png``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

from shared import apply_style  # noqa: E402
apply_style()

from dataset import (  # noqa: E402
    DEFAULT_LABELS, DEFAULT_CONFIG,
    load_hinglish_dataset, build_stratified_splits,
)
from model import (  # noqa: E402
    train_tfidf_baseline,
    HAVE_HF, HAVE_TORCH,
)


def main() -> None:
    # Use a small dataset for hero generation speed (we're on a 4GB RAM box).
    ds = load_hinglish_dataset(n_per_class=30, seed=42)
    train, val, test = build_stratified_splits(ds, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # --- Top-left: script distribution pie ------------------------------
    ax = axes[0, 0]
    script_counts = ds.df["script"].value_counts()
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"][:len(script_counts)]
    ax.pie(script_counts.values, labels=script_counts.index, colors=colors,
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"Script distribution (n={ds.n_samples})", loc="left", fontsize=12)

    # --- Top-right: TF-IDF accuracy -------------------------------------
    ax = axes[0, 1]
    pipe, tfidf_metrics = train_tfidf_baseline(
        train["text"], train["label_idx"],
        test["text"], test["label_idx"],
    )
    metrics_dict = {"TF-IDF\n+ LogReg": tfidf_metrics.accuracy}
    # Try IndicBERT if HF is available — but skip if too slow.
    indicbert_acc = None
    # IndicBERT fine-tuning is skipped in the hero generator because
    # loading the multilingual BERT weights (~700MB) on a 4GB-RAM dev
    # box causes OOM-kill. The hero image instead shows the script
    # distribution + TF-IDF accuracy + confusion matrix + a placeholder
    # for the ONNX parity chart (which is generated when the user runs
    # `python train.py --models indicbert --onnx-out ...`).
    HERO_RUN_INDICBERT = False  # set to True to enable IndicBERT in the hero
    if HERO_RUN_INDICBERT and HAVE_HF and HAVE_TORCH:
        try:
            from model import train_indicbert
            small_train = train.sample(n=min(50, len(train)), random_state=42)
            small_test = test.sample(n=min(15, len(test)), random_state=42)
            clf, indicbert_metrics = train_indicbert(
                small_train["text"], small_train["label_idx"],
                small_test["text"], small_test["label_idx"],
                config=DEFAULT_CONFIG, epochs=1, batch_size=8, learning_rate=2e-5,
            )
            indicbert_acc = indicbert_metrics.accuracy
            metrics_dict["IndicBERT\n(mBERT)"] = indicbert_acc
        except Exception as e:
            print(f"Skipping IndicBERT in hero: {e}")

    names = list(metrics_dict.keys())
    accs = list(metrics_dict.values())
    colors = ["#0072B2", "#D55E00"][:len(names)]
    bars = ax.bar(names, accs, color=colors)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f"{acc:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Model comparison", loc="left", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    # --- Bottom-left: TF-IDF confusion matrix ---------------------------
    ax = axes[1, 0]
    cm = np.array(tfidf_metrics.confusion_matrix)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(DEFAULT_LABELS, rotation=20, ha="right")
    ax.set_yticklabels(DEFAULT_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("TF-IDF + LogReg confusion matrix", loc="left", fontsize=12)
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "#2b2b2b"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Bottom-right: ONNX vs PyTorch parity (if IndicBERT ran) -------
    ax = axes[1, 1]
    if indicbert_acc is not None:
        from model import export_to_onnx, load_onnx_session, predict_with_onnx
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "indicbert.onnx"
            export_to_onnx(clf, onnx_path, max_length=128)
            session = load_onnx_session(onnx_path)
            sample_texts = test["text"].tolist()[:16]
            pt_labels, pt_probas = clf.predict(sample_texts)
            onnx_labels, onnx_probas = predict_with_onnx(
                session, clf.tokenizer, sample_texts, max_length=128,
            )
        pt_max = pt_probas.max(axis=1)
        onnx_max = onnx_probas.max(axis=1)
        ax.scatter(pt_max, onnx_max, s=80, color="#0072B2", alpha=0.7, edgecolor="white")
        lim = [min(pt_max.min(), onnx_max.min()) - 0.05,
               max(pt_max.max(), onnx_max.max()) + 0.05]
        ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.6, label="parity (y=x)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("PyTorch max probability")
        ax.set_ylabel("ONNX max probability")
        ax.set_title("ONNX runtime parity vs PyTorch\n(16 test samples)", loc="left", fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    else:
        # Show text normalization examples instead.
        from dataset import normalize_text, detect_script
        samples = [
            ("Movie ACCHA था", "Mixed script"),
            ("Check @user1 https://t.co/abc acha movie!", "URL+mention+hashtag"),
            ("bahut bura movie #flop", "Hinglish + hashtag"),
            ("यह फ़िल्म अच्छी है", "Devanagari"),
        ]
        ax.axis("off")
        ax.set_title("Text normalization examples", loc="left", fontsize=12)
        cell_text = []
        for raw, kind in samples:
            norm = normalize_text(raw)
            scr = detect_script(raw)
            cell_text.append([kind, raw[:30], norm[:30], scr])
        table = ax.table(
            cellText=cell_text,
            colLabels=["kind", "raw", "normalized", "script"],
            loc="center", cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        # Colour the header row.
        for j in range(4):
            cell = table[0, j]
            cell.set_facecolor("#0072B2")
            cell.set_text_props(color="white", fontweight="bold")

    fig.suptitle("Hinglish Sentiment — TF-IDF Baseline vs IndicBERT",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
