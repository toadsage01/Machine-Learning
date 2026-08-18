#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P14_indic_lm_from_scratch — pretrains a decoder-only
Transformer with AdamW + cosine schedule, evaluates validation perplexity,
demonstrates KV-cached autoregressive text generation, and exports checkpoint.

Usage
-----
::

    # 1. Default: synthetic data, 3 epochs
    python train.py

    # 2. Real text file
    python train.py --text-file corpus.txt

    # 3. Save checkpoint + metrics
    python train.py --checkpoint-out models/indic_lm.pth --metrics-json metrics.json

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
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch

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
    LMConfig, generate_synthetic_hindi_text, BPETokenizer,
    build_dataloaders, load_text_corpus,
)
from model import IndicLM, compute_perplexity  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("lm_train")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indic_lm_train",
        description="P14 Indic LM — decoder-only Transformer pretraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--text-file", default=None, help="Path to text corpus.")
    parser.add_argument("--n-sentences", type=int, default=500, help="Synthetic sentences (default: 500).")
    parser.add_argument("--vocab-size", type=int, default=500, help="BPE vocab size (default: 500).")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dim (default: 128).")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads (default: 4).")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers (default: 4).")
    parser.add_argument("--ff-dim", type=int, default=512, help="FFN dim (default: 512).")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length (default: 64).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4).")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01).")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps (default: 10).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", default=None, help="Save model checkpoint.")
    parser.add_argument("--metrics-json", default=None, help="Save training metrics.")
    parser.add_argument("--training-plot", default=None, help="Save loss curve.")
    parser.add_argument("--generate", action="store_true", help="Generate sample text after training.")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    return parser


# ---------------------------------------------------------------------------
# Cosine LR schedule with warmup
# ---------------------------------------------------------------------------
def get_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)

    torch.manual_seed(args.seed)

    # Step 1 — load corpus.
    try:
        log.info("Loading text corpus ...")
        sentences, source = load_text_corpus(args.text_file, args.n_sentences, args.seed)
        log.info("  %d sentences (source=%s)", len(sentences), source)
    except Exception as exc:
        log.error("Failed to load corpus: %s", exc)
        return 2

    # Step 2 — train tokenizer.
    try:
        log.info("Training BPE tokenizer (vocab=%d) ...", args.vocab_size)
        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(sentences)
        log.info("  Actual vocab size: %d", tokenizer.vocab_size_actual)
    except Exception as exc:
        log.error("Tokenizer training failed: %s", exc)
        return 2

    # Step 3 — build dataloaders.
    config = LMConfig(
        vocab_size=tokenizer.vocab_size_actual,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, ff_dim=args.ff_dim,
        seq_len=args.seq_len, batch_size=args.batch_size,
        seed=args.seed,
    )
    train_loader, val_loader = build_dataloaders(sentences, tokenizer, config)
    log.info("  Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

    # Step 4 — build model.
    model = IndicLM(
        vocab_size=tokenizer.vocab_size_actual,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, ff_dim=args.ff_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  Model: %.1fK params (%.2fM)", n_params / 1e3, n_params / 1e6)

    # Step 5 — optimizer + scheduler.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    log.info("  Total steps: %d (warmup=%d)", total_steps, args.warmup_steps)

    # Step 6 — training loop.
    history = []
    best_val_loss = float("inf")
    log.info("Training for %d epoch(s) ...", args.epochs)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            # Update LR.
            step = epoch * len(train_loader) + batch_idx
            lr = get_lr(step, total_steps, args.lr, args.warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, tokenizer.vocab_size_actual), y.view(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validation.
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, tokenizer.vocab_size_actual), y.view(-1),
                )
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)
        val_ppl = compute_perplexity(val_loss)

        log.info("  Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, val_ppl=%.2f",
                 epoch + 1, args.epochs, train_loss, val_loss, val_ppl)
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "val_ppl": val_ppl,
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # Step 7 — generation demo.
    if args.generate:
        log.info("Generating sample text ...")
        model.eval()
        # Encode a prompt.
        prompt_text = sentences[0][:20] if sentences else "accha"
        prompt_ids = tokenizer.encode(prompt_text)
        if len(prompt_ids) < 2:
            prompt_ids = [1, 5]
        prompt = torch.tensor([prompt_ids], dtype=torch.long)
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8,
                                      top_k=10, use_cache=True)
        generated_text = tokenizer.decode(generated[0].tolist())
        log.info("  Prompt: %s", prompt_text)
        log.info("  Generated: %s", generated_text)

    # Step 8 — checkpoint.
    if args.checkpoint_out:
        checkpoint = {
            "model_state": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size_actual,
                "embed_dim": args.embed_dim, "num_heads": args.num_heads,
                "num_layers": args.num_layers, "ff_dim": args.ff_dim,
            },
            "best_val_loss": best_val_loss,
        }
        path = Path(args.checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        log.info("  ✓ Checkpoint saved → %s", path.resolve())

    # Step 9 — metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {
                "vocab_size": tokenizer.vocab_size_actual,
                "embed_dim": args.embed_dim, "num_heads": args.num_heads,
                "num_layers": args.num_layers, "ff_dim": args.ff_dim,
                "seq_len": args.seq_len, "batch_size": args.batch_size,
                "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                "n_params": n_params,
            },
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_ppl": compute_perplexity(best_val_loss),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", metrics_path)

    # Step 10 — training plot.
    if args.training_plot:
        try:
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            epochs = [h["epoch"] for h in history]
            ax.plot(epochs, [h["train_loss"] for h in history], "o-", color="#0072B2", label="train loss")
            ax.plot(epochs, [h["val_loss"] for h in history], "s-", color="#D55E00", label="val loss")
            ax2 = ax.twinx()
            ax2.plot(epochs, [h["val_ppl"] for h in history], "^--", color="#009E73", label="val perplexity", alpha=0.7)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax2.set_ylabel("Perplexity")
            ax.set_title("Indic LM training curves", loc="left")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
            ax.grid(True, alpha=0.3)
            path = Path(args.training_plot)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160)
            plt.close(fig)
            log.info("Saved training plot → %s", path)
        except Exception as exc:
            log.warning("Failed to render plot: %s", exc)

    # Summary.
    final_val_loss = history[-1]["val_loss"] if history else 0.0
    print(f"FINAL_VAL_LOSS={final_val_loss:.4f}")
    print(f"FINAL_VAL_PPL={compute_perplexity(final_val_loss):.2f}")
    print(f"N_PARAMS={n_params}")
    print(f"VOCAB_SIZE={tokenizer.vocab_size_actual}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
