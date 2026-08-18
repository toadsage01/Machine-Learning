"""generate_hero — Hero image for P14 Indic LM README."""
from __future__ import annotations
import sys, math
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

import warnings; warnings.filterwarnings("ignore")
from shared import apply_style; apply_style()

from dataset import generate_synthetic_hindi_text, BPETokenizer, LMConfig, build_dataloaders
from model import IndicLM, RotaryPositionEmbedding, compute_perplexity


def main():
    sentences = generate_synthetic_hindi_text(n_sentences=200, seed=42)
    tok = BPETokenizer(vocab_size=200)
    tok.train(sentences)
    config = LMConfig(vocab_size=tok.vocab_size_actual, embed_dim=64, num_heads=4,
                      num_layers=2, ff_dim=128, seq_len=32, batch_size=8)
    train_loader, val_loader = build_dataloaders(sentences, tok, config)
    model = IndicLM(vocab_size=tok.vocab_size_actual, embed_dim=64, num_heads=4,
                   num_layers=2, ff_dim=128)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(10):
        total = 0; n = 0
        for x, y in train_loader:
            loss = torch.nn.functional.cross_entropy(
                model(x).view(-1, tok.vocab_size_actual), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        losses.append(total / max(n, 1))

    rope = RotaryPositionEmbedding(head_dim=16, max_seq_len=64)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # Top-left: RoPE cos/sin.
    ax = axes[0, 0]
    ax.plot(rope.cos[:32, :4].numpy(), "--", alpha=0.7)
    ax.set_title("RoPE cos values (first 4 dims, 32 positions)", loc="left", fontsize=11)
    ax.set_xlabel("Position"); ax.set_ylabel("cos(θ)"); ax.grid(True, alpha=0.3)

    # Top-right: training loss.
    ax = axes[0, 1]
    ax.plot(range(1, len(losses)+1), losses, "o-", color="#0072B2", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss (10 epochs)", loc="left", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Bottom-left: architecture diagram.
    ax = axes[1, 0]; ax.set_axis_off()
    ax.set_title("LLaMA-style architecture", loc="left", fontsize=11)
    steps = [
        ("Token IDs\n(1, seq_len)", 0.1, 0.9, "#0072B2"),
        ("Embedding\n(dim)", 0.1, 0.7, "#D55E00"),
        ("RoPE\n(position)", 0.35, 0.7, "#009E73"),
        ("MHA\n(causal)", 0.35, 0.5, "#CC79A7"),
        ("RMSNorm", 0.1, 0.5, "#56B4E9"),
        ("SwiGLU\nMLP", 0.35, 0.3, "#E69F00"),
        ("RMSNorm", 0.1, 0.3, "#56B4E9"),
        ("LM Head\n(vocab)", 0.1, 0.1, "#0072B2"),
        ("KV-Cache", 0.6, 0.5, "#D55E00"),
        ("Top-k/Top-p\nSampling", 0.6, 0.3, "#009E73"),
    ]
    for name, x, y, color in steps:
        ax.scatter(x, y, s=300, color=color, zorder=5, edgecolors="white", linewidth=1)
        ax.text(x, y, name, fontsize=6, ha="center", va="center", color="white",
                fontweight="bold", zorder=6)
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 1)

    # Bottom-right: sample text.
    ax = axes[1, 1]; ax.set_axis_off()
    ax.set_title("Generated text (after training)", loc="left", fontsize=11)
    model.eval()
    prompt_ids = tok.encode(sentences[0][:20])
    if len(prompt_ids) < 2: prompt_ids = [1, 5]
    prompt = torch.tensor([prompt_ids])
    gen = model.generate(prompt, max_new_tokens=15, temperature=0.8, top_k=10)
    text = tok.decode(gen[0].tolist())
    ax.text(0.05, 0.9, f"Prompt:\n{sentences[0][:30]}", fontsize=9, va="top",
            family="monospace", transform=ax.transAxes)
    ax.text(0.05, 0.5, f"Generated:\n{text}", fontsize=9, va="top",
            family="monospace", transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Indic LM From Scratch — Decoder-only Transformer (RoPE + SwiGLU + KV-Cache)",
                 fontsize=14, fontweight="bold", x=0.01, ha="left", y=1.02)
    out = PROJECT_ROOT / "assets" / "hero.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"Wrote hero image: {out}  ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
