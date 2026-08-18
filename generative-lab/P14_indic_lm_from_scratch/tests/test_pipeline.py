"""
tests/test_pipeline
===================

End-to-end tests for the P14 Indic LM from scratch.

Coverage:
    * RoPE tensor shapes (cos/sin).
    * KV-cache output parity against non-cached generation.
    * Loss reduction (training reduces loss).
    * Perplexity math (perplexity = exp(loss)).
    * Tokenizer encode/decode roundtrip.
    * CLI smoke test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from dataset import generate_synthetic_hindi_text, BPETokenizer, LMConfig, build_dataloaders  # noqa: E402
from model import IndicLM, RotaryPositionEmbedding, compute_perplexity  # noqa: E402


# ---------------------------------------------------------------------------
# RoPE tests
# ---------------------------------------------------------------------------
def test_rope_cos_sin_shapes():
    """RoPE cos/sin should have shape (max_seq_len, head_dim)."""
    rope = RotaryPositionEmbedding(head_dim=32, max_seq_len=128)
    assert rope.cos.shape == (128, 32), f"cos shape: {rope.cos.shape}"
    assert rope.sin.shape == (128, 32), f"sin shape: {rope.sin.shape}"


def test_rope_cos_sin_values_in_range():
    """cos and sin values must be in [-1, 1]."""
    rope = RotaryPositionEmbedding(head_dim=16, max_seq_len=64)
    assert (rope.cos >= -1.0).all() and (rope.cos <= 1.0).all()
    assert (rope.sin >= -1.0).all() and (rope.sin <= 1.0).all()


def test_rope_position_zero_is_identity():
    """At position 0, cos=1 and sin=0 (no rotation)."""
    rope = RotaryPositionEmbedding(head_dim=16, max_seq_len=64)
    assert torch.allclose(rope.cos[0], torch.ones(16), atol=1e-6)
    assert torch.allclose(rope.sin[0], torch.zeros(16), atol=1e-6)


# ---------------------------------------------------------------------------
# KV-cache parity tests
# ---------------------------------------------------------------------------
def test_kv_cache_output_matches_non_cached():
    """Generation with KV-cache should produce identical tokens to non-cached."""
    torch.manual_seed(42)
    model = IndicLM(vocab_size=100, embed_dim=32, num_heads=2, num_layers=1, ff_dim=64)
    model.eval()
    prompt = torch.tensor([[1, 5, 10, 15, 20]], dtype=torch.long)
    # Greedy (top_k=1, near-zero temperature) for deterministic comparison.
    out_no_cache = model.generate(prompt.clone(), max_new_tokens=5, temperature=1e-8, top_k=1, use_cache=False)
    out_cached = model.generate(prompt.clone(), max_new_tokens=5, temperature=1e-8, top_k=1, use_cache=True)
    assert torch.equal(out_no_cache, out_cached), (
        f"KV-cache mismatch:\n  no-cache: {out_no_cache[0].tolist()}\n  cached: {out_cached[0].tolist()}"
    )


# ---------------------------------------------------------------------------
# Loss reduction tests
# ---------------------------------------------------------------------------
def test_training_reduces_loss():
    """Training for a few steps should reduce the loss."""
    torch.manual_seed(42)
    sentences = generate_synthetic_hindi_text(n_sentences=100, seed=42)
    tok = BPETokenizer(vocab_size=200)
    tok.train(sentences)
    config = LMConfig(vocab_size=tok.vocab_size_actual, embed_dim=32, num_heads=2,
                       num_layers=1, ff_dim=64, seq_len=16, batch_size=4)
    train_loader, _ = build_dataloaders(sentences, tok, config)

    model = IndicLM(vocab_size=tok.vocab_size_actual, embed_dim=32, num_heads=2,
                     num_layers=1, ff_dim=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Get initial loss.
    model.eval()
    with torch.no_grad():
        x, y = next(iter(train_loader))
        initial_loss = torch.nn.functional.cross_entropy(
            model(x).view(-1, tok.vocab_size_actual), y.view(-1),
        ).item()

    # Train for 5 steps.
    model.train()
    for i, (x, y) in enumerate(train_loader):
        if i >= 5:
            break
        loss = torch.nn.functional.cross_entropy(
            model(x).view(-1, tok.vocab_size_actual), y.view(-1),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Get final loss.
    model.eval()
    with torch.no_grad():
        x, y = next(iter(train_loader))
        final_loss = torch.nn.functional.cross_entropy(
            model(x).view(-1, tok.vocab_size_actual), y.view(-1),
        ).item()

    assert final_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Perplexity tests
# ---------------------------------------------------------------------------
def test_perplexity_equals_exp_loss():
    """Perplexity = exp(loss) for reasonable loss values."""
    import math
    for loss in [1.0, 2.0, 3.0, 5.0, 10.0]:
        ppl = compute_perplexity(loss)
        expected = math.exp(loss)
        assert abs(ppl - expected) < 0.01, f"loss={loss}: ppl={ppl}, expected={expected}"


def test_perplexity_is_capped():
    """Perplexity should be capped at exp(20) to avoid overflow."""
    import math
    ppl = compute_perplexity(100.0)  # unreasonably high loss.
    assert ppl == math.exp(20.0), f"Expected capped perplexity {math.exp(20):.2f}, got {ppl}"


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------
def test_tokenizer_encode_decode_roundtrip():
    """encode → decode should recover the original text (approximately)."""
    sentences = generate_synthetic_hindi_text(n_sentences=50, seed=42)
    tok = BPETokenizer(vocab_size=200)
    tok.train(sentences)
    text = sentences[0]
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    # The decoded text may differ slightly (whitespace normalization) but
    # should contain the key words.
    for word in text.split():
        if word:
            assert word in decoded, f"Word '{word}' lost in roundtrip: '{decoded}'"


# ---------------------------------------------------------------------------
# Model shape tests
# ---------------------------------------------------------------------------
def test_model_output_shape():
    """Model forward pass should produce (batch, seq_len, vocab_size)."""
    model = IndicLM(vocab_size=100, embed_dim=32, num_heads=2, num_layers=1, ff_dim=64)
    x = torch.randint(0, 100, (4, 16))
    logits = model(x)
    assert logits.shape == (4, 16, 100), f"Expected (4, 16, 100), got {logits.shape}"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def test_cli_runs_end_to_end():
    """Full `python train.py` should exit 0 + write JSON."""
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--n-sentences", "50", "--vocab-size", "100",
        "--embed-dim", "32", "--num-heads", "2", "--num-layers", "1",
        "--ff-dim", "64", "--seq-len", "16", "--batch-size", "4",
        "--epochs", "1",
        "--metrics-json", "/tmp/_p14_cli_metrics.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"
    assert "FINAL_VAL_LOSS=" in result.stdout
    assert "N_PARAMS=" in result.stdout
    assert "VOCAB_SIZE=" in result.stdout
    assert Path("/tmp/_p14_cli_metrics.json").exists()
    import json
    payload = json.loads(Path("/tmp/_p14_cli_metrics.json").read_text())
    assert "history" in payload


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math
    tests = [
        test_rope_cos_sin_shapes,
        test_rope_cos_sin_values_in_range,
        test_rope_position_zero_is_identity,
        test_kv_cache_output_matches_non_cached,
        test_training_reduces_loss,
        test_perplexity_equals_exp_loss,
        test_perplexity_is_capped,
        test_tokenizer_encode_decode_roundtrip,
        test_model_output_shape,
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
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            n_failed += 1
    print(f"\n{n_passed} passed, {n_failed} failed (out of {len(tests)} total).")
    if n_failed > 0:
        sys.exit(1)
