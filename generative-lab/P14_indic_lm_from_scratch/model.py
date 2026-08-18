"""
model
=====

Decoder-only Transformer for Indic language modeling:
  - RoPE Rotary Position Embeddings
  - RMSNorm (replacing LayerNorm)
  - SwiGLU MLP (gated linear unit)
  - Multi-Head Self-Attention with Causal Mask & KV-Cache
  - Temperature / Top-k / Top-p sampling
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE: Rotary Position Embedding
# ---------------------------------------------------------------------------
class RotaryPositionEmbedding:
    """Precomputed RoPE cos/sin tables.

    RoPE rotates each pair of dimensions of the query/key vectors by an
    angle proportional to the position.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = positions[:, None] * freqs[None, :]
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        self.cos = cos
        self.sin = sin

    def apply(self, q: torch.Tensor, k: torch.Tensor,
              offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos[offset : offset + seq_len].to(q.device)
        sin = self.sin[offset : offset + seq_len].to(q.device)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        q_rotated = q * cos + rotate_half(q) * sin
        k_rotated = k * cos + rotate_half(k) * sin
        return q_rotated, k_rotated


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """SwiGLU gated MLP: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down"""

    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, ff_dim, bias=False)
        self.w_up = nn.Linear(dim, ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with Causal Mask + KV-Cache
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with KV-cache support."""

    def __init__(self, embed_dim: int, num_heads: int, head_dim: Optional[int] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.w_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len=1024)

        self._kv_cache_k: Optional[torch.Tensor] = None
        self._kv_cache_v: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, use_cache: bool = False,
                cache_offset: int = 0) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.w_qkv(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope.apply(q, k, offset=cache_offset)

        if use_cache:
            if self._kv_cache_k is not None:
                k = torch.cat([self._kv_cache_k, k], dim=2)
                v = torch.cat([self._kv_cache_v, v], dim=2)
            self._kv_cache_k = k
            self._kv_cache_v = v

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        S_q = q.shape[2]
        S_k = k.shape[2]
        if S_q > 1 or not use_cache:
            causal_mask = torch.triu(
                torch.ones(S_q, S_k, device=x.device, dtype=torch.bool),
                diagonal=S_k - S_q + 1,
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S_q, D)
        return self.w_o(out)

    def reset_cache(self) -> None:
        self._kv_cache_k = None
        self._kv_cache_v = None


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 head_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, head_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.mlp = SwiGLU(embed_dim, ff_dim)

    def forward(self, x: torch.Tensor, use_cache: bool = False,
                cache_offset: int = 0) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), use_cache=use_cache, cache_offset=cache_offset)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full language model
# ---------------------------------------------------------------------------
class IndicLM(nn.Module):
    """Decoder-only Transformer (LLaMA-style) for Indic language modeling."""

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 4, ff_dim: int = 512,
                 max_seq_len: int = 128, head_dim: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, head_dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying.
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor, use_cache: bool = False,
                cache_offset: int = 0) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, use_cache=use_cache, cache_offset=cache_offset)
        x = self.norm(x)
        return self.lm_head(x)

    def reset_cache(self) -> None:
        for block in self.blocks:
            block.attn.reset_cache()

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, use_cache: bool = True) -> torch.Tensor:
        self.eval()
        if use_cache:
            self.reset_cache()
        for step in range(max_new_tokens):
            if use_cache:
                if step == 0:
                    logits = self(token_ids, use_cache=True, cache_offset=0)
                else:
                    last = token_ids[:, -1:]
                    logits = self(last, use_cache=True, cache_offset=token_ids.shape[1] - 1)
            else:
                logits = self(token_ids, use_cache=False, cache_offset=0)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None and top_k > 0:
                top_k = min(top_k, next_logits.shape[-1])
                topk_vals, topk_idx = torch.topk(next_logits, top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(1, topk_idx, topk_vals)

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cum_probs - sorted_probs < top_p
                sorted_logits = sorted_logits * sorted_mask + float("-inf") * (~sorted_mask)
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------
def compute_perplexity(loss: float) -> float:
    """Perplexity = exp(loss)."""
    return float(math.exp(min(loss, 20.0)))


__all__ = [
    "RotaryPositionEmbedding",
    "RMSNorm",
    "SwiGLU",
    "MultiHeadAttention",
    "TransformerBlock",
    "IndicLM",
    "compute_perplexity",
]
