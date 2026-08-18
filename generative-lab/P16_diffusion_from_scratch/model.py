"""
model
=====

Complete Diffusion Engine from scratch:
  (a) NoiseScheduler: linear & cosine schedules with precomputed alpha bars.
  (b) UNet Denoiser: sinusoidal time embeddings, ResNet blocks, self-attention,
      class embeddings with null-token dropout for classifier-free guidance.
  (c) DDPM Sampler: T-step stochastic reverse process.
  (d) DDIM Sampler: accelerated deterministic sub-sampling.
  (e) Classifier-Free Guidance: CFG scale w.
  (f) FID Metric Calculator.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# (a) Noise Scheduler
# ---------------------------------------------------------------------------
class NoiseScheduler:
    """Precomputed noise schedule (linear or cosine).

    Stores:
        betas[t]           : noise variance at step t.
        alphas[t] = 1 - betas[t].
        alpha_bars[t] = prod(alphas[0..t]).
        sqrt_alpha_bars[t] = sqrt(alpha_bars[t]).
        sqrt_one_minus_alpha_bars[t].
    """

    def __init__(self, num_timesteps: int = 1000, schedule: str = "linear",
                 beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.schedule = schedule

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            # Cosine schedule from "Improved DDPM" (Nichol & Dhariwal 2021).
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
            alpha_bars = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clamp(betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Precompute all coefficients.
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # For posterior q(x_{t-1} | x_t, x_0):
        self.posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0]), alpha_bars[:-1]])) / (1.0 - alpha_bars)

    def to(self, device: torch.device) -> "NoiseScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
        sqrt_ab = self.sqrt_alpha_bars.gather(0, t).view(-1, 1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars.gather(0, t).view(-1, 1, 1, 1)
        return sqrt_ab * x_0 + sqrt_one_minus_ab * noise


# ---------------------------------------------------------------------------
# (b) UNet Denoiser
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResNetBlock(nn.Module):
    """ResNet block with time + class conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, num_classes: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.class_proj = nn.Embedding(num_classes + 1, out_ch)  # +1 for null token.
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                class_label: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # Add time + class embeddings.
        h = h + self.time_proj(t_emb)[:, :, None, None] + self.class_proj(class_label)[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention (spatial)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, head_dim, HW)
        attn = torch.einsum("bhdn,bhem->bhnm", q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhdn->bhdn", attn, v)  # (B, heads, head_dim, HW)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_classes, use_attn=False, num_heads=4):
        super().__init__()
        self.block = ResNetBlock(in_ch, out_ch, time_dim, num_classes)
        self.attn = SelfAttention(out_ch, num_heads) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb, cls):
        x = self.block(x, t_emb, cls)
        x = self.attn(x)
        return self.down(x), x  # (downsampled, residual for skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, time_dim, num_classes, use_attn=False, num_heads=4):
        super().__init__()
        # up: in_ch → out_ch (reduce channels before concat with skip).
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        # block: out_ch (upsampled) + skip_ch (skip connection) → out_ch.
        self.block = ResNetBlock(out_ch + skip_ch, out_ch, time_dim, num_classes)
        self.attn = SelfAttention(out_ch, num_heads) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb, cls):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t_emb, cls)
        return self.attn(x)


class UNet(nn.Module):
    """UNet denoiser for diffusion models.

    Architecture:
        input → down1 → down2 → mid → up2 → up1 → output
    with sinusoidal time embeddings + class embeddings (null-token dropout for CFG).
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32,
                 num_classes: int = 10, time_dim: int = 128,
                 class_dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.class_dropout = class_dropout
        self.null_token = num_classes  # index for null class (for CFG).

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # Encoder.
        self.down1 = DownBlock(base_ch, base_ch * 2, time_dim, num_classes, use_attn=True)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, time_dim, num_classes, use_attn=True)

        # Bottleneck.
        self.mid = ResNetBlock(base_ch * 4, base_ch * 4, time_dim, num_classes)
        self.mid_attn = SelfAttention(base_ch * 4)

        # Decoder.
        # up2: mid output (base_ch*4) → base_ch*2, skip from down2 (base_ch*4).
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 4, time_dim, num_classes, use_attn=True)
        # up1: up2 output (base_ch*2) → base_ch, skip from down1 (base_ch*2).
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch * 2, time_dim, num_classes, use_attn=True)

        self.output_norm = nn.GroupNorm(8, base_ch)
        self.output_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                class_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W) noisy image.
        t : (B,) diffusion timestep.
        class_label : (B,) class label (or None for unconditional).

        Returns
        -------
        (B, C, H, W) predicted noise.
        """
        B = x.shape[0]
        # Class label with null-token dropout for CFG training.
        if class_label is not None:
            # Dropout: replace with null token with probability class_dropout.
            mask = torch.rand(B, device=x.device) < self.class_dropout
            class_label = class_label.clone()
            class_label[mask] = self.null_token
        else:
            class_label = torch.full((B,), self.null_token, device=x.device, dtype=torch.long)

        t_emb = self.time_embed(t)

        x = self.input_conv(x)
        d1, s1 = self.down1(x, t_emb, class_label)
        d2, s2 = self.down2(d1, t_emb, class_label)

        m = self.mid(d2, t_emb, class_label)
        m = self.mid_attn(m)

        u2 = self.up2(m, s2, t_emb, class_label)
        u1 = self.up1(u2, s1, t_emb, class_label)

        out = F.silu(self.output_norm(u1))
        return self.output_conv(out)


# ---------------------------------------------------------------------------
# (c) DDPM Sampler
# ---------------------------------------------------------------------------
class DDPMSampler:
    """T-step stochastic reverse process.

    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1-alpha_bar_t) * eps_pred)
              + sigma_t * z   (z ~ N(0, I))
    """

    def __init__(self, scheduler: NoiseScheduler, model: UNet):
        self.scheduler = scheduler
        self.model = model

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], device: torch.device,
               class_labels: Optional[torch.Tensor] = None,
               cfg_scale: float = 1.0) -> torch.Tensor:
        """Generate images via DDPM reverse process."""
        self.scheduler.to(device)
        self.model.eval()
        B = shape[0]
        x = torch.randn(shape, device=device)
        T = self.scheduler.num_timesteps

        for t in reversed(range(T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Classifier-free guidance.
            if cfg_scale != 1.0 and class_labels is not None:
                eps_cond = self.model(x, t_batch, class_labels)
                eps_uncond = self.model(x, t_batch, None)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = self.model(x, t_batch, class_labels)

            alpha = self.scheduler.alphas[t]
            alpha_bar = self.scheduler.alpha_bars[t]
            beta = self.scheduler.betas[t]

            mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * eps)

            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.scheduler.posterior_variance[t])
                x = mean + sigma * noise
            else:
                x = mean

        return x


# ---------------------------------------------------------------------------
# (d) DDIM Sampler
# ---------------------------------------------------------------------------
class DDIMSampler:
    """Accelerated deterministic sub-sampling.

    Uses a subset of timesteps (tau_schedule) to skip steps, giving
    O(50) instead of O(1000) generation with minimal quality loss.
    """

    def __init__(self, scheduler: NoiseScheduler, model: UNet):
        self.scheduler = scheduler
        self.model = model

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], device: torch.device,
               class_labels: Optional[torch.Tensor] = None,
               cfg_scale: float = 1.0,
               num_steps: int = 50) -> torch.Tensor:
        """Generate images via DDIM with ``num_steps`` denoising steps."""
        self.scheduler.to(device)
        self.model.eval()
        B = shape[0]
        T = self.scheduler.num_timesteps

        # Sub-sample timesteps.
        step_indices = torch.linspace(0, T - 1, num_steps, dtype=torch.long)
        x = torch.randn(shape, device=device)

        for i in reversed(range(num_steps)):
            t = step_indices[i].item()
            t_prev = step_indices[i - 1].item() if i > 0 else -1

            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # CFG.
            if cfg_scale != 1.0 and class_labels is not None:
                eps_cond = self.model(x, t_batch, class_labels)
                eps_uncond = self.model(x, t_batch, None)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = self.model(x, t_batch, class_labels)

            alpha_bar_t = self.scheduler.alpha_bars[t]
            if t_prev >= 0:
                alpha_bar_t_prev = self.scheduler.alpha_bars[t_prev]
            else:
                alpha_bar_t_prev = torch.tensor(1.0, device=device)

            # DDIM update (deterministic — no noise).
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            x = torch.sqrt(alpha_bar_t_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_t_prev) * eps

        return x


# ---------------------------------------------------------------------------
# (e) Classifier-Free Guidance (integrated in samplers above)
# ---------------------------------------------------------------------------
def apply_cfg(eps_cond: torch.Tensor, eps_uncond: torch.Tensor,
              cfg_scale: float) -> torch.Tensor:
    """Apply classifier-free guidance.

    eps_guided = eps_uncond + w * (eps_cond - eps_uncond)

    When w=1: eps_guided = eps_cond (no guidance — pure conditional).
    When w=0: eps_guided = eps_uncond (unconditional).
    When w>1: amplifies the conditional direction.
    """
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


# ---------------------------------------------------------------------------
# (f) FID Metric Calculator
# ---------------------------------------------------------------------------
class FIDCalculator:
    """Frechet Inception Distance calculator.

    FID = ||mu_real - mu_fake||² + Tr(C_real + C_fake - 2*sqrt(C_real * C_fake))

    For simplicity (no Inception network), we compute FID directly on
    pixel-space statistics. This is a valid distance metric but less
    perceptually aligned than the standard Inception-feature FID.
    """

    @staticmethod
    def compute(real_images: np.ndarray, fake_images: np.ndarray) -> float:
        """Compute FID between real and fake image sets.

        Parameters
        ----------
        real_images : (N, C, H, W) float32 in [-1, 1].
        fake_images : (M, C, H, W) float32 in [-1, 1].

        Returns
        -------
        float
            FID score (lower = better).
        """
        real_flat = real_images.reshape(real_images.shape[0], -1).astype(np.float64)
        fake_flat = fake_images.reshape(fake_images.shape[0], -1).astype(np.float64)

        mu_real = real_flat.mean(axis=0)
        mu_fake = fake_flat.mean(axis=0)

        C_real = np.cov(real_flat, rowvar=False)
        C_fake = np.cov(fake_flat, rowvar=False)

        diff = mu_real - mu_fake
        fid = float(diff @ diff + np.trace(C_real) + np.trace(C_fake)
                    - 2.0 * np.trace(_sqrt_matrix(C_real @ C_fake)))
        return max(fid, 0.0)


def _sqrt_matrix(M: np.ndarray) -> np.ndarray:
    """Matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0.0)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


__all__ = [
    "NoiseScheduler",
    "SinusoidalTimeEmbedding",
    "ResNetBlock",
    "SelfAttention",
    "UNet",
    "DDPMSampler",
    "DDIMSampler",
    "apply_cfg",
    "FIDCalculator",
]
