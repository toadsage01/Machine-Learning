"""
tests/test_pipeline
===================

End-to-end tests for P16 Diffusion From Scratch — the FINAL capstone (16/16).

Coverage:
    * Forward noising alpha-bar limits: x_T ≈ N(0, I).
    * DDIM vs DDPM output shape parity.
    * CFG zero-guidance equivalence: w=1 → eps_guided = eps_cond.
    * UNet input-output shape matching.
    * FID math: same vs same → 0.
    * Noise schedule: alpha_bars monotonically decreasing.
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

from model import (  # noqa: E402
    NoiseScheduler, UNet, DDPMSampler, DDIMSampler, apply_cfg, FIDCalculator,
)


# ---------------------------------------------------------------------------
# Forward noising tests
# ---------------------------------------------------------------------------
def test_forward_noising_at_t_max_approaches_standard_normal():
    """At t=T-1, x_T = sqrt(alpha_bar_T) * x_0 + sqrt(1-alpha_bar_T) * noise
    should approach N(0, I) when alpha_bar_T is small."""
    sched = NoiseScheduler(num_timesteps=1000, schedule="linear")
    x_0 = torch.randn(1000, 1, 28, 28)
    noise = torch.randn_like(x_0)
    t = torch.full((1000,), 999, dtype=torch.long)
    x_t = sched.add_noise(x_0, noise, t)
    # alpha_bar at t=999 should be very small (≈0.0047 for linear).
    alpha_bar_T = sched.alpha_bars[999].item()
    assert alpha_bar_T < 0.01, f"alpha_bar_T={alpha_bar_T}, expected < 0.01"
    # x_T mean should be close to 0.
    assert abs(x_t.mean().item()) < 0.1, f"x_T mean={x_t.mean():.4f}, expected ~0"
    # x_T std should be close to 1.
    assert abs(x_t.std().item() - 1.0) < 0.1, f"x_T std={x_t.std():.4f}, expected ~1"


def test_alpha_bars_monotonically_decreasing():
    """alpha_bars[t] must be monotonically decreasing (since each alpha < 1)."""
    sched = NoiseScheduler(num_timesteps=100, schedule="linear")
    diffs = np.diff(sched.alpha_bars.numpy())
    assert (diffs <= 0).all(), "alpha_bars not monotonically decreasing"


def test_cosine_schedule_alpha_bars_also_decreasing():
    sched = NoiseScheduler(num_timesteps=100, schedule="cosine")
    diffs = np.diff(sched.alpha_bars.numpy())
    assert (diffs <= 0).all(), "cosine alpha_bars not monotonically decreasing"


# ---------------------------------------------------------------------------
# DDIM vs DDPM output shape parity
# ---------------------------------------------------------------------------
def test_ddim_ddpm_output_shape_parity():
    """DDPM and DDIM should produce images of the same shape."""
    sched = NoiseScheduler(num_timesteps=20, schedule="linear")
    model = UNet(in_channels=1, base_ch=8, num_classes=10, time_dim=32, class_dropout=0.0)
    shape = (4, 1, 28, 28)

    ddpm = DDPMSampler(sched, model)
    ddim = DDIMSampler(sched, model)

    ddpm_out = ddpm.sample(shape, device=torch.device("cpu"), class_labels=torch.tensor([0,1,2,3]))
    ddim_out = ddim.sample(shape, device=torch.device("cpu"), class_labels=torch.tensor([0,1,2,3]), num_steps=10)

    assert ddpm_out.shape == ddim_out.shape, (
        f"Shape mismatch: DDPM={ddpm_out.shape}, DDIM={ddim_out.shape}"
    )
    assert ddpm_out.shape == shape


# ---------------------------------------------------------------------------
# CFG tests
# ---------------------------------------------------------------------------
def test_cfg_w1_equals_conditional():
    """When w=1, eps_guided = eps_cond (no amplification)."""
    eps_cond = torch.randn(4, 1, 28, 28)
    eps_uncond = torch.randn(4, 1, 28, 28)
    guided = apply_cfg(eps_cond, eps_uncond, cfg_scale=1.0)
    assert torch.allclose(guided, eps_cond, atol=1e-6), "CFG w=1 should equal conditional"


def test_cfg_w0_equals_unconditional():
    """When w=0, eps_guided = eps_uncond (purely unconditional)."""
    eps_cond = torch.randn(4, 1, 28, 28)
    eps_uncond = torch.randn(4, 1, 28, 28)
    guided = apply_cfg(eps_cond, eps_uncond, cfg_scale=0.0)
    assert torch.allclose(guided, eps_uncond, atol=1e-6), "CFG w=0 should equal unconditional"


# ---------------------------------------------------------------------------
# UNet shape tests
# ---------------------------------------------------------------------------
def test_unet_input_output_shape_match():
    """UNet output must have the same shape as input."""
    model = UNet(in_channels=1, base_ch=16, num_classes=10, time_dim=64)
    x = torch.randn(4, 1, 28, 28)
    t = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    cls = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    out = model(x, t, cls)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_unet_works_without_class_labels():
    """UNet should work with class_label=None (unconditional)."""
    model = UNet(in_channels=1, base_ch=8, num_classes=10, time_dim=32)
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([5, 10], dtype=torch.long)
    out = model(x, t, None)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# FID tests
# ---------------------------------------------------------------------------
def test_fid_same_vs_same_is_zero():
    """FID between identical distributions should be ~0."""
    real = np.random.randn(100, 1, 28, 28).astype(np.float32)
    fid = FIDCalculator.compute(real, real)
    assert abs(fid) < 0.01, f"FID(same, same)={fid}, expected ~0"


def test_fid_different_distributions_is_positive():
    """FID between different distributions should be > 0."""
    real = np.random.randn(100, 1, 28, 28).astype(np.float32) * 0.5
    fake = np.random.randn(100, 1, 28, 28).astype(np.float32) * 2.0 + 1.0
    fid = FIDCalculator.compute(real, fake)
    assert fid > 0, f"FID(different)={fid}, expected > 0"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def test_cli_runs_end_to_end():
    """Full `python train.py` should exit 0."""
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--dataset", "synthetic", "--n-samples", "50",
        "--image-size", "16", "--channels", "1",
        "--base-ch", "8", "--time-dim", "32", "--timesteps", "10",
        "--batch-size", "8", "--epochs", "1",
        "--metrics-json", "/tmp/_p16_cli_metrics.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"
    assert "FINAL_LOSS=" in result.stdout
    assert "N_PARAMS=" in result.stdout
    assert Path("/tmp/_p16_cli_metrics.json").exists()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_forward_noising_at_t_max_approaches_standard_normal,
        test_alpha_bars_monotonically_decreasing,
        test_cosine_schedule_alpha_bars_also_decreasing,
        test_ddim_ddpm_output_shape_parity,
        test_cfg_w1_equals_conditional,
        test_cfg_w0_equals_unconditional,
        test_unet_input_output_shape_match,
        test_unet_works_without_class_labels,
        test_fid_same_vs_same_is_zero,
        test_fid_different_distributions_is_positive,
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
