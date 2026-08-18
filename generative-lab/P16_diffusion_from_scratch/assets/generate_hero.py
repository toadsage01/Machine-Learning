"""generate_hero — Hero image for P16 Diffusion README."""
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

from model import NoiseScheduler, UNet, DDPMSampler, DDIMSampler, FIDCalculator
from dataset import DiffusionConfig, load_diffusion_dataset


def main():
    config = DiffusionConfig(image_size=28, channels=1, num_classes=10, batch_size=16, seed=42)
    images, labels, source = load_diffusion_dataset("synthetic", n_samples=200, config=config)

    sched = NoiseScheduler(num_timesteps=50, schedule="linear")
    model = UNet(in_channels=1, base_ch=16, num_classes=10, time_dim=64, class_dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    losses = []
    for epoch in range(5):
        total = 0; n = 0
        for i in range(0, len(images), 16):
            x = torch.from_numpy(images[i:i+16]).float()
            y = torch.from_numpy(labels[i:i+16]).long()
            t = torch.randint(0, 50, (len(x),))
            noise = torch.randn_like(x)
            x_t = sched.add_noise(x, noise, t)
            pred = model(x_t, t, y)
            loss = torch.nn.functional.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        losses.append(total / max(n, 1))

    model.eval()
    ddpm = DDPMSampler(sched, model)
    ddim = DDIMSampler(sched, model)
    shape = (8, 1, 28, 28)
    cls = torch.arange(8)
    ddpm_out = ddpm.sample(shape, torch.device("cpu"), class_labels=cls, cfg_scale=1.0)
    ddim_out = ddim.sample(shape, torch.device("cpu"), class_labels=cls, cfg_scale=1.0, num_steps=25)
    fid = FIDCalculator.compute(images[:8], ddim_out.numpy())

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # Top-left: noise schedule.
    ax = axes[0, 0]
    t_vals = np.arange(50)
    ax.plot(t_vals, sched.alpha_bars.numpy(), "-", color="#0072B2", linewidth=2, label="alpha_bar (linear)")
    sched_c = NoiseScheduler(num_timesteps=50, schedule="cosine")
    ax.plot(t_vals, sched_c.alpha_bars.numpy(), "--", color="#D55E00", linewidth=2, label="alpha_bar (cosine)")
    ax.set_xlabel("Timestep t"); ax.set_ylabel("alpha_bar(t)")
    ax.set_title("Noise schedules (alpha_bar monotonically decreasing)", loc="left", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Top-right: training loss.
    ax = axes[0, 1]
    ax.plot(range(1, len(losses)+1), losses, "o-", color="#0072B2", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE noise loss")
    ax.set_title("DDPM training loss", loc="left", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Bottom-left: generated samples.
    ax = axes[1, 0]
    n_show = 8
    grid = np.zeros((2*28, n_show*28), dtype=np.float32)
    for i in range(n_show):
        grid[:28, i*28:(i+1)*28] = ddpm_out[i, 0].numpy()
        grid[28:, i*28:(i+1)*28] = ddim_out[i, 0].numpy()
    ax.imshow(grid, cmap="gray", vmin=-1, vmax=1)
    ax.set_title("Generated samples (top=DDPM, bottom=DDIM)", loc="left", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # Bottom-right: FID + architecture diagram.
    ax = axes[1, 1]; ax.set_axis_off()
    ax.set_title(f"FID = {fid:.2f} | Architecture", loc="left", fontsize=11)
    steps = [
        ("NoiseScheduler\n(linear/cosine)", 0.05, 0.85, "#0072B2"),
        ("UNet\n(time emb + ResNet + attn)", 0.05, 0.6, "#D55E00"),
        ("DDPM\n(stochastic)", 0.4, 0.7, "#009E73"),
        ("DDIM\n(deterministic)", 0.4, 0.45, "#CC79A7"),
        ("CFG\n(w=0,1,2,4)", 0.4, 0.2, "#E69F00"),
        ("FID\nCalculator", 0.7, 0.4, "#56B4E9"),
    ]
    for name, x, y, color in steps:
        ax.scatter(x, y, s=300, color=color, zorder=5, edgecolors="white", linewidth=1)
        ax.text(x, y, name, fontsize=6, ha="center", va="center", color="white", fontweight="bold", zorder=6)
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 1)

    fig.suptitle("Diffusion From Scratch — DDPM + DDIM + CFG + FID (16/16)",
                 fontsize=14, fontweight="bold", x=0.01, ha="left", y=1.02)
    out = PROJECT_ROOT / "assets" / "hero.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"Wrote hero image: {out}  ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
