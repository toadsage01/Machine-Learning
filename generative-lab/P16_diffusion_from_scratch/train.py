#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P16_diffusion_from_scratch — the FINAL capstone (16/16).

Trains DDPM on image dataset with MSE noise loss, logs loss progression,
generates image grids via DDPM and DDIM samplers across CFG scales,
evaluates FID, and saves checkpoint.

Usage
-----
::

    # 1. Default: synthetic shapes, 5 epochs
    python train.py

    # 2. MNIST
    python train.py --dataset mnist --epochs 10

    # 3. Generate samples after training
    python train.py --generate --checkpoint-out models/ddpm.pth

    # 4. Save metrics + plots
    python train.py --metrics-json metrics.json --sample-plot assets/samples.png
"""

from __future__ import annotations

import argparse
import json
import logging
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

from dataset import DiffusionConfig, load_diffusion_dataset, build_dataloaders  # noqa: E402
from model import (  # noqa: E402
    NoiseScheduler, UNet, DDPMSampler, DDIMSampler, FIDCalculator,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("diffusion_train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffusion_train",
        description="P16 Diffusion From Scratch — DDPM + DDIM + CFG + FID.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["synthetic", "mnist", "cifar10"],
                        default="synthetic")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--base-ch", type=int, default=32, help="UNet base channels.")
    parser.add_argument("--time-dim", type=int, default=128, help="Time embedding dim.")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps T.")
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate", action="store_true", help="Generate samples after training.")
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--sample-plot", default=None)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # Step 1 — load dataset.
    config = DiffusionConfig(
        image_size=args.image_size, channels=args.channels,
        num_classes=args.num_classes, batch_size=args.batch_size, seed=args.seed,
    )
    try:
        images, labels, source = load_diffusion_dataset(
            args.dataset, n_samples=args.n_samples, config=config,
        )
        log.info("Dataset: %s, shape=%s, range=[%.2f, %.2f], source=%s",
                 args.dataset, images.shape, images.min(), images.max(), source)
    except Exception as exc:
        log.error("Data loading failed: %s", exc)
        return 2

    train_loader, val_loader = build_dataloaders(images, labels, config)
    log.info("  Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

    # Step 2 — build model + scheduler.
    scheduler = NoiseScheduler(
        num_timesteps=args.timesteps, schedule=args.schedule,
    ).to(device)
    model = UNet(
        in_channels=args.channels, base_ch=args.base_ch,
        num_classes=args.num_classes, time_dim=args.time_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("UNet: %.1fK params (%.2fM), schedule=%s, T=%d",
             n_params / 1e3, n_params / 1e6, args.schedule, args.timesteps)

    # Step 3 — training loop (MSE noise loss).
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    log.info("Training for %d epoch(s) ...", args.epochs)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            B = x.shape[0]
            # Sample random timesteps.
            t = torch.randint(0, args.timesteps, (B,), device=device, dtype=torch.long)
            # Sample noise.
            noise = torch.randn_like(x)
            # Forward noising.
            x_t = scheduler.add_noise(x, noise, t)
            # Predict noise.
            pred_noise = model(x_t, t, y)
            # MSE loss.
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        log.info("  Epoch %d/%d — loss=%.6f", epoch + 1, args.epochs, avg_loss)
        history.append({"epoch": epoch + 1, "loss": avg_loss})

    # Step 4 — generate samples.
    sample_info = {}
    if args.generate:
        model.eval()
        n_gen = 16
        gen_labels = torch.arange(n_gen, device=device) % args.num_classes
        shape = (n_gen, args.channels, args.image_size, args.image_size)

        # DDPM sampling (use fewer timesteps for speed).
        log.info("Generating DDPM samples (T=%d) ...", min(args.timesteps, 50))
        ddpm_sampler = DDPMSampler(scheduler, model)
        # Use reduced timesteps for practical generation.
        sched_reduced = NoiseScheduler(num_timesteps=min(args.timesteps, 50),
                                        schedule=args.schedule).to(device)
        ddpm_sampler.scheduler = sched_reduced
        ddpm_samples = ddpm_sampler.sample(shape, device, class_labels=gen_labels, cfg_scale=1.0)
        log.info("  DDPM samples shape: %s", ddpm_samples.shape)

        # DDIM sampling (50 steps).
        log.info("Generating DDIM samples (50 steps) ...")
        ddim_sampler = DDIMSampler(scheduler, model)
        ddim_samples = ddim_sampler.sample(shape, device, class_labels=gen_labels,
                                            cfg_scale=1.0, num_steps=50)
        log.info("  DDIM samples shape: %s", ddim_samples.shape)

        # CFG scales comparison.
        cfg_scales = [0.0, 1.0, 2.0, 4.0]
        for w in cfg_scales:
            samples = ddim_sampler.sample((4, args.channels, args.image_size, args.image_size),
                                          device, class_labels=torch.tensor([0, 1, 2, 3]),
                                          cfg_scale=w, num_steps=50)
            log.info("  CFG w=%.1f: sample range=[%.3f, %.3f]", w, samples.min().item(), samples.max().item())

        # FID.
        real_subset = images[:n_gen]
        fake_np = ddim_samples.cpu().numpy()
        fid = FIDCalculator.compute(real_subset, fake_np)
        log.info("FID (real vs DDIM): %.2f", fid)
        sample_info = {"fid": fid}

        # Sample plot.
        if args.sample_plot:
            try:
                fig, axes = plt.subplots(2, n_gen, figsize=(2 * n_gen, 4), constrained_layout=True)
                for i in range(n_gen):
                    img = ddpm_samples[i, 0].cpu().numpy()
                    axes[0, i].imshow(img, cmap="gray", vmin=-1, vmax=1)
                    axes[0, i].set_title(f"DDPM cls={i % args.num_classes}", fontsize=7)
                    axes[0, i].axis("off")
                    img2 = ddim_samples[i, 0].cpu().numpy()
                    axes[1, i].imshow(img2, cmap="gray", vmin=-1, vmax=1)
                    axes[1, i].set_title(f"DDIM cls={i % args.num_classes}", fontsize=7)
                    axes[1, i].axis("off")
                axes[0, 0].set_ylabel("DDPM", fontsize=10)
                axes[1, 0].set_ylabel("DDIM", fontsize=10)
                fig.suptitle("DDPM vs DDIM generated samples", fontsize=12)
                path = Path(args.sample_plot)
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=140)
                plt.close(fig)
                log.info("Saved sample plot → %s", path)
            except Exception as exc:
                log.warning("Failed to render sample plot: %s", exc)

    # Step 5 — checkpoint.
    if args.checkpoint_out:
        checkpoint = {
            "model_state": model.state_dict(),
            "config": {
                "in_channels": args.channels, "base_ch": args.base_ch,
                "num_classes": args.num_classes, "time_dim": args.time_dim,
                "timesteps": args.timesteps, "schedule": args.schedule,
            },
        }
        path = Path(args.checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        log.info("Checkpoint saved → %s", path.resolve())

    # Step 6 — metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {
                "dataset": args.dataset, "image_size": args.image_size,
                "channels": args.channels, "num_classes": args.num_classes,
                "timesteps": args.timesteps, "schedule": args.schedule,
                "epochs": args.epochs, "lr": args.lr, "n_params": n_params,
            },
            "history": history,
            "samples": sample_info,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = Path(args.metrics_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", path)

    print(f"FINAL_LOSS={history[-1]['loss']:.6f}")
    print(f"N_PARAMS={n_params}")
    if "fid" in sample_info:
        print(f"FID={sample_info['fid']:.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
