"""
dataset
=======

Image dataset builder for MNIST / CIFAR-10 / synthetic geometric shapes,
normalized to [-1, 1], with class label mapping and PyTorch DataLoader.

Public surface
--------------
- ``DiffusionConfig``           : dataclass with image size + channels + num_classes.
- ``ImageDataset``              : PyTorch Dataset returning (image_tensor, label).
- ``generate_synthetic_shapes``  : synthetic geometric shapes (circles, squares, triangles).
- ``build_dataloaders``         : one-call helper returning train/val DataLoaders.
- ``load_diffusion_dataset``    : one-call loader.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("diffusion_dataset")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DiffusionConfig:
    image_size: int = 28
    channels: int = 1
    num_classes: int = 10
    batch_size: int = 64
    val_size: float = 0.1
    seed: int = 42


DEFAULT_CONFIG = DiffusionConfig()


# ---------------------------------------------------------------------------
# Synthetic shape generator
# ---------------------------------------------------------------------------
SHAPE_NAMES = ["circle", "square", "triangle", "star", "cross",
               "hexagon", "diamond", "heart", "arrow", "ellipse"]


def generate_synthetic_shapes(
    n_samples: int = 2000,
    image_size: int = 28,
    channels: int = 1,
    num_classes: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic geometric shape images normalized to [-1, 1].

    Each class draws a different geometric shape on a black background.
    Images are float32 in [-1, 1] (standard diffusion normalization).

    Returns
    -------
    (images, labels)
        images: (N, C, H, W) float32 in [-1, 1].
        labels: (N,) int64 in [0, num_classes).
    """
    if not HAVE_CV2:
        raise RuntimeError("opencv is required for generate_synthetic_shapes.")

    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, channels, image_size, image_size), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)

    cx, cy = image_size // 2, image_size // 2
    radius = image_size // 3

    for i in range(n_samples):
        cls = int(rng.integers(0, num_classes))
        labels[i] = cls
        img = np.zeros((image_size, image_size), dtype=np.uint8)
        color = 255
        offset_x = int(rng.integers(-3, 4))
        offset_y = int(rng.integers(-3, 4))
        r = max(3, radius + int(rng.integers(-3, 4)))

        if cls == 0:  # circle
            cv2.circle(img, (cx + offset_x, cy + offset_y), r, color, -1)
        elif cls == 1:  # square
            cv2.rectangle(img, (cx - r + offset_x, cy - r + offset_y),
                          (cx + r + offset_x, cy + r + offset_y), color, -1)
        elif cls == 2:  # triangle
            pts = np.array([[cx, cy - r + offset_y],
                            [cx - r + offset_x, cy + r + offset_y],
                            [cx + r + offset_x, cy + r + offset_y]], np.int32)
            cv2.fillPoly(img, [pts], color)
        elif cls == 3:  # star (5-pointed)
            pts = []
            for j in range(10):
                angle = np.pi / 5 * j - np.pi / 2
                rad = r if j % 2 == 0 else r // 2
                pts.append([int(cx + rad * np.cos(angle) + offset_x),
                            int(cy + rad * np.sin(angle) + offset_y)])
            cv2.fillPoly(img, [np.array(pts, np.int32)], color)
        elif cls == 4:  # cross
            cv2.rectangle(img, (cx - 3 + offset_x, cy - r + offset_y),
                          (cx + 3 + offset_x, cy + r + offset_y), color, -1)
            cv2.rectangle(img, (cx - r + offset_x, cy - 3 + offset_y),
                          (cx + r + offset_x, cy + 3 + offset_y), color, -1)
        elif cls == 5:  # hexagon
            pts = []
            for j in range(6):
                angle = np.pi / 3 * j
                pts.append([int(cx + r * np.cos(angle) + offset_x),
                            int(cy + r * np.sin(angle) + offset_y)])
            cv2.fillPoly(img, [np.array(pts, np.int32)], color)
        elif cls == 6:  # diamond
            pts = np.array([[cx, cy - r + offset_y],
                            [cx + r + offset_x, cy + offset_y],
                            [cx + offset_x, cy + r + offset_y],
                            [cx - r + offset_x, cy + offset_y]], np.int32)
            cv2.fillPoly(img, [pts], color)
        elif cls == 7:  # heart (approx)
            cv2.ellipse(img, (cx - r // 3 + offset_x, cy - r // 4 + offset_y),
                        (r // 3, r // 3), 0, 0, 360, color, -1)
            cv2.ellipse(img, (cx + r // 3 + offset_x, cy - r // 4 + offset_y),
                        (r // 3, r // 3), 0, 0, 360, color, -1)
            pts = np.array([[cx - r // 2 + offset_x, cy + offset_y],
                            [cx + r // 2 + offset_x, cy + offset_y],
                            [cx + offset_x, cy + r + offset_y]], np.int32)
            cv2.fillPoly(img, [pts], color)
        elif cls == 8:  # arrow
            cv2.arrowedLine(img, (cx - r + offset_x, cy + offset_y),
                            (cx + r + offset_x, cy + offset_y), color, 2, tipLength=0.3)
        else:  # ellipse
            cv2.ellipse(img, (cx + offset_x, cy + offset_y),
                        (r, r // 2), 0, 0, 360, color, -1)

        # Normalize to [-1, 1].
        img_f = img.astype(np.float32) / 127.5 - 1.0
        for c in range(channels):
            images[i, c] = img_f

    return images, labels


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class ImageDataset(Dataset):
    """Simple Dataset wrapping (images, labels) arrays."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------
def build_dataloaders(
    images: np.ndarray,
    labels: np.ndarray,
    config: DiffusionConfig = DEFAULT_CONFIG,
) -> Tuple["DataLoader", "DataLoader"]:
    """Build train + val DataLoaders."""
    n = len(images)
    n_val = max(1, int(n * config.val_size))
    train_ds = ImageDataset(images[:-n_val], labels[:-n_val])
    val_ds = ImageDataset(images[-n_val:], labels[-n_val:])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_diffusion_dataset(
    dataset_name: str = "synthetic",
    n_samples: int = 2000,
    config: DiffusionConfig = DEFAULT_CONFIG,
    data_dir: Optional[Path | str] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """One-call loader. Returns (images, labels, source).

    Parameters
    ----------
    dataset_name : str
        "synthetic" (default), "mnist", or "cifar10".
    n_samples : int
        Number of synthetic samples (ignored for real datasets).
    config : DiffusionConfig
    data_dir : optional path for real datasets.
    """
    if dataset_name == "synthetic":
        log.info("Generating %d synthetic shape images (%dx%dx%d)...",
                 n_samples, config.channels, config.image_size, config.image_size)
        images, labels = generate_synthetic_shapes(
            n_samples=n_samples, image_size=config.image_size,
            channels=config.channels, num_classes=config.num_classes,
            seed=config.seed,
        )
        return images, labels, "synthetic"

    if dataset_name in ("mnist", "cifar10"):
        try:
            import torchvision
            import torchvision.transforms as T

            transform = T.Compose([
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
                T.Normalize((0.5,) * config.channels, (0.5,) * config.channels),  # [-1, 1]
            ])

            root = str(data_dir or Path(__file__).parent / "data")
            if dataset_name == "mnist":
                ds = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
                config = DiffusionConfig(
                    image_size=config.image_size, channels=1, num_classes=10,
                    batch_size=config.batch_size, val_size=config.val_size, seed=config.seed,
                )
            else:
                ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
                config = DiffusionConfig(
                    image_size=config.image_size, channels=3, num_classes=10,
                    batch_size=config.batch_size, val_size=config.val_size, seed=config.seed,
                )

            images = np.stack([ds[i][0].numpy() for i in range(min(len(ds), n_samples))])
            labels = np.array([ds[i][1] for i in range(min(len(ds), n_samples))])
            return images, labels, dataset_name
        except Exception as exc:
            log.warning("Failed to load %s (%s); falling back to synthetic.", dataset_name, exc)
            images, labels = generate_synthetic_shapes(
                n_samples=n_samples, image_size=config.image_size,
                channels=config.channels, num_classes=config.num_classes, seed=config.seed,
            )
            return images, labels, "synthetic"

    raise ValueError(f"Unknown dataset: {dataset_name}")


__all__ = [
    "DiffusionConfig",
    "DEFAULT_CONFIG",
    "SHAPE_NAMES",
    "ImageDataset",
    "generate_synthetic_shapes",
    "build_dataloaders",
    "load_diffusion_dataset",
]
