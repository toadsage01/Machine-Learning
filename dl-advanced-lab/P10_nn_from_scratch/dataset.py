"""
dataset
=======

MNIST / CIFAR-10 ETL pipeline with synthetic fallback for offline execution.

Public surface
--------------
- ``DatasetConfig``               : dataclass with image size + class count.
- ``ImageDataset``                 : frozen value object bundling
                                      train/test arrays + provenance.
- ``generate_synthetic_mnist``   : synthetic MNIST-like digit generator.
- ``generate_synthetic_cifar``   : synthetic CIFAR-10-like image generator.
- ``load_mnist``                   : one-call MNIST loader (synthetic fallback).
- ``load_cifar10``                 : one-call CIFAR-10 loader (synthetic fallback).
- ``batch_generator``             : yields ``(images, labels)`` batches.

Design notes
------------
1. **Synthetic fallback for offline testing** — MNIST and CIFAR-10
   downloads require network access (and torchvision, which pulls in
   ~500 MB of dependencies). The synthetic generator produces
   realistic-shape images with class-dependent patterns so the entire
   training loop can be smoke-tested without network.

2. **Normalization** — images are scaled to [0, 1] and then mean/std
   normalized. For MNIST we use mean=0.1307, std=0.3081 (the canonical
   MNIST stats). For CIFAR-10 we use ImageNet stats (a reasonable
   approximation for natural images).

3. **Batch generator** — yields ``(images, labels)`` batches of shape
   ``(batch, C, H, W)`` for images and ``(batch,)`` for labels. The
   generator shuffles at the start of each epoch.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("nn_dataset")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"

# Canonical normalization stats.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616])


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for the image dataset."""

    name: str  # "mnist" | "cifar10"
    image_size: int  # 28 for MNIST, 32 for CIFAR-10
    channels: int  # 1 for MNIST, 3 for CIFAR-10
    num_classes: int  # 10 for both
    mean: float  # for normalization (scalar — broadcast per channel)
    std: float


MNIST_CONFIG = DatasetConfig(
    name="mnist", image_size=28, channels=1, num_classes=10,
    mean=MNIST_MEAN, std=MNIST_STD,
)
CIFAR_CONFIG = DatasetConfig(
    name="cifar10", image_size=32, channels=3, num_classes=10,
    mean=0.4734, std=0.2509,  # mean of CIFAR_MEAN / mean of CIFAR_STD (approximate scalar)
)


@dataclass(frozen=True)
class ImageDataset:
    """Bundle of train/test arrays + provenance."""

    config: DatasetConfig
    X_train: np.ndarray  # (N, C, H, W) float32 normalized
    y_train: np.ndarray   # (N,) int64
    X_test: np.ndarray
    y_test: np.ndarray
    source: str  # "synthetic" | "mnist" | "cifar10"
    sha256: str
    n_train: int = field(init=False)
    n_test: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_train", int(len(self.X_train)))
        object.__setattr__(self, "n_test", int(len(self.X_test)))


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------
def generate_synthetic_mnist(
    n_train: int = 1000, n_test: int = 200, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic MNIST-like digit images.

    Each class draws a digit-shaped bright region (a "stroke") on a
    dark background. The strokes are class-dependent (e.g. class 0 has
    a circular stroke, class 1 has a vertical bar).

    Returns
    -------
    (X_train, y_train, X_test, y_test)
        X_train: (n_train, 1, 28, 28) float32, normalized to MNIST stats.
        y_train: (n_train,) int64 labels.
    """
    rng = np.random.default_rng(seed)
    H = W = 28

    def _draw_digit(class_idx: int, img: np.ndarray) -> None:
        """Draw a class-dependent stroke on the 28×28 image."""
        cy, cx = 14, 14
        yy, xx = np.mgrid[:H, :W]
        if class_idx == 0:  # circle
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 < 50
            img[mask] = 1.0
        elif class_idx == 1:  # vertical bar
            img[:, 13:16] = 1.0
        elif class_idx == 2:  # horizontal bar
            img[13:16, :] = 1.0
        elif class_idx == 3:  # diagonal
            np.fill_diagonal(img, 1.0)
        elif class_idx == 4:  # L-shape
            img[:, 13:16] = 1.0
            img[20:23, :] = 1.0
        elif class_idx == 5:  # square
            img[10:18, 10:18] = 1.0
        elif class_idx == 6:  # triangle
            mask = (yy > cy - 5) & (np.abs(xx - cx) < (yy - cy + 5))
            img[mask] = 1.0
        elif class_idx == 7:  # X-shape
            for i in range(H):
                if 5 < i < 23:
                    img[i, i] = 1.0
                    img[i, W - 1 - i] = 1.0
        elif class_idx == 8:  # plus sign
            img[:, 13:16] = 1.0
            img[13:16, :] = 1.0
        elif class_idx == 9:  # diamond
            mask = np.abs(xx - cx) + np.abs(yy - cy) < 6
            img[mask] = 1.0

    def _make_batch(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        X = np.zeros((n, 1, H, W), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            cls = int(rng.integers(0, 10))
            _draw_digit(cls, X[i, 0])
            # Add noise.
            X[i, 0] += rng.normal(0, 0.1, (H, W))
            y[i] = cls
        return X, y

    X_train, y_train = _make_batch(n_train, rng)
    X_test, y_test = _make_batch(n_test, rng)
    return X_train, y_train, X_test, y_test


def generate_synthetic_cifar(
    n_train: int = 1000, n_test: int = 200, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic CIFAR-10-like images.

    Each class has a class-dependent colour + shape combination:
        * Class 0 (airplane):  light-blue + horizontal stripe.
        * Class 1 (car):      red + rectangle in the bottom half.
        * Class 2 (bird):     yellow + small circle top.
        * ...
    """
    rng = np.random.default_rng(seed)
    H = W = 32

    # Class colours (RGB).
    class_colors = [
        (0.6, 0.8, 1.0),  # airplane — light blue
        (1.0, 0.3, 0.3),  # car — red
        (1.0, 0.9, 0.3),  # bird — yellow
        (0.3, 0.7, 0.3),  # cat — green
        (0.5, 0.3, 0.7),  # deer — purple
        (0.7, 0.5, 0.3),  # dog — brown
        (0.3, 0.3, 0.3),  # frog — dark grey
        (1.0, 0.7, 0.3),  # horse — orange
        (0.9, 0.9, 0.9),  # ship — white
        (0.3, 0.3, 0.9),  # truck — blue
    ]

    def _draw_class(class_idx: int, img: np.ndarray) -> None:
        """Draw class-dependent pattern on (3, H, W) image."""
        color = class_colors[class_idx]
        # Background: a slightly noisy version of the class colour.
        for c in range(3):
            img[c] = color[c] * 0.3 + rng.normal(0, 0.05, (H, W))
        # Foreground shape.
        cy, cx = H // 2, W // 2
        yy, xx = np.mgrid[:H, :W]
        if class_idx == 0:  # airplane — horizontal stripe
            mask = (yy > 14) & (yy < 18)
            for c in range(3):
                img[c][mask] = color[c]
        elif class_idx == 1:  # car — rectangle bottom
            mask = (yy > 20) & (xx > 8) & (xx < 24)
            for c in range(3):
                img[c][mask] = color[c]
        elif class_idx == 2:  # bird — circle top
            mask = (yy - 8) ** 2 + (xx - 16) ** 2 < 25
            for c in range(3):
                img[c][mask] = color[c]
        elif class_idx == 3:  # cat — square center
            mask = (yy > 10) & (yy < 22) & (xx > 10) & (xx < 22)
            for c in range(3):
                img[c][mask] = color[c]
        elif class_idx == 4:  # deer — vertical stripes
            mask = (xx > 14) & (xx < 18)
            for c in range(3):
                img[c][mask] = color[c]
        elif class_idx == 5:  # dog — circle + square
            mask1 = (yy - 14) ** 2 + (xx - 16) ** 2 < 36
            for c in range(3):
                img[c][mask1] = color[c]
        elif class_idx == 6:  # frog — wavy line
            mask = np.abs(yy - 16 - np.sin(xx * 0.5) * 4) < 2
            for c in range(3):
                img[c][mask] = color[c]
        elif class_idx == 7:  # horse — diagonal
            mask = np.abs(xx - yy) < 3
            for c in range(3):
                img[c][mask] = color[c]
        elif class_idx == 8:  # ship — trapezoid bottom
            mask = (yy > 20) & (np.abs(xx - 16) < (24 - yy))
            for c in range(3):
                img[c][mask] = color[c]
        elif class_idx == 9:  # truck — rectangle
            mask = (yy > 12) & (yy < 20) & (xx > 8) & (xx < 24)
            for c in range(3):
                img[c][mask] = color[c]

    def _make_batch(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        X = np.zeros((n, 3, H, W), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            cls = int(rng.integers(0, 10))
            _draw_class(cls, X[i])
            y[i] = cls
        return X, y

    X_train, y_train = _make_batch(n_train, rng)
    X_test, y_test = _make_batch(n_test, rng)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Normalize images: (X - mean) / std (per channel)."""
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _sha256(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


def load_mnist(
    n_train_synthetic: int = 1000,
    n_test_synthetic: int = 200,
    seed: int = 42,
    use_real: bool = False,
) -> ImageDataset:
    """Load MNIST data (with synthetic fallback)."""
    if use_real:
        try:
            import torchvision
            import torchvision.transforms as T
            transform = T.Compose([T.ToTensor(), T.Normalize((MNIST_MEAN,), (MNIST_STD,))])
            train_set = torchvision.datasets.MNIST(
                root=str(PROJECT_DATA_DIR), train=True, download=True, transform=transform,
            )
            test_set = torchvision.datasets.MNIST(
                root=str(PROJECT_DATA_DIR), train=False, download=True, transform=transform,
            )
            X_train = np.stack([train_set[i][0].numpy() for i in range(len(train_set))])
            y_train = np.array([train_set[i][1] for i in range(len(train_set))])
            X_test = np.stack([test_set[i][0].numpy() for i in range(len(test_set))])
            y_test = np.array([test_set[i][1] for i in range(len(test_set))])
            source = "mnist"
            sha = _sha256(X_train.tobytes() + X_test.tobytes())
        except Exception as exc:
            log.warning("Real MNIST download failed (%s); using synthetic.", exc)
            X_train, y_train, X_test, y_test = generate_synthetic_mnist(
                n_train_synthetic, n_test_synthetic, seed,
            )
            X_train = normalize(X_train, MNIST_MEAN, MNIST_STD)
            X_test = normalize(X_test, MNIST_MEAN, MNIST_STD)
            source = "synthetic"
            sha = _sha256(X_train.tobytes() + X_test.tobytes())
    else:
        log.info("Using synthetic MNIST (n_train=%d, n_test=%d)", n_train_synthetic, n_test_synthetic)
        X_train, y_train, X_test, y_test = generate_synthetic_mnist(
            n_train_synthetic, n_test_synthetic, seed,
        )
        X_train = normalize(X_train, MNIST_MEAN, MNIST_STD)
        X_test = normalize(X_test, MNIST_MEAN, MNIST_STD)
        source = "synthetic"
        sha = _sha256(X_train.tobytes() + X_test.tobytes())

    return ImageDataset(
        config=MNIST_CONFIG,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        source=source, sha256=sha,
    )


def load_cifar10(
    n_train_synthetic: int = 1000,
    n_test_synthetic: int = 200,
    seed: int = 42,
    use_real: bool = False,
) -> ImageDataset:
    """Load CIFAR-10 data (with synthetic fallback)."""
    if use_real:
        try:
            import torchvision
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(tuple(CIFAR_MEAN.tolist()), tuple(CIFAR_STD.tolist())),
            ])
            train_set = torchvision.datasets.CIFAR10(
                root=str(PROJECT_DATA_DIR), train=True, download=True, transform=transform,
            )
            test_set = torchvision.datasets.CIFAR10(
                root=str(PROJECT_DATA_DIR), train=False, download=True, transform=transform,
            )
            X_train = np.stack([train_set[i][0].numpy() for i in range(len(train_set))])
            y_train = np.array([train_set[i][1] for i in range(len(train_set))])
            X_test = np.stack([test_set[i][0].numpy() for i in range(len(test_set))])
            y_test = np.array([test_set[i][1] for i in range(len(test_set))])
            source = "cifar10"
            sha = _sha256(X_train.tobytes() + X_test.tobytes())
        except Exception as exc:
            log.warning("Real CIFAR-10 download failed (%s); using synthetic.", exc)
            X_train, y_train, X_test, y_test = generate_synthetic_cifar(
                n_train_synthetic, n_test_synthetic, seed,
            )
            # Use scalar mean/std (broadcast per channel).
            X_train = normalize(X_train, CIFAR_CONFIG.mean, CIFAR_CONFIG.std)
            X_test = normalize(X_test, CIFAR_CONFIG.mean, CIFAR_CONFIG.std)
            source = "synthetic"
            sha = _sha256(X_train.tobytes() + X_test.tobytes())
    else:
        log.info("Using synthetic CIFAR-10 (n_train=%d, n_test=%d)", n_train_synthetic, n_test_synthetic)
        X_train, y_train, X_test, y_test = generate_synthetic_cifar(
            n_train_synthetic, n_test_synthetic, seed,
        )
        X_train = normalize(X_train, CIFAR_CONFIG.mean, CIFAR_CONFIG.std)
        X_test = normalize(X_test, CIFAR_CONFIG.mean, CIFAR_CONFIG.std)
        source = "synthetic"
        sha = _sha256(X_train.tobytes() + X_test.tobytes())

    return ImageDataset(
        config=CIFAR_CONFIG,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        source=source, sha256=sha,
    )


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------
def batch_generator(
    X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True,
    seed: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(images, labels)`` batches.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, H, W)
    y : np.ndarray, shape (N,)
    batch_size : int
    shuffle : bool
        If True, shuffle the data at the start of each epoch.
    seed : int, optional
        For reproducibility.
    """
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


__all__ = [
    "DatasetConfig",
    "ImageDataset",
    "MNIST_CONFIG",
    "CIFAR_CONFIG",
    "MNIST_MEAN", "MNIST_STD",
    "CIFAR_MEAN", "CIFAR_STD",
    "generate_synthetic_mnist",
    "generate_synthetic_cifar",
    "normalize",
    "load_mnist",
    "load_cifar10",
    "batch_generator",
]
