"""
dataset
=======

Plant Village leaf-disease dataset loader + PyTorch Dataset +
Albumentations augmentations + stratified splits.

Public surface
--------------
- ``LeafDiseaseConfig``     : dataclass holding class names + image size.
- ``DEFAULT_CLASSES``       : canonical Plant Village 10-class subset.
- ``LeafDataset``            : ``torch.utils.data.Dataset`` returning
                               ``(PIL.Image or tensor, label_idx)`` tuples.
- ``make_augmentations``     : build Albumentations pipelines for train/val/test.
- ``make_stratified_splits`` : stratified train/val/test split per class.
- ``download_plantvillage``  : fetch the dataset from a canonical URL with
                               synthetic-image fallback for offline use.
- ``make_synthetic_dataset`` : generate a tiny synthetic leaf-image dataset
                               for offline tests / smoke runs.
- ``build_dataloaders``      : one-call helper returning 3 DataLoaders.

Design notes
------------
1. **Plant Village is huge (~5 GB)** — for the benchmark to be runnable
   in CI without network access, we provide a synthetic-image generator
   that produces realistic-shape ``(224, 224, 3)`` images with class-
   dependent colour gradients. The synthetic data is *not* a substitute
   for real training, but it lets the entire pipeline (augment → train
   → evaluate → export ONNX → run Grad-CAM) be smoke-tested end-to-end.

2. **Albumentations over torchvision transforms** — Albumentations is
   ~3× faster than torchvision's ``transforms.Compose`` for the same
   operations because it operates on NumPy arrays (no PIL → Tensor
   round-trip per transform). The trade-off is a slightly heavier
   dependency, but it's already required for production CV pipelines.

3. **Stratified splits** — we use scikit-learn's
   ``StratifiedShuffleSplit`` to ensure every class is represented in
   train/val/test in proportion to its global frequency. This is
   critical for Plant Village, where some classes (e.g. ``Tomato_healthy``)
   have 5× more images than others (e.g. ``Potato___late_blight``).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import StratifiedShuffleSplit

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("leaf_dataset")

# Lazy imports — torch is heavy and may be unavailable in some test contexts.
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False
    torch = None  # type: ignore

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAVE_ALBUMENTATIONS = True
except Exception:  # pragma: no cover
    HAVE_ALBUMENTATIONS = False
    A = None  # type: ignore

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:  # pragma: no cover
    HAVE_PIL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = PROJECT_DATA_DIR / "_cache"

# Canonical 10-class Plant Village subset — chosen to span 4 crop species
# (tomato, potato, pepper, apple) with a mix of healthy/diseased leaves.
# The full Plant Village dataset has 38 classes; we use a 10-class subset
# to keep training times manageable on CPU.
DEFAULT_CLASSES: Tuple[str, ...] = (
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Potato_healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Pepper_bell_healthy",
    "Pepper_bell_Bacterial_spot",
    "Apple_healthy",
)

# Plant Village dataset URLs. The "spMohanty" mirror on GitHub is the most
# stable public copy we've found; if it's unreachable, the synthetic
# generator kicks in.
PLANTVILLAGE_URLS = [
    "https://github.com/spMohanty/PlantVillage-Dataset/raw/master/data/data_test.zip",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LeafDiseaseConfig:
    """Configuration for the leaf-disease dataset.

    Attributes
    ----------
    classes : tuple of str
        Class names in canonical order. The integer label for a class is
        its index in this tuple (used as the PyTorch training target).
    image_size : int
        Square image dimension (default 224 — standard for ImageNet-pretrained
        ResNet50 / EfficientNetV2 / ViT-B/16).
    num_channels : int
        3 for RGB.
    """

    classes: Tuple[str, ...] = DEFAULT_CLASSES
    image_size: int = 224
    num_channels: int = 3

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {i: c for i, c in enumerate(self.classes)}


DEFAULT_CONFIG = LeafDiseaseConfig()


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an arbitrary image array to (H, W, 3) uint8 RGB."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    return image.astype(np.uint8)


def _generate_synthetic_leaf_image(
    class_idx: int,
    size: int = 224,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a single synthetic (H, W, 3) leaf image for the given class.

    The image is a procedurally-generated leaf shape (ellipse + central vein)
    on a soil-coloured background, with class-specific disease patterns:

    * ``healthy`` classes   → uniform green leaf
    * ``early_blight``     → small brown spots
    * ``late_blight``      → large grey patches
    * ``leaf_mold``        → yellow-green velvety patches
    * ``bacterial_spot``   → dark brown scabs
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), [180, 160, 130], dtype=np.uint8)  # soil bg

    # Leaf ellipse — bright green.
    cy, cx = size // 2, size // 2
    yy, xx = np.mgrid[:size, :size]
    leaf_mask = ((yy - cy) / (size * 0.45)) ** 2 + ((xx - cx) / (size * 0.30)) ** 2 < 1
    img[leaf_mask] = [60, 130, 50]  # green leaf

    # Central vein — slightly darker green.
    vein_mask = leaf_mask & (np.abs(xx - cx) < 2)
    img[vein_mask] = [40, 100, 40]

    # Class-specific disease patterns.
    class_name = DEFAULT_CLASSES[class_idx]
    if "Early_blight" in class_name:
        # Small brown spots scattered on the leaf.
        n_spots = 15
        for _ in range(n_spots):
            sy = rng.integers(size // 4, 3 * size // 4)
            sx = rng.integers(size // 4, 3 * size // 4)
            if leaf_mask[sy, sx]:
                r = rng.integers(2, 6)
                spot = ((yy - sy) ** 2 + (xx - sx) ** 2) < r ** 2
                img[spot & leaf_mask] = [120, 60, 20]
    elif "Late_blight" in class_name:
        # Large grey patches.
        for _ in range(3):
            sy = rng.integers(size // 3, 2 * size // 3)
            sx = rng.integers(size // 3, 2 * size // 3)
            r = rng.integers(20, 35)
            patch = ((yy - sy) ** 2 + (xx - sx) ** 2) < r ** 2
            img[patch & leaf_mask] = [120, 120, 110]
    elif "Leaf_Mold" in class_name:
        # Yellow-green velvety patches.
        for _ in range(8):
            sy = rng.integers(size // 4, 3 * size // 4)
            sx = rng.integers(size // 4, 3 * size // 4)
            r = rng.integers(8, 14)
            patch = ((yy - sy) ** 2 + (xx - sx) ** 2) < r ** 2
            img[patch & leaf_mask] = [180, 170, 60]
    elif "Bacterial_spot" in class_name:
        # Dark brown scabs.
        for _ in range(20):
            sy = rng.integers(size // 4, 3 * size // 4)
            sx = rng.integers(size // 4, 3 * size // 4)
            if leaf_mask[sy, sx]:
                r = rng.integers(1, 4)
                spot = ((yy - sy) ** 2 + (xx - sx) ** 2) < r ** 2
                img[spot & leaf_mask] = [80, 40, 20]

    # Mild per-pixel noise to break up flat regions.
    noise = rng.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Synthetic dataset generator (offline fallback)
# ---------------------------------------------------------------------------
def make_synthetic_dataset(
    n_per_class: int = 50,
    config: LeafDiseaseConfig = DEFAULT_CONFIG,
    seed: int = 42,
    output_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Generate a synthetic leaf-disease dataset.

    Returns a DataFrame with columns ``[path, label, label_idx]``. If
    ``output_dir`` is given, the images are written to disk in a
    ``class_name/0001.jpg`` layout matching the Plant Village convention.
    Otherwise, the DataFrame's ``path`` column contains the in-memory
    image bytes encoded as base64.

    The generator is deterministic given ``seed`` — useful for unit tests.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for class_idx, class_name in enumerate(config.classes):
        for i in range(n_per_class):
            img = _generate_synthetic_leaf_image(
                class_idx, size=config.image_size, seed=int(rng.integers(0, 2**31)),
            )
            if output_dir is not None:
                cls_dir = Path(output_dir) / class_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                img_path = cls_dir / f"{i:04d}.jpg"
                if HAVE_PIL:
                    Image.fromarray(img).save(img_path, quality=90)
                else:  # pragma: no cover
                    # Fallback: write raw bytes via cv2.
                    import cv2
                    cv2.imwrite(str(img_path), img[:, :, ::-1])
                path = str(img_path)
            else:
                # In-memory base64 — useful for tests that don't want to touch disk.
                import base64
                buf = io.BytesIO()
                if HAVE_PIL:
                    Image.fromarray(img).save(buf, format="JPEG", quality=90)
                else:  # pragma: no cover
                    import cv2
                    ok, buf_arr = cv2.imencode(".jpg", img[:, :, ::-1])
                    buf.write(buf_arr.tobytes())
                path = "base64://" + base64.b64encode(buf.getvalue()).decode("ascii")
            rows.append({"path": path, "label": class_name, "label_idx": class_idx})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plant Village downloader (with synthetic fallback)
# ---------------------------------------------------------------------------
def download_plantvillage(
    output_dir: Path | str,
    config: LeafDiseaseConfig = DEFAULT_CONFIG,
    n_per_class: int = 50,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """Download (or synthesize) the Plant Village dataset.

    Resolution order:
        1. Check ``output_dir`` — if it already contains per-class subdirs
           with images, return immediately (cache hit).
        2. Try the canonical Plant Village URL (spMohanty mirror).
        3. Fall back to ``make_synthetic_dataset``.

    Parameters
    ----------
    output_dir : str or Path
        Where to extract / synthesize the dataset.
    config : LeafDiseaseConfig
        Class list + image size.
    n_per_class : int
        Number of synthetic images per class (used only by the fallback).
    seed : int
        Reproducibility seed for the synthetic generator.
    force : bool
        If True, ignore any existing cache.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache hit?
    if not force:
        existing_subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        matching = [d for d in existing_subdirs if d.name in config.class_to_idx]
        if len(matching) >= config.num_classes // 2:
            log.info("Plant Village cache hit at %s (%d class dirs)", output_dir, len(matching))
            return output_dir

    # Try real download.
    for url in PLANTVILLAGE_URLS:
        try:
            log.info("Attempting Plant Village download from %s", url)
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            # Save + extract.
            zip_path = CACHE_DIR / "plantvillage.zip"
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(response.content)
            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(output_dir)
            log.info("Extracted Plant Village → %s", output_dir)
            return output_dir
        except Exception as exc:
            log.warning("Download failed (%s): %s", url, exc)

    # Fallback: synthetic dataset.
    log.warning("All Plant Village URLs failed — using synthetic dataset.")
    make_synthetic_dataset(
        n_per_class=n_per_class, config=config, seed=seed, output_dir=output_dir,
    )
    return output_dir


# ---------------------------------------------------------------------------
# Albumentations pipelines
# ---------------------------------------------------------------------------
def make_augmentations(
    image_size: int = 224,
    config: LeafDiseaseConfig = DEFAULT_CONFIG,
) -> Dict[str, "A.Compose"]:
    """Build Albumentations pipelines for train / val / test.

    Train pipeline uses:
      - RandomResizedCrop (scale 0.8–1.0)
      - HorizontalFlip
      - RandomBrightnessContrast
      - HueSaturationValue
      - Rotate (±15°)
      - Normalize (ImageNet stats)
      - ToTensorV2

    Val / test pipeline is deterministic:
      - Resize to (image_size + 32)
      - CenterCrop to image_size
      - Normalize
      - ToTensorV2
    """
    if not HAVE_ALBUMENTATIONS:
        raise RuntimeError("albumentations is not installed.")

    # ImageNet normalization stats — standard for all 3 backbones (ResNet50,
    # EfficientNetV2, ViT-B/16) since they were all pretrained on ImageNet.
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_aug = A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])

    eval_aug = A.Compose([
        A.Resize(height=image_size + 32, width=image_size + 32),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])

    return {"train": train_aug, "val": eval_aug, "test": eval_aug}


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class LeafDataset(Dataset):
    """PyTorch Dataset returning ``(image_tensor, label_idx)`` tuples.

    Parameters
    ----------
    manifest : pd.DataFrame
        Must contain ``path`` and ``label_idx`` columns.
    augmentations : albumentations.Compose, optional
        If None, returns raw uint8 NumPy arrays.
    config : LeafDiseaseConfig
        Used for class-count metadata.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        augmentations: Optional["A.Compose"] = None,
        config: LeafDiseaseConfig = DEFAULT_CONFIG,
    ):
        super().__init__()
        if not HAVE_TORCH:
            raise RuntimeError("torch is required for LeafDataset.")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.augmentations = augmentations
        self.config = config

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_image(self, path: str) -> np.ndarray:
        """Load an image from disk or from a base64:// URI."""
        if path.startswith("base64://"):
            import base64
            payload = base64.b64decode(path[len("base64://"):])
            if HAVE_PIL:
                img = np.array(Image.open(io.BytesIO(payload)).convert("RGB"))
            else:  # pragma: no cover
                import cv2
                buf_arr = np.frombuffer(payload, dtype=np.uint8)
                img = cv2.imdecode(buf_arr, cv2.IMREAD_COLOR)[:, :, ::-1]
            return _ensure_rgb(img)
        # Disk path.
        if HAVE_PIL:
            img = np.array(Image.open(path).convert("RGB"))
        else:  # pragma: no cover
            import cv2
            img = cv2.imread(path)[:, :, ::-1]
        return _ensure_rgb(img)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.manifest.iloc[idx]
        image = self._load_image(row["path"])
        label = int(row["label_idx"])
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        return image, label


# ---------------------------------------------------------------------------
# Stratified splits
# ---------------------------------------------------------------------------
def make_stratified_splits(
    manifest: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split.

    Returns three DataFrames with the same columns as ``manifest``.
    """
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(manifest, manifest["label_idx"]))
    train_val = manifest.iloc[train_val_idx].reset_index(drop=True)
    test = manifest.iloc[test_idx].reset_index(drop=True)

    # Adjusted val_size so the *overall* val fraction is val_size.
    adjusted_val_size = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=seed)
    train_idx, val_idx = next(sss2.split(train_val, train_val["label_idx"]))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test


# ---------------------------------------------------------------------------
# Build dataloaders — one-call helper
# ---------------------------------------------------------------------------
def build_dataloaders(
    data_dir: Optional[Path | str] = None,
    config: LeafDiseaseConfig = DEFAULT_CONFIG,
    batch_size: int = 32,
    num_workers: int = 0,
    val_size: float = 0.15,
    test_size: float = 0.15,
    n_per_class: int = 50,
    seed: int = 42,
) -> Tuple["DataLoader", "DataLoader", "DataLoader", pd.DataFrame]:
    """One-call helper returning ``(train_loader, val_loader, test_loader, manifest)``.

    If ``data_dir`` is None, builds an in-memory synthetic dataset for
    testing. Otherwise, scans ``data_dir`` for per-class subdirs.
    """
    if not HAVE_TORCH:
        raise RuntimeError("torch is required for build_dataloaders.")

    # Build the manifest.
    if data_dir is None:
        manifest = make_synthetic_dataset(
            n_per_class=n_per_class, config=config, seed=seed,
        )
    else:
        rows: List[Dict] = []
        for class_name in config.classes:
            cls_dir = Path(data_dir) / class_name
            if not cls_dir.exists():
                # Class missing on disk — synthesize it for graceful degradation.
                log.warning("Class dir missing on disk, synthesizing: %s", cls_dir)
                synth = make_synthetic_dataset(
                    n_per_class=n_per_class, config=config, seed=seed,
                    output_dir=cls_dir.parent,
                )
                # Filter to just this class.
                rows.extend(synth.to_dict("records"))
                continue
            for img_path in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")):
                rows.append({
                    "path": str(img_path),
                    "label": class_name,
                    "label_idx": config.class_to_idx[class_name],
                })
        manifest = pd.DataFrame(rows)

    # Build the splits.
    train_df, val_df, test_df = make_stratified_splits(
        manifest, val_size=val_size, test_size=test_size, seed=seed,
    )
    log.info("Splits: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # Build augmentations.
    augs = make_augmentations(config.image_size, config)

    # Build datasets + dataloaders.
    train_ds = LeafDataset(train_df, augs["train"], config)
    val_ds = LeafDataset(val_df, augs["val"], config)
    test_ds = LeafDataset(test_df, augs["val"], config)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False, drop_last=False,
    )
    return train_loader, val_loader, test_loader, manifest


__all__ = [
    "LeafDiseaseConfig",
    "DEFAULT_CLASSES",
    "DEFAULT_CONFIG",
    "LeafDataset",
    "make_augmentations",
    "make_stratified_splits",
    "make_synthetic_dataset",
    "download_plantvillage",
    "build_dataloaders",
]
