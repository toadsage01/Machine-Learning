"""
dataset
=======

Face image ETL pipeline with facial landmark alignment, synthetic face
generator with identity labels, and image pre-processing utilities.

Public surface
--------------
- ``FaceConfig``                    : dataclass with image size + embedding dim.
- ``FaceImage``                     : value object holding image + landmarks + identity.
- ``FaceDataset``                   : bundle of images + identities + provenance.
- ``DEFAULT_LANDMARKS``             : canonical 5-point landmark template (eyes, nose, mouth corners).
- ``compute_affine_transform``       : 2D similarity transform from src→dst landmarks.
- ``align_face``                     : warp a face image to canonical landmark positions.
- ``generate_synthetic_face``       : single synthetic face with identity-specific features.
- ``generate_synthetic_face_dataset`` : batch synthetic face generator.
- ``load_face_dataset``              : one-call loader (synthetic | CSV index | directory).
- ``preprocess_for_arcface``         : normalize + resize image to ArcFace input shape.

Design notes
------------
1. **Landmark alignment is critical** — ArcFace embeddings are only
   comparable if the input faces are aligned to the same canonical pose.
   We use the standard 5-point alignment (left eye, right eye, nose,
   left mouth corner, right mouth corner) with a similarity transform
   (translation + rotation + uniform scale — no skew).

2. **Synthetic face generator for offline testing** — real face datasets
   (LFW, CASIA-WebFace) require auth or are gated. The synthetic
   generator produces identity-specific faces with class-dependent
   features (face shape, eye colour, skin tone, etc.) so the entire
   pipeline (detect → align → embed → index → retrieve) can be tested
   without downloading real faces.

3. **5-point landmarks are the standard** — RetinaFace, MTCNN, and
   MediaPipe all output 5 landmarks. We compute the similarity transform
   that maps these to a canonical template, then apply it via
   ``cv2.warpAffine``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("face_dataset")

try:
    import cv2
    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:  # pragma: no cover
    HAVE_PIL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"

# Standard ArcFace input size.
ARCFACE_IMAGE_SIZE = 112
ARCFACE_EMBEDDING_DIM = 512

# Canonical 5-point landmark template for a 112×112 aligned face.
# Points: left_eye, right_eye, nose, left_mouth, right_mouth.
# These are the standard positions used by InsightFace.
DEFAULT_LANDMARKS: np.ndarray = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Config & value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FaceConfig:
    """Configuration for face recognition."""

    image_size: int = ARCFACE_IMAGE_SIZE
    embedding_dim: int = ARCFACE_EMBEDDING_DIM
    num_landmarks: int = 5


DEFAULT_CONFIG = FaceConfig()


@dataclass
class FaceImage:
    """A single face image with optional landmarks + identity label."""

    image: np.ndarray  # (H, W, C) uint8 or float32
    landmarks: Optional[np.ndarray]  # (5, 2) float32 or None
    identity: str
    identity_idx: int

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.image.shape


@dataclass(frozen=True)
class FaceDataset:
    """Bundle of face images + identities + provenance."""

    images: List[FaceImage]
    identities: List[str]
    identity_to_idx: Dict[str, int]
    source: str
    sha256: str
    n_samples: int = field(init=False)
    n_identities: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_samples", len(self.images))
        object.__setattr__(self, "n_identities", len(self.identities))


# ---------------------------------------------------------------------------
# Landmark alignment
# ---------------------------------------------------------------------------
def compute_affine_transform(
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray = DEFAULT_LANDMARKS,
) -> Tuple[np.ndarray, float]:
    """Compute the optimal 2D similarity transform from src→dst landmarks.

    Uses Umeyama's method (same as scikit-image's ``SimilarityTransform``)
    to find the rotation + translation + uniform scale that minimizes
    the MSE between transformed ``src_landmarks`` and ``dst_landmarks``.

    Parameters
    ----------
    src_landmarks : np.ndarray, shape (5, 2)
        Source landmark positions (detected from the input image).
    dst_landmarks : np.ndarray, shape (5, 2)
        Destination (canonical) landmark positions.

    Returns
    -------
    (transform_matrix, residual)
        ``transform_matrix`` is a 2×3 affine matrix for ``cv2.warpAffine``.
        ``residual`` is the RMS landmark alignment error.
    """
    src = np.asarray(src_landmarks, dtype=np.float64)
    dst = np.asarray(dst_landmarks, dtype=np.float64)
    assert src.shape == dst.shape == (5, 2), (
        f"Expected (5, 2) landmark arrays, got src={src.shape}, dst={dst.shape}"
    )

    # Umeyama algorithm.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Covariance matrix.
    cov = src_centered.T @ dst_centered / len(src)

    # SVD of covariance.
    U, S, Vt = np.linalg.svd(cov)

    # Correct for reflection.
    d = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[1, 1] = -1

    # Rotation matrix.
    R = Vt.T @ d @ U.T

    # Uniform scale.
    src_var = np.sum(src_centered ** 2) / len(src)
    scale = np.trace(np.diag(S) @ d) / src_var if src_var > 0 else 1.0

    # Translation.
    t = dst_mean - scale * R @ src_mean

    # Build 2×3 affine matrix for cv2.warpAffine.
    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:, 2] = t

    # Compute residual (RMS alignment error).
    transformed = (scale * src @ R.T) + t
    residual = float(np.sqrt(np.mean((transformed - dst) ** 2)))

    return M, residual


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = ARCFACE_IMAGE_SIZE,
    dst_landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Align a face image to canonical landmark positions.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, C)
        Input image (uint8 or float32).
    landmarks : np.ndarray, shape (5, 2)
        Detected 5-point facial landmarks.
    output_size : int
        Output image size (square).
    dst_landmarks : np.ndarray, optional
        Canonical landmark positions. Defaults to ``DEFAULT_LANDMARKS``
        scaled to ``output_size``.

    Returns
    -------
    np.ndarray, shape (output_size, output_size, C)
        Aligned face image.
    """
    if not HAVE_CV2:
        raise RuntimeError("opencv is required for align_face.")

    if dst_landmarks is None:
        if output_size == ARCFACE_IMAGE_SIZE:
            dst = DEFAULT_LANDMARKS
        else:
            scale = output_size / ARCFACE_IMAGE_SIZE
            dst = DEFAULT_LANDMARKS * scale
    else:
        dst = np.asarray(dst_landmarks, dtype=np.float32)

    M, _ = compute_affine_transform(landmarks, dst)
    aligned = cv2.warpAffine(
        image, M, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def preprocess_for_arcface(
    image: np.ndarray,
    image_size: int = ARCFACE_IMAGE_SIZE,
) -> np.ndarray:
    """Preprocess a face image for ArcFace inference.

    Steps:
        1. Resize to (image_size, image_size) if needed.
        2. Convert to float32 and normalize to [-1, 1] (standard ArcFace
           preprocessing).
        3. Transpose to CHW format.
        4. Add batch dimension.

    Returns
    -------
    np.ndarray, shape (1, 3, H, W) float32
        Ready for model input.
    """
    if not HAVE_CV2:
        raise RuntimeError("opencv is required for preprocess_for_arcface.")

    # Ensure 3 channels.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Resize if needed.
    if image.shape[0] != image_size or image.shape[1] != image_size:
        image = cv2.resize(image, (image_size, image_size))

    # Normalize to [-1, 1].
    image = image.astype(np.float32) / 127.5 - 1.0

    # HWC → CHW.
    image = np.transpose(image, (2, 0, 1))

    # Add batch dim.
    return np.expand_dims(image, axis=0)


# ---------------------------------------------------------------------------
# Synthetic face generator
# ---------------------------------------------------------------------------
# Identity-specific features: face shape, skin tone, eye colour, hair colour,
# facial hair, etc. Each identity has a deterministic combination.
_FACE_SHAPES = ["oval", "round", "square", "heart"]
_SKIN_TONES = [
    (220, 180, 150),  # light
    (200, 160, 130),  # medium-light
    (170, 130, 100),  # medium
    (140, 100, 80),   # medium-dark
    (100, 70, 50),    # dark
]
_EYE_COLORS = [(50, 50, 50), (80, 50, 30), (50, 80, 100), (100, 80, 50), (60, 100, 60)]
_HAIR_COLORS = [(40, 30, 20), (80, 60, 40), (150, 130, 100), (200, 180, 150), (60, 40, 30)]
_FACIAL_HAIR = ["none", "mustache", "beard", "goatee"]


def generate_synthetic_face(
    identity_idx: int,
    image_size: int = 112,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generate a single synthetic face image with 5-point landmarks.

    The generator is deterministic per ``identity_idx`` — the same identity
    always produces the same facial features. Within an identity, slight
    random variation (expression, lighting) is added per call.

    Returns
    -------
    (image, landmarks, identity_name)
        ``image``: (H, W, 3) uint8 BGR.
        ``landmarks``: (5, 2) float32.
        ``identity_name``: e.g. "person_0".
    """
    if not HAVE_CV2:
        raise RuntimeError("opencv is required for generate_synthetic_face.")

    rng = np.random.default_rng(seed if seed is not None else identity_idx * 1000)

    # Identity-specific features (deterministic per identity_idx).
    shape_idx = identity_idx % len(_FACE_SHAPES)
    skin_tone = _SKIN_TONES[identity_idx % len(_SKIN_TONES)]
    eye_color = _EYE_COLORS[(identity_idx // 2) % len(_EYE_COLORS)]
    hair_color = _HAIR_COLORS[(identity_idx // 3) % len(_HAIR_COLORS)]
    facial_hair = _FACIAL_HAIR[(identity_idx // 4) % len(_FACIAL_HAIR)]

    # Background: neutral grey.
    img = np.full((image_size, image_size, 3), 128, dtype=np.uint8)

    # Face region.
    cx, cy = image_size // 2, image_size // 2 + 5
    face_h, face_w = image_size * 2 // 5, image_size * 2 // 5

    # Draw face shape.
    if _FACE_SHAPES[shape_idx] == "oval":
        cv2.ellipse(img, (cx, cy), (face_w, face_h), 0, 0, 360, skin_tone, -1)
    elif _FACE_SHAPES[shape_idx] == "round":
        cv2.circle(img, (cx, cy), face_w, skin_tone, -1)
    elif _FACE_SHAPES[shape_idx] == "square":
        cv2.rectangle(img, (cx - face_w, cy - face_h), (cx + face_w, cy + face_h),
                       skin_tone, -1)
    else:  # heart
        cv2.ellipse(img, (cx, cy + 5), (face_w, face_h), 0, 0, 360, skin_tone, -1)

    # Hair: arc on top.
    cv2.ellipse(img, (cx, cy - face_h + 5), (face_w + 5, face_h // 2),
                0, 180, 360, hair_color, -1)

    # Eyes.
    eye_y = cy - face_h // 4
    eye_dx = face_w // 3
    eye_r = max(3, face_w // 12)
    cv2.circle(img, (cx - eye_dx, eye_y), eye_r, (255, 255, 255), -1)
    cv2.circle(img, (cx + eye_dx, eye_y), eye_r, (255, 255, 255), -1)
    cv2.circle(img, (cx - eye_dx, eye_y), max(2, eye_r // 2), eye_color, -1)
    cv2.circle(img, (cx + eye_dx, eye_y), max(2, eye_r // 2), eye_color, -1)

    # Nose.
    cv2.ellipse(img, (cx, cy + 5), (max(2, face_w // 10), max(4, face_h // 6)),
                0, 0, 360, tuple(s // 2 for s in skin_tone), -1)

    # Mouth.
    mouth_y = cy + face_h // 3
    mouth_w = max(4, face_w // 4)
    cv2.ellipse(img, (cx, mouth_y), (mouth_w, max(3, face_h // 12)),
                0, 0, 360, (100, 50, 50), -1)

    # Facial hair.
    if facial_hair == "mustache":
        cv2.ellipse(img, (cx, mouth_y - 3), (mouth_w + 3, 3),
                    0, 0, 360, hair_color, -1)
    elif facial_hair == "beard":
        cv2.ellipse(img, (cx, mouth_y + 10), (face_w // 2, face_h // 3),
                    0, 0, 360, hair_color, -1)
    elif facial_hair == "goatee":
        cv2.ellipse(img, (cx, mouth_y + 8), (5, 8),
                    0, 0, 360, hair_color, -1)

    # Add per-call noise (slight expression / lighting variation).
    noise = rng.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Generate landmarks (slightly perturbed canonical positions).
    landmarks = DEFAULT_LANDMARKS.copy()
    # Scale to image_size.
    if image_size != ARCFACE_IMAGE_SIZE:
        scale = image_size / ARCFACE_IMAGE_SIZE
        landmarks = landmarks * scale
    # Add small per-call perturbation.
    landmarks += rng.normal(0, 1.5, landmarks.shape).astype(np.float32)

    identity_name = f"person_{identity_idx}"
    return img, landmarks, identity_name


def generate_synthetic_face_dataset(
    n_identities: int = 10,
    n_images_per_identity: int = 5,
    image_size: int = ARCFACE_IMAGE_SIZE,
    seed: int = 42,
) -> FaceDataset:
    """Generate a synthetic face dataset with identity labels.

    Each identity gets ``n_images_per_identity`` images with slight
    per-image variation (noise, landmark perturbation) but consistent
    facial features.
    """
    images: List[FaceImage] = []
    identities: List[str] = []
    identity_to_idx: Dict[str, int] = {}

    for identity_idx in range(n_identities):
        identity_name = f"person_{identity_idx}"
        identities.append(identity_name)
        identity_to_idx[identity_name] = identity_idx

        for img_idx in range(n_images_per_identity):
            img_seed = seed + identity_idx * 100 + img_idx
            img, landmarks, name = generate_synthetic_face(
                identity_idx, image_size=image_size, seed=img_seed,
            )
            images.append(FaceImage(
                image=img, landmarks=landmarks,
                identity=name, identity_idx=identity_idx,
            ))

    # Build a hash for provenance.
    sha_input = f"synthetic_faces_n{n_identities}_p{n_images_per_identity}_s{seed}".encode()
    sha = hashlib.sha256(sha_input).hexdigest()

    return FaceDataset(
        images=images,
        identities=identities,
        identity_to_idx=identity_to_idx,
        source="synthetic",
        sha256=sha,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_face_dataset(
    data_dir: Optional[Path | str] = None,
    n_identities_synthetic: int = 10,
    n_images_per_identity: int = 5,
    image_size: int = ARCFACE_IMAGE_SIZE,
    seed: int = 42,
) -> FaceDataset:
    """One-call loader for face data.

    Resolution order:
        1. ``data_dir`` (directory with per-identity subdirs of images).
        2. Synthetic generator.
    """
    if data_dir is not None:
        path = Path(data_dir)
        if not path.exists():
            raise FileNotFoundError(f"Face data directory not found: {path}")
        return _load_from_directory(path, image_size)

    log.info("Using synthetic face data (n_identities=%d, n_images_per_identity=%d)",
             n_identities_synthetic, n_images_per_identity)
    return generate_synthetic_face_dataset(
        n_identities=n_identities_synthetic,
        n_images_per_identity=n_images_per_identity,
        image_size=image_size,
        seed=seed,
    )


def _load_from_directory(
    data_dir: Path,
    image_size: int = ARCFACE_IMAGE_SIZE,
) -> FaceDataset:
    """Load faces from a directory structure: ``data_dir/<identity>/image.jpg``."""
    if not HAVE_PIL:
        raise RuntimeError("PIL is required for loading from directory.")

    images: List[FaceImage] = []
    identities: List[str] = []
    identity_to_idx: Dict[str, int] = {}

    for identity_dir in sorted(data_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        identity_name = identity_dir.name
        identity_idx = len(identities)
        identities.append(identity_name)
        identity_to_idx[identity_name] = identity_idx

        for img_path in sorted(identity_dir.glob("*.jpg")) + sorted(identity_dir.glob("*.png")):
            pil_img = Image.open(img_path).convert("RGB")
            img_array = np.array(pil_img)
            # BGR for OpenCV.
            if img_array.ndim == 3 and img_array.shape[2] == 3:
                img_array = img_array[:, :, ::-1].copy()

            # Use canonical landmarks (no detector in the loader).
            landmarks = DEFAULT_LANDMARKS.copy()
            if img_array.shape[0] != image_size:
                scale = image_size / ARCFACE_IMAGE_SIZE
                landmarks = landmarks * scale

            images.append(FaceImage(
                image=img_array, landmarks=landmarks,
                identity=identity_name, identity_idx=identity_idx,
            ))

    sha = hashlib.sha256(str(data_dir).encode()).hexdigest()
    return FaceDataset(
        images=images,
        identities=identities,
        identity_to_idx=identity_to_idx,
        source=str(data_dir),
        sha256=sha,
    )


__all__ = [
    "FaceConfig",
    "FaceImage",
    "FaceDataset",
    "DEFAULT_LANDMARKS",
    "ARCFACE_IMAGE_SIZE",
    "ARCFACE_EMBEDDING_DIM",
    "DEFAULT_CONFIG",
    "compute_affine_transform",
    "align_face",
    "preprocess_for_arcface",
    "generate_synthetic_face",
    "generate_synthetic_face_dataset",
    "load_face_dataset",
]
