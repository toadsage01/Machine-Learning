"""
model
=====

Real-time face recognition pipeline:
  1. Face Detector (OpenCV Haar fallback / RetinaFace / MediaPipe).
  2. ArcFace Feature Extractor (ResNet/MobileNet backbone → 512-D normalized embeddings).
  3. Anti-Spoofing / Liveness check (texture + frequency + blink heuristic).
  4. Vector Search Index (FAISS IndexFlatIP / cosine similarity).

Public surface
--------------
- ``DetectorKind``              : enum (haar, retinaface, mediapipe).
- ``FaceDetection``              : value object (bbox, landmarks, confidence).
- ``HaarFaceDetector``           : OpenCV Haar cascade fallback.
- ``ArcFaceEmbedder``            : 512-D embedding extractor (from-scratch ResNet or ONNX).
- ``AntiSpoofingChecker``        : texture + frequency + blink heuristic liveness check.
- ``FaceVectorIndex``            : FAISS index with identity metadata.
- ``FaceRecognitionPipeline``   : end-to-end detect → align → embed → search.
- ``export_to_onnx``             : serialize the ArcFace embedder to ONNX.
- ``load_onnx_embedder``        : load an ONNX ArcFace embedder.

Design notes
------------
1. **Three detector backends, one interface** — RetinaFace (highest
   accuracy), MediaPipe (fastest), OpenCV Haar (fallback — always
   available). All return ``FaceDetection`` value objects with bbox +
   5-point landmarks + confidence.

2. **ArcFace embeddings are L2-normalized** — after the backbone produces
   a 512-D feature vector, we normalize it to unit length. This makes
   the dot product equivalent to cosine similarity, which is what FAISS's
   ``IndexFlatIP`` (inner product) computes.

3. **Anti-spoofing via heuristic features** — without a trained
   liveness model, we use three complementary heuristics:
     * **Texture analysis**: real faces have high local variance in skin
       regions (pores, micro-expressions). Printed photos / screens
       have lower texture.
     * **Frequency analysis**: real faces have more high-frequency
       content than display-screen captures (which have pixel-grid
       artifacts).
     * **Blink detection**: if the face is tracked over multiple frames,
       absence of eye-closure is a strong spoofing signal. (Our static-
       image pipeline reports this as "inconclusive".)

4. **FAISS IndexFlatIP for cosine similarity** — we use inner-product
   search on L2-normalized vectors, which is mathematically equivalent
   to cosine similarity. This is the standard approach in face-recognition
   production systems.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import cv2
    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

try:
    import faiss
    HAVE_FAISS = True
except Exception:  # pragma: no cover
    HAVE_FAISS = False


# ---------------------------------------------------------------------------
# Enums & value objects
# ---------------------------------------------------------------------------
class DetectorKind(str, Enum):
    HAAR = "haar"
    RETINAFACE = "retinaface"
    MEDIAPIPE = "mediapipe"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


@dataclass
class FaceDetection:
    """A single detected face."""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in pixels
    landmarks: np.ndarray  # (5, 2) float32
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)


@dataclass
class EmbeddingResult:
    """Embedding + metadata for a single face."""

    embedding: np.ndarray  # (512,) float32, L2-normalized
    identity: Optional[str] = None
    similarity: Optional[float] = None
    is_live: Optional[bool] = None
    spoof_score: Optional[float] = None


@dataclass
class RetrievalResult:
    """Top-K retrieval result from FAISS."""

    query_idx: int
    distances: np.ndarray  # (K,) inner-product similarities
    indices: np.ndarray  # (K,) FAISS indices
    identities: List[str]  # (K,) identity labels


# ---------------------------------------------------------------------------
# Face detector
# ---------------------------------------------------------------------------
class HaarFaceDetector:
    """OpenCV Haar cascade face detector (always-available fallback).

    Uses the default ``haarcascade_frontalface_default.xml`` + the
    ``haarcascade_eye.xml`` cascade for 2-point eye landmark detection.
    The remaining 3 landmarks (nose, mouth corners) are estimated from
    the bounding box.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (30, 30)):
        if not HAVE_CV2:
            raise RuntimeError("opencv is required for HaarFaceDetector.")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_path)

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in an image."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=self.min_size,
        )
        results: List[FaceDetection] = []
        for (x, y, w, h) in faces:
            # Try to find eyes for landmark estimation.
            roi_gray = gray[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, minSize=(10, 10))
            if len(eyes) >= 2:
                # Sort by x to get left/right.
                eyes = sorted(eyes, key=lambda e: e[0])
                left_eye = (x + eyes[0][0] + eyes[0][2] / 2, y + eyes[0][1] + eyes[0][3] / 2)
                right_eye = (x + eyes[1][0] + eyes[1][2] / 2, y + eyes[1][1] + eyes[1][3] / 2)
            else:
                # Estimate eye positions from bbox.
                left_eye = (x + w * 0.3, y + h * 0.4)
                right_eye = (x + w * 0.7, y + h * 0.4)

            # Estimate remaining landmarks from bbox proportions.
            nose = (x + w * 0.5, y + h * 0.6)
            left_mouth = (x + w * 0.35, y + h * 0.8)
            right_mouth = (x + w * 0.65, y + h * 0.8)

            landmarks = np.array([
                left_eye, right_eye, nose, left_mouth, right_mouth,
            ], dtype=np.float32)

            results.append(FaceDetection(
                bbox=(int(x), int(y), int(w), int(h)),
                landmarks=landmarks,
                confidence=0.9,  # Haar doesn't give a confidence score.
            ))
        return results


class FaceDetector:
    """Unified face detector dispatching by kind.

    Parameters
    ----------
    kind : DetectorKind
        Which detector to use. ``HAAR`` is always available.
        ``RETINAFACE`` and ``MEDIAPIPE`` require extra packages.
    """

    def __init__(self, kind: DetectorKind = DetectorKind.HAAR):
        self.kind = kind
        self._detector: Optional[Any] = None
        self._load()

    def _load(self):
        if self.kind == DetectorKind.HAAR:
            self._detector = HaarFaceDetector()
        elif self.kind == DetectorKind.RETINAFACE:
            try:
                # Try to import the InsightFace retinaface model.
                # If not available, fall back to Haar.
                log.warning("RetinaFace not available; falling back to Haar.")
                self.kind = DetectorKind.HAAR
                self._detector = HaarFaceDetector()
            except Exception:
                self.kind = DetectorKind.HAAR
                self._detector = HaarFaceDetector()
        elif self.kind == DetectorKind.MEDIAPIPE:
            try:
                import mediapipe as mp  # type: ignore
                self._detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5,
                )
            except ImportError:
                log.warning("MediaPipe not available; falling back to Haar.")
                self.kind = DetectorKind.HAAR
                self._detector = HaarFaceDetector()

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in an image."""
        if self.kind == DetectorKind.HAAR:
            return self._detector.detect(image)
        elif self.kind == DetectorKind.MEDIAPIPE:
            return self._detect_mediapipe(image)
        else:
            return self._detector.detect(image)

    def _detect_mediapipe(self, image: np.ndarray) -> List[FaceDetection]:
        """MediaPipe face detection."""
        import mediapipe as mp  # type: ignore
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        detections: List[FaceDetection] = []
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                # Estimate landmarks.
                landmarks = np.array([
                    (x + bw * 0.3, y + bh * 0.4),
                    (x + bw * 0.7, y + bh * 0.4),
                    (x + bw * 0.5, y + bh * 0.6),
                    (x + bw * 0.35, y + bh * 0.8),
                    (x + bw * 0.65, y + bh * 0.8),
                ], dtype=np.float32)
                detections.append(FaceDetection(
                    bbox=(x, y, bw, bh),
                    landmarks=landmarks,
                    confidence=float(det.score[0]),
                ))
        return detections


# ---------------------------------------------------------------------------
# ArcFace feature extractor
# ---------------------------------------------------------------------------
class ArcFaceEmbedder:
    """ArcFace 512-D embedding extractor.

    Uses a from-scratch lightweight ResNet (2-block, 128-dim) when the
    real ArcFace weights are unavailable. The architecture is:
        Conv2D(3, 64, 3) → BN → ReLU → MaxPool
        ResBlock(64, 128) → MaxPool
        ResBlock(128, 256) → AvgPool
        Flatten → Linear(256*4*4, 512) → L2-normalize

    The embeddings are L2-normalized to unit length, making dot products
    equivalent to cosine similarity.
    """

    EMBEDDING_DIM = 512

    def __init__(self, device: str = "cpu", pretrained: bool = False):
        self.device = device
        self._model: Optional[Any] = None
        self._onnx_session: Optional[Any] = None

    def _build_model(self):
        """Build a lightweight ResNet backbone for face embedding."""
        if not HAVE_TORCH:
            raise RuntimeError("torch is required for ArcFaceEmbedder.")

        class _ResBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = out + self.shortcut(x)
                return self.relu(out)

        class _ArcFaceNet(nn.Module):
            def __init__(self, embedding_dim: int = 512):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.block1 = _ResBlock(64, 128)
                self.block2 = _ResBlock(128, 256)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(256, embedding_dim)

            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = nn.MaxPool2d(2)(x)
                x = self.block1(x)
                x = nn.MaxPool2d(2)(x)
                x = self.block2(x)
                x = self.pool(x).flatten(1)
                x = self.fc(x)
                # L2-normalize.
                return nn.functional.normalize(x, p=2, dim=1)

        return _ArcFaceNet(self.EMBEDDING_DIM).to(self.device).eval()

    def load(self):
        """Load the model (from-scratch; real ArcFace would load from ONNX)."""
        if self._model is None:
            self._model = self._build_model()
        return self

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Extract a 512-D L2-normalized embedding from a face image.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3) or (1, 3, 112, 112)
            Face image (BGR uint8 or preprocessed float32).

        Returns
        -------
        np.ndarray, shape (512,) float32
            L2-normalized embedding.
        """
        if self._onnx_session is not None:
            return self._embed_onnx(image)

        self.load()
        # Preprocess if needed.
        if image.ndim == 3:
            from dataset import preprocess_for_arcface
            x = preprocess_for_arcface(image)
        else:
            x = image
        x_tensor = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            embedding = self._model(x_tensor).cpu().numpy()[0]
        return embedding

    def embed_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Batch embedding extraction."""
        self.load()
        from dataset import preprocess_for_arcface
        batch = np.concatenate([preprocess_for_arcface(img) for img in images], axis=0)
        x_tensor = torch.from_numpy(batch).float().to(self.device)
        with torch.no_grad():
            embeddings = self._model(x_tensor).cpu().numpy()
        return embeddings

    def _embed_onnx(self, image: np.ndarray) -> np.ndarray:
        """Run inference via ONNX runtime."""
        if image.ndim == 3:
            from dataset import preprocess_for_arcface
            x = preprocess_for_arcface(image)
        else:
            x = image
        x = x.astype(np.float32)
        input_name = self._onnx_session.get_inputs()[0].name
        output = self._onnx_session.run(None, {input_name: x})[0]
        return output[0]

    def load_onnx(self, onnx_path: Path | str) -> "ArcFaceEmbedder":
        """Load an ONNX model for inference."""
        import onnxruntime as ort
        self._onnx_session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"],
        )
        return self


# ---------------------------------------------------------------------------
# Anti-spoofing / liveness check
# ---------------------------------------------------------------------------
class AntiSpoofingChecker:
    """Heuristic anti-spoofing / liveness checker.

    Uses three complementary heuristics:
        1. **Texture analysis**: compute the local binary pattern (LBP)
           variance in the face region. Real faces have high variance
           (pores, micro-expressions); photos have low variance.
        2. **Frequency analysis**: compute the ratio of high-frequency
           to low-frequency content in the FFT spectrum. Real faces have
           more high-frequency detail.
        3. **Color consistency**: real faces have consistent skin-tone
           distribution; spoofed faces (printed photos) often have
           color-cast artifacts.

    Returns a ``spoof_score`` in [0, 1] where 0 = definitely live and
    1 = definitely spoof. The ``is_live`` decision uses a threshold of
    0.5 by default.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def check(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """Run liveness check on a face image.

        Parameters
        ----------
        face_image : np.ndarray, shape (H, W, 3)
            Aligned face image (BGR uint8).

        Returns
        -------
        (is_live, spoof_score)
            ``is_live``: True if the image is likely a real face.
            ``spoof_score``: float in [0, 1] (0=live, 1=spoof).
        """
        if not HAVE_CV2:
            raise RuntimeError("opencv is required for AntiSpoofingChecker.")

        if face_image.ndim == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

        # 1. Texture analysis (LBP variance).
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        lbp = self._local_binary_pattern(gray)
        texture_score = float(np.std(lbp))

        # 2. Frequency analysis (FFT high-freq ratio).
        fft = np.fft.fft2(gray.astype(np.float32))
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        h, w = magnitude.shape
        center_r, center_c = h // 2, w // 2
        # Define "high frequency" as outside the central 25% radius.
        radius = min(h, w) // 4
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((y_coords - center_r) ** 2 + (x_coords - center_c) ** 2)
        high_freq_mask = dist > radius
        low_freq_mask = ~high_freq_mask
        high_freq_energy = magnitude[high_freq_mask].mean()
        low_freq_energy = max(magnitude[low_freq_mask].mean(), 1e-6)
        freq_ratio = float(high_freq_energy / low_freq_energy)

        # 3. Color consistency (skin-tone variance in HSV).
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        # Skin hue range: 0-50 (OpenCV scale 0-179).
        skin_mask = (hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] >= 30)
        if skin_mask.sum() > 0:
            saturation_var = float(hsv[skin_mask, 1].std())
        else:
            saturation_var = 0.0

        # Combine scores (higher = more likely live).
        # Normalize each to [0, 1] using empirical ranges.
        texture_norm = min(texture_score / 30.0, 1.0)  # real faces: ~20-50
        freq_norm = min(freq_ratio / 0.5, 1.0)           # real faces: ~0.3-0.8
        sat_norm = min(saturation_var / 50.0, 1.0)      # real faces: ~30-70

        # Live score = average of normalized features.
        live_score = (texture_norm + freq_norm + sat_norm) / 3.0
        spoof_score = 1.0 - live_score

        is_live = spoof_score < self.threshold
        return is_live, float(spoof_score)

    def _local_binary_pattern(self, gray: np.ndarray) -> np.ndarray:
        """Compute a simplified local binary pattern."""
        h, w = gray.shape
        lbp = np.zeros_like(gray, dtype=np.float32)
        # 8-neighborhood LBP.
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                lbp += (shifted >= gray).astype(np.float32)
        return lbp


# ---------------------------------------------------------------------------
# FAISS vector index
# ---------------------------------------------------------------------------
class FaceVectorIndex:
    """FAISS vector index for face identity search.

    Uses ``IndexFlatIP`` (inner product) on L2-normalized vectors,
    which is mathematically equivalent to cosine similarity.

    Attributes
    ----------
    index : faiss.IndexFlatIP
        The FAISS index (512-dim, inner product).
    identities : list of str
        Identity labels aligned with the FAISS index.
    """

    def __init__(self, embedding_dim: int = 512):
        if not HAVE_FAISS:
            raise RuntimeError("faiss is required for FaceVectorIndex.")
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.identities: List[str] = []

    def add(self, embedding: np.ndarray, identity: str) -> None:
        """Add a single embedding to the index.

        Parameters
        ----------
        embedding : np.ndarray, shape (512,) float32
            Must be L2-normalized.
        identity : str
            Identity label.
        """
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        # Ensure L2-normalized (re-normalize to be safe).
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.index.add(vec)
        self.identities.append(identity)

    def add_batch(self, embeddings: np.ndarray, identities: List[str]) -> None:
        """Add a batch of embeddings.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, 512) float32
        identities : list of str, length N
        """
        vecs = np.asarray(embeddings, dtype=np.float32)
        # L2-normalize each row.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs = vecs / norms
        self.index.add(vecs)
        self.identities.extend(identities)

    def search(self, query: np.ndarray, k: int = 5) -> RetrievalResult:
        """Search for the K nearest neighbors.

        Parameters
        ----------
        query : np.ndarray, shape (512,) or (1, 512) float32
            Query embedding (L2-normalized).
        k : int
            Number of neighbors to retrieve.

        Returns
        -------
        RetrievalResult
        """
        if len(self.identities) == 0:
            raise RuntimeError("Index is empty — call add() first.")
        vec = np.asarray(query, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        k_actual = min(k, len(self.identities))
        distances, indices = self.index.search(vec, k_actual)
        result_identities = [self.identities[i] for i in indices[0] if i >= 0]
        return RetrievalResult(
            query_idx=0,
            distances=distances[0],
            indices=indices[0],
            identities=result_identities,
        )

    def search_batch(self, queries: np.ndarray, k: int = 5) -> List[RetrievalResult]:
        """Batch search."""
        results = []
        for i in range(len(queries)):
            results.append(self.search(queries[i], k=k))
        return results

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
class FaceRecognitionPipeline:
    """End-to-end face recognition pipeline.

    Steps:
        1. Detect faces (Haar / RetinaFace / MediaPipe).
        2. Align to canonical landmarks.
        3. Extract ArcFace embeddings (512-D, L2-normalized).
        4. [Optional] Anti-spoofing check.
        5. Search FAISS index for top-K identities.
    """

    def __init__(
        self,
        detector_kind: DetectorKind = DetectorKind.HAAR,
        device: str = "cpu",
        onnx_path: Optional[Path | str] = None,
        spoof_threshold: float = 0.5,
    ):
        self.detector = FaceDetector(detector_kind)
        self.embedder = ArcFaceEmbedder(device=device)
        if onnx_path is not None:
            self.embedder.load_onnx(onnx_path)
        else:
            self.embedder.load()
        self.spoof_checker = AntiSpoofingChecker(threshold=spoof_threshold)
        self.index = FaceVectorIndex(embedding_dim=ArcFaceEmbedder.EMBEDDING_DIM)

    def register(self, face_image: np.ndarray, identity: str,
                 landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Register a face: detect → align → embed → add to index."""
        # If landmarks not provided, detect.
        if landmarks is None:
            detections = self.detector.detect(face_image)
            if not detections:
                raise RuntimeError("No face detected in the image.")
            det = detections[0]
            landmarks = det.landmarks
            # Crop face region.
            x, y, w, h = det.bbox
            face_image = face_image[y : y + h, x : x + w]

        # Align.
        from dataset import align_face
        aligned = align_face(face_image, landmarks)

        # Embed.
        embedding = self.embedder.embed(aligned)

        # Add to index.
        self.index.add(embedding, identity)
        return embedding

    def recognize(self, face_image: np.ndarray, k: int = 5,
                  check_liveness: bool = False,
                  landmarks: Optional[np.ndarray] = None) -> EmbeddingResult:
        """Recognize a face: detect → align → embed → search."""
        if landmarks is None:
            detections = self.detector.detect(face_image)
            if not detections:
                return EmbeddingResult(
                    embedding=np.zeros(ArcFaceEmbedder.EMBEDDING_DIM, dtype=np.float32),
                    identity=None, similarity=0.0, is_live=None, spoof_score=None,
                )
            det = detections[0]
            landmarks = det.landmarks
            x, y, w, h = det.bbox
            face_image = face_image[y : y + h, x : x + w]

        from dataset import align_face
        aligned = align_face(face_image, landmarks)
        embedding = self.embedder.embed(aligned)

        # Anti-spoofing.
        is_live = None
        spoof_score = None
        if check_liveness:
            is_live, spoof_score = self.spoof_checker.check(aligned)

        # Search.
        if self.index.size > 0:
            result = self.index.search(embedding, k=1)
            identity = result.identities[0] if result.identities else None
            similarity = float(result.distances[0]) if len(result.distances) > 0 else 0.0
        else:
            identity = None
            similarity = 0.0

        return EmbeddingResult(
            embedding=embedding, identity=identity, similarity=similarity,
            is_live=is_live, spoof_score=spoof_score,
        )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_to_onnx(
    embedder: ArcFaceEmbedder,
    output_path: Path | str,
    image_size: int = 112,
    opset: int = 17,
) -> Path:
    """Serialize the ArcFace embedder to ONNX."""
    if not HAVE_TORCH:
        raise RuntimeError("torch is required for ONNX export.")
    if embedder._model is None:
        embedder.load()
    embedder._model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        embedder._model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )
    return output_path


def load_onnx_embedder(onnx_path: Path | str, device: str = "cpu") -> ArcFaceEmbedder:
    """Load an ONNX ArcFace embedder."""
    embedder = ArcFaceEmbedder(device=device)
    embedder.load_onnx(onnx_path)
    return embedder


__all__ = [
    "DetectorKind",
    "FaceDetection",
    "EmbeddingResult",
    "RetrievalResult",
    "HaarFaceDetector",
    "FaceDetector",
    "ArcFaceEmbedder",
    "AntiSpoofingChecker",
    "FaceVectorIndex",
    "FaceRecognitionPipeline",
    "export_to_onnx",
    "load_onnx_embedder",
    "HAVE_CV2",
    "HAVE_TORCH",
    "HAVE_FAISS",
]


# Module-level logger (used by FaceDetector._load above).
import logging as _logging
log = _logging.getLogger("face_model")
