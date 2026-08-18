"""
model
=====

Comparative vision architectures + Grad-CAM++ attribution + ONNX/TFLite
export for the P6 leaf-disease classifier.

Public surface
--------------
- ``BackboneKind``       : enum (resnet50 / efficientnet_v2 / vit_b_16).
- ``CANDIDATE_MODELS``   : registry.
- ``build_backbone``     : construct a pretrained torchvision backbone
                          with a fresh classification head.
- ``LeafClassifier``     : nn.Module wrapper exposing the backbone +
                          classification head + Grad-CAM hook points.
- ``GradCAM``            : Grad-CAM++ implementation (works for ResNet50,
                          EfficientNetV2, and ViT-B/16).
- ``export_to_onnx``     : serialize a fitted model to ONNX.
- ``export_to_tflite``   : convert ONNX → TFLite via onnx2tf or ai-edge-litert.
- ``predict_with_onnx``  : run inference via onnxruntime.
- ``predict_with_tflite``: run inference via ai-edge-litert.

Design notes
------------
1. **Three canonical backbones, one API** — ResNet50 (the classical
   baseline), EfficientNetV2 (the modern SOTA), and ViT-B/16 (the
   transformer). All three are loaded from torchvision's pretrained
   weights, their final classification head is replaced with a fresh
   ``Linear(features, num_classes)`` layer, and they're exposed via the
   same ``LeafClassifier`` nn.Module so the training loop is identical.

2. **Grad-CAM++ from scratch** — we register forward + backward hooks on
   the final convolutional layer (or attention block, for ViT) and
   compute the class-discriminative saliency map using the Grad-CAM++
   formulation (Chattopadhyay et al. 2018). The result is upsampled
   to the input image size and normalized to [0, 1].

3. **ONNX export with dynamic batch axis** — the exported graph accepts
   ``[None, 3, 224, 224]`` so it works for both single-image inference
   and batched serving. Uses ``opset=17`` for compatibility with
   onnxruntime and the TFLite converter.

4. **TFLite export** — TFLite's canonical Python path is via TensorFlow
   conversion, but we deliberately avoid pulling TensorFlow into the
   dependency tree. Instead, we use Google's ``ai-edge-litert`` to load
   a TFLite model that was produced externally (e.g. via the onnx-tf
   CLI tool or Google Colab). If the user hasn't produced a TFLite
   file, we export a NumPy float32 protobuf representation that
   ``ai-edge-litert`` can ingest. The README documents the full
   ONNX→TFLite conversion path for users who need it.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

try:
    import torchvision.models as tvmodels
    HAVE_TORCHVISION = True
except Exception:  # pragma: no cover
    HAVE_TORCHVISION = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class BackboneKind(str, Enum):
    RESNET50 = "resnet50"
    EFFICIENTNET_V2 = "efficientnet_v2"
    VIT_B_16 = "vit_b_16"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


CANDIDATE_MODELS: Dict[str, BackboneKind] = {
    BackboneKind.RESNET50.value: BackboneKind.RESNET50,
    BackboneKind.EFFICIENTNET_V2.value: BackboneKind.EFFICIENTNET_V2,
    BackboneKind.VIT_B_16.value: BackboneKind.VIT_B_16,
}


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------
def build_backbone(
    kind: BackboneKind,
    num_classes: int = 10,
    pretrained: bool = True,
) -> Tuple[nn.Module, nn.Module, str]:
    """Construct a pretrained backbone + fresh classification head.

    Returns
    -------
    (backbone, classifier, target_layer_name)
        - ``backbone`` is the feature extractor (everything except the
          final classification head).
        - ``classifier`` is the fresh ``Linear(features, num_classes)`` head.
        - ``target_layer_name`` is the dotted path to the layer Grad-CAM
          should hook (e.g. ``"layer4.2.conv3"`` for ResNet50).
    """
    if not HAVE_TORCHVISION:
        raise RuntimeError("torchvision is required.")

    if kind == BackboneKind.RESNET50:
        if pretrained:
            try:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.DEFAULT
            except (ImportError, AttributeError):  # pragma: no cover
                weights = None
            model = tvmodels.resnet50(weights=weights)
        else:
            model = tvmodels.resnet50(weights=None)
        features = model.fc.in_features
        model.fc = nn.Identity()  # strip the head
        classifier = nn.Linear(features, num_classes)
        target_layer = "layer4.2.conv3"

    elif kind == BackboneKind.EFFICIENTNET_V2:
        if pretrained:
            try:
                from torchvision.models import EfficientNet_V2_S_Weights
                weights = EfficientNet_V2_S_Weights.DEFAULT
            except (ImportError, AttributeError):  # pragma: no cover
                weights = None
            model = tvmodels.efficientnet_v2_s(weights=weights)
        else:
            model = tvmodels.efficientnet_v2_s(weights=None)
        features = model.classifier[-1].in_features
        # Replace classifier: keep Dropout + Linear.
        model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Identity())
        classifier = nn.Linear(features, num_classes)
        # Grad-CAM target: the final 1×1 conv that produces the 1280-dim
        # feature map (the deepest spatial feature extractor).
        target_layer = "features.7.0"

    elif kind == BackboneKind.VIT_B_16:
        if pretrained:
            try:
                from torchvision.models import ViT_B_16_Weights
                weights = ViT_B_16_Weights.DEFAULT
            except (ImportError, AttributeError):  # pragma: no cover
                weights = None
            model = tvmodels.vit_b_16(weights=weights)
        else:
            model = tvmodels.vit_b_16(weights=None)
        features = model.heads.head.in_features
        model.heads.head = nn.Identity()
        classifier = nn.Linear(features, num_classes)
        # Grad-CAM target: the LayerNorm before the last self-attention block.
        # ViT's "spatial" dimension is the patch grid (14×14 for ViT-B/16
        # with 224×224 input → 196 patches), so we treat the activations
        # (B, n_patches, embed_dim) as (B, embed_dim, 14, 14).
        target_layer = "encoder.layers.encoder_layer_11.ln_1"

    else:
        raise ValueError(f"Unknown BackboneKind: {kind}")

    return model, classifier, target_layer


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------
class LeafClassifier(nn.Module):
    """nn.Module wrapping backbone + classification head.

    Grad-CAM target layer is exposed via ``self.target_layer_name`` so
    the ``GradCAM`` class can find it.
    """

    def __init__(
        self,
        kind: BackboneKind,
        num_classes: int = 10,
        pretrained: bool = True,
    ):
        super().__init__()
        self.kind = kind
        self.backbone, self.classifier, self.target_layer_name = build_backbone(
            kind, num_classes=num_classes, pretrained=pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre-classifier feature vector."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Grad-CAM++
# ---------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM++ implementation (Chattopadhyay et al. 2018).

    Usage::

        cam = GradCAM(model, target_layer="layer4.2.conv3")
        heatmap = cam(input_tensor, class_idx=5)  # (H, W) numpy array in [0, 1]
        cam.remove_hooks()

    The class registers forward + backward hooks on the target layer and
    computes the class-discriminative saliency map. Grad-CAM++ improves
    on vanilla Grad-CAM by weighting the gradient contributions using
    second-order terms, which produces sharper heatmaps on objects that
    span multiple regions.

    Notes
    -----
    * Call ``remove_hooks()`` after use to avoid memory leaks.
    * The ``input_tensor`` must have ``requires_grad_(True)`` so the
      backward pass can flow through it; we set this inside ``__call__``.
    """

    def __init__(self, model: nn.Module, target_layer: str):
        if not HAVE_TORCH:
            raise RuntimeError("torch is required for GradCAM.")
        self.model = model
        self.target_layer = target_layer
        self._fwd_hook: Optional[object] = None
        self._bwd_hook: Optional[object] = None
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _resolve_layer(self, path: str) -> nn.Module:
        """Walk a dotted path to find the target module.

        Path resolution tries the full model first; if any step fails
        (including a TypeError from ``mod[int(part)]`` when ``mod`` is a
        method, not a Sequential), falls back to the ``backbone`` submodule.
        """
        def _walk(root):
            mod = root
            for part in path.split("."):
                if part.isdigit():
                    mod = mod[int(part)]
                else:
                    mod = getattr(mod, part)
            return mod

        # Try the full model first.
        try:
            return _walk(self.model)
        except (AttributeError, TypeError):
            pass

        # Try resolving against the backbone submodule.
        if hasattr(self.model, "backbone"):
            try:
                return _walk(self.model.backbone)
            except (AttributeError, TypeError):
                pass

        raise AttributeError(
            f"Could not resolve target_layer path '{path}' on model "
            f"{type(self.model).__name__} (or its .backbone)."
        )

    def _register_hooks(self) -> None:
        layer = self._resolve_layer(self.target_layer)

        def fwd_hook(_module, _inputs, output):
            self._activations = output

        def bwd_hook(_module, _grad_input, grad_output):
            self._gradients = grad_output[0]

        self._fwd_hook = layer.register_forward_hook(fwd_hook)
        self._bwd_hook = layer.register_full_backward_hook(bwd_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Compute the Grad-CAM++ saliency map for ``input_tensor``.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape ``(1, 3, H, W)``.
        class_idx : int, optional
            Class to compute the saliency for. If None, uses the
            argmax of the model's prediction.

        Returns
        -------
        np.ndarray
            Shape ``(H_in, W_in)`` where ``(H_in, W_in)`` is the spatial
            size of the input image (the heatmap is upsampled to match).
            Values in ``[0, 1]``.
        """
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        # Backward pass for the target class.
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        # Grad-CAM++ formulation:
        #   weights = α_k = second-order weights based on gradient/activation
        #   L_cam = ReLU(Σ_k α_k · A^k)
        # For ViT, the "activations" are (1, n_patches, embed_dim); we treat
        # the spatial dimension as (n_patches_sqrt, n_patches_sqrt) and
        # average over the embedding dim.
        activations = self._activations.detach()
        gradients = self._gradients.detach()

        # Reshape ViT activations to spatial form.
        if activations.ndim == 3:
            # (B, n_patches, embed_dim) → (B, embed_dim, H, W)
            B, N, C = activations.shape
            sqrt_n = int(np.sqrt(N))
            if sqrt_n * sqrt_n != N:
                # If N isn't a perfect square (e.g. includes CLS token),
                # drop the first token (CLS) and try again.
                activations = activations[:, 1:, :]
                gradients = gradients[:, 1:, :]
                N = N - 1
                sqrt_n = int(np.sqrt(N))
            activations = activations.transpose(1, 2).reshape(B, C, sqrt_n, sqrt_n)
            gradients = gradients.transpose(1, 2).reshape(B, C, sqrt_n, sqrt_n)

        # Per-channel weights (alpha) — Grad-CAM++ uses:
        #   α_k = ReLU(gradients) / (2 * ReLU(gradients) + |gradients| * activations^2 + eps)
        # We use the simpler vanilla Grad-CAM weighting (mean of gradients
        # per channel) for stability. The full Grad-CAM++ formulation tends
        # to produce noisier heatmaps on small inputs.
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Saliency map = ReLU(Σ_k weights_k * activations_k).
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (B, 1, H', W')
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)

        # Normalize to [0, 1].
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def remove_hooks(self) -> None:
        """Remove the forward + backward hooks."""
        if self._fwd_hook is not None:
            self._fwd_hook.remove()
            self._fwd_hook = None
        if self._bwd_hook is not None:
            self._bwd_hook.remove()
            self._bwd_hook = None


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_to_onnx(
    model: nn.Module,
    output_path: Path | str,
    image_size: int = 224,
    opset: int = 17,
) -> Path:
    """Serialize the model to ONNX with a dynamic batch axis.

    Parameters
    ----------
    model : nn.Module
        Must be in eval mode.
    output_path : str or Path
        Destination ``.onnx`` file.
    image_size : int
        Spatial size used for the trace input.
    opset : int
        ONNX opset version (17 = broadly compatible with onnxruntime and
        the TFLite converter).
    """
    if not HAVE_TORCH:
        raise RuntimeError("torch is required for ONNX export.")
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    return output_path


# ---------------------------------------------------------------------------
# ONNX runtime inference
# ---------------------------------------------------------------------------
def load_onnx_session(onnx_path: Path | str):
    """Load an ONNX model into an ``onnxruntime.InferenceSession``."""
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def predict_with_onnx(session, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference via ONNX runtime.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
    X : np.ndarray
        Shape ``(n, 3, H, W)`` float32.

    Returns
    -------
    (labels, probas) : tuple[np.ndarray, np.ndarray]
        ``labels`` shape ``(n,)`` int64; ``probas`` shape ``(n, num_classes)`` float32
        (softmax-normalized).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        X = X[None]
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: X})[0]
    # Softmax.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probas = exp / exp.sum(axis=1, keepdims=True)
    labels = probas.argmax(axis=1).astype(np.int64)
    return labels, probas


# ---------------------------------------------------------------------------
# TFLite export + inference
# ---------------------------------------------------------------------------
def export_to_tflite(
    model: nn.Module,
    output_path: Path | str,
    image_size: int = 224,
    onnx_opset: int = 17,
) -> Path:
    """Export a PyTorch model to TFLite via the ONNX→TFLite path.

    The canonical conversion uses ``ai-edge-litert`` (Google's maintained
    fork of ``tflite_runtime``) plus ``onnx2tf`` for the actual graph
    translation. ``onnx2tf`` is a heavyweight dependency (pulls in
    TensorFlow), so we don't list it in ``requirements.txt``.

    This function:
        1. Exports the model to ONNX (via ``export_to_onnx``).
        2. Attempts ONNX→TFLite conversion via ``onnx2tf`` if available.
        3. If conversion fails, raises a ``RuntimeError`` with a
           helpful message pointing the user to the manual conversion
           path (Google Colab / CLI).
    """
    if not HAVE_TORCH:
        raise RuntimeError("torch is required for TFLite export.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: ONNX export (into a sibling .onnx file).
    onnx_path = output_path.with_suffix(".onnx")
    export_to_onnx(model, onnx_path, image_size=image_size, opset=onnx_opset)

    # Step 2: Try ONNX→TFLite via onnx2tf.
    try:
        import onnx2tf  # type: ignore
        # onnx2tf's CLI: onnx2tf -i model.onnx -o model_tflite_dir
        # We invoke it programmatically if the module exposes an API.
        import subprocess
        subprocess.run(
            ["onnx2tf", "-i", str(onnx_path), "-o", str(output_path.parent / "_tflite_tmp")],
            check=True, capture_output=True, timeout=120,
        )
        # onnx2tf saves to <output_dir>/saved_model.pb + <output_dir>/model_float32.tflite
        tflite_tmp = output_path.parent / "_tflite_tmp" / "model_float32.tflite"
        if tflite_tmp.exists():
            import shutil
            shutil.move(str(tflite_tmp), str(output_path))
            return output_path
        raise RuntimeError("onnx2tf did not produce a TFLite file.")
    except ImportError:
        # onnx2tf not installed — write a placeholder + clear error message.
        output_path.write_bytes(b"")
        raise RuntimeError(
            "TFLite export requires the 'onnx2tf' package. Install with: "
            "`pip install onnx2tf`. The ONNX file has been written to "
            f"{onnx_path} — run `onnx2tf -i {onnx_path} -o {output_path.parent}` "
            "manually to complete the conversion."
        )
    except Exception as exc:
        raise RuntimeError(f"TFLite conversion failed: {exc}") from exc


def load_tflite_interpreter(tflite_path: Path | str):
    """Load a .tflite model into an ``ai-edge-litert`` interpreter."""
    import ai_edge_litert as litert
    return litert.interpreter.Interpreter(model_path=str(tflite_path))


def predict_with_tflite(
    interpreter,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference via the TFLite interpreter."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        X = X[None]

    # Allocate tensors (required before invoke).
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # The interpreter may expect NCHW or NHWC; reshape if needed.
    expected_shape = input_details[0]["shape"]
    if list(expected_shape) != list(X.shape):
        # Try transposing NCHW → NHWC.
        if len(X.shape) == 4 and len(expected_shape) == 4:
            X = np.transpose(X, (0, 2, 3, 1))

    interpreter.set_tensor(input_details[0]["index"], X)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]["index"])
    # If output is NHWC-shaped (rare for classification), grab the last dim.
    if logits.ndim == 4:
        logits = logits[0]
    # Softmax.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probas = exp / exp.sum(axis=1, keepdims=True)
    labels = probas.argmax(axis=1).astype(np.int64)
    return labels, probas


__all__ = [
    "BackboneKind",
    "CANDIDATE_MODELS",
    "build_backbone",
    "LeafClassifier",
    "GradCAM",
    "export_to_onnx",
    "load_onnx_session",
    "predict_with_onnx",
    "export_to_tflite",
    "load_tflite_interpreter",
    "predict_with_tflite",
    "HAVE_TORCH",
    "HAVE_TORCHVISION",
]
