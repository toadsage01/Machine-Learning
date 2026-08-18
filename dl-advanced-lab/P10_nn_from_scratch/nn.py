"""
nn
==

PyTorch-like layer abstractions built on top of the custom autograd
``Tensor`` class. Provides Module, Linear, Conv2D, MaxPool2D, Sequential,
ReLU, Sigmoid, optimizers (SGD, Adam), and loss functions.

Public surface
--------------
- ``Module``           : base class for layers + models.
- ``Linear``            : fully-connected layer.
- ``Conv2D``            : 2D convolution (no padding, no stride > 1 —
                          the minimal reference implementation).
- ``MaxPool2D``         : 2D max pooling.
- ``Sequential``        : container that chains layers.
- ``ReLU``              : rectified linear unit.
- ``Sigmoid``           : sigmoid activation.
- ``Flatten``           : reshape (B, C, H, W) → (B, C*H*W).
- ``SGD``               : stochastic gradient descent.
- ``Adam``              : adaptive moment estimator.
- ``cross_entropy``     : functional loss.
- ``mse_loss``          : functional loss.

Design notes
------------
1. **Module tracks parameters** — every Module's ``parameters()`` method
   returns a list of all Tensors in its subtree that have
   ``requires_grad=True``. Optimizers consume this list.

2. **Conv2D is a reference impl** — uses an im2col trick for efficiency
   (transform the input into a 2D matrix of patches, then a single
   matmul produces the output). This is the same algorithm PyTorch
   uses internally, just unoptimized.

3. **MaxPool2D is implemented as a strided slice + reshape** — the
   forward pass extracts non-overlapping windows and takes the max
   over each. The backward pass uses ``np.where`` to route the
   gradient to the argmax position in each window.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from autograd import Tensor, zeros, randn  # noqa: E402


# ---------------------------------------------------------------------------
# Module base class
# ---------------------------------------------------------------------------
class Module:
    """Base class for all layers and models.

    Subclasses must:
        * Set ``self.training = True`` (inherited).
        * Implement ``forward(self, *args)``.
        * Register sub-modules via ``self._modules[name] = module`` or
          by assignment (``self.layer = Linear(...)``).
        * Register parameters via ``self._params[name] = tensor`` or
          by assignment with a Tensor that has ``requires_grad=True``.
    """

    def __init__(self):
        self._modules: Dict[str, "Module"] = {}
        self._params: Dict[str, Tensor] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        # Auto-register Module + Tensor (with requires_grad) attributes.
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor) and value.requires_grad:
            self._params[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> List[Tensor]:
        """Return all parameters in this module's subtree."""
        params = list(self._params.values())
        for mod in self._modules.values():
            params.extend(mod.parameters())
        return params

    def zero_grad(self) -> None:
        """Reset gradients on all parameters."""
        for p in self.parameters():
            p.grad = None

    def train(self) -> "Module":
        """Set training mode."""
        self.training = True
        for mod in self._modules.values():
            mod.train()
        return self

    def eval(self) -> "Module":
        """Set eval mode (no dropout / batchnorm updates)."""
        self.training = False
        for mod in self._modules.values():
            mod.eval()
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------
class Linear(Module):
    """Fully-connected layer: ``y = x @ W + b``.

    Parameters
    ----------
    in_features : int
    out_features : int
    bias : bool
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Kaiming initialization (good for ReLU networks).
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True,
        )
        if bias:
            self.bias = Tensor(
                np.zeros(out_features),
                requires_grad=True,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Conv2D (im2col reference implementation)
# ---------------------------------------------------------------------------
def _im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1) -> np.ndarray:
    """Transform a 4D input (B, C, H, W) into a 2D matrix of patches.

    Output shape: (B * out_h * out_w, C * kh * kw)
    where out_h = (H - kh) // stride + 1, out_w = (W - kw) // stride + 1.
    """
    B, C, H, W = x.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    # Build patch indices.
    cols = np.zeros((B, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_end = i + stride * out_h
        for j in range(kw):
            j_end = j + stride * out_w
            cols[:, :, i, j, :, :] = x[:, :, i:i_end:stride, j:j_end:stride]
    # Reshape: (B, C, kh, kw, out_h, out_w) → (B * out_h * out_w, C * kh * kw).
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(B * out_h * out_w, -1)
    return cols


class Conv2D(Module):
    """2D convolution (no padding, stride=1 only — reference impl).

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # He initialization.
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale,
            requires_grad=True,
        )
        self.bias = Tensor(
            np.zeros(out_channels),
            requires_grad=True,
        )
        # Cache for backward (set during forward).
        self._cache: Dict[str, Any] = {}

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, C, H, W). weight shape: (out_C, in_C, kh, kw).
        x_data = x.data
        B, C, H, W = x_data.shape
        out_C, _, kh, kw = self.weight.shape
        out_h = H - kh + 1
        out_w = W - kw + 1

        # im2col: (B * out_h * out_w, C * kh * kw).
        cols = _im2col(x_data, kh, kw, stride=1)
        # weight_reshaped: (out_C, C * kh * kw) → transpose → (C * kh * kw, out_C).
        w_reshaped = self.weight.data.reshape(out_C, -1).T
        # matmul: (B * out_h * out_w, out_C).
        out = cols @ w_reshaped + self.bias.data[None, :]
        out = out.reshape(B, out_h, out_w, out_C).transpose(0, 3, 1, 2)
        # out shape: (B, out_C, out_h, out_w).

        requires_grad = x.requires_grad or self.weight.requires_grad or self.bias.requires_grad
        out_t = Tensor(out, requires_grad=requires_grad,
                        _parents=(x,) if requires_grad else (),
                        _backward=None)

        # Cache for backward.
        self._cache = {
            "x_shape": (B, C, H, W),
            "cols": cols,
            "out_h": out_h, "out_w": out_w, "out_C": out_C,
        }

        def _backward():
            if not requires_grad:
                return
            grad = out_t.grad  # shape (B, out_C, out_h, out_w).
            # Reshape grad to (B * out_h * out_w, out_C).
            grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, out_C)
            # Bias grad: sum over batch + spatial dims → (out_C,).
            if self.bias.requires_grad:
                bias_grad = grad_reshaped.sum(axis=0)
                self.bias._accumulate_grad(bias_grad)
            # Weight grad: dL/dW = cols.T @ grad_reshaped → reshape to (out_C, in_C, kh, kw).
            if self.weight.requires_grad:
                w_grad_flat = cols.T @ grad_reshaped  # (C*kh*kw, out_C).
                w_grad = w_grad_flat.T.reshape(self.weight.shape)
                self.weight._accumulate_grad(w_grad)
            # Input grad: dL/dx = grad_reshaped @ w_reshaped.T → col2im back.
            if x.requires_grad:
                x_grad_cols = grad_reshaped @ w_reshaped.T  # (B*out_h*out_w, C*kh*kw).
                x_grad = _col2im(x_grad_cols, (B, C, H, W), kh, kw, out_h, out_w)
                x._accumulate_grad(x_grad)

        out_t._backward = _backward
        return out_t


def _col2im(cols: np.ndarray, x_shape: Tuple[int, int, int, int],
            kh: int, kw: int, out_h: int, out_w: int) -> np.ndarray:
    """Reverse of im2col: accumulate patches back into image shape."""
    B, C, H, W = x_shape
    cols_reshaped = cols.reshape(B, out_h, out_w, C, kh, kw)
    cols_reshaped = cols_reshaped.transpose(0, 3, 4, 5, 1, 2)
    # (B, C, kh, kw, out_h, out_w).
    x_grad = np.zeros((B, C, H, W), dtype=cols.dtype)
    for i in range(kh):
        i_end = i + out_h
        for j in range(kw):
            j_end = j + out_w
            x_grad[:, :, i:i_end, j:j_end] += cols_reshaped[:, :, i, j, :, :]
    return x_grad


# ---------------------------------------------------------------------------
# MaxPool2D
# ---------------------------------------------------------------------------
class MaxPool2D(Module):
    """2D max pooling with non-overlapping windows.

    Parameters
    ----------
    pool_size : int
    """

    def __init__(self, pool_size: int = 2):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x: Tensor) -> Tensor:
        ps = self.pool_size
        x_data = x.data
        B, C, H, W = x_data.shape
        out_h = H // ps
        out_w = W // ps
        # Reshape into windows: (B, C, out_h, ps, out_w, ps).
        windows = x_data[:, :, :out_h * ps, :out_w * ps].reshape(
            B, C, out_h, ps, out_w, ps
        )
        # Max over (ps, ps) windows.
        out = windows.max(axis=(3, 5))
        # out shape: (B, C, out_h, out_w).

        requires_grad = x.requires_grad
        out_t = Tensor(out, requires_grad=requires_grad,
                        _parents=(x,) if requires_grad else (),
                        _backward=None)

        def _backward():
            if not requires_grad:
                return
            grad = out_t.grad  # (B, C, out_h, out_w).
            # Build mask: where the max was.
            # Broadcast out back to the window shape, then compare.
            maxes = out[:, :, :, None, :, None]  # (B, C, out_h, 1, out_w, 1).
            mask = (windows == maxes).astype(np.float64)
            # Normalize mask if any window had ties (multiple maxima).
            # (We sum over (ps, ps) per window and divide — standard
            # gradient-routing trick for max-pooling with ties.)
            n_maxes = mask.sum(axis=(3, 5), keepdims=True)
            mask = mask / np.maximum(n_maxes, 1.0)
            # Multiply by the upstream gradient and reshape back.
            grad_expanded = grad[:, :, :, None, :, None]
            x_grad_windows = mask * grad_expanded
            # Reshape back to (B, C, H, W).
            x_grad = x_grad_windows.transpose(0, 1, 2, 3, 4, 5).reshape(
                B, C, out_h * ps, out_w * ps
            )
            # The leftover rows/cols (if H/W not divisible by ps) get zero grad.
            full_grad = np.zeros_like(x_data)
            full_grad[:, :, :out_h * ps, :out_w * ps] = x_grad
            x._accumulate_grad(full_grad)

        out_t._backward = _backward
        return out_t


# ---------------------------------------------------------------------------
# Activations + reshaping
# ---------------------------------------------------------------------------
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Flatten(Module):
    """Reshape (B, C, H, W) → (B, C*H*W)."""

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        return x.reshape(B, -1)


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------
class Sequential(Module):
    """Chain of layers applied in order."""

    def __init__(self, *layers: Module):
        super().__init__()
        self.layers: List[Module] = list(layers)
        # Register sub-modules so parameters() walks them.
        for i, layer in enumerate(self.layers):
            self._modules[f"layer_{i}"] = layer

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
class Optimizer:
    """Base optimizer class."""

    def __init__(self, params: List[Tensor]):
        self.params = params

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum.

    Update rule (with momentum):
        v ← β · v + grad
        p ← p − lr · v
    """

    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocities: List[np.ndarray] = [np.zeros_like(p.data) for p in params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            if self.momentum > 0:
                self._velocities[i] = self.momentum * self._velocities[i] + grad
                grad = self._velocities[i]
            p.data -= self.lr * grad


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba 2015).

    Update rule (bias-corrected):
        m ← β₁ · m + (1 - β₁) · g
        v ← β₂ · v + (1 - β₂) · g²
        m̂ ← m / (1 - β₁ᵗ)
        v̂ ← v / (1 - β₂ᵗ)
        p ← p − lr · m̂ / (√v̂ + ε)
    """

    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: List[np.ndarray] = [np.zeros_like(p.data) for p in params]
        self._v: List[np.ndarray] = [np.zeros_like(p.data) for p in params]
        self._t: int = 0

    def step(self) -> None:
        self._t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self._m[i] / (1 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1 - self.beta2 ** self._t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Functional losses
# ---------------------------------------------------------------------------
def cross_entropy(logits: Tensor, targets: Union[Tensor, np.ndarray, list]) -> Tensor:
    """Softmax cross-entropy loss (see ``Tensor.cross_entropy``)."""
    return logits.cross_entropy(targets)


def mse_loss(pred: Tensor, targets: Union[Tensor, np.ndarray, list]) -> Tensor:
    """Mean squared error loss (see ``Tensor.mse``)."""
    return pred.mse(targets)


__all__ = [
    "Module",
    "Linear",
    "Conv2D",
    "MaxPool2D",
    "Sequential",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Flatten",
    "Optimizer",
    "SGD",
    "Adam",
    "cross_entropy",
    "mse_loss",
]
