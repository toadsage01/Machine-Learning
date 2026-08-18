"""
autograd
========

N-dimensional Tensor class with reverse-mode automatic differentiation.

Public surface
--------------
- ``Tensor``               : the core value object. Supports numpy-like
                             broadcasting for elementwise ops and tracks
                             a DAG of operations for backward().
- ``unbroadcast``          : helper that sums a gradient back to its
                             original shape after a broadcasted op.

Supported ops (each with a custom backward):
    add, sub, mul, truediv, pow, neg, matmul, sum, mean,
    transpose, reshape, relu, sigmoid, tanh, log, exp,
    cross_entropy (softmax + NLL), mse, getitem (slicing).

Design notes
------------
1. **Reverse-mode autodiff via a DAG** — every Tensor stores a
   ``_backward`` closure and a set of parent Tensors. When ``backward()``
   is called on the loss, we topologically sort the DAG and walk it in
   reverse, calling each node's ``_backward`` closure to accumulate
   gradients into ``.grad``.

2. **Broadcasting-aware gradient accumulation** — when an op broadcasts
   a smaller tensor against a larger one (e.g. shape (3,) + shape (5, 3)),
   the upstream gradient has the broadcasted shape and must be summed
   back to the original shape. The ``unbroadcast`` helper handles this.

3. **Closure-based backward functions** — each op defines a closure
   that captures the input Tensors and computes the local gradient
   contribution. This is the same pattern that micrograd uses, extended
   to N-dimensional tensors.

4. **No in-place ops** — every op returns a new Tensor. In-place ops
   would break the DAG because the same Tensor might be used in multiple
   places. PyTorch allows in-place ops via version counters; we don't
   implement that complexity.

5. **PyTorch parity** — the test suite verifies that gradients computed
   by our engine match PyTorch's autograd to 1e-5 precision for every op.
   This is the canonical "is my autograd correct?" test.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Broadcasting helper
# ---------------------------------------------------------------------------
def unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` back to ``shape`` after a numpy broadcast.

    Numpy broadcasting rules:
        * Trailing dimensions are matched element-wise.
        * Dimensions of size 1 are broadcast to the larger size.
        * Missing leading dimensions are treated as size 1.

    To reverse a broadcast, we sum the gradient along any axis where the
    original shape had size 1 (or was missing entirely).
    """
    # Sum over extra leading dimensions that were added by broadcasting.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum over dimensions where the original shape was 1 (broadcasted).
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


# ---------------------------------------------------------------------------
# Matmul gradient helpers (handle 1-D and 2-D operands uniformly)
# ---------------------------------------------------------------------------
def _matmul_grad_a(grad: np.ndarray, a: np.ndarray, b: np.ndarray,
                    a_shape: Tuple[int, ...], b_shape: Tuple[int, ...]) -> np.ndarray:
    """Compute d(A @ B)/dA for various A/B shapes.

    Cases:
        (n,k) @ (k,)   → grad is (n,);   dA = outer(grad, b)
        (n,k) @ (k,m)  → grad is (n,m);  dA = grad @ b.T
        (b,n,k) @ (k,m) → grad is (b,n,m); dA = grad @ b.T (with broadcast)
    """
    if b.ndim == 1:
        # (..., k) @ (k,) → (...,). dA = outer(grad, b) per leading dim.
        # grad shape (..., k_after_reduce doesn't apply — grad is (...,))
        # We need dA shape (..., k). Use grad[..., None] * b[None, :].
        grad_reshaped = grad[..., None]   # (..., 1)
        b_reshaped = b[None, :]          # (1, k)
        return grad_reshaped * b_reshaped
    else:
        # (..., k) @ (k, m) → (..., m). dA = grad @ b.swapaxes(-1, -2).
        b_t = np.swapaxes(b, -1, -2)
        grad_a = np.matmul(grad, b_t)
        return unbroadcast(grad_a, a_shape)


def _matmul_grad_b(grad: np.ndarray, a: np.ndarray, b: np.ndarray,
                    a_shape: Tuple[int, ...], b_shape: Tuple[int, ...]) -> np.ndarray:
    """Compute d(A @ B)/dB for various A/B shapes.

    Cases:
        (n,k) @ (k,)   → grad is (n,);   db = a.T @ grad  (shape (k,))
        (n,k) @ (k,m)  → grad is (n,m);  db = a.T @ grad  (shape (k,m))
        (b,n,k) @ (k,m) → grad is (b,n,m); db = a.swapaxes(-1,-2) @ grad
    """
    if b.ndim == 1:
        # (..., k) @ (k,) → (...,). db = sum over leading dims of (a * grad[..., None]).
        # dL/db[k] = sum_n a[n, k] * grad[n].
        grad_reshaped = grad[..., None]    # (..., 1)
        # Multiply a (..., k) by grad (..., 1) and sum over leading dims.
        grad_b = (a * grad_reshaped).reshape(-1, a.shape[-1]).sum(axis=0)
        return grad_b
    else:
        # (..., k) @ (k, m) → (..., m). db = a.swapaxes(-1, -2) @ grad.
        a_t = np.swapaxes(a, -1, -2)
        grad_b = np.matmul(a_t, grad)
        return unbroadcast(grad_b, b_shape)


# ---------------------------------------------------------------------------
# Tensor class
# ---------------------------------------------------------------------------
class Tensor:
    """N-dimensional tensor with reverse-mode autodiff.

    Attributes
    ----------
    data : np.ndarray
        The forward-pass value.
    requires_grad : bool
        If True, gradients will be accumulated into ``.grad`` during
        ``backward()``.
    grad : np.ndarray or None
        The accumulated gradient (same shape as ``data``).
    _backward : callable
        Closure that computes the local backward pass for the op that
        produced this Tensor. Called once during ``backward()``.
    _parents : tuple of Tensors
        The Tensors that were inputs to the op that produced this Tensor.
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, "Tensor", float, int],
        requires_grad: bool = False,
        _parents: Tuple["Tensor", ...] = (),
        _backward: Optional[callable] = None,
    ):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward: callable = _backward or (lambda: None)
        self._parents: Tuple["Tensor", ...] = _parents

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_tensor(self, other: Union["Tensor", np.ndarray, float, int]) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(other, requires_grad=False)

    def zero_grad(self) -> None:
        """Reset the gradient to zero."""
        self.grad = np.zeros_like(self.data) if self.requires_grad else None

    def detach(self) -> "Tensor":
        """Return a new Tensor that shares data but is detached from the DAG."""
        return Tensor(self.data.copy(), requires_grad=False)

    # ------------------------------------------------------------------
    # Elementwise ops (broadcasting-aware)
    # ------------------------------------------------------------------
    def __add__(self, other: Union["Tensor", float]) -> "Tensor":
        other = self._ensure_tensor(other)
        out_data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad,
                      _parents=(self, other) if requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                grad = unbroadcast(out.grad, self.shape)
                self._accumulate_grad(grad)
            if other.requires_grad:
                grad = unbroadcast(out.grad, other.shape)
                other._accumulate_grad(grad)

        out._backward = _backward
        return out

    def __radd__(self, other: Union["Tensor", float]) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Union["Tensor", float]) -> "Tensor":
        other = self._ensure_tensor(other)
        return self.__add__(-other)

    def __rsub__(self, other: Union["Tensor", float]) -> "Tensor":
        other = self._ensure_tensor(other)
        return other.__add__(-self)

    def __neg__(self) -> "Tensor":
        return self.__mul__(-1.0)

    def __mul__(self, other: Union["Tensor", float]) -> "Tensor":
        other = self._ensure_tensor(other)
        out_data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad,
                      _parents=(self, other) if requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                # d(a*b)/da = b
                grad = unbroadcast(out.grad * other.data, self.shape)
                self._accumulate_grad(grad)
            if other.requires_grad:
                # d(a*b)/db = a
                grad = unbroadcast(out.grad * self.data, other.shape)
                other._accumulate_grad(grad)

        out._backward = _backward
        return out

    def __rmul__(self, other: Union["Tensor", float]) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Tensor", float]) -> "Tensor":
        other = self._ensure_tensor(other)
        return self.__mul__(other.__pow__(-1.0))

    def __rtruediv__(self, other: Union["Tensor", float]) -> "Tensor":
        other = self._ensure_tensor(other)
        return other.__mul__(self.__pow__(-1.0))

    def __pow__(self, p: Union[float, int]) -> "Tensor":
        assert isinstance(p, (int, float)), "Only scalar powers supported."
        out_data = self.data ** p
        requires_grad = self.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad,
                      _parents=(self,) if requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                # d(x^p)/dx = p * x^(p-1)
                grad = out.grad * (p * self.data ** (p - 1))
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Matrix multiply
    # ------------------------------------------------------------------
    def matmul(self, other: "Tensor") -> "Tensor":
        other = self._ensure_tensor(other)
        out_data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad,
                      _parents=(self, other) if requires_grad else (),
                      _backward=None)

        # Save the input shapes for the backward pass.
        a_shape = self.shape
        b_shape = other.shape

        def _backward():
            # Handle multiple matmul shapes:
            #   (n,k) @ (k,)   → (n,)     — vector dot per row
            #   (n,k) @ (k,m)  → (n,m)    — standard 2D matmul
            #   (b,n,k) @ (k,m) → (b,n,m) — batched matmul
            if self.requires_grad:
                grad = _matmul_grad_a(out.grad, self.data, other.data, a_shape, b_shape)
                self._accumulate_grad(grad)
            if other.requires_grad:
                grad = _matmul_grad_b(out.grad, self.data, other.data, a_shape, b_shape)
                other._accumulate_grad(grad)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
            keepdims: bool = False) -> "Tensor":
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad,
                      _parents=(self,) if requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # Broadcast the gradient back to the original shape.
                if axis is None:
                    grad = np.broadcast_to(grad, self.shape).copy()
                else:
                    if not keepdims:
                        # Re-insert the reduced dims.
                        if isinstance(axis, int):
                            axes = (axis,)
                        else:
                            axes = axis
                        axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                        for a in sorted(axes):
                            grad = np.expand_dims(grad, a)
                    grad = np.broadcast_to(grad, self.shape).copy()
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> "Tensor":
        """Mean over the given axis (or all axes if None)."""
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = int(np.prod([self.data.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------------
    # Shape ops
    # ------------------------------------------------------------------
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> "Tensor":
        out_data = np.transpose(self.data, axes=axes)
        requires_grad = self.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad,
                      _parents=(self,) if requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                if axes is None:
                    grad = np.transpose(out.grad)
                else:
                    # Invert the axes permutation.
                    inv = np.argsort(axes)
                    grad = np.transpose(out.grad, axes=inv)
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(int(s) for s in shape)
        out_data = self.data.reshape(shape)
        requires_grad = self.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad,
                      _parents=(self,) if requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.shape)
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def __getitem__(self, idx) -> "Tensor":
        out_data = self.data[idx]
        requires_grad = self.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad,
                      _parents=(self,) if requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, idx, out.grad)
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Nonlinearities
    # ------------------------------------------------------------------
    def relu(self) -> "Tensor":
        out_data = np.maximum(0, self.data)
        requires_grad = self.requires_grad
        out = Tensor(out_data, requires_grad=requires_grad,
                      _parents=(self,) if requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                grad = out.grad * (self.data > 0).astype(np.float64)
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        # Numerically-stable sigmoid.
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad,
                      _parents=(self,) if self.requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                grad = out.grad * s * (1.0 - s)
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad,
                      _parents=(self,) if self.requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                grad = out.grad * (1.0 - t * t)
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Exp / log
    # ------------------------------------------------------------------
    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad,
                      _parents=(self,) if self.requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                grad = out.grad * e
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out_data = np.log(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                      _parents=(self,) if self.requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                grad = out.grad / self.data
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def cross_entropy(self, targets: Union["Tensor", np.ndarray, list]) -> "Tensor":
        """Softmax cross-entropy loss.

        Parameters
        ----------
        targets : Tensor or array
            Integer class labels (shape ``(batch,)``) OR one-hot encoded
            probabilities (shape ``(batch, num_classes)``).

        Returns
        -------
        Tensor (scalar)
            Mean cross-entropy loss over the batch.
        """
        if isinstance(targets, Tensor):
            targets = targets.data.astype(int)
        targets = np.asarray(targets)
        logits = self.data  # shape (batch, num_classes)

        # Convert integer labels to one-hot if needed.
        if targets.ndim == 1:
            one_hot = np.zeros_like(logits)
            one_hot[np.arange(len(targets)), targets] = 1.0
        else:
            one_hot = targets.astype(np.float64)

        # Numerically-stable log-softmax.
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp  # shape (batch, num_classes)

        # Loss = -mean(sum(one_hot * log_probs, axis=1))
        batch_size = logits.shape[0]
        loss_value = -np.sum(one_hot * log_probs) / batch_size
        out = Tensor(loss_value, requires_grad=self.requires_grad,
                      _parents=(self,) if self.requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                # dL/d(logits) = (softmax(logits) - one_hot) / batch_size
                probs = np.exp(log_probs)  # softmax
                grad = (probs - one_hot) / batch_size
                grad = out.grad * grad  # out.grad is a scalar
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def mse(self, targets: Union["Tensor", np.ndarray, list]) -> "Tensor":
        """Mean squared error loss.

        ``self`` and ``targets`` must have the same shape.
        """
        if isinstance(targets, Tensor):
            targets_arr = targets.data
        else:
            targets_arr = np.asarray(targets, dtype=np.float64)
        diff = self.data - targets_arr
        n = diff.size
        loss_value = np.sum(diff * diff) / n
        out = Tensor(loss_value, requires_grad=self.requires_grad,
                      _parents=(self,) if self.requires_grad else (),
                      _backward=None)

        def _backward():
            if self.requires_grad:
                grad = 2.0 * diff / n * out.grad
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def _accumulate_grad(self, grad: np.ndarray) -> None:
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    def backward(self) -> None:
        """Run reverse-mode autodiff from this Tensor.

        Builds a topological sort of the DAG, then walks it in reverse
        calling each node's ``_backward`` closure. The gradient of this
        Tensor is initialized to 1.0 (scalar) before the walk.
        """
        # Topological sort.
        topo: List["Tensor"] = []
        visited: set = set()

        def build_topo(t: "Tensor"):
            if id(t) in visited:
                return
            visited.add(id(t))
            for parent in t._parents:
                build_topo(parent)
            topo.append(t)

        build_topo(self)

        # Initialize the gradient of the loss to 1.0.
        self.grad = np.ones_like(self.data)

        # Walk in reverse, calling each backward closure.
        for t in reversed(topo):
            t._backward()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def zeros(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def randn(*shape: int, requires_grad: bool = False, seed: Optional[int] = None) -> Tensor:
    if seed is not None:
        rng = np.random.default_rng(seed)
        return Tensor(rng.standard_normal(shape), requires_grad=requires_grad)
    return Tensor(np.random.standard_normal(shape), requires_grad=requires_grad)


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


__all__ = [
    "Tensor",
    "unbroadcast",
    "zeros",
    "randn",
    "tensor",
]
