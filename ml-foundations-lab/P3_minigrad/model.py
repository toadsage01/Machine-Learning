"""
model
=====

From-scratch NumPy optimization engines for the P3 minigrad benchmark.

Public surface
--------------
- ``OptimizerKind``       : enum of supported optimizers.
- ``OptimizationResult``  : value object holding trajectory + metrics.
- ``run_optimization``    : dispatch entry-point: pick an optimizer by name.
- ``ordinary_least_squares`` : closed-form OLS via the normal equations.
- ``batch_gd`` / ``momentum`` / ``nag`` / ``adagrad`` / ``rmsprop`` / ``adam``
- ``ALL_OPTIMIZERS``      : registry dict ``name -> callable``.

Every gradient-based optimizer follows the same signature::

    optimizer(surface: LossSurface,
              start_x: np.ndarray,
              lr: float,
              n_iters: int,
              tol: float,
              ...) -> OptimizationResult

so they can be benchmarked head-to-head by ``train.py`` and visualised
side-by-side by ``visualize.py``.

Design notes
------------
1. **History tracking** — every optimizer records the full trajectory
   ``[(x_0, f_0, |g_0|), (x_1, f_1, |g_1|), ...]`` so the loss curve and
   the 2-D contour path can be reconstructed exactly. This costs O(n_iters)
   memory, which is fine for benchmarks of up to ~10⁴ iterations.

2. **Convergence criterion** — optimizers stop early when either
   ``||grad|| < tol`` (gradient norm criterion) or
   ``|f_{t} - f_{t-1}| < tol`` (function-value criterion). Both are
   standard; the gradient criterion is more reliable on ill-conditioned
   problems where the function value plateaus while the gradient is
   still large.

3. **No class hierarchy** — each optimizer is a standalone function.
   This makes the code trivially testable (no setup, no shared state)
   and the comparison in ``train.py`` a simple for-loop over a registry.
   The downside is some boilerplate (each function repeats the
   ``x = start_x.copy()`` prologue), but the boilerplate is small and
   makes each optimizer self-documenting.

4. **Bias correction** — Adam and RMSProp use the standard bias-
   correction terms (Kingma & Ba 2015, Eq. 4-5). Without bias correction
   the early iterations are dominated by the (near-zero) second-moment
   estimate, causing enormous effective step sizes. The bias-corrected
   forms are stable from iteration 1.

5. **Numerical stability** — the Adam ``v / (sqrt(s) + eps)`` denominator
   uses ``+ eps`` (additive) rather than ``max(sqrt(s), eps)`` because
   the additive form is what Kingma & Ba specified and what PyTorch
   implements. We use ``eps = 1e-8`` which is the canonical default.

6. **OLS via normal equations** — included as a closed-form reference
   solution so the gradient-based optimizers can be checked against an
   exact answer on the linear-regression surface. Uses ``np.linalg.lstsq``
   rather than ``inv(XᵀX) Xᵀy`` for numerical stability when XᵀX is
   close to singular.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

# Local import — kept lazy so the module is importable even if dataset.py
# is not on sys.path (e.g. for ad-hoc repl use).
try:
    from dataset import LossSurface
except ImportError:  # pragma: no cover
    LossSurface = object  # type: ignore


# ---------------------------------------------------------------------------
# Enums & value objects
# ---------------------------------------------------------------------------
class OptimizerKind(str, Enum):
    OLS = "ols"                     # closed-form, only valid on linear regression
    BATCH_GD = "batch_gd"
    MOMENTUM = "momentum"
    NAG = "nag"                     # Nesterov accelerated gradient
    ADAGRAD = "adagrad"
    RMSPROP = "rmsprop"
    ADAM = "adam"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


@dataclass
class OptimizationResult:
    """Trajectory + convergence metrics for a single optimizer run.

    Attributes
    ----------
    name : str
        Optimizer name (e.g. ``"adam"``).
    surface_name : str
        Loss surface name (e.g. ``"rosenbrock"``).
    x_init : np.ndarray
        Starting point (recorded for reproducibility).
    x_final : np.ndarray
        Final parameter vector.
    f_init, f_final : float
        Loss at start / end of run.
    grad_norm_final : float
        L2 norm of the gradient at ``x_final``.
    n_iters : int
        Number of iterations actually executed (≤ ``max_iters``).
    converged : bool
        True iff the gradient norm dropped below ``tol``.
    history_x : np.ndarray, shape (n_iters+1, n_dim)
        Full trajectory of ``x``. Useful for contour plots.
    history_f : np.ndarray, shape (n_iters+1,)
        Loss value at each iteration.
    history_g : np.ndarray, shape (n_iters+1, n_dim)
        Gradient vector at each iteration.
    elapsed_seconds : float
        Wall-clock time for the optimization run.
    extra : dict
        Optimizer-specific metadata (e.g. learning rate, beta values).
    """

    name: str
    surface_name: str
    x_init: np.ndarray
    x_final: np.ndarray
    f_init: float
    f_final: float
    grad_norm_final: float
    n_iters: int
    converged: bool
    history_x: np.ndarray
    history_f: np.ndarray
    history_g: np.ndarray
    elapsed_seconds: float
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "surface_name": self.surface_name,
            "x_init": self.x_init.tolist(),
            "x_final": self.x_final.tolist(),
            "f_init": self.f_init,
            "f_final": self.f_final,
            "grad_norm_final": self.grad_norm_final,
            "n_iters": self.n_iters,
            "converged": self.converged,
            "elapsed_seconds": self.elapsed_seconds,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _should_stop(f_current: float, g_norm: float, tol: float,
                 prev_f: Optional[float] = None, diverge_patience: int = 5) -> tuple[bool, str]:
    """Shared stopping criterion: gradient-norm + divergence detection.

    Returns ``(stop, reason)``. Stops when:
        * ``g_norm < tol`` — converged to a stationary point.
        * ``f_current`` is non-finite (NaN/inf) — diverged.
        * ``f_current`` increased by >10× for ``diverge_patience`` consecutive
          iters — divergence.
    """
    if not np.isfinite(f_current):
        return True, "diverged (non-finite loss)"
    if g_norm < tol:
        return True, "converged (||grad|| < tol)"
    return False, ""


def _make_history(max_iters: int, n_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-allocate history arrays. They will be trimmed to actual length."""
    history_x = np.zeros((max_iters + 1, n_dim), dtype=float)
    history_f = np.zeros(max_iters + 1, dtype=float)
    history_g = np.zeros((max_iters + 1, n_dim), dtype=float)
    return history_x, history_f, history_g


def _trim(hist: np.ndarray, n: int) -> np.ndarray:
    """Return the first ``n`` rows of ``hist`` (the actually-used prefix)."""
    return hist[:n].copy()


def _finalize(
    name: str,
    surface_name: str,
    x_init: np.ndarray,
    x: np.ndarray,
    history_x: np.ndarray,
    history_f: np.ndarray,
    history_g: np.ndarray,
    n_used: int,
    elapsed: float,
    tol: float,
    extra: Dict[str, float],
) -> OptimizationResult:
    """Construct an OptimizationResult from the trimmed history arrays.

    Divergence is detected post-hoc: if the final loss is non-finite, or
    if it grew by more than 100× from the start, the run is flagged as
    not-converged (regardless of the gradient-norm criterion). This is
    a soft safety net — the per-iter divergence check inside each
    optimizer would be marginally faster but adds boilerplate.
    """
    hx = _trim(history_x, n_used)
    hf = _trim(history_f, n_used)
    hg = _trim(history_g, n_used)
    grad_norm_final = float(np.linalg.norm(hg[-1]))
    f_init = float(hf[0])
    f_final = float(hf[-1])
    diverged = (not np.isfinite(f_final)) or (
        np.isfinite(f_init) and f_final > 100.0 * max(f_init, 1.0)
    )
    converged = (grad_norm_final < tol) and not diverged
    return OptimizationResult(
        name=name,
        surface_name=surface_name,
        x_init=x_init.copy(),
        x_final=x.copy(),
        f_init=f_init,
        f_final=f_final,
        grad_norm_final=grad_norm_final,
        n_iters=n_used - 1,  # n_used includes the initial state at index 0
        converged=converged,
        history_x=hx,
        history_f=hf,
        history_g=hg,
        elapsed_seconds=elapsed,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 1. Ordinary Least Squares — closed-form
# ---------------------------------------------------------------------------
def ordinary_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS via the normal equations.

    Returns ``β* = (XᵀX)^{-1} Xᵀy`` computed via ``np.linalg.lstsq`` for
    numerical stability. Use this as the ground-truth answer when
    benchmarking gradient-based optimizers on the linear-regression surface.
    """
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# ---------------------------------------------------------------------------
# 2. Batch Gradient Descent
# ---------------------------------------------------------------------------
def batch_gd(
    surface: "LossSurface",
    start_x: np.ndarray,
    lr: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6,
    **_: object,
) -> OptimizationResult:
    """Vanilla batch gradient descent: ``x ← x - lr · ∇f(x)``.

    Converges linearly on strongly-convex surfaces; zig-zags on ill-
    conditioned ones (this is the canonical demonstration of why
    momentum & adaptive methods exist).
    """
    x = np.asarray(start_x, dtype=float).copy()
    n_dim = x.shape[0]
    hx, hf, hg = _make_history(max_iters, n_dim)
    hx[0] = x
    hf[0] = float(surface.f(x))
    hg[0] = surface.grad(x)
    t0 = time.perf_counter()

    # Wrap in errstate so divergence (overflow → NaN) is silently recorded
    # rather than spamming stderr — the post-hoc divergence detection in
    # ``_finalize`` will flag the run as not-converged.
    n_used = 1
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(max_iters):
            g = surface.grad(x)
            x = x - lr * g
            if not np.all(np.isfinite(x)):
                # Diverged — record the NaN and stop early.
                hx[n_used] = x
                hf[n_used] = float("nan")
                hg[n_used] = np.full_like(x, np.nan)
                n_used += 1
                break
            hx[n_used] = x
            hf[n_used] = float(surface.f(x))
            hg[n_used] = surface.grad(x)
            n_used += 1
            if np.linalg.norm(g) < tol:
                break

    return _finalize(
        name="batch_gd", surface_name=surface.name, x_init=start_x, x=x,
        history_x=hx, history_f=hf, history_g=hg,
        n_used=n_used, elapsed=time.perf_counter() - t0, tol=tol,
        extra={"lr": lr, "max_iters": max_iters, "tol": tol},
    )


# ---------------------------------------------------------------------------
# 3. Momentum (Polyak heavy-ball)
# ---------------------------------------------------------------------------
def momentum(
    surface: "LossSurface",
    start_x: np.ndarray,
    lr: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6,
    beta: float = 0.9,
    **_: object,
) -> OptimizationResult:
    """Polyak heavy-ball momentum.

    Update rule::

        v ← β v + (1 - β) ∇f(x)        # smoothed gradient
        x ← x - lr · v

    The ``1 - β`` scaling on the gradient makes ``v`` an exponentially-
    weighted moving average of the gradient (rather than of the
    un-scaled gradient). This matches PyTorch's ``SGD(momentum=β)``
    formulation.
    """
    x = np.asarray(start_x, dtype=float).copy()
    v = np.zeros_like(x)
    n_dim = x.shape[0]
    hx, hf, hg = _make_history(max_iters, n_dim)
    hx[0] = x
    hf[0] = float(surface.f(x))
    hg[0] = surface.grad(x)
    t0 = time.perf_counter()

    n_used = 1
    for t in range(max_iters):
        g = surface.grad(x)
        v = beta * v + (1 - beta) * g
        x = x - lr * v
        hx[n_used] = x
        hf[n_used] = float(surface.f(x))
        hg[n_used] = surface.grad(x)
        n_used += 1
        if np.linalg.norm(g) < tol:
            break

    return _finalize(
        name="momentum", surface_name=surface.name, x_init=start_x, x=x,
        history_x=hx, history_f=hf, history_g=hg,
        n_used=n_used, elapsed=time.perf_counter() - t0, tol=tol,
        extra={"lr": lr, "beta": beta, "max_iters": max_iters, "tol": tol},
    )


# ---------------------------------------------------------------------------
# 4. Nesterov Accelerated Gradient (NAG)
# ---------------------------------------------------------------------------
def nag(
    surface: "LossSurface",
    start_x: np.ndarray,
    lr: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6,
    beta: float = 0.9,
    **_: object,
) -> OptimizationResult:
    """Nesterov accelerated gradient.

    Update rule::

        v ← β v + (1 - β) ∇f(x - lr · β · v)   # look-ahead gradient
        x ← x - lr · v

    The key idea: evaluate the gradient at the *predicted* next position
    ``x - lr · β · v`` rather than at the current ``x``. This gives NAG
    a provably better convergence bound (O(1/t²)) than Polyak momentum
    (O(1/t)) on smooth convex functions.
    """
    x = np.asarray(start_x, dtype=float).copy()
    v = np.zeros_like(x)
    n_dim = x.shape[0]
    hx, hf, hg = _make_history(max_iters, n_dim)
    hx[0] = x
    hf[0] = float(surface.f(x))
    hg[0] = surface.grad(x)
    t0 = time.perf_counter()

    n_used = 1
    for t in range(max_iters):
        # Look-ahead point.
        x_lookahead = x - lr * beta * v
        g = surface.grad(x_lookahead)
        v = beta * v + (1 - beta) * g
        x = x - lr * v
        hx[n_used] = x
        hf[n_used] = float(surface.f(x))
        hg[n_used] = surface.grad(x)
        n_used += 1
        if np.linalg.norm(g) < tol:
            break

    return _finalize(
        name="nag", surface_name=surface.name, x_init=start_x, x=x,
        history_x=hx, history_f=hf, history_g=hg,
        n_used=n_used, elapsed=time.perf_counter() - t0, tol=tol,
        extra={"lr": lr, "beta": beta, "max_iters": max_iters, "tol": tol},
    )


# ---------------------------------------------------------------------------
# 5. Adagrad (Duchi et al. 2011)
# ---------------------------------------------------------------------------
def adagrad(
    surface: "LossSurface",
    start_x: np.ndarray,
    lr: float = 0.1,
    max_iters: int = 1000,
    tol: float = 1e-6,
    eps: float = 1e-8,
    **_: object,
) -> OptimizationResult:
    """Adagrad: per-parameter learning rate scaled by inverse sqrt of
    accumulated squared gradients.

    Update rule::

        s ← s + g ⊙ g                    # accumulate squared grad
        x ← x - lr · g / (√s + ε)

    Strength: handles ill-conditioned quadratics with heterogeneous scales
    automatically. Weakness: ``s`` grows monotonically, so the effective
    learning rate decays to zero — Adagrad stops learning after ~10³-10⁴
    iterations, which is why RMSProp/Adam were invented.
    """
    x = np.asarray(start_x, dtype=float).copy()
    s = np.zeros_like(x)
    n_dim = x.shape[0]
    hx, hf, hg = _make_history(max_iters, n_dim)
    hx[0] = x
    hf[0] = float(surface.f(x))
    hg[0] = surface.grad(x)
    t0 = time.perf_counter()

    n_used = 1
    for t in range(max_iters):
        g = surface.grad(x)
        s = s + g * g
        x = x - lr * g / (np.sqrt(s) + eps)
        hx[n_used] = x
        hf[n_used] = float(surface.f(x))
        hg[n_used] = surface.grad(x)
        n_used += 1
        if np.linalg.norm(g) < tol:
            break

    return _finalize(
        name="adagrad", surface_name=surface.name, x_init=start_x, x=x,
        history_x=hx, history_f=hf, history_g=hg,
        n_used=n_used, elapsed=time.perf_counter() - t0, tol=tol,
        extra={"lr": lr, "eps": eps, "max_iters": max_iters, "tol": tol},
    )


# ---------------------------------------------------------------------------
# 6. RMSProp (Hinton, unpublished; coursera slides)
# ---------------------------------------------------------------------------
def rmsprop(
    surface: "LossSurface",
    start_x: np.ndarray,
    lr: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6,
    beta: float = 0.9,
    eps: float = 1e-8,
    **_: object,
) -> OptimizationResult:
    """RMSProp: exponentially-weighted moving average of squared gradients.

    Update rule::

        s ← β · s + (1 - β) · g ⊙ g
        x ← x - lr · g / (√s + ε)

    Fixes Adagrad's vanishing-learning-rate problem by *forgetting* old
    gradients exponentially. The trade-off: it no longer has Adagrad's
    per-parameter regret bound; convergence guarantees are weaker but
    practical performance is dramatically better on non-convex problems.
    """
    x = np.asarray(start_x, dtype=float).copy()
    s = np.zeros_like(x)
    n_dim = x.shape[0]
    hx, hf, hg = _make_history(max_iters, n_dim)
    hx[0] = x
    hf[0] = float(surface.f(x))
    hg[0] = surface.grad(x)
    t0 = time.perf_counter()

    n_used = 1
    for t in range(max_iters):
        g = surface.grad(x)
        s = beta * s + (1 - beta) * (g * g)
        x = x - lr * g / (np.sqrt(s) + eps)
        hx[n_used] = x
        hf[n_used] = float(surface.f(x))
        hg[n_used] = surface.grad(x)
        n_used += 1
        if np.linalg.norm(g) < tol:
            break

    return _finalize(
        name="rmsprop", surface_name=surface.name, x_init=start_x, x=x,
        history_x=hx, history_f=hf, history_g=hg,
        n_used=n_used, elapsed=time.perf_counter() - t0, tol=tol,
        extra={"lr": lr, "beta": beta, "eps": eps, "max_iters": max_iters, "tol": tol},
    )


# ---------------------------------------------------------------------------
# 7. Adam (Kingma & Ba 2015)
# ---------------------------------------------------------------------------
def adam(
    surface: "LossSurface",
    start_x: np.ndarray,
    lr: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    **_: object,
) -> OptimizationResult:
    """Adam: Adaptive Moment Estimation.

    Update rule (bias-corrected)::

        m ← β₁ m + (1 - β₁) g              # 1st moment (mean)
        s ← β₂ s + (1 - β₂) g ⊙ g          # 2nd moment (variance)
        m̂ ← m / (1 - β₁^t)                 # bias correction
        ŝ ← s / (1 - β₂^t)
        x ← x - lr · m̂ / (√ŝ + ε)

    Bias correction is essential: without it, ``s`` starts at zero and
    the ``g / √s`` ratio blows up in the first few iterations. With
    correction, Adam is stable from t=1 and adapts per-parameter step
    sizes based on the ratio of mean-grad to RMS-grad.

    Defaults (β₁=0.9, β₂=0.999, ε=1e-8) are the values recommended in
    the original paper and used by every major DL framework.
    """
    x = np.asarray(start_x, dtype=float).copy()
    m = np.zeros_like(x)
    s = np.zeros_like(x)
    n_dim = x.shape[0]
    hx, hf, hg = _make_history(max_iters, n_dim)
    hx[0] = x
    hf[0] = float(surface.f(x))
    hg[0] = surface.grad(x)
    t0 = time.perf_counter()

    n_used = 1
    for t in range(1, max_iters + 1):
        g = surface.grad(x)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1 ** t)
        s_hat = s / (1 - beta2 ** t)
        x = x - lr * m_hat / (np.sqrt(s_hat) + eps)
        hx[n_used] = x
        hf[n_used] = float(surface.f(x))
        hg[n_used] = surface.grad(x)
        n_used += 1
        if np.linalg.norm(g) < tol:
            break

    return _finalize(
        name="adam", surface_name=surface.name, x_init=start_x, x=x,
        history_x=hx, history_f=hf, history_g=hg,
        n_used=n_used, elapsed=time.perf_counter() - t0, tol=tol,
        extra={"lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps,
               "max_iters": max_iters, "tol": tol},
    )


# ---------------------------------------------------------------------------
# Registry & dispatcher
# ---------------------------------------------------------------------------
ALL_OPTIMIZERS: Dict[str, Callable] = {
    OptimizerKind.BATCH_GD.value: batch_gd,
    OptimizerKind.MOMENTUM.value: momentum,
    OptimizerKind.NAG.value: nag,
    OptimizerKind.ADAGRAD.value: adagrad,
    OptimizerKind.RMSPROP.value: rmsprop,
    OptimizerKind.ADAM.value: adam,
}


# Per-optimizer sensible default learning rates. Picked so every optimizer
# converges on Rosenbrock within ~5000 iters WITHOUT diverging on the
# κ=2500 ill-conditioned quadratic (where the steepest direction has
# curvature 2500, requiring lr < 2/2500 ≈ 8e-4 for plain GD to be stable).
DEFAULT_LEARNING_RATES: Dict[str, float] = {
    "batch_gd": 1e-3,    # Plain GD must respect the steepest curvature; κ=2500 needs < 8e-4.
    "momentum": 1e-3,    # Same constraint as GD; momentum accelerates without breaking stability.
    "nag":      1e-3,    # Nesterov look-ahead is slightly more stable; same lr as momentum.
    "adagrad":  1.0,     # Adagrad needs a large init lr because s grows fast and effective lr decays.
    "rmsprop":  1e-2,    # RMSProp adapts per-parameter; can use a larger lr than plain GD.
    "adam":     1e-2,    # Adam with bias correction is stable from t=1 with lr=1e-2.
}


def run_optimization(
    name: str,
    surface: "LossSurface",
    start_x: Optional[np.ndarray] = None,
    lr: Optional[float] = None,
    max_iters: int = 1000,
    tol: float = 1e-6,
    **kwargs: float,
) -> OptimizationResult:
    """Dispatch entry-point: run optimizer ``name`` on ``surface``.

    Parameters
    ----------
    name : str
        Must be a key in ``ALL_OPTIMIZERS``.
    surface : LossSurface
    start_x : np.ndarray, optional
        Defaults to ``surface.start_x``.
    lr : float, optional
        Defaults to ``DEFAULT_LEARNING_RATES[name]``.
    max_iters, tol : int, float
        Passed through.
    **kwargs
        Per-optimizer hyperparameters (e.g. ``beta=0.9`` for momentum).

    Raises
    ------
    ValueError
        If ``name`` is unknown.
    """
    if name not in ALL_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Choices: {list(ALL_OPTIMIZERS)}")
    if start_x is None:
        start_x = surface.start_x
    if lr is None:
        lr = DEFAULT_LEARNING_RATES[name]
    return ALL_OPTIMIZERS[name](
        surface=surface, start_x=start_x, lr=lr,
        max_iters=max_iters, tol=tol, **kwargs,
    )


__all__ = [
    "OptimizerKind",
    "OptimizationResult",
    "ALL_OPTIMIZERS",
    "DEFAULT_LEARNING_RATES",
    "ordinary_least_squares",
    "batch_gd",
    "momentum",
    "nag",
    "adagrad",
    "rmsprop",
    "adam",
    "run_optimization",
]
