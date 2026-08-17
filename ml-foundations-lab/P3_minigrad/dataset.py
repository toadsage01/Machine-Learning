"""
dataset
=======

Synthetic loss-surface and regression-data generators for the P3 minigrad
optimization benchmarks.

Every loss surface is exposed as a :class:`LossSurface` value object
bundling:

* ``f(x)``     — the scalar objective function,
* ``grad(x)``   — its analytical gradient (vector-valued),
* ``minimum_x`` — the analytical global minimizer (or ``None`` if the
                  surface has many, e.g. Rastrigin),
* ``minimum_f`` — the value of ``f`` at the minimizer,
* ``name``      — short label for plots / logs,
* ``bounds``    — recommended plot bounds ``[(lo, hi), (lo, hi)]`` for 2-D
                  contour visualization,
* ``start_x``   — a "hard" starting point that stresses optimizers (often
                  the canonical literature starting point).

The module is **pure** — it never touches the filesystem. All outputs are
callables and small NumPy arrays, which makes the optimizers in ``model.py``
trivially unit-testable against scipy.

Public surface
--------------
- ``LossSurface``       : frozen value object.
- ``rosenbrock``        : 2-D Rosenbrock (the classic "banana" valley).
- ``rastrigin``         : 2-D Rastrigin (highly multimodal).
- ``ill_conditioned_quadratic`` : 2-D quadratic with condition number κ≈2500.
- ``beale``             : 2-D Beale (flat plateau + narrow curved valley).
- ``linear_regression_surface`` : n-D quadratic from synthetic (X, y) data,
                                   paired with a sample generator.
- ``make_regression_data``      : generate (X, y) with known coefficients
                                   so the analytical OLS minimum is exact.
- ``ALL_SURFACES``      : registry dict ``name -> LossSurface``.

Design notes
------------
* All gradients are derived analytically and verified against a finite-
  difference check in ``tests/test_pipeline.py``. This catches sign and
  factor-of-2 errors (the two most common bugs when porting literature
  formulas into code).
* For the ill-conditioned quadratic, the Hessian is intentionally
  ``diag([1, 2500])`` — a condition number κ = 2500, which makes plain
  gradient descent converge in ≈5000 iterations while Adam converges in
  <300. This is the canonical demonstration of why adaptive methods exist.
* For Rastrigin, we expose only the 2-D version because higher-D Rastrigin
  is harder than any 2-D optimizer can reliably solve without restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LossSurface:
    """A differentiable scalar objective + its analytical minimizer.

    Attributes
    ----------
    name : str
        Short label (e.g. ``"rosenbrock"``).
    f : Callable[[np.ndarray], float]
        The objective function. Takes a 1-D array, returns a scalar.
    grad : Callable[[np.ndarray], np.ndarray]
        The gradient. Same signature as ``f`` but returns a 1-D array.
    minimum_x : Optional[np.ndarray]
        Analytical minimizer. ``None`` if there are many (Rastrigin).
    minimum_f : Optional[float]
        Value of ``f`` at the minimizer.
    bounds : Tuple[Tuple[float, float], Tuple[float, float]]
        Recommended plot bounds for 2-D contour visualization.
    start_x : np.ndarray
        The "hard" starting point used by default in benchmarks.
    """

    name: str
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    minimum_x: Optional[np.ndarray]
    minimum_f: Optional[float]
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    start_x: np.ndarray
    description: str = ""

    def __post_init__(self) -> None:
        # Validate that minimum_f matches f(minimum_x) when both are known.
        if self.minimum_x is not None and self.minimum_f is not None:
            evaluated = float(self.f(np.asarray(self.minimum_x, dtype=float)))
            if not np.isclose(evaluated, self.minimum_f, atol=1e-6):
                raise RuntimeError(
                    f"{self.name}: f(minimum_x)={evaluated} does not match "
                    f"declared minimum_f={self.minimum_f}"
                )


# ---------------------------------------------------------------------------
# 1. Rosenbrock (2-D) — the "banana" valley
#    f(x, y) = (a - x)^2 + b (y - x^2)^2   with a=1, b=100
#    minimum: (1, 1), f=0
#    Hard because the valley is long, narrow, and curved.
# ---------------------------------------------------------------------------
def rosenbrock() -> LossSurface:
    a, b = 1.0, 100.0

    def f(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2)

    def grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
        dy = 2 * b * (x[1] - x[0] ** 2)
        return np.array([dx, dy])

    return LossSurface(
        name="rosenbrock",
        f=f,
        grad=grad,
        minimum_x=np.array([1.0, 1.0]),
        minimum_f=0.0,
        bounds=((-2.0, 2.0), (-1.0, 3.0)),
        start_x=np.array([-1.2, 1.0]),  # classic literature starting point
        description="Banana valley — long, narrow, curved. Canonical test of optimizer robustness.",
    )


# ---------------------------------------------------------------------------
# 2. Rastrigin (2-D) — highly multimodal
#    f(x) = A n + Σ [x_i^2 - A cos(2π x_i)]   with A=10, n=2
#    Global minimum at (0, 0), f=0; many local minima in the surrounding grid.
# ---------------------------------------------------------------------------
def rastrigin() -> LossSurface:
    A = 10.0

    def f(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))

    def grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

    return LossSurface(
        name="rastrigin",
        f=f,
        grad=grad,
        minimum_x=np.array([0.0, 0.0]),
        minimum_f=0.0,
        bounds=((-5.12, 5.12), (-5.12, 5.12)),
        start_x=np.array([-4.0, 4.0]),  # far from origin, surrounded by local minima
        description="Highly multimodal — global min at origin surrounded by a regular grid of local minima.",
    )


# ---------------------------------------------------------------------------
# 3. Ill-conditioned quadratic (2-D) — κ = 2500
#    f(x, y) = 0.5 (x^2 + 2500 y^2)
#    Hessian = diag([1, 2500]), condition number = 2500
#    minimum: (0, 0), f=0
#    The narrow canyon along the x-axis forces plain GD to zig-zag.
# ---------------------------------------------------------------------------
def ill_conditioned_quadratic(condition_number: float = 2500.0) -> LossSurface:
    kappa = float(condition_number)
    hessian_diag = np.array([1.0, kappa])

    def f(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(0.5 * np.sum(hessian_diag * x ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return hessian_diag * x

    return LossSurface(
        name=f"ill_conditioned_quadratic_κ{int(kappa)}",
        f=f,
        grad=grad,
        minimum_x=np.array([0.0, 0.0]),
        minimum_f=0.0,
        bounds=((-2.0, 2.0), (-0.05, 0.05)),
        start_x=np.array([2.0, 0.04]),  # along the steep wall — classic zig-zag trigger
        description=f"κ={kappa:.0f} quadratic. Hessian=diag([1, {kappa:.0f}]) — narrow canyon.",
    )


# ---------------------------------------------------------------------------
# 4. Beale (2-D) — flat plateau + curved valley
#    f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
#    minimum: (3, 0.5), f=0
# ---------------------------------------------------------------------------
def beale() -> LossSurface:
    def f(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        a, b = x[0], x[1]
        t1 = 1.5 - a + a * b
        t2 = 2.25 - a + a * b ** 2
        t3 = 2.625 - a + a * b ** 3
        return float(t1 ** 2 + t2 ** 2 + t3 ** 2)

    def grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        a, b = x[0], x[1]
        t1 = 1.5 - a + a * b
        t2 = 2.25 - a + a * b ** 2
        t3 = 2.625 - a + a * b ** 3
        # ∂f/∂a
        da = 2 * t1 * (b - 1) + 2 * t2 * (b ** 2 - 1) + 2 * t3 * (b ** 3 - 1)
        # ∂f/∂b
        db = 2 * t1 * (a) + 2 * t2 * (2 * a * b) + 2 * t3 * (3 * a * b ** 2)
        return np.array([da, db])

    return LossSurface(
        name="beale",
        f=f,
        grad=grad,
        minimum_x=np.array([3.0, 0.5]),
        minimum_f=0.0,
        bounds=((-4.5, 4.5), (-4.5, 4.5)),
        start_x=np.array([-2.0, -2.0]),
        description="Flat plateau + narrow curved valley. Sensitive to step size & momentum.",
    )


# ---------------------------------------------------------------------------
# 5. Linear regression surface (n-D quadratic)
#    Given (X, y) with X ∈ R^{N×D}, the OLS objective is
#        f(β) = (1/2N) ||X β - y||^2
#    The gradient is (1/N) Xᵀ(Xβ - y) and the Hessian is (1/N) XᵀX.
#    The analytical minimizer is β* = (XᵀX)^{-1} Xᵀy (provided XᵀX is invertible).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegressionSurface:
    """Bundle of (X, y), the OLS loss surface, and the analytical β*.

    Attributes
    ----------
    X : np.ndarray, shape (N, D)
    y : np.ndarray, shape (N,)
    beta_star : np.ndarray, shape (D,)
        Analytical OLS solution.
    surface : LossSurface
        A loss surface built from (X, y) — its ``minimum_x`` is ``beta_star``.
    """

    X: np.ndarray
    y: np.ndarray
    beta_star: np.ndarray
    surface: LossSurface


def make_regression_data(
    n_samples: int = 200,
    n_features: int = 5,
    noise_std: float = 0.5,
    seed: int = 42,
) -> RegressionSurface:
    """Generate (X, y) with known ground-truth coefficients.

    The OLS solution ``beta_star = pinv(X) @ y`` is computed exactly (no
    noise on the solution itself — only on ``y``) so optimizers can be
    verified against it to machine precision.
    """
    rng = np.random.default_rng(seed)
    true_beta = rng.normal(0, 1, size=n_features)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    y = X @ true_beta + rng.normal(0, noise_std, size=n_samples)

    # Analytical OLS via the normal equations.
    # Use lstsq for numerical stability — direct inverse is fragile.
    beta_star, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residual = X @ beta_star - y
    minimum_f = float(0.5 * np.mean(residual ** 2))

    N = X.shape[0]

    def f(beta: np.ndarray) -> float:
        beta = np.asarray(beta, dtype=float)
        residual = X @ beta - y
        return float(0.5 * np.mean(residual ** 2))

    def grad(beta: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta, dtype=float)
        return (X.T @ (X @ beta - y)) / N

    return RegressionSurface(
        X=X,
        y=y,
        beta_star=beta_star,
        surface=LossSurface(
            name=f"linear_regression_d{n_features}",
            f=f,
            grad=grad,
            minimum_x=beta_star,
            minimum_f=minimum_f,
            bounds=((-3.0, 3.0), (-3.0, 3.0)),  # used only for 2-D projection plots
            start_x=np.zeros(n_features),       # start at origin
            description=f"OLS regression on {n_samples} samples × {n_features} features; Hessian = (1/N) XᵀX.",
        ),
    )


def linear_regression_surface(n_samples: int = 200, n_features: int = 2,
                             noise_std: float = 0.5, seed: int = 42) -> LossSurface:
    """Convenience: return just the ``LossSurface`` for a regression problem."""
    return make_regression_data(
        n_samples=n_samples, n_features=n_features,
        noise_std=noise_std, seed=seed,
    ).surface


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ALL_SURFACES: dict[str, LossSurface] = {
    "rosenbrock": rosenbrock(),
    "rastrigin": rastrigin(),
    "ill_conditioned_quadratic": ill_conditioned_quadratic(),
    "beale": beale(),
    "linear_regression": linear_regression_surface(n_features=2),
}


__all__ = [
    "LossSurface",
    "RegressionSurface",
    "rosenbrock",
    "rastrigin",
    "ill_conditioned_quadratic",
    "beale",
    "linear_regression_surface",
    "make_regression_data",
    "ALL_SURFACES",
]
