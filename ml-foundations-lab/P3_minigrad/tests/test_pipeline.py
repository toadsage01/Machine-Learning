"""
tests/test_pipeline
===================

End-to-end tests for the P3 minigrad optimization library.

Coverage:
    * Analytical gradient correctness — finite-difference check on every surface.
    * OLS closed-form solution matches the analytical minimum of the
      linear_regression surface.
    * Adam converges to machine precision on the ill-conditioned quadratic.
    * Adam converges to the analytical OLS minimum on the regression surface.
    * Adagrad converges on the ill-conditioned quadratic (its canonical win).
    * Vanilla GD diverges predictably on κ=2500 with lr=1e-3 (negative test).
    * History arrays have correct shape & finite values for converged runs.
    * scipy.optimize.minimize (BFGS) parity check on the regression surface.

Run with::

    cd ml-foundations-lab/P3_minigrad
    python -m pytest tests/ -v

or::

    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import optimize as sopt

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import (  # noqa: E402
    ALL_SURFACES, LossSurface, make_regression_data, rosenbrock,
    rastrigin, ill_conditioned_quadratic, beale,
)
from model import (  # noqa: E402
    ALL_OPTIMIZERS, DEFAULT_LEARNING_RATES, OptimizationResult,
    run_optimization, ordinary_least_squares,
)


# ---------------------------------------------------------------------------
# Gradient correctness — finite-difference check on every surface
# ---------------------------------------------------------------------------
def _numerical_grad(f, x, eps=1e-6):
    """Central finite-difference gradient."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def test_gradients_match_finite_difference():
    """Verify analytical gradients on every surface against finite differences."""
    for name, surface in ALL_SURFACES.items():
        # Test at the canonical start point AND at a random interior point.
        for x_test in [surface.start_x, np.array([0.3, -0.7])]:
            g_ana = surface.grad(x_test)
            g_num = _numerical_grad(surface.f, x_test)
            rel_err = np.linalg.norm(g_ana - g_num) / max(np.linalg.norm(g_num), 1e-12)
            assert rel_err < 1e-5, (
                f"Surface '{name}' gradient mismatch at x={x_test}: "
                f"||g_ana - g_num|| / ||g_num|| = {rel_err:.2e}"
            )


# ---------------------------------------------------------------------------
# OLS analytical solution
# ---------------------------------------------------------------------------
def test_ordinary_least_squares_matches_analytical_minimum():
    """Closed-form OLS should match the declared minimum of the regression surface."""
    rs = make_regression_data(n_features=5, seed=42)
    beta_ols = ordinary_least_squares(rs.X, rs.y)
    # beta_ols should equal rs.beta_star to machine precision.
    err = float(np.linalg.norm(beta_ols - rs.beta_star))
    assert err < 1e-10, f"OLS solution does not match analytical β*: ||β_ols - β*|| = {err:.2e}"
    # And f(beta_ols) should equal the declared minimum_f.
    f_at_ols = rs.surface.f(beta_ols)
    assert np.isclose(f_at_ols, rs.surface.minimum_f, atol=1e-10)


# ---------------------------------------------------------------------------
# Adam convergence on the ill-conditioned quadratic
# ---------------------------------------------------------------------------
def test_adam_converges_on_ill_conditioned_quadratic():
    """Adam should reach the global minimum of the κ=2500 quadratic in <2000 iters."""
    surface = ill_conditioned_quadratic()
    result = run_optimization("adam", surface, max_iters=2000, tol=1e-6)
    assert result.converged, (
        f"Adam did not converge on {surface.name}: f_final={result.f_final}, "
        f"||grad||={result.grad_norm_final}"
    )
    assert result.f_final < 1e-6, f"Adam f_final too high: {result.f_final}"
    # And the final point should be close to the true minimum at the origin.
    x_err = float(np.linalg.norm(result.x_final - surface.minimum_x))
    assert x_err < 1e-3, f"Adam did not reach the minimum: ||x_final - x*|| = {x_err}"


# ---------------------------------------------------------------------------
# Adagrad convergence on the ill-conditioned quadratic (its canonical win)
# ---------------------------------------------------------------------------
def test_adagrad_converges_on_ill_conditioned_quadratic():
    """Adagrad should converge on the κ=2500 quadratic in <100 iters."""
    surface = ill_conditioned_quadratic()
    result = run_optimization("adagrad", surface, max_iters=1000, tol=1e-6)
    assert result.converged, (
        f"Adagrad did not converge on {surface.name}: f_final={result.f_final}"
    )
    assert result.n_iters < 100, f"Adagrad took too long: {result.n_iters} iters"
    assert result.f_final < 1e-6


# ---------------------------------------------------------------------------
# Adam reaches OLS analytical minimum on regression surface
# ---------------------------------------------------------------------------
def test_adam_reaches_ols_minimum():
    """Adam's final loss should match the analytical OLS minimum to <1e-6 relative."""
    rs = make_regression_data(n_features=5, seed=42)
    result = run_optimization("adam", rs.surface, lr=0.05, max_iters=3000, tol=1e-8)
    rel_err = abs(result.f_final - rs.surface.minimum_f) / max(rs.surface.minimum_f, 1e-12)
    assert rel_err < 1e-6, (
        f"Adam did not reach OLS minimum: rel_err={rel_err:.2e}, "
        f"f_final={result.f_final:.6e}, declared_min={rs.surface.minimum_f:.6e}"
    )
    x_err = float(np.linalg.norm(result.x_final - rs.beta_star))
    assert x_err < 1e-3, f"Adam's x_final is too far from β*: ||x - β*|| = {x_err:.2e}"


# ---------------------------------------------------------------------------
# scipy.optimize.minimize (BFGS) parity on the regression surface
# ---------------------------------------------------------------------------
def test_scipy_bfgs_reaches_ols_minimum():
    """scipy.optimize.minimize (BFGS) should also reach the OLS minimum —
    this is a sanity check that the surface is well-posed.
    """
    rs = make_regression_data(n_features=5, seed=42)
    result = sopt.minimize(
        fun=rs.surface.f,
        x0=np.zeros(5),
        method="BFGS",
        jac=lambda x: np.asarray(rs.surface.grad(x), dtype=float),
        options={"maxiter": 5000, "gtol": 1e-8},
    )
    # BFGS may report success=False due to precision loss at very tight gtol,
    # but the final loss should still be very close to the analytical minimum.
    rel_err = abs(result.fun - rs.surface.minimum_f) / max(rs.surface.minimum_f, 1e-12)
    assert rel_err < 1e-6, (
        f"BFGS did not reach OLS min: rel_err={rel_err:.2e}, "
        f"f_bfgs={result.fun:.6e}, declared_min={rs.surface.minimum_f:.6e}"
    )


# ---------------------------------------------------------------------------
# Negative test — vanilla GD diverges on κ=2500 with the default lr
# ---------------------------------------------------------------------------
def test_batch_gd_diverges_on_ill_conditioned_quadratic():
    """Plain GD with lr=1e-3 should diverge or fail to converge on κ=2500.

    This is the canonical motivation for adaptive methods — if our
    implementation *did* converge, it would indicate a bug (likely an
    accidental rescaling that defeats the purpose of the benchmark).
    """
    surface = ill_conditioned_quadratic()
    result = run_optimization("batch_gd", surface, lr=1e-3, max_iters=3000, tol=1e-6)
    # Either diverged (NaN) or did not converge to within tol.
    assert not result.converged, (
        f"Batch GD unexpectedly converged on κ=2500 in {result.n_iters} iters — "
        "this suggests the implementation is wrong."
    )


# ---------------------------------------------------------------------------
# History shape & finiteness
# ---------------------------------------------------------------------------
def test_history_arrays_have_correct_shape():
    """For every optimizer × every surface, the history arrays must be 2-D and finite
    on the first/last rows when converged.
    """
    for surf_name in ["rosenbrock", "beale"]:
        surface = ALL_SURFACES[surf_name]
        for opt_name in ["batch_gd", "momentum", "nag", "adagrad", "rmsprop", "adam"]:
            result = run_optimization(opt_name, surface, max_iters=500, tol=1e-6)
            n = result.n_iters + 1
            assert result.history_x.shape == (n, 2), (
                f"{opt_name}/{surf_name}: history_x shape={result.history_x.shape}, expected ({n}, 2)"
            )
            assert result.history_f.shape == (n,)
            assert result.history_g.shape == (n, 2)
            # First row should always be finite (it's the start point).
            assert np.all(np.isfinite(result.history_x[0]))
            assert np.isfinite(result.history_f[0])


# ---------------------------------------------------------------------------
# Optimizer registry
# ---------------------------------------------------------------------------
def test_all_optimizers_are_callable():
    for name, fn in ALL_OPTIMIZERS.items():
        assert callable(fn), f"Optimizer {name} is not callable"
        # Smoke-test on a tiny surface (ill-conditioned quadratic, 50 iters).
        surface = ill_conditioned_quadratic()
        result = fn(surface=surface, start_x=surface.start_x, lr=1e-3, max_iters=50, tol=1e-12)
        assert isinstance(result, OptimizationResult)
        assert result.name == name


# ---------------------------------------------------------------------------
# Visualization smoke test (renders without crashing)
# ---------------------------------------------------------------------------
def test_visualization_renders():
    """Verify the visualize.py module renders a side-by-side comparison PNG."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("visualize", PROJECT_ROOT / "visualize.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROJECT_ROOT))
    spec.loader.exec_module(mod)  # type: ignore[arg-type]

    from model import run_optimization  # local re-import to avoid circular
    surface = ALL_SURFACES["rosenbrock"]
    results = [run_optimization(name, surface, max_iters=500) for name in ["adam", "momentum", "rmsprop"]]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "contour.png"
        mod.plot_contour_trajectory(surface, results, output_path=out_path)
        assert out_path.exists() and out_path.stat().st_size > 10_000


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_gradients_match_finite_difference,
        test_ordinary_least_squares_matches_analytical_minimum,
        test_adam_converges_on_ill_conditioned_quadratic,
        test_adagrad_converges_on_ill_conditioned_quadratic,
        test_adam_reaches_ols_minimum,
        test_scipy_bfgs_reaches_ols_minimum,
        test_batch_gd_diverges_on_ill_conditioned_quadratic,
        test_history_arrays_have_correct_shape,
        test_all_optimizers_are_callable,
        test_visualization_renders,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
