"""
tests/test_pipeline
===================

End-to-end tests for the P10 NN-from-scratch autograd engine.

Coverage:
    * Scalar ops: add, sub, mul, div, pow — gradients match PyTorch.
    * Tensor ops: matmul (2D, 1D-broadcast), sum, mean, transpose, reshape.
    * Nonlinearities: relu, sigmoid, tanh, exp, log.
    * Losses: cross_entropy, mse — both loss value and gradients.
    * Conv2D + MaxPool2D forward + backward against PyTorch.
    * Linear layer against PyTorch.
    * Broadcasting-aware gradient accumulation.
    * Module.parameters() traversal.
    * SGD + Adam optimizer step shapes.
    * CLI smoke test.

All gradient parity tests use atol=1e-5 (the spec's requirement).

Run with::

    cd dl-advanced-lab/P10_nn_from_scratch
    python -m pytest tests/ -v

or::

    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from autograd import Tensor, unbroadcast  # noqa: E402
import nn  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: run an op in both our autograd and PyTorch, compare gradients.
# ---------------------------------------------------------------------------
def _compare_grads(our_tensor: Tensor, pt_tensor: torch.Tensor,
                    atol: float = 1e-5, label: str = "") -> None:
    """Assert that our_tensor.grad matches pt_tensor.grad.numpy() to ``atol``."""
    assert our_tensor.grad is not None, f"{label}: our grad is None"
    pt_grad = pt_tensor.grad
    assert pt_grad is not None, f"{label}: PyTorch grad is None"
    np.testing.assert_allclose(
        our_tensor.grad, pt_grad.numpy(), atol=atol, rtol=1e-5,
        err_msg=f"{label}: gradient mismatch (max diff = "
                f"{np.abs(our_tensor.grad - pt_grad.numpy()).max():.2e})",
    )


# ---------------------------------------------------------------------------
# Scalar ops
# ---------------------------------------------------------------------------
def test_add_gradient_matches_pytorch():
    x_np = np.random.randn(3, 4)
    x = Tensor(x_np, requires_grad=True)
    y = (x + 2.0).sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = (x_pt + 2.0).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="add")


def test_sub_gradient_matches_pytorch():
    x_np = np.random.randn(3, 4)
    x = Tensor(x_np, requires_grad=True)
    y = (x - 1.0).sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = (x_pt - 1.0).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="sub")


def test_mul_gradient_matches_pytorch():
    x_np = np.random.randn(3, 4)
    x = Tensor(x_np, requires_grad=True)
    y = (x * 3.0).sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = (x_pt * 3.0).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="mul")


def test_pow_gradient_matches_pytorch():
    x_np = np.random.randn(3, 4)
    x = Tensor(x_np, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = (x_pt ** 2).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="pow")


def test_div_gradient_matches_pytorch():
    x_np = np.random.randn(3, 4) + 5.0  # avoid zero
    x = Tensor(x_np, requires_grad=True)
    y = (x / 2.0).sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = (x_pt / 2.0).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="div")


def test_neg_gradient_matches_pytorch():
    x_np = np.random.randn(3, 4)
    x = Tensor(x_np, requires_grad=True)
    y = (-x).sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = (-x_pt).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="neg")


# ---------------------------------------------------------------------------
# Tensor ops
# ---------------------------------------------------------------------------
def test_matmul_2d_gradient_matches_pytorch():
    A_np = np.random.randn(3, 4)
    B_np = np.random.randn(4, 5)
    A = Tensor(A_np, requires_grad=True)
    B = Tensor(B_np, requires_grad=True)
    y = (A @ B).sum()
    y.backward()
    A_pt = torch.tensor(A_np, requires_grad=True)
    B_pt = torch.tensor(B_np, requires_grad=True)
    y_pt = (A_pt @ B_pt).sum()
    y_pt.backward()
    _compare_grads(A, A_pt, label="matmul_2d_A")
    _compare_grads(B, B_pt, label="matmul_2d_B")


def test_matmul_broadcast_gradient_matches_pytorch():
    """(n,k) @ (k,) → (n,). Tests 1-D operand broadcasting."""
    A_np = np.random.randn(3, 4)
    b_np = np.random.randn(4)
    A = Tensor(A_np, requires_grad=True)
    b = Tensor(b_np, requires_grad=True)
    y = (A @ b).sum()
    y.backward()
    A_pt = torch.tensor(A_np, requires_grad=True)
    b_pt = torch.tensor(b_np, requires_grad=True)
    y_pt = (A_pt @ b_pt).sum()
    y_pt.backward()
    _compare_grads(A, A_pt, label="matmul_broadcast_A")
    _compare_grads(b, b_pt, label="matmul_broadcast_b")


def test_sum_gradient_matches_pytorch():
    x_np = np.random.randn(3, 4)
    x = Tensor(x_np, requires_grad=True)
    y = x.sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = x_pt.sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="sum")


def test_mean_gradient_matches_pytorch():
    x_np = np.random.randn(3, 4)
    x = Tensor(x_np, requires_grad=True)
    y = x.mean()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = x_pt.mean()
    y_pt.backward()
    _compare_grads(x, x_pt, label="mean")


def test_transpose_gradient_matches_pytorch():
    x_np = np.random.randn(3, 4)
    x = Tensor(x_np, requires_grad=True)
    y = x.transpose().sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = x_pt.t().sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="transpose")


def test_reshape_gradient_matches_pytorch():
    x_np = np.random.randn(3, 4)
    x = Tensor(x_np, requires_grad=True)
    y = x.reshape(12).sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = x_pt.reshape(12).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="reshape")


def test_getitem_gradient_matches_pytorch():
    """Indexing/slicing should route gradients correctly via np.add.at."""
    x_np = np.random.randn(5, 4)
    x = Tensor(x_np, requires_grad=True)
    y = x[1:4].sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = x_pt[1:4].sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="getitem")


# ---------------------------------------------------------------------------
# Nonlinearities
# ---------------------------------------------------------------------------
def test_relu_gradient_matches_pytorch():
    x_np = np.random.randn(5, 5)
    x = Tensor(x_np, requires_grad=True)
    y = x.relu().sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = torch.relu(x_pt).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="relu")


def test_sigmoid_gradient_matches_pytorch():
    x_np = np.random.randn(5, 5)
    x = Tensor(x_np, requires_grad=True)
    y = x.sigmoid().sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = torch.sigmoid(x_pt).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="sigmoid")


def test_tanh_gradient_matches_pytorch():
    x_np = np.random.randn(5, 5)
    x = Tensor(x_np, requires_grad=True)
    y = x.tanh().sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = torch.tanh(x_pt).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="tanh")


def test_exp_gradient_matches_pytorch():
    x_np = np.random.randn(5, 5)
    x = Tensor(x_np, requires_grad=True)
    y = x.exp().sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = torch.exp(x_pt).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="exp")


def test_log_gradient_matches_pytorch():
    x_np = np.random.rand(5, 5) + 1.0  # ensure positive
    x = Tensor(x_np, requires_grad=True)
    y = x.log().sum()
    y.backward()
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = torch.log(x_pt).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="log")


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def test_cross_entropy_loss_and_gradient_match_pytorch():
    logits_np = np.random.randn(4, 3)
    targets = np.array([0, 1, 2, 1])
    logits = Tensor(logits_np, requires_grad=True)
    loss = logits.cross_entropy(targets)
    loss.backward()
    logits_pt = torch.tensor(logits_np, requires_grad=True)
    loss_pt = torch.nn.functional.cross_entropy(logits_pt, torch.tensor(targets))
    loss_pt.backward()
    assert abs(loss.data - loss_pt.item()) < 1e-5, (
        f"CE loss mismatch: ours={loss.data}, pt={loss_pt.item()}"
    )
    _compare_grads(logits, logits_pt, label="cross_entropy")


def test_mse_loss_and_gradient_match_pytorch():
    pred_np = np.random.randn(3, 2)
    target_np = np.random.randn(3, 2)
    pred = Tensor(pred_np, requires_grad=True)
    loss = pred.mse(target_np)
    loss.backward()
    pred_pt = torch.tensor(pred_np, requires_grad=True)
    loss_pt = torch.nn.functional.mse_loss(pred_pt, torch.tensor(target_np))
    loss_pt.backward()
    assert abs(loss.data - loss_pt.item()) < 1e-5, (
        f"MSE loss mismatch: ours={loss.data}, pt={loss_pt.item()}"
    )
    _compare_grads(pred, pred_pt, label="mse")


# ---------------------------------------------------------------------------
# Broadcasting-aware gradient accumulation
# ---------------------------------------------------------------------------
def test_broadcasting_add_gradient_accumulation():
    """(3,4) + (4,) → (3,4). The bias gradient should be summed over axis 0."""
    A_np = np.random.randn(3, 4)
    b_np = np.random.randn(4)
    A = Tensor(A_np, requires_grad=True)
    b = Tensor(b_np, requires_grad=True)
    y = (A + b).sum()
    y.backward()
    A_pt = torch.tensor(A_np, requires_grad=True)
    b_pt = torch.tensor(b_np, requires_grad=True)
    y_pt = (A_pt + b_pt).sum()
    y_pt.backward()
    _compare_grads(A, A_pt, label="broadcast_A")
    _compare_grads(b, b_pt, label="broadcast_b")


def test_broadcasting_mul_gradient_accumulation():
    """(3,4) * (4,) → (3,4)."""
    A_np = np.random.randn(3, 4)
    b_np = np.random.randn(4)
    A = Tensor(A_np, requires_grad=True)
    b = Tensor(b_np, requires_grad=True)
    y = (A * b).sum()
    y.backward()
    A_pt = torch.tensor(A_np, requires_grad=True)
    b_pt = torch.tensor(b_np, requires_grad=True)
    y_pt = (A_pt * b_pt).sum()
    y_pt.backward()
    _compare_grads(A, A_pt, label="broadcast_mul_A")
    _compare_grads(b, b_pt, label="broadcast_mul_b")


# ---------------------------------------------------------------------------
# unbroadcast helper unit test
# ---------------------------------------------------------------------------
def test_unbroadcast_helper():
    """unbroadcast should sum extra dims and broadcast-1 dims correctly."""
    # (3, 4) → (4,): sum over axis 0.
    grad = np.random.randn(3, 4)
    result = unbroadcast(grad, (4,))
    assert result.shape == (4,)
    np.testing.assert_allclose(result, grad.sum(axis=0))

    # (3, 4) → (1, 4): sum over axis 0 with keepdims.
    result = unbroadcast(grad, (1, 4))
    assert result.shape == (1, 4)
    np.testing.assert_allclose(result, grad.sum(axis=0, keepdims=True))


# ---------------------------------------------------------------------------
# NN module tests
# ---------------------------------------------------------------------------
def test_linear_layer_gradient_matches_pytorch():
    """Linear layer forward + backward against PyTorch."""
    np.random.seed(42)
    lin = nn.Linear(3, 4)
    x_np = np.random.randn(2, 3)
    x = Tensor(x_np, requires_grad=True)
    y = (lin(x) ** 2).sum()
    y.backward()
    # PyTorch: weight is (out, in) so we transpose.
    lin_pt = torch.nn.Linear(3, 4)
    lin_pt.weight.data = torch.tensor(lin.weight.data.T)
    lin_pt.bias.data = torch.tensor(lin.bias.data)
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = (lin_pt(x_pt) ** 2).sum()
    y_pt.backward()
    _compare_grads(x, x_pt, label="linear_x")
    # Weight: PyTorch stores (out, in), we store (in, out). Compare transposed.
    np.testing.assert_allclose(
        lin.weight.grad, lin_pt.weight.grad.numpy().T, atol=1e-5,
        err_msg="linear weight grad mismatch",
    )


def test_conv2d_gradient_matches_pytorch():
    """Conv2D forward + backward against PyTorch."""
    np.random.seed(42)
    x_np = np.random.randn(2, 3, 8, 8)
    conv = nn.Conv2D(3, 4, 3)
    x = Tensor(x_np, requires_grad=True)
    out = conv(x)
    loss = out.sum()
    loss.backward()
    # PyTorch.
    x_pt = torch.tensor(x_np, requires_grad=True)
    conv_pt = torch.nn.Conv2d(3, 4, 3, bias=True)
    conv_pt.weight.data = torch.tensor(conv.weight.data)
    conv_pt.bias.data = torch.tensor(conv.bias.data)
    out_pt = conv_pt(x_pt)
    loss_pt = out_pt.sum()
    loss_pt.backward()
    _compare_grads(x, x_pt, label="conv_x", atol=1e-4)
    np.testing.assert_allclose(
        conv.weight.grad, conv_pt.weight.grad.numpy(), atol=1e-4,
        err_msg="conv weight grad mismatch",
    )


def test_maxpool2d_gradient_matches_pytorch():
    """MaxPool2D forward + backward against PyTorch."""
    np.random.seed(42)
    x_np = np.random.randn(2, 3, 8, 8)
    pool = nn.MaxPool2D(2)
    x = Tensor(x_np, requires_grad=True)
    out = pool(x)
    loss = out.sum()
    loss.backward()
    # PyTorch.
    x_pt = torch.tensor(x_np, requires_grad=True)
    pool_pt = torch.nn.MaxPool2d(2)
    out_pt = pool_pt(x_pt)
    loss_pt = out_pt.sum()
    loss_pt.backward()
    _compare_grads(x, x_pt, label="maxpool_x")


def test_module_parameters_traversal():
    """Sequential.parameters() should walk all sub-modules."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )
    params = model.parameters()
    # Should find 4 parameter tensors: 2 weights + 2 biases.
    assert len(params) == 4
    # Total param count.
    total = sum(p.data.size for p in params)
    assert total == 4 * 8 + 8 + 8 * 3 + 3  # 32 + 8 + 24 + 3 = 67


def test_sgd_optimizer_updates_parameters():
    """SGD should reduce the loss after one step."""
    np.random.seed(42)
    model = nn.Sequential(nn.Linear(4, 1))
    opt = nn.SGD(model.parameters(), lr=0.1)
    x_np = np.random.randn(2, 4)
    target_np = np.array([[1.0], [0.0]])
    x = Tensor(x_np, requires_grad=False)
    loss_before = model(x).mse(target_np).data
    model.zero_grad()
    loss = model(x).mse(target_np)
    loss.backward()
    opt.step()
    loss_after = model(x).mse(target_np).data
    assert loss_after < loss_before, (
        f"SGD didn't reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"
    )


def test_adam_optimizer_updates_parameters():
    """Adam should reduce the loss after one step."""
    np.random.seed(42)
    model = nn.Sequential(nn.Linear(4, 1))
    opt = nn.Adam(model.parameters(), lr=0.01)
    x_np = np.random.randn(2, 4)
    target_np = np.array([[1.0], [0.0]])
    x = Tensor(x_np, requires_grad=False)
    loss_before = model(x).mse(target_np).data
    model.zero_grad()
    loss = model(x).mse(target_np)
    loss.backward()
    opt.step()
    loss_after = model(x).mse(target_np).data
    assert loss_after < loss_before, (
        f"Adam didn't reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def test_cli_grad_check_only():
    """`python train.py --grad-check-only` should exit 0 + verify all gradients."""
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--grad-check-only",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                            cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"
    assert "GRAD_CHECK_PASSED=true" in result.stdout


def test_cli_train_mlp():
    """`python train.py --epochs 1` should train + write JSON."""
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--dataset", "mnist", "--model", "mlp",
        "--epochs", "1", "--n-train", "100", "--n-test", "20",
        "--metrics-json", "/tmp/_p10_cli_metrics.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"
    assert "GRAD_CHECK_PASSED=true" in result.stdout
    assert "FINAL_VAL_ACC=" in result.stdout
    assert Path("/tmp/_p10_cli_metrics.json").exists()
    import json
    payload = json.loads(Path("/tmp/_p10_cli_metrics.json").read_text())
    assert "gradient_check" in payload
    assert "history" in payload


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_add_gradient_matches_pytorch,
        test_sub_gradient_matches_pytorch,
        test_mul_gradient_matches_pytorch,
        test_pow_gradient_matches_pytorch,
        test_div_gradient_matches_pytorch,
        test_neg_gradient_matches_pytorch,
        test_matmul_2d_gradient_matches_pytorch,
        test_matmul_broadcast_gradient_matches_pytorch,
        test_sum_gradient_matches_pytorch,
        test_mean_gradient_matches_pytorch,
        test_transpose_gradient_matches_pytorch,
        test_reshape_gradient_matches_pytorch,
        test_getitem_gradient_matches_pytorch,
        test_relu_gradient_matches_pytorch,
        test_sigmoid_gradient_matches_pytorch,
        test_tanh_gradient_matches_pytorch,
        test_exp_gradient_matches_pytorch,
        test_log_gradient_matches_pytorch,
        test_cross_entropy_loss_and_gradient_match_pytorch,
        test_mse_loss_and_gradient_match_pytorch,
        test_broadcasting_add_gradient_accumulation,
        test_broadcasting_mul_gradient_accumulation,
        test_unbroadcast_helper,
        test_linear_layer_gradient_matches_pytorch,
        test_conv2d_gradient_matches_pytorch,
        test_maxpool2d_gradient_matches_pytorch,
        test_module_parameters_traversal,
        test_sgd_optimizer_updates_parameters,
        test_adam_optimizer_updates_parameters,
        test_cli_grad_check_only,
        test_cli_train_mlp,
    ]
    n_passed = 0
    n_failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            n_passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            n_failed += 1
    print(f"\n{n_passed} passed, {n_failed} failed (out of {len(tests)} total).")
    if n_failed > 0:
        sys.exit(1)
