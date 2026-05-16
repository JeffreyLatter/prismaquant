import torch

from prismaquant.respinquant_core import (
    CayleySGD,
    TrainableRotation,
    fake_quantize_activation,
    orthogonality_error,
    paper_subspace_residual_transition,
    residual_transition_from_bases,
)


def test_trainable_rotation_hadamard_init_is_orthogonal():
    rot = TrainableRotation(8, seed=3, device="cpu")

    assert orthogonality_error(rot.weight) < 1e-6


def test_cayley_sgd_preserves_orthogonality_and_reduces_loss():
    target = torch.linalg.qr(torch.randn(8, 8)).Q
    if torch.linalg.det(target) < 0:
        target[:, 0] = -target[:, 0]
    rot = TrainableRotation(8, seed=5, device="cpu")
    opt = CayleySGD([rot.weight], lr=0.25)

    initial = float((rot.weight - target).pow(2).sum().item())
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        loss = (rot.weight - target).pow(2).sum()
        loss.backward()
        opt.step()

    final = float((rot.weight - target).pow(2).sum().item())
    assert final < initial
    assert orthogonality_error(rot.weight) < 1e-5


def test_fake_quantize_activation_is_straight_through():
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = fake_quantize_activation(x, bits=4, symmetric=False)

    assert y.shape == x.shape
    assert not torch.allclose(y.detach(), x.detach())
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_residual_transition_from_bases_row_convention():
    a = torch.linalg.qr(torch.randn(8, 8)).Q
    b = torch.linalg.qr(torch.randn(8, 8)).Q
    x = torch.randn(4, 8)

    transition = residual_transition_from_bases(a, b, convention="row")

    assert torch.allclose((x @ a) @ transition, x @ b, atol=1e-5, rtol=1e-5)


def test_paper_subspace_transition_exposes_plugin_tensors():
    a = torch.eye(8)
    b = torch.eye(8)
    theta = 0.2
    b[:2, :2] = torch.tensor([
        [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
        [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))],
    ])
    transition = residual_transition_from_bases(a, b, convention="row")

    approx = paper_subspace_residual_transition(
        transition,
        rank=2,
        device=torch.device("cpu"),
    )

    assert approx.metadata["mode"] == "paper_svd_polar"
    assert approx.metadata["sv_energy_retained"] > 0.999
    assert torch.allclose(approx.matrix, transition, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        torch.eye(8) + approx.u @ approx.v,
        approx.matrix,
        atol=1e-5,
        rtol=1e-5,
    )
