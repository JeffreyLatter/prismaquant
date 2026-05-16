"""Core math for ReSpinQuant-style trained residual rotations.

This module contains the reusable pieces that are independent of any one model
class or exporter:

* Hadamard initialization for dense orthogonal rotations.
* Cayley-transform optimization on the orthogonal manifold.
* Straight-through activation fake quantization for rotation training.
* The ReSpinQuant SVD/polar residual subspace approximation.

The runtime/export topology remains separate. Full ReSpinQuant is only
production-eligible when every residual-basis transition in the chosen model
can be represented by the artifact/runtime path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .halo import random_hadamard


@dataclass(frozen=True)
class ResidualSubspaceApproximation:
    """Low-rank representation of a residual-basis transition."""

    u: torch.Tensor
    v: torch.Tensor
    matrix: torch.Tensor
    metadata: dict[str, float | int | str]


def orthogonality_error(matrix: torch.Tensor) -> float:
    """Return max absolute error from ``matrix.T @ matrix == I``."""

    m = matrix.detach().float()
    eye = torch.eye(m.shape[1], device=m.device, dtype=m.dtype)
    return float((m.transpose(0, 1) @ m - eye).abs().max().item())


def hadamard_rotation_init(
    dim: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a randomized Hadamard orthogonal initializer."""

    return random_hadamard(dim, seed=seed, device=device, dtype=dtype)


class TrainableRotation(nn.Module):
    """Dense orthogonal rotation parameter initialized from Hadamard."""

    def __init__(
        self,
        dim: int,
        *,
        seed: int = 0,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        init = hadamard_rotation_init(dim, seed=seed, device=device)
        self.weight = nn.Parameter(init.to(torch.float32))

    def forward(self, x: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
        w = self.weight.to(device=x.device, dtype=x.dtype)
        if transpose:
            return x @ w.transpose(0, 1)
        return x @ w


def _cayley_update(
    x: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    *,
    max_lr: float | None = None,
) -> torch.Tensor:
    """One dense Cayley-transform descent step.

    For an orthogonal matrix ``X`` and Euclidean gradient ``G``, form the
    skew-symmetric generator ``A = G X^T - X G^T`` and update with the Cayley
    transform. This preserves orthogonality up to solve precision.
    """

    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"Cayley update expects a square matrix, got {tuple(x.shape)}")
    g = grad.to(device=x.device, dtype=torch.float32)
    x32 = x.to(dtype=torch.float32)
    a = g @ x32.transpose(0, 1) - x32 @ g.transpose(0, 1)
    # Bound the step by the generator's one-norm, matching the stability idea
    # used by Stiefel/Cayley optimizers without copying their implementation.
    one_norm = a.abs().sum(dim=0).max().clamp_min(1e-8)
    tau = float(lr)
    if max_lr is not None:
        tau = min(tau, float(max_lr))
    tau = min(tau, float(2.0 / one_norm.item()))
    eye = torch.eye(x32.shape[0], device=x32.device, dtype=x32.dtype)
    lhs = eye + 0.5 * tau * a
    rhs = (eye - 0.5 * tau * a) @ x32
    return torch.linalg.solve(lhs, rhs).to(dtype=x.dtype)


class CayleySGD(Optimizer):
    """SGD optimizer for dense square orthogonal matrices.

    Parameters in groups with ``stiefel=True`` are updated by a Cayley
    transform. Other parameters fall back to ordinary SGD, which keeps the
    optimizer usable in small tests.
    """

    def __init__(
        self,
        params,
        lr: float,
        *,
        stiefel: bool = True,
        momentum: float = 0.0,
        grad_clip: float | None = None,
        max_lr: float | None = None,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "stiefel": bool(stiefel),
            "momentum": float(momentum),
            "grad_clip": grad_clip,
            "max_lr": max_lr,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group.get("momentum", 0.0))
            grad_clip = group.get("grad_clip")
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if grad_clip is not None:
                    norm = grad.norm().clamp_min(1e-12)
                    if float(norm.item()) > float(grad_clip):
                        grad = grad * (float(grad_clip) / norm)
                if momentum:
                    state = self.state[param]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(grad)
                        state["momentum_buffer"] = buf
                    buf.mul_(momentum).add_(grad)
                    grad = buf
                if group.get("stiefel", True):
                    param.copy_(_cayley_update(
                        param.detach(),
                        grad,
                        lr,
                        max_lr=group.get("max_lr"),
                    ))
                else:
                    param.add_(grad, alpha=-lr)
        return loss


def fake_quantize_activation(
    x: torch.Tensor,
    *,
    bits: int = 4,
    symmetric: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Straight-through per-token activation fake quantization."""

    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    if bits >= 16:
        return x
    x32 = x.float()
    if symmetric:
        qmax = float((1 << (bits - 1)) - 1)
        scale = x32.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
        q = torch.round(x32 / scale).clamp(-qmax, qmax)
        deq = q * scale
    else:
        qmax = float((1 << bits) - 1)
        xmin = x32.amin(dim=-1, keepdim=True)
        xmax = x32.amax(dim=-1, keepdim=True)
        scale = (xmax - xmin).clamp_min(eps) / qmax
        zero = torch.round(-xmin / scale).clamp(0.0, qmax)
        q = torch.round(x32 / scale + zero).clamp(0.0, qmax)
        deq = (q - zero) * scale
    deq = deq.to(dtype=x.dtype)
    return x + (deq - x).detach()


def paper_subspace_residual_transition(
    transition: torch.Tensor,
    rank: int,
    *,
    device: torch.device | str,
) -> ResidualSubspaceApproximation:
    """Approximate a dense residual transition using ReSpinQuant Eq. 5-11."""

    dev = torch.device(device)
    t = transition.to(device=dev, dtype=torch.float32)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError(f"transition must be square, got {tuple(t.shape)}")
    dim = int(t.shape[0])
    r = min(max(int(rank), 0), dim)
    eye = torch.eye(dim, device=dev, dtype=torch.float32)
    if r <= 0:
        return ResidualSubspaceApproximation(
            u=torch.empty(dim, 0, dtype=torch.float32),
            v=torch.empty(0, dim, dtype=torch.float32),
            matrix=eye,
            metadata={
                "mode": "paper_svd_polar",
                "rank": 0,
                "relative_fro_error": float(torch.linalg.matrix_norm(t - eye).item()),
            },
        )

    delta = t - eye
    delta_norm = torch.linalg.matrix_norm(delta)
    if float(delta_norm.item()) <= 1e-12:
        return ResidualSubspaceApproximation(
            u=torch.zeros(dim, r, dtype=torch.float32),
            v=torch.zeros(r, dim, dtype=torch.float32),
            matrix=eye,
            metadata={
                "mode": "paper_svd_polar",
                "rank": int(r),
                "relative_fro_error": 0.0,
                "sv_energy_retained": 1.0,
                "identity_skip": 1,
            },
        )
    left, singular, _right_h = torch.linalg.svd(delta, full_matrices=False)
    q = left[:, :r].contiguous()
    t_sub = q.transpose(0, 1) @ t @ q
    u_sub, _s_sub, vh_sub = torch.linalg.svd(t_sub, full_matrices=False)
    r_sub = u_sub @ vh_sub
    if r_sub.numel() and torch.linalg.det(r_sub).item() < 0:
        u_sub[:, -1] = -u_sub[:, -1]
        r_sub = u_sub @ vh_sub
    m = r_sub - torch.eye(r, device=dev, dtype=torch.float32)
    approx = eye + q @ m @ q.transpose(0, 1)
    residual = t - approx
    denom = torch.linalg.matrix_norm(t - eye).clamp_min(1e-12)
    retained = (
        singular[:r].pow(2).sum() / singular.pow(2).sum().clamp_min(1e-12)
    )
    return ResidualSubspaceApproximation(
        u=q.detach().cpu(),
        v=(m @ q.transpose(0, 1)).detach().cpu(),
        matrix=approx,
        metadata={
            "mode": "paper_svd_polar",
            "rank": int(r),
            "relative_fro_error": float((torch.linalg.matrix_norm(residual) / denom).item()),
            "sv_energy_retained": float(retained.item()),
        },
    )


def residual_transition_from_bases(
    input_basis: torch.Tensor,
    output_basis: torch.Tensor,
    *,
    convention: str = "row",
) -> torch.Tensor:
    """Return the dense transition from one residual basis to another."""

    if input_basis.shape != output_basis.shape:
        raise ValueError(
            f"basis shapes differ: {tuple(input_basis.shape)} vs "
            f"{tuple(output_basis.shape)}"
        )
    if convention == "row":
        return input_basis.transpose(0, 1) @ output_basis
    if convention == "column":
        return output_basis @ input_basis.transpose(0, 1)
    raise ValueError(f"unsupported convention: {convention}")


def rotation_state_dict(rotations: dict[str, TrainableRotation]) -> dict[str, torch.Tensor]:
    """Return CPU tensors suitable for checkpointing."""

    return {
        name: module.weight.detach().cpu().float().contiguous()
        for name, module in rotations.items()
    }


def rotation_metadata(
    rotations: dict[str, TrainableRotation],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize trained rotations for run metadata."""

    data: dict[str, Any] = {
        "rotation_count": len(rotations),
        "orthogonality_max_abs": {
            name: orthogonality_error(module.weight)
            for name, module in rotations.items()
        },
    }
    if extra:
        data.update(extra)
    return data
