"""Tiny correctness tests for the isolated rotation/feedback pilot helpers."""
from __future__ import annotations

import itertools

import pytest
import torch

from prismaquant.rotation_ldlq_pilot import (
    block_error_feedback,
    inverse_hessian_cholesky,
    random_hadamard_signs,
    randomized_hadamard_right,
    relative_frobenius_gap,
)


def test_randomized_hadamard_preserves_linear_identity():
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(7, 16, generator=generator, dtype=torch.float64)
    w = torch.randn(9, 16, generator=generator, dtype=torch.float64)
    signs = random_hadamard_signs(16, seed=23)

    x_rot = randomized_hadamard_right(x, signs)
    w_rot = randomized_hadamard_right(w, signs)

    assert relative_frobenius_gap(x @ w.T, x_rot @ w_rot.T) < 2e-15


def test_scalar_feedback_reaches_known_discrete_optimum():
    # This fixed case has a coupled optimum that independent nearest rounding
    # misses: [-1,-1,-1] costs 1.9736 while [-1,-1,+1] costs 0.7888.
    weight = torch.tensor(
        [[-0.7882739181153038, -1.2492067122934476, -0.08155025071717491]],
        dtype=torch.float64,
    )
    hessian = torch.tensor(
        [
            [3.2422457736691603, -0.6854623922141154, 1.5023807483138358],
            [-0.6854623922141154, 0.5211500765493989, -0.3006832609372827],
            [1.5023807483138358, -0.3006832609372827, 1.1873827241327346],
        ],
        dtype=torch.float64,
    )
    upper, damping = inverse_hessian_cholesky(hessian, damping_fraction=0.0)
    assert damping == 0.0

    def binary_nearest(block: torch.Tensor, _start: int, _end: int) -> torch.Tensor:
        return torch.where(block >= 0, torch.ones_like(block), -torch.ones_like(block))

    feedback = block_error_feedback(
        weight,
        upper,
        binary_nearest,
        block_size=1,
    )
    candidates = [
        torch.tensor(values, dtype=torch.float64).reshape_as(weight)
        for values in itertools.product((-1.0, 1.0), repeat=weight.shape[1])
    ]

    def objective(candidate: torch.Tensor) -> float:
        error = weight - candidate
        return float(error @ hessian @ error.T)

    optimum = min(candidates, key=objective)
    plain = binary_nearest(weight, 0, weight.shape[1])
    assert feedback.tolist() == [[-1.0, -1.0, 1.0]]
    assert torch.equal(feedback, optimum)
    assert objective(feedback) == pytest.approx(0.7888320073901903)
    assert objective(feedback) < objective(plain)
