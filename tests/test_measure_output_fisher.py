"""Unit tests for prismaquant.measure_output_fisher.

These tests verify the analytic Fisher math (variance / covariance under
a categorical distribution) against hand-computed values, plus the
four-term identity decomposition Var(a+b) = Var(a) + 2 Cov(a,b) + Var(b).
"""
from __future__ import annotations

import math

import pytest
import torch

from prismaquant import measure_output_fisher as of


def _probs_from_logits(z: torch.Tensor) -> torch.Tensor:
    return torch.softmax(z, dim=-1)


def test_fisher_omega_ii_uniform_distribution():
    """Var_uniform(δz) reduces to the unbiased sample variance × (V-1)/V."""
    # Uniform p = 1/V; Var_p(δz) = (1/V) Σ δz_v² - (1/V Σ δz_v)²
    V = 4
    p = torch.full((1, V), 1.0 / V)
    dz = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # mean = 2.5, var = 1.25
    omega = of.fisher_omega_ii([p], [dz])
    # Expected: (1/2) × 1.25 = 0.625
    assert omega == pytest.approx(0.625, rel=1e-6)


def test_fisher_omega_ii_zero_perturbation():
    """δz = 0 → Ω_ii = 0."""
    p = torch.tensor([[0.1, 0.2, 0.7]])
    dz = torch.zeros((1, 3))
    omega = of.fisher_omega_ii([p], [dz])
    assert omega == pytest.approx(0.0, abs=1e-12)


def test_fisher_omega_ii_constant_perturbation_gives_zero():
    """A constant shift in logits doesn't change softmax — variance is 0."""
    p = torch.tensor([[0.1, 0.3, 0.6]])
    dz = torch.tensor([[5.0, 5.0, 5.0]])  # constant shift
    omega = of.fisher_omega_ii([p], [dz])
    assert omega == pytest.approx(0.0, abs=1e-7)


def test_fisher_omega_ij_zero_perturbation():
    """Either δz = 0 → Ω_ij = 0."""
    p = torch.tensor([[0.5, 0.5]])
    dz_a = torch.zeros((1, 2))
    dz_b = torch.tensor([[1.0, -1.0]])
    omega = of.fisher_omega_ij([p], [dz_a], [dz_b])
    assert omega == pytest.approx(0.0, abs=1e-12)


def test_fisher_omega_ij_uncorrelated_perturbations():
    """Cov_p(a, b) = 0 when a and b are linearly independent under p."""
    # Construct δz_a and δz_b that are uncorrelated under uniform p.
    V = 4
    p = torch.full((1, V), 1.0 / V)
    dz_a = torch.tensor([[1.0, -1.0, 1.0, -1.0]])  # mean 0
    dz_b = torch.tensor([[1.0, 1.0, -1.0, -1.0]])  # mean 0, orthogonal
    omega = of.fisher_omega_ij([p], [dz_a], [dz_b])
    # E[ab] = (1·1 + (-1)·1 + 1·(-1) + (-1)·(-1))/4 = 0
    # E[a]·E[b] = 0·0 = 0
    # → Cov = 0
    assert omega == pytest.approx(0.0, abs=1e-12)


def test_fisher_omega_ij_perfectly_correlated():
    """Cov_p(a, a) = Var_p(a) — Output-Fisher's diagonal == its self-covariance."""
    V = 5
    p = torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.2]])
    dz = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    omega_self_pair = of.fisher_omega_ij([p], [dz], [dz])  # = Var_p(dz)
    omega_unary = of.fisher_omega_ii([p], [dz])  # = (1/2) Var_p(dz)
    # Self-pair == 2 × unary
    assert omega_self_pair == pytest.approx(2.0 * omega_unary, rel=1e-6)


def test_four_term_identity_decomposition():
    """Var_p(a+b) = Var_p(a) + 2 Cov_p(a,b) + Var_p(b).

    This is the algebraic identity that makes the four-term identity
    consistent with the Fisher form.  With the (1/2) factor on Ω_ii:

        Ω_ii(a+b) = (1/2) Var_p(a+b)
                  = (1/2)[Var_p(a) + 2 Cov_p(a,b) + Var_p(b)]
                  = Ω_ii(a) + Ω_ii(b) + Cov_p(a,b)
                  = Ω_ii(a) + Ω_ii(b) + Ω_ij(a,b)
    """
    torch.manual_seed(0)
    V = 8
    p = torch.softmax(torch.randn(1, V), dim=-1)
    dz_a = torch.randn(1, V)
    dz_b = torch.randn(1, V)
    dz_sum = dz_a + dz_b

    omega_a = of.fisher_omega_ii([p], [dz_a])
    omega_b = of.fisher_omega_ii([p], [dz_b])
    omega_sum = of.fisher_omega_ii([p], [dz_sum])
    omega_ab = of.fisher_omega_ij([p], [dz_a], [dz_b])

    assert omega_sum == pytest.approx(omega_a + omega_b + omega_ab, rel=1e-6)


def test_fisher_omega_ii_matches_kl_at_small_delta():
    """KL(p ‖ p ⊕ δz) ≈ (1/2) Var_p(δz) for small δz.

    Compare exact KL to the Fisher quadratic-form approximation for a
    perturbation of magnitude 1e-3.  Should agree to within higher-order
    error ~ O(δz³).
    """
    torch.manual_seed(42)
    V = 16
    z_teacher = torch.randn(1, V) * 2.0
    p_teacher = torch.softmax(z_teacher, dim=-1)
    log_p = torch.log_softmax(z_teacher, dim=-1)

    dz = torch.randn(1, V) * 1e-3  # very small
    log_q = torch.log_softmax(z_teacher + dz, dim=-1)
    # Exact KL(p_teacher || p_student) per token:
    kl_exact = float((p_teacher * (log_p - log_q)).sum().item())

    # Fisher approximation:
    omega_ii = of.fisher_omega_ii([p_teacher], [dz])
    # Convert "average over tokens" Ω back to single-token KL:
    # Ω = (1/2) × mean_tokens Var = (1/2) × Var (one token).
    kl_fisher = omega_ii  # already per token

    # Higher-order error is O(δz³); for δz ~ 1e-3 we expect ~1% relative error.
    assert kl_fisher == pytest.approx(kl_exact, rel=5e-2)


def test_fisher_omega_ii_consistent_across_token_aggregation():
    """Splitting one [T, V] tensor into [1, V] chunks should give the same answer."""
    torch.manual_seed(1)
    V, T = 6, 5
    z = torch.randn(T, V) * 1.5
    p = torch.softmax(z, dim=-1)
    dz = torch.randn(T, V) * 0.1

    omega_full = of.fisher_omega_ii([p], [dz])

    # Split into per-token tensors
    p_chunks = [p[t:t+1] for t in range(T)]
    dz_chunks = [dz[t:t+1] for t in range(T)]
    omega_chunks = of.fisher_omega_ii(p_chunks, dz_chunks)

    assert omega_full == pytest.approx(omega_chunks, rel=1e-7)


def test_fisher_omega_ii_with_linear_offset_zero_offset_matches_no_offset():
    """When linear_offset is all zeros, sandwich Ω_ii reduces to BF16-centered."""
    torch.manual_seed(123)
    p = torch.softmax(torch.randn(2, 5), dim=-1)
    dz = torch.randn(2, 5) * 0.5
    zero_offset = [torch.zeros_like(p)]
    omega_no_offset = of.fisher_omega_ii([p], [dz])
    omega_zero_offset = of.fisher_omega_ii([p], [dz], linear_offset=zero_offset)
    assert omega_no_offset == pytest.approx(omega_zero_offset, rel=1e-7)


def test_fisher_omega_ii_sandwich_formula_matches_kl_difference():
    """For sandwich centered at z_c with student state p_c:
    Ω_ii ≈ KL(p_t || student_perturbed) − KL(p_t || p_c) at second order.

    Verify this matches the analytic decomposition
    ⟨p_c − p_t, δz⟩ + (1/2) Var_{p_c}(δz).
    """
    torch.manual_seed(7)
    V = 8
    z_t = torch.randn(1, V) * 1.0  # teacher logits
    p_t = torch.softmax(z_t, dim=-1)
    log_p_t = torch.log_softmax(z_t, dim=-1)

    # Centered student state at z_c = z_t + ε_c (small offset)
    eps_c = torch.randn(1, V) * 0.05
    z_c = z_t + eps_c
    p_c = torch.softmax(z_c, dim=-1)
    log_p_c = torch.log_softmax(z_c, dim=-1)

    # KL(p_t || p_c) per token, summed over vocab
    kl_c = float((p_t * (log_p_t - log_p_c)).sum().item())

    # Perturbation δz (from centered)
    dz = torch.randn(1, V) * 0.03  # small enough for second-order to dominate
    z_pert = z_c + dz
    log_p_pert = torch.log_softmax(z_pert, dim=-1)
    kl_pert = float((p_t * (log_p_t - log_p_pert)).sum().item())

    # Exact KL difference (the four-term Ω_ii at sandwich)
    omega_exact = kl_pert - kl_c

    # Output-Fisher analytic prediction: ⟨p_c − p_t, δz⟩ + (1/2) Var_{p_c}(δz)
    linear_offset = [(p_c - p_t)]
    omega_predicted = of.fisher_omega_ii([p_c], [dz], linear_offset=linear_offset)

    # Should agree to second order; remaining error is O(δz^3) ≈ O(1e-5)
    assert omega_predicted == pytest.approx(omega_exact, rel=5e-2, abs=1e-7)


def test_omega_ij_correct_when_aggregated_across_samples():
    """E_t Cov should aggregate correctly across multiple samples."""
    torch.manual_seed(5)
    p1 = torch.softmax(torch.randn(2, 4), dim=-1)
    p2 = torch.softmax(torch.randn(3, 4), dim=-1)
    dz_a1, dz_a2 = torch.randn(2, 4), torch.randn(3, 4)
    dz_b1, dz_b2 = torch.randn(2, 4), torch.randn(3, 4)

    omega_split = of.fisher_omega_ij(
        [p1, p2], [dz_a1, dz_a2], [dz_b1, dz_b2],
    )

    # Concatenate manually
    p_all = torch.cat([p1, p2], dim=0)
    a_all = torch.cat([dz_a1, dz_a2], dim=0)
    b_all = torch.cat([dz_b1, dz_b2], dim=0)
    omega_concat = of.fisher_omega_ij([p_all], [a_all], [b_all])

    assert omega_split == pytest.approx(omega_concat, rel=1e-6)
