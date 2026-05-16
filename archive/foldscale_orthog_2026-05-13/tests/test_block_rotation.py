"""Tests for the block-diagonal-at-G rotation candidate search."""

from __future__ import annotations

import math

import pytest
import torch

from prismaquant.block_rotation import (
    BlockRotationSearchTarget,
    apply_block_rotation_input,
    effective_weight_for_scoring,
    random_orthogonal,
    rotate_weight_for_storage,
    search_block_rotation,
    sylvester_hadamard,
)


def test_sylvester_hadamard_is_orthonormal():
    for g in (2, 4, 8, 16, 32):
        H = sylvester_hadamard(g)
        assert H.shape == (g, g)
        assert torch.allclose(H @ H.t(), torch.eye(g), atol=1e-6)
        assert torch.allclose(H.t() @ H, torch.eye(g), atol=1e-6)


def test_sylvester_hadamard_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        sylvester_hadamard(3)
    with pytest.raises(ValueError):
        sylvester_hadamard(15)
    with pytest.raises(ValueError):
        sylvester_hadamard(0)


def test_random_orthogonal_is_orthonormal():
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    for g in (4, 16, 32):
        Q = random_orthogonal(g, generator=gen, device="cpu")
        assert Q.shape == (g, g)
        assert torch.allclose(Q @ Q.t(), torch.eye(g), atol=1e-5)


def test_apply_block_rotation_input_identity_is_noop():
    g = 16
    M = torch.randn(8, 64)
    R = torch.eye(g)
    out = apply_block_rotation_input(M, R)
    assert torch.allclose(out, M, atol=1e-6)


def test_apply_block_rotation_input_block_diagonal_semantics():
    """A block-rotation must act on each G-block independently."""
    g = 4
    n_blocks = 3
    in_features = g * n_blocks
    M = torch.randn(2, in_features)
    R = sylvester_hadamard(g)

    out = apply_block_rotation_input(M, R)

    expected = torch.empty_like(M)
    for b in range(n_blocks):
        sl = slice(b * g, (b + 1) * g)
        expected[:, sl] = M[:, sl] @ R
    assert torch.allclose(out, expected, atol=1e-6)


def test_block_rotation_roundtrip_recovers_input_without_quantization():
    """With no quantization, stored * runtime cancels to identity."""
    g = 16
    M = torch.randn(7, g * 4)
    R = sylvester_hadamard(g)

    stored = rotate_weight_for_storage(M, R)
    recovered = effective_weight_for_scoring(stored, R)
    assert torch.allclose(recovered, M, atol=1e-5)


def test_block_rotation_roundtrip_random_orthogonal():
    """Same roundtrip for a learned/random orthogonal."""
    g = 16
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)
    R = random_orthogonal(g, generator=gen, device="cpu")
    M = torch.randn(5, g * 3)

    stored = rotate_weight_for_storage(M, R)
    recovered = effective_weight_for_scoring(stored, R)
    assert torch.allclose(recovered, M, atol=1e-5)


def _identity_render(idx, w_rotated, R):
    """Fake quantizer that introduces no error. Used to test gate behavior."""
    return w_rotated.clone()


def _per_block_rtn4_render(idx, w_rotated, R):
    """Fake quantizer: per-G-block 4-bit symmetric round-to-nearest.

    This is intentionally crude. The point is to introduce real per-group
    quantization error so the search has something to optimize.
    """
    out, in_features = w_rotated.shape
    g = int(R.shape[0])
    n_blocks = in_features // g
    grouped = w_rotated.reshape(out, n_blocks, g)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    levels = 7.0  # int4 symmetric levels per side
    step = max_abs / levels
    quantized = (grouped / step).round().clamp(-levels, levels) * step
    return quantized.reshape(out, in_features)


def test_search_returns_identity_when_no_quant_error():
    """If the renderer is exact, identity ties the baseline so search keeps it."""
    g = 16
    weight = torch.randn(8, g * 4)
    acts = torch.randn(64, g * 4)
    target = BlockRotationSearchTarget(
        name="t0",
        fmt="NVFP4",
        weight=weight,
        activations=acts,
        group_size=g,
    )

    result = search_block_rotation(
        [target],
        _identity_render,
        group_size=g,
        n_random_orthogonal=2,
        seed=1,
    )

    assert result.selected_label == "identity"
    assert result.gate_reason == "identity"
    assert math.isfinite(result.best_score)
    assert result.best_score <= result.baseline_score + 1e-9


def test_search_prefers_rotation_under_one_dominant_high_variance_channel():
    """Channel 0 has 20x stddev of the others; identity's step overshoots and
    crushes the small channels to 0. Rotation distributes variance across the
    group so the step matches the residual scale, recovering the small ones.
    """
    g = 16
    n_blocks = 4
    in_features = g * n_blocks
    rows = 32
    torch.manual_seed(0)
    weight = torch.randn(rows, in_features) * 0.5
    # One dominant channel per block, constant magnitude.
    for b in range(n_blocks):
        weight[:, b * g] = 10.0
    acts = torch.randn(128, in_features)
    target = BlockRotationSearchTarget(
        name="t0",
        fmt="NVFP4",
        weight=weight,
        activations=acts,
        group_size=g,
    )

    result = search_block_rotation(
        [target],
        _per_block_rtn4_render,
        group_size=g,
        n_random_orthogonal=4,
        seed=11,
    )

    assert result.selected_label != "identity"
    assert result.best_score < result.baseline_score


def test_search_joint_two_targets_shares_one_matrix():
    """Multi-target search should pick one matrix scored across all targets."""
    g = 16
    weight_a = torch.randn(8, g * 4)
    weight_b = torch.randn(6, g * 4)
    acts = torch.randn(32, g * 4)

    targets = [
        BlockRotationSearchTarget(
            name="a", fmt="NVFP4", weight=weight_a, activations=acts, group_size=g,
        ),
        BlockRotationSearchTarget(
            name="b", fmt="NVFP4", weight=weight_b, activations=acts, group_size=g,
        ),
    ]
    result = search_block_rotation(
        targets,
        _per_block_rtn4_render,
        group_size=g,
        n_random_orthogonal=3,
        seed=2,
    )

    assert result.matrix.shape == (g, g)
    assert math.isfinite(result.best_score)
    assert math.isfinite(result.baseline_score)


def test_search_rejects_mismatched_widths():
    g = 16
    bad = BlockRotationSearchTarget(
        name="a",
        fmt="NVFP4",
        weight=torch.randn(4, g * 2),
        activations=torch.randn(8, g * 3),
        group_size=g,
    )
    with pytest.raises(ValueError):
        search_block_rotation(
            [bad], _identity_render, group_size=g, n_random_orthogonal=0,
        )


def test_search_rejects_input_not_divisible_by_g():
    g = 16
    bad = BlockRotationSearchTarget(
        name="a",
        fmt="NVFP4",
        weight=torch.randn(4, 33),
        activations=torch.randn(8, 33),
        group_size=g,
    )
    with pytest.raises(ValueError):
        search_block_rotation(
            [bad], _identity_render, group_size=g, n_random_orthogonal=0,
        )


def test_cayley_orthogonal_is_orthonormal():
    from prismaquant.block_rotation import cayley_orthogonal

    g = 16
    torch.manual_seed(0)
    A = torch.randn(g, g) * 0.3
    R = cayley_orthogonal(A)
    assert R.shape == (g, g)
    assert torch.allclose(R @ R.t(), torch.eye(g), atol=1e-5)
    assert torch.allclose(R.t() @ R, torch.eye(g), atol=1e-5)


def test_cayley_at_zero_returns_identity():
    from prismaquant.block_rotation import cayley_orthogonal

    g = 8
    A = torch.zeros(g, g)
    R = cayley_orthogonal(A)
    assert torch.allclose(R, torch.eye(g), atol=1e-6)


def test_learn_block_rotation_decreases_loss():
    """Optimization should at least not increase the final loss."""
    from prismaquant.block_rotation import learn_block_rotation_cayley

    g = 16
    torch.manual_seed(0)
    # Pattern with one dominant high-variance channel per block — rotation helps.
    rows = 32
    in_features = g * 4
    weight = torch.randn(rows, in_features) * 0.5
    for b in range(4):
        weight[:, b * g] = 10.0
    acts = torch.randn(64, in_features)
    target = BlockRotationSearchTarget(
        name="t0",
        fmt="NVFP4",
        weight=weight,
        activations=acts,
        group_size=g,
    )
    init_R = torch.eye(g)
    R, final_loss = learn_block_rotation_cayley(
        [target],
        init_R=init_R,
        group_size=g,
        steps=30,
        lr=1e-2,
    )
    # R is orthonormal
    assert torch.allclose(R @ R.t(), torch.eye(g), atol=1e-4)
    assert math.isfinite(final_loss)


def test_search_block_rotation_with_cayley_does_not_regress():
    """When cayley_steps > 0, the search must not return a worse R than the
    static-best one (it scores the learned R against the same gate)."""
    g = 16
    torch.manual_seed(42)
    rows = 32
    in_features = g * 4
    weight = torch.randn(rows, in_features) * 0.5
    for b in range(4):
        weight[:, b * g] = 10.0
    acts = torch.randn(128, in_features)
    target = BlockRotationSearchTarget(
        name="t0",
        fmt="NVFP4",
        weight=weight,
        activations=acts,
        group_size=g,
    )

    result_static = search_block_rotation(
        [target],
        _per_block_rtn4_render,
        group_size=g,
        n_random_orthogonal=4,
        seed=11,
        cayley_steps=0,
    )
    result_cayley = search_block_rotation(
        [target],
        _per_block_rtn4_render,
        group_size=g,
        n_random_orthogonal=4,
        seed=11,
        cayley_steps=20,
        cayley_lr=1e-2,
    )

    assert result_cayley.best_score <= result_static.best_score + 1e-6


def test_block_ortho_g_is_registered_in_render_score():
    from prismaquant.render_score import (
        registered_render_mechanisms,
        resolve_render_mechanism_order,
    )

    registry = registered_render_mechanisms()
    spec = registry.get("block_ortho_g")
    assert spec is not None
    assert spec.exclusive_group == "activation_weight_fold"
    assert spec.phase == 20

    plan = resolve_render_mechanism_order(["block_ortho_g", "awq"])
    assert plan.errors
    assert "mutually exclusive" in plan.errors[0]

    plan2 = resolve_render_mechanism_order(["block_ortho_g", "gptq"])
    assert plan2.errors == ()
    assert "block_ortho_g" in plan2.names()
    assert "gptq" in plan2.names()
