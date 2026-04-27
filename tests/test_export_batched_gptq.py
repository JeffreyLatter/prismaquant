"""Equivalence tests for the batched NVFP4 GPTQ + scale-sweep path.

These verify that the batched output matches the per-Linear output
within numerical tolerance on a small synthetic example. The batched
math is bitwise the same algorithm; differences should only come from
floating-point reduction order (the batched Cholesky / bmm may pick a
different reduction tree than the per-Linear sequential path).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prismaquant.export_native_compressed import (
    _gptq_obs_rounding_nvfp4,
    _scale_sweep_nvfp4,
)
from prismaquant.export_batched_gptq import (
    gptq_obs_rounding_nvfp4_batched,
    scale_sweep_nvfp4_batched,
)


@pytest.fixture
def seeded():
    torch.manual_seed(0)


def _make_inputs(E: int, out_features: int, in_features: int, T: int):
    weights = torch.randn(
        E, out_features, in_features, dtype=torch.float32) * 0.1
    activations_list = [
        torch.randn(T + e, in_features, dtype=torch.float32) for e in range(E)
    ]
    return weights, activations_list


def test_gptq_batched_matches_per_linear(seeded):
    """Run GPTQ per-Linear and batched on the same inputs; outputs
    should agree to within FP32 reduction-order tolerance."""
    E, out_features, in_features = 4, 32, 64
    weights, activations_list = _make_inputs(E, out_features, in_features, T=128)

    per_outputs = []
    for e in range(E):
        per_outputs.append(_gptq_obs_rounding_nvfp4(
            weights[e], activations_list[e], group_size=16,
        ))
    per_stack = torch.stack(per_outputs, dim=0)

    batch_out = gptq_obs_rounding_nvfp4_batched(
        weights, activations_list, group_size=16, expert_chunk=2,
    )
    # Reduction-order may differ — allow a tight tolerance.
    assert torch.allclose(per_stack, batch_out, atol=1e-4, rtol=1e-3), (
        f"max abs diff: {(per_stack - batch_out).abs().max().item():.3e}\n"
        f"max rel diff: {((per_stack - batch_out).abs() / per_stack.abs().clamp_min(1e-12)).max().item():.3e}")


def test_gptq_batched_handles_dead_columns(seeded):
    """A Linear with a zero-column activation should not poison the
    rest of the batch — its Cholesky gets the identity fallback while
    the others run normally."""
    E, out_features, in_features = 3, 16, 32
    weights, activations_list = _make_inputs(E, out_features, in_features, T=64)
    # Zero out one activation tensor entirely.
    activations_list[1] = torch.zeros(0, in_features, dtype=torch.float32)

    out = gptq_obs_rounding_nvfp4_batched(
        weights, activations_list, group_size=16, expert_chunk=4,
    )
    # Linear 1 should have all-zero weights (dead-column handling).
    assert out[1].abs().max() == 0.0


def test_gptq_batched_with_global_real_overrides(seeded):
    """Per-Linear `global_real` matches between paths when explicit
    overrides are supplied (no implicit per-Linear computation)."""
    E, out_features, in_features = 4, 32, 64
    weights, activations_list = _make_inputs(E, out_features, in_features, T=128)
    overrides = torch.full((E,), 0.05, dtype=torch.float32)

    per_outputs = []
    for e in range(E):
        per_outputs.append(_gptq_obs_rounding_nvfp4(
            weights[e], activations_list[e], group_size=16,
            global_real_override=overrides[e],
        ))
    per_stack = torch.stack(per_outputs, dim=0)

    batch_out = gptq_obs_rounding_nvfp4_batched(
        weights, activations_list, group_size=16,
        global_real_overrides=overrides, expert_chunk=2,
    )
    assert torch.allclose(per_stack, batch_out, atol=1e-4, rtol=1e-3)


def test_scale_sweep_batched_matches_per_linear(seeded):
    """Same equivalence check for the scale-sweep path."""
    E, out_features, in_features = 4, 32, 64
    weights, activations_list = _make_inputs(E, out_features, in_features, T=128)
    reference_weights = weights.clone()

    per_outputs = []
    for e in range(E):
        per_outputs.append(_scale_sweep_nvfp4(
            weights[e], activations_list[e],
            reference_weight=reference_weights[e],
            group_size=16,
        ))
    per_stack = torch.stack(per_outputs, dim=0)

    batch_out = scale_sweep_nvfp4_batched(
        weights, activations_list,
        reference_weights=reference_weights,
        group_size=16, expert_chunk=2,
    )
    assert torch.allclose(per_stack, batch_out, atol=1e-4, rtol=1e-3), (
        f"max abs diff: {(per_stack - batch_out).abs().max().item():.3e}")


def test_scale_sweep_batched_with_overrides(seeded):
    E, out_features, in_features = 4, 32, 64
    weights, activations_list = _make_inputs(E, out_features, in_features, T=128)
    reference_weights = weights.clone()
    overrides = torch.full((E,), 0.05, dtype=torch.float32)

    per_outputs = []
    for e in range(E):
        per_outputs.append(_scale_sweep_nvfp4(
            weights[e], activations_list[e],
            reference_weight=reference_weights[e],
            group_size=16,
            global_real_override=overrides[e],
        ))
    per_stack = torch.stack(per_outputs, dim=0)

    batch_out = scale_sweep_nvfp4_batched(
        weights, activations_list,
        reference_weights=reference_weights,
        group_size=16,
        global_real_overrides=overrides,
        expert_chunk=2,
    )
    assert torch.allclose(per_stack, batch_out, atol=1e-4, rtol=1e-3)
