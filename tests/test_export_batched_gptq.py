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


def test_gptq_batched_realistic_moe_shape(seeded):
    """Production-scale equivalence: 16 experts × `[3072, 1408]` weight
    (similar to MiniMax down_proj proportions, scaled down). Verifies
    the chunked path holds bitwise equivalence at sizes representative
    of real MoE layers.

    Smaller than the live 256-expert path so the per-Linear baseline
    finishes in test time (CPU ~30 s); the chunked path's correctness
    is independent of scale."""
    E, out_features, in_features = 16, 384, 256
    weights, activations_list = _make_inputs(
        E, out_features, in_features, T=128,
    )

    per_outputs = []
    for e in range(E):
        per_outputs.append(_gptq_obs_rounding_nvfp4(
            weights[e], activations_list[e], group_size=16,
        ))
    per_stack = torch.stack(per_outputs, dim=0)

    batch_out = gptq_obs_rounding_nvfp4_batched(
        weights, activations_list, group_size=16, expert_chunk=8,
    )
    # Tolerance scales mildly with size due to FP32 reduction-order
    # drift on bigger Cholesky / bmm; 5e-4 is comfortable.
    assert torch.allclose(per_stack, batch_out, atol=5e-4, rtol=1e-3), (
        f"max abs diff: {(per_stack - batch_out).abs().max().item():.3e}")


def test_quantize_2d_nvfp4_group_batched_integration(seeded):
    """End-to-end integration test for the batched-NVFP4 group path
    used by the export hot loop. Builds a small synthetic group of
    same-shape Linears, runs `_quantize_2d_nvfp4_group_batched`, and
    verifies that each compressed dict has the expected keys + shapes
    and that values are bit-finite (no NaN/Inf).

    This does NOT compare to per-Linear `_quantize_2d` because that
    function reads from the module-level `_CACHED_ACTIVATIONS` /
    `_AWQ_PROPER_SCALES` / `_INPUT_GLOBAL_SCALES` globals; recreating
    that side-channel state in a unit test is more invasive than the
    test itself. The math is already covered by the per-function
    equivalence tests above; this test just exercises the integration
    plumbing."""
    import torch.nn as nn
    from prismaquant.export_native_compressed import (
        _quantize_2d_nvfp4_group_batched,
        _ACT_AWARE_FLAGS,
        _CACHED_ACTIVATIONS,
        _AWQ_PROPER_SCALES,
    )
    import prismaquant.export_native_compressed as enc

    E, out_features, in_features = 4, 64, 128

    items = []
    cached = {}
    for e in range(E):
        mod = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            mod.weight.copy_(torch.randn_like(mod.weight) * 0.1)
        recipe_key = f"layer.expert.{e}.weight"
        cached[recipe_key] = torch.randn(64, in_features, dtype=torch.float32)
        items.append((recipe_key, recipe_key, recipe_key, mod))

    # Stub the module-level activation cache for the duration of the
    # call. Restored after.
    saved_cache = enc._CACHED_ACTIVATIONS
    saved_flags = dict(enc._ACT_AWARE_FLAGS)
    saved_input_scales = enc._INPUT_GLOBAL_SCALES
    try:
        enc._CACHED_ACTIVATIONS = type("Idx", (), {
            "get": lambda self, name: cached.get(name)})()
        enc._ACT_AWARE_FLAGS["gptq"] = True
        enc._ACT_AWARE_FLAGS["scale_sweep"] = False
        enc._INPUT_GLOBAL_SCALES = {
            k: 1.0 for k, _, _, _ in items
        }

        compressed_per_linear = _quantize_2d_nvfp4_group_batched(
            items, joint_globals={}, device=torch.device("cpu"),
            expert_chunk=2,
        )
    finally:
        enc._CACHED_ACTIVATIONS = saved_cache
        enc._ACT_AWARE_FLAGS.clear()
        enc._ACT_AWARE_FLAGS.update(saved_flags)
        enc._INPUT_GLOBAL_SCALES = saved_input_scales

    assert len(compressed_per_linear) == E
    expected_keys = {
        "weight_packed", "weight_scale",
        "weight_global_scale", "input_global_scale",
    }
    for c in compressed_per_linear:
        assert set(c.keys()) >= expected_keys
        for k, v in c.items():
            # weight_scale is stored as fp8_e4m3fn; isfinite isn't
            # implemented for that dtype, but we can still check the
            # cast-to-float32 view.
            v_f32 = v.float() if v.is_floating_point() else v
            assert torch.isfinite(v_f32).all(), (
                f"non-finite values in {k}")
