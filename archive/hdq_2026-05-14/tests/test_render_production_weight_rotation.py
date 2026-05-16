"""Integration tests for the cluster_rotation hook in render_production_weight.

Exercises the Hadamard-DuQuant rotation wrapper without requiring a full
model — small synthetic weights + activations suffice to validate the
rotation flow (W → W @ M^T → render → @ M back to original coords) and
the gate_trace emission.
"""
from __future__ import annotations

import pytest
import torch

from prismaquant.hadamard_duquant import (
    NVFP4_GROUP_SIZE,
    apply_block_rotation_input,
    sylvester_hadamard,
)
from prismaquant.production_weight_cache import render_production_weight


def _make_levers() -> dict[str, object]:
    """Minimal levers dict for BF16 render path (skips GPTQ/scale_sweep)."""
    return {
        "gptq": False,
        "gptq_damp_sweep": False,
        "scale_sweep": False,
        "awq_round": False,
        "act_clip_solver": False,
        "fisher_gptq": False,
        "fisher_clip": False,
        "nvfp4_scale_rule": "static_6",
    }


def test_render_with_cluster_rotation_returns_same_shape():
    """The rotation wrapper preserves the output shape."""
    torch.manual_seed(0)
    weight = torch.randn(8, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)
    activations = {"q": torch.randn(32, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)}
    M = sylvester_hadamard(NVFP4_GROUP_SIZE, dtype=torch.float32)
    result = render_production_weight(
        weight, "BF16",
        qname="q",
        activations=activations,
        levers=_make_levers(),
        cluster_rotation=M,
    )
    assert result.shape == weight.shape
    assert result.dtype == weight.dtype


def test_render_without_rotation_matches_baseline():
    """cluster_rotation=None reproduces the pre-Phase-3 behavior."""
    torch.manual_seed(0)
    weight = torch.randn(8, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)
    activations = {"q": torch.randn(32, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)}
    baseline = render_production_weight(
        weight, "BF16",
        qname="q",
        activations=activations,
        levers=_make_levers(),
    )
    with_none = render_production_weight(
        weight, "BF16",
        qname="q",
        activations=activations,
        levers=_make_levers(),
        cluster_rotation=None,
    )
    torch.testing.assert_close(baseline, with_none)


def test_render_with_identity_rotation_matches_baseline_bf16():
    """Identity rotation should be a no-op for the BF16 (passthrough) path."""
    torch.manual_seed(0)
    weight = torch.randn(8, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)
    activations = {"q": torch.randn(32, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)}
    identity = torch.eye(NVFP4_GROUP_SIZE, dtype=torch.float32)
    baseline = render_production_weight(
        weight, "BF16",
        qname="q",
        activations=activations,
        levers=_make_levers(),
    )
    rotated = render_production_weight(
        weight, "BF16",
        qname="q",
        activations=activations,
        levers=_make_levers(),
        cluster_rotation=identity,
    )
    # BF16 is passthrough — the round trip (W @ I) → render → @ I should
    # leave the weight unchanged regardless of rotation.
    torch.testing.assert_close(rotated, baseline, atol=1e-6, rtol=1e-6)


def test_render_with_rotation_emits_gate_trace_entry():
    """The rotation wrapper appends a 'hadamard_duquant' entry to gate_trace."""
    torch.manual_seed(0)
    weight = torch.randn(8, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)
    activations = {"q": torch.randn(32, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)}
    M = sylvester_hadamard(NVFP4_GROUP_SIZE, dtype=torch.float32)
    gate_trace: list[dict[str, object]] = []
    render_production_weight(
        weight, "BF16",
        qname="q",
        activations=activations,
        levers=_make_levers(),
        cluster_rotation=M,
        gate_trace=gate_trace,
    )
    rotation_entries = [
        e for e in gate_trace if e.get("mechanism") == "hadamard_duquant"
    ]
    assert len(rotation_entries) == 1
    entry = rotation_entries[0]
    assert entry["accepted"] is True
    assert entry["selected"] == "rot"
    assert entry["G"] == NVFP4_GROUP_SIZE
    assert entry["reason"] == "allocator_pick"


def test_render_with_rotation_bf16_round_trip_is_lossless():
    """For BF16 (no quantization), the rotation wrapper is algebraically
    lossless: W → W @ M^T → render(BF16, identity) → @ M = W.

    Validates the per-block apply / fold-back algebra round-trips correctly
    when the inner render does no quantization."""
    torch.manual_seed(0)
    weight = torch.randn(8, NVFP4_GROUP_SIZE * 2, dtype=torch.float64)
    weight32 = weight.to(torch.float32)
    activations = {
        "q": torch.randn(32, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)
    }
    M = sylvester_hadamard(NVFP4_GROUP_SIZE, dtype=torch.float32)
    result = render_production_weight(
        weight32, "BF16",
        qname="q",
        activations=activations,
        levers=_make_levers(),
        cluster_rotation=M,
    )
    # BF16 path is passthrough; the rotation should fold back to W.
    # Compare in float32 with a small tolerance for f32 round-trip error.
    torch.testing.assert_close(result, weight32, atol=1e-5, rtol=1e-5)


def test_render_with_rotation_preserves_unrotated_activations_untouched():
    """The activations dict is copied — original activation tensors are
    not mutated even though the renderer needs a rotated view of qname."""
    torch.manual_seed(0)
    weight = torch.randn(8, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)
    original_x = torch.randn(32, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)
    activations = {
        "q": original_x,
        "other": torch.randn(32, 16, dtype=torch.float32),
    }
    original_x_clone = original_x.clone()
    M = sylvester_hadamard(NVFP4_GROUP_SIZE, dtype=torch.float32)
    render_production_weight(
        weight, "BF16",
        qname="q",
        activations=activations,
        levers=_make_levers(),
        cluster_rotation=M,
    )
    # Original activation tensor should be untouched.
    torch.testing.assert_close(original_x, original_x_clone)
