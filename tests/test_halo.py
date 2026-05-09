"""Unit tests for the HALO rotation preprocessor.

Critical correctness check: applying HALO rotation must produce
mathematically identical model outputs (up to FP32 numerical tolerance)
because the rotation is a no-op at runtime — every R cancels with R^T
along its path through the network.

If this test fails, the rotation is incorrectly absorbed and quantizing
the resulting weights would silently corrupt the model.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant.halo import (
    apply_halo_to_head,
    apply_halo_to_layer,
    apply_halo_rotation,
    block_hadamard_matrix,
    block_specs_for_layer,
    default_block_specs,
    fold_gamma_into_linears,
    hadamard_matrix,
    random_hadamard,
)


# ---------------------------------------------------------------------------
# Property tests for the Hadamard generators
# ---------------------------------------------------------------------------

def test_hadamard_matrix_orthogonal_pow2():
    for d in (2, 4, 8, 16, 64, 256):
        H = hadamard_matrix(d, dtype=torch.float64)
        I = torch.eye(d, dtype=torch.float64)
        assert torch.allclose(H @ H.t(), I, atol=1e-9), \
            f"Hadamard d={d} not orthogonal"
        assert torch.allclose(H.t() @ H, I, atol=1e-9), \
            f"Hadamard d={d} not orthogonal (transpose)"


def test_hadamard_matrix_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power-of-2"):
        hadamard_matrix(12)


def test_block_hadamard_matrix_supports_non_power_of_two():
    H = block_hadamard_matrix(12, dtype=torch.float64)
    I = torch.eye(12, dtype=torch.float64)
    assert torch.allclose(H @ H.t(), I, atol=1e-9)
    # Greedy partition is 8+4, so the off-block regions remain zero.
    assert torch.count_nonzero(H[:8, 8:]).item() == 0
    assert torch.count_nonzero(H[8:, :8]).item() == 0


def test_random_hadamard_orthogonal():
    for d in (8, 12, 16, 64, 96, 4096):
        R = random_hadamard(d, seed=42, dtype=torch.float64)
        I = torch.eye(d, dtype=torch.float64)
        assert torch.allclose(R @ R.t(), I, atol=1e-9), \
            f"random Hadamard d={d} not orthogonal"


def test_random_hadamard_seed_reproducible():
    R1 = random_hadamard(64, seed=7)
    R2 = random_hadamard(64, seed=7)
    R3 = random_hadamard(64, seed=8)
    assert torch.equal(R1, R2)
    assert not torch.equal(R1, R3)


# ---------------------------------------------------------------------------
# RMSNorm gamma folding
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """Minimal RMSNorm for testing."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class _Qwen35RMSNorm(nn.Module):
    """Offset-residual RMSNorm matching transformers Qwen3_5RMSNorm.

    Forward computes `(1 + weight) * normalize(x)`; weight is initialized to
    zeros and learned as a centered residual around 1. HALO's gamma fold must
    treat the effective gamma as `1 + weight` rather than `weight`, otherwise
    the post-fold residual stream is mismultiplied by `(2 * weight) / (1 +
    weight)` per fold site (catastrophic for small pretrained weights).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Match transformers' init: zeros, not ones.
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (1.0 + self.weight) * x


class _NormThenLinear(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = _RMSNorm(d)
        self.lin = nn.Linear(d, d, bias=False)
        # Initialize gamma to non-trivial values so the test is meaningful.
        with torch.no_grad():
            self.norm.weight.copy_(
                torch.linspace(0.5, 1.5, d, dtype=torch.float32))

    def forward(self, x):
        return self.lin(self.norm(x))


class _OffsetResidualNormThenLinear(nn.Module):
    """`_NormThenLinear` variant using `_Qwen35RMSNorm` with non-trivial
    learned residual."""
    def __init__(self, d: int):
        super().__init__()
        self.norm = _Qwen35RMSNorm(d)
        self.lin = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            # Centered residual ~0.0 with light spread, mimicking pretrained
            # qwen3_5 magnitudes (effective gamma = 1 + weight ≈ [0.7, 1.3]).
            self.norm.weight.copy_(
                torch.linspace(-0.3, 0.3, d, dtype=torch.float32))

    def forward(self, x):
        return self.lin(self.norm(x))


def test_fold_gamma_preserves_output():
    torch.manual_seed(0)
    d = 32
    model = _NormThenLinear(d).eval()
    x = torch.randn(4, d, dtype=torch.float32)
    with torch.no_grad():
        y_before = model(x)
        fold_gamma_into_linears(model, "norm", ["lin"])
        y_after = model(x)
    # Norm.weight is now 1.0 everywhere; gamma is in lin.weight.
    assert torch.allclose(model.norm.weight, torch.ones(d), atol=1e-7)
    assert torch.allclose(y_before, y_after, atol=1e-5), \
        f"max diff = {(y_before - y_after).abs().max().item()}"


def test_fold_gamma_offset_residual_preserves_output():
    """Offset-residual (Qwen3_5-style) RMSNorm fold must absorb effective
    gamma `(1 + weight)` and reset `weight = 0` post-fold so the norm computes
    identity scaling. Verifies the patch that fixes the qwen3_5_dense HALO
    smoke regression on 2026-05-09."""
    torch.manual_seed(0)
    d = 32
    model = _OffsetResidualNormThenLinear(d).eval()
    x = torch.randn(4, d, dtype=torch.float32)
    with torch.no_grad():
        y_before = model(x)
        fold_gamma_into_linears(model, "norm", ["lin"], offset_residual=True)
        y_after = model(x)
    # For offset-residual norms, post-fold weight must be 0 (so `(1 + 0) = 1`
    # — identity scaling), NOT 1 (which would scale by 2).
    assert torch.allclose(model.norm.weight, torch.zeros(d), atol=1e-7), (
        f"offset-residual norm.weight should reset to 0; got "
        f"{model.norm.weight}")
    assert torch.allclose(y_before, y_after, atol=1e-5), \
        f"max diff = {(y_before - y_after).abs().max().item()}"


class Qwen3_5RMSNorm(nn.Module):
    """Test-local class whose name matches the production transformers
    `Qwen3_5RMSNorm`, so HALO's class-name-based auto-detect of the
    offset-residual convention fires without importing transformers."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (1.0 + self.weight) * x


def test_fold_gamma_offset_residual_autodetects_qwen35_class():
    """`fold_gamma_into_linears` should auto-detect the offset-residual
    convention when the norm class name matches the production
    `Qwen3_5RMSNorm`. No explicit override should be required."""
    torch.manual_seed(0)
    d = 32

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = Qwen3_5RMSNorm(d)
            self.lin = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.norm.weight.copy_(
                    torch.linspace(-0.3, 0.3, d, dtype=torch.float32))

        def forward(self, x):
            return self.lin(self.norm(x))

    model = _Wrapper().eval()
    x = torch.randn(4, d, dtype=torch.float32)
    with torch.no_grad():
        y_before = model(x)
        # No explicit override — auto-detect via class name must fire.
        fold_gamma_into_linears(model, "norm", ["lin"])
        y_after = model(x)
    assert torch.allclose(model.norm.weight, torch.zeros(d), atol=1e-7), (
        f"offset-residual auto-detect failed; norm.weight={model.norm.weight}")
    assert torch.allclose(y_before, y_after, atol=1e-5), \
        f"max diff = {(y_before - y_after).abs().max().item()}"


def test_fold_gamma_llama_style_unaffected_by_patch():
    """Regression guard: the Llama-style fold must still work the same
    (weight=1 post-fold, output preserved) when offset_residual is left at
    default and the norm class isn't in the offset-residual whitelist."""
    torch.manual_seed(0)
    d = 32
    model = _NormThenLinear(d).eval()
    x = torch.randn(4, d, dtype=torch.float32)
    with torch.no_grad():
        y_before = model(x)
        fold_gamma_into_linears(model, "norm", ["lin"])  # auto-detect: False
        y_after = model(x)
    # Standard Llama-style fold: weight should land at 1.0.
    assert torch.allclose(model.norm.weight, torch.ones(d), atol=1e-7)
    assert torch.allclose(y_before, y_after, atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end rotation invariance on a tiny standard-transformer-like model
# ---------------------------------------------------------------------------

class _Attn(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        # Single-head attention for simplicity.
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        d = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
        attn = torch.softmax(scores, dim=-1)
        return self.o_proj(attn @ V)


class _MLP(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))
                              * self.up_proj(x))


class _DecoderLayer(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.input_layernorm = _RMSNorm(d)
        self.self_attn = _Attn(d)
        self.post_attention_layernorm = _RMSNorm(d)
        self.mlp = _MLP(d, ff)
        # Non-trivial gammas so folding actually does work.
        with torch.no_grad():
            self.input_layernorm.weight.copy_(
                torch.linspace(0.7, 1.3, d, dtype=torch.float32))
            self.post_attention_layernorm.weight.copy_(
                torch.linspace(0.6, 1.4, d, dtype=torch.float32))

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _ModelInner(nn.Module):
    def __init__(self, d: int, ff: int, n_layers: int, vocab: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_DecoderLayer(d, ff)
                                     for _ in range(n_layers)])
        self.norm = _RMSNorm(d)
        with torch.no_grad():
            self.norm.weight.copy_(
                torch.linspace(0.8, 1.2, d, dtype=torch.float32))

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _TinyModel(nn.Module):
    def __init__(self, d: int = 32, ff: int = 64, n_layers: int = 2,
                 vocab: int = 100):
        super().__init__()
        self.model = _ModelInner(d, ff, n_layers, vocab)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        h = self.model(ids)
        return self.lm_head(h)


def test_halo_rotation_preserves_output():
    """The gold-standard correctness test: apply HALO, verify the model
    produces identical logits."""
    torch.manual_seed(0)
    model = _TinyModel(d=32, ff=64, n_layers=2, vocab=100).eval()
    ids = torch.randint(0, 100, (3, 16))
    with torch.no_grad():
        logits_before = model(ids)
        spec = default_block_specs(model)
        # spec should pick up 2 layers × 2 blocks each = 4 block specs
        assert len(spec.block_specs) == 4, \
            f"expected 4 block specs, got {len(spec.block_specs)}"
        R = apply_halo_rotation(model, spec, seed=7)
        # Verify R is orthogonal
        assert torch.allclose(R @ R.t(), torch.eye(32), atol=1e-5)
        logits_after = model(ids)
    diff = (logits_before - logits_after).abs().max().item()
    # FP32 cancellation errors compound through 2 layers × 2 blocks
    # but should stay well under a sensible tolerance.
    assert diff < 1e-3, (
        f"HALO rotation broke model output: max abs diff = {diff}\n"
        "This indicates incorrect absorption math or an architecture "
        "mismatch between default_block_specs and the actual model.")


def test_halo_rotation_preserves_output_non_power_of_two_hidden():
    torch.manual_seed(0)
    model = _TinyModel(d=24, ff=48, n_layers=1, vocab=80).eval()
    ids = torch.randint(0, 80, (2, 8))
    with torch.no_grad():
        logits_before = model(ids)
        spec = default_block_specs(model)
        R = apply_halo_rotation(model, spec, seed=11)
        assert torch.allclose(R @ R.t(), torch.eye(24), atol=1e-5)
        logits_after = model(ids)
    diff = (logits_before - logits_after).abs().max().item()
    assert diff < 1e-3


def test_halo_actually_changes_weights():
    """Sanity check: rotation must actually modify weights, not be a
    silent no-op (which would also pass `test_halo_rotation_preserves_output`).
    """
    torch.manual_seed(0)
    model = _TinyModel(d=32, ff=64, n_layers=1, vocab=100).eval()
    q_before = model.model.layers[0].self_attn.q_proj.weight.clone()
    o_before = model.model.layers[0].self_attn.o_proj.weight.clone()
    with torch.no_grad():
        spec = default_block_specs(model)
        apply_halo_rotation(model, spec, seed=7)
    q_after = model.model.layers[0].self_attn.q_proj.weight
    o_after = model.model.layers[0].self_attn.o_proj.weight
    assert not torch.allclose(q_before, q_after), \
        "q_proj weight unchanged — rotation didn't apply"
    assert not torch.allclose(o_before, o_after), \
        "o_proj weight unchanged — rotation didn't apply"


def test_streaming_layer_halo_strict_rejects_unsupported_topology():
    model = nn.Module()
    model.layer = nn.Module()
    with pytest.raises(RuntimeError, match="no supported standard"):
        apply_halo_to_layer(model, model.layer, "layer", torch.eye(8))


def test_qwen_linear_attention_layer_specs_are_rotated():
    class _LinearAttn(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.in_proj_qkv = nn.Linear(d, d, bias=False)
            self.in_proj_z = nn.Linear(d, d, bias=False)
            self.in_proj_b = nn.Linear(d, 4, bias=False)
            self.in_proj_a = nn.Linear(d, 4, bias=False)
            self.out_proj = nn.Linear(d, d, bias=False)

    class _Layer(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.input_layernorm = _RMSNorm(d)
            self.linear_attn = _LinearAttn(d)
            self.post_attention_layernorm = _RMSNorm(d)
            self.mlp = _MLP(d, d * 2)

    layer = _Layer(16)
    specs = block_specs_for_layer(layer, "model.layers.0", 16)
    names = {s.name for s in specs}
    assert "model.layers.0.linear_attn" in names
    lin = next(s for s in specs if s.name == "model.layers.0.linear_attn")
    assert "model.layers.0.linear_attn.in_proj_qkv" in lin.input_linears
    assert "model.layers.0.linear_attn.in_proj_a" in lin.input_linears
    assert lin.output_linears == ["model.layers.0.linear_attn.out_proj"]


def test_head_halo_strict_rejects_tied_embeddings():
    class _HeadOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(16, 8)
            self.lm_head = nn.Linear(8, 16, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight

    with pytest.raises(RuntimeError, match="tied"):
        apply_halo_to_head(
            _HeadOnly(), torch.eye(8), final_norm_qname=None)


def test_export_halo_validator_rejects_unsupported_configs():
    from prismaquant.export_native_compressed import (
        _validate_halo_export_support,
    )

    default_profile = SimpleNamespace(name="default", has_mtp=lambda: False)
    qwen35_profile = SimpleNamespace(name="qwen3_5", has_mtp=lambda: False)
    qwen35_dense_mtp_profile = SimpleNamespace(
        name="qwen3_5_dense", has_mtp=lambda: True)
    tied_cfg = SimpleNamespace(hidden_size=64, tie_word_embeddings=True)
    untied_cfg = SimpleNamespace(hidden_size=64, tie_word_embeddings=False)
    nonpow_cfg = SimpleNamespace(hidden_size=96, tie_word_embeddings=False)

    with pytest.raises(RuntimeError, match="tied embeddings"):
        _validate_halo_export_support(default_profile, tied_cfg, 64)
    with pytest.raises(RuntimeError, match="profile 'qwen3_5'"):
        _validate_halo_export_support(qwen35_profile, untied_cfg, 64)
    _validate_halo_export_support(default_profile, nonpow_cfg, 96)
    _validate_halo_export_support(qwen35_dense_mtp_profile, nonpow_cfg, 96)


def test_export_halo_rejects_prerendered_production_cache():
    from prismaquant.export_native_compressed import (
        _validate_halo_cache_inputs,
    )

    _validate_halo_cache_inputs(
        SimpleNamespace(halo_mode="off", production_weight_cache="cache.pkl")
    )
    _validate_halo_cache_inputs(
        SimpleNamespace(halo_mode="random", production_weight_cache=None)
    )
    with pytest.raises(RuntimeError, match="production-weight-cache"):
        _validate_halo_cache_inputs(
            SimpleNamespace(
                halo_mode="random",
                production_weight_cache="cache.pkl",
            )
        )
