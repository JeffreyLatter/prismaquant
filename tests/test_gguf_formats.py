"""GGUF k-quant format tests: exact bpp accounting, allocator round-trip,
and bit-exact agreement between emulation QDQ and the export byte packers
as decoded by gguf-py (the llama.cpp reference reader)."""

import numpy as np
import pytest
import torch

from prismaquant.format_registry import get_format
from prismaquant.gguf_formats import (
    GGUF_BLOCK_BYTES,
    gguf_pack,
    gguf_quantize_dequantize,
)
from prismaquant.layer_config import canonicalize_format
from prismaquant.serving_profiles import load_serving_profile

gguf = pytest.importorskip("gguf")

GGUF_BPW = {
    "Q2_K": 2.625,
    "Q3_K": 3.4375,
    "Q4_K": 4.5,
    "Q5_K": 5.5,
    "Q6_K": 6.5625,
    "Q8_0": 8.5,
}


def _weights(rows=64, cols=1024, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g, dtype=torch.float32)
    return w * torch.rand(rows, 1, generator=g).exp()


@pytest.mark.parametrize("name,bpw", sorted(GGUF_BPW.items()))
def test_effective_bits_exact(name, bpw):
    spec = get_format(name)
    assert spec.effective_bits_for_shape((64, 1024)) == pytest.approx(bpw, abs=1e-9)
    block, type_size = GGUF_BLOCK_BYTES[name]
    assert type_size * 8 / block == pytest.approx(bpw)
    assert spec.memory_bytes_for_shape((64, 1024)) == 64 * 1024 // block * type_size


@pytest.mark.parametrize("name", sorted(GGUF_BPW))
def test_canonicalize_round_trip(name):
    spec = get_format(name)
    assert canonicalize_format(spec.autoround_config()) == name
    assert canonicalize_format(name) == name
    assert canonicalize_format(name.lower()) == name


def test_canonicalize_rejects_unknown_gguf_type():
    with pytest.raises(ValueError):
        canonicalize_format({"data_type": "gguf", "gguf_type": "IQ9_Z", "bits": 2})


@pytest.mark.parametrize("name", sorted(GGUF_BPW))
def test_pack_matches_emulation_bit_exact(name):
    """gguf-py dequantize(pack(w)) must equal our registry emulation exactly:
    the cost the allocator measures IS the artifact llama.cpp/vLLM serves."""
    w = _weights()
    packed = gguf_pack(w, name)
    block, type_size = GGUF_BLOCK_BYTES[name]
    assert packed.shape == (64, 1024 // block * type_size)
    assert packed.dtype == np.uint8

    qt = getattr(gguf.GGMLQuantizationType, name)
    decoded = gguf.quants.dequantize(packed, qt)
    emulated = gguf_quantize_dequantize(w, name).numpy()
    np.testing.assert_array_equal(decoded, emulated)


@pytest.mark.parametrize("name", sorted(GGUF_BPW))
def test_error_ladder_and_edge_cases(name):
    w = _weights()
    out = gguf_quantize_dequantize(w, name)
    rel = (out - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt()
    # RTN error floors: monotone-ish ladder, sane magnitudes.
    ceiling = {"Q2_K": 0.45, "Q3_K": 0.25, "Q4_K": 0.11, "Q5_K": 0.06,
               "Q6_K": 0.035, "Q8_0": 0.01}[name]
    assert 0 < float(rel) < ceiling

    zeros = torch.zeros(4, 512)
    assert gguf_quantize_dequantize(zeros, name).abs().sum() == 0

    bf16 = w.to(torch.bfloat16)
    assert gguf_quantize_dequantize(bf16, name).dtype == torch.bfloat16


def test_qdq_pads_odd_shapes_but_pack_refuses():
    w = _weights(cols=1000)  # not a multiple of 256
    out = gguf_quantize_dequantize(w, "Q2_K")
    assert out.shape == w.shape
    with pytest.raises(ValueError, match="multiple of 256"):
        gguf_pack(w, "Q2_K")


def test_pack_handles_stacked_expert_tensors():
    w = torch.randn(4, 8, 512)  # (experts, out, in)
    packed = gguf_pack(w, "Q3_K")
    block, type_size = GGUF_BLOCK_BYTES["Q3_K"]
    assert packed.shape == (4, 8, 512 // block * type_size)


def test_gguf_serving_profile_gates_formats_and_shapes():
    profile = load_serving_profile("gguf")

    assert profile.check_format("model.layers.0.mlp.down_proj", "Q2_K").legal
    assert profile.check_format("model.layers.0.mlp.down_proj", "BF16").legal
    assert not profile.check_format("model.layers.0.mlp.down_proj", "NVFP4").legal
    assert not profile.check_format("model.layers.0.mlp.down_proj", "FP8_E4M3").legal

    assert profile.check_shape(
        "Q2_K", qname="model.layers.0.mlp.down_proj",
        in_features=1024, out_features=512,
    ).legal
    assert not profile.check_shape(
        "Q2_K", qname="model.layers.0.mlp.down_proj",
        in_features=1000, out_features=512,
    ).legal
    assert profile.check_shape(
        "Q8_0", qname="model.layers.0.self_attn.q_proj",
        in_features=1024, out_features=512,
    ).legal
