"""GGUF IQ-quant format tests: exact bpp/byte accounting, allocator/canonical
round-trip, imatrix behaviour, and — the load-bearing contract — bit-exact
agreement between the emulation QDQ and the export byte packers as decoded by
gguf-py (the llama.cpp reference reader)."""

import numpy as np
import pytest
import torch

from prismaquant.format_registry import get_format
from prismaquant.gguf_formats import gguf_pack, gguf_quantize_dequantize
from prismaquant.gguf_iq_formats import IQ_BLOCK_BYTES
from prismaquant.layer_config import canonicalize_format
from prismaquant.serving_profiles import load_serving_profile

gguf = pytest.importorskip("gguf")

# Exact ggml bpw = type_size*8/block_size.
IQ_BPW = {
    "IQ2_XXS": 66 * 8 / 256,
    "IQ2_XS": 74 * 8 / 256,
    "IQ2_S": 82 * 8 / 256,
    "IQ3_XXS": 98 * 8 / 256,
    "IQ3_S": 110 * 8 / 256,
    "IQ4_XS": 136 * 8 / 256,
    "IQ4_NL": 18 * 8 / 32,
}
_NAMES = sorted(IQ_BPW)
_GRID_NAMES = ["IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S"]


def _weights(rows=64, cols=1024, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g, dtype=torch.float32)
    return w * torch.rand(rows, 1, generator=g).exp()


def test_block_bytes_match_gguf_sizes():
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    for name, (block, ts) in IQ_BLOCK_BYTES.items():
        qt = getattr(GGMLQuantizationType, name)
        assert GGML_QUANT_SIZES[qt] == (block, ts)


@pytest.mark.parametrize("name", _NAMES)
def test_effective_bits_exact(name):
    spec = get_format(name)
    bpw = IQ_BPW[name]
    assert spec.effective_bits_for_shape((64, 1024)) == pytest.approx(bpw, abs=1e-9)
    block, type_size = IQ_BLOCK_BYTES[name]
    assert type_size * 8 / block == pytest.approx(bpw)
    assert spec.memory_bytes_for_shape((64, 1024)) == 64 * 1024 // block * type_size


@pytest.mark.parametrize("name", _NAMES)
def test_canonicalize_round_trip(name):
    spec = get_format(name)
    assert canonicalize_format(spec.autoround_config()) == name
    assert canonicalize_format(name) == name
    assert canonicalize_format(name.lower()) == name


@pytest.mark.parametrize("name", _NAMES)
def test_pack_matches_emulation_bit_exact(name):
    """gguf-py dequantize(pack(w)) must equal our registry emulation exactly:
    the cost the allocator measures IS the artifact llama.cpp/vLLM serves."""
    w = _weights()
    packed = gguf_pack(w, name)
    block, type_size = IQ_BLOCK_BYTES[name]
    assert packed.shape == (64, 1024 // block * type_size)
    assert packed.dtype == np.uint8

    qt = getattr(gguf.GGMLQuantizationType, name)
    decoded = gguf.quants.dequantize(packed, qt)
    emulated = gguf_quantize_dequantize(w, name).numpy()
    np.testing.assert_array_equal(decoded, emulated)


@pytest.mark.parametrize("name", _NAMES)
def test_error_ladder_and_edge_cases(name):
    w = _weights()
    out = gguf_quantize_dequantize(w, name)
    rel = (out - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt()
    ceiling = {"IQ2_XXS": 0.42, "IQ2_XS": 0.36, "IQ2_S": 0.30,
               "IQ3_XXS": 0.22, "IQ3_S": 0.17, "IQ4_XS": 0.10,
               "IQ4_NL": 0.10}[name]
    assert 0 < float(rel) < ceiling

    zeros = torch.zeros(4, 512)
    assert gguf_quantize_dequantize(zeros, name).abs().sum() == 0

    bf16 = w.to(torch.bfloat16)
    assert gguf_quantize_dequantize(bf16, name).dtype == torch.bfloat16


@pytest.mark.parametrize("name", _GRID_NAMES)
def test_grid_formats_are_exact_fixed_points(name):
    """Re-quantizing the dequant of a grid format is a fixed point (the grid
    entries/signs/scales all re-select identically)."""
    w = _weights()
    out = gguf_quantize_dequantize(w, name)
    out2 = gguf_quantize_dequantize(out, name)
    torch.testing.assert_close(out2, out, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["IQ4_XS", "IQ4_NL"])
def test_iq4_idempotence_is_tight(name):
    """The non-linear-codebook scale re-estimation is not a bit-exact fixed
    point, but re-quantizing the dequant must stay within a tight RMS band."""
    w = _weights()
    out = gguf_quantize_dequantize(w, name)
    out2 = gguf_quantize_dequantize(out, name)
    rms = (out2 - out).pow(2).mean().sqrt() / out.pow(2).mean().sqrt()
    assert float(rms) < 0.01


def test_bpw_ladder_is_monotone():
    w = _weights(seed=3)
    errs = [
        (IQ_BPW[n], float((gguf_quantize_dequantize(w, n) - w).pow(2).mean()))
        for n in _NAMES
    ]
    errs.sort()
    for (_, lo), (_, hi) in zip(errs[1:], errs[:-1]):
        assert lo <= hi + 1e-9  # more bits => not-worse RTN error


@pytest.mark.parametrize("name", _NAMES)
def test_imatrix_changes_result_and_reduces_weighted_mse(name):
    w = _weights()
    g = torch.Generator().manual_seed(7)
    qw = torch.rand(1024, generator=g) + 0.05

    packed = gguf_pack(w, name, col_weights=qw)
    qt = getattr(gguf.GGMLQuantizationType, name)
    decoded = gguf.quants.dequantize(packed, qt)
    emulated = gguf_quantize_dequantize(w, name, col_weights=qw).numpy()
    np.testing.assert_array_equal(decoded, emulated)

    unweighted = gguf_quantize_dequantize(w, name)
    assert not np.array_equal(emulated, unweighted.numpy())

    wmse_u = float((qw * (unweighted - w) ** 2).mean())
    wmse_w = float((qw * (torch.from_numpy(emulated) - w) ** 2).mean())
    assert wmse_w < wmse_u


@pytest.mark.parametrize("name", ["IQ2_XXS", "IQ2_S", "IQ3_XXS", "IQ4_XS"])
def test_dead_imatrix_columns_do_not_erase_weights(name):
    w = _weights(rows=4)
    g = torch.Generator().manual_seed(9)
    qw = torch.rand(1024, generator=g) + 0.05
    qw[:256] = 0.0  # a whole superblock column-range never activated
    out = gguf_quantize_dequantize(w, name, col_weights=qw)
    assert out[:, :256].abs().sum() > 0
    packed = gguf_pack(w, name, col_weights=qw)
    decoded = gguf.quants.dequantize(packed, getattr(gguf.GGMLQuantizationType, name))
    np.testing.assert_array_equal(decoded, out.numpy())


@pytest.mark.parametrize("name", _NAMES)
def test_batched_cost_path_matches_unbatched(name):
    """The batched cost measurement must equal per-slice registry QDQ, else
    the allocator's cost diverges from the shipped bytes."""
    from prismaquant.measure_quant_cost import _batched_quantize

    spec = get_format(name)
    g = torch.Generator().manual_seed(2)
    stacked = torch.randn(3, 8, 512, generator=g) * torch.rand(3, 1, 1, generator=g).exp()
    batched = _batched_quantize(spec, stacked)
    per_slice = torch.stack(
        [gguf_quantize_dequantize(stacked[i], name) for i in range(3)]
    )
    torch.testing.assert_close(batched, per_slice, rtol=0, atol=0)

    qw = torch.rand(3, 1, 512, generator=g) + 0.05
    batched_w = _batched_quantize(spec, stacked, col_weights=qw)
    per_slice_w = torch.stack([
        gguf_quantize_dequantize(stacked[i], name, col_weights=qw[i, 0])
        for i in range(3)
    ])
    torch.testing.assert_close(batched_w, per_slice_w, rtol=0, atol=0)


def test_pack_handles_stacked_expert_tensors():
    w = torch.randn(4, 8, 512)
    packed = gguf_pack(w, "IQ3_XXS")
    block, type_size = IQ_BLOCK_BYTES["IQ3_XXS"]
    assert packed.shape == (4, 8, 512 // block * type_size)


def test_qdq_pads_odd_shapes_but_256_pack_refuses():
    w = _weights(cols=1000)  # not a multiple of 256
    out = gguf_quantize_dequantize(w, "IQ2_XXS")
    assert out.shape == w.shape
    with pytest.raises(ValueError, match="multiple of 256"):
        gguf_pack(w, "IQ2_XXS")


def test_iq4_nl_is_the_block32_rung():
    """IQ4_NL is the only IQ rung usable when in_features % 256 != 0."""
    w = _weights(cols=1024 + 32)  # multiple of 32, not of 256
    packed = gguf_pack(w, "IQ4_NL")
    block, type_size = IQ_BLOCK_BYTES["IQ4_NL"]
    assert packed.shape == (64, (1024 + 32) // block * type_size)
    decoded = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.IQ4_NL)
    emulated = gguf_quantize_dequantize(w, "IQ4_NL").numpy()
    np.testing.assert_array_equal(decoded, emulated)


def test_gguf_serving_profile_gates_iq_formats_and_shapes():
    profile = load_serving_profile("gguf")
    q = "model.layers.0.mlp.down_proj"

    for name in _NAMES:
        assert profile.check_format(q, name).legal
    assert not profile.check_format(q, "NVFP4").legal

    # 256-superblock IQ types need in_features % 256 == 0.
    assert profile.check_shape("IQ3_XXS", qname=q, in_features=1024,
                               out_features=512).legal
    assert not profile.check_shape("IQ3_XXS", qname=q, in_features=1000,
                                   out_features=512).legal
    assert not profile.check_shape("IQ3_XXS", qname=q, in_features=1056,
                                   out_features=512).legal  # %32 not %256
    # IQ4_NL is legal at %32 but not arbitrary widths.
    assert profile.check_shape("IQ4_NL", qname=q, in_features=1056,
                               out_features=512).legal
    assert not profile.check_shape("IQ4_NL", qname=q, in_features=1000,
                                   out_features=512).legal
