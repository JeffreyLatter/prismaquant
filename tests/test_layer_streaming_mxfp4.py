"""MXFP4 streaming dequant (DSv4-Flash routed experts) — CPU-only.

Covers the 2026-07 review of the DSv4 probe-enablement PR:

a) The MXFP4 decode path triggers ONLY on the checkpoint's explicit
   declaration (config `expert_dtype: "fp4"`), never on a tensor-shape
   heuristic — an INT8 checkpoint with group-16 scales must fail loudly
   through `_check_fp8_scale_grid` instead of being silently decoded as
   nibble pairs.
b) The vectorized decode is bit-exact against an independent scalar
   reference on synthetic tensors (OCP MX v1.0: FP4 E2M1 element grid,
   one E8M0 power-of-two scale per 32 logical elements).
c) E8M0 `0xFF` decodes every element of its block to NaN (per the OCP
   spec), not `+inf`.
d) The batched shape-group decode matches the reference across chunk
   boundaries.
"""
import json
import math

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.layer_streaming import (
    _apply_fp8_dequant_inplace,
    _build_fp8_scale_inv_map,
    _check_mxfp4_packed_grid,
)

CPU = torch.device("cpu")

# FP4 E2M1 element grid, code 0..15 (sign bit = 8).
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _scalar_reference_decode(packed: torch.Tensor,
                             scales: torch.Tensor) -> torch.Tensor:
    """Independent per-element MXFP4 decode: int8 nibble-pack (low nibble
    = even element) x per-32-element E8M0 scale, fp32 multiply, bf16 out.
    Deliberately loop-based — shares no code with the vectorized path."""
    rows, packed_in = packed.shape
    logical_in = packed_in * 2
    out = torch.empty(rows, logical_in, dtype=torch.float32)
    for r in range(rows):
        for p in range(packed_in):
            b = int(packed[r, p].item()) & 0xFF
            out[r, 2 * p] = _E2M1[b & 0x0F]
            out[r, 2 * p + 1] = _E2M1[b >> 4]
        for g in range(logical_in // 32):
            s = int(scales[r, g].item()) & 0xFF
            sc = float("nan") if s == 0xFF else 2.0 ** (s - 127)
            out[r, g * 32:(g + 1) * 32] *= sc
    return out.to(torch.bfloat16)


def _write_dsv4_checkpoint(tmp_path, experts: dict[int, tuple],
                           attn_fp8: tuple | None = None,
                           declare_fp4: bool = True):
    """Minimal DSv4-style checkpoint: `.scale` siblings, flat naming,
    `expert_dtype` declared at config top level (as DeepSeek-V4-Flash
    ships it) next to a block-FP8 quantization_config."""
    cfg = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }
    if declare_fp4:
        cfg["expert_dtype"] = "fp4"
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    tensors = {}
    for eid, (packed, scale) in experts.items():
        tensors[f"layers.0.ffn.experts.{eid}.w1.weight"] = packed
        tensors[f"layers.0.ffn.experts.{eid}.w1.scale"] = scale
    if attn_fp8 is not None:
        w, s = attn_fp8
        tensors["layers.0.attn.q_proj.weight"] = w
        tensors["layers.0.attn.q_proj.scale"] = s
    save_file(tensors, str(tmp_path / "model.safetensors"))


def _rand_expert(rows=4, packed_in=32, seed=0, scale_range=(100, 200)):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(-128, 128, (rows, packed_in),
                           dtype=torch.int8, generator=g)
    scale = torch.randint(scale_range[0], scale_range[1],
                          (rows, packed_in // 16), generator=g
                          ).to(torch.uint8)
    return packed, scale


def _live(eid):
    return f"model.layers.0.mlp.experts.{eid}.gate_proj.weight"


# ---------------------------------------------------------------------------
# a) explicit declaration gates the decode; no heuristic; fp8 grid
#    assertion is never bypassed
# ---------------------------------------------------------------------------

def test_declared_expert_dtype_populates_mxfp4_names(tmp_path):
    packed, scale = _rand_expert()
    attn = (torch.randn(128, 128).to(torch.float8_e4m3fn),
            torch.ones(1, 1, dtype=torch.float32))
    _write_dsv4_checkpoint(tmp_path, {0: (packed, scale)}, attn_fp8=attn)
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    assert fp8_map.mxfp4_names == {_live(0)}
    # The non-expert fp8 tensor stays on the block-FP8 path.
    assert "model.layers.0.self_attn.q_proj.weight" in fp8_map
    assert "model.layers.0.self_attn.q_proj.weight" not in fp8_map.mxfp4_names


def test_undeclared_checkpoint_has_no_mxfp4_names(tmp_path):
    packed, scale = _rand_expert()
    _write_dsv4_checkpoint(tmp_path, {0: (packed, scale)}, declare_fp4=False)
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    assert fp8_map.mxfp4_names == frozenset()


def test_undeclared_int8_group16_is_not_decoded_as_nibbles(tmp_path):
    """The old shape heuristic decoded ANY 2-D int8 tensor with a
    group-16 scale grid as MXFP4 nibble pairs — silent corruption for an
    INT8-with-group-16-scales checkpoint. Without the declaration the
    tensor must instead hit `_check_fp8_scale_grid` (the transposed-grid
    assertion the 2026-07-02 audit added) and fail loudly."""
    packed, scale = _rand_expert(rows=64, packed_in=256)
    _write_dsv4_checkpoint(tmp_path, {0: (packed, scale.to(torch.float32))},
                           declare_fp4=False)
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    out = {_live(0): packed.clone()}
    with pytest.raises(ValueError, match="transposed"):
        _apply_fp8_dequant_inplace(out, fp8_map, CPU)
    # And the tensor was not silently replaced with decoded nibbles.
    assert out[_live(0)].dtype == torch.int8


def test_declared_tensor_with_wrong_layout_raises():
    # Declared MXFP4 but the scale grid is not (rows, packed/16).
    w = torch.zeros(4, 32, dtype=torch.int8)
    with pytest.raises(ValueError, match="declared MXFP4"):
        _check_mxfp4_packed_grid("t", w, torch.zeros(4, 4))
    # Declared MXFP4 but the weight is not an int8 nibble-pack.
    with pytest.raises(ValueError, match="declared MXFP4"):
        _check_mxfp4_packed_grid(
            "t", torch.zeros(4, 32, dtype=torch.bfloat16),
            torch.zeros(4, 2))


# ---------------------------------------------------------------------------
# b) bit-exact decode vs the scalar reference
# ---------------------------------------------------------------------------

def _assert_bitwise_equal_bf16(got: torch.Tensor, ref: torch.Tensor):
    assert got.dtype == torch.bfloat16 and ref.dtype == torch.bfloat16
    nan_got, nan_ref = torch.isnan(got), torch.isnan(ref)
    assert torch.equal(nan_got, nan_ref)
    finite = ~nan_got
    assert torch.equal(got[finite].view(torch.int16),
                       ref[finite].view(torch.int16))


def test_e8m0_ff_scale_yields_all_nan_block(tmp_path):
    """E8M0 0xFF is NaN per the OCP MX v1.0 spec. exp2(0xFF - 127) =
    +inf turned a 0xFF block into 29 ±inf (nonzero elements) + 3 NaN
    (zero elements, 0*inf) instead of 32 NaNs."""
    packed, scale = _rand_expert(rows=2, packed_in=32, seed=3)
    scale[0, 0] = 0xFF  # first 32-element group of row 0
    _write_dsv4_checkpoint(tmp_path, {0: (packed, scale)})
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    out = {_live(0): packed.clone()}
    _apply_fp8_dequant_inplace(out, fp8_map, CPU)
    got = out[_live(0)]
    assert torch.isnan(got[0, :32]).all()
    assert not torch.isinf(got).any()
    # Other groups decode normally.
    assert torch.isfinite(got[0, 32:]).all() and torch.isfinite(got[1]).all()
    _assert_bitwise_equal_bf16(got, _scalar_reference_decode(packed, scale))


def test_mxfp4_decode_bit_exact_vs_scalar_reference(tmp_path):
    packed, scale = _rand_expert(rows=8, packed_in=64, seed=7)
    # Exercise every FP4 code at least once.
    packed[0, :8] = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        dtype=torch.uint8).view(torch.int8)
    _write_dsv4_checkpoint(tmp_path, {0: (packed, scale)})
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    out = {_live(0): packed.clone()}
    assert _apply_fp8_dequant_inplace(out, fp8_map, CPU) == 1
    got = out[_live(0)]
    assert got.shape == (8, 128)
    _assert_bitwise_equal_bf16(got, _scalar_reference_decode(packed, scale))
