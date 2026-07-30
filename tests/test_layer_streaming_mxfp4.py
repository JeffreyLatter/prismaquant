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
                           declare_fp4: bool = True,
                           scale_fmt: str | None = "ue8m0"):
    """Minimal DSv4-style checkpoint: `.scale` siblings, flat naming,
    `expert_dtype` declared at config top level (as DeepSeek-V4-Flash
    ships it) next to a block-FP8 quantization_config.

    `scale_fmt=None` omits the scale-format declaration (real DSv4-Flash
    ships `expert_dtype` without one)."""
    cfg = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }
    if scale_fmt is not None:
        cfg["quantization_config"]["scale_fmt"] = scale_fmt
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
        _check_mxfp4_packed_grid("t", w, torch.zeros(4, 4, dtype=torch.uint8))
    # Declared MXFP4 but the weight is not an int8 nibble-pack.
    with pytest.raises(ValueError, match="declared MXFP4"):
        _check_mxfp4_packed_grid(
            "t", torch.zeros(4, 32, dtype=torch.bfloat16),
            torch.zeros(4, 2, dtype=torch.uint8))


def test_declared_scale_plane_must_be_e8m0_dtype():
    """Step 3b reinterprets the scale sibling as a raw E8M0 byte plane
    (`view(torch.uint8)` + `exp2(b - 127)`). float8_e4m3fn is the same
    width, so a width check would let it through and decode every block at
    a wrong power-of-two scale. uint8 / int8 / float8_e8m0fnu pass."""
    w = torch.zeros(4, 32, dtype=torch.int8)
    with pytest.raises(ValueError, match="declared MXFP4"):
        _check_mxfp4_packed_grid(
            "t", w, torch.zeros(4, 2).to(torch.float8_e4m3fn))
    # fp32 scale planes are rejected too (they are not a byte plane).
    with pytest.raises(ValueError, match="declared MXFP4"):
        _check_mxfp4_packed_grid("t", w, torch.zeros(4, 2))
    for dt in (torch.uint8, torch.int8, torch.float8_e8m0fnu):
        _check_mxfp4_packed_grid("t", w, torch.zeros(4, 2).to(dt))


def test_declared_non_e8m0_scale_fmt_raises(tmp_path):
    """The decode hardcodes the E8M0 exponent interpretation, so a
    checkpoint that declares a different scale encoding must fail loudly
    at map-build time rather than have its bytes reinterpreted."""
    packed, scale = _rand_expert()
    _write_dsv4_checkpoint(tmp_path, {0: (packed, scale)}, scale_fmt="e4m3")
    with pytest.raises(ValueError, match="scale_fmt"):
        _build_fp8_scale_inv_map(str(tmp_path))


def test_missing_scale_fmt_is_not_fatal(tmp_path):
    """Real DSv4-Flash declares `expert_dtype` with no scale-format field;
    the per-tensor dtype allow-list is what guards the reinterpretation."""
    packed, scale = _rand_expert()
    _write_dsv4_checkpoint(tmp_path, {0: (packed, scale)}, scale_fmt=None)
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    assert fp8_map.mxfp4_names == {_live(0)}
    out = {_live(0): packed.clone()}
    assert _apply_fp8_dequant_inplace(out, fp8_map, CPU) == 1
    _assert_bitwise_equal_bf16(
        out[_live(0)], _scalar_reference_decode(packed, scale))


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


# ---------------------------------------------------------------------------
# resident-size estimators price declared-MXFP4 I8 as 2 elements/byte
# ---------------------------------------------------------------------------

def test_layer_cache_estimate_prices_declared_fp4_experts_4x(tmp_path):
    """`_estimate_layer_cache_bytes` must price declared-MXFP4 I8 experts
    at 2 logical elements x target dtype per packed byte (4x at bf16);
    an undeclared checkpoint keeps the verbatim 1 byte/elem. The 4x
    undercount made prepare_for_load() under-evict and prefetch refuse
    layers that actually fit."""
    from prismaquant.streaming_model import _estimate_layer_cache_bytes
    key_e = "model.layers.0.mlp.experts.0.gate_proj.weight"
    key_d = "model.layers.0.self_attn.q_proj.weight"
    shard = str(tmp_path / "model.safetensors")
    save_file({key_e: torch.zeros(64, 32, dtype=torch.int8),
               key_d: torch.zeros(16, 16, dtype=torch.bfloat16)}, shard)
    kw = dict(weight_shard={key_e: shard, key_d: shard},
              weight_ckpt={key_e: key_e, key_d: key_d},
              layers_prefix="model.layers.", num_layers=1,
              target_dtype=torch.bfloat16)
    est_fp4, sizes_fp4 = _estimate_layer_cache_bytes(fp4_experts=True, **kw)
    assert sizes_fp4[0] == 64 * 32 * 2 * 2 + 16 * 16 * 2
    est_i8, sizes_i8 = _estimate_layer_cache_bytes(fp4_experts=False, **kw)
    assert sizes_i8[0] == 64 * 32 * 1 + 16 * 16 * 2


@pytest.mark.parametrize("packed_dtype", [torch.int8, torch.uint8])
def test_layer_cache_estimate_matches_real_resident_bytes(tmp_path,
                                                          packed_dtype):
    """The pre-load estimate must equal what `LayerCache` actually accounts
    for the decoded tensor. `_check_mxfp4_packed_grid` accepts int8 AND
    uint8 nibble-packs, so both spellings must price at 2 logical elements
    x execution dtype per packed byte — sizing only I8 left a U8 checkpoint
    with the original 4x undercount that under-evicts and makes prefetch
    refuse layers."""
    from prismaquant.layer_streaming import LayerCache
    from prismaquant.streaming_model import _estimate_layer_cache_bytes

    rows, packed_in = 4, 32
    g = torch.Generator().manual_seed(11)
    packed = torch.randint(0, 256, (rows, packed_in), dtype=torch.uint8,
                           generator=g)
    if packed_dtype is torch.int8:
        packed = packed.view(torch.int8)
    scale = torch.full((rows, packed_in // 16), 127, dtype=torch.uint8)
    _write_dsv4_checkpoint(tmp_path, {0: (packed, scale)})

    # Real decode -> real LayerCache accounting.
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    assert fp8_map.mxfp4_names == {_live(0)}
    out = {_live(0): packed.clone()}
    assert _apply_fp8_dequant_inplace(out, fp8_map, CPU) == 1
    resident = LayerCache(max_bytes=1 << 30)._sizeof(out)

    shard = str(tmp_path / "model.safetensors")
    ckpt = "layers.0.ffn.experts.0.w1.weight"
    kw = dict(weight_shard={_live(0): shard},
              weight_ckpt={_live(0): ckpt},
              layers_prefix="model.layers.", num_layers=1,
              target_dtype=torch.bfloat16)
    _est, sizes = _estimate_layer_cache_bytes(fp4_experts=True, **kw)
    assert sizes[0] == resident == rows * packed_in * 2 * 2
    # No declaration -> genuine 1-byte integer tensor, sized verbatim.
    _est_u, sizes_undeclared = _estimate_layer_cache_bytes(
        fp4_experts=False, **kw)
    assert sizes_undeclared[0] == rows * packed_in * 1


# ---------------------------------------------------------------------------
# d) batched shape-group decode matches the reference across chunk
#    boundaries
# ---------------------------------------------------------------------------

def test_batched_decode_matches_reference_across_chunks(tmp_path, monkeypatch):
    from prismaquant import layer_streaming as LS
    monkeypatch.setattr(LS, "_MXFP4_DECODE_CHUNK", 2)
    # 5 experts of one shape (splits 2+2+1) — exercises chunk boundaries
    # and the shape-group stacking.
    experts = {eid: _rand_expert(rows=4, packed_in=32, seed=10 + eid)
               for eid in range(5)}
    _write_dsv4_checkpoint(tmp_path, experts)
    fp8_map = _build_fp8_scale_inv_map(str(tmp_path))
    assert fp8_map.mxfp4_names == {_live(e) for e in experts}
    out = {_live(e): p.clone() for e, (p, _s) in experts.items()}
    assert _apply_fp8_dequant_inplace(out, fp8_map, CPU) == 5
    for eid, (packed, scale) in experts.items():
        _assert_bitwise_equal_bf16(
            out[_live(eid)], _scalar_reference_decode(packed, scale))
