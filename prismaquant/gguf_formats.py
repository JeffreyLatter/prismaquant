"""GGUF k-quant weight formats: torch quantizers, emulation QDQ, byte packers.

GGUF k-quants are two-tier superblock formats along the input dimension:
a 256-element superblock carries an fp16 super-scale (``d``, plus ``dmin``
for the asymmetric types) and per-sub-block *quantized* scales (and mins).
The serving dequant is ``w = d*sc[i]*q - dmin*m[i]`` (asymmetric) or
``w = d*sc[i]*q`` (symmetric).

One field-quantizer per format is the single source of the quantization
math; the emulation ``quantize_dequantize`` (used by cost measurement) and
the byte packer (used by export) both consume its output, so measured
error and shipped bytes cannot diverge. Byte layouts are the exact
inverses of gguf-py's ``dequantize_blocks`` (validated bit-exact in
tests/test_gguf_formats.py).

Scale selection here is RTN-grade (min-max / max-abs, like the llama.cpp
reference quantizers but without their weighted grid search). Activation
emulation models the ggml MMQ/MMVQ compute path, which quantizes
activations to Q8_1 (per-32 symmetric int8).
"""
from __future__ import annotations

import numpy as np
import torch

QK_K = 256

# name -> (block_size, type_size_bytes); mirrors gguf.GGML_QUANT_SIZES.
GGUF_BLOCK_BYTES: dict[str, tuple[int, int]] = {
    "Q2_K": (QK_K, 84),
    "Q3_K": (QK_K, 110),
    "Q4_K": (QK_K, 144),
    "Q5_K": (QK_K, 176),
    "Q6_K": (QK_K, 210),
    "Q8_0": (32, 34),
}

# gguf.GGMLQuantizationType values (stable on-disk enum).
GGML_TYPE_IDS: dict[str, int] = {
    "F32": 0, "F16": 1, "Q8_0": 8,
    "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14,
    "BF16": 30,
}


def _fp16r(t: torch.Tensor) -> torch.Tensor:
    """Round through fp16 storage (the super-scales are stored as fp16)."""
    return t.to(torch.float16).to(torch.float32)


def _safe_inv(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t != 0, 1.0 / torch.where(t == 0, torch.ones_like(t), t),
                       torch.zeros_like(t))


def _round_half_away(t: torch.Tensor) -> torch.Tensor:
    """roundf() semantics (half away from zero), matching ggml/np_roundf."""
    return torch.sign(t) * torch.floor(t.abs() + 0.5)


# ---------------------------------------------------------------------------
# Field quantizers.  Input: (N, 256) float32 superblocks.  Output: dict of
# integer fields + fp16-rounded super-scales, everything needed to either
# reconstruct values or pack bytes.
# ---------------------------------------------------------------------------

def _fields_asym(blocks: torch.Tensor, sub: int, qmax: int,
                 scale_max: int) -> dict[str, torch.Tensor]:
    """Shared asymmetric two-tier quantizer (Q2_K sub=16, Q4_K/Q5_K sub=32).

    Per sub-block min-max affine onto q in [0, qmax]; sub-scale and sub-min
    quantized to [0, scale_max] under fp16 super-scales d, dmin.
    """
    n = blocks.shape[0]
    sb = blocks.reshape(n, QK_K // sub, sub)
    mn = sb.amin(dim=2).clamp_max(0.0)
    mx = sb.amax(dim=2).clamp_min(0.0)
    sub_min = -mn
    sub_scale = (mx - mn) / qmax

    d = _fp16r(sub_scale.amax(dim=1, keepdim=True) / scale_max)
    dmin = _fp16r(sub_min.amax(dim=1, keepdim=True) / scale_max)
    sc = torch.round(sub_scale * _safe_inv(d)).clamp(0, scale_max).to(torch.uint8)
    m = torch.round(sub_min * _safe_inv(dmin)).clamp(0, scale_max).to(torch.uint8)

    dl = (d * sc.float()).unsqueeze(-1)
    ml = (dmin * m.float()).unsqueeze(-1)
    q = torch.round((sb + ml) * _safe_inv(dl)).clamp(0, qmax).to(torch.uint8)
    return {"d": d, "dmin": dmin, "sc": sc, "m": m, "q": q.reshape(n, QK_K)}


def _recon_asym(f: dict[str, torch.Tensor], sub: int) -> torch.Tensor:
    n = f["q"].shape[0]
    dl = (f["d"] * f["sc"].float()).unsqueeze(-1)
    ml = (f["dmin"] * f["m"].float()).unsqueeze(-1)
    q = f["q"].reshape(n, QK_K // sub, sub).float()
    return (dl * q - ml).reshape(n, QK_K)


def _fields_q2_k(blocks: torch.Tensor) -> dict[str, torch.Tensor]:
    return _fields_asym(blocks, sub=16, qmax=3, scale_max=15)


def _fields_q4_k(blocks: torch.Tensor) -> dict[str, torch.Tensor]:
    return _fields_asym(blocks, sub=32, qmax=15, scale_max=63)


def _fields_q5_k(blocks: torch.Tensor) -> dict[str, torch.Tensor]:
    return _fields_asym(blocks, sub=32, qmax=31, scale_max=63)


def _fields_q3_k(blocks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Symmetric 3-bit: q in [-4, 3], 6-bit signed sub-scales, fp16 d."""
    n = blocks.shape[0]
    sb = blocks.reshape(n, QK_K // 16, 16)
    amax = sb.abs().amax(dim=2)
    sub_scale = amax / 4.0

    d = _fp16r(sub_scale.amax(dim=1, keepdim=True) / 31.0)
    sc = torch.round(sub_scale * _safe_inv(d)).clamp(-32, 31).to(torch.int8)

    dl = (d * sc.float()).unsqueeze(-1)
    q = torch.round(sb * _safe_inv(dl)).clamp(-4, 3).to(torch.int8)
    return {"d": d, "sc": sc, "q": q.reshape(n, QK_K)}


def _recon_q3_k(f: dict[str, torch.Tensor]) -> torch.Tensor:
    n = f["q"].shape[0]
    dl = (f["d"] * f["sc"].float()).unsqueeze(-1)
    return (dl * f["q"].reshape(n, QK_K // 16, 16).float()).reshape(n, QK_K)


def _fields_q6_k(blocks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Symmetric 6-bit: q in [-32, 31], int8 sub-scales, fp16 d."""
    n = blocks.shape[0]
    sb = blocks.reshape(n, QK_K // 16, 16)
    amax = sb.abs().amax(dim=2)
    sub_scale = amax / 32.0

    d = _fp16r(sub_scale.amax(dim=1, keepdim=True) / 127.0)
    sc = torch.round(sub_scale * _safe_inv(d)).clamp(-128, 127).to(torch.int8)

    dl = (d * sc.float()).unsqueeze(-1)
    q = torch.round(sb * _safe_inv(dl)).clamp(-32, 31).to(torch.int8)
    return {"d": d, "sc": sc, "q": q.reshape(n, QK_K)}


def _recon_q6_k(f: dict[str, torch.Tensor]) -> torch.Tensor:
    n = f["q"].shape[0]
    dl = (f["d"] * f["sc"].float()).unsqueeze(-1)
    return (dl * f["q"].reshape(n, QK_K // 16, 16).float()).reshape(n, QK_K)


def _fields_q8_0(blocks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-32 symmetric int8 (blocks input is (N, 32)); half-away rounding
    to stay bit-exact with the ggml/gguf-py reference quantizer."""
    d = _fp16r(blocks.abs().amax(dim=1, keepdim=True) / 127.0)
    q = _round_half_away(blocks * _safe_inv(d)).clamp(-128, 127).to(torch.int8)
    return {"d": d, "q": q}


def _recon_q8_0(f: dict[str, torch.Tensor]) -> torch.Tensor:
    return f["d"] * f["q"].float()


# ---------------------------------------------------------------------------
# Emulation QDQ (registry quantize_dequantize).  Pads the input dim with
# zeros when it is not a multiple of the block size — zero sub-blocks get
# zero scales and cannot perturb the real columns.
# ---------------------------------------------------------------------------

_FIELDS = {
    "Q2_K": (_fields_q2_k, lambda f: _recon_asym(f, 16), QK_K),
    "Q3_K": (_fields_q3_k, _recon_q3_k, QK_K),
    "Q4_K": (_fields_q4_k, lambda f: _recon_asym(f, 32), QK_K),
    "Q5_K": (_fields_q5_k, lambda f: _recon_asym(f, 32), QK_K),
    "Q6_K": (_fields_q6_k, _recon_q6_k, QK_K),
    "Q8_0": (_fields_q8_0, _recon_q8_0, 32),
}


def gguf_quantize_dequantize(w: torch.Tensor, fmt: str) -> torch.Tensor:
    fields_fn, recon_fn, block = _FIELDS[fmt]
    orig_shape = w.shape
    in_f = int(orig_shape[-1])
    flat = w.reshape(-1, in_f).to(torch.float32)
    pad = (-in_f) % block
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block)
    out = recon_fn(fields_fn(blocks)).reshape(flat.shape)
    if pad:
        out = out[:, :in_f]
    return out.reshape(orig_shape).to(w.dtype)


def make_gguf_qdq(fmt: str):
    def f(w: torch.Tensor) -> torch.Tensor:
        return gguf_quantize_dequantize(w, fmt)
    return f


# ---------------------------------------------------------------------------
# Byte packers (export path).  Layout-exact inverses of gguf-py
# dequantize_blocks; consume the same fields as the emulation.
# ---------------------------------------------------------------------------

def _pack_2bit(q: torch.Tensor) -> torch.Tensor:
    """(n, 256) values in [0,3] -> (n, 64) bytes.
    Element e -> byte (e//128)*32 + e%32, shift 2*((e%128)//32)."""
    n = q.shape[0]
    v = q.reshape(n, 2, 4, 32).to(torch.int32)
    shifts = torch.tensor([0, 2, 4, 6], device=q.device).view(1, 1, 4, 1)
    return (v << shifts).sum(dim=2).reshape(n, 64).to(torch.uint8)


def _pack_nibbles(q: torch.Tensor, chunk: int) -> torch.Tensor:
    """(n, 256) values in [0,15] -> (n, 128) bytes.
    chunk=32 (Q4/Q5: byte (e//64)*32+e%32, shift 4*((e%64)//32));
    chunk=64 (Q6 ql: byte (e//128)*64+e%64, shift 4*((e%128)//64))."""
    n = q.shape[0]
    v = q.reshape(n, QK_K // (2 * chunk), 2, chunk).to(torch.int32)
    return (v[:, :, 0, :] | (v[:, :, 1, :] << 4)).reshape(n, 128).to(torch.uint8)


def _pack_bits(bits: torch.Tensor, nbytes: int) -> torch.Tensor:
    """(n, 256) values in [0,1] -> (n, nbytes) with byte e%nbytes, bit e//nbytes."""
    n = bits.shape[0]
    v = bits.reshape(n, QK_K // nbytes, nbytes).to(torch.int32)
    shifts = torch.arange(QK_K // nbytes, device=bits.device).view(1, -1, 1)
    return (v << shifts).sum(dim=1).to(torch.uint8)


def _pack_2bit_chunk32(q: torch.Tensor) -> torch.Tensor:
    """(n, 256) values in [0,3] -> (n, 64) bytes for Q6_K qh:
    byte (e//128)*32 + e%32, shift 2*((e%128)//32)."""
    return _pack_2bit(q)


def _fp16_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16).view(torch.uint8)


def _pack_scales_k(sc: torch.Tensor, mn: torch.Tensor) -> torch.Tensor:
    """(n, 8) 6-bit scales + mins -> (n, 12) bytes (Q4_K/Q5_K layout)."""
    sc = sc.to(torch.int32)
    mn = mn.to(torch.int32)
    d_b = (sc[:, :4] & 0x3F) | (((sc[:, 4:] >> 4) & 0x03) << 6)
    m_b = (mn[:, :4] & 0x3F) | (((mn[:, 4:] >> 4) & 0x03) << 6)
    md_b = (sc[:, 4:] & 0x0F) | ((mn[:, 4:] & 0x0F) << 4)
    return torch.cat([d_b, m_b, md_b], dim=1).to(torch.uint8)


def _pack_scales_q3(sc: torch.Tensor) -> torch.Tensor:
    """(n, 16) 6-bit signed scales (stored +32) -> (n, 12) bytes."""
    sc6 = (sc.to(torch.int32) + 32)
    lo, hi = sc6 & 0x0F, sc6 >> 4
    out = torch.zeros(sc.shape[0], 12, dtype=torch.int32, device=sc.device)
    out[:, :8] = lo[:, :8] | (lo[:, 8:] << 4)
    for t in range(4):
        out[:, 8:12] |= hi[:, 4 * t: 4 * t + 4] << (2 * t)
    return out.to(torch.uint8)


def _pack_blocks(w: torch.Tensor, fmt: str) -> torch.Tensor:
    fields_fn, _, block = _FIELDS[fmt]
    flat = w.to(torch.float32).reshape(-1, block)
    f = fields_fn(flat)
    n = flat.shape[0]
    if fmt == "Q2_K":
        scales_b = (f["sc"] | (f["m"] << 4)).to(torch.uint8)
        return torch.cat([scales_b, _pack_2bit(f["q"]),
                          _fp16_bytes(f["d"]), _fp16_bytes(f["dmin"])], dim=1)
    if fmt == "Q3_K":
        q = f["q"].to(torch.int32)
        ql = (q & 3).to(torch.uint8)
        hbit = (q >= 0).to(torch.uint8)  # stored 1 = no -4 offset
        return torch.cat([_pack_bits(hbit, 32), _pack_2bit(ql),
                          _pack_scales_q3(f["sc"]), _fp16_bytes(f["d"])], dim=1)
    if fmt == "Q4_K":
        return torch.cat([_fp16_bytes(f["d"]), _fp16_bytes(f["dmin"]),
                          _pack_scales_k(f["sc"], f["m"]),
                          _pack_nibbles(f["q"], 32)], dim=1)
    if fmt == "Q5_K":
        q = f["q"].to(torch.int32)
        return torch.cat([_fp16_bytes(f["d"]), _fp16_bytes(f["dmin"]),
                          _pack_scales_k(f["sc"], f["m"]),
                          _pack_bits((q >> 4).to(torch.uint8), 32),
                          _pack_nibbles((q & 0x0F).to(torch.uint8), 32)], dim=1)
    if fmt == "Q6_K":
        q = (f["q"].to(torch.int32) + 32)
        return torch.cat([_pack_nibbles((q & 0x0F).to(torch.uint8), 64),
                          _pack_2bit_chunk32((q >> 4).to(torch.uint8)),
                          f["sc"].view(torch.uint8), _fp16_bytes(f["d"])], dim=1)
    if fmt == "Q8_0":
        return torch.cat([_fp16_bytes(f["d"]), f["q"].view(torch.uint8)], dim=1)
    raise ValueError(f"unsupported GGUF pack format: {fmt}")


def gguf_pack(w: torch.Tensor, fmt: str) -> np.ndarray:
    """Quantize + bit-pack a 2-D (or stacked 3-D) weight into GGUF bytes.

    Returns uint8 of shape ``(*w.shape[:-1], row_bytes)`` — the shape the
    GGUF writer needs so tensor metadata records the logical dims.
    """
    block, type_size = GGUF_BLOCK_BYTES[fmt]
    in_f = int(w.shape[-1])
    if in_f % block:
        raise ValueError(
            f"{fmt} requires the input dim to be a multiple of {block}; "
            f"got shape {tuple(w.shape)}"
        )
    packed = _pack_blocks(w, fmt)
    out_shape = tuple(w.shape[:-1]) + (in_f // block * type_size,)
    return packed.reshape(out_shape).cpu().numpy()
