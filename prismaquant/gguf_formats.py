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

Scale selection ports llama.cpp's reference quantizers (ggml-quants.c):
``make_qkx2_quants`` for the asymmetric types (weighted grid search over
candidate scales + weighted least-squares (scale, min) refit per
sub-block) and ``make_qx_quants`` for the symmetric types (sign-aware
search that maps the extremum onto the extra negative level, weighted-LS
scale refit), vectorized over all sub-blocks in torch. Weight functions
match the no-imatrix reference: |x| (Q2_K), av_x+|x| (Q4_K/Q5_K), x^2
(Q3_K/Q6_K); Q3_K additionally LS-quantizes its 16 sub-scales onto the
6-bit grid with per-sub-block weight mass, as the reference does.

Activation emulation models the ggml MMQ/MMVQ compute path, which
quantizes activations to Q8_1 (per-32 symmetric int8).
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

def _search_asym(sb: torch.Tensor, qmax: int, weights: torch.Tensor,
                 rmin: float, rdelta: float, nstep: int,
                 use_mad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized make_qkx2_quants: weighted (scale, min) search per sub-block.

    sb, weights: (..., sub). Returns float (sub_scale, sub_min), both >= 0.
    """
    mn = sb.amin(dim=-1).clamp_max(0.0)
    mx = sb.amax(dim=-1)
    span = mx - mn
    degenerate = span <= 0

    sum_w = weights.sum(dim=-1)
    sum_x = (weights * sb).sum(dim=-1)

    def _err(scale, minv, L):
        diff = scale.unsqueeze(-1) * L + minv.unsqueeze(-1) - sb
        diff = diff.abs() if use_mad else diff * diff
        return (weights * diff).sum(dim=-1)

    iscale0 = qmax * _safe_inv(span)
    L = torch.round(iscale0.unsqueeze(-1) * (sb - mn.unsqueeze(-1)))
    L = L.clamp(0, qmax)
    best_scale = _safe_inv(iscale0)
    best_min = mn
    best_err = _err(best_scale, best_min, L)

    for step in range(nstep + 1):
        iscale = (rmin + rdelta * step + qmax) * _safe_inv(span)
        Laux = torch.round(iscale.unsqueeze(-1) * (sb - mn.unsqueeze(-1)))
        Laux = Laux.clamp(0, qmax)
        wl = weights * Laux
        sum_l = wl.sum(dim=-1)
        sum_l2 = (wl * Laux).sum(dim=-1)
        sum_xl = (wl * sb).sum(dim=-1)
        D = sum_w * sum_l2 - sum_l * sum_l
        this_scale = (sum_w * sum_xl - sum_x * sum_l) * _safe_inv(D)
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) * _safe_inv(D)
        # A positive min is illegal (mins are stored >= 0 and subtracted):
        # clamp to 0 and refit scale alone, as the reference does.
        pos = this_min > 0
        this_scale = torch.where(pos, sum_xl * _safe_inv(sum_l2), this_scale)
        this_min = torch.where(pos, torch.zeros_like(this_min), this_min)
        cur_err = _err(this_scale, this_min, Laux)
        better = (D > 0) & (cur_err < best_err)
        best_err = torch.where(better, cur_err, best_err)
        best_scale = torch.where(better, this_scale, best_scale)
        best_min = torch.where(better, this_min, best_min)

    zero = torch.zeros_like(best_scale)
    best_scale = torch.where(degenerate, zero, best_scale)
    best_min = torch.where(degenerate, -mn, -best_min)
    return best_scale, best_min.clamp_min(0.0)


def _search_sym(sb: torch.Tensor, nmax: int,
                weights: torch.Tensor) -> torch.Tensor:
    """Vectorized make_qx_quants: sign-aware weighted-LS scale per sub-block.

    sb, weights: (..., sub). Quant grid is [-nmax, nmax-1]; the initial
    candidate maps the (signed) extremum onto -nmax so the asymmetric
    integer range is fully used. Returns the SIGNED float scale.
    """
    amax, idx = sb.abs().max(dim=-1)
    signed_max = torch.gather(sb, -1, idx.unsqueeze(-1)).squeeze(-1)
    degenerate = amax < 1e-30

    def _fit(iscale):
        L = torch.round(iscale.unsqueeze(-1) * sb).clamp(-nmax, nmax - 1)
        wl = weights * L
        sumlx = (wl * sb).sum(dim=-1)
        suml2 = (wl * L).sum(dim=-1)
        return sumlx, suml2

    iscale0 = -nmax * _safe_inv(signed_max)
    sumlx, suml2 = _fit(iscale0)
    scale = sumlx * _safe_inv(suml2)
    best = scale * sumlx

    for step in range(-9, 10):
        if step == 0:
            continue
        iscale = -(nmax + 0.1 * step) * _safe_inv(signed_max)
        slx, sl2 = _fit(iscale)
        better = (sl2 > 0) & (slx * slx > best * sl2)
        new_scale = slx * _safe_inv(sl2)
        scale = torch.where(better, new_scale, scale)
        best = torch.where(better, new_scale * slx, best)

    return torch.where(degenerate, torch.zeros_like(scale), scale)


def _imatrix_weights(blocks: torch.Tensor, qw: torch.Tensor, sub: int,
                     sigma2_factor: float) -> torch.Tensor:
    """llama.cpp imatrix composition: qw * sqrt(sigma2 + x^2), with sigma2
    the (factor-scaled) mean square over the whole superblock."""
    n = blocks.shape[0]
    sigma2 = sigma2_factor * blocks.pow(2).mean(dim=-1, keepdim=True)
    sb = blocks.reshape(n, QK_K // sub, sub)
    return qw.reshape(n, QK_K // sub, sub) * (
        sigma2.unsqueeze(-1) + sb * sb
    ).sqrt()


def _fields_asym(blocks: torch.Tensor, sub: int, qmax: int, scale_max: int,
                 rmin: float, rdelta: float, nstep: int,
                 use_mad: bool, weight_kind: str,
                 qw: torch.Tensor | None = None,
                 sigma2_factor: float = 2.0) -> dict[str, torch.Tensor]:
    """Shared asymmetric two-tier quantizer (Q2_K sub=16, Q4_K/Q5_K sub=32).

    llama.cpp-reference weighted search per sub-block, then sub-scale and
    sub-min quantized to [0, scale_max] under fp16 super-scales d, dmin,
    and q recomputed against the *quantized* (dl, ml).
    """
    n = blocks.shape[0]
    sb = blocks.reshape(n, QK_K // sub, sub)
    if qw is not None:
        weights = _imatrix_weights(blocks, qw, sub, sigma2_factor)
    elif weight_kind == "abs":
        weights = sb.abs()
    else:  # "avx_abs": av_x + |x|, the Q4_K/Q5_K reference weighting
        av_x = sb.pow(2).mean(dim=-1, keepdim=True).sqrt()
        weights = av_x + sb.abs()

    sub_scale, sub_min = _search_asym(
        sb, qmax, weights, rmin=rmin, rdelta=rdelta, nstep=nstep,
        use_mad=use_mad,
    )

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


def _fields_q2_k(blocks: torch.Tensor,
                 qw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    return _fields_asym(blocks, sub=16, qmax=3, scale_max=15,
                        rmin=-0.5, rdelta=0.1, nstep=15, use_mad=True,
                        weight_kind="abs", qw=qw, sigma2_factor=1.0)


def _fields_q4_k(blocks: torch.Tensor,
                 qw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    return _fields_asym(blocks, sub=32, qmax=15, scale_max=63,
                        rmin=-1.0, rdelta=0.1, nstep=20, use_mad=False,
                        weight_kind="avx_abs", qw=qw, sigma2_factor=2.0)


def _fields_q5_k(blocks: torch.Tensor,
                 qw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    return _fields_asym(blocks, sub=32, qmax=31, scale_max=63,
                        rmin=-0.5, rdelta=0.1, nstep=15, use_mad=False,
                        weight_kind="avx_abs", qw=qw, sigma2_factor=2.0)


def _fields_q3_k(blocks: torch.Tensor,
                 qw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Symmetric 3-bit: q in [-4, 3], 6-bit signed sub-scales, fp16 d.

    x^2-weighted LS sub-scales; the 16 float sub-scales are themselves
    LS-quantized onto [-32, 31] with per-sub-block weight mass (the
    reference's second make_qx_quants pass), then q recomputed against
    the quantized two-tier scale.
    """
    n = blocks.shape[0]
    sb = blocks.reshape(n, QK_K // 16, 16)
    if qw is not None:
        weights = _imatrix_weights(blocks, qw, 16, sigma2_factor=2.0)
    else:
        weights = sb * sb
    sub_scale = _search_sym(sb, nmax=4, weights=weights)

    sw = weights.sum(dim=-1)
    d = _fp16r(
        _search_sym(sub_scale, nmax=32, weights=sw).unsqueeze(-1)
    )
    sc = torch.round(sub_scale * _safe_inv(d)).clamp(-32, 31).to(torch.int8)

    dl = (d * sc.float()).unsqueeze(-1)
    q = torch.round(sb * _safe_inv(dl)).clamp(-4, 3).to(torch.int8)
    return {"d": d, "sc": sc, "q": q.reshape(n, QK_K)}


def _recon_q3_k(f: dict[str, torch.Tensor]) -> torch.Tensor:
    n = f["q"].shape[0]
    dl = (f["d"] * f["sc"].float()).unsqueeze(-1)
    return (dl * f["q"].reshape(n, QK_K // 16, 16).float()).reshape(n, QK_K)


def _fields_q6_k(blocks: torch.Tensor,
                 qw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Symmetric 6-bit: q in [-32, 31], int8 sub-scales, fp16 d.

    x^2-weighted LS sub-scales (raw qw when an imatrix is given, as the
    reference does); the super-scale maps the largest (signed) sub-scale
    onto -128, using the extra negative int8 level as the reference does.
    """
    n = blocks.shape[0]
    sb = blocks.reshape(n, QK_K // 16, 16)
    if qw is not None:
        weights = qw.reshape(n, QK_K // 16, 16)
    else:
        weights = sb * sb
    sub_scale = _search_sym(sb, nmax=32, weights=weights)

    amax, idx = sub_scale.abs().max(dim=1, keepdim=True)
    signed_max = torch.gather(sub_scale, 1, idx)
    d = _fp16r(-signed_max / 128.0)
    sc = torch.round(sub_scale * _safe_inv(d)).clamp(-128, 127).to(torch.int8)

    dl = (d * sc.float()).unsqueeze(-1)
    q = torch.round(sb * _safe_inv(dl)).clamp(-32, 31).to(torch.int8)
    return {"d": d, "sc": sc, "q": q.reshape(n, QK_K)}


def _recon_q6_k(f: dict[str, torch.Tensor]) -> torch.Tensor:
    n = f["q"].shape[0]
    dl = (f["d"] * f["sc"].float()).unsqueeze(-1)
    return (dl * f["q"].reshape(n, QK_K // 16, 16).float()).reshape(n, QK_K)


def _fields_q8_0(blocks: torch.Tensor,
                 qw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Per-32 symmetric int8 (blocks input is (N, 32)); half-away rounding
    to stay bit-exact with the ggml/gguf-py reference quantizer. The
    reference Q8_0 quantizer ignores imatrix weights."""
    del qw
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


def _qw_blocks(qw: torch.Tensor, w_shape: tuple[int, ...], pad: int,
               block: int) -> torch.Tensor:
    """Broadcast column weights to the flat superblock layout.

    ``qw`` is either (in_features,) — one importance vector for the whole
    tensor — or any shape broadcastable to ``w_shape`` (e.g. (N, 1, in)
    for a stacked batch with per-item vectors).
    """
    qw = qw.to(torch.float32)
    full = torch.broadcast_to(qw, w_shape).reshape(-1, w_shape[-1])
    if pad:
        full = torch.nn.functional.pad(full, (0, pad))
    return full.reshape(-1, block)


def gguf_quantize_dequantize(
    w: torch.Tensor, fmt: str, col_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Emulation QDQ. ``col_weights`` is an optional per-input-column
    importance vector (llama.cpp imatrix semantics: mean squared
    activation per column); it biases scale selection, never the grid."""
    fields_fn, recon_fn, block = _FIELDS[fmt]
    orig_shape = w.shape
    in_f = int(orig_shape[-1])
    flat = w.reshape(-1, in_f).to(torch.float32)
    pad = (-in_f) % block
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block)
    qw = None
    if col_weights is not None:
        qw = _qw_blocks(
            col_weights.to(w.device), tuple(orig_shape), pad, block,
        )
    out = recon_fn(fields_fn(blocks, qw=qw)).reshape(flat.shape)
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


def _pack_blocks(w: torch.Tensor, fmt: str,
                 col_weights: torch.Tensor | None = None) -> torch.Tensor:
    fields_fn, _, block = _FIELDS[fmt]
    in_f = int(w.shape[-1])
    flat = w.to(torch.float32).reshape(-1, block)
    qw = None
    if col_weights is not None:
        qw = _qw_blocks(col_weights.to(w.device), tuple(w.shape), 0, block)
    f = fields_fn(flat, qw=qw)
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


def gguf_pack(w: torch.Tensor, fmt: str,
              col_weights: torch.Tensor | None = None) -> np.ndarray:
    """Quantize + bit-pack a 2-D (or stacked 3-D) weight into GGUF bytes.

    Returns uint8 of shape ``(*w.shape[:-1], row_bytes)`` — the shape the
    GGUF writer needs so tensor metadata records the logical dims.
    ``col_weights`` (in_features,) biases scale selection exactly as the
    emulation's ``col_weights`` does — the two stay bit-identical.
    """
    block, type_size = GGUF_BLOCK_BYTES[fmt]
    in_f = int(w.shape[-1])
    if in_f % block:
        raise ValueError(
            f"{fmt} requires the input dim to be a multiple of {block}; "
            f"got shape {tuple(w.shape)}"
        )
    packed = _pack_blocks(w, fmt, col_weights=col_weights)
    out_shape = tuple(w.shape[:-1]) + (in_f // block * type_size,)
    return packed.reshape(out_shape).cpu().numpy()
