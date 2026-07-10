"""GGUF IQ-quant weight formats: torch field quantizers, emulation recon,
and byte packers — the sub-Q2_K / non-linear-4-bit ggml lane.

Seven ggml types live here:

* IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S — *grid* (codebook-of-vectors)
  formats. Weights are grouped (8 for IQ2, 4 for IQ3); every group's magnitude
  pattern is one entry of a fixed grid, its per-element signs are a packed
  sign code, and a two-tier fp16 super-scale + 4-bit sub-scale sets the group
  amplitude. The grid magnitudes, the 7-bit sign codebook (``ksigns``) and the
  non-linear IQ4 value table are lifted from gguf-py's decoded tables
  (``prismaquant/data/iq_grids.pt``; regenerate with ``scripts/gen_iq_grids.py``)
  so the encoder reconstructs the exact numbers gguf-py / llama.cpp decode.
* IQ4_XS / IQ4_NL — non-linear 4-bit codebook (``kvalues_iq4nl``) with a
  two-tier (IQ4_XS) or single (IQ4_NL, block-32) scale.

As with the k-quants (``gguf_formats``) there is one math path per format: the
field quantizer feeds *both* the emulation ``reconstruct`` (what cost
measurement scores) and the byte packer (what export ships), and gguf-py's
``dequantize`` of those bytes is pinned bit-identical to the emulation in
``tests/test_gguf_iq_formats.py``.

Scale/grid selection does an *exhaustive* GPU codebook argmin (a
``groups × grid`` weighted-distance minimization) rather than llama.cpp's
precomputed-neighbour heuristic — plain tensor math that dominates the
reference search on quality. imatrix per-column weights bias the objective
exactly as llama.cpp's ``quant_weights`` do (``qw · sqrt(sigma2 + x²)``).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

QK_K = 256
_DATA = Path(__file__).resolve().parent / "data" / "iq_grids.pt"

# name -> (block_size, type_size_bytes); mirrors gguf.GGML_QUANT_SIZES.
IQ_BLOCK_BYTES: dict[str, tuple[int, int]] = {
    "IQ2_XXS": (QK_K, 66),
    "IQ2_XS": (QK_K, 74),
    "IQ2_S": (QK_K, 82),
    "IQ3_XXS": (QK_K, 98),
    "IQ3_S": (QK_K, 110),
    "IQ4_XS": (QK_K, 136),
    "IQ4_NL": (32, 18),
}

# Per grid format: grid element group size (ge), grid entry count (ngrid),
# scale-groups per superblock (n_sg), grid entries per scale-group (eps),
# sign encoding ("ksigns" 7-bit+parity or "direct" 8-bit), the continuous
# per-group scale -> reference-unit multiplier K, and the (base, coef) of the
# super-scale reconstruction db = d * coef * (base + l).
_META: dict[str, dict] = {
    "IQ2_XXS": dict(key="grid_iq2_xxs", ge=8, ngrid=256, n_sg=8, eps=4,
                    sign="ksigns", K=8.0, base=0.5, coef=0.25),
    "IQ2_XS": dict(key="grid_iq2_xs", ge=8, ngrid=512, n_sg=16, eps=2,
                   sign="ksigns", K=8.0, base=0.5, coef=0.25),
    "IQ2_S": dict(key="grid_iq2_s", ge=8, ngrid=1024, n_sg=16, eps=2,
                  sign="direct", K=8.0, base=0.5, coef=0.25),
    "IQ3_XXS": dict(key="grid_iq3_xxs", ge=4, ngrid=256, n_sg=8, eps=8,
                    sign="ksigns", K=4.0, base=0.5, coef=0.5),
    "IQ3_S": dict(key="grid_iq3_s", ge=4, ngrid=512, n_sg=8, eps=8,
                  sign="direct", K=1.0, base=0.5, coef=2.0),
}


@lru_cache(maxsize=None)
def _tables(device: str) -> dict[str, torch.Tensor]:
    raw = torch.load(_DATA, map_location="cpu", weights_only=True)
    dev = torch.device(device)
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("grid_"):
            out[k] = v.to(dev, torch.float32)
    out["ksigns"] = raw["ksigns"].to(dev).long()
    out["kvalues_iq4nl"] = raw["kvalues_iq4nl"].to(dev, torch.float32)
    return out


def _fp16r(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16).to(torch.float32)


def _safe_inv(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t != 0, 1.0 / torch.where(t == 0, torch.ones_like(t), t),
                       torch.zeros_like(t))


def _round_half_away(t: torch.Tensor) -> torch.Tensor:
    return torch.sign(t) * torch.floor(t.abs() + 0.5)


# ---------------------------------------------------------------------------
# Byte helpers: little-endian widen (matches gguf-py's ndarray .view).
# ---------------------------------------------------------------------------

def _u16_to_bytes(t: torch.Tensor) -> torch.Tensor:
    t = t.to(torch.int64)
    b = torch.stack([t & 0xFF, (t >> 8) & 0xFF], dim=-1)
    return b.reshape(*t.shape[:-1], -1).to(torch.uint8)


def _u32_to_bytes(t: torch.Tensor) -> torch.Tensor:
    t = t.to(torch.int64)
    b = torch.stack([(t >> (8 * i)) & 0xFF for i in range(4)], dim=-1)
    return b.reshape(*t.shape[:-1], -1).to(torch.uint8)


def _fp16_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16).view(torch.uint8)


def _bits_to_pm(codes: torch.Tensor) -> torch.Tensor:
    """8-bit sign masks (..., n8) -> +-1 floats (..., n8*8), bit set => -1."""
    shifts = torch.arange(8, device=codes.device)
    bits = (codes.unsqueeze(-1) >> shifts) & 1
    return torch.where(bits == 0, 1.0, -1.0).reshape(
        *codes.shape[:-1], codes.shape[-1] * 8)


# ---------------------------------------------------------------------------
# imatrix weighting (llama.cpp composition qw * sqrt(sigma2 + x^2)).
# ---------------------------------------------------------------------------

def _weights(blocks: torch.Tensor, qw: torch.Tensor | None,
             sigma2_factor: float, group: int) -> torch.Tensor:
    n, block = blocks.shape
    sigma2 = sigma2_factor * blocks.pow(2).mean(dim=-1, keepdim=True)
    base = (sigma2 + blocks * blocks).sqrt()
    if qw is None:
        return base
    w = qw * base
    # Dead calibration groups (all-zero imatrix mass) would collapse the
    # scale search and erase real weights; fall back to the unweighted base.
    g = block // group
    mass = w.reshape(n, g, group).sum(dim=-1, keepdim=True)
    return torch.where(mass == 0, base.reshape(n, g, group),
                       w.reshape(n, g, group)).reshape(n, block)


# ---------------------------------------------------------------------------
# Grid (IQ2/IQ3) field quantizer.
# ---------------------------------------------------------------------------

def _sign_fields(x: torch.Tensor, w: torch.Tensor, mode: str,
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per group-of-8 sign handling. Returns (y, codes):

    y = applied_sign * x (the search target; grid magnitudes are >= 0), and
    codes = stored sign integer per group (7-bit for ksigns, 8-bit direct).
    """
    n = x.shape[0]
    g8 = x.reshape(n, -1, 8)
    w8 = w.reshape(n, -1, 8)
    neg = g8 < 0
    if mode == "direct":
        codes = (neg.long() << torch.arange(8, device=x.device)).sum(-1)
        return g8.abs().reshape(n, -1), codes
    # ksigns: applied sign mask must have even parity; if odd, flip the
    # least-important element (min w*x^2), exactly as llama.cpp does.
    mask = neg.long()
    odd = (mask.sum(-1) % 2) == 1
    imp = w8 * g8 * g8
    flip = imp.argmin(dim=-1)
    flip_oh = torch.nn.functional.one_hot(flip, 8).to(mask.dtype)
    mask = torch.where(odd.unsqueeze(-1).expand_as(mask),
                       mask ^ flip_oh, mask)
    applied = torch.where(mask.bool(), -1.0, 1.0)
    y = (applied * g8).reshape(n, -1)
    codes7 = (mask << torch.arange(8, device=x.device)).sum(-1) & 0x7F
    return y, codes7


def _grid_fields(blocks: torch.Tensor, fmt: str,
                 qw: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Exhaustive weighted grid + two-tier scale quantizer for one IQ2/IQ3
    format. Minimizes sum_i w_i (db*grid[g,i] - y_i)^2 per grid entry, with a
    per-scale-group amplitude db shared across its eps entries, swept and
    refined by weighted least squares, then quantized into the fp16 super-scale
    d + 4-bit sub-scale l and the grid entries re-selected at the quantized db.

    A, B, C (the per-entry weighted moments over the grid) are computed once and
    reused across every scale candidate and the final re-selection.
    """
    m = _META[fmt]
    grid = _tables(str(blocks.device))[m["key"]]
    n = blocks.shape[0]
    n_sg, eps, ge = m["n_sg"], m["eps"], m["ge"]
    w = _weights(blocks, qw, sigma2_factor=1.0, group=8)
    y, codes = _sign_fields(blocks, w, m["sign"])

    nm = n * n_sg * eps
    ye = y.reshape(nm, ge)
    we = w.reshape(nm, ge)
    A = we @ (grid * grid).transpose(0, 1).contiguous()       # (nm, ngrid)
    B = (we * ye) @ grid.transpose(0, 1).contiguous()         # (nm, ngrid)
    C = (we * ye * ye).sum(-1)                                 # (nm,)

    def pick(db_sg: torch.Tensor):
        """db_sg: (n, n_sg). Returns per-entry argmin idx (nm,) and per-scale-
        group summed error (n, n_sg)."""
        db_e = db_sg.reshape(n, n_sg, 1).expand(n, n_sg, eps).reshape(nm, 1)
        d2 = db_e * db_e * A - 2.0 * db_e * B + C.unsqueeze(-1)
        idx = d2.argmin(dim=-1)
        err = d2.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        return idx, err.reshape(n, n_sg, eps).sum(-1)

    def refit(idx: torch.Tensor):
        gsel = grid[idx]                                       # (nm, ge)
        num = (we * ye * gsel).reshape(n, n_sg, eps * ge).sum(-1)
        den = (we * gsel * gsel).reshape(n, n_sg, eps * ge).sum(-1)
        return num * _safe_inv(den)                           # (n, n_sg)

    db0 = y.reshape(n, n_sg, eps * ge).abs().amax(dim=-1) / grid.max()
    best_db = db0.clone()
    best_err = torch.full((n, n_sg), float("inf"), device=y.device)

    def consider(db_sg: torch.Tensor):
        nonlocal best_db, best_err
        _, errsg = pick(db_sg)
        better = errsg < best_err
        best_err = torch.where(better, errsg, best_err)
        best_db = torch.where(better, db_sg, best_db)

    for f in torch.linspace(0.5, 1.8, 27, device=y.device).tolist():
        consider(db0 * f)
    for _ in range(3):
        consider(refit(pick(best_db)[0]))

    scaleunit = best_db * m["K"]
    d = _fp16r(scaleunit.amax(dim=1, keepdim=True) / 31.0)
    l = _round_half_away(0.5 * (scaleunit * _safe_inv(d) - 1.0))
    l = l.clamp(0, 15).to(torch.int64)
    db_q = d * m["coef"] * (m["base"] + l.float())            # (n, n_sg)
    idx, _ = pick(db_q)                                        # final selection
    return {"d": d, "l": l, "idx": idx.reshape(n, n_sg * eps), "sign": codes}


def _grid_recon(f: dict[str, torch.Tensor], fmt: str) -> torch.Tensor:
    m = _META[fmt]
    grid = _tables(str(f["d"].device))[m["key"]]
    ge, n_sg = m["ge"], m["n_sg"]
    n = f["d"].shape[0]
    db = f["d"] * m["coef"] * (m["base"] + f["l"].float())     # (n, n_sg)
    db_elem = db.repeat_interleave(QK_K // n_sg, dim=1)         # (n, 256)
    gvals = grid[f["idx"]].reshape(n, QK_K)                     # (n, 256)
    if m["sign"] == "direct":
        sign = _bits_to_pm(f["sign"])                       # sign per g8
    else:
        s8 = _tables(str(f["d"].device))["ksigns"][f["sign"]]
        sign = _bits_to_pm(s8)
    return db_elem * gvals * sign


# ---------------------------------------------------------------------------
# Grid byte packers.
# ---------------------------------------------------------------------------

def _assemble_grid(f: dict[str, torch.Tensor], fmt: str) -> torch.Tensor:
    n = f["d"].shape[0]
    d_b = _fp16_bytes(f["d"])
    idx = f["idx"].to(torch.int64)
    sign = f["sign"].to(torch.int64)
    l = f["l"].to(torch.int64)
    if fmt == "IQ2_XXS":
        gi = idx.reshape(n, 8, 4)
        q_lo = sum((gi[:, :, k] << (8 * k)) for k in range(4))       # (n,8)
        s7 = sign.reshape(n, 8, 4)
        q_hi = sum((s7[:, :, k] << (7 * k)) for k in range(4)) | (l << 28)
        q = torch.stack([q_lo, q_hi], dim=-1).reshape(n, 16)
        return torch.cat([d_b, _u32_to_bytes(q)], dim=1)
    if fmt == "IQ2_XS":
        qs = idx | (sign << 9)                                        # (n,32)
        ll = l.reshape(n, 8, 2)
        sc = (ll[:, :, 0] | (ll[:, :, 1] << 4)).to(torch.uint8)
        return torch.cat([d_b, _u16_to_bytes(qs), sc], dim=1)
    if fmt == "IQ2_S":
        qs = (idx & 0xFF).to(torch.uint8)                             # (n,32)
        signs = sign.to(torch.uint8)                                  # (n,32)
        gi = idx.reshape(n, 8, 4)
        qh = sum((((gi[:, :, b] >> 8) & 3) << (2 * b)) for b in range(4))
        ll = l.reshape(n, 8, 2)
        sc = (ll[:, :, 0] | (ll[:, :, 1] << 4)).to(torch.uint8)
        return torch.cat([d_b, qs, signs, qh.to(torch.uint8), sc], dim=1)
    if fmt == "IQ3_XXS":
        qs = idx.to(torch.uint8)                                      # (n,64)
        s7 = sign.reshape(n, 8, 4)
        sas = sum((s7[:, :, k] << (7 * k)) for k in range(4)) | (l << 28)
        return torch.cat([d_b, qs, _u32_to_bytes(sas)], dim=1)
    if fmt == "IQ3_S":
        qs = (idx & 0xFF).to(torch.uint8)                             # (n,64)
        gi = idx.reshape(n, 8, 8)
        qh = sum((((gi[:, :, b] >> 8) & 1) << b) for b in range(8))
        signs = sign.to(torch.uint8)                                  # (n,32)
        ll = l.reshape(n, 4, 2)
        sc = (ll[:, :, 0] | (ll[:, :, 1] << 4)).to(torch.uint8)
        return torch.cat([d_b, qs, qh.to(torch.uint8), signs, sc], dim=1)
    raise ValueError(fmt)


# ---------------------------------------------------------------------------
# IQ4 (non-linear codebook) field quantizer.
# ---------------------------------------------------------------------------

def _best_kv(vals: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    return (vals.unsqueeze(-1) - kv).abs().argmin(dim=-1)


def _iq4_block_scale(xb: torch.Tensor, wb: torch.Tensor, kv: torch.Tensor,
                     ntry: int) -> torch.Tensor:
    """Continuous per-32-block scale by the llama.cpp ntry sweep + LS refit.
    xb, wb: (..., 32). Returns signed scale (...,)."""
    amax, ai = xb.abs().max(dim=-1)
    xmax = xb.gather(-1, ai.unsqueeze(-1)).squeeze(-1)        # signed extremum
    k0 = kv[0]

    def fit(scale):
        idl = _safe_inv(scale)
        q = kv[_best_kv(idl.unsqueeze(-1) * xb, kv)]
        sumqx = (wb * q * xb).sum(-1)
        sumq2 = (wb * q * q).sum(-1)
        return sumqx, sumq2

    scale = -xmax / k0
    sumqx, sumq2 = fit(scale)
    best_scale = sumqx * _safe_inv(sumq2)
    best = best_scale * sumqx
    for itry in range(-ntry, ntry + 1):
        id_try = (itry + k0) * _safe_inv(xmax)
        cand = _safe_inv(id_try)
        sumqx, sumq2 = fit(cand)
        cs = sumqx * _safe_inv(sumq2)
        better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        best_scale = torch.where(better, cs, best_scale)
        best = torch.where(better, cs * sumqx, best)
    dead = amax < 1e-30
    return torch.where(dead, torch.zeros_like(best_scale), best_scale)


def _fields_iq4_xs(blocks: torch.Tensor,
                   qw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    n = blocks.shape[0]
    kv = _tables(str(blocks.device))["kvalues_iq4nl"]
    sigma2 = 2.0 * blocks.pow(2).mean(dim=-1, keepdim=True)
    if qw is None:
        w = blocks * blocks
    else:
        base = (sigma2 + blocks * blocks).sqrt()
        w = qw * base
        mass = w.reshape(n, 8, 32).sum(dim=-1, keepdim=True)
        w = torch.where(mass == 0, base.reshape(n, 8, 32),
                        w.reshape(n, 8, 32)).reshape(n, QK_K)
    xb = blocks.reshape(n, 8, 32)
    wb = w.reshape(n, 8, 32)
    scales = _iq4_block_scale(xb, wb, kv, ntry=7)              # (n, 8)

    amax, ai = scales.abs().max(dim=-1, keepdim=True)
    max_scale = scales.gather(-1, ai)                          # signed
    d = _fp16r(-max_scale / 32.0)                              # (n, 1)
    l = _round_half_away(scales * _safe_inv(d)).clamp(-32, 31)
    dl = (d * l).unsqueeze(-1)                                 # (n, 8, 1)
    idx = _best_kv(xb * _safe_inv(dl), kv)                     # (n, 8, 32)
    return {"d": d, "scale6": (l + 32).to(torch.int64),
            "idx": idx.reshape(n, QK_K)}


def _recon_iq4_xs(f: dict[str, torch.Tensor]) -> torch.Tensor:
    kv = _tables(str(f["d"].device))["kvalues_iq4nl"]
    n = f["d"].shape[0]
    dl = f["d"] * (f["scale6"].float() - 32.0)                 # (n, 8)
    dl_elem = dl.repeat_interleave(32, dim=1)                  # (n, 256)
    return dl_elem * kv[f["idx"]]


def _assemble_iq4_xs(f: dict[str, torch.Tensor]) -> torch.Tensor:
    n = f["d"].shape[0]
    d_b = _fp16_bytes(f["d"])
    s6 = f["scale6"].to(torch.int64)                          # (n, 8)
    sh = sum((((s6[:, i] >> 4) & 3) << (2 * i)) for i in range(8))
    scales_h = _u16_to_bytes(sh.reshape(n, 1))
    sl = s6.reshape(n, 4, 2)
    scales_l = ((sl[:, :, 0] & 0xF) | ((sl[:, :, 1] & 0xF) << 4)).to(torch.uint8)
    idx = f["idx"].reshape(n, 8, 32).to(torch.int64)
    qs = (idx[:, :, :16] | (idx[:, :, 16:] << 4)).reshape(n, 128).to(torch.uint8)
    return torch.cat([d_b, scales_h, scales_l, qs], dim=1)


def _fields_iq4_nl(blocks: torch.Tensor,
                   qw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    n = blocks.shape[0]
    kv = _tables(str(blocks.device))["kvalues_iq4nl"]
    sigma2 = 2.0 * blocks.pow(2).mean(dim=-1, keepdim=True)
    if qw is None:
        w = blocks * blocks
    else:
        base = (sigma2 + blocks * blocks).sqrt()
        w = qw * base
        mass = w.sum(dim=-1, keepdim=True)
        w = torch.where(mass == 0, base, w)
    scale = _iq4_block_scale(blocks.reshape(n, 1, 32), w.reshape(n, 1, 32),
                             kv, ntry=7).reshape(n, 1)
    d = _fp16r(scale)
    idx = _best_kv(blocks * _safe_inv(d), kv)
    return {"d": d, "idx": idx.reshape(n, 32)}


def _recon_iq4_nl(f: dict[str, torch.Tensor]) -> torch.Tensor:
    kv = _tables(str(f["d"].device))["kvalues_iq4nl"]
    return f["d"] * kv[f["idx"]]


def _assemble_iq4_nl(f: dict[str, torch.Tensor]) -> torch.Tensor:
    n = f["d"].shape[0]
    d_b = _fp16_bytes(f["d"])
    idx = f["idx"].to(torch.int64)
    qs = (idx[:, :16] | (idx[:, 16:] << 4)).to(torch.uint8)
    return torch.cat([d_b, qs], dim=1)


# ---------------------------------------------------------------------------
# Public dispatch: fields / reconstruct / assemble, keyed by format.
# ---------------------------------------------------------------------------

_GRID_FMTS = ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S")
_CHUNK_ELEMS = 128 * 1024 * 1024


def iq_fields(blocks: torch.Tensor, fmt: str,
              qw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Quantize (n_blocks, block) float32 superblocks into IQ fields.

    Chunks along the block axis to bound the exhaustive-search temporaries.
    """
    block = IQ_BLOCK_BYTES[fmt][0]
    n = blocks.shape[0]
    ng = _META[fmt]["ngrid"] if fmt in _GRID_FMTS else 16
    step = max(1, _CHUNK_ELEMS // max(block * ng, 1))
    if n <= step:
        return _iq_fields_one(blocks, fmt, qw)
    parts: list[dict[str, torch.Tensor]] = []
    for i in range(0, n, step):
        cw = qw[i:i + step] if (qw is not None and qw.dim() >= 2) else qw
        parts.append(_iq_fields_one(blocks[i:i + step], fmt, cw))
    return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}


def _iq_fields_one(blocks, fmt, qw):
    if fmt in _GRID_FMTS:
        return _grid_fields(blocks, fmt, qw)
    if fmt == "IQ4_XS":
        return _fields_iq4_xs(blocks, qw)
    if fmt == "IQ4_NL":
        return _fields_iq4_nl(blocks, qw)
    raise ValueError(f"unsupported IQ format: {fmt}")


def iq_reconstruct(f: dict[str, torch.Tensor], fmt: str) -> torch.Tensor:
    if fmt in _GRID_FMTS:
        return _grid_recon(f, fmt)
    if fmt == "IQ4_XS":
        return _recon_iq4_xs(f)
    if fmt == "IQ4_NL":
        return _recon_iq4_nl(f)
    raise ValueError(f"unsupported IQ format: {fmt}")


def iq_assemble_bytes(f: dict[str, torch.Tensor], fmt: str) -> torch.Tensor:
    if fmt in _GRID_FMTS:
        return _assemble_grid(f, fmt)
    if fmt == "IQ4_XS":
        return _assemble_iq4_xs(f)
    if fmt == "IQ4_NL":
        return _assemble_iq4_nl(f)
    raise ValueError(f"unsupported IQ format: {fmt}")
