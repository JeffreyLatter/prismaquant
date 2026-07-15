"""NVFP4-CB / FP8-CB vector-quantization codebook formats (emulation).

A codeword is a d=8 vector of grid values (E2M1 for the ``fp4`` family, E4M3
for the ``fp8`` family). A k-bit index per 8 weights gives ``k/8`` index bpw;
the fp4 family adds NVFP4-identical group-16 E4M3 scales (+0.5 bpw), the fp8
family a per-output-channel fp32 scale (negligible bpw). A decoded fp4 tile is
bit-compatible NVFP4 by construction (grid values on the E2M1 grid, NVFP4 group
scale), so it feeds the CUTLASS FP4 path unchanged.

This module is Milestone A: emulation only. One weighted-VQ field quantizer
feeds the emulation ``reconstruct`` (what cost measurement scores); the byte
packer and exporter land in Milestone B and must share this exact math path.

Two VQ modes:

* ``full`` — one ``2^k`` codebook over the 8-dim vector, exhaustive weighted
  argmin (chunked). Only feasible for k<=14; raises above without an explicit
  codebook.
* ``product`` (default) — the 8-dim vector splits into two 4-dim halves, each
  with its own ``2^(k/2)`` sub-codebook (ceil/floor bit split for odd k). Feasible
  for the whole NVFP4-CB ladder (k=12..24).

The weighted objective is llama.cpp/imatrix style:
``sum_j w_j (x_j - c_j)^2`` per codeword, with per-input-column ``col_weights``
(the same plumbing as the GGUF lane).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

VEC_DIM = 8
SUPERBLOCK = 256
FP4_GROUP = 16
FP8_ELEMENT_MAX = 448.0
# Flat-table feasibility ceiling (encode-side exhaustive argmin + serve-side
# LUT). Above this a structured/learned codebook must be supplied explicitly.
MAX_FLAT_K = 14
# Slice stacked/huge tensors along the leading dim to bound VQ temporaries
# (mirrors gguf_slice_max_elems' 64M IQ threshold — UMA swap-kill guard).
_SLICE_MAX_ELEMS = 64 * 1024 * 1024
# Row-chunk bound for the (rows*nvec, K) distance sweep.
_SCORE_CHUNK_ELEMS = 1 << 26

_DATA = Path(__file__).resolve().parent / "data" / "nvfp4_cb_lattices.pt"
_LATTICE_SEED = 1234
_LATTICE_SAMPLES = 1 << 17
_LATTICE_ITERS = 12

# E2M1: {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _grid_split(k: int) -> tuple[int, int]:
    """Codeword-index bit split for product mode: (ceil, floor)."""
    return (k + 1) // 2, k // 2


@lru_cache(maxsize=None)
def _e2m1_grid(device: str) -> torch.Tensor:
    signed = {0.0}
    for v in _E2M1_VALUES[1:]:
        signed.add(v)
        signed.add(-v)
    return torch.tensor(sorted(signed), dtype=torch.float32,
                        device=torch.device(device))


def _snap_to_grid(t: torch.Tensor, grid: str) -> torch.Tensor:
    """Project every coordinate onto the element grid (nearest)."""
    if grid == "fp8":
        return (t.clamp(-FP8_ELEMENT_MAX, FP8_ELEMENT_MAX)
                .to(torch.float8_e4m3fn).to(torch.float32))
    if grid != "fp4":
        raise ValueError(f"unknown grid {grid!r} (expected 'fp4' or 'fp8')")
    cb = _e2m1_grid(str(t.device))
    x = t.to(torch.float32).contiguous()
    idx = torch.bucketize(x, cb)
    lo = cb[(idx - 1).clamp_min(0)]
    hi = cb[idx.clamp_max(cb.numel() - 1)]
    return torch.where((hi - x).abs() < (x - lo).abs(), hi, lo)


# ---------------------------------------------------------------------------
# Weighted VQ assignment (the imatrix-weighted exhaustive argmin).
# ---------------------------------------------------------------------------

def _vq_assign(x: torch.Tensor, cb: torch.Tensor,
               wq: torch.Tensor | None) -> torch.Tensor:
    """Argmin_c sum_j wq_j (x_j - cb[c]_j)^2 per row of ``x``.

    ``x`` is (m, d), ``cb`` is (K, d), ``wq`` is (m, d) or None. The additive
    sum_j wq_j x_j^2 term is constant per row and dropped (cancels in argmin).
    """
    m, K = x.shape[0], cb.shape[0]
    cb = cb.to(x.device, torch.float32)
    cb_sq = cb * cb
    cb_t = cb.t().contiguous()
    idx = torch.empty(m, dtype=torch.long, device=x.device)
    chunk = max(1, _SCORE_CHUNK_ELEMS // max(K, 1))
    if wq is None:
        cb_sqnorm = cb_sq.sum(dim=-1)
        for a in range(0, m, chunk):
            b = min(m, a + chunk)
            dist = cb_sqnorm - 2.0 * (x[a:b] @ cb_t)
            idx[a:b] = dist.argmin(dim=-1)
        return idx
    cb_sq_t = cb_sq.t().contiguous()
    for a in range(0, m, chunk):
        b = min(m, a + chunk)
        wc = wq[a:b]
        term1 = (wc * x[a:b]) @ cb_t
        term2 = wc @ cb_sq_t
        idx[a:b] = (term2 - 2.0 * term1).argmin(dim=-1)
    return idx


# ---------------------------------------------------------------------------
# Fixed lattice + learned codebook (weighted Lloyd on the element grid).
# ---------------------------------------------------------------------------

def _lloyd(samples: torch.Tensor, init: torch.Tensor, grid: str,
           weights: torch.Tensor | None, iters: int, seed: int) -> torch.Tensor:
    """Grid-snapped weighted Lloyd. Every centroid coordinate is projected
    onto the element grid after each update, so codewords stay grid-valued."""
    cb = _snap_to_grid(init.to(torch.float32), grid)
    K, d = cb.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    for _ in range(int(iters)):
        assign = _vq_assign(samples, cb, weights)
        onehot = torch.zeros(samples.shape[0], K, device=samples.device)
        onehot.scatter_(1, assign.unsqueeze(1), 1.0)
        if weights is None:
            counts = onehot.sum(dim=0)                       # (K,)
            summ = onehot.t() @ samples                      # (K, d)
            new = summ / counts.clamp_min(1.0).unsqueeze(-1)
        else:
            wsum = onehot.t() @ weights                      # (K, d)
            summ = onehot.t() @ (weights * samples)          # (K, d)
            new = summ / wsum.clamp_min(1e-12)
            counts = onehot.sum(dim=0)
        empty = counts == 0
        if bool(empty.any()):
            n_empty = int(empty.sum())
            pick = torch.randint(0, samples.shape[0], (n_empty,),
                                 generator=gen).to(samples.device)
            new[empty] = samples[pick]
        cb = _snap_to_grid(new, grid)
    return cb


@lru_cache(maxsize=None)
def _lattice_file() -> dict[str, torch.Tensor]:
    if _DATA.exists():
        return torch.load(_DATA, map_location="cpu", weights_only=True)
    return {}


def _lattice_key(k: int, grid: str, d: int) -> str:
    return f"{grid}_d{d}_k{k}"


@lru_cache(maxsize=None)
def _fixed_lattice_cpu(k: int, grid: str, d: int) -> torch.Tensor:
    if k > MAX_FLAT_K:
        raise ValueError(
            f"flat codebook infeasible at k={k} (2^{k} codewords > "
            f"2^{MAX_FLAT_K}); provide an explicit/structured codebook")
    cached = _lattice_file().get(_lattice_key(k, grid, d))
    if cached is not None:
        return cached.to(torch.float32).contiguous()
    return _build_lattice(k, grid, d)


def _build_lattice(k: int, grid: str, d: int) -> torch.Tensor:
    """Deterministic universal lattice: grid-snapped Lloyd on seeded
    standard-Gaussian d-dim samples. Regenerated identically on cache miss."""
    K = 1 << k
    gen = torch.Generator(device="cpu").manual_seed(_LATTICE_SEED + k * 131 + d)
    m = max(_LATTICE_SAMPLES, K * 16)
    samples = torch.randn(m, d, generator=gen)
    perm = torch.randperm(m, generator=gen)[:K]
    init = samples[perm]
    return _lloyd(samples, init, grid, None, _LATTICE_ITERS, _LATTICE_SEED)


def fixed_lattice(k: int, grid: str, d: int = 8) -> torch.Tensor:
    """Universal (2^k, d) codebook of grid-valued codewords."""
    return _fixed_lattice_cpu(int(k), str(grid), int(d))


def learn_codebook(vectors: torch.Tensor, k: int, *, grid: str,
                   col_weights: torch.Tensor | None = None,
                   init: torch.Tensor | None = None, iters: int = 4,
                   seed: int = 0) -> torch.Tensor:
    """Weighted Lloyd codebook on the element grid, deterministic given
    ``seed`` + ``init``. Returns a (2^k, d) grid-valued tensor."""
    vectors = vectors.to(torch.float32)
    d = vectors.shape[-1]
    vectors = vectors.reshape(-1, d)
    if init is None:
        init = fixed_lattice(k, grid, d).to(vectors.device)
    else:
        init = init.to(vectors.device, torch.float32)
    if (1 << int(k)) != init.shape[0]:
        raise ValueError(f"init has {init.shape[0]} entries, expected 2^{k}")
    weights = None
    if col_weights is not None:
        weights = torch.broadcast_to(
            col_weights.to(vectors.device, torch.float32), vectors.shape
        ).contiguous()
    return _lloyd(vectors, init, grid, weights, iters, seed)


def _resolve_codebook(k: int, grid: str, mode: str,
                      codebook: torch.Tensor | tuple | None,
                      device: torch.device):
    if mode == "full":
        if codebook is None:
            cb = fixed_lattice(k, grid, VEC_DIM)
        else:
            cb = codebook
        return cb.to(device, torch.float32)
    if mode == "product":
        k_lo, k_hi = _grid_split(k)
        if codebook is None:
            lo = fixed_lattice(k_lo, grid, VEC_DIM // 2)
            hi = fixed_lattice(k_hi, grid, VEC_DIM // 2)
        else:
            lo, hi = codebook
        return (lo.to(device, torch.float32), hi.to(device, torch.float32))
    raise ValueError(f"unknown mode {mode!r} (expected 'full' or 'product')")


# ---------------------------------------------------------------------------
# Scale + vectorize.
# ---------------------------------------------------------------------------

def _fp4_group_scale(w2d: torch.Tensor) -> torch.Tensor:
    """NVFP4 group-16 effective scale (rows, in//16), via the export codec so
    resident emulation == served bytes."""
    from . import export_native_compressed as enc

    rows, in_f = w2d.shape
    grouped = w2d.to(torch.float32).reshape(rows, in_f // FP4_GROUP, FP4_GROUP)
    scale_real, global_real = enc._select_nvfp4_pack_scales_and_global(grouped)
    return enc._nvfp4_effective_scale_from_real(
        scale_real, global_real, quantize_fp8=True)


def _per_element_scale(scales: torch.Tensor, grid: str,
                       in_f: int) -> torch.Tensor:
    if grid == "fp4":
        return scales.repeat_interleave(FP4_GROUP, dim=1)
    return scales.expand(scales.shape[0], in_f)


def _scale_and_vectorize(w2d: torch.Tensor, grid: str):
    """Return (vectors (nvec, 8), scales, per_element_scale). ``scales`` is the
    stored scale plane: (rows, in//16) for fp4, (rows, 1) for fp8."""
    rows, in_f = w2d.shape
    wf = w2d.to(torch.float32)
    if grid == "fp4":
        scales = _fp4_group_scale(wf)
    elif grid == "fp8":
        scales = (wf.abs().amax(dim=-1, keepdim=True) / FP8_ELEMENT_MAX
                  ).clamp_min(1e-12)
    else:
        raise ValueError(f"unknown grid {grid!r}")
    pes = _per_element_scale(scales, grid, in_f)
    x = wf / pes
    vectors = x.reshape(rows * (in_f // VEC_DIM), VEC_DIM)
    return vectors, scales, pes


# ---------------------------------------------------------------------------
# Fields / reconstruct.
# ---------------------------------------------------------------------------

def _col_weight_vectors(cw2d: torch.Tensor) -> torch.Tensor:
    """Reshape a per-element (rows, in) weight to (nvec, 8) with a dead-vector
    guard (all-zero weight -> unweighted)."""
    wq = cw2d.reshape(-1, VEC_DIM)
    mass = wq.sum(dim=-1, keepdim=True)
    return torch.where(mass == 0, torch.ones_like(wq), wq)


def _fields_block(w2d: torch.Tensor, k: int, grid: str, mode: str,
                  cb, cw2d: torch.Tensor | None) -> dict:
    rows, in_f = w2d.shape
    vectors, scales, _ = _scale_and_vectorize(w2d, grid)
    nvec_per_row = in_f // VEC_DIM
    wq = None
    if cw2d is not None:
        wq = _col_weight_vectors(cw2d)
    if mode == "full":
        idx = _vq_assign(vectors, cb, wq)
        indices = idx.reshape(rows, nvec_per_row)
    else:
        lo, hi = cb
        wq_lo = wq[:, :VEC_DIM // 2] if wq is not None else None
        wq_hi = wq[:, VEC_DIM // 2:] if wq is not None else None
        idx_lo = _vq_assign(vectors[:, :VEC_DIM // 2], lo, wq_lo)
        idx_hi = _vq_assign(vectors[:, VEC_DIM // 2:], hi, wq_hi)
        indices = torch.stack([idx_lo, idx_hi], dim=-1).reshape(
            rows, nvec_per_row, 2)
    return {"indices": indices, "scales": scales}


def nvfp4_cb_fields(w: torch.Tensor, k: int, *, grid: str = "fp4",
                    mode: str = "product",
                    col_weights: torch.Tensor | None = None,
                    codebook: torch.Tensor | tuple | None = None) -> dict:
    """Quantize ``w`` (2-D or 3-D stacked experts) into VQ fields.

    Returns at least {"indices", "scales"}; the resolved codebook is echoed
    back under "codebook" so reconstruct and the packer share one table.
    """
    in_f = int(w.shape[-1])
    if in_f % SUPERBLOCK != 0:
        raise ValueError(
            f"in_features={in_f} must be a multiple of {SUPERBLOCK}")
    orig_shape = tuple(w.shape)
    w2d = w.reshape(-1, in_f)
    rows = w2d.shape[0]
    cb = _resolve_codebook(k, grid, mode, codebook, w2d.device)

    cw2d = None
    if col_weights is not None:
        cw2d = torch.broadcast_to(
            col_weights.to(w2d.device, torch.float32), orig_shape
        ).reshape(rows, in_f)

    row_step = max(1, _SLICE_MAX_ELEMS // max(in_f, 1))
    if rows <= row_step:
        out = _fields_block(w2d, k, grid, mode, cb, cw2d)
    else:
        parts = []
        for a in range(0, rows, row_step):
            b = min(rows, a + row_step)
            cw = cw2d[a:b] if cw2d is not None else None
            parts.append(_fields_block(w2d[a:b], k, grid, mode, cb, cw))
        out = {
            "indices": torch.cat([p["indices"] for p in parts], dim=0),
            "scales": torch.cat([p["scales"] for p in parts], dim=0),
        }
    out["shape"] = orig_shape
    out["codebook"] = cb
    return out


def nvfp4_cb_reconstruct(fields: dict, k: int, *, grid: str = "fp4",
                         mode: str = "product",
                         codebook: torch.Tensor | tuple | None = None
                         ) -> torch.Tensor:
    shape = fields.get("shape")
    indices = fields["indices"]
    scales = fields["scales"]
    rows = indices.shape[0]
    in_f = int(shape[-1]) if shape is not None else scales.shape[-1] * (
        FP4_GROUP if grid == "fp4" else 1)
    cb = fields.get("codebook")
    if cb is None:
        cb = _resolve_codebook(k, grid, mode, codebook, indices.device)
    if mode == "full":
        vecs = cb[indices]                                   # (rows, nvec, 8)
        recon = vecs.reshape(rows, in_f)
    else:
        lo, hi = cb
        vlo = lo[indices[..., 0]]
        vhi = hi[indices[..., 1]]
        recon = torch.cat([vlo, vhi], dim=-1).reshape(rows, in_f)
    pes = _per_element_scale(scales.to(recon.dtype), grid, in_f)
    recon = recon * pes
    if shape is not None:
        recon = recon.reshape(shape)
    return recon


def make_nvfp4_cb_qdq(k: int, grid: str = "fp4", mode: str = "product"):
    """Single-source emulation closure ``(w, col_weights=None) -> w_hat``
    used by both cost and (Milestone B) the packer."""
    def f(w: torch.Tensor, col_weights: torch.Tensor | None = None
          ) -> torch.Tensor:
        fields = nvfp4_cb_fields(w, k, grid=grid, mode=mode,
                                 col_weights=col_weights)
        return nvfp4_cb_reconstruct(fields, k, grid=grid, mode=mode).to(w.dtype)
    return f
