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

Three VQ modes:

* ``full`` — one ``2^k`` codebook over the 8-dim vector, exhaustive weighted
  argmin (chunked). Only feasible for k<=14; raises above without an explicit
  codebook.
* ``product`` (default) — the 8-dim vector splits into two 4-dim halves, each
  with its own ``2^(k/2)`` sub-codebook (ceil/floor bit split for odd k). Feasible
  for the whole NVFP4-CB ladder (k=12..24).
* ``signed`` — sign-magnitude factorization (the IQ-family move): 8 explicit
  sign bits + an ``m = k-8``-bit index into a MAGNITUDE codebook over the
  positive half-grid. A flat codebook burns most of its entries covering sign
  patterns (~2^8 per magnitude shape at d=8, leaving ~2^(k-8) effective
  magnitude shapes); factoring the signs out spends all 2^m entries on
  magnitude shapes (exp-1 diagnosis: this is why IQ2_S beat flat CB +66% at
  matched bytes). Encode is exactly separable under weighted L2 (see
  ``_fields_block``); tables are tiny (m=5..8 -> 32..256 entries).

The weighted objective is llama.cpp/imatrix style:
``sum_j w_j (x_j - c_j)^2`` per codeword, with per-input-column ``col_weights``
(the same plumbing as the GGUF lane).

Scale search: CB encode sweeps the per-group (fp4) / per-row (fp8) scale over a
grid of E4M3-legal candidates and picks the one minimizing weighted
reconstruction error in the ORIGINAL weight domain, then refines with 2
WLS-refit fixed-point iterations — rendering parity with the IQ lane's
27-candidate ``_grid_fields`` sweep. ``scale_sweep=True`` is the default for all
three modes and both grids. The amax/6 (fp4) / amax/448 (fp8) one-shot scale is
always in the candidate set, so the sweep is never worse than one-shot. **The
Phase-0 exp-1 0.6B results (pre-c3f8c6d) used one-shot scales for CB while the
IQ arms got their scale sweep — that rendering asymmetry is corrected here;
re-run before trusting any CB-vs-IQ delta.**

Scale coding (fp4 family): the v1 plane stores each group-16 scale as a bare
E4M3 byte — on real LLM weights ~90% of group scales sit in e4m3's SUBNORMAL
band, where the sweep's candidates collapse to ~1-2 distinct values
(two-tier-scale-spec.md §3). Layout v2 ("two_tier", opt-in until the serving
gates clear) stores a per-superblock E8M0 super `2^(E-127)` plus 16 4-bit sub
codes into an e4m3-exact multiplier table; the composition IS an E4M3 scale by
construction, the encoder explores every reachable value across the ideal-scale
window (~20+ distinct candidates where v1 had 1-2), and the scale plane shrinks
16 B -> 9 B per superblock (0.5 -> 0.28125 bpw). Every fp4 exp-1/1b number was
measured under v1 coding.
"""
from __future__ import annotations

import os
import threading
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import torch

from .cb_layout import (
    CODEWORDS_PER_SUPERBLOCK,
    FP4_GROUP,
    FP4_SCALE_GROUPS_PER_SUPERBLOCK,
    INDEX_BYTES_PER_K,
    SCALE_CODING_TWO_TIER,
    SCALE_CODING_V1,
    SCALE_PLANE_BYTES,
    SUPERBLOCK,
    VEC_DIM,
    codebook_subtable_shapes,
    family_for,
    subtable_bit_widths,
    type_size as _serialized_type_size,
)

FP8_ELEMENT_MAX = 448.0
NVFP4_GRID_MAX = 6.0            # max(|E2M1|); amax/6 == no-clip one-shot scale
# Flat-table feasibility ceiling (encode-side exhaustive argmin + serve-side
# LUT). Above this a structured/learned codebook must be supplied explicitly.
MAX_FLAT_K = 14
# Slice stacked/huge tensors along the leading dim to bound VQ temporaries
# (mirrors gguf_slice_max_elems' 64M IQ threshold — UMA swap-kill guard).
_SLICE_MAX_ELEMS = 64 * 1024 * 1024
# Row-chunk bound for the (rows*nvec, K) distance sweep.
_SCORE_CHUNK_ELEMS = 1 << 26

# Scale search (rendering parity with the IQ lane's _grid_fields sweep):
# number of clipping-level candidates and fixed-point refit iterations. The
# fp4 grid sweeps amax/L for L spanning [6, 4] (grid max 6; the JSO {6,4}
# insight as a grid); fp8 spans amax/L for L in [448, 448*4/6]. L=grid-max is
# candidate 0 == the amax/grid-max one-shot, so the sweep is never worse.
_SCALE_SWEEP_CANDIDATES = 16
_SCALE_SWEEP_REFIT_ITERS = 2
_E4M3 = torch.float8_e4m3fn
# Smallest positive fp8_e4m3fn value (subnormal 2^-9), an E4M3-exact floor
# that keeps a chosen block scale strictly positive (never underflow to 0).
_E4M3_MIN_POS = 2.0 ** -9

# --- Two-tier scale coding (layout v2, fp4 family only) -----------------
# docs/lanes/nvfp4-cb/two-tier-scale-spec.md: per-256 E8M0 super (2^(E-127))
# x per-16 4-bit sub code into a fixed table of e4m3-exact multipliers;
# the composition lands exactly on E4M3 by construction (legality mask, no
# rounding anywhere). Scale plane 16 B -> 9 B per superblock (0.28125 bpw).
TWO_TIER_SUPER_BIAS = 127
# T4_2oct8m (spec §1.3): all 8 e4m3 mantissa steps x 2 octaves.
TWO_TIER_SUB_TABLE = (1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
                      2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75)
# Window margin around [min_ideal/T_max, 1.5*max_ideal] (spec §1.4 derives the
# E window from the ideal group scales; the conservatism note says a wider
# window can only improve, so pad one octave on each side).
_TWO_TIER_WINDOW_PAD = 1
_TWO_TIER_MAX_WINDOW = 14

# --- Tiered encoder (docs/lanes/nvfp4-cb/encode_tiers.md) -----------------
# PRISMAQUANT_CB_ENCODE_TIER in {fast, balanced, max}:
#   max      — the original exhaustive sweep, bit-identical
#              (regression-pinned);
#   balanced — analytic scale init (usage-calibrated second-moment match,
#              s0 = sqrt(sum q w^2 / (sum q * m2_used)); m2_used measured
#              from a pilot encode of the tensor's leading rows) + a
#              log-spaced micro-sweep of +-2 neighbors + the amax/grid-max
#              guarantee candidate, scored from per-(vector, entry) moments
#              (err = C - 2sB + s^2A; A, B scale-independent, built once
#              per chunk — the _grid_fields/_sweep_errs identity), then the
#              exact 2 WLS refits;
#   fast     — same, +-1 neighbors + 1 refit.
# Measured basis (encode_cost_4b.json + encode_tiers.md): the 4B FP8_CB
# winners spread across high-clip candidates (no JSO-style collapse; pruning
# alone <= 2.6x) and refits accept at 99.5%/95.2% (>=1 refit everywhere);
# the naive all-codeword m2 lands the wrong basin (2-3x err) so m2_used is
# calibrated from actually-USED assignments; a scalar RTN-snap proxy ranking
# was measured INVALID (+21% recon fp8) and dropped. The micro-sweep span
# covers the measured s0-vs-sweep ratio error (<=1.15).
_ENCODE_TIER_ENV = "PRISMAQUANT_CB_ENCODE_TIER"
_ENCODE_TIERS = ("fast", "balanced", "max")
_ENCODE_TIER_DEFAULT = "balanced"
_TIER_REFITS = {"fast": 1, "balanced": _SCALE_SWEEP_REFIT_ITERS,
                "max": _SCALE_SWEEP_REFIT_ITERS}
_TIER_MICRO_SPAN = {"fast": 1, "balanced": 2}     # +- steps around s0
_TIER_MICRO_RATIO = {"fast": 1.1, "balanced": 1.075}
# Per-group hill-climb extension after the micro-grid (measured: the reach,
# not granularity, sets quality — q_proj span curve hits max-parity at
# +-24% and BEATS max at +-34%; down_proj needs per-group adaptive reach).
_TIER_HILL_ITERS = {"fast": 2, "balanced": 4}
_PILOT_ROWS = 256
_ENCODE_COMPILE_ENV = "PRISMAQUANT_CB_ENCODE_COMPILE"


def _resolve_encode_tier(tier: str | None) -> str:
    t = tier if tier is not None else os.environ.get(
        _ENCODE_TIER_ENV, _ENCODE_TIER_DEFAULT)
    t = str(t).strip().lower()
    if t not in _ENCODE_TIERS:
        raise ValueError(
            f"unknown encode tier {t!r} (expected one of {_ENCODE_TIERS})")
    return t

_DATA = Path(__file__).resolve().parent / "data" / "nvfp4_cb_lattices.pt"
_LATTICE_SEED = 1234
_LATTICE_SAMPLES = 1 << 17
_LATTICE_ITERS = 12

# E2M1: {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@lru_cache(maxsize=None)
def _e2m1_grid(device: str) -> torch.Tensor:
    signed = {0.0}
    for v in _E2M1_VALUES[1:]:
        signed.add(v)
        signed.add(-v)
    return torch.tensor(sorted(signed), dtype=torch.float32,
                        device=torch.device(device))


def _snap_to_grid(t: torch.Tensor, grid: str,
                  positive: bool = False) -> torch.Tensor:
    """Project every coordinate onto the element grid (nearest).

    ``positive=True`` restricts to the non-negative half-grid (magnitude
    codebooks for signed mode): clamp to >=0 first — the nearest full-grid
    value of a non-negative input is itself non-negative, so a plain snap
    then lands on the half-grid."""
    if positive:
        t = t.clamp_min(0)
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

def _vq_dist_argmin_eager(term2: torch.Tensor,
                          term1: torch.Tensor) -> torch.Tensor:
    """argmin_c of ``term2 - 2*term1`` over the trailing codeword axis."""
    return (term2 - 2.0 * term1).argmin(dim=-1)


@lru_cache(maxsize=None)
def _vq_dist_argmin_compiled():
    return torch.compile(_vq_dist_argmin_eager, dynamic=True)


def _vq_dist_argmin(term2: torch.Tensor, term1: torch.Tensor) -> torch.Tensor:
    """Fused distance + argmin.

    Eagerly this materializes a whole (m, K) fp32 distance plane, writes it,
    and reads it straight back for the reduction — three passes over the
    largest tensor in the encoder for one index per row. Compiled, the
    subtraction folds into the reduction and the plane never exists (measured
    2.9x on the production shapes). Same per-element arithmetic and the same
    first-occurrence tie rule, so the chosen codewords are unchanged."""
    if _encode_compile_on():
        try:
            _raise_encode_recompile_limit()
            return _vq_dist_argmin_compiled()(term2, term1)
        except Exception:
            pass
    return _vq_dist_argmin_eager(term2, term1)


def _vq_assign(x: torch.Tensor, cb: torch.Tensor,
               wq: torch.Tensor | None,
               wq_period: int | None = None) -> torch.Tensor:
    """Argmin_c sum_j wq_j (x_j - cb[c]_j)^2 per row of ``x``.

    ``x`` is (m, d), ``cb`` is (K, d), ``wq`` is (m, d) or None. The additive
    sum_j wq_j x_j^2 term is constant per row and dropped (cancels in argmin).

    ``wq_period``: ``wq`` repeats every ``wq_period`` rows (per-column imatrix
    broadcast over rows), so the scale-independent ``wq @ cb_sq^T`` term is
    built once from the base block and broadcast instead of being materialized
    at (m, K) on every call. Same per-element values, same reduction axis.
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
            idx[a:b] = _vq_dist_argmin(cb_sqnorm, x[a:b] @ cb_t)
        return idx
    cb_sq_t = cb_sq.t().contiguous()
    P = wq_period
    if not (P and 0 < P < m and m % P == 0 and chunk % P == 0):
        P = None
    if P is not None:
        term2 = wq[:P] @ cb_sq_t                      # (P, K)
        for a in range(0, m, chunk):
            b = min(m, a + chunk)
            r = (b - a) // P
            term1 = (wq[a:b] * x[a:b]) @ cb_t
            idx[a:b] = _vq_dist_argmin(
                term2.reshape(1, P, K), term1.reshape(r, P, K)).reshape(-1)
        return idx
    for a in range(0, m, chunk):
        b = min(m, a + chunk)
        wc = wq[a:b]
        term1 = (wc * x[a:b]) @ cb_t
        term2 = wc @ cb_sq_t
        idx[a:b] = _vq_dist_argmin(term2, term1)
    return idx


# ---------------------------------------------------------------------------
# Fixed lattice + learned codebook (weighted Lloyd on the element grid).
# ---------------------------------------------------------------------------

def _lloyd(samples: torch.Tensor, init: torch.Tensor, grid: str,
           weights: torch.Tensor | None, iters: int, seed: int,
           positive: bool = False) -> torch.Tensor:
    """Grid-snapped weighted Lloyd. Every centroid coordinate is projected
    onto the element grid after each update, so codewords stay grid-valued
    (the positive half-grid for magnitude codebooks)."""
    cb = _snap_to_grid(init.to(torch.float32), grid, positive=positive)
    K, d = cb.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    for _ in range(int(iters)):
        assign = _vq_assign(samples, cb, weights)
        # Index-based accumulation: a dense (m, K) one-hot is ~51 GB fp32 at
        # 27B-Linear scale (m~3.1M, K=4096) and swap-kills a UMA box.
        counts = torch.bincount(assign, minlength=K).to(samples.dtype)
        if weights is None:
            summ = torch.zeros(K, d, dtype=samples.dtype,
                               device=samples.device)
            summ.index_add_(0, assign, samples)
            new = summ / counts.clamp_min(1.0).unsqueeze(-1)
        else:
            wsum = torch.zeros(K, d, dtype=samples.dtype,
                               device=samples.device)
            wsum.index_add_(0, assign, weights)
            summ = torch.zeros(K, d, dtype=samples.dtype,
                               device=samples.device)
            summ.index_add_(0, assign, weights * samples)
            new = summ / wsum.clamp_min(1e-12)
        empty = counts == 0
        if bool(empty.any()):
            n_empty = int(empty.sum())
            pick = torch.randint(0, samples.shape[0], (n_empty,),
                                 generator=gen).to(samples.device)
            new[empty] = samples[pick]
        cb = _snap_to_grid(new, grid, positive=positive)
    return cb


@lru_cache(maxsize=None)
def _lattice_file() -> dict[str, torch.Tensor]:
    if _DATA.exists():
        return torch.load(_DATA, map_location="cpu", weights_only=True)
    return {}


def _lattice_key(k: int, grid: str, d: int, positive: bool = False) -> str:
    return f"{grid}{'pos' if positive else ''}_d{d}_k{k}"


@lru_cache(maxsize=None)
def _fixed_lattice_cpu(k: int, grid: str, d: int,
                       positive: bool = False) -> torch.Tensor:
    if k > MAX_FLAT_K:
        raise ValueError(
            f"flat codebook infeasible at k={k} (2^{k} codewords > "
            f"2^{MAX_FLAT_K}); provide an explicit/structured codebook")
    cached = _lattice_file().get(_lattice_key(k, grid, d, positive))
    if cached is not None:
        return cached.to(torch.float32).contiguous()
    return _build_lattice(k, grid, d, positive=positive)


def _build_lattice(k: int, grid: str, d: int,
                   positive: bool = False) -> torch.Tensor:
    """Deterministic universal lattice: grid-snapped Lloyd on seeded samples
    drawn from the *post-normalization* distribution each grid's encoder
    actually produces. Regenerated on cache miss.

    Both families must train at the data scale or the codewords cluster far
    from the data and reconstruction collapses (2026-07-15: the original fp4
    path trained on standard N(0,1) while NVFP4 group-16 normalization yields
    normalized weights of std ~2.9 / absmax ~6, giving whole-model emulated
    KL ~15 / top1 ~0 — a measurement bug that would have falsely killed the
    family). Fixes:
      * fp8 — rows scale by amax/448, so scaled vectors live at
        ~sigma·448/amax_sigma; train at that scale (amax ~ 4 sigma).
      * fp4 — group-16 amax→6 normalization; train on genuinely NVFP4-
        normalized Gaussian weights via the encoder's own
        ``_scale_and_vectorize`` (no hand-tuned scale constant), so the
        lattice matches the exact distribution the encoder feeds it.
    """
    K = 1 << k
    gen = torch.Generator(device="cpu").manual_seed(_LATTICE_SEED + k * 131 + d)
    m = max(_LATTICE_SAMPLES, K * 16)
    if grid == "fp8":
        samples = torch.randn(m, d, generator=gen) * (FP8_ELEMENT_MAX / 4.0)
    else:  # fp4: normalized-weight samples at the true encoder scale.
        in_f = 512  # multiple of the group-16 scale window
        n8 = (m * d + VEC_DIM - 1) // VEC_DIM
        rows = (n8 * VEC_DIM + in_f - 1) // in_f
        w = torch.randn(rows, in_f, generator=gen)
        vectors, _, _ = _scale_and_vectorize(w, "fp4")   # (rows*64, 8), std~2.9
        samples = vectors.reshape(-1, d)[:m].contiguous()
    if positive:
        # Magnitude lattice: train on |x| of the same post-normalization
        # distribution (exactly what the signed-mode encoder searches over).
        samples = samples.abs()
    if torch.cuda.is_available():
        samples = samples.cuda()
    perm = torch.randperm(samples.shape[0], generator=gen).to(samples.device)[:K]
    init = samples[perm]
    return _lloyd(samples, init, grid, None, _LATTICE_ITERS, _LATTICE_SEED,
                  positive=positive).cpu()


def fixed_lattice(k: int, grid: str, d: int = 8,
                  positive: bool = False) -> torch.Tensor:
    """Universal (2^k, d) codebook of grid-valued codewords (positive
    half-grid magnitude codewords when ``positive=True``)."""
    return _fixed_lattice_cpu(int(k), str(grid), int(d), bool(positive))


def learn_codebook(vectors: torch.Tensor, k: int, *, grid: str,
                   col_weights: torch.Tensor | None = None,
                   init: torch.Tensor | None = None, iters: int = 4,
                   seed: int = 0, positive: bool = False) -> torch.Tensor:
    """Weighted Lloyd codebook on the element grid. Returns a (2^k, d)
    grid-valued tensor (positive half-grid when ``positive=True`` — pass
    ``|vectors|`` to learn a signed-mode magnitude codebook). Deterministic
    given ``seed`` + ``init`` on CPU; on CUDA the index_add_ float atomics
    can flip grid-snap ties across runs, so ship the resulting codebook
    rather than regenerating it."""
    vectors = vectors.to(torch.float32)
    d = vectors.shape[-1]
    vectors = vectors.reshape(-1, d)
    if init is None:
        init = fixed_lattice(k, grid, d, positive=positive).to(vectors.device)
    else:
        init = init.to(vectors.device, torch.float32)
    if (1 << int(k)) != init.shape[0]:
        raise ValueError(f"init has {init.shape[0]} entries, expected 2^{k}")
    weights = None
    if col_weights is not None:
        weights = torch.broadcast_to(
            col_weights.to(vectors.device, torch.float32), vectors.shape
        ).contiguous()
    return _lloyd(vectors, init, grid, weights, iters, seed,
                  positive=positive)


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
        n_sub = family_for(grid, mode).n_sub
        expected_shapes = codebook_subtable_shapes(k, mode, n_sub)
        if codebook is None:
            tables = tuple(
                fixed_lattice(bits, grid, sub_dim)
                for bits, (_, sub_dim) in zip(
                    subtable_bit_widths(k, mode, n_sub), expected_shapes
                )
            )
        else:
            tables = tuple(codebook)
        actual_shapes = tuple(tuple(int(dim) for dim in table.shape)
                              for table in tables)
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"{grid} {mode} k={k} codebook shapes {actual_shapes} do "
                f"not match canonical serialized shapes {expected_shapes}"
            )
        return tuple(t.to(device, torch.float32) for t in tables)
    if mode == "signed":
        n_sub = family_for(grid, mode).n_sub
        (m,) = subtable_bit_widths(k, mode, n_sub)
        (expected_shape,) = codebook_subtable_shapes(k, mode, n_sub)
        if codebook is None:
            cb = fixed_lattice(m, grid, VEC_DIM, positive=True)
        else:
            cb = codebook
        cb = cb.to(device, torch.float32)
        actual_shape = tuple(int(dim) for dim in cb.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{grid} {mode} k={k} codebook shape {actual_shape} does "
                f"not match canonical serialized shape {expected_shape}"
            )
        if bool((cb < 0).any()):
            raise ValueError(
                "signed-mode magnitude codebook must be non-negative "
                "(sign-optimality requires codewords on the positive "
                "half-grid)")
        return cb
    raise ValueError(
        f"unknown mode {mode!r} (expected 'full', 'product' or 'signed')")


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


def _mode_encode(vectors: torch.Tensor, mode: str, cb, wq,
                 wq_period: int | None = None) -> dict:
    """VQ-assign scaled ``vectors`` (nvec, 8) under one mode. Returns per-mode
    index fields ({"idx": (nvec,) or (nvec, n_sub)}, + "signs" for signed)."""
    if mode == "full":
        return {"idx": _vq_assign(vectors, cb, wq, wq_period)}
    if mode == "signed":
        # Exactly separable under weighted L2: for any magnitude codeword
        # c >= 0, sum_j w_j (x_j - s_j c_j)^2 is minimized over s_j in {+-1}
        # by s_j = sign(x_j) (the cross-term -2 w_j s_j x_j c_j is largest
        # when s_j x_j >= 0, independent of which codeword is chosen), and at
        # that sign the objective equals sum_j w_j (|x_j| - c_j)^2. So the
        # weighted argmin over |x| plus signs = sign(x) IS the joint optimum
        # — no sign x magnitude search needed. Zero-safe: sign(0) -> +1.
        return {"idx": _vq_assign(vectors.abs(), cb, wq, wq_period),
                "signs": torch.where(vectors < 0, -1.0, 1.0)}
    n_sub = len(cb)
    sub_dim = VEC_DIM // n_sub
    idxs = []
    for i, table in enumerate(cb):
        xs = vectors[:, i * sub_dim:(i + 1) * sub_dim]
        ws = wq[:, i * sub_dim:(i + 1) * sub_dim] if wq is not None else None
        idxs.append(_vq_assign(xs, table, ws, wq_period))
    return {"idx": torch.stack(idxs, dim=-1)}


def _mode_decode(enc: dict, mode: str, cb) -> torch.Tensor:
    """Scaled-domain grid reconstruction (nvec, 8) from index fields."""
    if mode == "full":
        return cb[enc["idx"]]
    if mode == "signed":
        return cb[enc["idx"]] * enc["signs"]
    parts = [table[enc["idx"][:, i]] for i, table in enumerate(cb)]
    return torch.cat(parts, dim=-1)


def _enc_to_fields(enc: dict, mode: str, cb, rows: int, in_f: int,
                   nvec_per_row: int) -> dict:
    if mode == "full":
        return {"indices": enc["idx"].reshape(rows, nvec_per_row)}
    if mode == "signed":
        return {"indices": enc["idx"].reshape(rows, nvec_per_row),
                "signs": enc["signs"].reshape(rows, in_f)}
    return {"indices": enc["idx"].reshape(rows, nvec_per_row, len(cb))}


def _group_amax(w2d: torch.Tensor, grid: str) -> torch.Tensor:
    rows, in_f = w2d.shape
    if grid == "fp4":
        return w2d.reshape(rows, in_f // FP4_GROUP, FP4_GROUP).abs().amax(-1)
    return w2d.abs().amax(-1, keepdim=True)


def _group_reduce(err: torch.Tensor, grid: str) -> torch.Tensor:
    rows, in_f = err.shape
    if grid == "fp4":
        return err.reshape(rows, in_f // FP4_GROUP, FP4_GROUP).sum(-1)
    return err.sum(-1, keepdim=True)


def _snap_scale(s: torch.Tensor, grid: str) -> torch.Tensor:
    """Project a per-group scale onto the legal grid: E4M3 (fp4 block scale,
    exactly like NVFP4) or fp32 (fp8 per-channel). Floored strictly positive."""
    if grid == "fp4":
        return _snap_to_grid(s, "fp8").clamp_min(_E4M3_MIN_POS)
    return s.clamp_min(1e-12)


def _candidate_scales(amax: torch.Tensor, grid: str, n: int) -> torch.Tensor:
    """(n, *amax.shape) legal candidate scales sweeping the clipping level.
    Candidate 0 is L=grid-max == the one-shot amax/grid-max scale, so an
    argmin over these is never worse than the one-shot."""
    if grid == "fp4":
        levels = torch.linspace(NVFP4_GRID_MAX, 4.0, n, device=amax.device)
    else:
        levels = torch.linspace(FP8_ELEMENT_MAX, FP8_ELEMENT_MAX * 4.0 / 6.0,
                                n, device=amax.device)
    shape = (n,) + (1,) * amax.dim()
    return _snap_scale(amax.unsqueeze(0) / levels.reshape(shape), grid)


def _eval_candidate(w2d: torch.Tensor, wq: torch.Tensor | None,
                    s: torch.Tensor, grid: str, mode: str, cb,
                    wq_period: int | None = None):
    """Encode ``w2d`` at per-group scale ``s`` and score the WEIGHTED
    reconstruction error in the ORIGINAL weight domain (so the scale choice
    is judged on real error, not scaled-domain error). Returns
    (err_group (rows, ngroups), enc, grid_decode (rows, in))."""
    rows, in_f = w2d.shape
    pes = _per_element_scale(s, grid, in_f)              # (rows, in)
    pes_vec = pes.reshape(-1, VEC_DIM)                   # (nvec, 8)
    wvec = w2d.reshape(-1, VEC_DIM)
    x = wvec / pes_vec
    enc = _mode_encode(x, mode, cb, wq, wq_period)
    dec = _mode_decode(enc, mode, cb)                    # (nvec, 8) grid
    recon = dec * pes_vec                                # original domain
    err = (recon - wvec).pow(2)
    if wq is not None:
        err = err * wq
    err_group = _group_reduce(err.reshape(rows, in_f), grid)
    return err_group, enc, dec.reshape(rows, in_f)


# ---------------------------------------------------------------------------
# Moment-scored sweep (fast/balanced tiers). The candidate error decomposes
# as err(s) = C - 2s*B + s^2*A with per-(vector, entry) moments A, B that do
# NOT depend on the scale (the _grid_fields/_sweep_errs identity), so all
# candidates score from moments built ONCE per chunk instead of recomputing
# a full VQ distance matrix per candidate. C is constant per vector across
# candidates and cancels in every argmin, so it is dropped. The selection
# matches the exact sweep up to fp-rounding ties; refits and the final
# assignment stay on the exact direct-eval path.
# ---------------------------------------------------------------------------

# --- Row-periodic A ---------------------------------------------------------
# A = sum_j w_j t_j^2 depends only on the COLUMN weights, and the production
# imatrix is one per-input-column vector broadcast over rows. wq therefore
# repeats every ``in_f // VEC_DIM`` vectors, so A repeats too: the (m, K)
# moment is ``m / P`` identical copies of a (P, K) base. Keeping only the base
# removes half the moment build and half the bytes every scoring pass has to
# stream (the scan is bandwidth-bound on exactly these two planes), and the
# base is small enough to stay resident in L2. Values are untouched — each
# output element is the same length-``sub_dim`` dot product — so every emitted
# byte is unchanged.


def _periodic_split(A: torch.Tensor, B: torch.Tensor):
    """(P, R) when A is a row-periodic base for B, else (None, None)."""
    if A.dim() != 2:
        return None, None
    P = A.shape[0]
    m = B.shape[0]
    if P == m or P <= 0 or m % P:
        return None, None
    return P, m // P


def _score_min_eager(A: torch.Tensor, B: torch.Tensor,
                     s: torch.Tensor) -> torch.Tensor:
    """min over entries of s^2*A - 2s*B; A (m,K), (P,K) or (K,), B (m,K),
    s (m,1)."""
    P, R = _periodic_split(A, B)
    if P is None:
        return ((s * s) * A - (2.0 * s) * B).min(dim=-1).values
    K = B.shape[-1]
    sv = s.reshape(R, P, 1)
    d = (sv * sv) * A.reshape(1, P, K) - (2.0 * sv) * B.reshape(R, P, K)
    return d.min(dim=-1).values.reshape(-1)


def _score_argmin_eager(A: torch.Tensor, B: torch.Tensor, s: torch.Tensor):
    P, R = _periodic_split(A, B)
    if P is None:
        d = (s * s) * A - (2.0 * s) * B
        v, i = d.min(dim=-1)
        return v, i
    K = B.shape[-1]
    sv = s.reshape(R, P, 1)
    d = (sv * sv) * A.reshape(1, P, K) - (2.0 * sv) * B.reshape(R, P, K)
    v, i = d.min(dim=-1)
    return v.reshape(-1), i.reshape(-1)


@lru_cache(maxsize=None)
def _score_min_compiled():
    return torch.compile(_score_min_eager, dynamic=True)


@lru_cache(maxsize=None)
def _score_argmin_compiled():
    return torch.compile(_score_argmin_eager, dynamic=True)


def _encode_compile_on() -> bool:
    return os.environ.get(_ENCODE_COMPILE_ENV, "1").lower() not in (
        "0", "false", "no")


_ENCODE_RECOMPILE_LIMIT_RAISED = False


def _raise_encode_recompile_limit() -> None:
    """The moment-scoring kernels are compiled once (dynamic=True) but see a
    handful of distinct (K, S) specializations across grids/tiers/formats
    (fp8 K2048/S14, fp4 K256/S16, the S1 refit argmins, ...). The default
    dynamo recompile_limit=8 is exceeded across a mixed-format run, silently
    dropping the compiled path back to EAGER — which materializes the whole
    (m, K, S) intermediate and runs ~30x slower. Raise the limit so every
    specialization stays compiled (mirrors _make_rtn's defensive bump)."""
    global _ENCODE_RECOMPILE_LIMIT_RAISED
    if _ENCODE_RECOMPILE_LIMIT_RAISED:
        return
    try:
        torch._dynamo.config.recompile_limit = max(
            int(getattr(torch._dynamo.config, "recompile_limit", 8)), 256)
        torch._dynamo.config.accumulated_recompile_limit = max(
            int(getattr(torch._dynamo.config,
                        "accumulated_recompile_limit", 256)), 4096)
    except Exception:
        pass
    _ENCODE_RECOMPILE_LIMIT_RAISED = True


def _score_min(A, B, s):
    if _encode_compile_on():
        try:
            _raise_encode_recompile_limit()
            return _score_min_compiled()(A, B, s)
        except Exception:
            pass
    return _score_min_eager(A, B, s)


def _score_min_batched_eager(A: torch.Tensor, B: torch.Tensor,
                             s: torch.Tensor) -> torch.Tensor:
    """min over entries of s^2 A - 2s B for a BATCH of per-vector scales.

    ``A``/``B`` are (m, K); ``s`` is (m, S). Returns (m, S) — the per-vector
    min-over-K at each of the S scales. torch.compile fuses the elementwise
    scoring and the K-reduction so the (m, K) moments are read ONCE for all
    S scales (vs S separate reductions), the launch-bound/volume fix for the
    27B-scale sweep. ``A`` may also be a (P, K) row-periodic base."""
    P, R = _periodic_split(A, B)
    if P is None:
        s2 = (s * s).unsqueeze(1)                    # (m, 1, S)
        ts = (2.0 * s).unsqueeze(1)                  # (m, 1, S)
        d = s2 * A.unsqueeze(-1) - ts * B.unsqueeze(-1)   # (m, K, S)
        return d.min(dim=1).values                   # (m, S)
    K, S = B.shape[-1], s.shape[-1]
    sv = s.reshape(R, P, 1, S)
    d = (sv * sv) * A.reshape(1, P, K, 1) - (2.0 * sv) * B.reshape(R, P, K, 1)
    return d.min(dim=2).values.reshape(-1, S)


def _score_minargmin_batched_eager(A: torch.Tensor, B: torch.Tensor,
                                   s: torch.Tensor):
    """Batched min AND argmin over K for S per-vector scales. Returns
    (values (m, S), indices (m, S)). The argmin comes free with the min
    reduction, so scoring the scale grid ALSO yields the assignment at each
    candidate — folding the separate init-argmin pass into the scan."""
    P, R = _periodic_split(A, B)
    if P is None:
        s2 = (s * s).unsqueeze(1)
        ts = (2.0 * s).unsqueeze(1)
        d = s2 * A.unsqueeze(-1) - ts * B.unsqueeze(-1)   # (m, K, S)
        v, i = d.min(dim=1)
        return v, i
    K, S = B.shape[-1], s.shape[-1]
    sv = s.reshape(R, P, 1, S)
    d = (sv * sv) * A.reshape(1, P, K, 1) - (2.0 * sv) * B.reshape(R, P, K, 1)
    v, i = d.min(dim=2)
    return v.reshape(-1, S), i.reshape(-1, S)


@lru_cache(maxsize=None)
def _score_min_batched_compiled():
    return torch.compile(_score_min_batched_eager, dynamic=True)


@lru_cache(maxsize=None)
def _score_minargmin_batched_compiled():
    return torch.compile(_score_minargmin_batched_eager, dynamic=True)


def _score_minargmin_batched(A, B, s):
    if _encode_compile_on():
        try:
            _raise_encode_recompile_limit()
            return _score_minargmin_batched_compiled()(A, B, s)
        except Exception:
            pass
    vs, is_ = [], []
    for i in range(s.shape[-1]):
        v, ix = _score_argmin_eager(A, B, s[:, i:i + 1])
        vs.append(v)
        is_.append(ix)
    return torch.stack(vs, dim=-1), torch.stack(is_, dim=-1)


def _score_min_batched(A, B, s):
    if _encode_compile_on():
        try:
            _raise_encode_recompile_limit()
            return _score_min_batched_compiled()(A, B, s)
        except Exception:
            pass
    # Eager fallback: loop the S columns so the (m, K, S) intermediate is
    # never materialized (it would OOM at 27B scale — the fused compiled
    # path reads (m, K) once instead). Each column is one (m, K) reduction.
    return torch.stack(
        [_score_min_eager(A, B, s[:, i:i + 1]) for i in range(s.shape[-1])],
        dim=-1)


def _score_argmin(A, B, s):
    if _encode_compile_on():
        try:
            _raise_encode_recompile_limit()
            return _score_argmin_compiled()(A, B, s)
        except Exception:
            pass
    return _score_argmin_eager(A, B, s)


def _mode_streams(wvec: torch.Tensor, mode: str, cb, wq):
    """Per-sub scoring streams [(x, wq_sub, table)] in the ORIGINAL weight
    domain. signed scores on |w| (err (w - s*sign(w)*t)^2 == (|w| - s*t)^2
    for t >= 0); product splits sub-vectors (independent argmins)."""
    if mode == "full":
        return [(wvec, wq, cb)]
    if mode == "signed":
        return [(wvec.abs(), wq, cb)]
    n_sub = len(cb)
    sd = VEC_DIM // n_sub
    return [(wvec[:, i * sd:(i + 1) * sd],
             wq[:, i * sd:(i + 1) * sd] if wq is not None else None,
             cb[i]) for i in range(n_sub)]


def _stream_moments(x: torch.Tensor, ws: torch.Tensor | None,
                    table: torch.Tensor, ws_period: int | None = None):
    """A = sum_j w_j t_j^2 (per entry), B = sum_j w_j x_j t_j.

    When ``ws_period`` is given, ``ws`` is known to repeat every ``ws_period``
    rows (a per-column imatrix broadcast over rows), so A is built from the
    base block alone and returned as (ws_period, K); the scorers broadcast it.
    Each A entry is the same length-``sub_dim`` dot product either way."""
    t = table.to(x.device, torch.float32)
    tt = t.t().contiguous()
    B = ((ws * x) if ws is not None else x) @ tt
    if ws is not None:
        base = ws
        if (ws_period is not None and 0 < ws_period < ws.shape[0]
                and ws.shape[0] % ws_period == 0):
            base = ws[:ws_period]
        A = base @ (t * t).t().contiguous()
    else:
        A = (t * t).sum(dim=-1)
    return A, B


def _moment_rows_step(cb, vec_per_row: int) -> int:
    tables = cb if isinstance(cb, tuple) else (cb,)
    k_max = max(int(t.shape[0]) for t in tables)
    return max(1, (_SCORE_CHUNK_ELEMS // max(k_max, 1)) // max(vec_per_row, 1))


def _moment_err_groups(moms, s_g: torch.Tensor, grid: str, in_f: int,
                       vec_per_group: int) -> torch.Tensor:
    """Per-group error (minus the constant C term) at per-group scale
    ``s_g`` (rc, G), scored from cached moments."""
    rc, G = s_g.shape
    s_v = s_g.repeat_interleave(vec_per_group, dim=1).reshape(-1, 1)
    err_v = None
    for (A, B) in moms:
        v = _score_min(A, B, s_v)
        err_v = v if err_v is None else err_v + v
    return err_v.reshape(rc, G, vec_per_group).sum(dim=-1)


def _moment_err_groups_batched(moms, s_g: torch.Tensor, vec_per_group: int
                               ) -> torch.Tensor:
    """Per-group error for a BATCH of candidate scales in one fused pass.

    ``s_g`` is (rc, G, S). Returns (rc, G, S). The batched min reads each
    stream's (m, K) moments ONCE for all S candidates (vs S passes), so the
    whole scale sweep is a single volume pass instead of S — the fix for the
    launch/volume blowup on 27B-scale Linears."""
    rc, G, S = s_g.shape
    s_v = s_g.repeat_interleave(vec_per_group, dim=1).reshape(-1, S)
    err_v = None
    for (A, B) in moms:
        v = _score_min_batched(A, B, s_v)                 # (m, S)
        err_v = v if err_v is None else err_v + v
    return err_v.reshape(rc, G, vec_per_group, S).sum(dim=2)


def _chunk_moments(wvec, wqc, mode, cb, wq_period: int | None = None):
    """Build the per-stream (A, B) moments for one row-chunk ONCE, reusable
    across the whole per-chunk sweep + argmin + refits (eliminates the 4x
    moment rebuild the old scan/eval/refit split incurred)."""
    return [_stream_moments(x, ws, t, ws_period=wq_period)
            for (x, ws, t) in _mode_streams(wvec, mode, cb, wqc)]


def _scan_and_assign(moms, wvec, grid_s_c, mode, cb, grid, in_f,
                     vec_per_group):
    """Batched exhaustive scale-grid scan that ALSO returns the assignment at
    the chosen (per-group) scale — one fused (m, K) pass does both the min
    (scale selection) and the argmin (codeword assignment), no separate
    init-argmin pass. ``grid_s_c`` is (rc, G, S). Returns
    (best_s (rc, G), err (rc, G), enc, dec (rc, in_f))."""
    rc, G, S = grid_s_c.shape
    s_v = grid_s_c.repeat_interleave(vec_per_group, dim=1).reshape(-1, S)
    err = None
    per_stream_idx = []
    for (A, B) in moms:
        v, i = _score_minargmin_batched(A, B, s_v)        # (m, S), (m, S)
        err = v if err is None else err + v
        per_stream_idx.append(i)
    err_g = err.reshape(rc, G, vec_per_group, S).sum(dim=2)   # (rc, G, S)
    best_col = err_g.argmin(dim=-1)                        # first-min on ties
    best_s = torch.gather(grid_s_c, -1, best_col.unsqueeze(-1)).squeeze(-1)
    err_c = torch.gather(err_g, -1, best_col.unsqueeze(-1)).squeeze(-1)
    col_v = best_col.repeat_interleave(vec_per_group, dim=1).reshape(-1, 1)
    idxs = [torch.gather(i, -1, col_v).squeeze(-1) for i in per_stream_idx]
    if mode == "full":
        enc = {"idx": idxs[0]}
    elif mode == "signed":
        enc = {"idx": idxs[0], "signs": torch.where(wvec < 0, -1.0, 1.0)}
    else:
        enc = {"idx": torch.stack(idxs, dim=-1)}
    dec = _mode_decode(enc, mode, cb).reshape(rc, in_f)
    return best_s, err_c, enc, dec


def _argmin_from_moments(moms, wvec, s_g, mode, cb, grid, in_f,
                         vec_per_group):
    """Assignment + decode at per-group scale ``s_g`` (rc, G) from RESIDENT
    moments (no rebuild). Returns (err (rc, G), enc, dec (rc, in_f))."""
    rc, G = s_g.shape
    s_v = s_g.repeat_interleave(vec_per_group, dim=1).reshape(-1, 1)
    err_v = None
    idxs = []
    for (A, B) in moms:
        v, i = _score_argmin(A, B, s_v)
        err_v = v if err_v is None else err_v + v
        idxs.append(i)
    if mode == "full":
        enc = {"idx": idxs[0]}
    elif mode == "signed":
        enc = {"idx": idxs[0], "signs": torch.where(wvec < 0, -1.0, 1.0)}
    else:
        enc = {"idx": torch.stack(idxs, dim=-1)}
    err = err_v.reshape(rc, G, vec_per_group).sum(dim=-1)
    dec = _mode_decode(enc, mode, cb).reshape(rc, in_f)
    return err, enc, dec


def _calibrate_m2_used(w2d, wq2d, grid, mode, cb,
                       wq_period=None) -> float:
    """Usage-calibrated mean-square codeword coordinate from a pilot encode
    of the tensor's leading rows: the pilot runs the full 16-candidate
    moment-scored sweep, then m2_used = sum(q * dec^2) / sum(q) over the
    pilot's ACTUAL assignments. (The naive all-codeword table m2 lands the
    wrong basin — 2-3x worse error that refits cannot escape; per-tensor
    pilots also absorb the measured per-role m2 variation.)"""
    p1 = min(w2d.shape[0], _PILOT_ROWS)
    pilot = w2d[:p1]
    wp2d = wq2d[:p1] if wq2d is not None else None
    in_f = pilot.shape[1]
    vec_per_group = (FP4_GROUP if grid == "fp4" else in_f) // VEC_DIM
    vec_per_row = in_f // VEC_DIM
    amax = _group_amax(pilot, grid)
    cands = _candidate_scales(amax, grid, _SCALE_SWEEP_CANDIDATES)   # (S,rc,G)
    grid_s = cands.permute(1, 2, 0).contiguous()                    # (rc,G,S)
    rows_step = _moment_rows_step(cb, vec_per_row)
    num_acc = 0.0
    den_acc = 0.0
    for r0 in range(0, p1, rows_step):
        r1 = min(p1, r0 + rows_step)
        wvec = pilot[r0:r1].reshape(-1, VEC_DIM)
        wqc = (wp2d[r0:r1].reshape(-1, VEC_DIM)
               if wp2d is not None else None)
        moms = _chunk_moments(wvec, wqc, mode, cb, wq_period)
        errs = _moment_err_groups_batched(moms, grid_s[r0:r1], vec_per_group)
        best_col = errs.argmin(dim=-1)
        best_s = torch.gather(grid_s[r0:r1], -1,
                              best_col.unsqueeze(-1)).squeeze(-1)
        _, _, dec = _argmin_from_moments(
            moms, wvec, best_s, mode, cb, grid, in_f, vec_per_group)
        wcol = (wp2d[r0:r1] if wp2d is not None
                else torch.ones_like(dec))
        num_acc += float((wcol * dec * dec).sum())
        den_acc += float(wcol.sum())
    m2 = num_acc / max(den_acc, 1e-30)
    return max(m2, 1e-30)


def _analytic_s0(w2d, wq2d, grid, m2_used: float) -> torch.Tensor:
    """Per-group second-moment-match scale s0 (rows, G)."""
    wcol = wq2d if wq2d is not None else torch.ones_like(w2d)
    num = _group_reduce(wcol * w2d * w2d, grid)
    den = _group_reduce(wcol, grid) * m2_used
    return (num / den.clamp_min(1e-30)).sqrt()


def _tier_scale_grid(s0, w2d, grid, tier):
    """Exhaustive per-group candidate scale grid (rows, G, S): s0*ratio^i for
    i spanning the FULL reach the micro-sweep + greedy hill-climb could visit
    (+-(span+hill_iters)), plus the amax/grid-max guarantee.

    A single global argmin over this grid is a strict SUPERSET of every scale
    the sequential greedy hill explores, so it is provably no worse than the
    old greedy result (equal for the unimodal RD error that is the actual
    case) while collapsing the ~14 sequential scored passes into ONE batched
    pass. Snapped legal; the guarantee keeps the never-worse-than-one-shot
    contract."""
    ratio = _TIER_MICRO_RATIO[tier]
    reach = _TIER_MICRO_SPAN[tier] + _TIER_HILL_ITERS[tier]
    mults = [ratio ** i for i in range(-reach, reach + 1)]
    cols = [_snap_scale(s0 * m, grid) for m in mults]
    amax = _group_amax(w2d, grid)
    cols.append(_candidate_scales(amax, grid, 1)[0])
    return torch.stack(cols, dim=-1)                       # (rows, G, S)


def _sweep_encode_moment(w2d: torch.Tensor, grid: str, mode: str, cb,
                         wq: torch.Tensor | None, tier: str,
                         wq_period: int | None = None):
    """fast/balanced v1 sweep. Unified per-chunk pipeline: build the (m, K)
    moments ONCE per chunk, batch-score the exhaustive scale grid in a single
    fused pass, then argmin + WLS refits all off the RESIDENT moments (no
    rebuild). The scale grid is a superset of the old greedy hill's reach, so
    encode choices are no worse (equal on unimodal groups)."""
    rows, in_f = w2d.shape
    wq2d = wq.reshape(rows, in_f) if wq is not None else None
    m2 = _calibrate_m2_used(w2d, wq2d, grid, mode, cb, wq_period)
    s0 = _analytic_s0(w2d, wq2d, grid, m2)
    grid_s = _tier_scale_grid(s0, w2d, grid, tier)         # (rows, G, S)
    vec_per_group = (FP4_GROUP if grid == "fp4" else in_f) // VEC_DIM
    vec_per_row = in_f // VEC_DIM
    refits = _TIER_REFITS[tier]
    best_s = torch.empty(rows, s0.shape[1], device=w2d.device)
    enc_parts: list[dict] = []
    rows_step = _moment_rows_step(cb, vec_per_row)
    for r0 in range(0, rows, rows_step):
        r1 = min(rows, r0 + rows_step)
        wvec = w2d[r0:r1].reshape(-1, VEC_DIM)
        wqc = (wq2d[r0:r1].reshape(-1, VEC_DIM)
               if wq2d is not None else None)
        moms = _chunk_moments(wvec, wqc, mode, cb, wq_period)
        # Exhaustive scale grid + assignment in ONE fused batched pass.
        s_c, err, enc, dec = _scan_and_assign(
            moms, wvec, grid_s[r0:r1], mode, cb, grid, in_f, vec_per_group)
        w_chunk = w2d[r0:r1]
        wcol = (wq2d[r0:r1] if wq2d is not None else torch.ones_like(w_chunk))
        for _ in range(refits):
            num = _group_reduce(wcol * dec * w_chunk, grid)
            den = _group_reduce(wcol * dec * dec, grid)
            s_star = _snap_scale(
                torch.where(den > 0, num / den.clamp_min(1e-30), s_c), grid)
            err_star, enc_star, dec_star = _argmin_from_moments(
                moms, wvec, s_star, mode, cb, grid, in_f, vec_per_group)
            better = err_star < err
            s_c = torch.where(better, s_star, s_c)
            err = torch.where(better, err_star, err)
            bvec = better.repeat_interleave(vec_per_group, dim=1).reshape(-1)
            for key in enc:
                cur, star = enc[key], enc_star[key]
                mask = bvec if cur.dim() == 1 else bvec.reshape(
                    (-1,) + (1,) * (cur.dim() - 1)).expand_as(cur)
                enc[key] = torch.where(mask, star, cur)
            belem = better.repeat_interleave(
                (FP4_GROUP if grid == "fp4" else in_f), dim=1)
            dec = torch.where(belem, dec_star, dec)
        best_s[r0:r1] = s_c
        enc_parts.append(enc)
    enc = {key: torch.cat([e[key] for e in enc_parts], dim=0)
           for key in enc_parts[0]}
    return best_s, enc


def _sweep_encode(w2d: torch.Tensor, grid: str, mode: str, cb,
                  wq: torch.Tensor | None, wq_period: int | None = None):
    """Joint scale sweep + WLS-refit fixed point (mirrors _grid_fields). Picks
    the per-group scale minimizing weighted real error over the E4M3-legal
    candidate grid, then refines with continuous WLS refits accepted per group
    only when strictly better. Returns (best_scales (rows, ng), enc)."""
    rows, in_f = w2d.shape
    amax = _group_amax(w2d, grid)                        # (rows, ng)
    cands = _candidate_scales(amax, grid, _SCALE_SWEEP_CANDIDATES)
    best_err, _, _ = _eval_candidate(w2d, wq, cands[0], grid, mode, cb,
                                     wq_period)
    best_s = cands[0]
    for si in range(1, cands.shape[0]):
        err, _, _ = _eval_candidate(w2d, wq, cands[si], grid, mode, cb,
                                    wq_period)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, cands[si], best_s)
    # WLS refit: optimal continuous scale s* = sum(w g v) / sum(w g^2) per
    # group at the current (fixed) assignment, snapped legal, accepted per
    # group only when it strictly lowers real error.
    for _ in range(_SCALE_SWEEP_REFIT_ITERS):
        err_cur, _, g = _eval_candidate(w2d, wq, best_s, grid, mode, cb,
                                        wq_period)
        wcol = wq.reshape(rows, in_f) if wq is not None else torch.ones_like(g)
        num = _group_reduce(wcol * g * w2d, grid)
        den = _group_reduce(wcol * g * g, grid)
        s_star = _snap_scale(torch.where(den > 0, num / den.clamp_min(1e-30),
                                         best_s), grid)
        err_star, _, _ = _eval_candidate(w2d, wq, s_star, grid, mode, cb,
                                         wq_period)
        better = err_star < err_cur
        best_s = torch.where(better, s_star, best_s)
    _, enc, _ = _eval_candidate(w2d, wq, best_s, grid, mode, cb, wq_period)
    return best_s, enc


# ---------------------------------------------------------------------------
# Two-tier scale coding (layout v2): compose/legality tables + encoder.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _two_tier_tables(device: str):
    """Return (table (16,), compose (256, 16) fp32, legal (256, 16) bool).

    ``compose[E, c] = T[c] * 2^(E - 127)``; a pair is legal iff the composed
    value round-trips ``float8_e4m3fn`` bit-exactly and lies in (0, 448]
    (spec §1.2) — so every emitted scale is exact E4M3 by construction."""
    dev = torch.device(device)
    table = torch.tensor(TWO_TIER_SUB_TABLE, dtype=torch.float32, device=dev)
    snapped = table.to(_E4M3).to(torch.float32)
    if not torch.equal(snapped, table):
        raise AssertionError("TWO_TIER_SUB_TABLE entries must be e4m3-exact")
    exps = torch.arange(256, dtype=torch.float64, device=dev)
    compose64 = table.to(torch.float64) * torch.pow(
        2.0, exps - TWO_TIER_SUPER_BIAS).unsqueeze(-1)          # (256, 16)
    compose = compose64.to(torch.float32)
    finite = torch.isfinite(compose)
    rt = torch.where(finite, compose, torch.zeros_like(compose)).to(
        _E4M3).to(torch.float32)
    legal = (finite & (compose > 0) & (compose <= FP8_ELEMENT_MAX)
             & (rt == compose)
             & (compose.to(torch.float64) == compose64))
    return table, compose, legal


@lru_cache(maxsize=None)
def _two_tier_legal_e_range(device: str) -> tuple[int, int]:
    """(e_min, e_max): the first/last super-exponent with ANY legal sub entry.

    A pure function of the (device-independent) legality table, so it is
    resolved once per device instead of costing two ``nonzero`` launches plus
    two device->host syncs on every single encoded tensor. Same integers, so
    the window — and therefore every emitted byte — is unchanged."""
    _, _, legal = _two_tier_tables(device)
    any_legal = legal.any(dim=-1)
    nz = torch.nonzero(any_legal)
    return int(nz[0]), int(nz[-1])


def _two_tier_window(amax: torch.Tensor, ideal: torch.Tensor | None = None,
                     pad: int = _TWO_TIER_WINDOW_PAD):
    """Per-superblock E window (E_lo (rows, n_sb) int64, W int) from the ideal
    group scales (spec §1.4): E_lo so the table top reaches min_ideal, E_hi so
    the table bottom reaches 1.5*max_ideal, padded ``pad`` octaves each side.
    ``ideal`` defaults to amax/6 (the exact path); the analytic tiers pass the
    s0-calibrated ideals so the window centers where winners actually live."""
    rows, G = amax.shape
    n_sb = G * FP4_GROUP // SUPERBLOCK
    if ideal is None:
        ideal = amax / NVFP4_GRID_MAX
    ideal = ideal.clamp_min(_E4M3_MIN_POS)
    ideal_sb = ideal.reshape(rows, n_sb, SUPERBLOCK // FP4_GROUP)
    min_i = ideal_sb.amin(dim=-1)
    max_i = ideal_sb.amax(dim=-1)
    t_max = float(TWO_TIER_SUB_TABLE[-1])
    e_lo = (torch.ceil(torch.log2(min_i / t_max)) + TWO_TIER_SUPER_BIAS
            - pad)
    e_hi = (torch.floor(torch.log2(max_i * 1.5)) + TWO_TIER_SUPER_BIAS
            + pad)
    e_min, e_max = _two_tier_legal_e_range(str(amax.device))
    e_lo = e_lo.clamp(e_min, e_max).to(torch.int64)
    e_hi = e_hi.clamp(e_min, e_max).to(torch.int64)
    e_hi = torch.maximum(e_hi, e_lo)
    W = min(int((e_hi - e_lo).max()) + 1, _TWO_TIER_MAX_WINDOW)
    # When the cap truncates an extreme-spread superblock, keep the TOP of
    # the window: groups below the reachable floor snap UP with error bounded
    # by their (small) magnitude (spec §1.2 zero/degenerate rules), while
    # losing the top would cost ~amax^2 on the largest group.
    e_lo = torch.maximum(e_lo, e_hi - (W - 1))
    return e_lo, e_hi, W


def _sb_to_groups(t_sb: torch.Tensor) -> torch.Tensor:
    """(rows, n_sb) -> (rows, G): broadcast a per-superblock value to its 16
    group-16 slots."""
    rows, n_sb = t_sb.shape
    reps = SUPERBLOCK // FP4_GROUP
    return t_sb.unsqueeze(-1).expand(rows, n_sb, reps).reshape(rows, -1)


def _two_tier_eval_entry(w2d, wq, comp_sb, legal_sb, mode, cb,
                         wq_period=None):
    """Score one sub-table entry at per-superblock composed scale ``comp_sb``
    ((rows, n_sb), maybe illegal): weighted original-domain error per group,
    +inf where the (E, c) pair is illegal."""
    safe = torch.where(legal_sb, comp_sb, torch.ones_like(comp_sb))
    err_g, _, _ = _eval_candidate(w2d, wq, _sb_to_groups(safe), "fp4", mode,
                                  cb, wq_period)
    inf = torch.tensor(float("inf"), device=err_g.device)
    return torch.where(_sb_to_groups(legal_sb.to(torch.bool)), err_g, inf)


def _sweep_encode_two_tier(w2d: torch.Tensor, mode: str, cb,
                           wq: torch.Tensor | None,
                           wq_period: int | None = None):
    """Layout-v2 encoder (spec §1.4): the sweep machinery with the candidate
    set restricted to the two-tier reachable set.

    1. sweep E per superblock over the ideal-scale window; per (E, group) the
       best legal sub-table entry via the weighted original-domain eval;
    2. pick E per superblock by total weighted error;
    3. per-group entry argmin at the frozen E (strict <, so an all-zero group
       deterministically takes the FIRST legal entry — spec zero rule);
    4. WLS refits snapped to the frozen-E reachable set, accepted per group
       only when strictly better.

    Returns (scales (rows, G) composed e4m3-exact, enc, super_E (rows, n_sb),
    sub_codes (rows, G))."""
    rows, in_f = w2d.shape
    n_sb = in_f // SUPERBLOCK
    dev = str(w2d.device)
    _, compose, legal = _two_tier_tables(dev)
    amax = _group_amax(w2d, "fp4")                              # (rows, G)
    e_lo, e_hi, W = _two_tier_window(amax)                      # (rows, n_sb)

    inf = torch.tensor(float("inf"), device=w2d.device)
    best_tot = torch.full((rows, n_sb), float("inf"), device=w2d.device)
    best_e = e_lo.clone()
    for i in range(W):
        E = torch.minimum(e_lo + i, e_hi)
        valid_i = (e_lo + i) <= e_hi
        err_best_g = torch.full_like(amax, float("inf"))
        for c in range(len(TWO_TIER_SUB_TABLE)):
            err_g = _two_tier_eval_entry(
                w2d, wq, compose[E, c], legal[E, c], mode, cb, wq_period)
            err_best_g = torch.minimum(err_best_g, err_g)
        tot = err_best_g.reshape(rows, n_sb, -1).sum(-1)
        tot = torch.where(valid_i, tot, inf)
        better = tot < best_tot
        best_tot = torch.where(better, tot, best_tot)
        best_e = torch.where(better, E, best_e)

    # Per-group entry selection at the frozen per-superblock E. Strict < keeps
    # the FIRST legal entry on ties (all-zero groups -> deterministic bytes).
    best_err_g = torch.full_like(amax, float("inf"))
    best_s = torch.full_like(amax, _E4M3_MIN_POS)
    best_c = torch.zeros(rows, amax.shape[1], dtype=torch.int64,
                         device=w2d.device)
    for c in range(len(TWO_TIER_SUB_TABLE)):
        err_g = _two_tier_eval_entry(
            w2d, wq, compose[best_e, c], legal[best_e, c], mode, cb, wq_period)
        better = err_g < best_err_g
        best_err_g = torch.where(better, err_g, best_err_g)
        best_s = torch.where(better, _sb_to_groups(compose[best_e, c]), best_s)
        best_c = torch.where(better, torch.full_like(best_c, c), best_c)

    # WLS refit on the frozen-E reachable set (spec §1.4 step 3).
    reach = compose[best_e]                                     # (rows,n_sb,16)
    reach_legal = legal[best_e]
    reps = SUPERBLOCK // FP4_GROUP
    reach_g = reach.unsqueeze(2).expand(rows, n_sb, reps, -1)
    legal_g = reach_legal.unsqueeze(2).expand(rows, n_sb, reps, -1)
    for _ in range(_SCALE_SWEEP_REFIT_ITERS):
        err_cur, _, g = _eval_candidate(w2d, wq, best_s, "fp4", mode, cb,
                                        wq_period)
        wcol = wq.reshape(rows, in_f) if wq is not None else torch.ones_like(g)
        num = _group_reduce(wcol * g * w2d, "fp4")
        den = _group_reduce(wcol * g * g, "fp4")
        s_star = torch.where(den > 0, num / den.clamp_min(1e-30), best_s)
        dist = (s_star.reshape(rows, n_sb, reps, 1) - reach_g).abs()
        dist = torch.where(legal_g, dist, inf)
        c_star = dist.argmin(dim=-1)                            # (rows,n_sb,16)
        s_snap = torch.gather(reach_g, -1, c_star.unsqueeze(-1)).squeeze(-1)
        s_snap = s_snap.reshape(rows, -1)
        err_star, _, _ = _eval_candidate(w2d, wq, s_snap, "fp4", mode, cb,
                                         wq_period)
        better = err_star < err_cur
        best_s = torch.where(better, s_snap, best_s)
        best_c = torch.where(better, c_star.reshape(rows, -1), best_c)

    _, enc, _ = _eval_candidate(w2d, wq, best_s, "fp4", mode, cb, wq_period)
    return best_s, enc, best_e, best_c


def _sweep_encode_two_tier_moment(w2d: torch.Tensor, mode: str, cb,
                                  wq: torch.Tensor | None, tier: str,
                                  wq_period: int | None = None):
    """fast/balanced layout-v2 encoder: the windowed-E x entry search scored
    from cached moments (the W*16 combos reuse ONE moment build per chunk),
    with the E window centered on the s0-calibrated ideals (analytic init;
    pad 1 octave balanced / 0 fast), exact refits snapped to the frozen-E
    reachable set. Selection order and tie rules mirror the exact path
    (strict <, first-legal wins).

    NOTE: layout-v2 is the production writer default. Its W*16
    windowed-entry search does not batch cleanly (the launch-bound fix targets
    the v1/fp8-shaped search), so this keeps the original bit-preserving
    structure. Legacy-v1 remains an explicit read/reproduction mode."""
    rows, in_f = w2d.shape
    n_sb = in_f // SUPERBLOCK
    dev = str(w2d.device)
    _, compose, legal = _two_tier_tables(dev)
    amax = _group_amax(w2d, "fp4")
    wq2d_w = wq.reshape(rows, in_f) if wq is not None else None
    m2 = _calibrate_m2_used(w2d, wq2d_w, "fp4", mode, cb, wq_period)
    wcol = wq2d_w if wq2d_w is not None else torch.ones_like(w2d)
    s0 = (_group_reduce(wcol * w2d * w2d, "fp4")
          / (_group_reduce(wcol, "fp4") * m2).clamp_min(1e-30)).sqrt()
    refits = _TIER_REFITS[tier]
    e_lo, e_hi, W = _two_tier_window(
        amax, ideal=s0, pad=1 if tier == "balanced" else 0)
    G = amax.shape[1]
    gps = SUPERBLOCK // FP4_GROUP
    vec_per_row = in_f // VEC_DIM
    vec_per_group = FP4_GROUP // VEC_DIM
    n_ent = len(TWO_TIER_SUB_TABLE)
    inf = torch.tensor(float("inf"), device=w2d.device)

    best_e = e_lo.clone()
    best_s = torch.full((rows, G), _E4M3_MIN_POS, device=w2d.device)
    best_c = torch.zeros(rows, G, dtype=torch.int64, device=w2d.device)
    rows_step = _moment_rows_step(cb, vec_per_row)
    wq2d = wq.reshape(rows, in_f) if wq is not None else None
    for r0 in range(0, rows, rows_step):
        r1 = min(rows, r0 + rows_step)
        rc = r1 - r0
        wvec = w2d[r0:r1].reshape(-1, VEC_DIM)
        wqc = (wq2d[r0:r1].reshape(-1, VEC_DIM)
               if wq2d is not None else None)
        moms = _chunk_moments(wvec, wqc, mode, cb, wq_period)

        def entry_err_all(E):
            """All n_ent entries in ONE batched moment pass. Returns
            (err (rc, G, n_ent) inf-masked, s_g (rc, G, n_ent)). Values are
            the same per-(element, entry) arithmetic as the scalar path; the
            16-entry python loop was 2ms-kernel launch-bound (33.6s of a
            46.8s E=24 encode, 2026-07-19)."""
            comp = compose[E]                              # (rc, n_sb, n_ent)
            leg = legal[E]
            s_sb = torch.where(leg, comp, torch.ones_like(comp))
            s_g = (s_sb.unsqueeze(2).expand(rc, n_sb, gps, n_ent)
                   .reshape(rc, G, n_ent))
            err_g = _moment_err_groups_batched(moms, s_g, vec_per_group)
            leg_g = (leg.unsqueeze(2).expand(rc, n_sb, gps, n_ent)
                     .reshape(rc, G, n_ent))
            return torch.where(leg_g, err_g, inf), s_g

        # Phase 1 — E per superblock by total error (running strict-min in
        # window order, matching the exact path; min over entries is
        # order-free so the batched min is value-identical).
        lo, hi = e_lo[r0:r1], e_hi[r0:r1]
        best_tot = torch.full((rc, n_sb), float("inf"), device=w2d.device)
        for i in range(W):
            E = torch.minimum(lo + i, hi)
            err_best_g = entry_err_all(E)[0].min(dim=-1).values
            tot = err_best_g.reshape(rc, n_sb, gps).sum(-1)
            tot = torch.where((lo + i) <= hi, tot, inf)
            better = tot < best_tot
            best_tot = torch.where(better, tot, best_tot)
            best_e[r0:r1] = torch.where(better, E, best_e[r0:r1])

        # Phase 2 — per-group entry at the frozen E. torch.min's documented
        # first-occurrence tie rule IS the sequential strict-<, first-legal
        # rule (candidates scanned in c order from an inf init).
        Eb = best_e[r0:r1]
        err_all, s_all = entry_err_all(Eb)
        vals, idx = err_all.min(dim=-1)
        finite = vals < inf
        best_s[r0:r1] = torch.where(
            finite, torch.gather(s_all, -1, idx.unsqueeze(-1)).squeeze(-1),
            best_s[r0:r1])
        best_c[r0:r1] = torch.where(finite, idx, best_c[r0:r1])

    # Phase 3 — exact WLS refits on the frozen-E reachable set.
    #
    # State (err, enc, grid-decode) is CARRIED across refits instead of being
    # re-derived by re-evaluating at ``best_s`` each iteration. `_eval_candidate`
    # is group-local in every step — the per-element scale comes from the
    # element's own group, an 8-wide codeword never straddles a group-16
    # boundary, and the error reduction is per group — so evaluating at the
    # per-group-merged scale is EXACTLY the per-group selection between the two
    # evaluations already in hand. That drops the exact evaluations from
    # 2*refits+1 (5 at balanced) to refits+1 (3) with byte-identical output.
    reach = compose[best_e]
    reach_legal = legal[best_e]
    reps = SUPERBLOCK // FP4_GROUP
    reach_g = reach.unsqueeze(2).expand(rows, n_sb, reps, -1)
    legal_g = reach_legal.unsqueeze(2).expand(rows, n_sb, reps, -1)
    err_cur, enc, g = _eval_candidate(w2d, wq, best_s, "fp4", mode, cb,
                                      wq_period)
    for _ in range(int(refits)):
        wcol = wq.reshape(rows, in_f) if wq is not None else torch.ones_like(g)
        num = _group_reduce(wcol * g * w2d, "fp4")
        den = _group_reduce(wcol * g * g, "fp4")
        s_star = torch.where(den > 0, num / den.clamp_min(1e-30), best_s)
        dist = (s_star.reshape(rows, n_sb, reps, 1) - reach_g).abs()
        dist = torch.where(legal_g, dist, inf)
        c_star = dist.argmin(dim=-1)
        s_snap = torch.gather(reach_g, -1, c_star.unsqueeze(-1)).squeeze(-1)
        s_snap = s_snap.reshape(rows, -1)
        err_star, enc_star, g_star = _eval_candidate(
            w2d, wq, s_snap, "fp4", mode, cb, wq_period)
        better = err_star < err_cur
        best_s = torch.where(better, s_snap, best_s)
        best_c = torch.where(better, c_star.reshape(rows, -1), best_c)
        err_cur = torch.where(better, err_star, err_cur)
        g = torch.where(better.repeat_interleave(FP4_GROUP, dim=1), g_star, g)
        bvec = better.repeat_interleave(
            FP4_GROUP // VEC_DIM, dim=1).reshape(-1)
        for key in enc:
            cur, star = enc[key], enc_star[key]
            mask = bvec if cur.dim() == 1 else bvec.reshape(
                (-1,) + (1,) * (cur.dim() - 1)).expand_as(cur)
            enc[key] = torch.where(mask, star, cur)
    return best_s, enc, best_e, best_c


def _fields_block(w2d: torch.Tensor, k: int, grid: str, mode: str,
                  cb, cw2d: torch.Tensor | None, scale_sweep: bool,
                  scale_coding: str = SCALE_CODING_V1,
                  encode_tier: str = "max",
                  cw_row_broadcast: bool = False,
                  warm_scale_state: dict[str, torch.Tensor] | None = None,
                  ) -> dict:
    rows, in_f = w2d.shape
    nvec_per_row = in_f // VEC_DIM
    wq = _col_weight_vectors(cw2d) if cw2d is not None else None
    # A per-input-column imatrix repeats every ``nvec_per_row`` weight vectors,
    # so the scale-independent moments built from it do too. Only claim the
    # period when the caller proved every row of cw2d came from one vector.
    wq_period = (nvec_per_row
                 if (wq is not None and cw_row_broadcast and rows > 1)
                 else None)
    if warm_scale_state is not None:
        # Warm state is only the sweep argmin.  Assignment still runs through
        # the ordinary weighted VQ evaluator, and assembly still runs through
        # the ordinary packer; no codeword/index bytes are trusted or reused.
        scales = warm_scale_state["scales"].to(
            device=w2d.device, dtype=torch.float32
        )
        _, enc, _ = _eval_candidate(
            w2d, wq, scales, grid, mode, cb, wq_period
        )
        if scale_coding == SCALE_CODING_TWO_TIER:
            super_e = warm_scale_state["scale_super"].to(w2d.device)
            sub_c = warm_scale_state["scale_sub"].to(w2d.device)
    elif scale_coding == SCALE_CODING_TWO_TIER:
        if grid != "fp4":
            raise ValueError("two-tier scale coding is fp4-family only "
                             "(fp8 has no per-superblock scale plane)")
        if not scale_sweep:
            raise ValueError("two-tier scale coding IS the sweep encoder "
                             "(spec §1.4); scale_sweep=False is undefined")
        if encode_tier == "max":
            scales, enc, super_e, sub_c = _sweep_encode_two_tier(
                w2d, mode, cb, wq, wq_period)
        else:
            scales, enc, super_e, sub_c = _sweep_encode_two_tier_moment(
                w2d, mode, cb, wq, encode_tier, wq_period)
    elif scale_sweep:
        if encode_tier == "max":
            scales, enc = _sweep_encode(w2d, grid, mode, cb, wq, wq_period)
        else:
            scales, enc = _sweep_encode_moment(
                w2d, grid, mode, cb, wq, encode_tier, wq_period)
    else:
        vectors, scales, _ = _scale_and_vectorize(w2d, grid)
        enc = _mode_encode(vectors, mode, cb, wq, wq_period)
    out = _enc_to_fields(enc, mode, cb, rows, in_f, nvec_per_row)
    out["scales"] = scales
    if scale_coding == SCALE_CODING_TWO_TIER:
        out["scale_super"] = super_e.to(torch.uint8)
        out["scale_sub"] = sub_c
    return out


def nvfp4_cb_fields(w: torch.Tensor, k: int, *, grid: str = "fp4",
                    mode: str = "product",
                    col_weights: torch.Tensor | None = None,
                    codebook: torch.Tensor | tuple | None = None,
                    scale_sweep: bool = True,
                    scale_coding: str = SCALE_CODING_V1,
                    encode_tier: str | None = None,
                    warm_scale_state: dict[str, torch.Tensor] | None = None,
                    ) -> dict:
    """Quantize ``w`` (2-D or 3-D stacked experts) into VQ fields.

    ``scale_sweep`` (default True) jointly optimizes the per-group scale over
    the E4M3-legal candidate grid (IQ-rendering parity); set False for the
    one-shot amax/grid-max scale (A/B and the pre-c3f8c6d rendering).

    ``encode_tier``: fast / balanced / max speed-accuracy tier (None reads
    ``PRISMAQUANT_CB_ENCODE_TIER``, default balanced). max reproduces the
    original sweep bit-identically; see docs/lanes/nvfp4-cb/encode_tiers.md.

    ``scale_coding``: ``"v1"`` (default; bare e4m3 plane) or ``"two_tier"``
    (layout v2, fp4 only: per-superblock E8M0 super + 4-bit sub codes; the
    stored plane is still the composed E4M3-exact per-group scale). The low-
    level codec defaults to v1 for read compatibility; production callers bind
    an explicit serialization context and select v2.

    Returns at least {"indices", "scales"}; the resolved codebook is echoed
    back under "codebook" so reconstruct and the packer share one table.
    """
    in_f = int(w.shape[-1])
    if in_f % SUPERBLOCK != 0:
        raise ValueError(
            f"in_features={in_f} must be a multiple of {SUPERBLOCK}")
    if scale_coding not in (SCALE_CODING_V1, SCALE_CODING_TWO_TIER):
        raise ValueError(f"unknown scale_coding {scale_coding!r}")
    tier = _resolve_encode_tier(encode_tier)
    orig_shape = tuple(w.shape)
    w2d = w.reshape(-1, in_f)
    rows = w2d.shape[0]
    cb = _resolve_codebook(k, grid, mode, codebook, w2d.device)

    warm_state = None
    if warm_scale_state is not None:
        # File-level validation lives in cb_warm_state.  These shape checks
        # are the codec boundary's final defence for direct library callers.
        groups = in_f // FP4_GROUP if grid == "fp4" else 1
        expected = (rows, groups)
        scales = torch.as_tensor(warm_scale_state.get("scales"))
        if tuple(scales.shape) != expected:
            raise ValueError(
                f"warm scales shape {tuple(scales.shape)} != {expected}"
            )
        warm_state = {"scales": scales}
        if scale_coding == SCALE_CODING_TWO_TIER:
            n_sb = in_f // SUPERBLOCK
            super_e = torch.as_tensor(warm_scale_state.get("scale_super"))
            sub_c = torch.as_tensor(warm_scale_state.get("scale_sub"))
            if tuple(super_e.shape) != (rows, n_sb):
                raise ValueError("warm two-tier super-scale shape differs")
            if tuple(sub_c.shape) != expected:
                raise ValueError("warm two-tier sub-scale shape differs")
            warm_state.update(scale_super=super_e, scale_sub=sub_c)

    # col_weights stays a broadcast VIEW; blocks materialize only their rows
    # (a full-shape fp32 copy is another ~10GB on a Hy3 expert stack).
    cw_view = None
    cw_row_broadcast = False
    if col_weights is not None:
        cw_view = torch.broadcast_to(
            col_weights.to(w2d.device, torch.float32), orig_shape)
        # Every leading (row) dim stride 0 <=> one per-input-column vector
        # replicated across rows, which is what the production imatrix is.
        # Only then do the col-weight moments repeat per row (see
        # _stream_moments/_vq_assign ``wq_period``).
        cw_row_broadcast = all(
            cw_view.stride(d) == 0 for d in range(cw_view.dim() - 1))

    def _cw_rows(a: int, b: int) -> torch.Tensor | None:
        if cw_view is None:
            return None
        idx = torch.unravel_index(
            torch.arange(a, b, device=w2d.device), orig_shape[:-1])
        return cw_view[idx]                                  # (b-a, in_f)

    row_step = max(1, _SLICE_MAX_ELEMS // max(in_f, 1))

    def _warm_rows(a: int, b: int):
        if warm_state is None:
            return None
        return {key: value[a:b] for key, value in warm_state.items()}

    if rows <= row_step:
        out = _fields_block(w2d, k, grid, mode, cb, _cw_rows(0, rows),
                            scale_sweep, scale_coding, tier,
                            cw_row_broadcast, _warm_rows(0, rows))
    else:
        parts = []
        for a in range(0, rows, row_step):
            b = min(rows, a + row_step)
            parts.append(
                _fields_block(w2d[a:b], k, grid, mode, cb, _cw_rows(a, b),
                              scale_sweep, scale_coding, tier,
                              cw_row_broadcast, _warm_rows(a, b)))
        out = {key: torch.cat([p[key] for p in parts], dim=0)
               for key in parts[0]}
    out["shape"] = orig_shape
    out["codebook"] = cb
    if scale_coding == SCALE_CODING_TWO_TIER:
        out["scale_coding"] = SCALE_CODING_TWO_TIER
    return out


LDLQ_BLOCK_SIZE = 64
LDLQ_DAMPING_FRACTION = 0.01
_LDLQ_FACTOR_CACHE_MAX = 512
_LDLQ_FACTOR_CACHE: OrderedDict[tuple, tuple[weakref.ReferenceType, torch.Tensor]] = (
    OrderedDict()
)
_LDLQ_FACTOR_CACHE_LOCK = threading.Lock()


def _ldlq_inverse_factor_cached(
    activation_rows: torch.Tensor,
    *,
    device: torch.device,
    damping_fraction: float,
) -> torch.Tensor:
    """Reuse the exact format-independent factor across adjacent CB rungs."""
    source = torch.as_tensor(activation_rows)
    key = (
        id(source),
        source.data_ptr(),
        source.storage_offset(),
        tuple(source.shape),
        tuple(source.stride()),
        source.device,
        source.dtype,
        device,
        float(damping_fraction),
    )
    with _LDLQ_FACTOR_CACHE_LOCK:
        cached = _LDLQ_FACTOR_CACHE.get(key)
        if cached is not None and cached[0]() is source:
            _LDLQ_FACTOR_CACHE.move_to_end(key)
            return cached[1]
        if cached is not None:
            del _LDLQ_FACTOR_CACHE[key]

    from .rotation_ldlq_pilot import inverse_hessian_cholesky

    x = source.to(device=device, dtype=torch.float32)
    factor = inverse_hessian_cholesky(
        x.T @ x,
        damping_fraction=float(damping_fraction),
    )[0]
    with _LDLQ_FACTOR_CACHE_LOCK:
        _LDLQ_FACTOR_CACHE[key] = (weakref.ref(source), factor)
        _LDLQ_FACTOR_CACHE.move_to_end(key)
        if len(_LDLQ_FACTOR_CACHE) > _LDLQ_FACTOR_CACHE_MAX:
            dead = [
                old_key for old_key, (reference, _value)
                in _LDLQ_FACTOR_CACHE.items() if reference() is None
            ]
            for old_key in dead:
                del _LDLQ_FACTOR_CACHE[old_key]
        while len(_LDLQ_FACTOR_CACHE) > _LDLQ_FACTOR_CACHE_MAX:
            _LDLQ_FACTOR_CACHE.popitem(last=False)
    return factor


def _ldlq_reassign_fields_2d(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    activation_rows: torch.Tensor,
    *,
    grid: str,
    mode: str,
    block_size: int,
    damping_fraction: float,
) -> dict:
    """Replace only fixed-codebook assignments using block Hessian feedback."""
    from .rotation_ldlq_pilot import block_error_feedback

    if weight.ndim != 2:
        raise ValueError(f"LDLQ weight must be 2-D, got {tuple(weight.shape)}")
    rows, columns = map(int, weight.shape)
    x = torch.as_tensor(activation_rows)
    if x.ndim != 2 or int(x.shape[1]) != columns:
        raise ValueError(
            "LDLQ activation rows must have shape (rows, in_features), got "
            f"{tuple(x.shape)} for weight {tuple(weight.shape)}"
        )
    if int(x.shape[0]) == 0:
        raise ValueError("LDLQ activation rows must be non-empty")
    if columns % int(block_size) or int(block_size) % FP4_GROUP:
        raise ValueError(
            f"LDLQ block_size={block_size} must divide in_features={columns} "
            f"and preserve group-{FP4_GROUP} scales"
        )

    upper = _ldlq_inverse_factor_cached(
        x,
        device=weight.device,
        damping_fraction=float(damping_fraction),
    )

    scales = fields["scales"].to(weight.device, torch.float32)
    codebook = fields["codebook"]
    if isinstance(codebook, tuple):
        codebook = tuple(table.to(weight.device, torch.float32) for table in codebook)
    else:
        codebook = codebook.to(weight.device, torch.float32)
    cw = torch.broadcast_to(
        torch.as_tensor(col_weights).to(weight.device, torch.float32),
        weight.shape,
    )
    assignment_parts: list[dict[str, torch.Tensor]] = []

    def quantize_block(block: torch.Tensor, start: int, end: int) -> torch.Tensor:
        width = end - start
        block_scales = (
            scales[:, start // FP4_GROUP:end // FP4_GROUP]
            if grid == "fp4"
            else scales
        )
        wq = _col_weight_vectors(cw[:, start:end])
        _err, enc, decoded = _eval_candidate(
            block.to(torch.float32),
            wq,
            block_scales,
            grid,
            mode,
            codebook,
        )
        assignment_parts.append(
            _enc_to_fields(enc, mode, codebook, rows, width, width // VEC_DIM)
        )
        return decoded * _per_element_scale(block_scales, grid, width)

    # The returned reconstruction is intentionally discarded: export needs
    # the assignments, and reconstructing those fields is the shared decoder.
    block_error_feedback(
        weight,
        upper,
        quantize_block,
        block_size=int(block_size),
    )
    updated = dict(fields)
    updated["indices"] = torch.cat(
        [part["indices"] for part in assignment_parts], dim=1
    )
    if mode == "signed":
        updated["signs"] = torch.cat(
            [part["signs"] for part in assignment_parts], dim=1
        )
    return updated


def _ldlq_reassign_fields_3d_batched(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    activation_rows: Sequence[torch.Tensor],
    *,
    grid: str,
    mode: str,
    block_size: int,
    damping_fraction: float,
) -> dict:
    """Vectorize independent expert LDLQ solves over a batch dimension.

    Experts remain independent and the column-block loop retains the serial
    path's exact within-expert order.  Fixed-codebook assignment, triangular
    block solve, and feedback update are batched over the expert axis.  The
    inverse-Hessian factors remain the serial bit-identity anchors: CUDA's
    stacked Cholesky chooses a different numerical kernel and changes real
    expert indices at exact VQ boundaries.  Repeated cold-prior inputs share
    one exact factor, so identical work is still deduplicated.
    """
    if weight.ndim != 3:
        raise ValueError(
            f"batched LDLQ weight must be 3-D, got {tuple(weight.shape)}"
        )
    experts, rows, columns = map(int, weight.shape)
    activations = tuple(activation_rows)
    if len(activations) != experts:
        raise ValueError(
            f"LDLQ expert activation count {len(activations)} != "
            f"stack size {experts}"
        )
    if columns % int(block_size) or int(block_size) % FP4_GROUP:
        raise ValueError(
            f"LDLQ block_size={block_size} must divide in_features={columns} "
            f"and preserve group-{FP4_GROUP} scales"
        )

    raw_expert_batch = os.environ.get(
        "PRISMAQUANT_CB_LDLQ_EXPERT_BATCH", "16"
    ).strip()
    try:
        expert_batch = int(raw_expert_batch)
    except ValueError as exc:
        raise ValueError(
            "PRISMAQUANT_CB_LDLQ_EXPERT_BATCH must be a positive integer"
        ) from exc
    if expert_batch <= 0:
        raise ValueError(
            "PRISMAQUANT_CB_LDLQ_EXPERT_BATCH must be a positive integer"
        )
    if experts > expert_batch:
        # A single E=256 launch makes the tall assignment and feedback GEMMs
        # slower on GB10 than several resident same-shape batches.  Chunking
        # changes only the independent expert batch dimension; column blocks
        # and every operation within one expert retain their original order.
        indices = fields["indices"].reshape(experts * rows, -1)
        signs = fields.get("signs")
        if signs is not None:
            signs = signs.reshape(experts * rows, -1)
        scales = fields["scales"].reshape(experts * rows, -1)
        chunk_ranges = [
            (first, min(first + expert_batch, experts))
            for first in range(0, experts, expert_batch)
        ]

        def encode_chunk(first: int, last: int) -> dict:
            last = min(first + expert_batch, experts)
            row_first, row_last = first * rows, last * rows
            chunk_fields = dict(fields)
            chunk_fields["indices"] = indices[row_first:row_last]
            chunk_fields["scales"] = scales[row_first:row_last]
            if signs is not None:
                chunk_fields["signs"] = signs[row_first:row_last]
            chunk_fields["shape"] = (last - first, rows, columns)
            return _ldlq_reassign_fields_3d_batched(
                weight[first:last],
                chunk_fields,
                torch.broadcast_to(
                    torch.as_tensor(col_weights), weight.shape
                )[first:last],
                activations[first:last],
                grid=grid,
                mode=mode,
                block_size=block_size,
                damping_fraction=damping_fraction,
            )

        raw_streams = os.environ.get(
            "PRISMAQUANT_CB_LDLQ_BATCH_STREAMS", "1"
        ).strip()
        try:
            batch_streams = int(raw_streams)
        except ValueError as exc:
            raise ValueError(
                "PRISMAQUANT_CB_LDLQ_BATCH_STREAMS must be a positive integer"
            ) from exc
        if batch_streams <= 0:
            raise ValueError(
                "PRISMAQUANT_CB_LDLQ_BATCH_STREAMS must be a positive integer"
            )
        chunk_results: list[dict | None] = [None] * len(chunk_ranges)
        if batch_streams > 1 and weight.device.type == "cuda":
            from concurrent.futures import ThreadPoolExecutor

            stream_count = min(batch_streams, len(chunk_ranges))
            streams = [
                torch.cuda.Stream(device=weight.device)
                for _ in range(stream_count)
            ]

            def encode_stream(stream_id: int) -> list[tuple[int, dict]]:
                encoded: list[tuple[int, dict]] = []
                with torch.cuda.device(weight.device), torch.cuda.stream(
                    streams[stream_id]
                ):
                    for chunk_id in range(
                        stream_id, len(chunk_ranges), stream_count
                    ):
                        first, last = chunk_ranges[chunk_id]
                        encoded.append(
                            (chunk_id, encode_chunk(first, last))
                        )
                return encoded

            with ThreadPoolExecutor(max_workers=stream_count) as pool:
                for encoded in pool.map(encode_stream, range(stream_count)):
                    for chunk_id, result in encoded:
                        chunk_results[chunk_id] = result
            current = torch.cuda.current_stream(weight.device)
            for stream in streams:
                current.wait_stream(stream)
        else:
            for chunk_id, (first, last) in enumerate(chunk_ranges):
                chunk_results[chunk_id] = encode_chunk(first, last)
        ready = [result for result in chunk_results if result is not None]
        if len(ready) != len(chunk_ranges):
            raise RuntimeError("batched LDLQ stream lost an expert chunk")
        updated = dict(fields)
        updated["indices"] = torch.cat(
            [result["indices"] for result in ready], dim=0
        )
        if signs is not None:
            updated["signs"] = torch.cat(
                [result["signs"] for result in ready], dim=0
            )
        return updated

    xs: list[torch.Tensor] = []
    for x in activations:
        x = torch.as_tensor(x)
        if x.ndim != 2 or int(x.shape[1]) != columns:
            raise ValueError(
                "LDLQ activation rows must have shape (rows, in_features), got "
                f"{tuple(x.shape)} for expert weight {(rows, columns)}"
            )
        if int(x.shape[0]) == 0:
            raise ValueError("LDLQ activation rows must be non-empty")
        xs.append(x)

    upper_parts = [
        _ldlq_inverse_factor_cached(
            x,
            device=weight.device,
            damping_fraction=float(damping_fraction),
        )
        for x in xs
    ]
    upper = torch.stack(upper_parts)
    del xs, upper_parts

    scales = fields["scales"].to(weight.device, torch.float32).reshape(
        experts, rows, -1
    )
    codebook = fields["codebook"]
    if isinstance(codebook, tuple):
        codebook = tuple(
            table.to(weight.device, torch.float32) for table in codebook
        )
    else:
        codebook = codebook.to(weight.device, torch.float32)
    cw = torch.broadcast_to(
        torch.as_tensor(col_weights).to(weight.device, torch.float32),
        weight.shape,
    )
    work = weight.to(torch.float32).clone()
    assignment_parts: list[dict[str, torch.Tensor]] = []

    for start in range(0, columns, int(block_size)):
        end = start + int(block_size)
        width = end - start
        block = work[:, :, start:end]
        block_scales = (
            scales[:, :, start // FP4_GROUP:end // FP4_GROUP]
            if grid == "fp4"
            else scales
        )
        flat_block = block.reshape(experts * rows, width)
        flat_scales = block_scales.reshape(experts * rows, -1)
        wq = _col_weight_vectors(
            cw[:, :, start:end].reshape(experts * rows, width)
        )
        _err, enc, decoded = _eval_candidate(
            flat_block,
            wq,
            flat_scales,
            grid,
            mode,
            codebook,
        )
        assignment_parts.append(
            _enc_to_fields(
                enc,
                mode,
                codebook,
                experts * rows,
                width,
                width // VEC_DIM,
            )
        )
        qblock = (
            decoded
            * _per_element_scale(flat_scales, grid, width)
        ).reshape(experts, rows, width)
        residual = block - qblock
        diagonal_block = upper[:, start:end, start:end]
        scaled_error = torch.linalg.solve_triangular(
            diagonal_block.transpose(-2, -1),
            residual.transpose(-2, -1),
            upper=False,
        ).transpose(-2, -1)
        work[:, :, start:] -= torch.bmm(
            scaled_error,
            upper[:, start:end, start:],
        )

    updated = dict(fields)
    updated["indices"] = torch.cat(
        [part["indices"] for part in assignment_parts], dim=1
    )
    if mode == "signed":
        updated["signs"] = torch.cat(
            [part["signs"] for part in assignment_parts], dim=1
        )
    return updated


def _ldlq_reassign_fields_3d_threaded(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    activation_rows: Sequence[torch.Tensor],
    *,
    grid: str,
    mode: str,
    block_size: int,
    damping_fraction: float,
    workers: int,
) -> dict:
    """Feed exact per-expert LDLQ streams from multiple host threads.

    This is the secondary lever for rungs whose large fixed-codebook search
    does not benefit from flattening experts into one assignment batch.  One
    expert still executes the byte-pinned 2-D path, on one CUDA stream, in
    exactly the legacy order; threads only make independent units resident
    concurrently so a single Python core cannot starve the device.
    """
    from concurrent.futures import ThreadPoolExecutor

    if weight.device.type != "cuda":
        raise ValueError("threaded LDLQ feeder requires CUDA expert weights")
    experts, rows, columns = map(int, weight.shape)
    activations = tuple(activation_rows)
    if len(activations) != experts:
        raise ValueError(
            f"LDLQ expert activation count {len(activations)} != "
            f"stack size {experts}"
        )
    workers = min(max(1, int(workers)), experts)
    indices = fields["indices"].reshape(experts * rows, -1)
    scales = fields["scales"].reshape(experts * rows, -1)
    signs = fields.get("signs")
    if signs is not None:
        signs = signs.reshape(experts * rows, -1)
    cw = torch.broadcast_to(torch.as_tensor(col_weights), weight.shape)
    streams = [torch.cuda.Stream(device=weight.device) for _ in range(workers)]

    def encode_worker(worker: int) -> list[tuple[int, dict]]:
        encoded: list[tuple[int, dict]] = []
        stream = streams[worker]
        with torch.cuda.device(weight.device), torch.cuda.stream(stream):
            for expert in range(worker, experts, workers):
                first, last = expert * rows, (expert + 1) * rows
                expert_fields = dict(fields)
                expert_fields["indices"] = indices[first:last]
                expert_fields["scales"] = scales[first:last]
                if signs is not None:
                    expert_fields["signs"] = signs[first:last]
                expert_fields["shape"] = (rows, columns)
                result = _ldlq_reassign_fields_2d(
                    weight[expert],
                    expert_fields,
                    cw[expert],
                    activations[expert],
                    grid=grid,
                    mode=mode,
                    block_size=block_size,
                    damping_fraction=damping_fraction,
                )
                encoded.append((expert, result))
        return encoded

    results: list[dict | None] = [None] * experts
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for encoded in pool.map(encode_worker, range(workers)):
            for expert, result in encoded:
                results[expert] = result
    current = torch.cuda.current_stream(weight.device)
    for stream in streams:
        current.wait_stream(stream)
    ready = [result for result in results if result is not None]
    if len(ready) != experts:
        raise RuntimeError("threaded LDLQ feeder lost an expert result")
    updated = dict(fields)
    updated["indices"] = torch.cat(
        [result["indices"] for result in ready], dim=0
    )
    if signs is not None:
        updated["signs"] = torch.cat(
            [result["signs"] for result in ready], dim=0
        )
    return updated


LDLQ_GATE_ENV = "PRISMAQUANT_CB_LDLQ_GATE"
LDLQ_GATE_EPSILON = 1e-12


def _ldlq_gate_enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    raw = str(values.get(LDLQ_GATE_ENV, "1")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{LDLQ_GATE_ENV} must be a boolean 0/1 setting, got {raw!r}")


def _ldlq_weighted_mse(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    col_weights: torch.Tensor,
) -> torch.Tensor:
    """Per-row col-weighted MSE, matching the cost's activation-aware branch."""
    err = (weight.to(torch.float32) - reconstruction.to(torch.float32)).pow(2)
    if col_weights is not None:
        cw = torch.broadcast_to(
            torch.as_tensor(col_weights).to(weight.device, torch.float32),
            weight.shape,
        ).to(torch.float32)
        # Match _col_weight_vectors dead-vector guard: zero-mass -> unweighted.
        if err.dim() == 2:
            mass = cw.sum(dim=-1, keepdim=True)
            cw = torch.where(mass == 0, torch.ones_like(cw), cw)
        elif err.dim() == 3:
            mass = cw.sum(dim=-1, keepdim=True)
            cw = torch.where(mass == 0, torch.ones_like(cw), cw)
        err = err * cw
        denom = cw.sum(dim=-1).clamp_min(1e-30).mean().clamp_min(1e-30)
        # Use mean over all elements weighted by cw; denom above keeps scale
        # stable when some rows have zero mass. For per-expert gating we need
        # per-slice value, so caller handles slicing.
        return err.mean()
    return err.mean()


def _ldlq_per_expert_weighted_mse(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    col_weights: torch.Tensor,
) -> list[float]:
    """Return per-expert col-weighted MSE for a 3-D stack or single Linear."""
    if weight.ndim == 2:
        return [_ldlq_weighted_mse(weight, reconstruction, col_weights).item()]
    # 3-D: (E, R, C)
    values: list[float] = []
    for idx in range(int(weight.shape[0])):
        w = weight[idx]
        r = reconstruction[idx] if reconstruction.ndim == 3 else reconstruction
        cw = col_weights[idx] if col_weights.ndim == 3 else col_weights
        # col_weights for experts is (E,1,C) broadcast; slice matches.
        if cw is not None and cw.ndim == 3:
            cw_slice = cw[idx] if cw.shape[0] == weight.shape[0] else cw
        else:
            cw_slice = cw
        err = (w.to(torch.float32) - r.to(torch.float32)).pow(2)
        if cw_slice is not None:
            cw_b = torch.broadcast_to(
                torch.as_tensor(cw_slice).to(w.device, torch.float32), w.shape
            ).to(torch.float32)
            mass = cw_b.sum(dim=-1, keepdim=True)
            cw_b = torch.where(mass == 0, torch.ones_like(cw_b), cw_b)
            err = err * cw_b
        values.append(float(err.mean().item()))
    return values


def _ldlq_activation_mse(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    activation_rows: torch.Tensor | Sequence[torch.Tensor] | None,
) -> float | None:
    """Activation-weighted output MSE, or None if no activation rows.

    This is the declared gate metric ``activation_output_mse``.  It is fail-
    closed on malformed rows: a non-2-D tensor, a width mismatch, or any
    other structural error raises immediately so the caller can record an
    explicit fallback reason or abort, rather than silently falling back to
    a different metric.
    """
    if activation_rows is None:
        return None
    if isinstance(activation_rows, torch.Tensor):
        act = torch.as_tensor(activation_rows)
        if act.numel() == 0 or act.shape[0] == 0:
            return None
        if act.ndim != 2:
            raise ValueError(f"activation rows must be rank-2, got shape {tuple(act.shape)}")
        if int(act.shape[1]) != int(weight.shape[-1]):
            raise ValueError(
                f"activation width {act.shape[1]} != weight in_features {weight.shape[-1]}"
            )
        if weight.ndim == 3:
            total = 0.0
            for idx in range(int(weight.shape[0])):
                w = weight[idx].to(torch.float32)
                r = reconstruction[idx].to(torch.float32) if reconstruction.ndim == 3 else reconstruction.to(torch.float32)
                err = (act.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item()
                total += float(err)
            return total / max(int(weight.shape[0]), 1)
        w = weight.to(torch.float32)
        r = reconstruction.to(torch.float32)
        return float((act.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item())
    # Sequence per expert
    seq = tuple(activation_rows)
    if not seq:
        return None
    # Check for any empty or malformed entry; empty is not an error but a
    # missing-data signal that the caller must handle explicitly.
    has_data = False
    for act in seq:
        act_t = torch.as_tensor(act)
        if act_t.numel() == 0 or act_t.shape[0] == 0:
            continue
        has_data = True
        if act_t.ndim != 2:
            raise ValueError(f"per-expert activation rows must be rank-2, got {tuple(act_t.shape)}")
        if int(act_t.shape[1]) != int(weight.shape[-1]):
            raise ValueError(
                f"per-expert activation width {act_t.shape[1]} != weight in_features {weight.shape[-1]}"
            )
    if not has_data:
        return None
    if weight.ndim == 2:
        act = torch.as_tensor(seq[0])
        if act.numel() == 0:
            return None
        w = weight.to(torch.float32)
        r = reconstruction.to(torch.float32)
        return float((act.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item())
    total = 0.0
    count = 0
    for idx, act in enumerate(seq):
        act_t = torch.as_tensor(act)
        if act_t.numel() == 0 or act_t.shape[0] == 0:
            continue
        w = weight[idx].to(torch.float32)
        r = reconstruction[idx].to(torch.float32) if reconstruction.ndim == 3 else reconstruction.to(torch.float32)
        total += float((act_t.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item())
        count += 1
    return total / max(count, 1) if count else None


def _ldlq_per_expert_activation_mse(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    activation_rows: Sequence[torch.Tensor],
) -> list[float] | None:
    if weight.ndim != 3:
        return None
    seq = tuple(activation_rows)
    if len(seq) != int(weight.shape[0]):
        raise ValueError(
            f"per-expert activation count {len(seq)} != stack size {weight.shape[0]}"
        )
    values: list[float] = []
    for idx, act in enumerate(seq):
        act_t = torch.as_tensor(act)
        if act_t.numel() == 0 or act_t.shape[0] == 0:
            # Missing rows for this expert is not silently inf; the caller
            # must decide to fallback per expert with an explicit reason.
            # We return inf as a sentinel that the caller will interpret as
            # missing, but we do not swallow malformed shapes.
            values.append(float("inf"))
            continue
        if act_t.ndim != 2:
            raise ValueError(f"per-expert activation rows must be rank-2, got {tuple(act_t.shape)} for expert {idx}")
        if int(act_t.shape[1]) != int(weight.shape[-1]):
            raise ValueError(
                f"per-expert activation width {act_t.shape[1]} != weight in_features {weight.shape[-1]} for expert {idx}"
            )
        w = weight[idx].to(torch.float32)
        r = reconstruction[idx].to(torch.float32) if reconstruction.ndim == 3 else reconstruction.to(torch.float32)
        err = float((act_t.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item())
        values.append(err)
    return values


def ldlq_reassign_cb_fields(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    activation_rows: torch.Tensor | Sequence[torch.Tensor],
    *,
    grid: str,
    mode: str,
    block_size: int = LDLQ_BLOCK_SIZE,
    damping_fraction: float = LDLQ_DAMPING_FRACTION,
    batch_experts: bool | None = None,
) -> dict:
    """Run deterministic fixed-scale/codebook LDLQ assignment.

    Scale and codebook fitting have already completed. For stacked experts a
    sequence supplies one activation matrix (and Hessian) per expert; a single
    matrix deliberately shares one Hessian across the stack.
    """
    if batch_experts is None:
        raw_batch = os.environ.get(
            "PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS", "1"
        ).strip().lower()
        if raw_batch not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError(
                "PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS must be 0 or 1"
            )
        batch_experts = raw_batch in {"1", "true", "yes", "on"}
    if weight.ndim == 2 or isinstance(activation_rows, torch.Tensor):
        flat = weight.reshape(-1, weight.shape[-1])
        flat_col_weights = torch.broadcast_to(
            torch.as_tensor(col_weights), weight.shape
        ).reshape_as(flat)
        return _ldlq_reassign_fields_2d(
            flat,
            fields,
            flat_col_weights,
            torch.as_tensor(activation_rows),
            grid=grid,
            mode=mode,
            block_size=block_size,
            damping_fraction=damping_fraction,
        )
    if weight.ndim != 3:
        raise ValueError(
            "per-slice LDLQ activations require a 3-D expert stack, got "
            f"{tuple(weight.shape)}"
        )
    activations = tuple(activation_rows)
    if len(activations) != int(weight.shape[0]):
        raise ValueError(
            f"LDLQ expert activation count {len(activations)} != "
            f"stack size {weight.shape[0]}"
        )
    if batch_experts:
        raw_workers = os.environ.get(
            "PRISMAQUANT_CB_LDLQ_FEEDER_THREADS", "0"
        ).strip()
        try:
            feeder_workers = int(raw_workers)
        except ValueError as exc:
            raise ValueError(
                "PRISMAQUANT_CB_LDLQ_FEEDER_THREADS must be a non-negative "
                "integer"
            ) from exc
        if feeder_workers < 0:
            raise ValueError(
                "PRISMAQUANT_CB_LDLQ_FEEDER_THREADS must be a non-negative "
                "integer"
            )
        if feeder_workers and weight.device.type == "cuda":
            return _ldlq_reassign_fields_3d_threaded(
                weight,
                fields,
                col_weights,
                activations,
                grid=grid,
                mode=mode,
                block_size=block_size,
                damping_fraction=damping_fraction,
                workers=feeder_workers,
            )
        return _ldlq_reassign_fields_3d_batched(
            weight,
            fields,
            col_weights,
            activations,
            grid=grid,
            mode=mode,
            block_size=block_size,
            damping_fraction=damping_fraction,
        )

    # Retained as the bit-identity reference.  Production takes the batched
    # arm; tests and measurement gates exercise both on identical inputs.
    cw = torch.broadcast_to(torch.as_tensor(col_weights), weight.shape)
    rows_per_expert = int(weight.shape[1])
    assignment_parts: list[dict] = []
    slice_keys = {"indices", "signs", "scales", "scale_super", "scale_sub"}
    for expert, x in enumerate(activations):
        start = expert * rows_per_expert
        end = start + rows_per_expert
        local = {
            key: (value[start:end] if key in slice_keys else value)
            for key, value in fields.items()
        }
        local["shape"] = tuple(weight[expert].shape)
        assignment_parts.append(_ldlq_reassign_fields_2d(
            weight[expert],
            local,
            cw[expert],
            x,
            grid=grid,
            mode=mode,
            block_size=block_size,
            damping_fraction=damping_fraction,
        ))
    updated = dict(fields)
    updated["indices"] = torch.cat(
        [part["indices"] for part in assignment_parts], dim=0
    )
    if mode == "signed":
        updated["signs"] = torch.cat(
            [part["signs"] for part in assignment_parts], dim=0
        )
    return updated


def ldlq_reassign_cb_fields_gated(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    activation_rows: torch.Tensor | Sequence[torch.Tensor],
    *,
    grid: str,
    mode: str,
    k: int,
    block_size: int = LDLQ_BLOCK_SIZE,
    damping_fraction: float = LDLQ_DAMPING_FRACTION,
    batch_experts: bool | None = None,
    gate: bool | None = None,
) -> tuple[dict, dict]:
    """Fixed-codebook LDLQ with per-unit do-no-harm fallback.

    Returns ``(fields, gate_info)`` where ``gate_info`` records whether the
    LDLQ arm was kept per Linear / per expert slice.  The byte payload is
    identical in either arm, so this is a pure quality gate.  When ``gate``
    is False the raw LDLQ result is returned verbatim (no comparison).
    """
    if gate is None:
        gate = _ldlq_gate_enabled()
    if not gate:
        ldlq_fields = ldlq_reassign_cb_fields(
            weight, fields, col_weights, activation_rows,
            grid=grid, mode=mode, block_size=block_size,
            damping_fraction=damping_fraction, batch_experts=batch_experts,
        )
        return ldlq_fields, {"gate": "disabled", "kept_ldlq": True}
    # Raw reconstruction (no LDLQ) for comparison.
    raw_recon = nvfp4_cb_reconstruct(fields, k, grid=grid, mode=mode).to(weight.dtype)
    ldlq_fields = ldlq_reassign_cb_fields(
        weight, fields, col_weights, activation_rows,
        grid=grid, mode=mode, block_size=block_size,
        damping_fraction=damping_fraction, batch_experts=batch_experts,
    )
    ldlq_recon = nvfp4_cb_reconstruct(ldlq_fields, k, grid=grid, mode=mode).to(weight.dtype)
    # Per-unit gate: for packed experts decide per slice, otherwise whole tensor.
    # Declared metric is activation_output_mse when activation rows were supplied;
    # missing or malformed rows select raw with an explicit reason, never silently
    # changing metric.
    if weight.ndim == 3:
        # Activation rows must be a per-expert sequence for 3-D; a single tensor
        # is not a valid per-expert activation for the gate - treat as missing.
        if activation_rows is None:
            # No activation supplied: fail closed to raw with explicit reason.
            return fields, {
                "gate": "raw_fallback_no_activation",
                "kept_ldlq": False,
                "reason": "activation_rows is None but LDLQ gate requires activation_output_mse",
                "metric": "activation_output_mse",
            }
        if isinstance(activation_rows, torch.Tensor):
            # Single tensor for 3-D is ambiguous: it would be shared across experts,
            # but the declared per-expert gate requires per-expert rows. Fall back
            # to raw with explicit reason.
            return fields, {
                "gate": "raw_fallback_shared_activation_for_packed",
                "kept_ldlq": False,
                "reason": "3-D weight requires per-expert activation sequence, got single tensor",
                "metric": "activation_output_mse",
            }
        try:
            raw_act_per = _ldlq_per_expert_activation_mse(weight, raw_recon, activation_rows)
            ldlq_act_per = _ldlq_per_expert_activation_mse(weight, ldlq_recon, activation_rows)
        except ValueError as exc:
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": str(exc),
                "metric": "activation_output_mse",
            }
        # Check for missing per-expert rows (inf sentinel)
        if any(v == float("inf") for v in raw_act_per + ldlq_act_per):
            # Missing rows for some experts: per-expert fallback for those, keep
            # LDLQ for others where data exists. Record explicit per-expert reason.
            missing = [i for i, v in enumerate(raw_act_per) if v == float("inf") or ldlq_act_per[i] == float("inf")]
            keep_mask_missing: list[bool] = []
            for idx, (raw_err, ldlq_err) in enumerate(zip(raw_act_per, ldlq_act_per)):
                if idx in missing:
                    keep_mask_missing.append(False)
                else:
                    keep = ldlq_err <= raw_err + LDLQ_GATE_EPSILON * max(abs(raw_err), abs(ldlq_err), 1.0)
                    keep_mask_missing.append(bool(keep))
            if all(not k for k in keep_mask_missing):
                return fields, {
                    "gate": "raw_kept_all_missing_activation",
                    "kept_ldlq": False,
                    "per_expert_kept": keep_mask_missing,
                    "missing_experts": missing,
                    "raw_mse_per_expert": raw_act_per,
                    "ldlq_mse_per_expert": ldlq_act_per,
                    "metric": "activation_output_mse",
                    "reason": f"missing activation for experts {missing}",
                }
            # For mixed missing case, use the already computed keep_mask and
            # raw/ldlq per-expert values for the mixed construction below.
            raw_per, ldlq_per = raw_act_per, ldlq_act_per
            keep_mask = keep_mask_missing
        else:
            raw_per, ldlq_per = raw_act_per, ldlq_act_per
            keep_mask = []
            for raw_err, ldlq_err in zip(raw_per, ldlq_per):
                keep = ldlq_err <= raw_err + LDLQ_GATE_EPSILON * max(abs(raw_err), abs(ldlq_err), 1.0)
                keep_mask.append(bool(keep))
        if all(keep_mask):
            return ldlq_fields, {
                "gate": "ldlq_kept_all",
                "kept_ldlq": True,
                "per_expert_kept": keep_mask,
                "raw_mse_per_expert": raw_per,
                "ldlq_mse_per_expert": ldlq_per,
            }
        if not any(keep_mask):
            return fields, {
                "gate": "raw_kept_all",
                "kept_ldlq": False,
                "per_expert_kept": keep_mask,
                "raw_mse_per_expert": raw_per,
                "ldlq_mse_per_expert": ldlq_per,
            }
        # Mixed: keep LDLQ slices where it wins, raw elsewhere.  We must
        # splice indices/scales per expert from the winning arm.  The simplest
        # byte-correct splice is to rebuild fields per expert from the two
        # sources.
        # ``fields`` and ``ldlq_fields`` share scales/codebook (LDLQ is
        # fixed-codebook, fixed-scale), only indices/signs differ, so splicing
        # indices is sufficient and byte-identical to re-encoding the winner.
        # For 3-D, indices are (E*R, nvec) flattened; slice per expert.
        rows_per_expert = int(weight.shape[1])
        # Indices are (E*R, ...) flattened; recover per-expert blocks.
        def _slice_indices(src: dict) -> list[torch.Tensor]:
            idx = src["indices"]
            # idx shape (E*R, ...) -> per expert (R, ...)
            return [idx[e*rows_per_expert:(e+1)*rows_per_expert] for e in range(int(weight.shape[0]))]
        raw_slices = _slice_indices(fields)
        ldlq_slices = _slice_indices(ldlq_fields)
        mixed_slices = [
            ldlq_slices[e] if keep_mask[e] else raw_slices[e]
            for e in range(len(keep_mask))
        ]
        mixed_indices = torch.cat(mixed_slices, dim=0)
        updated = dict(ldlq_fields if any(keep_mask) else fields)
        updated["indices"] = mixed_indices
        if mode == "signed":
            raw_sign_slices = [fields["signs"][e*rows_per_expert:(e+1)*rows_per_expert] for e in range(len(keep_mask))]
            ldlq_sign_slices = [ldlq_fields["signs"][e*rows_per_expert:(e+1)*rows_per_expert] for e in range(len(keep_mask))]
            mixed_signs = [
                ldlq_sign_slices[e] if keep_mask[e] else raw_sign_slices[e]
                for e in range(len(keep_mask))
            ]
            updated["signs"] = torch.cat(mixed_signs, dim=0)
        return updated, {
            "gate": "mixed_per_expert",
            "kept_ldlq": keep_mask,
            "per_expert_kept": keep_mask,
            "raw_mse_per_expert": raw_per,
            "ldlq_mse_per_expert": ldlq_per,
        }
    # 2-D case: whole-tensor gate. Activation is required; missing selects raw
    # with explicit reason, never silently changes metric.
    if activation_rows is None:
        return fields, {
            "gate": "raw_fallback_no_activation",
            "kept_ldlq": False,
            "reason": "activation_rows is None but LDLQ gate requires activation_output_mse",
            "metric": "activation_output_mse",
        }
    try:
        raw_act = _ldlq_activation_mse(weight, raw_recon, activation_rows)
        ldlq_act = _ldlq_activation_mse(weight, ldlq_recon, activation_rows)
    except ValueError as exc:
        return fields, {
            "gate": "raw_fallback_malformed_activation",
            "kept_ldlq": False,
            "reason": str(exc),
            "metric": "activation_output_mse",
        }
    if raw_act is None or ldlq_act is None:
        return fields, {
            "gate": "raw_fallback_missing_activation",
            "kept_ldlq": False,
            "reason": "activation rows empty or missing for 2-D gate",
            "metric": "activation_output_mse",
        }
    raw_err, ldlq_err = float(raw_act), float(ldlq_act)
    if ldlq_err <= raw_err + LDLQ_GATE_EPSILON * max(abs(raw_err), abs(ldlq_err), 1.0):
        return ldlq_fields, {
            "gate": "ldlq_kept",
            "kept_ldlq": True,
            "raw_mse": raw_err,
            "ldlq_mse": ldlq_err,
        }
    return fields, {
        "gate": "raw_kept",
        "kept_ldlq": False,
        "raw_mse": raw_err,
        "ldlq_mse": ldlq_err,
    }


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
    elif mode == "signed":
        vecs = cb[indices]                                   # (rows, nvec, 8)
        recon = vecs.reshape(rows, in_f) * fields["signs"]
    else:
        parts = [table[indices[..., i]] for i, table in enumerate(cb)]
        recon = torch.cat(parts, dim=-1).reshape(rows, in_f)
    pes = _per_element_scale(scales.to(recon.dtype), grid, in_f)
    recon = recon * pes
    if shape is not None:
        recon = recon.reshape(shape)
    return recon


def make_nvfp4_cb_qdq(k: int, grid: str = "fp4", mode: str = "product",
                      scale_sweep: bool = True,
                      scale_coding: str = SCALE_CODING_V1,
                      encode_tier: str | None = None):
    """Single-source emulation closure ``(w, col_weights=None) -> w_hat``
    used by both cost and (Milestone B) the packer. ``scale_sweep`` defaults
    True (joint scale search, IQ-rendering parity); ``scale_coding``
    selects the v1 e4m3 plane (default) or the layout-v2 two-tier coding;
    ``encode_tier`` the fast/balanced/max speed-accuracy tier (None ->
    PRISMAQUANT_CB_ENCODE_TIER, resolved per call)."""
    def f(w: torch.Tensor, col_weights: torch.Tensor | None = None
          ) -> torch.Tensor:
        fields = nvfp4_cb_fields(w, k, grid=grid, mode=mode,
                                 col_weights=col_weights,
                                 scale_sweep=scale_sweep,
                                 scale_coding=scale_coding,
                                 encode_tier=encode_tier)
        return nvfp4_cb_reconstruct(fields, k, grid=grid, mode=mode).to(w.dtype)
    return f


# ---------------------------------------------------------------------------
# RD-law ladder interpolation (cost path, opt-in): encode a few ANCHOR rungs,
# fit the one-parameter rate-distortion law D(k) = C * 2^(-k/4) per Linear
# (log2 D linear in k with fixed slope -1/4), predict weighted-recon cost at
# every other rung, and report a HOLDOUT check so the caller can fall back to
# full measurement where the fit is not trusted. This is "surrogate proposes,
# measurement verifies" — the anchors and the holdout ARE measurements; no
# unverified closed form ships (the analytical-damp graveyard rule).
# Wiring point: the local-cost path may consult this behind
# PRISMAQUANT_CB_LADDER_INTERP=1 (default OFF; cost-path wiring belongs to
# the menu-integration workstream — this module only provides the helper).
# ---------------------------------------------------------------------------

RD_LAW_SLOPE_BITS = 0.25          # log2 D drop per index bit (D ~ 2^(-k/4))


def _weighted_recon_cost(w: torch.Tensor, k: int, *, grid: str, mode: str,
                         col_weights: torch.Tensor | None,
                         scale_coding: str, encode_tier: str | None) -> float:
    qdq = make_nvfp4_cb_qdq(k, grid, mode, scale_coding=scale_coding,
                            encode_tier=encode_tier)
    w_hat = qdq(w, col_weights)
    err = (w.to(torch.float32) - w_hat.to(torch.float32)).pow(2)
    if col_weights is not None:
        cw = torch.broadcast_to(
            col_weights.to(w.device, torch.float32), w.shape)
        err = err * cw
    return float(err.sum())


def predict_cb_ladder_costs(w: torch.Tensor, ks: tuple[int, ...], *,
                            grid: str = "fp4", mode: str = "product",
                            col_weights: torch.Tensor | None = None,
                            anchors: tuple[int, ...] = (12, 18, 24),
                            holdout: int | None = None,
                            scale_coding: str = SCALE_CODING_V1,
                            encode_tier: str | None = None) -> dict:
    """Predict per-rung weighted-recon cost from a few measured anchors.

    Encodes ``anchors`` (and ``holdout`` if given) for real, fits
    ``log2 D(k) = log2 C - k/4`` (one parameter ``C``), and returns::

        {"measured": {k: D}, "predicted": {k: D_hat for k in ks},
         "log2_C": float, "holdout": {"k", "measured", "predicted",
                                      "rel_error"} | None}

    The caller trusts the interpolation only where the holdout relative
    error clears its noise floor (docs/lanes/nvfp4-cb/encode_tiers.md §B),
    falling back to full per-rung measurement elsewhere.
    """
    measured: dict[int, float] = {}
    for ak in anchors:
        measured[int(ak)] = _weighted_recon_cost(
            w, int(ak), grid=grid, mode=mode, col_weights=col_weights,
            scale_coding=scale_coding, encode_tier=encode_tier)
    logc = sum(
        (torch.log2(torch.tensor(d)).item() + RD_LAW_SLOPE_BITS * ak)
        for ak, d in measured.items()) / len(measured)
    predicted = {int(k): float(2.0 ** (logc - RD_LAW_SLOPE_BITS * int(k)))
                 for k in ks}
    hold = None
    if holdout is not None:
        h_meas = _weighted_recon_cost(
            w, int(holdout), grid=grid, mode=mode, col_weights=col_weights,
            scale_coding=scale_coding, encode_tier=encode_tier)
        h_pred = float(2.0 ** (logc - RD_LAW_SLOPE_BITS * int(holdout)))
        hold = {"k": int(holdout), "measured": h_meas, "predicted": h_pred,
                "rel_error": abs(h_pred - h_meas) / max(h_meas, 1e-30)}
    return {"measured": measured, "predicted": predicted,
            "log2_C": float(logc), "holdout": hold}


# ---------------------------------------------------------------------------
# Milestone B — byte packers (export path). Bit-exact on-disk layout:
# docs/lanes/nvfp4-cb/format-pipeline.md §1 / docs/lanes/nvfp4-cb/LAYOUT.md.
#
# The index body, scale-plane sizes, and total type size come only from
# ``cb_layout``. The packed tensor is 2-D uint8 (rows, bytes_per_row) — NEVER
# a flat 1-D buffer (the GGUF
# lesson: a flat store loses the logical row/superblock structure the reader
# and serving kernel index into). fp8 ships NO scale plane in the weight bytes;
# its per-output-channel fp32 scales are a separate ``<name>.weight_scale``.
# ---------------------------------------------------------------------------

def nvfp4_cb_type_size(k: int, grid: str = "fp4",
                       scale_coding: str = SCALE_CODING_V1) -> int:
    """Public: on-disk bytes per 256-weight superblock for a CB rung."""
    return _serialized_type_size(k, grid, scale_coding)


def nvfp4_cb_effective_bits(k: int, grid: str = "fp4",
                            scale_coding: str = SCALE_CODING_V1) -> float:
    """Version-keyed body bpw (spec §2): fp4 v1 ``k/8 + 0.5``, fp4 v2
    two-tier ``k/8 + 0.28125``, fp8 ``k/8`` (per-channel scale separate).
    Registered FormatSpec rates are nominal compatibility metadata; exact
    producer pricing is versioned by ``CBSerializationContext`` and asserted
    against the serialized payload."""
    return _serialized_type_size(k, grid, scale_coding) * 8.0 / SUPERBLOCK


def _vector_codes(fields: dict, k: int, grid: str, mode: str) -> torch.Tensor:
    """Per-8-weight-vector k-bit codeword (rows, nvec_per_row), int64.

    Bit layout inside the k-bit field (LSB-first), per §1.1 + the PLAN's
    product decomposition:
      * full   — the codebook index itself (k bits);
      * product— the n_sub sub-indices contiguous, sub0 in the low bits
                 (bit widths ``subtable_bit_widths(k, "product", n_sub)``,
                 ceil-first);
      * signed — 8 sign bits (bit j == coord j is negative) in the low byte,
                 then the (k-8)-bit magnitude index above them.
    """
    idx = fields["indices"]
    rows = idx.shape[0]
    if mode == "full":
        return idx.reshape(rows, -1).to(torch.int64)
    if mode == "signed":
        mag = idx.reshape(rows, -1).to(torch.int64)               # (rows, nvec)
        signs = fields["signs"].reshape(rows, -1, VEC_DIM)        # (rows,nvec,8)
        neg = (signs < 0).to(torch.int64)
        shifts = torch.arange(VEC_DIM, device=neg.device)
        sign_byte = (neg << shifts).sum(dim=-1)                   # (rows, nvec)
        return sign_byte | (mag << VEC_DIM)
    # product: idx is (rows, nvec, canonical family n_sub)
    n_sub = family_for(grid, mode).n_sub
    if idx.shape[-1] != n_sub:
        raise ValueError(
            f"{grid} {mode} index stream has {idx.shape[-1]} subtables; "
            f"serialized family requires {n_sub}"
        )
    bits = subtable_bit_widths(k, mode, n_sub)
    code = torch.zeros(idx.shape[:-1], dtype=torch.int64, device=idx.device)
    off = 0
    for i in range(n_sub):
        code = code | (idx[..., i].to(torch.int64) << off)
        off += bits[i]
    return code.reshape(rows, -1)


def _pack_codes_to_bytes(codes: torch.Tensor, k: int) -> torch.Tensor:
    """Pack k-bit codewords into the canonical superblock index body.

    Each superblock is byte-aligned, so the codewords pack contiguously
    LSB-first (codeword c
    occupies stream bits [c*k, c*k+k), its own LSB first; bytes fill LSB-first).
    """
    rows, nvec = codes.shape
    if nvec % CODEWORDS_PER_SUPERBLOCK:
        raise ValueError(
            f"codeword count {nvec} is not divisible by "
            f"{CODEWORDS_PER_SUPERBLOCK}"
        )
    n_sb = nvec // CODEWORDS_PER_SUPERBLOCK
    shifts = torch.arange(k, device=codes.device)
    wt = 1 << torch.arange(8, device=codes.device)
    # Chunk over rows: the (chunk, nvec, k) int64 bitstream transient is
    # ~16x the packed bytes — unchunked it is ~155GB on a Hy3 192-expert
    # stack (three box-wide OOMs, 2026-07-19). Rows are independent, so
    # chunking is bit-identical.
    step = max(1, _SLICE_MAX_ELEMS // max(nvec * k, 1))
    index_bytes = INDEX_BYTES_PER_K * k
    out = torch.empty(rows, n_sb, index_bytes, dtype=torch.uint8,
                      device=codes.device)
    for a in range(0, rows, step):
        b = min(rows, a + step)
        bits = (codes[a:b].unsqueeze(-1) >> shifts) & 1      # (chunk, nvec, k)
        bits = bits.reshape(b - a, n_sb, index_bytes, 8)
        out[a:b] = (bits * wt).sum(dim=-1).to(torch.uint8)
    return out                                               # (rows,n_sb,4k)


def _unpack_bytes_to_codes(idx_bytes: torch.Tensor, k: int) -> torch.Tensor:
    """Inverse of :func:`_pack_codes_to_bytes` under ``cb_layout``."""
    rows, n_sb, _ = idx_bytes.shape
    bshift = torch.arange(8, device=idx_bytes.device)
    bits = (idx_bytes.to(torch.int64).unsqueeze(-1) >> bshift) & 1
    bits = bits.reshape(
        rows, n_sb, CODEWORDS_PER_SUPERBLOCK, k
    )                                                          # k bits/codeword
    kshift = torch.arange(k, device=idx_bytes.device)
    codes = (bits << kshift).sum(dim=-1)                        # (rows,n_sb,32)
    return codes.reshape(rows, n_sb * CODEWORDS_PER_SUPERBLOCK)


def _scale_plane_bytes(scales: torch.Tensor, n_sb: int) -> torch.Tensor:
    """Encode the canonical fp4-v1 scale plane. ``scales`` (rows, in//16)
    are already E4M3-exact (snapped by the encoder), so the E4M3 byte view is
    lossless."""
    rows = scales.shape[0]
    s = scales.reshape(
        rows, n_sb, FP4_SCALE_GROUPS_PER_SUPERBLOCK
    ).to(_E4M3)
    return s.contiguous().view(torch.uint8)


def _two_tier_scale_bytes(super_e: torch.Tensor, sub_c: torch.Tensor,
                          n_sb: int) -> torch.Tensor:
    """Encode the canonical fp4-v2 two-tier scale plane.

    The first byte is E8M0; remaining bytes pack two 4-bit subscale codes,
    with the even group in the low nibble. Spec §5.1.
    """
    rows = super_e.shape[0]
    sup = super_e.reshape(rows, n_sb, 1).to(torch.uint8)
    c = sub_c.reshape(
        rows, n_sb, FP4_SCALE_GROUPS_PER_SUPERBLOCK
    ).to(torch.int64)
    if bool((c < 0).any()) or bool((c > 15).any()):
        raise ValueError("two-tier sub codes must be 4-bit (0..15)")
    pairs = c.reshape(
        rows, n_sb, FP4_SCALE_GROUPS_PER_SUPERBLOCK // 2, 2
    )
    sub = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)
    return torch.cat([sup, sub], dim=-1)


def _two_tier_scale_unpack(sc_bytes: torch.Tensor):
    """Decode the canonical fp4-v2 scale plane to exact E4M3 scales."""
    expected = SCALE_PLANE_BYTES[("fp4", SCALE_CODING_TWO_TIER)]
    if sc_bytes.shape[-1] != expected:
        raise ValueError(
            f"two-tier scale plane has {sc_bytes.shape[-1]} bytes; "
            f"expected {expected}"
        )
    rows, n_sb, _ = sc_bytes.shape
    super_e = sc_bytes[..., 0].to(torch.int64)
    sub = sc_bytes[..., 1:].to(torch.int64)                     # (rows,n_sb,8)
    lo = sub & 0xF
    hi = (sub >> 4) & 0xF
    codes = torch.stack([lo, hi], dim=-1).reshape(rows, n_sb, -1)
    _, compose, legal = _two_tier_tables(str(sc_bytes.device))
    e_exp = super_e.unsqueeze(-1).expand_as(codes)
    if not bool(legal[e_exp, codes].all()):
        raise ValueError(
            "two-tier scale bytes contain an illegal (super, sub) pair")
    scales = compose[e_exp, codes].reshape(
        rows, n_sb * FP4_SCALE_GROUPS_PER_SUPERBLOCK
    )
    return super_e, codes.reshape(rows, -1), scales


def nvfp4_cb_assemble_bytes(fields: dict, k: int, grid: str = "fp4",
                            mode: str = "product") -> torch.Tensor:
    """Bit-pack VQ ``fields`` into the §1 on-disk byte layout.

    Returns a 2-D uint8 tensor ``(rows, n_superblocks * type_size)`` on the
    fields' device. The scale coding is taken from the fields (a two-tier
    encode carries ``scale_super``/``scale_sub``); the size is resolved from
    :mod:`prismaquant.cb_layout` and asserted.
    """
    k = int(k)
    scale_coding = fields.get("scale_coding", SCALE_CODING_V1)
    codes = _vector_codes(fields, k, grid, mode)                # (rows, nvec)
    rows, nvec = codes.shape
    if nvec % CODEWORDS_PER_SUPERBLOCK != 0:
        raise ValueError(
            f"in_features={nvec * VEC_DIM} is not a multiple of {SUPERBLOCK}")
    n_sb = nvec // CODEWORDS_PER_SUPERBLOCK
    idx_bytes = _pack_codes_to_bytes(codes, k)
    if grid == "fp4" and scale_coding == SCALE_CODING_TWO_TIER:
        sc_bytes = _two_tier_scale_bytes(
            fields["scale_super"], fields["scale_sub"], n_sb)
        block = torch.cat([idx_bytes, sc_bytes], dim=-1)
    elif grid == "fp4":
        sc_bytes = _scale_plane_bytes(fields["scales"], n_sb)
        block = torch.cat([idx_bytes, sc_bytes], dim=-1)
    elif grid == "fp8":
        block = idx_bytes
    else:
        raise ValueError(f"unknown grid {grid!r}")
    ts = _serialized_type_size(k, grid, scale_coding)
    assert block.shape[-1] == ts, (
        f"type_size mismatch: packed {block.shape[-1]} bytes/superblock, "
        f"expected {ts} for k={k} grid={grid} scale_coding={scale_coding}")
    return block.reshape(rows, n_sb * ts).contiguous()


def nvfp4_cb_unpack(packed: torch.Tensor, k: int, grid: str, mode: str,
                    shape: tuple[int, ...],
                    codebook: torch.Tensor | tuple | None = None,
                    scales: torch.Tensor | None = None,
                    scale_coding: str = SCALE_CODING_V1) -> dict:
    """Inverse of :func:`nvfp4_cb_assemble_bytes`: byte tensor -> VQ ``fields``
    ready for :func:`nvfp4_cb_reconstruct`.

    fp4 scales are recovered from the packed scale section — the v1 e4m3
    plane by default; pass ``scale_coding="two_tier"`` for layout-v2 bytes
    (absence of the scheme's ``scale_coding`` key means v1, so old artifacts
    decode unchanged, forever). fp8 has no scale plane on disk — pass the
    per-output-channel ``scales`` tensor (``<name>.weight_scale``)
    explicitly. ``codebook`` (the resolved learned / lattice table) is echoed
    into the fields so reconstruct uses the exact table the packer encoded
    against.
    """
    k = int(k)
    in_f = int(shape[-1])
    if in_f % SUPERBLOCK != 0:
        raise ValueError(
            f"in_features={in_f} must be a multiple of {SUPERBLOCK}")
    rows = int(packed.shape[0])
    n_sb = in_f // SUPERBLOCK
    ts = _serialized_type_size(k, grid, scale_coding)
    if tuple(packed.shape) != (rows, n_sb * ts):
        raise ValueError(
            f"packed shape {tuple(packed.shape)} != expected "
            f"{(rows, n_sb * ts)} for k={k} grid={grid} in_features={in_f} "
            f"scale_coding={scale_coding}")
    block = packed.reshape(rows, n_sb, ts)
    index_bytes = INDEX_BYTES_PER_K * k
    codes = _unpack_bytes_to_codes(block[..., :index_bytes], k)
    nvec = in_f // VEC_DIM

    if mode == "full":
        fields: dict = {"indices": codes.reshape(rows, nvec)}
    elif mode == "signed":
        sign_byte = codes & 0xFF
        mag = codes >> VEC_DIM
        shifts = torch.arange(VEC_DIM, device=codes.device)
        neg = ((sign_byte.unsqueeze(-1) >> shifts) & 1).bool()  # (rows,nvec,8)
        signs = torch.where(neg, -1.0, 1.0).reshape(rows, in_f)
        fields = {"indices": mag.reshape(rows, nvec), "signs": signs}
    elif mode == "product":
        n_sub = family_for(grid, "product").n_sub
        bits = subtable_bit_widths(k, "product", n_sub)
        subs, off = [], 0
        for i in range(n_sub):
            subs.append((codes >> off) & ((1 << bits[i]) - 1))
            off += bits[i]
        fields = {"indices": torch.stack(subs, dim=-1).reshape(
            rows, nvec, n_sub)}
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if grid == "fp4" and scale_coding == SCALE_CODING_TWO_TIER:
        super_e, sub_c, composed = _two_tier_scale_unpack(
            block[..., index_bytes:
                  index_bytes + SCALE_PLANE_BYTES[
                      ("fp4", SCALE_CODING_TWO_TIER)
                  ]])
        fields["scales"] = composed
        fields["scale_super"] = super_e.to(torch.uint8)
        fields["scale_sub"] = sub_c
        fields["scale_coding"] = SCALE_CODING_TWO_TIER
    elif grid == "fp4":
        scale_plane_bytes = SCALE_PLANE_BYTES[("fp4", SCALE_CODING_V1)]
        sc = block[..., index_bytes:index_bytes + scale_plane_bytes].reshape(
            rows, n_sb * FP4_SCALE_GROUPS_PER_SUPERBLOCK
        )
        fields["scales"] = sc.contiguous().view(_E4M3).to(torch.float32)
    else:
        if scales is None:
            raise ValueError(
                "fp8 CB has no on-disk scale plane; pass the per-channel "
                "`scales` (<name>.weight_scale) to unpack")
        fields["scales"] = scales
    fields["shape"] = tuple(int(d) for d in shape)
    if codebook is not None:
        fields["codebook"] = codebook
    return fields


def nvfp4_cb_pack(w: torch.Tensor, k: int, *, grid: str = "fp4",
                  mode: str = "product",
                  col_weights: torch.Tensor | None = None,
                  codebook: torch.Tensor | tuple | None = None,
                  scale_sweep: bool = True,
                  scale_coding: str = SCALE_CODING_V1,
                  encode_tier: str | None = None,
                  warm_scale_state: dict[str, torch.Tensor] | None = None,
                  ldlq: bool = False,
                  activation_rows: torch.Tensor | Sequence[torch.Tensor] | None = None,
                  ) -> tuple[torch.Tensor, dict]:
    """Quantize + bit-pack a weight in one call (mirrors ``gguf_pack``).

    Returns ``(packed uint8 (rows, bytes_per_row), fields)``; ``fields``
    carries ``scales`` (per-channel fp8 scale plane the exporter ships
    separately) and the resolved ``codebook``.
    """
    fields = nvfp4_cb_fields(w, k, grid=grid, mode=mode,
                             col_weights=col_weights, codebook=codebook,
                             scale_sweep=scale_sweep,
                             scale_coding=scale_coding,
                             encode_tier=encode_tier,
                             warm_scale_state=warm_scale_state)
    if ldlq:
        if col_weights is None:
            raise ValueError("LDLQ CB packing requires activation-weighted col_weights")
        if activation_rows is None:
            raise ValueError("LDLQ CB packing requires calibration activation rows")
        fields = ldlq_reassign_cb_fields(
            w,
            fields,
            col_weights,
            activation_rows,
            grid=grid,
            mode=mode,
        )
    packed = nvfp4_cb_assemble_bytes(fields, k, grid=grid, mode=mode)
    return packed, fields
