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
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

VEC_DIM = 8
SUPERBLOCK = 256
FP4_GROUP = 16
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

_DATA = Path(__file__).resolve().parent / "data" / "nvfp4_cb_lattices.pt"
_LATTICE_SEED = 1234
_LATTICE_SAMPLES = 1 << 17
_LATTICE_ITERS = 12

# E2M1: {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _product_n_sub(grid: str) -> int:
    """Sub-vectors per 8-dim vector in product mode. fp4 splits into two
    4-dim halves; fp8 into four 2-dim sub-vectors so every FP8_CB rung's
    sub-table stays flat-searchable (k=36..48 -> 9..12-bit sub-tables)."""
    return 2 if grid == "fp4" else 4


def _bit_split(k: int, n_sub: int) -> tuple[int, ...]:
    """Split k index bits across n_sub sub-tables as evenly as possible
    (ceil-first, so n_sub=2 keeps the historical (ceil, floor) split)."""
    base, extra = divmod(k, n_sub)
    return tuple(base + (1 if i < extra else 0) for i in range(n_sub))


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
        if codebook is None:
            n_sub = _product_n_sub(grid)
            sub_dim = VEC_DIM // n_sub
            tables = tuple(fixed_lattice(bits, grid, sub_dim)
                           for bits in _bit_split(k, n_sub))
        else:
            tables = tuple(codebook)
            if VEC_DIM % len(tables) != 0:
                raise ValueError(
                    f"product codebook count {len(tables)} must divide "
                    f"{VEC_DIM}")
        return tuple(t.to(device, torch.float32) for t in tables)
    if mode == "signed":
        m = k - VEC_DIM                       # 8 explicit sign bits inside k
        if m < 1:
            raise ValueError(f"signed mode needs k > {VEC_DIM} (got k={k})")
        if codebook is None:
            cb = fixed_lattice(m, grid, VEC_DIM, positive=True)
        else:
            cb = codebook
        cb = cb.to(device, torch.float32)
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


def _mode_encode(vectors: torch.Tensor, mode: str, cb, wq) -> dict:
    """VQ-assign scaled ``vectors`` (nvec, 8) under one mode. Returns per-mode
    index fields ({"idx": (nvec,) or (nvec, n_sub)}, + "signs" for signed)."""
    if mode == "full":
        return {"idx": _vq_assign(vectors, cb, wq)}
    if mode == "signed":
        # Exactly separable under weighted L2: for any magnitude codeword
        # c >= 0, sum_j w_j (x_j - s_j c_j)^2 is minimized over s_j in {+-1}
        # by s_j = sign(x_j) (the cross-term -2 w_j s_j x_j c_j is largest
        # when s_j x_j >= 0, independent of which codeword is chosen), and at
        # that sign the objective equals sum_j w_j (|x_j| - c_j)^2. So the
        # weighted argmin over |x| plus signs = sign(x) IS the joint optimum
        # — no sign x magnitude search needed. Zero-safe: sign(0) -> +1.
        return {"idx": _vq_assign(vectors.abs(), cb, wq),
                "signs": torch.where(vectors < 0, -1.0, 1.0)}
    n_sub = len(cb)
    sub_dim = VEC_DIM // n_sub
    idxs = []
    for i, table in enumerate(cb):
        xs = vectors[:, i * sub_dim:(i + 1) * sub_dim]
        ws = wq[:, i * sub_dim:(i + 1) * sub_dim] if wq is not None else None
        idxs.append(_vq_assign(xs, table, ws))
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
                    s: torch.Tensor, grid: str, mode: str, cb):
    """Encode ``w2d`` at per-group scale ``s`` and score the WEIGHTED
    reconstruction error in the ORIGINAL weight domain (so the scale choice
    is judged on real error, not scaled-domain error). Returns
    (err_group (rows, ngroups), enc, grid_decode (rows, in))."""
    rows, in_f = w2d.shape
    pes = _per_element_scale(s, grid, in_f)              # (rows, in)
    pes_vec = pes.reshape(-1, VEC_DIM)                   # (nvec, 8)
    wvec = w2d.reshape(-1, VEC_DIM)
    x = wvec / pes_vec
    enc = _mode_encode(x, mode, cb, wq)
    dec = _mode_decode(enc, mode, cb)                    # (nvec, 8) grid
    recon = dec * pes_vec                                # original domain
    err = (recon - wvec).pow(2)
    if wq is not None:
        err = err * wq
    err_group = _group_reduce(err.reshape(rows, in_f), grid)
    return err_group, enc, dec.reshape(rows, in_f)


def _sweep_encode(w2d: torch.Tensor, grid: str, mode: str, cb,
                  wq: torch.Tensor | None):
    """Joint scale sweep + WLS-refit fixed point (mirrors _grid_fields). Picks
    the per-group scale minimizing weighted real error over the E4M3-legal
    candidate grid, then refines with continuous WLS refits accepted per group
    only when strictly better. Returns (best_scales (rows, ng), enc)."""
    rows, in_f = w2d.shape
    amax = _group_amax(w2d, grid)                        # (rows, ng)
    cands = _candidate_scales(amax, grid, _SCALE_SWEEP_CANDIDATES)
    best_err, _, _ = _eval_candidate(w2d, wq, cands[0], grid, mode, cb)
    best_s = cands[0]
    for si in range(1, cands.shape[0]):
        err, _, _ = _eval_candidate(w2d, wq, cands[si], grid, mode, cb)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, cands[si], best_s)
    # WLS refit: optimal continuous scale s* = sum(w g v) / sum(w g^2) per
    # group at the current (fixed) assignment, snapped legal, accepted per
    # group only when it strictly lowers real error.
    for _ in range(_SCALE_SWEEP_REFIT_ITERS):
        err_cur, _, g = _eval_candidate(w2d, wq, best_s, grid, mode, cb)
        wcol = wq.reshape(rows, in_f) if wq is not None else torch.ones_like(g)
        num = _group_reduce(wcol * g * w2d, grid)
        den = _group_reduce(wcol * g * g, grid)
        s_star = _snap_scale(torch.where(den > 0, num / den.clamp_min(1e-30),
                                         best_s), grid)
        err_star, _, _ = _eval_candidate(w2d, wq, s_star, grid, mode, cb)
        better = err_star < err_cur
        best_s = torch.where(better, s_star, best_s)
    _, enc, _ = _eval_candidate(w2d, wq, best_s, grid, mode, cb)
    return best_s, enc


def _fields_block(w2d: torch.Tensor, k: int, grid: str, mode: str,
                  cb, cw2d: torch.Tensor | None, scale_sweep: bool) -> dict:
    rows, in_f = w2d.shape
    nvec_per_row = in_f // VEC_DIM
    wq = _col_weight_vectors(cw2d) if cw2d is not None else None
    if scale_sweep:
        scales, enc = _sweep_encode(w2d, grid, mode, cb, wq)
    else:
        vectors, scales, _ = _scale_and_vectorize(w2d, grid)
        enc = _mode_encode(vectors, mode, cb, wq)
    out = _enc_to_fields(enc, mode, cb, rows, in_f, nvec_per_row)
    out["scales"] = scales
    return out


def nvfp4_cb_fields(w: torch.Tensor, k: int, *, grid: str = "fp4",
                    mode: str = "product",
                    col_weights: torch.Tensor | None = None,
                    codebook: torch.Tensor | tuple | None = None,
                    scale_sweep: bool = True) -> dict:
    """Quantize ``w`` (2-D or 3-D stacked experts) into VQ fields.

    ``scale_sweep`` (default True) jointly optimizes the per-group scale over
    the E4M3-legal candidate grid (IQ-rendering parity); set False for the
    one-shot amax/grid-max scale (A/B and the pre-c3f8c6d rendering).

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
        out = _fields_block(w2d, k, grid, mode, cb, cw2d, scale_sweep)
    else:
        parts = []
        for a in range(0, rows, row_step):
            b = min(rows, a + row_step)
            cw = cw2d[a:b] if cw2d is not None else None
            parts.append(
                _fields_block(w2d[a:b], k, grid, mode, cb, cw, scale_sweep))
        out = {key: torch.cat([p[key] for p in parts], dim=0)
               for key in parts[0]}
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
                      scale_sweep: bool = True):
    """Single-source emulation closure ``(w, col_weights=None) -> w_hat``
    used by both cost and (Milestone B) the packer. ``scale_sweep`` defaults
    True (joint scale search, IQ-rendering parity)."""
    def f(w: torch.Tensor, col_weights: torch.Tensor | None = None
          ) -> torch.Tensor:
        fields = nvfp4_cb_fields(w, k, grid=grid, mode=mode,
                                 col_weights=col_weights,
                                 scale_sweep=scale_sweep)
        return nvfp4_cb_reconstruct(fields, k, grid=grid, mode=mode).to(w.dtype)
    return f


# ---------------------------------------------------------------------------
# Milestone B — byte packers (export path). Bit-exact on-disk layout:
# docs/nvfp4-cb-plan/format-pipeline.md §1 / docs/nvfp4-cb-plan/LAYOUT.md.
#
# Per 256-weight superblock along the input dim:
#   * 4k index bytes — 32 k-bit codewords (one per 8-dim vector), LSB-first;
#   * fp4 only: 16 E4M3 group scale bytes (identical to NVFP4's block scales).
# type_size = 4k + 16 (fp4) / 4k (fp8), integer for every k. The packed tensor
# is 2-D uint8 (rows, bytes_per_row) — NEVER a flat 1-D buffer (the GGUF
# lesson: a flat store loses the logical row/superblock structure the reader
# and serving kernel index into). fp8 ships NO scale plane in the weight bytes;
# its per-output-channel fp32 scales are a separate ``<name>.weight_scale``.
# ---------------------------------------------------------------------------

def _type_size(k: int, grid: str) -> int:
    """Bytes per 256-weight superblock (§1.1): 4k index bytes + 16 E4M3 scale
    bytes (fp4) / 4k index bytes only (fp8)."""
    return 4 * int(k) + (16 if grid == "fp4" else 0)


def nvfp4_cb_type_size(k: int, grid: str = "fp4") -> int:
    """Public: on-disk bytes per 256-weight superblock for a CB rung."""
    return _type_size(k, grid)


def _vector_codes(fields: dict, k: int, grid: str, mode: str) -> torch.Tensor:
    """Per-8-weight-vector k-bit codeword (rows, nvec_per_row), int64.

    Bit layout inside the k-bit field (LSB-first), per §1.1 + the PLAN's
    product decomposition:
      * full   — the codebook index itself (k bits);
      * product— the n_sub sub-indices contiguous, sub0 in the low bits
                 (bit widths ``_bit_split(k, n_sub)``, ceil-first);
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
    # product: idx is (rows, nvec, n_sub)
    n_sub = idx.shape[-1]
    bits = _bit_split(k, n_sub)
    code = torch.zeros(idx.shape[:-1], dtype=torch.int64, device=idx.device)
    off = 0
    for i in range(n_sub):
        code = code | (idx[..., i].to(torch.int64) << off)
        off += bits[i]
    return code.reshape(rows, -1)


def _pack_codes_to_bytes(codes: torch.Tensor, k: int) -> torch.Tensor:
    """(rows, nvec) k-bit codewords -> (rows, n_superblocks, 4k) uint8.

    Each 256-weight superblock is 32 codewords = 32k bits = 4k bytes and is
    byte-aligned, so the codewords pack contiguously LSB-first (codeword c
    occupies stream bits [c*k, c*k+k), its own LSB first; bytes fill LSB-first).
    """
    rows, nvec = codes.shape
    n_sb = nvec // 32
    shifts = torch.arange(k, device=codes.device)
    bitstream = (codes.unsqueeze(-1) >> shifts) & 1              # (rows, nvec,k)
    bitstream = bitstream.reshape(rows, n_sb, 4 * k, 8).to(torch.int64)
    wt = 1 << torch.arange(8, device=codes.device)
    return (bitstream * wt).sum(dim=-1).to(torch.uint8)         # (rows,n_sb,4k)


def _unpack_bytes_to_codes(idx_bytes: torch.Tensor, k: int) -> torch.Tensor:
    """(rows, n_sb, 4k) uint8 -> (rows, n_sb*32) int64 codewords (inverse of
    ``_pack_codes_to_bytes``)."""
    rows, n_sb, _ = idx_bytes.shape
    bshift = torch.arange(8, device=idx_bytes.device)
    bits = (idx_bytes.to(torch.int64).unsqueeze(-1) >> bshift) & 1
    bits = bits.reshape(rows, n_sb, 32, k)                      # k bits/codeword
    kshift = torch.arange(k, device=idx_bytes.device)
    codes = (bits << kshift).sum(dim=-1)                        # (rows,n_sb,32)
    return codes.reshape(rows, n_sb * 32)


def _scale_plane_bytes(scales: torch.Tensor, n_sb: int) -> torch.Tensor:
    """fp4 scale plane -> (rows, n_sb, 16) uint8. ``scales`` (rows, in//16) are
    already E4M3-exact (snapped by the encoder), so the E4M3 byte view is
    lossless."""
    rows = scales.shape[0]
    s = scales.reshape(rows, n_sb, FP4_GROUP).to(_E4M3)
    return s.contiguous().view(torch.uint8)


def nvfp4_cb_assemble_bytes(fields: dict, k: int, grid: str = "fp4",
                            mode: str = "product") -> torch.Tensor:
    """Bit-pack VQ ``fields`` into the §1 on-disk byte layout.

    Returns a 2-D uint8 tensor ``(rows, n_superblocks * type_size)`` on the
    fields' device. ``type_size == 4k + 16`` (fp4) / ``4k`` (fp8), asserted.
    """
    k = int(k)
    codes = _vector_codes(fields, k, grid, mode)                # (rows, nvec)
    rows, nvec = codes.shape
    if nvec % 32 != 0:
        raise ValueError(
            f"in_features={nvec * VEC_DIM} is not a multiple of {SUPERBLOCK}")
    n_sb = nvec // 32
    idx_bytes = _pack_codes_to_bytes(codes, k)                  # (rows,n_sb,4k)
    if grid == "fp4":
        sc_bytes = _scale_plane_bytes(fields["scales"], n_sb)   # (rows,n_sb,16)
        block = torch.cat([idx_bytes, sc_bytes], dim=-1)
    elif grid == "fp8":
        block = idx_bytes
    else:
        raise ValueError(f"unknown grid {grid!r}")
    ts = _type_size(k, grid)
    assert block.shape[-1] == ts, (
        f"type_size mismatch: packed {block.shape[-1]} bytes/superblock, "
        f"expected {ts} for k={k} grid={grid}")
    return block.reshape(rows, n_sb * ts).contiguous()


def nvfp4_cb_unpack(packed: torch.Tensor, k: int, grid: str, mode: str,
                    shape: tuple[int, ...],
                    codebook: torch.Tensor | tuple | None = None,
                    scales: torch.Tensor | None = None) -> dict:
    """Inverse of :func:`nvfp4_cb_assemble_bytes`: byte tensor -> VQ ``fields``
    ready for :func:`nvfp4_cb_reconstruct`.

    fp4 scales are recovered from the packed scale plane. fp8 has no scale
    plane on disk — pass the per-output-channel ``scales`` tensor
    (``<name>.weight_scale``) explicitly. ``codebook`` (the resolved learned /
    lattice table) is echoed into the fields so reconstruct uses the exact
    table the packer encoded against.
    """
    k = int(k)
    in_f = int(shape[-1])
    if in_f % SUPERBLOCK != 0:
        raise ValueError(
            f"in_features={in_f} must be a multiple of {SUPERBLOCK}")
    rows = int(packed.shape[0])
    n_sb = in_f // SUPERBLOCK
    ts = _type_size(k, grid)
    if tuple(packed.shape) != (rows, n_sb * ts):
        raise ValueError(
            f"packed shape {tuple(packed.shape)} != expected "
            f"{(rows, n_sb * ts)} for k={k} grid={grid} in_features={in_f}")
    block = packed.reshape(rows, n_sb, ts)
    codes = _unpack_bytes_to_codes(block[..., :4 * k], k)
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
        n_sub = _product_n_sub(grid)
        bits = _bit_split(k, n_sub)
        subs, off = [], 0
        for i in range(n_sub):
            subs.append((codes >> off) & ((1 << bits[i]) - 1))
            off += bits[i]
        fields = {"indices": torch.stack(subs, dim=-1).reshape(
            rows, nvec, n_sub)}
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if grid == "fp4":
        sc = block[..., 4 * k:4 * k + FP4_GROUP].reshape(rows, n_sb * FP4_GROUP)
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
                  scale_sweep: bool = True) -> tuple[torch.Tensor, dict]:
    """Quantize + bit-pack a weight in one call (mirrors ``gguf_pack``).

    Returns ``(packed uint8 (rows, bytes_per_row), fields)``; ``fields``
    carries ``scales`` (per-channel fp8 scale plane the exporter ships
    separately) and the resolved ``codebook``.
    """
    fields = nvfp4_cb_fields(w, k, grid=grid, mode=mode,
                             col_weights=col_weights, codebook=codebook,
                             scale_sweep=scale_sweep)
    packed = nvfp4_cb_assemble_bytes(fields, k, grid=grid, mode=mode)
    return packed, fields
