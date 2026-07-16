"""NVFP4-CB / FP8-CB codebook format tests (Milestone A emulation +
Milestone B byte packers / exporter)."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import layer_config as lc
from prismaquant import nvfp4_cb_formats as cb

_NVFP4_KS = list(range(12, 25))
_FP8_KS = [36, 40, 44, 48]
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _wmse(w, r, cw=None):
    e = (w - r).float().pow(2)
    if cw is not None:
        e = e * cw
    return float(e.mean())


# (a) effective-bits accounting, exact, for every rung.
@pytest.mark.parametrize("k", _NVFP4_KS)
def test_nvfp4_cb_effective_bits_exact(k):
    spec = fr.get_format(f"NVFP4_CB_K{k}")
    assert spec.effective_bits == pytest.approx(k / 8 + 0.5, abs=1e-9)
    assert spec.effective_bits_for_shape((64, 2048)) == pytest.approx(
        k / 8 + 0.5, abs=1e-9)
    assert spec.memory_bytes_for_shape((64, 2048)) == 64 * (2048 // 256) * (
        4 * k + 16)


@pytest.mark.parametrize("k", _FP8_KS)
def test_fp8_cb_effective_bits_exact(k):
    spec = fr.get_format(f"FP8_CB_K{k}")
    # Registry body = index stream only, k/8 bpw exact (no group scale plane).
    # The per-output-channel fp32 scale is the authoritative footprint's
    # concern (nvfp4_cb_footprint), not the single-scale FormatSpec.
    assert spec.effective_bits == pytest.approx(k / 8, abs=1e-9)
    assert spec.effective_bits_for_shape((64, 2048)) == pytest.approx(
        k / 8, abs=1e-9)
    assert spec.memory_bytes_for_shape((128, 256)) == 128 * (256 // 256) * (
        4 * k)


# (b) decode validity: every reconstructed value == a grid point * group scale.
@pytest.mark.parametrize("mode", ["full", "product"])
def test_decode_on_grid_times_scale(mode):
    torch.manual_seed(0)
    w = torch.randn(64, 512)
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode=mode)
    recon = cb.nvfp4_cb_reconstruct(fields, 12, grid="fp4", mode=mode)
    pes = cb._per_element_scale(fields["scales"], "fp4", 512)
    q = recon / pes
    grid = cb._e2m1_grid("cpu")
    dist = (q.unsqueeze(-1) - grid).abs().min(dim=-1).values
    assert float(dist.max()) < 1e-5


# (c) determinism: bit-identical, eager, per device.
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["full", "product"])
def test_determinism_per_device(device, mode):
    torch.manual_seed(3)
    w = torch.randn(48, 512, device=device)
    qdq = cb.make_nvfp4_cb_qdq(12, "fp4", mode)
    a, b = qdq(w), qdq(w)
    assert torch.equal(a, b)


# (d) col_weights changes the assignment and reduces weighted MSE.
def test_col_weights_reduces_weighted_mse():
    torch.manual_seed(1)
    w = torch.randn(64, 512)
    cw = torch.rand(512) + 0.05
    f0 = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="full")
    fw = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="full", col_weights=cw)
    assert not torch.equal(f0["indices"], fw["indices"])
    r0 = cb.nvfp4_cb_reconstruct(f0, 12, grid="fp4", mode="full")
    rw = cb.nvfp4_cb_reconstruct(fw, 12, grid="fp4", mode="full")
    assert _wmse(w, rw, cw) <= _wmse(w, r0, cw) + 1e-9


# (e) learned codebook (k=12, full) beats-or-ties the fixed lattice.
def test_learned_codebook_beats_fixed():
    torch.manual_seed(2)
    w = torch.randn(96, 512)
    cw = torch.rand(512) + 0.05
    vecs, _, _ = cb._scale_and_vectorize(w, "fp4")
    learned = cb.learn_codebook(vecs, 12, grid="fp4", iters=8)
    f_fix = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="full", col_weights=cw)
    f_lrn = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="full",
                              col_weights=cw, codebook=learned)
    r_fix = cb.nvfp4_cb_reconstruct(f_fix, 12, grid="fp4", mode="full")
    r_lrn = cb.nvfp4_cb_reconstruct(f_lrn, 12, grid="fp4", mode="full",
                                    codebook=learned)
    assert _wmse(w, r_lrn, cw) <= _wmse(w, r_fix, cw) + 1e-9
    # learned codebook is grid-valued (E2M1) so a decoded tile stays NVFP4.
    grid = cb._e2m1_grid("cpu")
    dist = (learned.unsqueeze(-1) - grid).abs().min(dim=-1).values
    assert float(dist.max()) < 1e-5


# scale sweep: joint per-group scale search (IQ-rendering parity).
def _sweep_total_err(w, cw, grid, mode, k, scale):
    cw2d = torch.broadcast_to(cw, w.shape).contiguous()
    wq = cb._col_weight_vectors(cw2d)
    C = cb._resolve_codebook(k, grid, mode, None, w.device)
    err, _, _ = cb._eval_candidate(w, wq, scale, grid, mode, C)
    return float(err.sum())


@pytest.mark.parametrize("grid,mode,k", [
    ("fp4", "full", 13), ("fp4", "product", 14), ("fp4", "signed", 14),
    ("fp8", "product", 40),
])
def test_scale_sweep_never_worse_than_one_shot(grid, mode, k):
    torch.manual_seed(0)
    w = torch.randn(64, 512) * 0.3
    cw = torch.rand(512) + 0.05
    C = cb._resolve_codebook(k, grid, mode, None, w.device)
    amax = cb._group_amax(w, grid)
    cands = cb._candidate_scales(amax, grid, cb._SCALE_SWEEP_CANDIDATES)
    # candidate 0 is the amax/grid-max one-shot; it is IN the sweep set.
    one_shot = _sweep_total_err(w, cw, grid, mode, k, cands[0])
    fields = cb.nvfp4_cb_fields(w, k, grid=grid, mode=mode, col_weights=cw,
                                scale_sweep=True)
    swept = _sweep_total_err(w, cw, grid, mode, k, fields["scales"])
    assert swept <= one_shot + 1e-4


@pytest.mark.parametrize("mode,k", [
    ("full", 13), ("product", 14), ("signed", 14)])
def test_scale_sweep_fp4_scales_are_e4m3_legal(mode, k):
    torch.manual_seed(1)
    w = torch.randn(48, 512) * 0.3
    cw = torch.rand(512) + 0.05
    fields = cb.nvfp4_cb_fields(w, k, grid="fp4", mode=mode, col_weights=cw,
                                scale_sweep=True)
    s = fields["scales"]
    assert torch.equal(s, s.to(torch.float8_e4m3fn).to(torch.float32))
    assert bool((s > 0).all())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["full", "product", "signed"])
def test_scale_sweep_determinism(device, mode):
    torch.manual_seed(2)
    w = (torch.randn(48, 512, device=device) * 0.3)
    qdq = cb.make_nvfp4_cb_qdq(14, "fp4", mode, scale_sweep=True)
    assert torch.equal(qdq(w), qdq(w))


def test_scale_sweep_toggle_changes_output_and_default_on():
    torch.manual_seed(3)
    w = torch.randn(64, 512) * 0.3
    swept = cb.make_nvfp4_cb_qdq(14, "fp4", "product", scale_sweep=True)(w)
    one_shot = cb.make_nvfp4_cb_qdq(14, "fp4", "product", scale_sweep=False)(w)
    default = cb.make_nvfp4_cb_qdq(14, "fp4", "product")(w)
    assert torch.equal(default, swept)          # default is scale_sweep=True
    assert not torch.equal(swept, one_shot)     # the sweep actually moved


def test_scale_sweep_decode_validity_holds():
    # swept scales are still one E4M3 value per group-16, so decode == grid*scale
    torch.manual_seed(4)
    w = torch.randn(64, 512) * 0.3
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product",
                                scale_sweep=True)
    recon = cb.nvfp4_cb_reconstruct(fields, 12, grid="fp4", mode="product")
    pes = cb._per_element_scale(fields["scales"], "fp4", 512)
    grid = cb._e2m1_grid("cpu")
    dist = ((recon / pes).unsqueeze(-1) - grid).abs().min(dim=-1).values
    assert float(dist.max()) < 1e-5


def test_scale_sweep_does_not_change_effective_bits():
    for k in (12, 14):
        spec = fr.get_format(f"NVFP4_CB_K{k}")
        assert spec.effective_bits == pytest.approx(k / 8 + 0.5, abs=1e-9)


# signed mode: 8 explicit sign bits + (k-8)-bit positive-grid magnitude index.
_SIGNED_KS = [13, 14, 15, 16]


@pytest.mark.parametrize("k", _SIGNED_KS)
def test_signed_effective_bits_exact(k):
    spec = fr.get_format(f"NVFP4_CB_S{k}")
    assert spec.effective_bits == pytest.approx(k / 8 + 0.5, abs=1e-9)
    assert spec.effective_bits_for_shape((64, 2048)) == pytest.approx(
        k / 8 + 0.5, abs=1e-9)


@pytest.mark.parametrize("k", _SIGNED_KS)
def test_signed_decode_on_pos_grid_times_scale(k):
    torch.manual_seed(8)
    w = torch.randn(48, 512)
    w[0, :16] = 0.0                       # zero coords: sign must be +1-safe
    fields = cb.nvfp4_cb_fields(w, k, grid="fp4", mode="signed")
    assert torch.equal(
        fields["signs"].abs(), torch.ones_like(fields["signs"]))
    recon = cb.nvfp4_cb_reconstruct(fields, k, grid="fp4", mode="signed")
    pes = cb._per_element_scale(fields["scales"], "fp4", 512)
    q = (recon / pes).abs()               # |value| on the positive half-grid
    pos = torch.tensor(cb._E2M1_VALUES)
    dist = (q.unsqueeze(-1) - pos).abs().min(dim=-1).values
    assert float(dist.max()) < 1e-5
    # magnitude codebook itself is non-negative and grid-valued
    mag = fields["codebook"]
    assert bool((mag >= 0).all())
    assert torch.equal(cb._snap_to_grid(mag, "fp4"), mag)


def test_signed_separable_encode_is_joint_optimum():
    # For c >= 0 the optimal sign is sign(x) independent of the codeword, so
    # weighted argmin over |x| + explicit signs must EXACTLY match the
    # exhaustive joint search over all 2^8 sign patterns x magnitudes.
    torch.manual_seed(9)
    w = torch.randn(8, 256)
    cwq = torch.rand(256) + 0.05
    # sign-separability is a property of the fixed-scale encode; pin the
    # one-shot scale so the joint search below sees the same scaled vectors.
    fields = cb.nvfp4_cb_fields(w, 13, grid="fp4", mode="signed",
                                col_weights=cwq, scale_sweep=False)
    mag = fields["codebook"]
    vecs, _, _ = cb._scale_and_vectorize(w, "fp4")
    wq = cb._col_weight_vectors(
        torch.broadcast_to(cwq, (8, 256)).reshape(8, 256))
    signs_all = torch.tensor(
        [[1.0 if (s >> j) & 1 == 0 else -1.0 for j in range(8)]
         for s in range(256)])
    joint = (signs_all.unsqueeze(1) * mag.unsqueeze(0)).reshape(-1, 8)
    idx_joint = cb._vq_assign(vecs, joint, wq)
    err_joint = (wq * (vecs - joint[idx_joint]).pow(2)).sum()
    rec_sep = mag[fields["indices"].reshape(-1)] * fields["signs"].reshape(
        -1, 8)
    err_sep = (wq * (vecs - rec_sep).pow(2)).sum()
    assert float(err_sep) <= float(err_joint) + 1e-3


def test_signed_extends_ladder_beyond_flat_ceiling():
    # k=15,16 have no flat-full twin (MAX_FLAT_K=14); signed reaches them
    # with tiny tables. At the one-shot scale signed edges product there
    # (the scale sweep narrows this to a wash — see the module note); the
    # durable invariant is that signed extends the exhaustive-optimal ladder.
    torch.manual_seed(10)
    w = torch.randn(128, 1024)
    cw = torch.rand(1024) + 0.05
    for k in (15, 16):
        rs = cb.make_nvfp4_cb_qdq(k, "fp4", "signed", scale_sweep=False)(w)
        rp = cb.make_nvfp4_cb_qdq(k, "fp4", "product", scale_sweep=False)(w)
        assert _wmse(w, rs, cw) <= _wmse(w, rp, cw) + 1e-9
        with pytest.raises(ValueError, match="infeasible"):
            cb.fixed_lattice(k, "fp4", 8)


@pytest.mark.parametrize("device", _DEVICES)
def test_signed_determinism_per_device(device):
    torch.manual_seed(11)
    w = torch.randn(48, 512, device=device)
    qdq = cb.make_nvfp4_cb_qdq(14, "fp4", "signed")
    assert torch.equal(qdq(w), qdq(w))


def test_signed_learned_magnitude_roundtrip():
    torch.manual_seed(12)
    w = torch.randn(96, 512)
    cw = torch.rand(512) + 0.05
    vecs, _, _ = cb._scale_and_vectorize(w, "fp4")
    mag = cb.learn_codebook(vecs.abs(), 6, grid="fp4", positive=True,
                            iters=6)
    assert mag.shape == (64, 8)
    assert bool((mag >= 0).all())
    f_fix = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="signed",
                               col_weights=cw)
    f_lrn = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="signed",
                               col_weights=cw, codebook=mag)
    r_fix = cb.nvfp4_cb_reconstruct(f_fix, 14, grid="fp4", mode="signed")
    r_lrn = cb.nvfp4_cb_reconstruct(f_lrn, 14, grid="fp4", mode="signed",
                                    codebook=mag)
    assert _wmse(w, r_lrn, cw) <= _wmse(w, r_fix, cw) + 1e-9
    # negative-entry codebooks are rejected (breaks sign optimality)
    with pytest.raises(ValueError, match="non-negative"):
        cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="signed",
                           codebook=mag - 1.0)


def test_signed_needs_more_than_sign_bits():
    with pytest.raises(ValueError, match="signed mode needs k"):
        cb.nvfp4_cb_fields(torch.randn(8, 256), 8, grid="fp4", mode="signed")


# FP8_CB: every registered rung is functional through the qdq closure —
# product mode splits into four 2-dim sub-vectors (9..12-bit sub-tables).
@pytest.mark.parametrize("k", _FP8_KS)
def test_fp8_cb_qdq_roundtrip_valid(k):
    torch.manual_seed(6)
    w = torch.randn(32, 512) * 0.3
    qdq = cb.make_nvfp4_cb_qdq(k, "fp8", "product")
    a, b = qdq(w), qdq(w)
    assert torch.equal(a, b)
    fields = cb.nvfp4_cb_fields(w, k, grid="fp8", mode="product")
    assert fields["indices"].shape[-1] == 4
    for table in fields["codebook"]:
        assert torch.equal(cb._snap_to_grid(table, "fp8"), table)
        assert table.shape == (1 << (k // 4), 2)
    recon = cb.nvfp4_cb_reconstruct(fields, k, grid="fp8", mode="product")
    # decode validity: recon / per-row scale recovers an E4M3 grid value
    # (up to the 1-ulp fp32 (c*s)/s roundtrip).
    pes = cb._per_element_scale(fields["scales"], "fp8", 512)
    q = recon / pes
    snap = cb._snap_to_grid(q, "fp8")
    rel = (q - snap).abs() / snap.abs().clamp_min(1e-12)
    assert float(rel.max()) < 1e-6


def test_product_n_sub4_determinism_pin():
    torch.manual_seed(7)
    w = torch.randn(24, 256) * 0.5
    f1 = cb.nvfp4_cb_fields(w, 40, grid="fp8", mode="product")
    f2 = cb.nvfp4_cb_fields(w, 40, grid="fp8", mode="product")
    assert torch.equal(f1["indices"], f2["indices"])
    assert torch.equal(f1["scales"], f2["scales"])


def test_bit_split_even_and_ceil_first():
    assert cb._bit_split(13, 2) == (7, 6)
    assert cb._bit_split(12, 2) == (6, 6)
    assert cb._bit_split(36, 4) == (9, 9, 9, 9)
    assert cb._bit_split(48, 4) == (12, 12, 12, 12)


# Lloyd at scale: the old dense one-hot path materialized (m, K) fp32 —
# 2M x 4096 = 32 GB — and would OOM here; index_add accumulation must not.
def test_lloyd_scale_no_dense_onehot():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device="cpu").manual_seed(11)
    vectors = torch.randn(2_000_000, 8, generator=gen).to(device)
    learned = cb.learn_codebook(vectors, 12, grid="fp4", iters=1)
    assert learned.shape == (4096, 8)
    grid = cb._e2m1_grid(device)
    dist = (learned.unsqueeze(-1) - grid).abs().min(dim=-1).values
    assert float(dist.max()) < 1e-5


# (f) product and full both reconstruct valid values at k=12.
def test_product_and_full_valid():
    torch.manual_seed(4)
    w = torch.randn(32, 768)
    for mode in ("full", "product"):
        r = cb.make_nvfp4_cb_qdq(12, "fp4", mode)(w)
        assert r.shape == w.shape
        assert torch.isfinite(r).all()


# (g) 3-D stacked experts round-trip with per-expert col_weights.
def test_stacked_experts_roundtrip():
    torch.manual_seed(5)
    w = torch.randn(3, 64, 256)
    cw = torch.rand(3, 1, 256) + 0.05
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product",
                                col_weights=cw)
    recon = cb.nvfp4_cb_reconstruct(fields, 12, grid="fp4", mode="product")
    assert recon.shape == w.shape
    assert fields["indices"].shape == (3 * 64, 256 // cb.VEC_DIM, 2)
    # each expert uses its own scale plane -> per-expert reconstruction differs.
    assert not torch.equal(recon[0], recon[1])


# (h) in_features % 256 != 0 raises.
def test_superblock_constraint():
    with pytest.raises(ValueError, match="multiple of 256"):
        cb.nvfp4_cb_fields(torch.randn(8, 300), 12)


def test_flat_k_ceiling_raises():
    with pytest.raises(ValueError, match="infeasible"):
        cb.fixed_lattice(15, "fp4", 8)


# (i) menu: all rungs register, resolve, sort by effective_bits.
def test_menu_registers_and_resolves():
    names = [f"NVFP4_CB_K{k}" for k in _NVFP4_KS] + \
            [f"NVFP4_CB_S{k}" for k in _SIGNED_KS] + \
            [f"FP8_CB_K{k}" for k in _FP8_KS]
    for name in names:
        spec = fr.get_format(name)
        assert spec is not None
        assert lc.canonicalize_format(name.lower()) == name
    # dict-form canonicalization (custom quant-config JSON shape).
    assert lc.canonicalize_format(
        {"data_type": "nvfp4_cb", "cb_k": 20}) == "NVFP4_CB_K20"
    assert lc.canonicalize_format(
        {"data_type": "nvfp4_cb", "cb_k": 14, "cb_mode": "signed"},
    ) == "NVFP4_CB_S14"
    assert lc.canonicalize_format(
        {"data_type": "fp8_cb", "cb_k": 44}) == "FP8_CB_K44"
    fam = [s for s in fr.list_formats() if s.family in ("nvfp4_cb", "fp8_cb")]
    assert len(fam) == len(names)
    bpps = [s.effective_bits for s in fam]
    assert bpps == sorted(bpps)


# ===========================================================================
# Milestone B — byte packers (format-pipeline.md §1 / LAYOUT.md contract).
# ===========================================================================

# (grid, mode, k): full/product/signed × both grids, the required matrix.
_PACK_CASES = [
    ("fp4", "product", 12), ("fp4", "product", 14), ("fp4", "product", 16),
    ("fp4", "full", 12), ("fp4", "full", 14), ("fp4", "full", 16),
    ("fp4", "signed", 16),
    ("fp8", "product", 36), ("fp8", "product", 44),
]


def _codebook_for(w, grid, mode, k):
    """Fixed lattice by default; an explicit grid-valued table where the flat
    lattice is infeasible (full mode, k>MAX_FLAT_K) so k=16-full is still
    covered. The pack/unpack round-trip is codebook-quality-agnostic, so a
    cheap snapped random table suffices (no need for a full 2^16 Lloyd)."""
    if mode == "full" and k > cb.MAX_FLAT_K:
        g = torch.Generator(device=w.device).manual_seed(k)
        raw = torch.randn(1 << k, cb.VEC_DIM, generator=g, device=w.device)
        return cb._snap_to_grid(raw, grid)
    return None


@pytest.mark.parametrize("grid,mode,k", _PACK_CASES)
def test_nvfp4_cb_type_size_and_packed_shape(grid, mode, k):
    ts = cb.nvfp4_cb_type_size(k, grid)
    assert ts == 4 * k + (16 if grid == "fp4" else 0)
    w = torch.randn(48, 512) * 0.3
    C = _codebook_for(w, grid, mode, k)
    fields = cb.nvfp4_cb_fields(w, k, grid=grid, mode=mode, codebook=C)
    packed = cb.nvfp4_cb_assemble_bytes(fields, k, grid, mode)
    assert packed.dtype == torch.uint8
    assert packed.ndim == 2                      # (rows, bytes) — never flat
    assert packed.shape == (48, (512 // 256) * ts)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("grid,mode,k", _PACK_CASES)
def test_nvfp4_cb_pack_unpack_matches_emulation(device, grid, mode, k):
    """THE contract: reconstruct(unpack(assemble(fields))) is BIT-IDENTICAL to
    the emulation qdq output the cost measurement scored — with scale_sweep on,
    on CPU and CUDA, for every mode×grid×k."""
    torch.manual_seed(0)
    w = torch.randn(48, 512, device=device) * 0.3
    cw = torch.rand(512, device=device) + 0.05
    C = _codebook_for(w, grid, mode, k)
    fields = cb.nvfp4_cb_fields(w, k, grid=grid, mode=mode, col_weights=cw,
                                codebook=C, scale_sweep=True)
    packed = cb.nvfp4_cb_assemble_bytes(fields, k, grid, mode)
    assert packed.device == w.device
    scales = fields["scales"] if grid == "fp8" else None
    up = cb.nvfp4_cb_unpack(packed, k, grid, mode, tuple(w.shape),
                            codebook=fields["codebook"], scales=scales)
    rec = cb.nvfp4_cb_reconstruct(up, k, grid=grid, mode=mode).to(w.dtype)
    emu = cb.nvfp4_cb_reconstruct(fields, k, grid=grid, mode=mode).to(w.dtype)
    assert torch.equal(rec, emu)


def test_nvfp4_cb_assemble_asserts_type_size():
    # A tampered fields dict whose scale plane is the wrong width must trip the
    # type_size assert rather than silently emit off-layout bytes.
    w = torch.randn(16, 512) * 0.3
    fields = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product")
    fields["scales"] = fields["scales"][:, :-1]      # drop one group scale
    with pytest.raises(Exception):
        cb.nvfp4_cb_assemble_bytes(fields, 14, "fp4", "product")


def test_nvfp4_cb_unpack_fp8_requires_scales():
    w = torch.randn(16, 512) * 0.3
    fields = cb.nvfp4_cb_fields(w, 40, grid="fp8", mode="product")
    packed = cb.nvfp4_cb_assemble_bytes(fields, 40, "fp8", "product")
    with pytest.raises(ValueError, match="no on-disk scale plane"):
        cb.nvfp4_cb_unpack(packed, 40, "fp8", "product", tuple(w.shape),
                           codebook=fields["codebook"])


def test_nvfp4_cb_pack_stacked_experts():
    torch.manual_seed(1)
    w = torch.randn(3, 32, 256) * 0.3
    cw = torch.rand(3, 1, 256) + 0.05
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product",
                                col_weights=cw)
    packed = cb.nvfp4_cb_assemble_bytes(fields, 12, "fp4", "product")
    assert packed.shape == (3 * 32, (256 // 256) * cb.nvfp4_cb_type_size(
        12, "fp4"))
    up = cb.nvfp4_cb_unpack(packed, 12, "fp4", "product", tuple(w.shape),
                            codebook=fields["codebook"])
    rec = cb.nvfp4_cb_reconstruct(up, 12, grid="fp4", mode="product").to(
        w.dtype)
    emu = cb.nvfp4_cb_reconstruct(fields, 12, grid="fp4",
                                  mode="product").to(w.dtype)
    assert torch.equal(rec, emu)


# ===========================================================================
# Milestone B — exporter (prismaquant.export_nvfp4_cb). CPU-only for
# bit-exactness (learned Lloyd / VQ argmin ties are device-dependent).
# ===========================================================================

# Mandated scratch root — never /tmp (CLAUDE.md landmine).
_EXPORT_ROOT = Path("/home/rob/dq-runs/nvfp4-cb-phase0/export-test/pytest")


@pytest.fixture
def export_dir():
    _EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=_EXPORT_ROOT))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _tiny_model(mdl: Path, in_f: int = 256):
    """2-layer synthetic HF dir: two 256-in Linears + a norm sidecar."""
    from safetensors.torch import save_file

    mdl.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    tens = {
        "model.layers.0.mlp.gate_proj.weight":
            (torch.randn(128, in_f) * 0.3).to(torch.bfloat16),
        "model.layers.1.mlp.gate_proj.weight":
            (torch.randn(128, in_f) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(in_f, dtype=torch.bfloat16),
    }
    save_file(tens, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(
        json.dumps({"architectures": ["Tiny"], "hidden_size": in_f}))
    return tens


def _write_assignment(path: Path, mapping: dict):
    path.write_text(json.dumps(mapping))


@pytest.mark.parametrize("source", ["lattice", "learned"])
def test_exporter_roundtrip_equals_emulation(export_dir, source):
    from safetensors.torch import load_file

    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl, out = export_dir / "model", export_dir / "out"
    tens = _tiny_model(mdl)
    assign = {
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
        "model.layers.1.mlp.gate_proj": {"data_type": "fp8_cb", "cb_k": 40},
    }
    apath = export_dir / "assign.json"
    _write_assignment(apath, assign)
    cw = {q: torch.rand(256) + 0.05 for q in assign}
    spec = ({"source": "learned", "train": True}
            if source == "learned" else {"source": "lattice"})

    counts = export_nvfp4_cb(mdl, apath, out, cw,
                             shared_codebook_spec=spec, device="cpu")
    assert counts["NVFP4_CB_K16"] == 1 and counts["FP8_CB_K40"] == 1

    qc = json.loads((out / "quant_config.json").read_text())
    assert qc["quant_method"] == "prismaquant"
    # non-target norm copied verbatim; config.json copied + pointer injected.
    ot = load_file(str(out / "model.safetensors"))
    assert "model.norm.weight" in ot
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"]["quant_method"] == "prismaquant"

    for g in qc["config_groups"].values():
        s = g["scheme"]
        grid, mode, k = s["grid"], s["mode"], s["k"]
        ref = s["codebook_ref"]
        codebook = (tuple(ot[r].float() for r in ref)
                    if isinstance(ref, list) else ot[ref].float())
        for q in g["targets"]:
            w = tens[q + ".weight"].float()
            packed = ot[q + ".cb_qweight"]
            assert packed.dtype == torch.uint8
            scales = ot.get(q + ".weight_scale")
            if grid == "fp8":
                assert scales is not None and scales.numel() == 128
                scales = scales.reshape(-1, 1)
            else:
                assert (q + ".weight_scale") not in ot   # fp4: scales in bytes
            up = cb.nvfp4_cb_unpack(packed, k, grid, mode, tuple(w.shape),
                                    codebook=codebook, scales=scales)
            rec = cb.nvfp4_cb_reconstruct(up, k, grid=grid, mode=mode).float()
            emu_f = cb.nvfp4_cb_fields(w, k, grid=grid, mode=mode,
                                       col_weights=cw[q], codebook=codebook)
            emu = cb.nvfp4_cb_reconstruct(emu_f, k, grid=grid,
                                          mode=mode).float()
            assert torch.equal(rec, emu)


def test_exporter_rejects_unknown_format(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl = export_dir / "model"
    _tiny_model(mdl)
    apath = export_dir / "assign.json"
    _write_assignment(apath, {
        "model.layers.0.mlp.gate_proj": {"data_type": "nv_fp", "bits": 4},
    })
    cw = {"model.layers.0.mlp.gate_proj": torch.rand(256) + 0.05}
    with pytest.raises(ValueError, match="cannot carry"):
        export_nvfp4_cb(mdl, apath, export_dir / "out", cw, device="cpu")


def test_exporter_rejects_non_multiple_of_256(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl = export_dir / "model"
    _tiny_model(mdl, in_f=300)          # 300 % 256 != 0
    apath = export_dir / "assign.json"
    _write_assignment(apath, {
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    })
    cw = {"model.layers.0.mlp.gate_proj": torch.rand(300) + 0.05}
    with pytest.raises(ValueError, match="multiple of 256"):
        export_nvfp4_cb(mdl, apath, export_dir / "out", cw, device="cpu")


def test_exporter_rejects_missing_col_weights(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl = export_dir / "model"
    _tiny_model(mdl)
    apath = export_dir / "assign.json"
    _write_assignment(apath, {
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    })
    with pytest.raises(ValueError, match="no col_weights"):
        export_nvfp4_cb(mdl, apath, export_dir / "out", {}, device="cpu")


def test_exporter_rejects_missing_learned_sidecar(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl = export_dir / "model"
    _tiny_model(mdl)
    apath = export_dir / "assign.json"
    _write_assignment(apath, {
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    })
    cw = {"model.layers.0.mlp.gate_proj": torch.rand(256) + 0.05}
    # learned source, no training, no supplied codebooks -> missing sidecar.
    with pytest.raises(ValueError, match="missing learned sidecar"):
        export_nvfp4_cb(mdl, apath, export_dir / "out", cw,
                        shared_codebook_spec={"source": "learned",
                                              "codebooks": {}}, device="cpu")


def test_export_native_compressed_hard_fails_on_cb():
    """A CB assignment reaching the compressed-tensors exporter must raise,
    not silently coerce to BF16 (mirrors the GGUF wrong-container guard)."""
    from prismaquant.export_native_compressed import (
        _coerce_runtime_legal_assignment,
    )
    with pytest.raises(ValueError, match="nvfp4_cb container"):
        _coerce_runtime_legal_assignment(
            "unused-model",
            {"model.layers.0.mlp.gate_proj": "NVFP4_CB_K16"},
        )


# ===========================================================================
# Layout v2 — two-tier scale coding (docs/nvfp4-cb-plan/two-tier-scale-spec.md).
# ===========================================================================

def _real_magnitude_w(rows=64, in_f=512, seed=0):
    """0.6B-magnitude weights: group scales land in e4m3's subnormal band
    (the regime where the v1 candidate sweep collapses)."""
    torch.manual_seed(seed)
    return torch.randn(rows, in_f) * 0.02


# T1 — compose exactness, exhaustive over all (E, c) pairs.
def test_two_tier_compose_exact_exhaustive():
    table, compose, legal = cb._two_tier_tables("cpu")
    assert table.shape == (16,) and compose.shape == (256, 16)
    assert torch.equal(table.to(torch.float8_e4m3fn).to(torch.float32), table)
    lv = compose[legal]
    assert int(legal.sum()) > 0
    # every legal pair round-trips e4m3 bit-exactly and lies in (0, 448].
    assert torch.equal(lv.to(torch.float8_e4m3fn).to(torch.float32), lv)
    assert bool((lv > 0).all()) and bool((lv <= 448.0).all())
    # every ILLEGAL finite positive pair fails the round-trip or the range.
    ill = ~legal & torch.isfinite(compose) & (compose > 0)
    iv = compose[ill]
    rt = iv.to(torch.float8_e4m3fn).to(torch.float32)
    assert bool(((rt != iv) | (iv > 448.0)).all())
    # union of legal compositions covers every positive e4m3 value (spec §1.2)
    e4m3_pos = sorted({float(torch.tensor(b, dtype=torch.uint8).view(
        torch.float8_e4m3fn).to(torch.float32)) for b in range(256)
        if 0 < float(torch.tensor(b, dtype=torch.uint8).view(
            torch.float8_e4m3fn).to(torch.float32)) <= 448.0})
    reachable = set(lv.tolist())
    assert set(e4m3_pos) <= reachable


# T1b — encoder fuzz: emitted (super, sub) pairs are always legal and the
# stored plane equals the composition.
@pytest.mark.parametrize("mode", ["full", "product", "signed"])
def test_two_tier_encoder_emits_only_legal_pairs(mode):
    k = 13 if mode == "full" else 14
    w = _real_magnitude_w(seed=1)
    fields = cb.nvfp4_cb_fields(w, k, grid="fp4", mode=mode,
                                scale_coding="two_tier")
    _, compose, legal = cb._two_tier_tables("cpu")
    e = fields["scale_super"].to(torch.int64)
    c = fields["scale_sub"]
    e_g = e.unsqueeze(-1).expand(*e.shape, 16).reshape(e.shape[0], -1)
    assert bool(legal[e_g, c].all())
    assert torch.equal(compose[e_g, c], fields["scales"])
    s = fields["scales"]
    assert torch.equal(s.to(torch.float8_e4m3fn).to(torch.float32), s)


# T2 — pack -> unpack -> reconstruct == emulation, bit-exact, all modes.
@pytest.mark.parametrize("mode", ["full", "product", "signed"])
def test_two_tier_pack_unpack_matches_emulation(mode):
    k = 13 if mode == "full" else 14
    w = _real_magnitude_w(seed=2)
    cw = torch.rand(512) + 0.05
    packed, fields = cb.nvfp4_cb_pack(w, k, grid="fp4", mode=mode,
                                      col_weights=cw,
                                      scale_coding="two_tier")
    up = cb.nvfp4_cb_unpack(packed, k, "fp4", mode, tuple(w.shape),
                            codebook=fields["codebook"],
                            scale_coding="two_tier")
    rec = cb.nvfp4_cb_reconstruct(up, k, grid="fp4", mode=mode)
    emu = cb.nvfp4_cb_reconstruct(fields, k, grid="fp4", mode=mode)
    assert torch.equal(rec, emu)
    assert torch.equal(up["scales"], fields["scales"])
    assert torch.equal(up["scale_super"], fields["scale_super"])
    assert torch.equal(up["scale_sub"], fields["scale_sub"])


# T3 — byte accounting: type_size 4k+9, packed nbytes, §2.1 bpw ladder.
def test_two_tier_type_size_and_effective_bits():
    for k, bpw in ((12, 1.78125), (13, 1.90625), (14, 2.03125),
                   (16, 2.28125), (18, 2.53125), (20, 2.78125),
                   (24, 3.28125)):
        assert cb.nvfp4_cb_type_size(k, "fp4", "two_tier") == 4 * k + 9
        assert cb.nvfp4_cb_effective_bits(
            k, "fp4", "two_tier") == pytest.approx(bpw, abs=1e-12)
        assert cb.nvfp4_cb_effective_bits(
            k, "fp4", "v1") == pytest.approx(k / 8 + 0.5, abs=1e-12)
    w = _real_magnitude_w(seed=3)
    packed, _ = cb.nvfp4_cb_pack(w, 14, grid="fp4", mode="product",
                                 scale_coding="two_tier")
    assert packed.shape == (64, (512 // 256) * (4 * 14 + 9))
    # registered rungs stay on v1 accounting until the serving gates clear.
    assert fr.get_format("NVFP4_CB_K14").effective_bits == pytest.approx(
        2.25, abs=1e-12)


# T4 — v1 regression: default decode path is v1 and unchanged.
def test_two_tier_v1_fixture_still_decodes():
    w = _real_magnitude_w(seed=4)
    packed, fields = cb.nvfp4_cb_pack(w, 14, grid="fp4", mode="product")
    assert packed.shape[-1] == (512 // 256) * (4 * 14 + 16)   # v1 type_size
    up = cb.nvfp4_cb_unpack(packed, 14, "fp4", "product", tuple(w.shape),
                            codebook=fields["codebook"])      # no scale_coding
    rec = cb.nvfp4_cb_reconstruct(up, 14, grid="fp4", mode="product")
    emu = cb.nvfp4_cb_reconstruct(fields, 14, grid="fp4", mode="product")
    assert torch.equal(rec, emu)
    assert "scale_super" not in up


# T5 — determinism: encode twice => identical bytes (CPU and CUDA).
@pytest.mark.parametrize("device", _DEVICES)
def test_two_tier_determinism(device):
    w = _real_magnitude_w(seed=5).to(device)
    cw = (torch.rand(512) + 0.05).to(device)
    p1, _ = cb.nvfp4_cb_pack(w, 14, grid="fp4", mode="product",
                             col_weights=cw, scale_coding="two_tier")
    p2, _ = cb.nvfp4_cb_pack(w, 14, grid="fp4", mode="product",
                             col_weights=cw, scale_coding="two_tier")
    assert torch.equal(p1, p2)


# T6 — edges: all-zero group / superblock, 448 top, subnormal snap-up.
def test_two_tier_edges():
    torch.manual_seed(6)
    w = torch.randn(4, 512) * 0.02
    w[0, :16] = 0.0                                  # all-zero group
    w[1, :256] = 0.0                                 # all-zero superblock
    w[2, :16] = 448.0 * 6.0                          # amax at the 448 scale top
    fields = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product",
                                scale_coding="two_tier")
    s = fields["scales"]
    assert bool((s > 0).all())                       # T has no zero (spec)
    assert torch.equal(s.to(torch.float8_e4m3fn).to(torch.float32), s)
    assert float(s[2, 0]) == 448.0                   # top reachable: 1.75*2^8
    # zero regions: scale is the deterministic first-legal candidate (bytes
    # pinned below); recon is bounded by grid*scale (the lattice need not
    # contain an exact zero codeword — same as v1).
    recon = cb.nvfp4_cb_reconstruct(fields, 14, grid="fp4", mode="product")
    assert float(recon[0, :16].abs().max()) <= 6.0 * float(s[0, 0])
    assert float(recon[1, :256].abs().max()) <= 6.0 * float(s[1, :16].max())
    # determinism of the degenerate bytes
    p1 = cb.nvfp4_cb_assemble_bytes(fields, 14, "fp4", "product")
    f2 = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product",
                            scale_coding="two_tier")
    p2 = cb.nvfp4_cb_assemble_bytes(f2, 14, "fp4", "product")
    assert torch.equal(p1, p2)


def test_two_tier_snap_up_no_clip():
    # One tiny-amax group among 15 big ones: its ideal scale sits below the
    # superblock's reachable floor at the chosen E -> snaps UP (>= ideal), the
    # no-clip direction: |w/s| <= 6 everywhere in that group.
    torch.manual_seed(7)
    w = torch.randn(1, 256) * 0.05
    w[0, :16] *= 1e-4                                # tiny group 0
    fields = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product",
                                scale_coding="two_tier")
    s0 = float(fields["scales"][0, 0])
    ideal0 = float(w[0, :16].abs().amax() / 6.0)
    assert s0 >= ideal0
    assert float((w[0, :16].abs() / s0).max()) <= 6.0 + 1e-6


# Candidate diversity: the un-collapse the two-tier coding buys. On a
# subnormal-band tensor the v1 clip sweep collapses to a handful of distinct
# e4m3 candidates per group; the two-tier window offers >= 16 distinct legal
# reachable values per superblock.
def test_two_tier_candidate_diversity_vs_v1():
    w = _real_magnitude_w(seed=8)
    amax = cb._group_amax(w, "fp4")
    v1 = cb._candidate_scales(amax, "fp4", cb._SCALE_SWEEP_CANDIDATES)
    v1_distinct = torch.tensor([
        len(set(v1[:, r, g].tolist()))
        for r in range(0, 64, 16) for g in range(0, 32, 8)])
    assert float(v1_distinct.float().mean()) <= 8.0   # the collapse (v1)
    _, compose, legal = cb._two_tier_tables("cpu")
    e_lo, e_hi, W = cb._two_tier_window(amax)
    for r in range(0, 64, 16):
        for sb in range(2):
            vals = set()
            for i in range(W):
                E = min(int(e_lo[r, sb]) + i, int(e_hi[r, sb]))
                vals |= set(compose[E][legal[E]].tolist())
            assert len(vals) >= 16


# v2 sweep quality: never worse than the v1 one-shot; empirically also beats
# the v1 free sweep on subnormal-band tensors (the spec §3 negative tax).
@pytest.mark.parametrize("mode", ["product", "signed"])
def test_two_tier_beats_one_shot_and_v1_sweep(mode):
    w = _real_magnitude_w(seed=9)
    cw = torch.rand(512) + 0.05
    cw2d = torch.broadcast_to(cw, w.shape).contiguous()
    wq = cb._col_weight_vectors(cw2d)
    C = cb._resolve_codebook(14, "fp4", mode, None, w.device)

    def err_of(scales):
        e, _, _ = cb._eval_candidate(w, wq, scales, "fp4", mode, C)
        return float(e.sum())

    one_shot = cb._candidate_scales(cb._group_amax(w, "fp4"), "fp4", 16)[0]
    f_v1 = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode=mode, col_weights=cw)
    f_v2 = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode=mode, col_weights=cw,
                              scale_coding="two_tier")
    e_one, e_v1, e_v2 = (err_of(one_shot), err_of(f_v1["scales"]),
                         err_of(f_v2["scales"]))
    assert e_v2 <= e_one + 1e-6
    assert e_v2 <= e_v1 + 1e-6


def test_two_tier_rejects_fp8_and_no_sweep():
    w = _real_magnitude_w(seed=10)
    with pytest.raises(ValueError, match="fp4-family only"):
        cb.nvfp4_cb_fields(w, 40, grid="fp8", mode="product",
                           scale_coding="two_tier")
    with pytest.raises(ValueError, match="IS the sweep"):
        cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product",
                           scale_sweep=False, scale_coding="two_tier")
    with pytest.raises(ValueError, match="scale_coding"):
        cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product",
                           scale_coding="v3")


def test_two_tier_exporter_writes_layout_version(export_dir):
    from safetensors.torch import load_file

    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl, out = export_dir / "model", export_dir / "out"
    tens = _tiny_model(mdl)
    assign = {
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
        "model.layers.1.mlp.gate_proj": {"data_type": "fp8_cb", "cb_k": 40},
    }
    apath = export_dir / "assign.json"
    _write_assignment(apath, assign)
    cw = {q: torch.rand(256) + 0.05 for q in assign}
    export_nvfp4_cb(mdl, apath, out, cw, device="cpu",
                    scale_coding="two_tier")
    qc = json.loads((out / "quant_config.json").read_text())
    assert qc["layout_version"] == 2
    assert qc["provenance"]["scale_coding"] == "two_tier"
    ot = load_file(str(out / "model.safetensors"))
    for g in qc["config_groups"].values():
        s = g["scheme"]
        if s["grid"] == "fp4":
            sc = s["scale_coding"]
            assert sc["kind"] == "two_tier" and sc["sub_bits"] == 4
            assert sc["super_bias"] == 127 and len(sc["table"]) == 16
            tb = torch.tensor(sc["table"], dtype=torch.float32)
            assert torch.equal(
                tb.to(torch.float8_e4m3fn).to(torch.float32), tb)
            assert s["type_size"] == 4 * s["k"] + 9
            # round-trip through the v2 scheme
            q = g["targets"][0]
            ref = s["codebook_ref"]
            codebook = (tuple(ot[r].float() for r in ref)
                        if isinstance(ref, list) else ot[ref].float())
            w = tens[q + ".weight"].float()
            up = cb.nvfp4_cb_unpack(ot[q + ".cb_qweight"], s["k"], "fp4",
                                    s["mode"], tuple(w.shape),
                                    codebook=codebook,
                                    scale_coding="two_tier")
            emu_f = cb.nvfp4_cb_fields(w, s["k"], grid="fp4", mode=s["mode"],
                                       col_weights=cw[q], codebook=codebook,
                                       scale_coding="two_tier")
            assert torch.equal(
                cb.nvfp4_cb_reconstruct(up, s["k"], grid="fp4",
                                        mode=s["mode"]),
                cb.nvfp4_cb_reconstruct(emu_f, s["k"], grid="fp4",
                                        mode=s["mode"]))
        else:
            assert "scale_coding" not in s          # fp8: no scale plane
            assert s["type_size"] == 4 * s["k"]
