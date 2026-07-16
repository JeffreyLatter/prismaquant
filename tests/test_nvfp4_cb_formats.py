"""NVFP4-CB / FP8-CB codebook format tests (Milestone A, emulation)."""
from __future__ import annotations

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
    fields = cb.nvfp4_cb_fields(w, 13, grid="fp4", mode="signed",
                                col_weights=cwq)
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
    # with tiny tables and beats product mode there.
    torch.manual_seed(10)
    w = torch.randn(128, 1024)
    cw = torch.rand(1024) + 0.05
    for k in (15, 16):
        rs = cb.make_nvfp4_cb_qdq(k, "fp4", "signed")(w)
        rp = cb.make_nvfp4_cb_qdq(k, "fp4", "product")(w)
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
