"""Phase-0 NVFP4-CB measurement-harness tests.

Covers the three harness modules authored for Phase 0:
  * nvfp4_cb_footprint  — sidecar-aware byte accountant (§1.2 table)
  * index_entropy       — index-stream entropy / redundancy
  * emu_forward_kl      — whole-model emulated forward KL-vs-BF16 (GPU)

These are format-agnostic: they use the already-registered GGUF/NVFP4 formats
plus the §2 CB byte parameters, so they do not depend on the (separately
authored) nvfp4_cb_formats.py registration landing first.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from prismaquant.nvfp4_cb_footprint import cb_footprint
from prismaquant.index_entropy import index_entropy


# ---------------------------------------------------------------------------
# Footprint — reproduces the docs §1.2 bpw table exactly (fixed lattice)
# ---------------------------------------------------------------------------

# From docs/nvfp4-cb-plan/format-pipeline.md §1.2: (k, bpw, type_size B/256).
_SEC_1_2 = [
    (12, 2.000, 64), (13, 2.125, 68), (14, 2.250, 72), (15, 2.375, 76),
    (16, 2.500, 80), (17, 2.625, 84), (18, 2.750, 88), (19, 2.875, 92),
    (20, 3.000, 96), (21, 3.125, 100), (22, 3.250, 104), (23, 3.375, 108),
    (24, 3.500, 112),
]


@pytest.mark.parametrize("k,bpw,type_size", _SEC_1_2)
def test_footprint_reproduces_section_1_2_bpw(k, bpw, type_size):
    # in_features multiple of 256, out arbitrary → registry-exact.
    shape = (128, 256)
    n = shape[0] * shape[1]
    fmt = f"NVFP4_CB_K{k}"
    fp = cb_footprint({"w": fmt}, {"w": shape})
    # Fixed-lattice body bpw is exactly k/8 + 0.5.
    assert fp["body_bpw"] == pytest.approx(bpw, abs=1e-9)
    # type_size (bytes per 256 weights) matches the table.
    expected_body_bytes = (n // 256) * type_size
    assert fp["body_bytes"] == expected_body_bytes
    assert fp["per_tensor"]["w"]["k"] == k
    # No learned codebook ⇒ no sidecar; only the per-tensor global scale sits
    # on top of the registry-exact body.
    assert fp["sidecar_bytes"] == 0
    assert fp["global_scale_bytes"] == 4


def test_footprint_adds_learned_sidecar():
    k = 16
    shape = (256, 256)
    fmt = f"NVFP4_CB_K{k}"
    base = cb_footprint({"w": fmt}, {"w": shape})
    learned = cb_footprint(
        {"w": fmt}, {"w": shape}, codebook_sources={"w": "learned"})
    expected_sidecar = (1 << k) * 4  # 2^k entries × 4 bytes = 256 KB
    assert base["sidecar_bytes"] == 0
    assert learned["sidecar_bytes"] == expected_sidecar
    assert learned["total_bytes"] == base["total_bytes"] + expected_sidecar
    # Body bpw is unchanged (registry-exact); total bpw is strictly larger.
    assert learned["body_bpw"] == pytest.approx(base["body_bpw"], abs=1e-9)
    assert learned["total_bpw"] > learned["body_bpw"]


def test_footprint_shared_codebook_charged_once():
    k = 12
    shape = (256, 256)
    fmt = f"NVFP4_CB_K{k}"
    src = {"a": {"learned": "cb", "group": "role0"},
           "b": {"learned": "cb", "group": "role0"}}
    fp = cb_footprint(
        {"a": fmt, "b": fmt}, {"a": shape, "b": shape}, codebook_sources=src)
    # One shared codebook for both tensors → charged once.
    assert fp["sidecar_bytes"] == (1 << k) * 4


def test_footprint_mixed_registry_format():
    # A stock (non-CB) format still accounts via the registry; no CB sidecar
    # or global scale for it.
    fp = cb_footprint({"w": "NVFP4"}, {"w": (256, 256)})
    assert fp["sidecar_bytes"] == 0
    assert fp["global_scale_bytes"] == 0
    assert fp["body_bpw"] == pytest.approx(4.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Index entropy
# ---------------------------------------------------------------------------

def test_entropy_uniform_indices_approaches_k():
    k = 10
    torch.manual_seed(0)
    # Large uniform sample over all 2^k symbols → H ≈ k.
    idx = torch.randint(0, 1 << k, (400_000,))
    r = index_entropy(idx, k)
    assert r["H"] == pytest.approx(k, abs=0.05)
    assert r["redundancy"] == pytest.approx(0.0, abs=0.05)
    assert r["redundancy"] == pytest.approx(k - r["H"], abs=1e-9)


def test_entropy_constant_indices_zero():
    idx = torch.full((10_000,), 7, dtype=torch.long)
    r = index_entropy(idx, 12)
    assert r["H"] == pytest.approx(0.0, abs=1e-9)
    assert r["redundancy"] == pytest.approx(12.0, abs=1e-9)
    assert r["H_conditional"] == pytest.approx(0.0, abs=1e-9)


def test_entropy_two_symbol_exact():
    # Balanced two-symbol stream → H = 1 bit exactly.
    idx = torch.tensor([0, 1] * 5000, dtype=torch.long)
    r = index_entropy(idx, 4)
    assert r["H"] == pytest.approx(1.0, abs=1e-6)
    assert r["redundancy"] == pytest.approx(3.0, abs=1e-6)
    # Perfectly predictable from the previous symbol → conditional H ≈ 0.
    assert r["H_conditional"] == pytest.approx(0.0, abs=1e-6)
    assert r["conditional_gain"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Emulated forward KL (GPU)
# ---------------------------------------------------------------------------

_MODEL = "/home/rob/models/Qwen3-0.6B"


def _tiny_dataset(tmp_path: Path) -> str:
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Quantization allocates bits per linear layer to minimize divergence. "
        "Vector codebooks decode to the native floating-point grid.\n\n"
    ) * 6
    p = tmp_path / "held_out.txt"
    p.write_text(text)
    return str(p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not Path(_MODEL).exists(), reason="Qwen3-0.6B not present")
def test_emu_kl_identity_is_zero(tmp_path):
    from prismaquant.emu_forward_kl import measure_emulated_kl
    from prismaquant.measure_quant_cost import canonical_linear_name
    from transformers import AutoModelForCausalLM
    import torch.nn as nn

    ds = _tiny_dataset(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    fmap = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            fmap[canonical_linear_name(name)] = {"format": "BF16",
                                                 "col_weights": None}
    del model

    res = measure_emulated_kl(
        _MODEL, fmap, ds, device="cuda", seqlen=128, max_tokens=256)
    # BF16 passthrough is bit-identical → KL is exactly zero.
    assert res["kl_all"] == 0.0
    assert res["kl_confident"] == 0.0
    assert res["top1_agreement"] == 1.0
    assert res["n_positions"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not Path(_MODEL).exists(), reason="Qwen3-0.6B not present")
def test_emu_kl_q4k_positive_and_deterministic(tmp_path):
    from prismaquant.emu_forward_kl import measure_emulated_kl
    from prismaquant.measure_quant_cost import canonical_linear_name
    from transformers import AutoModelForCausalLM
    import torch.nn as nn

    ds = _tiny_dataset(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    fmap = {}
    for name, mod in model.named_modules():
        # Quantize only the MLP/attention projections (in_features % 256 == 0
        # is not required for GGUF emulation, but keep it representative).
        if isinstance(mod, nn.Linear):
            fmap[canonical_linear_name(name)] = {"format": "Q4_K",
                                                 "col_weights": None}
    del model

    a = measure_emulated_kl(
        _MODEL, fmap, ds, device="cuda", seqlen=128, max_tokens=256)
    b = measure_emulated_kl(
        _MODEL, fmap, ds, device="cuda", seqlen=128, max_tokens=256)
    assert a["kl_all"] > 0.0
    assert math.isfinite(a["kl_all"])
    assert math.isfinite(a["kl_confident"])
    # Deterministic across runs (greedy forward, fixed seed).
    assert a["kl_all"] == pytest.approx(b["kl_all"], rel=0, abs=0.0)
    assert a["provenance"]["assignment_sha256"] == b["provenance"]["assignment_sha256"]
