"""Pipeline-integration tests for the NVFP4-CB / FP8-CB codebook lane.

Milestone-B pipeline plumbing (serving profile + allocator mixed-menu flow).
The byte-format contract is pinned by test_nvfp4_cb_formats.py; this file pins
that the CB rungs flow through the STANDARD allocator machinery as first-class
menu rungs alongside plain NVFP4/FP8_DYNAMIC/BF16 (the mixed container,
PLAN.md decision #1 / format-pipeline.md §5), namely:

  - serving_profile_specs/nvfp4_cb.json loads, is discoverable, allows both CB
    families + NVFP4/FP8_DYNAMIC/BF16, and enforces the in_features%256 shape
    rule (mirrors gguf.json);
  - a mixed menu (NVFP4_CB_K14,NVFP4_CB_K16,FP8_CB_K44,NVFP4,FP8_DYNAMIC,BF16)
    flows through build_candidates legality + the knapsack + fused-sibling
    promotion without error;
  - CB rungs respect in_features%256 legality via the existing group_size=256
    divisibility check (fall back to a coarser legal rung / BF16 when violated);
  - passthrough integrity is untouched — CB is SYNTHESIZED, never passthrough;
  - the family-coherence gate WARNS but does not block an intentional
    intra-family CB ladder (hard-fails only under --enforce-family-coherence).
"""
from __future__ import annotations

import json
import pickle
import sys

import pytest

import torch

from prismaquant import format_registry as fr
from prismaquant import layer_config as lcfg
from prismaquant import measure_quant_cost as mqc
from prismaquant import serving_profiles as sp
from prismaquant.allocator import (
    aggregate_fused_siblings,
    build_candidates,
    expand_fused_sibling_assignment,
    solve_with_promotion,
)
from prismaquant.allocator_candidates import (
    PASSTHROUGH_SOURCE_REQUIREMENTS,
    check_format_applicability,
)
from prismaquant.allocator_solver import promote_serving_units
import prismaquant.allocator as alloc


# The task's canonical mixed menu: two CB families + their native carriers.
_MIXED_MENU = [
    "NVFP4_CB_K14", "NVFP4_CB_K16", "FP8_CB_K44",
    "NVFP4", "FP8_DYNAMIC", "BF16",
]
_ALL_CB_RUNGS = (
    [f"NVFP4_CB_K{k}" for k in range(12, 25)]
    + [f"NVFP4_CB_S{k}" for k in (13, 14, 15, 16)]
    + [f"FP8_CB_K{k}" for k in (36, 40, 44, 48)]
)


class _FakeProfile:
    """q/k/v -> one fused group; gate/up -> one fused group (like Qwen3)."""

    def fused_sibling_group(self, name: str) -> str | None:
        if name.endswith((".q_proj", ".k_proj", ".v_proj")):
            return name.rsplit(".", 1)[0] + ".qkv_proj"
        if name.endswith((".gate_proj", ".up_proj")):
            return name.rsplit(".", 1)[0] + ".gate_up_proj"
        return None


def _menu_specs(menu=_MIXED_MENU):
    return [fr.get_format(n) for n in menu]


def _canon(menu=_MIXED_MENU):
    # Canonical registry names (FP8_DYNAMIC -> FP8_E4M3, etc.) — Candidate.fmt
    # and the emitted assignment use these, not the raw menu aliases.
    return {fr.get_format(n).name for n in menu}


def _cost_entry(dloss: float) -> dict:
    return {"weight_mse": max(dloss, 0.0), "predicted_dloss": max(dloss, 0.0)}


def _costs_for(menu_specs, h_trace: float) -> dict:
    # Monotone: fewer bits -> higher dloss, so the DP has a real tradeoff.
    dloss_by_bpp = lambda s: 0.02 * h_trace / max(s.effective_bits, 1.0)
    return {s.name: _cost_entry(dloss_by_bpp(s)) for s in menu_specs}


def _dense_model(menu_specs):
    """One decoder layer: q/k/v/o + gate/up/down, all in_features % 256 == 0."""
    layer = "model.layers.0"
    # (out, in) — every in_features divisible by 256 AND by 16 (NVFP4 legal).
    shapes = {
        "self_attn.q_proj": (2048, 1024),
        "self_attn.k_proj": (256, 1024),
        "self_attn.v_proj": (256, 1024),
        "self_attn.o_proj": (1024, 2048),
        "mlp.gate_proj": (3072, 1024),
        "mlp.up_proj": (3072, 1024),
        "mlp.down_proj": (1024, 3072),
    }
    h = {
        "self_attn.q_proj": 0.5, "self_attn.k_proj": 0.3, "self_attn.v_proj": 0.7,
        "self_attn.o_proj": 0.4, "mlp.gate_proj": 0.8, "mlp.up_proj": 0.6,
        "mlp.down_proj": 0.9,
    }
    stats, costs = {}, {}
    for leaf, (d_out, d_in) in shapes.items():
        name = f"{layer}.{leaf}"
        stats[name] = {
            "h_trace": h[leaf], "n_params": d_out * d_in,
            "in_features": d_in, "out_features": d_out,
        }
        costs[name] = _costs_for(menu_specs, h[leaf])
    return stats, costs


# ---------------------------------------------------------------------------
# (A) serving profile: discoverable, allows both families + carriers, %256.
# ---------------------------------------------------------------------------

def test_nvfp4_cb_profile_discoverable_and_metadata():
    assert "nvfp4_cb" in sp.serving_profile_names()
    prof = sp.load_serving_profile("nvfp4_cb")
    assert prof.id == "nvfp4_cb"
    assert prof.runtime == "gridbook_plugin"


def test_nvfp4_cb_profile_allows_all_cb_rungs_and_carriers():
    prof = sp.load_serving_profile("nvfp4_cb")
    for name in _ALL_CB_RUNGS + ["NVFP4", "FP8_DYNAMIC", "BF16"]:
        d = prof.check_format(None, name)
        assert d.legal, f"nvfp4_cb profile must allow {name}: {d.reason} {d.detail}"


def test_nvfp4_cb_profile_denies_out_of_family_format():
    prof = sp.load_serving_profile("nvfp4_cb")
    # A registered vLLM format that is NOT on the CB menu is structurally
    # unavailable in this container.
    d = prof.check_format(None, "MXFP4")
    assert not d.legal and d.reason == "profile_mismatch"


@pytest.mark.parametrize("rung", ["NVFP4_CB_K12", "NVFP4_CB_S16", "FP8_CB_K44"])
def test_nvfp4_cb_profile_shape_rule_256(rung):
    ok = sp.check_serving_shape(
        "nvfp4_cb", rung, in_features=2048, out_features=512)
    bad = sp.check_serving_shape(
        "nvfp4_cb", rung, in_features=2064, out_features=512)  # 2064 % 256 == 16
    assert ok.legal, f"{rung} should be legal at in=2048: {ok.detail}"
    assert not bad.legal and bad.reason == "kernel_shape"


# ---------------------------------------------------------------------------
# (B) build_candidates legality: CB rungs kept when aligned, masked otherwise;
#     passthrough integrity untouched (CB synthesized, never passthrough).
# ---------------------------------------------------------------------------

def test_mixed_menu_build_candidates_keeps_cb_rungs():
    specs = _menu_specs()
    stats, costs = _dense_model(specs)
    cands = build_candidates(stats, costs, specs, target_profile="nvfp4_cb")
    for name in stats:
        fmts = {c.fmt for c in cands[name]}
        # Every menu rung is legal on these 256-aligned shapes (canonical names).
        assert _canon() <= fmts, f"{name} lost menu rungs: {sorted(fmts)}"


def test_cb_masked_when_in_features_not_256_falls_back():
    specs = _menu_specs()
    # in_features = 2064: NVFP4 (group16) legal, but 2064 % 256 == 16 -> CB
    # rungs illegal. The Linear must keep NVFP4/BF16 and drop CB (coarser
    # fallback), never crash.
    name = "model.layers.0.self_attn.o_proj"
    stats = {name: {"h_trace": 0.5, "n_params": 1024 * 2064,
                    "in_features": 2064, "out_features": 1024}}
    costs = {name: _costs_for(specs, 0.5)}
    mask_records: list[dict] = []
    cands = build_candidates(stats, costs, specs, target_profile="nvfp4_cb",
                             mask_records=mask_records)
    fmts = {c.fmt for c in cands[name]}
    assert "NVFP4" in fmts and "BF16" in fmts, f"fallback rungs lost: {fmts}"
    for cb_rung in ("NVFP4_CB_K14", "NVFP4_CB_K16", "FP8_CB_K44"):
        assert cb_rung not in fmts, f"{cb_rung} should be masked at in=2064"
    masked_cb = {r["format"] for r in mask_records if r["format"].endswith(
        ("K14", "K16", "K44"))}
    assert {"NVFP4_CB_K14", "NVFP4_CB_K16", "FP8_CB_K44"} <= masked_cb
    # The 256-superblock divisibility is what masks CB (group_size double duty).
    assert all(r["reason"] in ("group_divisibility", "kernel_shape")
               for r in mask_records)


def test_cb_is_synthesized_never_passthrough():
    # CB rungs are absent from the passthrough integrity table...
    for rung in _ALL_CB_RUNGS:
        assert rung not in PASSTHROUGH_SOURCE_REQUIREMENTS
    # ...and are legal with no source dtype (they are synthesized, like NVFP4),
    # while genuine passthrough formats still require their source dtype.
    shape = (512, 2048)  # 2048 % 256 == 0
    cb_ok = check_format_applicability(
        shape, "NVFP4_CB_K16", source_kind=None, target_profile="nvfp4_cb")
    assert cb_ok.legal, cb_ok.detail
    # Passthrough integrity is UNTOUCHED: FP8_SOURCE still needs an fp8 source.
    fp8src = check_format_applicability(
        shape, "FP8_SOURCE", source_kind=None, target_profile="research")
    assert not fp8src.legal and fp8src.reason == "source_dtype_mismatch"
    fp8src_ok = check_format_applicability(
        shape, "FP8_SOURCE", source_kind="fp8", target_profile="research")
    assert fp8src_ok.legal


# ---------------------------------------------------------------------------
# (C) knapsack + fused-sibling promotion over the mixed CB menu.
# ---------------------------------------------------------------------------

def test_mixed_menu_solve_and_fused_promotion_uniform():
    specs = _menu_specs()
    stats, costs = _dense_model(specs)
    profile = _FakeProfile()
    cands = build_candidates(stats, costs, specs, target_profile="nvfp4_cb")

    stats_x, costs_x, cands_x = aggregate_fused_siblings(
        stats, costs, specs, cands, profile)
    format_specs = {s.name: s for s in specs}
    format_rank = {s.name: i for i, s in enumerate(
        sorted(specs, key=lambda s: s.effective_bits))}

    assignment, achieved = solve_with_promotion(
        stats_x, cands_x, target_bits=4.0,
        format_specs=format_specs, format_rank=format_rank,
        bit_precision=0.001, profile=profile)
    assert assignment is not None
    assert isinstance(achieved, float)

    expanded = expand_fused_sibling_assignment(assignment, stats_x)
    # Fused siblings must be uniform (one format per group) — the union-find
    # coherence invariant, now proven to hold with CB rungs in the menu.
    qkv = [expanded[f"model.layers.0.self_attn.{p}_proj"] for p in "qkv"]
    gu = [expanded[f"model.layers.0.mlp.{p}_proj"] for p in ("gate", "up")]
    assert len(set(qkv)) == 1, f"q/k/v not uniform: {qkv}"
    assert len(set(gu)) == 1, f"gate/up not uniform: {gu}"
    # Every chosen format is a real menu rung.
    assert set(expanded.values()) <= _canon()


def test_promote_serving_units_lifts_mixed_group_to_max_rank():
    # A fused group that the DP left mixed across CB + carrier rungs must be
    # promoted UP to its highest-rank (most-bits) member — over the CB ladder
    # exactly as over any other menu (format-pipeline.md §5).
    specs = _menu_specs()
    format_rank = {s.name: i for i, s in enumerate(
        sorted(specs, key=lambda s: s.effective_bits))}
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4_CB_K14",
        "model.layers.0.self_attn.k_proj": "NVFP4_CB_K16",
        "model.layers.0.self_attn.v_proj": "NVFP4",
    }
    promoted = promote_serving_units(
        assignment, format_rank, profile=_FakeProfile(),
        include_fused=True, include_moe=True)
    chosen = set(promoted.values())
    assert chosen == {"NVFP4"}, f"group should lift to max-rank NVFP4: {promoted}"


# ---------------------------------------------------------------------------
# (D) family-coherence: WARNS on an intra-family ladder, does NOT block;
#     hard-fails only under --enforce-family-coherence. Driven end-to-end
#     through the real allocator.main() (mirrors test_allocator_main_*).
# ---------------------------------------------------------------------------

def _write_alloc_fixture(tmp_path, menu):
    # 256-aligned dense fixture the CB rungs survive on.
    specs = _menu_specs(menu)
    stats, costs = _dense_model(specs)
    probe = {"stats": stats, "meta": {"model": None}}
    cost_blob = {"costs": costs, "meta": {"formats": list(menu)}}
    p = tmp_path / "probe.pkl"
    c = tmp_path / "cost.pkl"
    p.write_bytes(pickle.dumps(probe))
    c.write_bytes(pickle.dumps(cost_blob))
    return p, c


def _run_main(tmp_path, monkeypatch, menu, *, enforce, target="3.0"):
    probe_p, cost_p = _write_alloc_fixture(tmp_path, menu)
    lc = tmp_path / "layer_config.json"
    csv = tmp_path / "pareto.csv"
    argv = [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", ",".join(menu),
        "--target-bits", target,
        "--pareto-targets", target,
        "--layer-config", str(lc),
        "--pareto-csv", str(csv),
        "--target-profile", "nvfp4_cb",
        "--allow-default-profile",
    ]
    if enforce:
        argv.append("--enforce-family-coherence")
    monkeypatch.setattr(sys, "argv", argv)
    alloc.main()
    return lc


# NVFP4_CB_K15 (2.375) and K16 (2.5) both bucket to the 2.5 bit-tier -> the
# family-coherence gate collides on an intentional intra-family CB ladder.
_ADJACENT_LADDER = ["NVFP4_CB_K15", "NVFP4_CB_K16", "BF16"]


def test_family_coherence_warns_but_does_not_block(tmp_path, monkeypatch, capsys):
    lc = _run_main(tmp_path, monkeypatch, _ADJACENT_LADDER, enforce=False)
    out = capsys.readouterr().out
    assert "multiple candidates at the same bit tier" in out
    assert "WARNING" in out
    assert lc.exists(), "warn-not-block: the allocation must still be emitted"
    emitted = json.loads(lc.read_text())
    names = emitted.get("assignment", emitted)
    assert names, "emitted layer_config must carry an assignment"


def test_family_coherence_enforced_raises(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch, _ADJACENT_LADDER, enforce=True)


def test_task_example_mixed_menu_flows_end_to_end(tmp_path, monkeypatch):
    # The canonical mixed menu is well-spaced (no 0.25-tier collision), so it
    # flows through main() cleanly and emits a CB-containing assignment.
    lc = _run_main(tmp_path, monkeypatch, _MIXED_MENU, enforce=False, target="4.0")
    assert lc.exists()
    emitted = json.loads(lc.read_text())
    names = emitted.get("assignment", emitted)
    assert names
    # Layer-config entries are rich dicts (AutoRound schema); canonicalize each.
    chosen = {lcfg.canonicalize_format(v) for v in names.values()}
    assert chosen <= _canon() | {"BF16"}, f"off-menu format chosen: {chosen}"
    assert any(c.startswith(("NVFP4_CB", "FP8_CB")) for c in chosen), (
        f"mixed menu produced no CB rung: {chosen}")


# ---------------------------------------------------------------------------
# (E) CB local-cost imatrix lockstep: the batched cost render must use the SAME
#     imatrix-weighted VQ the exporter ships (one-cache/no-confound rule).
#     measure_quant_cost._batched_quantize now renders the CB families (before
#     the fix it raised "Unknown weight_element_dtype" outright — CB cost was
#     broken in batched mode, which run-pipeline.sh uses). Mirrors
#     test_gguf_formats.test_batched_cost_path_matches_unbatched.
# ---------------------------------------------------------------------------
_CB_COST_RUNGS = ["NVFP4_CB_K16", "NVFP4_CB_S16", "FP8_CB_K44"]


@pytest.mark.parametrize("rung", _CB_COST_RUNGS)
def test_cb_batched_cost_matches_direct_qdq(rung):
    # The batched cost path must equal the per-slice registry qdq the exporter
    # uses — unweighted AND under a per-item imatrix — or the allocator's cost
    # diverges from the shipped bytes. (in_features=512 % 256 == 0.)
    spec = fr.get_format(rung)
    torch.manual_seed(0)
    stacked = torch.randn(3, 256, 512) * torch.rand(3, 1, 1).exp()
    batched = mqc._batched_quantize(spec, stacked)
    per_slice = torch.stack(
        [spec.quantize_dequantize(stacked[i]) for i in range(3)])
    torch.testing.assert_close(batched, per_slice, rtol=0, atol=0)
    # Per-item imatrix (N,1,in) — the batched cost path's shape.
    qw = torch.rand(3, 1, 512) + 0.05
    batched_w = mqc._batched_quantize(spec, stacked, col_weights=qw)
    per_slice_w = torch.stack([
        spec.quantize_dequantize(stacked[i], col_weights=qw[i, 0])
        for i in range(3)])
    torch.testing.assert_close(batched_w, per_slice_w, rtol=0, atol=0)


@pytest.mark.parametrize("rung", _CB_COST_RUNGS)
def test_cb_imatrix_changes_cost_and_lowers_weighted_error(rung):
    # "The CB cost entry changes when the act cache provides imatrix vs not":
    # the weighted render differs from the unweighted one, and the weighted VQ
    # search lowers the weighted MSE it optimizes — so the DP sees a different,
    # exporter-faithful cost.
    spec = fr.get_format(rung)
    torch.manual_seed(1)
    w = torch.randn(128, 512)
    qw = torch.rand(512) + 0.05
    unweighted = spec.quantize_dequantize(w.clone())
    weighted = spec.quantize_dequantize(w.clone(), col_weights=qw)
    assert not torch.equal(unweighted, weighted), "imatrix did not change render"
    wmse = lambda r: float(((w - r).pow(2) * qw).mean())
    assert wmse(weighted) <= wmse(unweighted) * 1.0001 + 1e-9


def test_cost_render_uses_imatrix_predicate_and_gguf_toggle(monkeypatch):
    # CB families are ALWAYS imatrix-weighted (their export always is, no
    # toggle); gguf tracks PRISMAQUANT_GGUF_IMATRIX; plain nv is never weighted.
    cb = fr.get_format("NVFP4_CB_K16")
    fp8cb = fr.get_format("FP8_CB_K44")
    gguf = fr.get_format("Q4_K")
    nvfp4 = fr.get_format("NVFP4")
    monkeypatch.delenv("PRISMAQUANT_GGUF_IMATRIX", raising=False)
    assert mqc._cost_render_uses_imatrix(cb)
    assert mqc._cost_render_uses_imatrix(fp8cb)
    assert mqc._cost_render_uses_imatrix(gguf)          # default on
    assert not mqc._cost_render_uses_imatrix(nvfp4)
    monkeypatch.setenv("PRISMAQUANT_GGUF_IMATRIX", "0")
    assert not mqc._cost_render_uses_imatrix(gguf)      # toggled off
    assert mqc._cost_render_uses_imatrix(cb)            # CB unaffected
    assert mqc._cost_render_uses_imatrix(fp8cb)


def test_batched_quantize_still_rejects_col_weights_for_unsupported_family():
    # The guard must remain for genuinely-unsupported families (MXFP4 has no
    # imatrix render) — only gguf + CB families were opened.
    spec = fr.get_format("MXFP4")
    with pytest.raises(ValueError, match="gguf-family and CB codebook"):
        mqc._batched_quantize(spec, torch.randn(2, 64, 128),
                              col_weights=torch.rand(1, 1, 128))


# ---------------------------------------------------------------------------
# (F) Mixed-container STOCK-RUNG export: export_nvfp4_cb now packs stock NVFP4 /
#     FP8_DYNAMIC CT-style (reusing the export_native_compressed codecs) so the
#     plugin can delegate them to vLLM's CompressedTensors path — the 27B
#     production menu carries them alongside the CB rungs. Before this the
#     coverage gate hard-rejected any non-CB/non-BF16 format.
# ---------------------------------------------------------------------------
import json as _json
from pathlib import Path as _Path


def _make_synth_model(tmp_path, weights: dict) -> dict:
    from safetensors.torch import save_file
    tens = {}
    for q, (o, i) in weights.items():
        tens[q + ".weight"] = torch.randn(o, i).bfloat16()
    tens["model.norm.weight"] = torch.randn(64).bfloat16()  # 1D verbatim copy
    save_file(tens, str(_Path(tmp_path) / "model.safetensors"))
    (_Path(tmp_path) / "config.json").write_text(
        _json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    return {q: tens[q + ".weight"] for q in weights}


def _write_layer_config(tmp_path, assignment: dict) -> str:
    p = _Path(tmp_path) / "layer_config.json"
    p.write_text(_json.dumps(assignment))
    return str(p)


def test_stock_nvfp4_export_bitexact_vs_ct_codec(tmp_path):
    # Round-trip: a stock-NVFP4 Linear must ship EXACTLY the CT codec's own
    # output (no re-derivation — the M19 scale-fidelity guarantee).
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    from prismaquant import export_native_compressed as enc
    from safetensors.torch import load_file
    q = "model.layers.0.self_attn.o_proj"  # not a fused sibling -> singleton
    src = _make_synth_model(tmp_path, {q: (256, 512)})
    lc = _write_layer_config(tmp_path, {q: "NVFP4"})
    out = _Path(tmp_path) / "exp"
    export_nvfp4_cb(str(tmp_path), lc, str(out), col_weights={}, device="cpu")
    st = load_file(str(out / "model.safetensors"))
    ref = enc._quantize_2d(src[q].to("cpu"), "NVFP4")
    assert set(ref) == {"weight_packed", "weight_scale",
                        "weight_global_scale", "input_global_scale"}
    for suffix, t in ref.items():
        assert torch.equal(st[f"{q}.{suffix}"], t.cpu()), f"{suffix} not bit-exact"


def test_stock_fp8_export_bitexact_vs_ct_codec(tmp_path):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    from prismaquant import export_native_compressed as enc
    from safetensors.torch import load_file
    q = "model.layers.0.mlp.down_proj"
    src = _make_synth_model(tmp_path, {q: (256, 512)})
    lc = _write_layer_config(tmp_path, {q: "FP8_DYNAMIC"})  # -> FP8_E4M3
    out = _Path(tmp_path) / "exp"
    export_nvfp4_cb(str(tmp_path), lc, str(out), col_weights={}, device="cpu")
    st = load_file(str(out / "model.safetensors"))
    ref = enc._quantize_2d(src[q].to("cpu"), "FP8_E4M3")
    assert set(ref) == {"weight", "weight_scale"}
    for suffix, t in ref.items():
        assert torch.equal(st[f"{q}.{suffix}"], t.cpu()), f"{suffix} not bit-exact"


def test_mixed_container_config_groups_schema(tmp_path):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    from safetensors.torch import load_file
    qcb = "model.layers.0.mlp.gate_proj"       # CB   (in=256, 256-legal)
    qnv = "model.layers.0.self_attn.o_proj"    # NVFP4
    qfp = "model.layers.0.mlp.up_proj"         # FP8_DYNAMIC
    qbf = "model.layers.0.self_attn.q_proj"    # BF16
    _make_synth_model(tmp_path, {qcb: (128, 256), qnv: (256, 512),
                                 qfp: (256, 512), qbf: (256, 512)})
    lc = _write_layer_config(tmp_path, {
        qcb: "NVFP4_CB_K16", qnv: "NVFP4", qfp: "FP8_DYNAMIC", qbf: "BF16"})
    out = _Path(tmp_path) / "exp"
    export_nvfp4_cb(str(tmp_path), lc, str(out),
                    col_weights={qcb: torch.rand(256) + 0.05}, device="cpu")
    qc = _json.loads((out / "quant_config.json").read_text())
    groups = qc["config_groups"]
    # CB groups carry a "scheme" (custom vocab); stock CT groups do NOT (the
    # dispatch marker) and use the exact CT scheme vocabulary.
    cb_g = [g for g in groups.values() if "scheme" in g]
    nv_g = [g for g in groups.values() if g.get("format") == "nvfp4-pack-quantized"]
    fp_g = [g for g in groups.values() if g.get("format") == "float-quantized"]
    assert len(cb_g) == 1 and cb_g[0]["scheme"]["k"] == 16 and cb_g[0]["scheme"]["grid"] == "fp4"
    assert len(nv_g) == 1 and "scheme" not in nv_g[0]
    assert nv_g[0]["weights"]["num_bits"] == 4 and "input_activations" in nv_g[0]
    assert len(fp_g) == 1 and "scheme" not in fp_g[0]
    assert fp_g[0]["weights"]["num_bits"] == 8
    # stock targets are CT regexes and are NOT ignored; BF16 IS ignored.
    assert any("o_proj" in t and t.startswith("re:^") for t in nv_g[0]["targets"])
    assert qbf in qc["ignore"]
    assert qnv not in qc["ignore"] and qfp not in qc["ignore"]
    # on-disk tensors: NVFP4 quad, FP8 pair, BF16 verbatim, CB packed.
    st = load_file(str(out / "model.safetensors"))
    for suf in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"):
        assert f"{qnv}.{suf}" in st
    for suf in ("weight", "weight_scale"):
        assert f"{qfp}.{suf}" in st
    assert f"{qbf}.weight" in st and f"{qcb}.cb_qweight" in st
    assert qc["provenance"]["stock_ct_targets"] == 2
    assert qc["provenance"]["cb_targets"] == 1


def test_stock_rungs_no_longer_rejected_but_junk_still_is(tmp_path):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    q = "model.layers.0.self_attn.o_proj"
    _make_synth_model(tmp_path, {q: (256, 512)})
    out = _Path(tmp_path) / "exp"
    # stock rungs: accepted now.
    for fmt in ("NVFP4", "FP8_DYNAMIC"):
        lc = _write_layer_config(tmp_path, {q: fmt})
        export_nvfp4_cb(str(tmp_path), lc, str(out), col_weights={}, device="cpu")
    # a genuinely-unsupported format still hard-fails coverage.
    lc = _write_layer_config(tmp_path, {q: "MXFP4"})
    with pytest.raises(ValueError, match="cannot carry"):
        export_nvfp4_cb(str(tmp_path), lc, str(out), col_weights={}, device="cpu")


def test_fused_nvfp4_siblings_share_weight_global_scale(tmp_path):
    # q/k/v that all land on NVFP4 must ship ONE shared weight_global_scale
    # (else vLLM's fused qkv loader sees inconsistent per-tensor globals).
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    from prismaquant import export_native_compressed as enc
    from prismaquant.model_profiles import detect_profile
    from safetensors.torch import load_file
    qs = [f"model.layers.0.self_attn.{p}_proj" for p in "qkv"]
    src = _make_synth_model(tmp_path, {q: (256, 512) for q in qs})
    # Only meaningful if the profile actually groups q/k/v; else each is a
    # singleton and the assertion is vacuously the per-Linear scale.
    prof = None
    try:
        prof = detect_profile(str(tmp_path))
    except Exception:
        pass
    grouped = prof is not None and prof.fused_sibling_group(qs[0]) is not None
    lc = _write_layer_config(tmp_path, {q: "NVFP4" for q in qs})
    out = _Path(tmp_path) / "exp"
    export_nvfp4_cb(str(tmp_path), lc, str(out), col_weights={}, device="cpu")
    st = load_file(str(out / "model.safetensors"))
    globals_ = [st[f"{q}.weight_global_scale"] for q in qs]
    if grouped:
        assert all(torch.equal(globals_[0], g) for g in globals_[1:]), \
            "fused NVFP4 siblings must share one weight_global_scale"
        expected = torch.stack([
            enc.compute_nvfp4_global_real(src[q].to("cpu"), 16).reshape(())
            for q in qs]).max()
        # stored weight_global_scale is the DIVISOR 1/global_real.
        assert torch.allclose(globals_[0].reshape(()), 1.0 / expected, rtol=1e-4)


class _NestedVLMProfile:
    """Hybrid VLM name mapping: the LM nests under a `model.language_model.`
    infix that the allocator's recipe names strip (Qwen3.6-27B / Hy3 / DSv4)."""
    def source_tensor_name(self, qname):
        if qname.startswith("model.layers."):
            return "model.language_model.layers." + qname[len("model.layers."):]
        return qname
    def checkpoint_to_live_name(self, ckpt_key, *, multimodal=False):
        if ckpt_key.endswith(".weight_scale_inv"):
            return None
        if ckpt_key.startswith(("model.visual.", "model.vision_tower.")):
            return None
        if not multimodal and ckpt_key.startswith("model.language_model.layers."):
            return "model.layers." + ckpt_key[len("model.language_model.layers."):]
        return ckpt_key
    def fused_sibling_group(self, qname):
        return None


def test_exporter_resolves_nested_prefix_skeleton(tmp_path, monkeypatch):
    # Hybrid Qwen3.6-27B/Hy3/DSv4: the assignment uses recipe names
    # (model.layers.N.*) but the checkpoint nests the LM under
    # model.language_model.*. The exporter must resolve BOTH directions via the
    # profile: TENSORS ship under the CHECKPOINT convention, config_groups
    # targets under the SERVING-canonical one (3be09e4 — vLLM's class mapper
    # serves the LM at model.layers.* whatever the on-disk infix, so
    # checkpoint-namespace targets matched nothing and every layer loaded
    # unquantized, 2026-07-22).
    from prismaquant import model_profiles
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    from safetensors.torch import save_file, load_file
    canon = "model.layers.0.mlp.gate_proj"
    nested = "model.language_model.layers.0.mlp.gate_proj"
    tens = {
        nested + ".weight": torch.randn(128, 256).bfloat16(),        # CB target
        "model.language_model.norm.weight": torch.randn(64).bfloat16(),   # 1-D verbatim
        "model.visual.patch_embed.weight": torch.randn(32, 64).bfloat16(),  # visual verbatim
    }
    save_file(tens, str(_Path(tmp_path) / "model.safetensors"))
    (_Path(tmp_path) / "config.json").write_text(
        _json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
    lc = _write_layer_config(tmp_path, {canon: "NVFP4_CB_K16"})
    monkeypatch.setattr(model_profiles, "detect_profile",
                        lambda *a, **k: _NestedVLMProfile())
    out = _Path(tmp_path) / "exp"
    export_nvfp4_cb(str(tmp_path), lc, str(out),
                    col_weights={canon: torch.rand(256) + 0.05}, device="cpu")
    st = load_file(str(out / "model.safetensors"))
    # Packed tensor carries the NESTED (checkpoint/vLLM) name, not the recipe one.
    assert f"{nested}.cb_qweight" in st, sorted(st)
    assert f"{canon}.cb_qweight" not in st
    # config_groups target is the CANONICAL name so the plugin matches the
    # modules vLLM actually instantiates. (Artifacts exported before 3be09e4
    # carry checkpoint-namespace targets; the plugin canonicalizes those at
    # parse time — e765301 — so back-compat lives there, not here.)
    qc = _json.loads((out / "quant_config.json").read_text())
    cb_g = next(g for g in qc["config_groups"].values() if "scheme" in g)
    assert canon in cb_g["targets"] and nested not in cb_g["targets"]
    # Non-target tensors copied verbatim under their checkpoint names.
    assert "model.language_model.norm.weight" in st
    assert "model.visual.patch_embed.weight" in st


def test_codebook_pqcb_sidecar_contract(tmp_path):
    # Codebooks ship in cb_codebooks.pqcb (safetensors under a non-globbed
    # extension), named by each scheme's codebook_ref, and are NOT in
    # model.safetensors — the exact contract plugins/gridbook config.py
    # get_codebooks() -> load_file(model_dir/cb_codebooks.pqcb) consumes, and
    # linear.py looks up as codebooks[n] for n in codebook_ref. (Format-level
    # fixture: the plugin's own reader is not importable without vLLM.)
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    from safetensors.torch import load_file
    qcb = "model.layers.0.mlp.gate_proj"
    qfp = "model.layers.1.mlp.gate_proj"
    _make_synth_model(tmp_path, {qcb: (128, 256), qfp: (128, 256)})
    lc = _write_layer_config(tmp_path, {qcb: "NVFP4_CB_K16", qfp: "FP8_CB_K40"})
    out = _Path(tmp_path) / "exp"
    export_nvfp4_cb(str(tmp_path), lc, str(out),
                    col_weights={qcb: torch.rand(256) + 0.05,
                                 qfp: torch.rand(256) + 0.05}, device="cpu")
    qc = _json.loads((out / "quant_config.json").read_text())
    cfg = _json.loads((out / "config.json").read_text())
    assert qc["codebook_file"] == "cb_codebooks.pqcb"
    assert cfg["quantization_config"]["codebook_file"] == "cb_codebooks.pqcb"
    # The sidecar loads and carries EXACTLY the codebook_ref names (the plugin
    # indexes codebooks[n] for every n in codebook_ref — all must resolve).
    pqcb = load_file(str(out / "cb_codebooks.pqcb"))
    refs = set()
    for g in qc["config_groups"].values():
        r = g["scheme"]["codebook_ref"]
        refs.update(r if isinstance(r, list) else [r])
    assert refs and set(pqcb) == refs, f"{set(pqcb)} != {refs}"
    for r in refs:
        assert pqcb[r].dtype == torch.float16  # grid-exact fp16 tables
    # Sidecar-only: no cb_codebook.* tensors leak into the globbed weight file.
    ot = load_file(str(out / "model.safetensors"))
    assert not any(k.startswith("cb_codebook") for k in ot)
