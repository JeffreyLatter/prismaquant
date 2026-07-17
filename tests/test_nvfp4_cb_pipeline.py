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

from prismaquant import format_registry as fr
from prismaquant import layer_config as lcfg
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
    assert prof.runtime == "vllm_prismaquant_plugin"


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
