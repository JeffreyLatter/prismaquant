"""Unit tests for prismaquant.block_clado solver and payload."""
from __future__ import annotations

import json

import pytest
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant import block_clado as bc


def _mk_unit(name: str, block_id: str, options):
    member_qnames = (f"{name}.weight",)
    opts = []
    for fmt, omega_ii, bits_per_param, memory_bytes in options:
        opts.append(bc.FormatCost(
            fmt=fmt,
            omega_ii=float(omega_ii),
            bits_per_param=float(bits_per_param),
            memory_bytes=int(memory_bytes),
        ))
    opts.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
    return bc.DecisionUnit(
        name=name,
        block_id=block_id,
        member_qnames=member_qnames,
        options=tuple(opts),
    )


# ---------------------------------------------------------------------------
# block_id and fused-group helpers
# ---------------------------------------------------------------------------


def test_block_id_for_transformer_layers():
    assert bc.block_id_from_qname("model.layers.0.self_attn.q_proj") == "model.layers.0"
    assert bc.block_id_from_qname("model.layers.27.mlp.down_proj") == "model.layers.27"


def test_block_id_for_singletons():
    assert bc.block_id_from_qname("lm_head") == "lm_head"
    assert bc.block_id_from_qname("model.embed_tokens") == "model.embed_tokens"


def test_pair_key_roundtrip():
    key = bc._pair_key("NVFP4", "MXFP8_E4M3")
    a, b = bc._parse_pair_key(key)
    assert (a, b) == ("NVFP4", "MXFP8_E4M3")


def test_discover_units_drops_runtime_illegal_mxfp8_shape():
    model = nn.Module()
    model.small = nn.Linear(5120, 48, bias=False)
    model.large = nn.Linear(5120, 128, bias=False)
    formats = [fr.get_format("NVFP4"), fr.get_format("MXFP8"), fr.get_format("BF16")]

    blocks, singletons, _ = bc.discover_units(model, None, formats)
    units = {unit.name: unit for unit in singletons}
    for unit_list in blocks.values():
        units.update({unit.name: unit for unit in unit_list})

    small_formats = {option.fmt for option in units["small"].options}
    large_formats = {option.fmt for option in units["large"].options}

    assert small_formats == {"NVFP4", "BF16"}
    assert "MXFP8_E4M3" in large_formats


# ---------------------------------------------------------------------------
# score_block_assignment
# ---------------------------------------------------------------------------


def test_score_block_assignment_pair_term():
    # Two units, two formats each: BF16 (cheap, no perturbation) and a low-bit
    # option whose pair interaction is +0.5 (synergistic worsening).
    block_id = "model.layers.0"
    unit_a = _mk_unit("model.layers.0.attn.qkv", block_id, [
        ("BF16", 0.0, 16.0, 1024),
        ("LOWBIT", 0.1, 4.0, 256),
    ])
    unit_b = _mk_unit("model.layers.0.attn.o", block_id, [
        ("BF16", 0.0, 16.0, 1024),
        ("LOWBIT", 0.2, 4.0, 256),
    ])
    pair = bc.BlockPair(
        unit_a=unit_a.name,
        unit_b=unit_b.name,
        block_id=block_id,
        omega_ij={
            ("BF16", "BF16"): 0.0,
            ("BF16", "LOWBIT"): 0.0,
            ("LOWBIT", "BF16"): 0.0,
            ("LOWBIT", "LOWBIT"): 0.5,
        },
    )
    cost_bf16, bits_bf16 = bc.score_block_assignment(
        [unit_a, unit_b],
        {unit_a.name: "BF16", unit_b.name: "BF16"},
        [pair],
    )
    assert cost_bf16 == pytest.approx(0.0)
    assert bits_bf16 == pytest.approx(16384.0)  # 2 × 1024 × 8
    cost_lowbit, bits_lowbit = bc.score_block_assignment(
        [unit_a, unit_b],
        {unit_a.name: "LOWBIT", unit_b.name: "LOWBIT"},
        [pair],
    )
    # 0.1 + 0.2 unary + 0.5 pair = 0.8
    assert cost_lowbit == pytest.approx(0.8)
    assert bits_lowbit == pytest.approx(4096.0)


def test_score_block_assignment_missing_pair_treated_as_zero():
    block_id = "model.layers.0"
    unit_a = _mk_unit("a", block_id, [("BF16", 0.0, 16.0, 1024), ("LO", 0.1, 4.0, 256)])
    unit_b = _mk_unit("b", block_id, [("BF16", 0.0, 16.0, 1024), ("LO", 0.2, 4.0, 256)])
    pair = bc.BlockPair(
        unit_a=unit_a.name, unit_b=unit_b.name, block_id=block_id,
        omega_ij={("LO", "LO"): 0.5},  # missing other entries
    )
    cost, _ = bc.score_block_assignment(
        [unit_a, unit_b],
        {unit_a.name: "BF16", unit_b.name: "BF16"},
        [pair],
    )
    # No omega_ij entry for ("BF16", "BF16") → treated as 0.
    assert cost == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# enumerate_block_states (Pareto filter)
# ---------------------------------------------------------------------------


def test_enumerate_block_states_recovers_known_optimum():
    block_id = "model.layers.0"
    unit_a = _mk_unit("a", block_id, [("BF16", 0.0, 16.0, 1024), ("LO", 0.1, 4.0, 256)])
    unit_b = _mk_unit("b", block_id, [("BF16", 0.0, 16.0, 1024), ("LO", 0.2, 4.0, 256)])
    pair = bc.BlockPair(
        unit_a=unit_a.name, unit_b=unit_b.name, block_id=block_id,
        omega_ij={
            ("BF16", "BF16"): 0.0,
            ("BF16", "LO"): 0.0,
            ("LO", "BF16"): 0.0,
            ("LO", "LO"): 0.5,
        },
    )
    states = bc.enumerate_block_states([unit_a, unit_b], [pair])
    # Possible: BB (0, 16384), BL (0.2, 10240), LB (0.1, 10240), LL (0.8, 4096).
    # After Pareto filter (sorted by bits asc, kept if cost strictly improves):
    #   LL (0.8, 4096), LB (0.1, 10240) — LB is cheaper and bigger so kept.
    #   BL would be (0.2, 10240) — same bits as LB but worse cost, dropped.
    #   BB (0.0, 16384) — strictly improves cost, kept.
    bits_to_cost = {state.bits_total: state.cost for state in states}
    assert pytest.approx(bits_to_cost[4096.0]) == 0.8
    assert pytest.approx(bits_to_cost[10240.0]) == 0.1
    assert pytest.approx(bits_to_cost[16384.0]) == 0.0


def test_enumerate_block_states_rejects_pathologically_large_blocks():
    # 1000 units × 5 formats = 5^1000, impossible.
    block_id = "model.layers.0"
    units = [_mk_unit(f"u{i}", block_id, [
        ("F0", 0.0, 16.0, 1024),
        ("F1", 0.1, 8.0, 512),
        ("F2", 0.2, 4.0, 256),
        ("F3", 0.3, 3.0, 192),
        ("F4", 0.4, 2.0, 128),
    ]) for i in range(1000)]
    with pytest.raises(ValueError, match="format tuples"):
        bc.enumerate_block_states(units, [], max_states=1024)


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------


def test_singleton_states_have_no_pair_terms():
    unit = _mk_unit("lm_head", "lm_head", [
        ("BF16", 0.0, 16.0, 4096),
        ("NVFP4", 0.05, 4.5, 1152),
    ])
    states = bc.enumerate_singleton_states(unit)
    # Both states are Pareto-optimal — different bits, different costs.
    assert len(states) == 2
    by_bits = {s.bits_total: s.cost for s in states}
    assert pytest.approx(by_bits[32768.0]) == 0.0
    assert pytest.approx(by_bits[9216.0]) == 0.05


# ---------------------------------------------------------------------------
# Lagrangian / λ-sweep
# ---------------------------------------------------------------------------


def test_lambda_sweep_traces_two_block_frontier():
    # Two independent blocks, each with a clear "small but expensive" vs
    # "large but cheap" tradeoff.  Sweeping λ should hit (cheap, cheap),
    # mixed, and (expensive, expensive) endpoints.
    block_a = [
        bc.BlockSolution("A", {"x": "BF16"}, cost=0.0, bits_total=16000.0),
        bc.BlockSolution("A", {"x": "LO"},   cost=1.0, bits_total=4000.0),
    ]
    block_b = [
        bc.BlockSolution("B", {"y": "BF16"}, cost=0.0, bits_total=16000.0),
        bc.BlockSolution("B", {"y": "LO"},   cost=2.0, bits_total=4000.0),
    ]
    block_states = {"A": block_a, "B": block_b}
    results = bc.lambda_sweep(
        block_states,
        lambda_min=1e-9,
        lambda_max=1.0,
        n_lambdas=21,
    )
    # Should recover at least three distinct frontier points: (0, 32k),
    # (1.0, 20k) i.e. only A goes low, and (3.0, 8k) i.e. both go low.
    bits_seen = {round(r.bits_total) for r in results}
    cost_by_bits = {round(r.bits_total): r.cost_total for r in results}
    assert 32000 in bits_seen
    assert 8000 in bits_seen
    assert pytest.approx(cost_by_bits[32000]) == 0.0
    assert pytest.approx(cost_by_bits[8000]) == 3.0


def test_lambda_sweep_skips_dominated_pockets_but_finds_endpoints():
    # Strictly convex frontier: all 4 corners reachable.
    block_a = [
        bc.BlockSolution("A", {"x": "HI"}, cost=0.0, bits_total=160.0),
        bc.BlockSolution("A", {"x": "MID"}, cost=0.5, bits_total=80.0),
        bc.BlockSolution("A", {"x": "LO"},  cost=1.5, bits_total=40.0),
    ]
    results = bc.lambda_sweep(
        {"A": block_a},
        lambda_min=1e-9,
        lambda_max=1.0,
        n_lambdas=51,
    )
    bits = sorted({round(r.bits_total) for r in results})
    assert 160 in bits
    assert 80 in bits  # MID lies on the convex hull, λ-sweep finds it
    assert 40 in bits


# ---------------------------------------------------------------------------
# Budget DP (multi-choice knapsack)
# ---------------------------------------------------------------------------


def test_solve_budget_finds_smallest_cost_within_budget():
    # Block A has options at (cost=0, bits=160), (cost=1, bits=80).
    # Block B has options at (cost=0, bits=160), (cost=2, bits=80).
    # Budget 200 bits forces at least one block to go low.  Minimal-cost
    # solution: A goes low (cost 1), B stays high (160 bits).  Total bits =
    # 80 + 160 = 240 > 200 — infeasible at 200.  Try budget 240 → feasible.
    block_a = [
        bc.BlockSolution("A", {"x": "HI"}, cost=0.0, bits_total=160.0),
        bc.BlockSolution("A", {"x": "LO"}, cost=1.0, bits_total=80.0),
    ]
    block_b = [
        bc.BlockSolution("B", {"y": "HI"}, cost=0.0, bits_total=160.0),
        bc.BlockSolution("B", {"y": "LO"}, cost=2.0, bits_total=80.0),
    ]
    states = {"A": block_a, "B": block_b}
    result = bc.solve_budget(states, bits_budget=240.0, bit_precision_bits=8.0)
    assert result is not None
    assert result.bits_total <= 240.0 + 1e-6
    assert pytest.approx(result.cost_total) == 1.0
    assert result.assignment == {"x": "LO", "y": "HI"}


def test_solve_budget_returns_none_when_infeasible():
    block = [bc.BlockSolution("A", {"x": "HI"}, cost=0.0, bits_total=200.0)]
    result = bc.solve_budget({"A": block}, bits_budget=100.0, bit_precision_bits=10.0)
    assert result is None


# ---------------------------------------------------------------------------
# Payload roundtrip
# ---------------------------------------------------------------------------


def test_payload_roundtrip_preserves_units_and_pairs():
    block_id = "model.layers.0"
    unit_a = _mk_unit("a", block_id, [("BF16", 0.0, 16.0, 1024), ("LO", 0.1, 4.0, 256)])
    unit_b = _mk_unit("b", block_id, [("BF16", 0.0, 16.0, 1024), ("LO", 0.2, 4.0, 256)])
    singleton = _mk_unit("lm_head", "lm_head", [
        ("BF16", 0.0, 16.0, 4096),
        ("LO", 0.05, 4.5, 1152),
    ])
    pair = bc.BlockPair(
        unit_a=unit_a.name, unit_b=unit_b.name, block_id=block_id,
        omega_ij={
            ("BF16", "BF16"): 0.0,
            ("BF16", "LO"): 0.0,
            ("LO", "BF16"): 0.0,
            ("LO", "LO"): 0.5,
        },
    )
    payload = bc.units_and_pairs_to_payload(
        blocks={block_id: [unit_a, unit_b]},
        singletons=[singleton],
        pairs_by_block={block_id: [pair]},
        meta={"test_marker": True},
    )
    # JSON-serialisable
    serialised = json.loads(json.dumps(payload))
    blocks_back, singletons_back, pairs_back = bc.parse_payload(serialised)
    assert set(blocks_back.keys()) == {block_id}
    units_back = {u.name: u for u in blocks_back[block_id]}
    assert units_back["a"].member_qnames == ("a.weight",)
    a_options = {opt.fmt: opt for opt in units_back["a"].options}
    assert a_options["BF16"].omega_ii == pytest.approx(0.0)
    assert a_options["LO"].omega_ii == pytest.approx(0.1)
    assert set(a_options.keys()) == {"BF16", "LO"}
    assert {(p.unit_a, p.unit_b) for p in pairs_back[block_id]} == {("a", "b")}
    pair_back = pairs_back[block_id][0]
    assert pytest.approx(pair_back.omega_ij[("LO", "LO")]) == 0.5
    assert {u.name for u in singletons_back} == {"lm_head"}


# ---------------------------------------------------------------------------
# Total params back-derivation
# ---------------------------------------------------------------------------


def test_centered_measurement_recovers_center_kl_in_meta():
    """When ``meta.center_kl`` is non-zero, ``score_block_assignment`` should
    still recover sensible costs at the centered formats (zero unary, zero
    pairs)."""
    block_id = "model.layers.0"
    # Centered at NVFP4 for unit a, NVFP4 for unit b → omega_ii at NVFP4 is
    # 0 by convention.  At BF16 it's a delta from the centered KL, so it can
    # be negative.
    unit_a = _mk_unit("a", block_id, [
        ("BF16", -0.05, 16, 1024),  # delta from center
        ("NVFP4", 0.0, 4.5, 256),    # the centered format
        ("MXFP8_E4M3", 0.02, 8, 512),
    ])
    unit_b = _mk_unit("b", block_id, [
        ("BF16", -0.04, 16, 1024),
        ("NVFP4", 0.0, 4.5, 256),
        ("MXFP8_E4M3", 0.03, 8, 512),
    ])
    pair = bc.BlockPair(
        unit_a="a", unit_b="b", block_id=block_id,
        omega_ij={
            ("BF16", "BF16"): -0.01,  # interaction once both upgrade
            ("BF16", "NVFP4"): 0.0,
            ("NVFP4", "BF16"): 0.0,
            ("NVFP4", "NVFP4"): 0.0,
            ("BF16", "MXFP8_E4M3"): 0.005,
            ("MXFP8_E4M3", "BF16"): 0.005,
            ("MXFP8_E4M3", "MXFP8_E4M3"): 0.01,
            ("NVFP4", "MXFP8_E4M3"): 0.0,
            ("MXFP8_E4M3", "NVFP4"): 0.0,
        },
    )
    payload = bc.units_and_pairs_to_payload(
        blocks={block_id: [unit_a, unit_b]},
        singletons=[],
        pairs_by_block={block_id: [pair]},
        meta={"center_kl": 0.087, "centered": True},
    )
    # center_kl roundtrips
    assert bc.center_kl_from_payload(payload) == pytest.approx(0.087)

    # Score at the centered formats should be zero.
    units = [unit_a, unit_b]
    cost, bits = bc.score_block_assignment(
        units, {"a": "NVFP4", "b": "NVFP4"}, [pair],
    )
    assert cost == pytest.approx(0.0)

    # Score at all-BF16: −0.05 −0.04 + (−0.01) = −0.10.
    cost_bf16, _ = bc.score_block_assignment(
        units, {"a": "BF16", "b": "BF16"}, [pair],
    )
    assert cost_bf16 == pytest.approx(-0.10)


def test_total_param_count_recovers_member_count():
    # 1024 params × 16 bpp = 16384 bits → 2048 bytes.  Memory bytes / bpp
    # back-derives the param count.
    block_id = "model.layers.0"
    unit = _mk_unit("a", block_id, [("BF16", 0.0, 16.0, 2048), ("LO", 0.1, 4.0, 512)])
    payload = bc.units_and_pairs_to_payload(
        blocks={block_id: [unit]},
        singletons=[],
        pairs_by_block={block_id: []},
    )
    assert bc.total_param_count(payload) == 1024


# ---------------------------------------------------------------------------
# Build-block-states integration
# ---------------------------------------------------------------------------


def test_build_block_states_yields_pareto_states_per_block():
    block_id = "model.layers.0"
    unit_a = _mk_unit("a", block_id, [("BF16", 0.0, 16.0, 1024), ("LO", 0.1, 4.0, 256)])
    unit_b = _mk_unit("b", block_id, [("BF16", 0.0, 16.0, 1024), ("LO", 0.2, 4.0, 256)])
    singleton = _mk_unit("lm_head", "lm_head", [
        ("BF16", 0.0, 16.0, 4096),
        ("LO", 0.05, 4.5, 1152),
    ])
    pair = bc.BlockPair(
        unit_a="a", unit_b="b", block_id=block_id,
        omega_ij={
            ("BF16", "BF16"): 0.0,
            ("BF16", "LO"): 0.0,
            ("LO", "BF16"): 0.0,
            ("LO", "LO"): 0.5,
        },
    )
    payload = bc.units_and_pairs_to_payload(
        blocks={block_id: [unit_a, unit_b]},
        singletons=[singleton],
        pairs_by_block={block_id: [pair]},
    )
    states = bc.build_block_states(payload)
    assert set(states.keys()) == {block_id, "lm_head"}
    block_pareto = states[block_id]
    # Block has 4 raw states, 3 are Pareto-non-dominated.
    assert len(block_pareto) == 3
    singleton_pareto = states["lm_head"]
    assert len(singleton_pareto) == 2
