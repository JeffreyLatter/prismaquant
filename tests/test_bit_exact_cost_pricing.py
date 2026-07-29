"""Bit-exact pricing: weight_mse == 0.0 short-circuits to zero dloss ONLY
for formats whose activation path is the identity.

Regression #1 (the original motivation): MX re-encodes of QAT/FP8 sources
store the source weights verbatim (``weight_mse == 0.0`` exactly), yet the
cost pipeline records a positive ``output_mse`` for the weight-only
passthrough formats too (kernel dequant dtype noise), which inverted
dominance against lossy k-quants.

Regression #2 (the review catch): the short-circuit must NOT fire for
W·A· formats. ``measure_quant_cost`` applies
``activation_quantize_dequantize(X)`` before computing ``output_mse``, so
for a weight-lossless activation-quantizing format (MXFP4 re-encode of an
MXFP4-packed source, MXFP8_E4M3 of an FP8-block source, ...) that
output_mse is REAL A-side error. Pricing those entries at dloss 0.0 makes
them the unbeatable global minimum at any budget while the served
activations are still quantized.

Contract pinned here:
  - weight-bit-exact + PASSTHROUGH-activation format (act_bits is None)
    short-circuits to dloss 0.0 ("bit_exact" source);
  - weight-bit-exact + ACTIVATION-QUANTIZING format keeps its measured
    output_mse pricing — at entry, build_candidates, and packed-group
    aggregation level;
  - unknown formats and explicit-cost_source entries never short-circuit;
  - the dtype-level fact used for the gate (FormatSpec.act_bits is None
    <=> activation_quantize_dequantize is the identity) holds for every
    registered format.
"""
from __future__ import annotations

import torch

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    build_candidates,
    cost_entry_is_bit_exact,
    cost_entry_predicted_dloss,
    cost_entry_source,
    cost_entry_uses_measured_output_mse,
)
from prismaquant.allocator_solver import solve_with_promotion


_STATS = {
    "h_trace": 2.5, "n_params": 1024 * 1024,
    "in_features": 1024, "out_features": 1024,
}

# A weight-only re-encode format: activations pass through unquantized.
_PASSTHROUGH_FMT = "MXFP8A16"
# A W·A· format: serving quantizes the input activations to FP4.
_ACTQUANT_FMT = "MXFP4"


def _bit_exact_entry():
    # Lossless weights; output probe carries A-side error and/or noise.
    return {"weight_mse": 0.0, "output_mse": 6.3e-3,
            "rel_output_mse": 8.7e-3}


def _lossy_entry():
    return {"weight_mse": 2.0e-6, "output_mse": 1.4e-3,
            "rel_output_mse": 1.9e-3}


def test_registry_act_bits_declaration_matches_activation_callable():
    """act_bits is None <=> activation_quantize_dequantize is the identity.

    This is the dtype-level fact the bit-exact short-circuit relies on;
    pin it against the actual callables so the registry cannot drift.
    """
    torch.manual_seed(1234)
    x = torch.randn(4, 512, dtype=torch.float32) * 3.1
    for name, spec in sorted(fr.REGISTRY.items()):
        out = spec.activation_quantize_dequantize(x.clone())
        if spec.act_bits is None:
            assert not spec.act_quant_changes_input
            assert torch.equal(out, x), (
                f"{name}: act_bits=None declares a passthrough activation "
                "path, but the callable changed the input")
        else:
            assert spec.act_quant_changes_input
            assert not torch.equal(out, x), (
                f"{name}: act_bits={spec.act_bits} declares activation "
                "quantization, but the callable is the identity")


def test_passthrough_activation_bit_exact_prices_at_zero_dloss():
    entry = _bit_exact_entry()
    assert cost_entry_is_bit_exact(entry, _PASSTHROUGH_FMT)
    assert cost_entry_predicted_dloss(
        _STATS, entry, format_name=_PASSTHROUGH_FMT) == 0.0
    # gain multipliers cannot resurrect a cost that is zero by construction
    assert cost_entry_predicted_dloss(
        _STATS, entry, gain=3.7, format_name=_PASSTHROUGH_FMT) == 0.0
    assert cost_entry_source(_STATS, entry, _PASSTHROUGH_FMT) == "bit_exact"
    assert not cost_entry_uses_measured_output_mse(
        _STATS, entry, _PASSTHROUGH_FMT)


def test_activation_quantizing_bit_exact_keeps_measured_a_side_cost():
    """weight_mse == 0.0 on a W·A· format proves nothing about the output:
    measure_quant_cost quantized the activations before measuring
    output_mse, so the measured cost is real and must be charged."""
    entry = _bit_exact_entry()
    assert not cost_entry_is_bit_exact(entry, _ACTQUANT_FMT)
    assert cost_entry_uses_measured_output_mse(_STATS, entry, _ACTQUANT_FMT)
    assert cost_entry_source(_STATS, entry, _ACTQUANT_FMT) == "output_mse"
    expected = 0.5 * _STATS["h_trace"] * entry["output_mse"]
    got = cost_entry_predicted_dloss(_STATS, entry, format_name=_ACTQUANT_FMT)
    assert abs(got - expected) < 1e-12
    # Every registered activation-quantizing format is gated the same way.
    for name, spec in sorted(fr.REGISTRY.items()):
        if spec.act_quant_changes_input:
            assert not cost_entry_is_bit_exact(_bit_exact_entry(), name), name


def test_unknown_format_never_short_circuits():
    """No format identity, no proof: the caller that cannot name the
    format keeps the conservative (pre-shortcut) pricing."""
    entry = _bit_exact_entry()
    assert not cost_entry_is_bit_exact(entry)
    assert not cost_entry_is_bit_exact(entry, None)
    assert not cost_entry_is_bit_exact(entry, "NOT_A_FORMAT")
    assert cost_entry_uses_measured_output_mse(_STATS, entry)
    expected = 0.5 * _STATS["h_trace"] * entry["output_mse"]
    assert abs(cost_entry_predicted_dloss(_STATS, entry) - expected) < 1e-12


def test_explicit_cost_source_entries_are_never_bit_exact():
    """Entries with an explicit cost_source (e.g. the production-render
    score pipeline) default weight_mse to 0.0 as a placeholder, not a
    measurement — the short-circuit must not override their own pricing,
    matching the precedence cost_entry_source gives the explicit source."""
    entry = {"predicted_dloss": 12.0, "weight_mse": 0.0, "output_mse": 0.0,
             "output_mse_measured": False,
             "cost_source": "production_render_score"}
    assert not cost_entry_is_bit_exact(entry, _PASSTHROUGH_FMT)
    assert cost_entry_predicted_dloss(
        _STATS, entry, format_name=_PASSTHROUGH_FMT) == 12.0
    assert cost_entry_source(
        _STATS, entry, _PASSTHROUGH_FMT) == "production_render_score"


def test_lossy_entries_keep_measured_output_mse_pricing():
    entry = _lossy_entry()
    assert not cost_entry_is_bit_exact(entry, _PASSTHROUGH_FMT)
    assert cost_entry_uses_measured_output_mse(_STATS, entry, _PASSTHROUGH_FMT)
    expected = 0.5 * _STATS["h_trace"] * entry["output_mse"]
    assert abs(
        cost_entry_predicted_dloss(
            _STATS, entry, format_name=_PASSTHROUGH_FMT
        ) - expected
    ) < 1e-12
    # Near-zero is not zero: only an exact 0.0 proves losslessness.
    tiny = dict(entry, weight_mse=1e-300)
    assert not cost_entry_is_bit_exact(tiny, _PASSTHROUGH_FMT)


def test_build_candidates_splits_passthrough_and_actquant_bit_exact():
    stats = {"row": dict(_STATS)}
    costs = {"row": {_PASSTHROUGH_FMT: _bit_exact_entry(),
                     _ACTQUANT_FMT: _bit_exact_entry(),
                     "Q4_K": _lossy_entry()}}
    specs = [fr.get_format(_PASSTHROUGH_FMT), fr.get_format(_ACTQUANT_FMT),
             fr.get_format("Q4_K")]
    cands = build_candidates(stats, costs, specs)
    by_fmt = {c.fmt: c for c in cands["row"]}
    assert by_fmt[_PASSTHROUGH_FMT].predicted_dloss == 0.0
    expected_a_side = 0.5 * _STATS["h_trace"] * _bit_exact_entry()["output_mse"]
    assert abs(by_fmt[_ACTQUANT_FMT].predicted_dloss - expected_a_side) < 1e-12
    assert by_fmt["Q4_K"].predicted_dloss > 0.0
    # The A-side cost keeps MXFP4 comparable with the lossy k-quant on the
    # SAME footing (both model their compute-path activation error) instead
    # of an unbeatable 0.0.
    assert by_fmt[_ACTQUANT_FMT].predicted_dloss > by_fmt["Q4_K"].predicted_dloss


def test_zero_dloss_fewer_bits_strictly_dominates_in_the_dp():
    """A measured-zero-dloss candidate at FEWER bits must displace a
    positive-dloss candidate at MORE bits whenever the DP funds an
    upgrade — dloss == 0 is a valid, optimal, measured cost."""
    from prismaquant.allocator_solver import Candidate

    stats = {}
    cands = {}
    menu = ["IQ2_XXS", "MXFP4", "Q4_K"]
    dloss = {"IQ2_XXS": 50.0, "MXFP4": 0.0, "Q4_K": 1.7}
    for i in range(4):
        name = f"model.layers.{i}.self_attn.o_proj"
        n = 1 << 20
        stats[name] = {"h_trace": 1.0, "n_params": n,
                       "in_features": 1024, "out_features": 1024}
        cands[name] = []
        for f in menu:
            bpp = fr.get_format(f).effective_bits
            cands[name].append(Candidate(
                fmt=f, bits_per_param=bpp,
                memory_bytes=int(round(bpp * n / 8.0)),
                predicted_dloss=dloss[f]))
    specs = {f: fr.get_format(f) for f in menu}
    rank = {f: i for i, f in enumerate(menu)}

    for target in (4.3, 4.6, 16.0):  # fits MXFP4 only / both / everything
        assign, achieved = solve_with_promotion(
            stats, cands, target, specs, rank, bit_precision=0.001)
        assert assign is not None
        assert achieved <= target + 0.01
        assert all(f == "MXFP4" for f in assign.values()), (
            f"target={target}: zero-dloss fewer-bits format must win, "
            f"got {assign}")


def test_packed_group_pricing_splits_passthrough_and_actquant():
    """Group level: members bit-exact at a passthrough-activation format
    sum to a zero group cost; the same members at a weight-lossless W·A·
    format carry the summed A-side cost."""
    from prismaquant.allocator_candidates import (
        _PACKED_GROUP_MARKER,
        aggregate_packed_serving_groups,
    )

    class _P:
        def packed_expert_format_group(self, name):
            return "g" if ".experts." in name else None

    n_members = 4
    stats, costs = {}, {}
    specs = [fr.get_format(_PASSTHROUGH_FMT), fr.get_format(_ACTQUANT_FMT)]
    for e in range(n_members):
        name = f"model.layers.0.mlp.experts.{e}.gate_proj"
        stats[name] = {"h_trace": 0.5, "n_params": 65536,
                       "in_features": 256, "out_features": 256}
        costs[name] = {_PASSTHROUGH_FMT: _bit_exact_entry(),
                       _ACTQUANT_FMT: _bit_exact_entry()}
    cands = build_candidates(stats, costs, specs)
    _stats2, _costs2, cands2 = aggregate_packed_serving_groups(
        stats, costs, specs, cands, _P())
    super_name = next(n for n in cands2 if _PACKED_GROUP_MARKER in n)
    by_fmt = {c.fmt: c for c in cands2[super_name]}
    assert by_fmt[_PASSTHROUGH_FMT].predicted_dloss == 0.0
    per_member = 0.5 * 0.5 * _bit_exact_entry()["output_mse"]
    assert abs(
        by_fmt[_ACTQUANT_FMT].predicted_dloss - n_members * per_member
    ) < 1e-12
