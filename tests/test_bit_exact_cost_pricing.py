"""Bit-exact (measured weight_mse == 0.0) formats price at zero dloss.

Regression: MX re-encodes of QAT/FP8 sources are
bit-exact — ``weight_mse == 0.0`` exactly, i.e. the format stores the
source weights verbatim — yet the cost pipeline also records a small
positive ``output_mse`` for them (kernel dequant dtype + activation
sampling noise; a bit-identical weight tensor cannot really perturb the
output). The allocator's measured-output_mse-first pricing then charged
bit-exact MXFP4 (output_mse 6.3e-3 of pure noise) ABOVE lossy Q4_K
(output_mse 1.4e-3) on every expert row, so no MX format was ever
selected at any rung or budget.

Contract pinned here: a measured-zero-dloss candidate at FEWER bits
strictly dominates a positive-dloss candidate at MORE bits — the DP must
select it whenever either fits; dloss == 0 is a valid, optimal, measured
cost (the allocator's ``_log_error_values`` documents zero as real for
FP8-native passthrough).
"""
from __future__ import annotations

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


def _bit_exact_entry():
    # The real-world contradiction: lossless weights, noisy output probe.
    return {"weight_mse": 0.0, "output_mse": 6.3e-3,
            "rel_output_mse": 8.7e-3}


def _lossy_entry():
    return {"weight_mse": 2.0e-6, "output_mse": 1.4e-3,
            "rel_output_mse": 1.9e-3}


def test_measured_zero_weight_mse_prices_at_zero_dloss():
    assert cost_entry_is_bit_exact(_bit_exact_entry())
    assert cost_entry_predicted_dloss(_STATS, _bit_exact_entry()) == 0.0
    # gain multipliers cannot resurrect a cost that is zero by construction
    assert cost_entry_predicted_dloss(
        _STATS, _bit_exact_entry(), gain=3.7) == 0.0
    assert cost_entry_source(_STATS, _bit_exact_entry()) == "bit_exact"
    assert not cost_entry_uses_measured_output_mse(
        _STATS, _bit_exact_entry())


def test_explicit_cost_source_entries_are_never_bit_exact():
    """Entries with an explicit cost_source (e.g. the production-render
    score pipeline) default weight_mse to 0.0 as a placeholder, not a
    measurement — the short-circuit must not override their own pricing,
    matching the precedence cost_entry_source gives the explicit source."""
    entry = {"predicted_dloss": 12.0, "weight_mse": 0.0, "output_mse": 0.0,
             "output_mse_measured": False,
             "cost_source": "production_render_score"}
    assert not cost_entry_is_bit_exact(entry)
    assert cost_entry_predicted_dloss(_STATS, entry) == 12.0
    assert cost_entry_source(_STATS, entry) == "production_render_score"


def test_lossy_entries_keep_measured_output_mse_pricing():
    entry = _lossy_entry()
    assert not cost_entry_is_bit_exact(entry)
    assert cost_entry_uses_measured_output_mse(_STATS, entry)
    expected = 0.5 * _STATS["h_trace"] * entry["output_mse"]
    assert abs(cost_entry_predicted_dloss(_STATS, entry) - expected) < 1e-12
    # Near-zero is not zero: only an exact 0.0 proves losslessness.
    tiny = dict(entry, weight_mse=1e-300)
    assert not cost_entry_is_bit_exact(tiny)


def test_build_candidates_carries_zero_dloss_for_bit_exact_formats():
    stats = {"row": dict(_STATS)}
    costs = {"row": {"MXFP4": _bit_exact_entry(), "Q4_K": _lossy_entry()}}
    specs = [fr.get_format("MXFP4"), fr.get_format("Q4_K")]
    cands = build_candidates(stats, costs, specs)
    by_fmt = {c.fmt: c for c in cands["row"]}
    assert by_fmt["MXFP4"].predicted_dloss == 0.0
    assert by_fmt["Q4_K"].predicted_dloss > 0.0
    assert by_fmt["MXFP4"].bits_per_param < by_fmt["Q4_K"].bits_per_param


def test_zero_dloss_fewer_bits_strictly_dominates_in_the_dp():
    """MXFP4 (4.25 bpw, measured dloss 0) must displace Q4_K (4.5 bpw,
    dloss > 0) whenever the DP funds an upgrade — at a budget that fits
    either, and at an unconstrained budget where everything fits."""
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
            from prismaquant.allocator_solver import Candidate
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


def test_bit_exact_members_sum_to_zero_group_cost():
    """A packed group whose members are all bit-exact at a format prices
    the whole group at exactly zero for that format."""
    from prismaquant.allocator_candidates import (
        _PACKED_GROUP_MARKER,
        aggregate_packed_serving_groups,
    )

    class _P:
        def packed_expert_format_group(self, name):
            return "g" if ".experts." in name else None

    stats, costs, cands = {}, {}, {}
    specs = [fr.get_format("MXFP4"), fr.get_format("Q4_K")]
    for e in range(4):
        name = f"model.layers.0.mlp.experts.{e}.gate_proj"
        stats[name] = {"h_trace": 0.5, "n_params": 65536,
                       "in_features": 256, "out_features": 256}
        costs[name] = {"MXFP4": _bit_exact_entry(), "Q4_K": _lossy_entry()}
    cands = build_candidates(stats, costs, specs)
    _stats2, _costs2, cands2 = aggregate_packed_serving_groups(
        stats, costs, specs, cands, _P())
    super_name = next(n for n in cands2 if _PACKED_GROUP_MARKER in n)
    by_fmt = {c.fmt: c for c in cands2[super_name]}
    assert by_fmt["MXFP4"].predicted_dloss == 0.0
    assert by_fmt["Q4_K"].predicted_dloss > 0.0
