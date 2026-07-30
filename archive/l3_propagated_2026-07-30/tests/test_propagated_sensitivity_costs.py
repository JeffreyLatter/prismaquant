"""The propagated-sensitivity penalty must land in the field the allocator
actually reads — including on bit-exact rows.

Regression: ``cost_entry_predicted_dloss`` short-circuits a bit-exact row
(measured ``weight_mse == 0.0`` AND an identity activation path) to 0.0 by
construction, reading NEITHER ``output_mse`` NOR ``predicted_dloss``. The
propagated path picks its injection field from
``cost_entry_uses_measured_output_mse``, which returns False for exactly those
rows, so the penalty was written into ``predicted_dloss`` and then silently
discarded by the short-circuit — while ``total_scaled_member_penalty`` still
reported it as applied. An FP8_SOURCE row on a native-FP8 source is the live
case (its ``weight_mse`` is exactly 0.0 because the format's
``quantize_dequantize`` is the identity).

Contract pinned here:
  - a bit-exact row carrying a penalty prices at the penalty, not 0.0 (and not
    at its noise-level measured output_mse either);
  - the summary's ``total_scaled_member_penalty`` equals the total dloss the DP
    actually charges, i.e. Σ(after) − Σ(before);
  - the authoritative-source stamp is applied ONLY to bit-exact rows, and never
    overwrites a producer's own ``cost_source`` provenance.
"""
from __future__ import annotations

import pytest

from prismaquant.allocator_candidates import (
    cost_entry_predicted_dloss,
    cost_entry_source,
    cost_entry_uses_measured_output_mse,
)
from prismaquant.propagated_sensitivity_costs import (
    apply_propagated_sensitivity_penalty,
)

_NAME = "model.layers.0.mlp.down_proj"
_STAT = {"h_trace": 2.0, "n_params": 1024, "in_features": 32, "out_features": 32}
_PROPAGATED_KL = 0.05


def _stats():
    return {_NAME: dict(_STAT)}


def _bit_exact_passthrough_entry():
    """FP8_SOURCE on a native-FP8 source: the format stores the source bytes
    verbatim, so weight_mse is EXACTLY 0.0; output_mse is measured but is
    noise for an identity weight + identity activation path."""
    return {"weight_mse": 0.0, "output_mse": 3e-4, "rel_output_mse": 1e-3}


def _lossy_actquant_entry():
    return {"weight_mse": 1e-5, "output_mse": 2e-3, "rel_output_mse": 5e-3}


def _report(current_fmt: str, *, key: str = "unit0"):
    return {
        "rows": [{
            "key": key,
            "members": [_NAME],
            "propagated_kl": _PROPAGATED_KL,
            "bits_delta": 4.0,
            "candidate_lane_override": {_NAME: current_fmt},
        }],
    }


def test_bit_exact_row_charges_the_propagated_penalty():
    """The whole point: a row the allocator would price at 0.0 must still be
    charged the propagated penalty."""
    stats = _stats()
    costs = {_NAME: {"FP8_SOURCE": _bit_exact_passthrough_entry()}}

    # Sanity: before the penalty, this row prices at exactly 0.0 (short-circuit).
    assert cost_entry_predicted_dloss(
        stats[_NAME], costs[_NAME]["FP8_SOURCE"],
        format_name="FP8_SOURCE") == 0.0

    adjusted, summary = apply_propagated_sensitivity_penalty(
        costs,
        stats=stats,
        report=_report("FP8_SOURCE"),
        scale=1.0,
        target_format="BF16",
        format_extrapolation="current_only",
    )
    entry = adjusted[_NAME]["FP8_SOURCE"]

    # current_only + a single member => the penalty is the measured KL exactly.
    assert entry["propagated_kl_penalty"] == pytest.approx(_PROPAGATED_KL)
    assert entry[
        "base_predicted_dloss_before_propagated_serving_unit_penalty"] == 0.0

    charged = cost_entry_predicted_dloss(
        stats[_NAME], entry, format_name="FP8_SOURCE")
    assert charged == pytest.approx(_PROPAGATED_KL), (
        "bit-exact row must be charged the propagated penalty, not "
        f"short-circuited to 0.0 (got {charged})")
    # ...and NOT priced from the noise-level measured output_mse.
    assert charged != pytest.approx(0.5 * _STAT["h_trace"] * 3e-4)

    # Both halves of the authoritative-source contract are present: the
    # explicit source defeats the bit-exact short-circuit, and
    # output_mse_measured=False stops the noise output_mse outranking it.
    assert entry["cost_source"] == "propagated_serving_unit_penalty"
    assert entry["output_mse_measured"] is False
    assert not cost_entry_uses_measured_output_mse(
        stats[_NAME], entry, "FP8_SOURCE")
    assert cost_entry_source(
        stats[_NAME], entry, "FP8_SOURCE") == "propagated_serving_unit_penalty"
    assert summary["total_scaled_member_penalty"] == pytest.approx(
        _PROPAGATED_KL)


def test_summary_penalty_equals_the_dloss_the_dp_actually_charges():
    """Accounting invariant: Σ(charged after) − Σ(charged before) ==
    total_scaled_member_penalty. This is what broke — the summary counted a
    penalty on the bit-exact row that the DP never saw."""
    stats = _stats()
    costs = {_NAME: {
        "FP8_SOURCE": _bit_exact_passthrough_entry(),
        "NVFP4": _lossy_actquant_entry(),
    }}
    before = {
        fmt: cost_entry_predicted_dloss(
            stats[_NAME], entry, format_name=fmt)
        for fmt, entry in costs[_NAME].items()
    }
    assert before["FP8_SOURCE"] == 0.0          # short-circuited
    assert before["NVFP4"] > 0.0                # measured output_mse

    adjusted, summary = apply_propagated_sensitivity_penalty(
        costs,
        stats=stats,
        report=_report("NVFP4"),
        scale=1.0,
        target_format="BF16",
        format_extrapolation="local_mse_ratio",
    )
    after = {
        fmt: cost_entry_predicted_dloss(
            stats[_NAME], entry, format_name=fmt)
        for fmt, entry in adjusted[_NAME].items()
    }

    charged_delta = sum(after.values()) - sum(before.values())
    assert charged_delta == pytest.approx(
        summary["total_scaled_member_penalty"]), (
        f"summary claims {summary['total_scaled_member_penalty']} of penalty "
        f"but the DP's cost only moved by {charged_delta}")
    # Every adjusted entry moved: none of the penalty evaporated.
    assert summary["adjusted_entries"] == 2
    for fmt in ("FP8_SOURCE", "NVFP4"):
        assert after[fmt] > before[fmt], fmt


def test_stamp_is_limited_to_bit_exact_rows():
    """A row the allocator already prices from predicted_dloss needs no stamp:
    only the bit-exact short-circuit discards the penalty."""
    stats = _stats()
    # Unmeasured output_mse => the allocator reads predicted_dloss already,
    # and weight_mse > 0 means the row was never bit-exact.
    entry = {"weight_mse": 1e-5, "output_mse": 0.0,
             "output_mse_measured": False, "predicted_dloss": 0.25}
    costs = {_NAME: {"NVFP4": dict(entry)}}

    adjusted, summary = apply_propagated_sensitivity_penalty(
        costs,
        stats=stats,
        report=_report("NVFP4"),
        scale=1.0,
        target_format="BF16",
        format_extrapolation="current_only",
    )
    out = adjusted[_NAME]["NVFP4"]
    assert "cost_source" not in out
    assert out["predicted_dloss"] == pytest.approx(0.25 + _PROPAGATED_KL)
    assert cost_entry_predicted_dloss(
        stats[_NAME], out, format_name="NVFP4") == pytest.approx(
            0.25 + _PROPAGATED_KL)
    assert summary["total_scaled_member_penalty"] == pytest.approx(
        _PROPAGATED_KL)


def test_producer_cost_source_provenance_is_preserved():
    """An entry that already declares its own authoritative source keeps that
    provenance (the ``[alloc] cost-source usage`` accounting must not be
    rewritten to the penalty label)."""
    stats = _stats()
    # aura-style row: explicit source, predicted_dloss authoritative, and a
    # weight_mse of 0.0 that must NOT be mistaken for a bit-exact measurement.
    costs = {_NAME: {"NVFP4": {
        "weight_mse": 0.0, "predicted_dloss": 0.5,
        "output_mse_measured": False, "cost_source": "aura",
    }}}

    adjusted, _summary = apply_propagated_sensitivity_penalty(
        costs,
        stats=stats,
        report=_report("NVFP4"),
        scale=1.0,
        target_format="BF16",
        format_extrapolation="current_only",
    )
    out = adjusted[_NAME]["NVFP4"]
    assert out["cost_source"] == "aura"
    assert cost_entry_predicted_dloss(
        stats[_NAME], out, format_name="NVFP4") == pytest.approx(
            0.5 + _PROPAGATED_KL)
