"""Propagated-sensitivity cost augmentation utilities."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import cost_entry_predicted_dloss
from prismaquant.mse_promotion import _bits_delta


def apply_propagated_sensitivity_penalty(
    costs: Mapping[str, object],
    *,
    stats: Mapping[str, Mapping],
    report: Mapping[str, object],
    scale: float,
    target_format: str | None = None,
    score_field: str = "propagated_kl",
    metadata_prefix: str = "propagated_serving_unit",
) -> tuple[dict[str, object], dict[str, object]]:
    """Return a copy of ``costs`` with propagated KL folded into output MSE.

    ``sensitivity_propagated_group_report.py`` measures a serving unit at its
    current assignment versus ``target_format``.  This function injects that
    end-to-end penalty into every non-target candidate for the same members so
    the normal allocator can spend bits on units that have high propagated
    sensitivity, not just high local output MSE.

    The unit penalty is counted once.  For fused units, it is distributed over
    members by each member's added-bit share from current format to target
    format.  Alternative candidate formats are scaled by their local output-MSE
    ratio relative to the measured current format.
    """
    target_fmt = fr.canonical_format_name(target_format or report.get("target_format", "BF16"))
    out = copy.deepcopy(dict(costs))
    rows = list(report.get("rows", ()))

    adjusted_entries = 0
    skipped = 0
    total_scaled_member_penalty = 0.0
    total_unscaled_propagated = 0.0
    scale_f = float(scale)

    for row in rows:
        propagated = _finite_float(row.get(score_field))
        if propagated <= 0.0:
            continue
        members = [str(member) for member in row.get("members", ()) if str(member) in stats]
        overrides = row.get("candidate_lane_override", {})
        if not isinstance(overrides, Mapping) or not members:
            skipped += 1
            continue
        total_unscaled_propagated += propagated

        member_bit_deltas: dict[str, float] = {}
        for member in members:
            current_fmt = _canonical(overrides.get(member))
            if not current_fmt:
                continue
            try:
                member_bit_deltas[member] = max(
                    _bits_delta(stats[member], current_fmt, target_fmt),
                    0.0,
                )
            except Exception:
                member_bit_deltas[member] = 0.0
        total_bits = sum(member_bit_deltas.values())
        if total_bits <= 0.0:
            total_bits = _finite_float(row.get("bits_delta"))
        if total_bits <= 0.0:
            skipped += 1
            continue

        for member in members:
            per_name = out.get(member)
            stat = stats.get(member)
            current_fmt = _canonical(overrides.get(member))
            if not isinstance(per_name, Mapping) or not isinstance(stat, Mapping) or not current_fmt:
                skipped += 1
                continue
            current_entry = per_name.get(current_fmt)
            if not isinstance(current_entry, Mapping):
                skipped += 1
                continue
            current_output_mse = _finite_float(current_entry.get("output_mse"))
            if current_output_mse <= 0.0:
                skipped += 1
                continue
            h_trace = _finite_float(stat.get("h_trace"))
            if h_trace <= 0.0:
                skipped += 1
                continue
            member_share = member_bit_deltas.get(member, 0.0) / total_bits
            if member_share <= 0.0:
                member_share = 1.0 / max(float(len(members)), 1.0)

            for fmt, entry in list(per_name.items()):
                fmt_c = _canonical(fmt)
                if fmt_c == target_fmt or not isinstance(entry, Mapping) or "error" in entry:
                    continue
                output_mse = _finite_float(entry.get("output_mse"))
                if output_mse < 0.0:
                    output_mse = 0.0
                format_ratio = output_mse / max(current_output_mse, 1e-30)
                penalty = propagated * scale_f * member_share * format_ratio
                if penalty <= 0.0:
                    continue
                base_predicted = cost_entry_predicted_dloss(dict(stat), dict(entry))
                delta_output_mse = penalty / max(0.5 * h_trace, 1e-30)
                new_entry = copy.deepcopy(dict(entry))
                new_entry[f"base_output_mse_before_{metadata_prefix}_penalty"] = output_mse
                new_entry[f"base_predicted_dloss_before_{metadata_prefix}_penalty"] = base_predicted
                new_entry[f"{metadata_prefix}_key"] = str(row.get("key", ""))
                new_entry[f"{metadata_prefix}_kl"] = propagated
                new_entry[f"{metadata_prefix}_member_share"] = float(member_share)
                new_entry[f"{metadata_prefix}_format_ratio"] = float(format_ratio)
                new_entry["propagated_kl_penalty_scale"] = scale_f
                new_entry["propagated_kl_penalty"] = float(penalty)
                new_entry["output_mse"] = float(output_mse + delta_output_mse)
                new_entry["output_mse_measured"] = True
                new_entry["predicted_dloss"] = float(base_predicted + penalty)
                per_name[fmt] = new_entry
                adjusted_entries += 1
                total_scaled_member_penalty += penalty

    summary = {
        "schema": "prismaquant.propagated_sensitivity_costs.summary.v1",
        "scale": scale_f,
        "target_format": target_fmt,
        "score_field": str(score_field),
        "metadata_prefix": str(metadata_prefix),
        "measured_units": len(rows),
        "adjusted_entries": int(adjusted_entries),
        "skipped": int(skipped),
        "total_unscaled_propagated_kl": float(total_unscaled_propagated),
        "total_scaled_member_penalty": float(total_scaled_member_penalty),
        "penalty_distribution": (
            "member added-bit share; format local-output-mse ratio to current "
            "format; fused unit sums once after allocator aggregation"
        ),
    }
    return out, summary


def _canonical(value: object) -> str:
    if value is None:
        return ""
    try:
        return fr.canonical_format_name(str(value))
    except Exception:
        return str(value)


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out
