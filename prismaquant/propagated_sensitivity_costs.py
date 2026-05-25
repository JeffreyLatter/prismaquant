"""Propagated-sensitivity cost augmentation utilities."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    cost_entry_predicted_dloss,
    cost_entry_uses_measured_output_mse,
)
from prismaquant.mse_promotion import _bits_delta


def apply_propagated_sensitivity_penalty(
    costs: Mapping[str, object],
    *,
    stats: Mapping[str, Mapping],
    report: Mapping[str, object],
    scale: float,
    target_format: str | None = None,
    score_field: str = "propagated_kl",
    format_extrapolation: str = "local_mse_ratio",
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
    format.  Alternative candidate formats are scaled according to
    ``format_extrapolation``:

    - ``local_mse_ratio``: local output-MSE ratio to the current format.
    - ``current_only``: apply the measured penalty only to the current format.
    - ``bits_interp``: linearly scale by remaining added bits to target.
    """
    target_fmt = fr.canonical_format_name(target_format or report.get("target_format", "BF16"))
    extrapolation = str(format_extrapolation)
    if extrapolation not in {"local_mse_ratio", "current_only", "bits_interp"}:
        raise ValueError(
            "format_extrapolation must be one of: local_mse_ratio, "
            f"current_only, bits_interp; got {format_extrapolation!r}"
        )
    out = copy.deepcopy(dict(costs))
    rows = list(report.get("rows", ()))

    adjusted_entries = 0
    skipped = 0
    total_scaled_member_penalty = 0.0
    total_scaled_current_format_penalty = 0.0
    total_unscaled_propagated = 0.0
    max_current_format_penalty_abs_error = 0.0
    large_shift_count = 0
    large_shift_sample: list[dict[str, object]] = []
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

        row_current_format_penalty = 0.0
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
            if current_output_mse <= 0.0 and extrapolation == "local_mse_ratio":
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
                format_ratio = _format_extrapolation_ratio(
                    extrapolation,
                    stats_entry=stat,
                    candidate_fmt=fmt_c,
                    current_fmt=current_fmt,
                    target_fmt=target_fmt,
                    candidate_output_mse=output_mse,
                    current_output_mse=current_output_mse,
                )
                penalty = propagated * scale_f * member_share * format_ratio
                if penalty <= 0.0:
                    continue
                base_predicted = cost_entry_predicted_dloss(dict(stat), dict(entry))
                new_entry = copy.deepcopy(dict(entry))
                uses_output_mse = cost_entry_uses_measured_output_mse(
                    dict(stat),
                    dict(entry),
                )
                new_entry[f"{metadata_prefix}_uses_output_mse"] = bool(uses_output_mse)
                if uses_output_mse:
                    delta_output_mse = penalty / max(0.5 * h_trace, 1e-30)
                    new_entry[f"base_output_mse_before_{metadata_prefix}_penalty"] = output_mse
                    new_entry["output_mse"] = float(output_mse + delta_output_mse)
                    shifted_cost = new_entry["output_mse"]
                    base_cost = max(output_mse, 1e-30)
                else:
                    new_entry["predicted_dloss"] = float(base_predicted + penalty)
                    shifted_cost = new_entry["predicted_dloss"]
                    base_cost = max(base_predicted, 1e-30)
                new_entry[f"base_predicted_dloss_before_{metadata_prefix}_penalty"] = base_predicted
                new_entry[f"{metadata_prefix}_key"] = str(row.get("key", ""))
                new_entry[f"{metadata_prefix}_kl"] = propagated
                new_entry[f"{metadata_prefix}_member_share"] = float(member_share)
                new_entry[f"{metadata_prefix}_format_ratio"] = float(format_ratio)
                new_entry[f"{metadata_prefix}_format_extrapolation"] = extrapolation
                new_entry["propagated_kl_penalty_scale"] = scale_f
                new_entry["propagated_kl_penalty"] = float(penalty)
                per_name[fmt] = new_entry
                adjusted_entries += 1
                total_scaled_member_penalty += penalty
                shift_ratio = shifted_cost / base_cost
                if shift_ratio > 5.0:
                    large_shift_count += 1
                    if len(large_shift_sample) < 20:
                        large_shift_sample.append({
                            "name": member,
                            "format": fmt_c,
                            "shift_ratio": float(shift_ratio),
                            "base_cost": float(base_cost),
                            "shifted_cost": float(shifted_cost),
                        })
                if fmt_c == current_fmt:
                    row_current_format_penalty += penalty

        expected_current_penalty = propagated * scale_f
        if row_current_format_penalty > 0.0:
            total_scaled_current_format_penalty += row_current_format_penalty
            max_current_format_penalty_abs_error = max(
                max_current_format_penalty_abs_error,
                abs(row_current_format_penalty - expected_current_penalty),
            )

    summary = {
        "schema": "prismaquant.propagated_sensitivity_costs.summary.v1",
        "scale": scale_f,
        "target_format": target_fmt,
        "score_field": str(score_field),
        "format_extrapolation": extrapolation,
        "metadata_prefix": str(metadata_prefix),
        "report_base_assignment": report.get("base_assignment"),
        "report_base_bpp": report.get("base_bpp"),
        "measured_units": len(rows),
        "adjusted_entries": int(adjusted_entries),
        "skipped": int(skipped),
        "total_unscaled_propagated_kl": float(total_unscaled_propagated),
        "total_scaled_member_penalty": float(total_scaled_member_penalty),
        "total_scaled_current_format_penalty": float(
            total_scaled_current_format_penalty
        ),
        "max_current_format_penalty_abs_error": float(
            max_current_format_penalty_abs_error
        ),
        "large_shift_threshold": 5.0,
        "large_shift_count": int(large_shift_count),
        "large_shift_sample": large_shift_sample,
        "penalty_distribution": (
            "member added-bit share; selected format_extrapolation across "
            "candidate formats; fused unit sums once after allocator aggregation"
        ),
    }
    return out, summary


def _format_extrapolation_ratio(
    method: str,
    *,
    stats_entry: Mapping,
    candidate_fmt: str,
    current_fmt: str,
    target_fmt: str,
    candidate_output_mse: float,
    current_output_mse: float,
) -> float:
    if method == "local_mse_ratio":
        return candidate_output_mse / max(current_output_mse, 1e-30)
    if method == "current_only":
        return 1.0 if candidate_fmt == current_fmt else 0.0
    if method == "bits_interp":
        current_delta = max(_bits_delta(stats_entry, current_fmt, target_fmt), 0.0)
        if current_delta <= 0.0:
            return 0.0
        candidate_delta = max(_bits_delta(stats_entry, candidate_fmt, target_fmt), 0.0)
        return candidate_delta / current_delta
    raise AssertionError(f"unhandled format extrapolation method {method!r}")


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
