"""Empirical response analysis for measured KL sensitivity probes.

This module does not choose formats.  It turns the already-measured
``kl_sensitivity_probe`` candidate rows into quantitative sensitivity
coefficients:

    measured_gain = floor_kl - candidate_kl
    sensitivity   = measured_gain / induced_error

where ``induced_error`` can be the local render ``output_mse``, allocator
``predicted_dloss``, or added bits.  That separates "this layer had a large
quantization error" from "the model is unusually sensitive to that error".
"""
from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import cost_entry_predicted_dloss


_LAYER_RE = re.compile(r"(?:^|[.])layers[.](\d+)[.]")
_VISUAL_BLOCK_RE = re.compile(r"(?:^|[.])visual[.]blocks[.](\d+)[.]")


def semantic_category(name: str) -> str:
    """Return the coarse functional bucket for a Linear/decision unit."""
    value = str(name)
    if ".visual." in value or value.startswith("model.visual."):
        return "visual"
    if ".mlp.shared_expert." in value or value.endswith(".mlp.shared_expert_gate"):
        return "shared_expert"
    if ".self_attn." in value:
        return "self_attn"
    if ".linear_attn." in value:
        return "linear_attn"
    if ".mlp.experts." in value:
        return "routed_experts"
    if value.startswith("mtp."):
        return "mtp"
    return "other"


def layer_number(name: str) -> str:
    match = _LAYER_RE.search(str(name))
    if match:
        return match.group(1)
    match = _VISUAL_BLOCK_RE.search(str(name))
    if match:
        return match.group(1)
    return "unknown"


def _canonical(fmt: object) -> str:
    return fr.canonical_format_name(str(fmt).strip().upper())


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _ratio(num: float, den: float) -> float | None:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1e-30:
        return None
    return float(num / den)


def _cost_entry(costs: Mapping[str, object] | None, qname: str, fmt: str) -> Mapping | None:
    if not isinstance(costs, Mapping):
        return None
    per_name = costs.get(qname)
    if not isinstance(per_name, Mapping):
        return None
    seen: set[str] = set()
    for candidate in (fmt, _canonical(fmt), *fr.aliases_for(fmt)):
        key = str(candidate).strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        entry = per_name.get(key)
        if isinstance(entry, Mapping) and "error" not in entry:
            return entry
    return None


def _baseline_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    baseline: dict[str, Mapping[str, object]] = {}
    for row in rows:
        qname = str(row.get("qname", ""))
        if not qname:
            continue
        if abs(_finite_float(row.get("bits_delta"))) <= 1e-9:
            baseline.setdefault(qname, row)
    return baseline


def _member_cost_totals(
    members: Sequence[str],
    *,
    baseline_by_qname: Mapping[str, Mapping[str, object]],
    costs: Mapping[str, object] | None,
    stats: Mapping[str, object] | None,
) -> dict[str, object]:
    output_mse_sum = 0.0
    rel_output_mse_sum = 0.0
    fisher_output_mse_sum = 0.0
    weight_mse_param_sum = 0.0
    predicted_dloss_sum = 0.0
    params = 0
    missing_cost: list[str] = []
    baseline_formats: Counter[str] = Counter()

    for qname in members:
        baseline_row = baseline_by_qname.get(qname, {})
        baseline_fmt = _canonical(baseline_row.get("format", "BF16"))
        baseline_formats[baseline_fmt] += 1
        shape = baseline_row.get("shape")
        if isinstance(shape, Sequence) and len(shape) == 2:
            n_params = int(shape[0]) * int(shape[1])
        else:
            stat = stats.get(qname) if isinstance(stats, Mapping) else None
            n_params = int(stat.get("n_params", 0) or 0) if isinstance(stat, Mapping) else 0
        params += int(n_params)

        entry = _cost_entry(costs, qname, baseline_fmt)
        if entry is None:
            missing_cost.append(qname)
            continue
        output_mse_sum += _finite_float(entry.get("output_mse"))
        rel_output_mse_sum += _finite_float(entry.get("rel_output_mse"))
        fisher_output_mse_sum += _finite_float(entry.get("fisher_output_mse"))
        weight_mse_param_sum += _finite_float(entry.get("weight_mse")) * float(n_params)
        if entry.get("predicted_dloss") is not None:
            predicted_dloss_sum += _finite_float(entry.get("predicted_dloss"))
        else:
            stat = stats.get(qname) if isinstance(stats, Mapping) else None
            if isinstance(stat, Mapping):
                try:
                    predicted_dloss_sum += float(
                        cost_entry_predicted_dloss(
                            dict(stat), dict(entry),
                            format_name=baseline_fmt,
                        )
                    )
                except Exception:
                    pass

    return {
        "params": int(params),
        "baseline_formats": dict(sorted(baseline_formats.items())),
        "output_mse_sum": float(output_mse_sum),
        "rel_output_mse_sum": float(rel_output_mse_sum),
        "fisher_output_mse_sum": float(fisher_output_mse_sum),
        "weight_mse_param_sum": float(weight_mse_param_sum),
        "weight_mse_weighted_mean": _ratio(weight_mse_param_sum, float(params)),
        "predicted_dloss_sum": float(predicted_dloss_sum),
        "missing_cost_count": len(missing_cost),
        "missing_cost_sample": missing_cost[:8],
    }


def build_response_report(
    kl_probe_payload: Mapping[str, object],
    *,
    costs: Mapping[str, object] | None = None,
    stats: Mapping[str, object] | None = None,
    target_format: str = "BF16",
) -> dict[str, object]:
    """Build unit and category sensitivity coefficients from KL probe rows."""
    rows = kl_probe_payload.get("rows")
    if not isinstance(rows, Sequence):
        raise ValueError("KL probe payload does not contain a rows list")
    floor = kl_probe_payload.get("floor")
    if not isinstance(floor, Mapping):
        raise ValueError("KL probe payload does not contain a floor mapping")
    floor_kl = _finite_float(floor.get("kl"))
    target_fmt = _canonical(target_format)
    baseline_by_qname = _baseline_rows(rows)

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        fmt = _canonical(row.get("format", ""))
        if fmt != target_fmt:
            continue
        qname = str(row.get("qname", ""))
        if not qname:
            continue
        if abs(_finite_float(row.get("bits_delta"))) <= 1e-9:
            continue
        decision_unit = str(row.get("decision_unit") or qname)
        grouped[(decision_unit, fmt)].append(row)

    unit_rows: list[dict[str, object]] = []
    for (decision_unit, fmt), member_rows in sorted(grouped.items()):
        members = sorted(str(row["qname"]) for row in member_rows)
        candidate_values = [_finite_float(row.get("candidate_kl")) for row in member_rows]
        candidate_kl = sum(candidate_values) / max(len(candidate_values), 1)
        candidate_kl_min = min(candidate_values) if candidate_values else candidate_kl
        candidate_kl_max = max(candidate_values) if candidate_values else candidate_kl
        kl_gain = float(floor_kl - candidate_kl)
        bits_delta = float(sum(_finite_float(row.get("bits_delta")) for row in member_rows))
        row_sensitivity_sum = float(
            sum(_finite_float(row.get("sensitivity")) for row in member_rows)
        )
        cost_totals = _member_cost_totals(
            members,
            baseline_by_qname=baseline_by_qname,
            costs=costs,
            stats=stats,
        )
        category_counts = Counter(semantic_category(member) for member in members)
        category = category_counts.most_common(1)[0][0] if category_counts else "other"
        layer_counts = Counter(layer_number(member) for member in members)
        layer = layer_counts.most_common(1)[0][0] if layer_counts else "unknown"
        gbits_delta = bits_delta / 1e9
        output_mse_sum = float(cost_totals["output_mse_sum"])
        predicted_dloss_sum = float(cost_totals["predicted_dloss_sum"])
        unit_rows.append({
            "decision_unit": decision_unit,
            "category": category,
            "layer": layer,
            "target_format": fmt,
            "member_count": len(members),
            "members": members,
            "baseline_formats": cost_totals["baseline_formats"],
            "params": int(cost_totals["params"]),
            "bits_delta": bits_delta,
            "gbits_delta": gbits_delta,
            "floor_kl": float(floor_kl),
            "candidate_kl": float(candidate_kl),
            "candidate_kl_min": float(candidate_kl_min),
            "candidate_kl_max": float(candidate_kl_max),
            "kl_gain": kl_gain,
            "row_sensitivity_sum": row_sensitivity_sum,
            "output_mse_sum": output_mse_sum,
            "rel_output_mse_sum": float(cost_totals["rel_output_mse_sum"]),
            "fisher_output_mse_sum": float(cost_totals["fisher_output_mse_sum"]),
            "weight_mse_param_sum": float(cost_totals["weight_mse_param_sum"]),
            "weight_mse_weighted_mean": cost_totals["weight_mse_weighted_mean"],
            "predicted_dloss_sum": predicted_dloss_sum,
            "missing_cost_count": int(cost_totals["missing_cost_count"]),
            "missing_cost_sample": cost_totals["missing_cost_sample"],
            "kl_gain_per_gbit": _ratio(kl_gain, gbits_delta),
            "kl_gain_per_output_mse": _ratio(kl_gain, output_mse_sum),
            "kl_gain_per_predicted_dloss": _ratio(kl_gain, predicted_dloss_sum),
            "prediction_gain_ratio": _ratio(kl_gain, predicted_dloss_sum),
        })

    unit_rows.sort(
        key=lambda row: (
            -_finite_float(row.get("kl_gain")),
            -_finite_float(row.get("kl_gain_per_gbit")),
            str(row.get("decision_unit")),
        )
    )
    category_rows = _category_summary(unit_rows)
    return {
        "schema": "prismaquant.sensitivity_response.v1",
        "source_schema": kl_probe_payload.get("schema"),
        "source_git_commit": kl_probe_payload.get("git_commit"),
        "target_format": target_fmt,
        "floor_kl": float(floor_kl),
        "unit_count": len(unit_rows),
        "category_count": len(category_rows),
        "units": unit_rows,
        "categories": category_rows,
    }


def _category_summary(unit_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_category: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in unit_rows:
        by_category[str(row.get("category", "other"))].append(row)

    total_positive_gain = sum(max(_finite_float(row.get("kl_gain")), 0.0) for row in unit_rows)
    total_output_mse = sum(_finite_float(row.get("output_mse_sum")) for row in unit_rows)
    total_predicted = sum(_finite_float(row.get("predicted_dloss_sum")) for row in unit_rows)

    summaries: list[dict[str, object]] = []
    for category, rows in sorted(by_category.items()):
        gain_sum = sum(_finite_float(row.get("kl_gain")) for row in rows)
        positive_gain_sum = sum(max(_finite_float(row.get("kl_gain")), 0.0) for row in rows)
        negative_gain_sum = sum(min(_finite_float(row.get("kl_gain")), 0.0) for row in rows)
        bits_delta = sum(_finite_float(row.get("bits_delta")) for row in rows)
        gbits_delta = bits_delta / 1e9
        output_mse_sum = sum(_finite_float(row.get("output_mse_sum")) for row in rows)
        predicted_dloss_sum = sum(_finite_float(row.get("predicted_dloss_sum")) for row in rows)
        params = sum(int(row.get("params", 0) or 0) for row in rows)
        share_positive_gain = _ratio(positive_gain_sum, total_positive_gain)
        share_output_mse = _ratio(output_mse_sum, total_output_mse)
        share_predicted = _ratio(predicted_dloss_sum, total_predicted)
        summaries.append({
            "category": category,
            "unit_count": len(rows),
            "positive_unit_count": sum(1 for row in rows if _finite_float(row.get("kl_gain")) > 0.0),
            "member_count": sum(int(row.get("member_count", 0) or 0) for row in rows),
            "params": int(params),
            "bits_delta": float(bits_delta),
            "gbits_delta": float(gbits_delta),
            "kl_gain_sum": float(gain_sum),
            "positive_kl_gain_sum": float(positive_gain_sum),
            "negative_kl_gain_sum": float(negative_gain_sum),
            "output_mse_sum": float(output_mse_sum),
            "predicted_dloss_sum": float(predicted_dloss_sum),
            "kl_gain_per_gbit": _ratio(gain_sum, gbits_delta),
            "positive_kl_gain_per_gbit": _ratio(positive_gain_sum, gbits_delta),
            "kl_gain_per_output_mse": _ratio(gain_sum, output_mse_sum),
            "positive_kl_gain_per_output_mse": _ratio(positive_gain_sum, output_mse_sum),
            "kl_gain_per_predicted_dloss": _ratio(gain_sum, predicted_dloss_sum),
            "positive_kl_gain_per_predicted_dloss": _ratio(positive_gain_sum, predicted_dloss_sum),
            "share_positive_gain": share_positive_gain,
            "share_output_mse": share_output_mse,
            "share_predicted_dloss": share_predicted,
            "positive_gain_enrichment_vs_output_mse": (
                _ratio(share_positive_gain, share_output_mse)
                if share_positive_gain is not None and share_output_mse is not None
                else None
            ),
            "positive_gain_enrichment_vs_predicted_dloss": (
                _ratio(share_positive_gain, share_predicted)
                if share_positive_gain is not None and share_predicted is not None
                else None
            ),
            "top_units": [
                str(row.get("decision_unit"))
                for row in sorted(rows, key=lambda item: -_finite_float(item.get("kl_gain")))[:8]
            ],
        })
    summaries.sort(
        key=lambda row: (
            -_finite_float(row.get("positive_kl_gain_sum")),
            -_finite_float(row.get("positive_gain_enrichment_vs_output_mse")),
            str(row.get("category")),
        )
    )
    return summaries


_UNIT_CSV_FIELDS = (
    "decision_unit",
    "category",
    "layer",
    "target_format",
    "member_count",
    "params",
    "bits_delta",
    "gbits_delta",
    "floor_kl",
    "candidate_kl",
    "kl_gain",
    "row_sensitivity_sum",
    "output_mse_sum",
    "predicted_dloss_sum",
    "kl_gain_per_gbit",
    "kl_gain_per_output_mse",
    "kl_gain_per_predicted_dloss",
    "prediction_gain_ratio",
    "missing_cost_count",
)

_CATEGORY_CSV_FIELDS = (
    "category",
    "unit_count",
    "positive_unit_count",
    "member_count",
    "params",
    "bits_delta",
    "gbits_delta",
    "kl_gain_sum",
    "positive_kl_gain_sum",
    "negative_kl_gain_sum",
    "output_mse_sum",
    "predicted_dloss_sum",
    "kl_gain_per_gbit",
    "positive_kl_gain_per_gbit",
    "kl_gain_per_output_mse",
    "positive_kl_gain_per_output_mse",
    "kl_gain_per_predicted_dloss",
    "positive_kl_gain_per_predicted_dloss",
    "share_positive_gain",
    "share_output_mse",
    "share_predicted_dloss",
    "positive_gain_enrichment_vs_output_mse",
    "positive_gain_enrichment_vs_predicted_dloss",
)


def write_unit_csv(path: str | Path, units: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, units, _UNIT_CSV_FIELDS)


def write_category_csv(path: str | Path, categories: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, categories, _CATEGORY_CSV_FIELDS)


def _write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
