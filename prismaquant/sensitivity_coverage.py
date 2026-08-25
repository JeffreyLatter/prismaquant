"""Coverage summaries for propagated-sensitivity reports."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence


def summarize_sensitivity_coverage(
    report: Mapping[str, object],
    assignment: Mapping[str, str],
    *,
    top_ns: Sequence[int] = (10, 20, 40),
    order_by: str = "propagated_kl_per_added_bit",
) -> dict[str, object]:
    """Summarize how an assignment covers top propagated-sensitive units."""
    rows = _ranked_rows(report, order_by)
    unit_rows = [_unit_row(row, assignment, idx) for idx, row in enumerate(rows, start=1)]
    top_summary = {
        str(int(n)): _top_summary(unit_rows[: max(int(n), 0)])
        for n in top_ns
    }
    return {
        "schema": "prismaquant.sensitivity_coverage.v1",
        "source_schema": report.get("schema"),
        "order_by": str(order_by),
        "row_count": len(unit_rows),
        "top": top_summary,
        "units": unit_rows,
    }


def _ranked_rows(
    report: Mapping[str, object],
    order_by: str,
) -> list[Mapping[str, object]]:
    rows = [
        row for row in report.get("rows", ())
        if isinstance(row, Mapping)
    ]
    return sorted(
        rows,
        key=lambda row: (
            -_finite_float(row.get(order_by)),
            -_finite_float(row.get("propagated_kl")),
            str(row.get("key", "")),
        ),
    )


def _unit_row(
    row: Mapping[str, object],
    assignment: Mapping[str, str],
    rank: int,
) -> dict[str, object]:
    members = [str(member) for member in row.get("members", ())]
    formats = Counter(str(assignment.get(member, "MISSING")) for member in members)
    format_key = _format_key(formats)
    return {
        "rank": int(rank),
        "key": str(row.get("key", "")),
        "category": str(row.get("category", "")),
        "layer": str(row.get("layer", "")),
        "members": members,
        "member_count": len(members),
        "formats": dict(sorted(formats.items())),
        "format_key": format_key,
        "all_bf16": format_key == "BF16",
        "has_nvfp4": formats.get("NVFP4", 0) > 0,
        "no_nvfp4": formats.get("NVFP4", 0) == 0 and formats.get("MISSING", 0) == 0,
        "has_missing": formats.get("MISSING", 0) > 0,
        "propagated_kl": _finite_float(row.get("propagated_kl")),
        "propagated_kl_per_added_bit": _finite_float(
            row.get("propagated_kl_per_added_bit")
        ),
    }


def _top_summary(unit_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    format_units = Counter(str(row.get("format_key", "")) for row in unit_rows)
    categories = Counter(str(row.get("category", "")) for row in unit_rows)
    return {
        "unit_count": len(unit_rows),
        "all_bf16_units": sum(1 for row in unit_rows if row.get("all_bf16")),
        "no_nvfp4_units": sum(1 for row in unit_rows if row.get("no_nvfp4")),
        "nvfp4_units": sum(1 for row in unit_rows if row.get("has_nvfp4")),
        "missing_units": sum(1 for row in unit_rows if row.get("has_missing")),
        "format_units": dict(sorted(format_units.items())),
        "categories": dict(sorted(categories.items())),
    }


def _format_key(formats: Counter[str]) -> str:
    if not formats:
        return "MISSING"
    if len(formats) == 1:
        return next(iter(formats))
    return "mixed:" + ",".join(f"{fmt}={count}" for fmt, count in sorted(formats.items()))


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out
