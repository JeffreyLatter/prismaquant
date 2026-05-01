"""L3 propagated-cost utilities.

This module owns the final-pass "L3 polish" path: select a small allocator
neighborhood around the converged L2 assignment, measure propagated costs for
that neighborhood, and re-solve only those measured choices while freezing the
rest of the L2 assignment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import cost_entry_predicted_dloss
from prismaquant.allocator_solver import Candidate, _shape_from_stats, solve_allocation


@dataclass(frozen=True)
class L3NeighborhoodEntry:
    name: str
    current_format: str
    formats: tuple[str, ...]
    margin: float
    l2_current_cost: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


class FrozenBudgetError(RuntimeError):
    """Raised when frozen L2 choices make the L3 neighborhood infeasible."""


def _cost_entry(costs: Mapping, name: str, fmt: str) -> dict | None:
    per_name = costs.get(name, {})
    if not isinstance(per_name, Mapping):
        return None
    for alias in fr.aliases_for(fmt):
        entry = per_name.get(alias)
        if isinstance(entry, dict) and "error" not in entry:
            return entry
    return None


def l2_cost_value(stats: Mapping, costs: Mapping, name: str, fmt: str) -> float | None:
    """Return the allocator's L2 scalar cost for one existing cost entry."""
    entry = _cost_entry(costs, name, fmt)
    if entry is None or name not in stats:
        return None
    return float(cost_entry_predicted_dloss(stats[name], entry))


def _memory_bytes_for_format(
    stats_entry: Mapping,
    spec: fr.FormatSpec,
) -> int:
    memory_map = stats_entry.get("_memory_bytes_by_format")
    if isinstance(memory_map, Mapping) and spec.name in memory_map:
        return int(memory_map[spec.name])
    return int(spec.memory_bytes_for_shape(_shape_from_stats(dict(stats_entry))))


def assignment_bit_total(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    specs_by_name: Mapping[str, fr.FormatSpec],
) -> float:
    """Return total assigned bits, not average bits."""
    total = 0.0
    for name, fmt in assignment.items():
        if name not in stats:
            continue
        spec = specs_by_name[fr.canonical_format_name(fmt)]
        total += 8.0 * _memory_bytes_for_format(stats[name], spec)
    return total


def _available_formats_for_name(
    stats: Mapping,
    costs: Mapping,
    name: str,
    specs: list[fr.FormatSpec],
) -> list[str]:
    available: list[str] = []
    for spec in specs:
        if l2_cost_value(stats, costs, name, spec.name) is not None:
            available.append(spec.name)
    return available


def _bits_for_name(stats: Mapping, name: str, spec: fr.FormatSpec) -> float:
    shape = _shape_from_stats(dict(stats[name]))
    return float(spec.effective_bits_for_shape(shape))


def select_formats_for_l3(
    stats: Mapping,
    costs: Mapping,
    assignment: Mapping[str, str],
    name: str,
    specs: list[fr.FormatSpec],
) -> tuple[str, ...]:
    """Choose current + one cheaper + one more accurate + BF16 when present."""
    if name not in stats or name not in assignment:
        return ()
    specs_by_name = {s.name: s for s in specs}
    current = fr.canonical_format_name(assignment[name])
    available = _available_formats_for_name(stats, costs, name, specs)
    if current not in available:
        return ()

    ordered = sorted(
        available,
        key=lambda fmt: (
            _bits_for_name(stats, name, specs_by_name[fmt]),
            fmt,
        ),
    )
    idx = ordered.index(current)
    chosen = {current}
    if idx > 0:
        chosen.add(ordered[idx - 1])
    if idx + 1 < len(ordered):
        chosen.add(ordered[idx + 1])
    if "BF16" in available:
        chosen.add("BF16")
    return tuple(
        sorted(
            chosen,
            key=lambda fmt: (
                _bits_for_name(stats, name, specs_by_name[fmt]),
                fmt,
            ),
        )
    )


def _relative_margin(values: list[float], current_cost: float) -> float:
    margins = []
    for value in values:
        denom = max(abs(current_cost), abs(value), 1e-12)
        margins.append(abs(value - current_cost) / denom)
    if not margins:
        return float("inf")
    return float(min(margins))


def select_l3_neighborhood(
    stats: Mapping,
    costs: Mapping,
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
    *,
    uncertainty_rel_tol: float = 0.10,
    min_fraction: float = 0.05,
    max_fraction: float = 0.10,
    safety_fraction: float = 0.02,
) -> list[L3NeighborhoodEntry]:
    """Select the small L2 neighborhood that L3 is allowed to re-optimize."""
    eligible: list[L3NeighborhoodEntry] = []
    for name in sorted(set(stats) & set(assignment)):
        current = fr.canonical_format_name(assignment[name])
        current_cost = l2_cost_value(stats, costs, name, current)
        if current_cost is None:
            continue
        fmts = select_formats_for_l3(stats, costs, assignment, name, specs)
        if not fmts:
            continue
        alt_costs = [
            value
            for fmt in fmts
            if fmt != current
            for value in [l2_cost_value(stats, costs, name, fmt)]
            if value is not None
        ]
        margin = _relative_margin(alt_costs, current_cost)
        eligible.append(
            L3NeighborhoodEntry(
                name=name,
                current_format=current,
                formats=fmts,
                margin=margin,
                l2_current_cost=current_cost,
            )
        )

    if not eligible:
        return []

    total = len([name for name in assignment if name in stats])
    min_count = max(1, int(math.ceil(total * min_fraction)))
    max_count = max(min_count, int(math.ceil(total * max_fraction)))
    safety_count = int(math.ceil(total * safety_fraction))

    by_name: dict[str, L3NeighborhoodEntry] = {}

    def _add(entry: L3NeighborhoodEntry, reason: str) -> None:
        existing = by_name.get(entry.name)
        reasons = set(existing.reasons if existing is not None else entry.reasons)
        reasons.add(reason)
        by_name[entry.name] = L3NeighborhoodEntry(
            name=entry.name,
            current_format=entry.current_format,
            formats=entry.formats,
            margin=entry.margin,
            l2_current_cost=entry.l2_current_cost,
            reasons=tuple(sorted(reasons)),
        )

    for entry in eligible:
        if entry.margin <= uncertainty_rel_tol:
            _add(entry, "uncertain")

    for entry in sorted(eligible, key=lambda e: (-e.l2_current_cost, e.name))[:safety_count]:
        _add(entry, "high_l2_cost")

    if len(by_name) < min_count:
        for entry in sorted(eligible, key=lambda e: (e.margin, -e.l2_current_cost, e.name)):
            _add(entry, "fill_min_fraction")
            if len(by_name) >= min_count:
                break

    selected = list(by_name.values())
    if len(selected) > max_count:
        safety = [e for e in selected if "high_l2_cost" in e.reasons]
        rest = [e for e in selected if "high_l2_cost" not in e.reasons]
        rest.sort(key=lambda e: (e.margin, -e.l2_current_cost, e.name))
        selected = (safety + rest)[:max_count]

    return sorted(selected, key=lambda e: e.name)


def build_l3_candidates(
    stats: Mapping,
    propagated_costs: Mapping[str, Mapping[str, Mapping]],
    specs: list[fr.FormatSpec],
) -> dict[str, list[Candidate]]:
    """Build DP candidates from propagated end-KL costs only."""
    specs_by_name = {s.name: s for s in specs}
    out: dict[str, list[Candidate]] = {}
    for name, per_name in propagated_costs.items():
        if name not in stats or not isinstance(per_name, Mapping):
            continue
        shape = _shape_from_stats(dict(stats[name]))
        cands: list[Candidate] = []
        for fmt, entry in per_name.items():
            canonical = fr.canonical_format_name(fmt)
            if canonical not in specs_by_name or not isinstance(entry, Mapping):
                continue
            if "error" in entry or "propagated_end_kl" not in entry:
                continue
            spec = specs_by_name[canonical]
            cands.append(
                Candidate(
                    fmt=canonical,
                    bits_per_param=spec.effective_bits_for_shape(shape),
                    memory_bytes=_memory_bytes_for_format(stats[name], spec),
                    predicted_dloss=max(float(entry["propagated_end_kl"]), 0.0),
                )
            )
        if cands:
            out[name] = cands
    return out


def solve_frozen_l3_neighborhood(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    l3_candidates: Mapping[str, list[Candidate]],
    specs: list[fr.FormatSpec],
    *,
    target_bits: float,
    bit_precision: float,
) -> tuple[dict[str, str], dict[str, Candidate]]:
    """Solve L3 candidates while freezing all non-neighborhood L2 choices."""
    specs_by_name = {s.name: s for s in specs}
    all_names = set(stats) & set(assignment)
    open_names = set(l3_candidates)
    frozen_assignment = {
        name: assignment[name]
        for name in sorted(all_names - open_names)
    }
    total_params = sum(int(stats[n].get("n_params", 0) or 0) for n in all_names)
    open_params = sum(int(stats[n].get("n_params", 0) or 0) for n in open_names)
    if total_params <= 0:
        return dict(assignment), {}

    target_total_bits = float(target_bits) * float(total_params)
    frozen_bits = assignment_bit_total(stats, frozen_assignment, specs_by_name)
    remaining_bits = target_total_bits - frozen_bits
    if remaining_bits < -1e-6:
        raise FrozenBudgetError(
            "L3 polish infeasible: frozen L2 choices already exceed target "
            f"budget ({frozen_bits / total_params:.6f} bpp frozen vs "
            f"{target_bits:.6f} bpp target)."
        )
    if open_params <= 0:
        return dict(assignment), {}

    open_target_bits = remaining_bits / float(open_params)
    open_stats = {name: dict(stats[name]) for name in sorted(open_names)}
    open_cands = {name: list(l3_candidates[name]) for name in sorted(open_names)}
    result = solve_allocation(open_stats, open_cands, open_target_bits, bit_precision)
    if result is None:
        raise FrozenBudgetError(
            "L3 polish infeasible: measured neighborhood cannot satisfy "
            f"remaining budget {open_target_bits:.6f} bpp after frozen choices."
        )
    open_assignment, chosen = result
    merged = dict(assignment)
    merged.update(open_assignment)
    return merged, chosen
