"""L3 propagated-cost utilities.

This module owns the final-pass "L3 polish" path: select a small allocator
neighborhood around the converged L2 assignment, measure propagated costs for
that neighborhood, and re-solve only those measured choices while freezing the
rest of the L2 assignment.
"""
from __future__ import annotations

import inspect
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    _stats_indicates_packed_expert,
    cost_entry_predicted_dloss,
)
from prismaquant.allocator_solver import Candidate, _shape_from_stats, solve_allocation
from prismaquant.build_rtn_cache import kl_divergence
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    build_quantizable_map,
    calibration_data_hash,
    iter_calibration_forwards,
)
from prismaquant.layer_state_cache import LayerHiddenStateCache


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


class L3UnsupportedTargetError(RuntimeError):
    """Raised when L3 selection reaches targets the hook path cannot measure."""


@dataclass(frozen=True)
class _LaneSpec:
    name: str
    fmt: str
    baseline_index: int | None
    is_baseline: bool


@dataclass
class QuantWeightCache:
    cache: dict[tuple[str, str], torch.Tensor]

    def get(self, module_name: str, fmt: str) -> torch.Tensor | None:
        seen: set[str] = set()
        for candidate in (fmt, fr.canonical_format_name(fmt), *fr.aliases_for(fmt)):
            if candidate in seen:
                continue
            seen.add(candidate)
            cached = self.cache.get((module_name, candidate))
            if cached is not None:
                return cached
        return None


def _env_flag_enabled(name: str, *, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return max(parsed, 0)


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
    seen: set[str] = set()
    for spec in specs:
        canonical = fr.canonical_format_name(spec.name)
        if (
            canonical not in seen
            and l2_cost_value(stats, costs, name, canonical) is not None
        ):
            available.append(canonical)
            seen.add(canonical)
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
    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
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


def _is_l3_unsupported_target(stats_entry: Mapping) -> bool:
    """Return True for probe entries whose live module is not L3-hookable."""
    return _stats_indicates_packed_expert(dict(stats_entry))


def _current_has_cheaper_available_format(
    stats: Mapping,
    costs: Mapping,
    assignment: Mapping[str, str],
    name: str,
    specs: list[fr.FormatSpec],
) -> bool:
    current = fr.canonical_format_name(assignment[name])
    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
    if current not in specs_by_name:
        return False
    current_bits = _bits_for_name(stats, name, specs_by_name[current])
    for fmt in _available_formats_for_name(stats, costs, name, specs):
        if fmt == current or fmt not in specs_by_name:
            continue
        if _bits_for_name(stats, name, specs_by_name[fmt]) < current_bits - 1e-12:
            return True
    return False


def select_l3_neighborhood(
    stats: Mapping,
    costs: Mapping,
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
    *,
    uncertainty_rel_tol: float = 0.10,
    min_fraction: float = 0.05,
    max_fraction: float = 0.30,
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
    max_count = max(1, int(math.ceil(total * max_fraction)))
    min_count = min(max_count, max(1, int(math.ceil(total * min_fraction))))
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

    def _add_until_full(entries: list[L3NeighborhoodEntry], reason: str) -> None:
        for entry in entries:
            if entry.name not in by_name and len(by_name) >= max_count:
                continue
            _add(entry, reason)

    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}

    def _expected_flip_benefit(entry: L3NeighborhoodEntry) -> float:
        current = entry.current_format
        if current not in specs_by_name:
            return float("-inf")
        current_bits = _bits_for_name(stats, entry.name, specs_by_name[current])
        best = float("-inf")
        for fmt in entry.formats:
            if fmt == current or fmt not in specs_by_name:
                continue
            if _bits_for_name(stats, entry.name, specs_by_name[fmt]) >= current_bits:
                continue
            alt_cost = l2_cost_value(stats, costs, entry.name, fmt)
            if alt_cost is not None:
                best = max(best, entry.l2_current_cost - alt_cost)
        return best

    uncertain = [
        entry
        for entry in eligible
        if entry.margin <= uncertainty_rel_tol
    ]
    uncertain.sort(key=lambda e: (e.margin, -e.l2_current_cost, e.name))

    confident_non_cheapest = [
        entry
        for entry in eligible
        if _current_has_cheaper_available_format(
            stats, costs, assignment, entry.name, specs
        )
    ]
    benefit_by_name = {
        entry.name: _expected_flip_benefit(entry)
        for entry in confident_non_cheapest
    }
    confident_non_cheapest.sort(
        key=lambda e: (
            -benefit_by_name[e.name],
            -e.l2_current_cost,
            e.margin,
            e.name,
        )
    )

    safety = sorted(eligible, key=lambda e: (-e.l2_current_cost, e.name))[:safety_count]

    unsupported = sorted(
        {
            entry.name
            for entry in (*confident_non_cheapest, *uncertain, *safety)
            if _is_l3_unsupported_target(stats[entry.name])
        }
    )
    if unsupported:
        raise L3UnsupportedTargetError(
            "L3 polish does not yet support packed expert tensors. "
            f"Unsupported targets: {unsupported}. "
            "Re-run without --l3-polish for L2-only allocation, or wait "
            "for packed-expert L3 support."
        )

    _add_until_full(confident_non_cheapest, "confident_non_cheapest")
    _add_until_full(uncertain, "uncertain")
    for entry in safety:
        if entry.name not in by_name and len(by_name) >= max_count:
            continue
        _add(entry, "high_l2_cost")

    if len(by_name) < min_count:
        fill = sorted(eligible, key=lambda e: (e.margin, -e.l2_current_cost, e.name))
        for entry in fill:
            if entry.name not in by_name and len(by_name) >= max_count:
                break
            _add(entry, "fill_min_fraction")
            if len(by_name) >= min_count:
                break

    unsupported = sorted(
        name
        for name in by_name
        if _is_l3_unsupported_target(stats[name])
    )
    if unsupported:
        raise L3UnsupportedTargetError(
            "L3 polish does not yet support packed expert tensors. "
            f"Unsupported targets: {unsupported}. "
            "Re-run without --l3-polish for L2-only allocation, or wait "
            "for packed-expert L3 support."
        )

    return sorted(by_name.values(), key=lambda e: e.name)


def build_global_l3_neighborhood(
    stats: Mapping,
    costs: Mapping,
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
) -> list[L3NeighborhoodEntry]:
    """Build an L3 measurement neighborhood covering every eligible Linear."""
    selected: list[L3NeighborhoodEntry] = []
    unsupported: list[str] = []
    for name in sorted(set(stats) & set(assignment)):
        current = fr.canonical_format_name(assignment[name])
        current_cost = l2_cost_value(stats, costs, name, current)
        if current_cost is None:
            continue
        fmts = select_formats_for_l3(stats, costs, assignment, name, specs)
        if not fmts:
            continue
        if _is_l3_unsupported_target(stats[name]):
            unsupported.append(name)
            continue
        alt_costs = [
            value
            for fmt in fmts
            if fmt != current
            for value in [l2_cost_value(stats, costs, name, fmt)]
            if value is not None
        ]
        selected.append(
            L3NeighborhoodEntry(
                name=name,
                current_format=current,
                formats=fmts,
                margin=_relative_margin(alt_costs, current_cost),
                l2_current_cost=current_cost,
                reasons=("global",),
            )
        )
    if unsupported:
        raise L3UnsupportedTargetError(
            "L3 polish does not yet support packed expert tensors. "
            f"Unsupported targets: {sorted(unsupported)}. "
            "Re-run without --l3-polish for L2-only allocation, or wait "
            "for packed-expert L3 support."
        )
    return selected


def build_l3_candidates(
    stats: Mapping,
    propagated_costs: Mapping[str, Mapping[str, Mapping]],
    specs: list[fr.FormatSpec],
) -> dict[str, list[Candidate]]:
    """Build DP candidates from propagated end-KL costs only."""
    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
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


def _candidate_total_bits(candidate: Candidate) -> float:
    return 8.0 * float(candidate.memory_bytes)


def _greedy_l3_under_budget(
    open_cands: Mapping[str, list[Candidate]],
    current_assignment: Mapping[str, str],
    remaining_bits: float,
    budget_ceiling_bits: float | None = None,
) -> tuple[dict[str, str], dict[str, Candidate], dict]:
    names = sorted(open_cands)
    chosen: dict[str, Candidate] = {}
    for name in names:
        by_fmt = {c.fmt: c for c in open_cands[name]}
        current_fmt = fr.canonical_format_name(current_assignment.get(name, "BF16"))
        chosen[name] = by_fmt.get(current_fmt) or min(
            open_cands[name],
            key=lambda c: (c.predicted_dloss, _candidate_total_bits(c), c.fmt),
        )

    used_bits = sum(_candidate_total_bits(c) for c in chosen.values())
    budget_ceiling_bits = (
        float(remaining_bits)
        if budget_ceiling_bits is None
        else float(budget_ceiling_bits)
    )
    eps = 1e-12
    attempts = []
    for name in names:
        current = chosen[name]
        for cand in open_cands[name]:
            if cand.fmt == current.fmt:
                continue
            improvement = current.predicted_dloss - cand.predicted_dloss
            bit_delta = _candidate_total_bits(cand) - _candidate_total_bits(current)
            if bit_delta < -eps and improvement >= -eps:
                priority = 0
            elif bit_delta < -eps:
                priority = 1
            elif improvement > eps:
                priority = 2
            else:
                priority = 3
            attempts.append((priority, improvement, name, cand))
    attempts.sort(
        key=lambda item: (
            item[0],
            -item[1],
            _candidate_total_bits(item[3]),
            item[2],
            item[3].fmt,
        )
    )

    stats = {
        "attempts": 0,
        "accepted": 0,
        "rejected_not_better": 0,
        "rejected_budget": 0,
        "accepted_budget_reducing_nonworse": 0,
        "accepted_budget_reducing_worse": 0,
        "accepted_cost_improving": 0,
        "start_bits": used_bits,
        "end_bits": None,
        "remaining_bits": float(remaining_bits),
        "budget_ceiling_bits": float(budget_ceiling_bits),
    }
    swapped_names: set[str] = set()
    for _priority, improvement, name, cand in attempts:
        if name in swapped_names:
            continue
        stats["attempts"] += 1
        current = chosen[name]
        current_bits = _candidate_total_bits(current)
        cand_bits = _candidate_total_bits(cand)
        next_bits = used_bits - current_bits + cand_bits
        if next_bits > budget_ceiling_bits + 1e-6:
            stats["rejected_budget"] += 1
            continue
        overshoot_before = max(float(used_bits) - float(remaining_bits), 0.0)
        overshoot_after = max(float(next_bits) - float(remaining_bits), 0.0)
        reduces_overshoot = overshoot_after < overshoot_before - 1e-6
        cost_worsens = improvement < -eps
        cost_improves = improvement > eps
        if reduces_overshoot:
            if cost_worsens:
                stats["accepted_budget_reducing_worse"] += 1
            else:
                stats["accepted_budget_reducing_nonworse"] += 1
        elif cost_improves:
            stats["accepted_cost_improving"] += 1
        else:
            stats["rejected_not_better"] += 1
            continue
        chosen[name] = cand
        used_bits = next_bits
        swapped_names.add(name)
        stats["accepted"] += 1

    stats["end_bits"] = used_bits
    assignment = {name: chosen[name].fmt for name in names}
    return assignment, chosen, stats


def solve_frozen_l3_neighborhood(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    l3_candidates: Mapping[str, list[Candidate]],
    specs: list[fr.FormatSpec],
    *,
    target_bits: float,
    bit_precision: float,
    budget_tolerance: float = 0.0,
    return_metadata: bool = False,
) -> tuple[dict[str, str], dict[str, Candidate]]:
    """Solve L3 candidates while freezing all non-neighborhood L2 choices."""
    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
    all_names = set(stats) & set(assignment)
    open_names = set(l3_candidates)
    frozen_assignment = {
        name: assignment[name]
        for name in sorted(all_names - open_names)
    }
    total_params = sum(int(stats[n].get("n_params", 0) or 0) for n in all_names)
    open_params = sum(int(stats[n].get("n_params", 0) or 0) for n in open_names)
    if total_params <= 0:
        result = (dict(assignment), {})
        if return_metadata:
            return (*result, {"frozen_dp_precision_used": "none"})
        return result

    target_total_bits = float(target_bits) * float(total_params)
    budget_tolerance_bits = max(0.0, float(budget_tolerance)) * target_total_bits
    frozen_bits = assignment_bit_total(stats, frozen_assignment, specs_by_name)
    remaining_bits = target_total_bits - frozen_bits
    if remaining_bits < -1e-6:
        raise FrozenBudgetError(
            "L3 polish infeasible: frozen L2 choices already exceed target "
            f"budget ({frozen_bits / total_params:.6f} bpp frozen vs "
            f"{target_bits:.6f} bpp target)."
        )
    if open_params <= 0:
        result = (dict(assignment), {})
        if return_metadata:
            return (*result, {"frozen_dp_precision_used": "none"})
        return result

    open_target_bits = remaining_bits / float(open_params)
    open_stats = {name: dict(stats[name]) for name in sorted(open_names)}
    open_cands = {name: list(l3_candidates[name]) for name in sorted(open_names)}
    result = solve_allocation(open_stats, open_cands, open_target_bits, bit_precision)
    precision_used: float | str = float(bit_precision)
    dp_attempts = [{"precision": float(bit_precision), "result": "ok" if result is not None else "failed"}]
    if result is None:
        print(
            f"[l3] frozen DP precision {float(bit_precision):g}: failed",
            flush=True,
        )
        for fallback_precision in (0.01, 0.05, 0.25, 0.5, 1.0):
            result = solve_allocation(
                open_stats,
                open_cands,
                open_target_bits,
                fallback_precision,
            )
            dp_attempts.append({
                "precision": fallback_precision,
                "result": "ok" if result is not None else "failed",
            })
            if result is not None:
                precision_used = fallback_precision
                print(
                    f"[l3] frozen DP precision {fallback_precision:g}: ok",
                    flush=True,
                )
                break
            print(
                f"[l3] frozen DP precision {fallback_precision:g}: failed",
                flush=True,
            )
    if result is None:
        open_current_assignment = {
            name: assignment[name]
            for name in open_cands
            if name in assignment
        }
        open_assignment, chosen, greedy_stats = _greedy_l3_under_budget(
            open_cands,
            open_current_assignment,
            remaining_bits,
            remaining_bits + budget_tolerance_bits,
        )
        result = (open_assignment, chosen)
        precision_used = "greedy"
        print(
            "[l3] frozen DP greedy: "
            f"attempts={greedy_stats['attempts']} "
            f"accepted={greedy_stats['accepted']} "
            f"rejected_not_better={greedy_stats['rejected_not_better']} "
            f"rejected_budget={greedy_stats['rejected_budget']} "
            f"budget_ceiling_bits={greedy_stats['budget_ceiling_bits']:.1f}",
            flush=True,
        )
    else:
        greedy_stats = None
    open_assignment, chosen = result
    merged = dict(assignment)
    merged.update(open_assignment)
    if return_metadata:
        return merged, chosen, {
            "frozen_dp_precision_used": precision_used,
            "frozen_dp_attempts": dp_attempts,
            "frozen_dp_greedy": greedy_stats,
            "frozen_dp_budget_tolerance": float(budget_tolerance),
            "frozen_dp_budget_tolerance_bits": float(budget_tolerance_bits),
        }
    return merged, chosen


_LAYER_DEPTH_RE = re.compile(r"(?:^|[.])layers[.](\d+)(?:[.]|$)")


def layer_depth(name: str) -> int | None:
    """Best-effort decoder-layer depth parser for depth-grouped L3 batches."""
    m = _LAYER_DEPTH_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def _group_neighborhood_by_depth(
    entries: list[L3NeighborhoodEntry],
) -> list[tuple[str, list[L3NeighborhoodEntry]]]:
    grouped: dict[str, list[L3NeighborhoodEntry]] = {}
    for entry in entries:
        depth = layer_depth(entry.name)
        key = f"layer:{depth:05d}" if depth is not None else f"name:{entry.name}"
        grouped.setdefault(key, []).append(entry)
    return [(key, grouped[key]) for key in sorted(grouped)]


def _canonical_assignment(
    assignment: Mapping[str, str],
) -> dict[str, str]:
    return {
        str(name): fr.canonical_format_name(fmt)
        for name, fmt in assignment.items()
    }


def _first_tensor_batch_size(args, kwargs) -> int:
    for value in list(args) + list((kwargs or {}).values()):
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return int(value.size(0))
    raise ValueError("could not infer calibration batch size from model inputs")


def _repeat_value_for_lanes(value, lane_count: int):
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        repeats = (int(lane_count),) + (1,) * (value.dim() - 1)
        return value.repeat(repeats)
    if isinstance(value, Mapping):
        return {
            key: _repeat_value_for_lanes(child, lane_count)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_repeat_value_for_lanes(child, lane_count) for child in value)
    if isinstance(value, list):
        return [_repeat_value_for_lanes(child, lane_count) for child in value]
    return value


def _repeat_inputs_for_lanes(args, kwargs, lane_count: int):
    return (
        tuple(_repeat_value_for_lanes(value, lane_count) for value in args),
        {
            key: _repeat_value_for_lanes(value, lane_count)
            for key, value in (kwargs or {}).items()
        },
    )


def _extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


def _first_tensor_output(output) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, Mapping):
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    return None


def _replace_first_tensor_output(output, replacement: torch.Tensor):
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        values = list(output)
        for idx, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                values[idx] = replacement
                return tuple(values)
    if isinstance(output, dict):
        values = dict(output)
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = replacement
                return values
    return output


def _decoder_stack(model: nn.Module):
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
    ]
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        candidates.append(getattr(language_model, "model", None))
    for base in candidates:
        if base is None:
            continue
        layers = getattr(base, "layers", None)
        if layers is not None and hasattr(layers, "__len__"):
            return base, layers
    return None, None


def _replace_first_tensor_call(args, kwargs, replacement: torch.Tensor):
    args = list(args)
    for idx, value in enumerate(args):
        if isinstance(value, torch.Tensor):
            args[idx] = replacement
            return tuple(args), dict(kwargs or {})
    kwargs = dict(kwargs or {})
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            kwargs[key] = replacement
            return tuple(args), kwargs
    return (replacement, *tuple(args)), kwargs


def _repeat_layer_value_for_lanes(value, lane_count: int, base_batch: int):
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and int(value.size(0)) == int(base_batch):
            repeats = (int(lane_count),) + (1,) * (value.dim() - 1)
            return value.repeat(repeats)
        return value
    if isinstance(value, Mapping):
        return {
            key: _repeat_layer_value_for_lanes(child, lane_count, base_batch)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _repeat_layer_value_for_lanes(child, lane_count, base_batch)
            for child in value
        )
    if isinstance(value, list):
        return [
            _repeat_layer_value_for_lanes(child, lane_count, base_batch)
            for child in value
        ]
    return value


def _repeat_layer_call_for_lanes(args, kwargs, lane_count: int, base_batch: int):
    return (
        tuple(
            _repeat_layer_value_for_lanes(value, lane_count, base_batch)
            for value in args
        ),
        {
            key: _repeat_layer_value_for_lanes(value, lane_count, base_batch)
            for key, value in (kwargs or {}).items()
        },
    )


class _TailLayerCaptureDone(Exception):
    pass


def _clone_layer_value_for_cache(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_layer_value_for_cache(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_layer_value_for_cache(child) for child in value)
    if isinstance(value, list):
        return [_clone_layer_value_for_cache(child) for child in value]
    return value


def _move_cached_layer_value(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {
            key: _move_cached_layer_value(child, device)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_cached_layer_value(child, device) for child in value)
    if isinstance(value, list):
        return [_move_cached_layer_value(child, device) for child in value]
    return value


def _move_cached_layer_call(cached_call, device):
    args, kwargs, base_batch = cached_call
    return (
        tuple(_move_cached_layer_value(value, device) for value in args),
        {
            key: _move_cached_layer_value(value, device)
            for key, value in kwargs.items()
        },
        base_batch,
    )


def _model_accepts_kwarg(model: nn.Module, name: str) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


def _capture_layer_call(model: nn.Module, layer: nn.Module, args, kwargs):
    captured = {}

    def _hook(_module, hook_args, hook_kwargs):
        layer_kwargs = dict(hook_kwargs or {})
        if "use_cache" in layer_kwargs:
            layer_kwargs["use_cache"] = False
        if "past_key_value" in layer_kwargs:
            layer_kwargs["past_key_value"] = None
        captured["args"] = tuple(hook_args)
        captured["kwargs"] = layer_kwargs
        raise _TailLayerCaptureDone

    handle = layer.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        try:
            model(*args, **(kwargs or {}))
        except _TailLayerCaptureDone:
            pass
    finally:
        handle.remove()
    if "args" not in captured:
        raise RuntimeError("tail-only L3 could not capture decoder layer inputs")
    return captured["args"], captured["kwargs"]


def _capture_all_layer_calls(
    model: nn.Module,
    layers,
    layer_indices: set[int],
    calibration_data,
    device,
) -> dict[int, list[tuple[tuple, dict, int]]]:
    captured: dict[int, list[tuple[tuple, dict, int]]] = {
        idx: [] for idx in sorted(layer_indices)
    }
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(_module, hook_args, hook_kwargs):
            layer_kwargs = dict(hook_kwargs or {})
            if "use_cache" in layer_kwargs:
                layer_kwargs["use_cache"] = False
            if "past_key_value" in layer_kwargs:
                layer_kwargs["past_key_value"] = None
            base_batch = _first_tensor_batch_size(hook_args, layer_kwargs)
            captured[layer_idx].append(
                (
                    tuple(_clone_layer_value_for_cache(value) for value in hook_args),
                    {
                        key: _clone_layer_value_for_cache(value)
                        for key, value in layer_kwargs.items()
                    },
                    int(base_batch),
                )
            )

        return _hook

    for layer_idx in sorted(layer_indices):
        layer = layers[layer_idx]
        handles.append(layer.register_forward_pre_hook(
            _make_hook(layer_idx),
            with_kwargs=True,
        ))
    try:
        for args, kwargs in iter_calibration_forwards(calibration_data, device):
            call_kwargs = dict(kwargs or {})
            if _model_accepts_kwarg(model, "use_cache"):
                call_kwargs["use_cache"] = False
            model(*args, **call_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _tail_forward_eager(
    model: nn.Module,
    layer_idx: int,
    layer_args,
    layer_kwargs,
    hidden_state: torch.Tensor,
) -> torch.Tensor:
    """Run decoder layers after ``layer_idx`` plus final norm and LM head."""
    base, layers = _decoder_stack(model)
    if layers is None:
        raise RuntimeError("tail-only L3 requires a decoder layer stack")
    hidden = hidden_state
    for next_idx in range(int(layer_idx) + 1, len(layers)):
        call_args, call_kwargs = _replace_first_tensor_call(
            layer_args,
            layer_kwargs,
            hidden,
        )
        output = layers[next_idx](*call_args, **call_kwargs)
        next_hidden = _first_tensor_output(output)
        if next_hidden is None:
            raise RuntimeError("tail-only L3 decoder layer returned no tensor")
        hidden = next_hidden
    norm = getattr(base, "norm", None)
    if norm is not None:
        hidden = norm(hidden)
    lm_head = getattr(model, "lm_head", None) or getattr(base, "lm_head", None)
    if lm_head is not None:
        return lm_head(hidden)
    return hidden


def _tensor_tree_signature(value):
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__name__,
            tuple(
                sorted(
                    (str(key), _tensor_tree_signature(child))
                    for key, child in value.items()
                )
            ),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_tensor_tree_signature(child) for child in value))
    if isinstance(value, list):
        return ("list", tuple(_tensor_tree_signature(child) for child in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", type(value).__name__, value)
    return ("object", type(value).__name__, id(value))


def _clone_static_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_static_tree(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_static_tree(child) for child in value)
    if isinstance(value, list):
        return [_clone_static_tree(child) for child in value]
    return value


def _copy_static_tree(src, dst) -> bool:
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        if (
            tuple(src.shape) != tuple(dst.shape)
            or src.dtype != dst.dtype
            or src.device != dst.device
        ):
            return False
        dst.copy_(src)
        return True
    if isinstance(src, Mapping) and isinstance(dst, Mapping):
        if set(src.keys()) != set(dst.keys()):
            return False
        return all(_copy_static_tree(src[key], dst[key]) for key in src)
    if isinstance(src, tuple) and isinstance(dst, tuple):
        if len(src) != len(dst):
            return False
        return all(_copy_static_tree(a, b) for a, b in zip(src, dst))
    if isinstance(src, list) and isinstance(dst, list):
        if len(src) != len(dst):
            return False
        return all(_copy_static_tree(a, b) for a, b in zip(src, dst))
    if src is dst:
        return True
    if src is None or isinstance(src, (bool, int, float, str)):
        return src == dst
    return False


def _first_cuda_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value if value.is_cuda else None
    if isinstance(value, Mapping):
        for child in value.values():
            found = _first_cuda_tensor(child)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for child in value:
            found = _first_cuda_tensor(child)
            if found is not None:
                return found
    return None


def _clone_cuda_graph_output(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_cuda_graph_output(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_cuda_graph_output(child) for child in value)
    if isinstance(value, list):
        return [_clone_cuda_graph_output(child) for child in value]
    return value


_CUDA_GRAPH_WARNED_LABELS: set[str] = set()


def _warn_cuda_graph_fallback_once(label: str, exc: BaseException) -> None:
    if label in _CUDA_GRAPH_WARNED_LABELS:
        return
    _CUDA_GRAPH_WARNED_LABELS.add(label)
    print(
        "[cuda-graphs] warning: "
        f"{label} capture/replay failed once; using eager for that shape "
        f"({type(exc).__name__}: {exc})",
        file=sys.stderr,
        flush=True,
    )


@dataclass
class _CUDAGraphEntry:
    graph: object
    static_args: tuple
    static_kwargs: dict
    static_output: object
    keepalive: tuple[object, ...] = field(default_factory=tuple)


class CUDAGraphRegistry:
    """Bounded LRU CUDA graph cache for fixed-shape tensor forwards.

    Each entry owns graph activation memory plus static input/output tensors.
    The default cap is intentionally small and can be overridden per path with
    the registry's ``max_entries_env`` variable.
    """

    def __init__(
        self,
        *,
        label: str,
        max_entries: int = 4,
        max_entries_env: str | None = None,
        warmup_iters: int = 2,
    ):
        self.label = str(label)
        self.default_max_entries = max(int(max_entries), 0)
        self.max_entries_env = max_entries_env
        self.warmup_iters = max(int(warmup_iters), 0)
        self.entries: OrderedDict[tuple, _CUDAGraphEntry] = OrderedDict()
        self.disabled_keys: set[tuple] = set()

    def clear(self) -> None:
        self.entries.clear()
        self.disabled_keys.clear()

    def _max_entries(self) -> int:
        if self.max_entries_env is None:
            return self.default_max_entries
        return _env_int(self.max_entries_env, self.default_max_entries)

    def _evict_if_needed(self) -> None:
        max_entries = self._max_entries()
        if max_entries <= 0:
            self.entries.clear()
            return
        while len(self.entries) > max_entries:
            self.entries.popitem(last=False)

    def run(
        self,
        label: str,
        key: tuple,
        fn: Callable,
        *args,
        enabled: bool = True,
        device: torch.device | None = None,
        keepalive: tuple[object, ...] = (),
        **kwargs,
    ):
        cuda_tensor = _first_cuda_tensor((args, kwargs))
        graph_device = device
        if graph_device is None and cuda_tensor is not None:
            graph_device = cuda_tensor.device
        if (
            not enabled
            or not torch.cuda.is_available()
            or graph_device is None
            or torch.device(graph_device).type != "cuda"
            or self._max_entries() <= 0
        ):
            return fn(*args, **kwargs)

        full_key = (
            self.label,
            str(label),
            tuple(key),
            _tensor_tree_signature(args),
            _tensor_tree_signature(kwargs),
        )
        entry = self.entries.get(full_key)
        if entry is not None:
            self.entries.move_to_end(full_key)
            if not (
                _copy_static_tree(tuple(args), entry.static_args)
                and _copy_static_tree(dict(kwargs), entry.static_kwargs)
            ):
                return fn(*args, **kwargs)
            try:
                entry.graph.replay()
                return _clone_cuda_graph_output(entry.static_output)
            except Exception as exc:
                self.entries.pop(full_key, None)
                self.disabled_keys.add(full_key)
                _warn_cuda_graph_fallback_once(str(label), exc)
                return fn(*args, **kwargs)
        if full_key in self.disabled_keys:
            return fn(*args, **kwargs)

        try:
            entry = self._capture(
                fn,
                args,
                kwargs,
                torch.device(graph_device),
                keepalive=keepalive,
            )
        except Exception as exc:
            self.disabled_keys.add(full_key)
            _warn_cuda_graph_fallback_once(str(label), exc)
            return fn(*args, **kwargs)
        self.entries[full_key] = entry
        self._evict_if_needed()
        return _clone_cuda_graph_output(entry.static_output)

    def _capture(
        self,
        fn: Callable,
        args: tuple,
        kwargs: Mapping,
        device: torch.device,
        *,
        keepalive: tuple[object, ...],
    ) -> _CUDAGraphEntry:
        static_args = tuple(_clone_static_tree(value) for value in args)
        static_kwargs = {
            key: _clone_static_tree(value)
            for key, value in dict(kwargs).items()
        }
        current_stream = torch.cuda.current_stream(device)
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream), torch.no_grad():
            for _ in range(self.warmup_iters):
                fn(*static_args, **static_kwargs)
        current_stream.wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            static_output = fn(*static_args, **static_kwargs)
        return _CUDAGraphEntry(
            graph=graph,
            static_args=static_args,
            static_kwargs=static_kwargs,
            static_output=static_output,
            keepalive=tuple(keepalive),
        )


_COORD_LANE_CUDA_GRAPH_REGISTRY = CUDAGraphRegistry(
    label="coord-lane",
    max_entries=4,
    max_entries_env="PRISMAQUANT_COORD_LANE_CUDA_GRAPH_CACHE_SIZE",
)


@dataclass
class _TailCudaGraphEntry:
    graph: object
    static_hidden: torch.Tensor
    static_args: tuple
    static_kwargs: dict
    static_output: torch.Tensor


class _TailCudaGraphCache:
    def __init__(self, *, enabled: bool):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.entries: dict[tuple, _TailCudaGraphEntry] = {}
        self.disabled_keys: set[tuple] = set()

    def clear(self) -> None:
        self.entries.clear()
        self.disabled_keys.clear()

    def run(
        self,
        model: nn.Module,
        layer_idx: int,
        layer_args,
        layer_kwargs,
        hidden_state: torch.Tensor,
        *,
        lane_count: int,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or not isinstance(hidden_state, torch.Tensor)
            or not hidden_state.is_cuda
        ):
            return _tail_forward_eager(
                model,
                layer_idx,
                layer_args,
                layer_kwargs,
                hidden_state,
            )
        key = (
            id(model),
            int(layer_idx),
            int(lane_count),
            _tensor_tree_signature(hidden_state),
            _tensor_tree_signature(layer_args),
            _tensor_tree_signature(layer_kwargs or {}),
        )
        entry = self.entries.get(key)
        if entry is not None:
            if not (
                _copy_static_tree(hidden_state, entry.static_hidden)
                and _copy_static_tree(tuple(layer_args), entry.static_args)
                and _copy_static_tree(dict(layer_kwargs or {}), entry.static_kwargs)
            ):
                return _tail_forward_eager(
                    model,
                    layer_idx,
                    layer_args,
                    layer_kwargs,
                    hidden_state,
                )
            entry.graph.replay()
            return entry.static_output.clone()
        if key in self.disabled_keys:
            return _tail_forward_eager(
                model,
                layer_idx,
                layer_args,
                layer_kwargs,
                hidden_state,
            )
        try:
            entry = self._capture(
                model,
                layer_idx,
                layer_args,
                layer_kwargs,
                hidden_state,
            )
        except Exception:
            self.disabled_keys.add(key)
            return _tail_forward_eager(
                model,
                layer_idx,
                layer_args,
                layer_kwargs,
                hidden_state,
            )
        self.entries[key] = entry
        return entry.static_output.clone()

    def _capture(
        self,
        model: nn.Module,
        layer_idx: int,
        layer_args,
        layer_kwargs,
        hidden_state: torch.Tensor,
    ) -> _TailCudaGraphEntry:
        static_hidden = hidden_state.detach().clone()
        static_args = tuple(_clone_static_tree(value) for value in layer_args)
        static_kwargs = {
            key: _clone_static_tree(value)
            for key, value in (layer_kwargs or {}).items()
        }
        device = hidden_state.device
        current_stream = torch.cuda.current_stream(device)
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                _tail_forward_eager(
                    model,
                    layer_idx,
                    static_args,
                    static_kwargs,
                    static_hidden,
                )
        current_stream.wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = _tail_forward_eager(
                model,
                layer_idx,
                static_args,
                static_kwargs,
                static_hidden,
            )
        return _TailCudaGraphEntry(
            graph=graph,
            static_hidden=static_hidden,
            static_args=static_args,
            static_kwargs=static_kwargs,
            static_output=static_output,
        )


def tail_forward_from_layer(
    model: nn.Module,
    layer_idx: int,
    layer_args,
    layer_kwargs,
    hidden_state: torch.Tensor,
    *,
    cuda_graph_cache: _TailCudaGraphCache | None = None,
    lane_count: int | None = None,
) -> torch.Tensor:
    if cuda_graph_cache is not None:
        return cuda_graph_cache.run(
            model,
            layer_idx,
            layer_args,
            layer_kwargs,
            hidden_state,
            lane_count=lane_count or 1,
        )
    return _tail_forward_eager(
        model,
        layer_idx,
        layer_args,
        layer_kwargs,
        hidden_state,
    )


def _split_lanes(tensor: torch.Tensor, base_batch: int, lane_count: int):
    if tensor.dim() == 0 or tensor.size(0) != base_batch * lane_count:
        return None
    return tensor.split(base_batch, dim=0)


def _coord_replay_target_keys(
    replay_cache: LayerHiddenStateCache,
    target_names: set[str],
) -> tuple[set[object], set[int]]:
    by_name = getattr(replay_cache, "_linear_targets_by_name", {})
    target_keys: set[object] = set()
    module_ids: set[int] = set()
    for raw_name in target_names:
        candidates = [raw_name]
        if raw_name.endswith(".weight"):
            candidates.append(raw_name[:-7])
        else:
            candidates.append(f"{raw_name}.weight")
        for name in candidates:
            target = by_name.get(name)
            if target is None:
                continue
            target_keys.add(target.key)
            module_ids.add(id(target.module))
            break
    return target_keys, module_ids


def _repeat_replay_template_for_lanes(template, lane_count: int, base_batch: int):
    return replace(
        template,
        args=tuple(
            _repeat_layer_value_for_lanes(value, lane_count, base_batch)
            for value in template.args
        ),
        kwargs={
            key: _repeat_layer_value_for_lanes(value, lane_count, base_batch)
            for key, value in template.kwargs.items()
        },
    )


def _lane_replay_cache_logits(
    replay_cache: LayerHiddenStateCache,
    layer_idx: int,
    *,
    lane_count: int,
    base_batch: int,
    target_names: set[str],
) -> torch.Tensor:
    """Replay a populated LayerHiddenStateCache with lane-repeated state.

    LayerHiddenStateCache intentionally exposes scalar replay. Coord descent
    keeps lane semantics here by temporarily repeating the cached layer input
    and non-hidden layer-call tensors, while leaving target modules at live
    BF16 weights so _DepthGroupTargetHooks can choose the per-lane format.
    """
    original_inputs = list(replay_cache.layer_inputs)
    original_templates = list(getattr(replay_cache, "_layer_call_templates"))
    original_baseline_weights = dict(
        getattr(replay_cache, "_baseline_weight_values")
    )
    original_activation_quantizers = dict(
        getattr(replay_cache, "_activation_quantizers")
    )
    target_keys, target_module_ids = _coord_replay_target_keys(
        replay_cache,
        target_names,
    )
    try:
        replay_cache.layer_inputs = list(original_inputs)
        replay_cache.layer_inputs[layer_idx] = _repeat_layer_value_for_lanes(
            original_inputs[layer_idx],
            lane_count,
            base_batch,
        )
        repeated_templates = list(original_templates)
        for idx in range(layer_idx, len(repeated_templates)):
            repeated_templates[idx] = _repeat_replay_template_for_lanes(
                repeated_templates[idx],
                lane_count,
                base_batch,
            )
        replay_cache._layer_call_templates = repeated_templates
        replay_cache._baseline_weight_values = {
            key: value
            for key, value in original_baseline_weights.items()
            if key not in target_keys
        }
        replay_cache._activation_quantizers = {
            module_id: value
            for module_id, value in original_activation_quantizers.items()
            if module_id not in target_module_ids
        }
        return replay_cache.replay_from(layer_idx)
    finally:
        replay_cache.layer_inputs = original_inputs
        replay_cache._layer_call_templates = original_templates
        replay_cache._baseline_weight_values = original_baseline_weights
        replay_cache._activation_quantizers = original_activation_quantizers


def _l3_quantizable_map(model: nn.Module) -> dict[str, tuple[nn.Module, str]]:
    """Map L3 names to modules, including tiny nn.Linear modules in tests."""
    out = dict(build_quantizable_map(model))
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        names = {name, f"{name}.weight"}
        if name.startswith("model."):
            suffix = name[len("model."):]
            names.add(f"model.language_model.{suffix}")
            names.add(f"model.language_model.{suffix}.weight")
        for candidate in names:
            out.setdefault(candidate, (module, "weight"))
    return out


def apply_format_quantization(
    weight: torch.Tensor,
    spec: fr.FormatSpec,
) -> torch.Tensor:
    return spec.quantize_dequantize(weight.detach().clone())


def build_quant_weight_cache(
    model: nn.Module,
    neighborhood: list[L3NeighborhoodEntry],
    specs: list[fr.FormatSpec],
    *,
    skip_bf16: bool = True,
) -> QuantWeightCache:
    quant_map = _l3_quantizable_map(model)
    cache: dict[tuple[str, str], torch.Tensor] = {}
    for entry in neighborhood:
        target = quant_map.get(entry.name)
        if target is None:
            continue
        linear, attr = target
        if not isinstance(linear, nn.Linear) or attr != "weight":
            continue
        name_keys = {
            name
            for name, (candidate_module, candidate_attr) in quant_map.items()
            if candidate_module is linear and candidate_attr == attr
        }
        name_keys.add(entry.name)
        original_weight = linear.weight.data
        for spec in specs:
            canonical = fr.canonical_format_name(spec.name)
            if skip_bf16 and canonical == "BF16":
                continue
            quantized = apply_format_quantization(original_weight, spec).to(
                device=original_weight.device,
                dtype=original_weight.dtype,
            )
            quantized = quantized.contiguous()
            fmt_keys = {canonical, spec.name, *fr.aliases_for(spec.name)}
            for name_key in name_keys:
                for fmt_key in fmt_keys:
                    cache[(name_key, fmt_key)] = quantized
    return QuantWeightCache(cache)


class _DepthGroupTargetHooks:
    """Apply lane-specific target formats for one depth-group microbatch.

    The normal L2 context hooks are installed for every non-target module.
    Group targets are excluded from that context, then these hooks apply either
    the lane's candidate format, that target's paired BF16 baseline, or the
    target's original L2 format for lanes belonging to other targets in the
    same depth group. This avoids double-quantizing a target module while
    preserving "all other modules at the L2 assignment" semantics.
    """

    def __init__(
        self,
        model: nn.Module,
        assignment: Mapping[str, str],
        specs_by_name: Mapping[str, fr.FormatSpec],
        lanes: list[_LaneSpec],
        *,
        base_batch: int,
        quant_weight_cache: QuantWeightCache | None = None,
    ):
        self.model = model
        self.assignment = _canonical_assignment(assignment)
        self.specs_by_name = specs_by_name
        self.lanes = lanes
        self.base_batch = int(base_batch)
        self.quant_weight_cache = quant_weight_cache
        self.handles = []
        self.missing: list[str] = []

    def install(self) -> None:
        quant_map = _l3_quantizable_map(self.model)
        target_names = sorted({lane.name for lane in self.lanes})
        for name in target_names:
            target = quant_map.get(name)
            if target is None:
                self.missing.append(name)
                continue
            module, _attr = target
            if not isinstance(module, nn.Linear):
                self.missing.append(name)
                continue
            self.handles.append(
                module.register_forward_hook(
                    self._make_hook(name),
                    with_kwargs=True,
                )
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _format_for_lane(self, module_name: str, lane: _LaneSpec) -> str:
        if lane.name == module_name:
            return "BF16" if lane.is_baseline else lane.fmt
        return self.assignment.get(module_name, "BF16")

    def _make_hook(self, module_name: str):
        def _hook(module, args, kwargs, output):
            y = _first_tensor_output(output)
            if y is None:
                return output
            chunks = _split_lanes(y, self.base_batch, len(self.lanes))
            if chunks is None:
                return output
            x = None
            for value in list(args) + list((kwargs or {}).values()):
                if isinstance(value, torch.Tensor):
                    x = value
                    break
            if x is None:
                return output
            x_chunks = _split_lanes(x, self.base_batch, len(self.lanes))
            if x_chunks is None:
                return output

            out_chunks = []
            weight = module.weight.detach()
            bias = module.bias.detach() if module.bias is not None else None
            for lane, y_lane, x_lane in zip(self.lanes, chunks, x_chunks):
                fmt = self._format_for_lane(module_name, lane)
                if fmt == "BF16":
                    out_chunks.append(y_lane)
                    continue
                spec = self.specs_by_name.get(fmt)
                if spec is None:
                    out_chunks.append(y_lane)
                    continue
                w_hat = None
                if self.quant_weight_cache is not None:
                    w_hat = self.quant_weight_cache.get(module_name, fmt)
                if w_hat is None:
                    w_hat = apply_format_quantization(weight, spec)
                x_hat = spec.activation_quantize_dequantize(x_lane)
                out_chunks.append(F.linear(x_hat, w_hat.to(weight.dtype), bias))
            replacement = torch.cat(out_chunks, dim=0)
            return _replace_first_tensor_output(output, replacement)

        return _hook


class _LaneOutputMSE:
    def __init__(
        self,
        model: nn.Module,
        names: list[str],
        lanes: list[_LaneSpec],
        *,
        base_batch: int,
    ):
        self.model = model
        self.names = names
        self.lanes = lanes
        self.base_batch = int(base_batch)
        self.handles = []
        self.total_by_lane = [0.0 for _ in lanes]
        self.batch_count = 0

    def install(self) -> None:
        quant_map = _l3_quantizable_map(self.model)
        seen_modules: set[int] = set()
        for name in self.names:
            target = quant_map.get(name)
            if target is None:
                continue
            module, _attr = target
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            self.handles.append(
                module.register_forward_hook(self._hook, with_kwargs=True)
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def mark_batch(self) -> None:
        self.batch_count += 1

    def _hook(self, _module, _args, _kwargs, output):
        y = _first_tensor_output(output)
        if y is None:
            return output
        chunks = _split_lanes(y.detach(), self.base_batch, len(self.lanes))
        if chunks is None:
            return output
        for idx, lane in enumerate(self.lanes):
            if lane.is_baseline or lane.baseline_index is None:
                continue
            base = chunks[lane.baseline_index].float()
            cand = chunks[idx].float()
            self.total_by_lane[idx] += float((cand - base).pow(2).mean().item())
        return output

    def value_for_lane(self, lane_index: int) -> float:
        return self.total_by_lane[lane_index] / max(self.batch_count, 1)


def _ordered_quantizable_names(model: nn.Module, assignment_names: set[str]) -> list[str]:
    quant_map = _l3_quantizable_map(model)
    module_to_names: dict[int, list[str]] = {}
    for name in assignment_names:
        target = quant_map.get(name)
        if target is None:
            continue
        module_to_names.setdefault(id(target[0]), []).append(name)

    ordered: list[str] = []
    seen: set[str] = set()
    for _qname, module in model.named_modules():
        for name in sorted(module_to_names.get(id(module), [])):
            if name not in seen:
                ordered.append(name)
                seen.add(name)
    return ordered


def _downstream_names_for_group(
    ordered_names: list[str],
    group_names: set[str],
) -> list[str]:
    positions = [
        idx for idx, name in enumerate(ordered_names)
        if name in group_names
    ]
    if not positions:
        return []
    return ordered_names[min(positions):]


def _lane_specs_for_entries(
    entries: list[L3NeighborhoodEntry],
    *,
    include_baseline: bool = True,
) -> list[_LaneSpec]:
    lanes: list[_LaneSpec] = []
    for entry in entries:
        candidate_fmts = [
            fr.canonical_format_name(fmt)
            for fmt in entry.formats
            if not include_baseline or fr.canonical_format_name(fmt) != "BF16"
        ]
        if not candidate_fmts:
            continue
        if not include_baseline:
            for fmt in candidate_fmts:
                lanes.append(
                    _LaneSpec(
                        name=entry.name,
                        fmt=fmt,
                        baseline_index=None,
                        is_baseline=False,
                    )
                )
            continue
        baseline_idx = len(lanes)
        lanes.append(
            _LaneSpec(
                name=entry.name,
                fmt="BF16",
                baseline_index=None,
                is_baseline=True,
            )
        )
        for fmt in candidate_fmts:
            lanes.append(
                _LaneSpec(
                    name=entry.name,
                    fmt=fmt,
                    baseline_index=baseline_idx,
                    is_baseline=False,
                )
            )
    return lanes


def _lane_microbatches_for_entries(
    entries: list[L3NeighborhoodEntry],
    max_lanes_per_batch: int,
    *,
    include_baseline: bool = True,
) -> list[list[_LaneSpec]]:
    batches: list[list[_LaneSpec]] = []
    current: list[_LaneSpec] = []
    max_lanes = max(int(max_lanes_per_batch), 1)
    for entry in entries:
        entry_lanes = _lane_specs_for_entries(
            [entry],
            include_baseline=include_baseline,
        )
        if not entry_lanes:
            continue
        if current and len(current) + len(entry_lanes) > max_lanes:
            batches.append(current)
            current = []
        if len(entry_lanes) > max_lanes:
            batches.append(entry_lanes)
        else:
            current.extend(entry_lanes)
    if current:
        batches.append(current)
    return batches


def _specs_by_canonical_name(format_names: set[str]) -> dict[str, fr.FormatSpec]:
    specs_by_name: dict[str, fr.FormatSpec] = {}
    for fmt in sorted(format_names):
        canonical = fr.canonical_format_name(fmt)
        if canonical == "BF16":
            continue
        spec = fr.get_format(canonical)
        specs_by_name[spec.name] = spec
        specs_by_name[canonical] = spec
        for alias in fr.aliases_for(spec.name):
            specs_by_name[alias] = spec
    return specs_by_name


def _entries_for_candidate_flips(
    candidate_flips: list[tuple[str, str]],
    assignment: Mapping[str, str],
) -> list[L3NeighborhoodEntry]:
    return [
        L3NeighborhoodEntry(
            name=str(name),
            current_format=assignment.get(str(name), "BF16"),
            formats=(fr.canonical_format_name(fmt),),
            margin=0.0,
            l2_current_cost=0.0,
        )
        for name, fmt in candidate_flips
    ]


def _cuda_graph_lane_count(lane_count: int) -> int:
    for candidate in (1, 2, 4, 8, 16, 32, 64):
        if int(lane_count) <= candidate:
            return candidate
    return int(lane_count)


def _pad_lanes_for_cuda_graph(lanes: list[_LaneSpec]) -> list[_LaneSpec]:
    padded_count = _cuda_graph_lane_count(len(lanes))
    if padded_count <= len(lanes) or not lanes:
        return lanes
    dummy_source = lanes[-1]
    padded = list(lanes)
    padded.extend(
        _LaneSpec(
            name=dummy_source.name,
            fmt="BF16",
            baseline_index=None,
            is_baseline=True,
        )
        for _ in range(padded_count - len(lanes))
    )
    return padded


def _calibration_sample_tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_calibration_sample_tensor_bytes(child) for child in value.values())
    if isinstance(value, tuple | list):
        return sum(_calibration_sample_tensor_bytes(child) for child in value)
    return 0


def _estimate_l3_microbatch_memory_bytes(calibration_data, lane_count: int) -> int:
    if isinstance(calibration_data, torch.Tensor):
        if calibration_data.dim() == 0 or calibration_data.size(0) == 0:
            sample = calibration_data
        else:
            sample = calibration_data[:1]
        base_bytes = _calibration_sample_tensor_bytes(sample)
    elif isinstance(calibration_data, Mapping):
        base_bytes = _calibration_sample_tensor_bytes(calibration_data)
    elif isinstance(calibration_data, (tuple, list)) and calibration_data:
        base_bytes = _calibration_sample_tensor_bytes(calibration_data[0])
    else:
        base_bytes = 0
    return int(base_bytes * max(int(lane_count), 1) * 4)


def _adjust_l3_max_lanes_for_memory(
    max_lanes_per_batch: int,
    calibration_data,
    device: torch.device,
) -> int:
    requested = max(int(max_lanes_per_batch), 1)
    if device.type != "cuda" or not torch.cuda.is_available():
        return requested
    headroom_gb = max(
        _env_float("PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_GB", 8.0),
        0.0,
    )
    headroom_bytes = int(headroom_gb * 1024 ** 3)
    estimated_bytes = _estimate_l3_microbatch_memory_bytes(
        calibration_data,
        requested,
    )
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    except TypeError:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    if int(free_bytes) < headroom_bytes + estimated_bytes and requested > 1:
        return max(requested // 2, 1)
    return requested


def _output_mse_names_reach_tail(
    names: list[str],
    group_depth: int | None,
) -> bool:
    if group_depth is None:
        return bool(names)
    for name in names:
        depth = layer_depth(name)
        if depth is None or depth > group_depth:
            return True
    return False


@torch.no_grad()
def measure_lane_batched_kl_deltas(
    model: nn.Module,
    baseline_assignment: Mapping[str, str],
    candidate_flips: list[tuple[str, str]],
    calib_ids: torch.Tensor,
    ref_log_probs: list[torch.Tensor],
    *,
    work_root: Path,
    max_lanes_per_batch: int = 64,
    profile=None,
    replay_cache: LayerHiddenStateCache | None = None,
) -> list[float]:
    """Measure end-KL for each candidate flip applied to baseline_assignment.

    Each lane is one ``(Linear, format)`` override. Lanes may target different
    Linear modules; target hooks apply the candidate format for the matching
    lane and the baseline assignment for all other target modules in that lane.
    """
    if not candidate_flips:
        return []

    assignment_c = _canonical_assignment(baseline_assignment)
    flips = [
        (str(name), fr.canonical_format_name(fmt))
        for name, fmt in candidate_flips
    ]
    format_names = set(assignment_c.values()) | {fmt for _name, fmt in flips}
    specs_by_name = _specs_by_canonical_name(format_names)

    device = next(model.parameters()).device
    requested_max_lanes_per_batch = max(int(max_lanes_per_batch), 1)
    max_lanes_per_batch = _adjust_l3_max_lanes_for_memory(
        requested_max_lanes_per_batch,
        calib_ids,
        device,
    )
    entries = _entries_for_candidate_flips(flips, assignment_c)
    batches = _lane_microbatches_for_entries(
        entries,
        max_lanes_per_batch,
        include_baseline=False,
    )
    cal_hash = calibration_data_hash(calib_ids)
    tmp_parent = str(work_root) if work_root is not None else None
    use_prequant_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_PREQUANT_CACHE",
        default=True,
    )
    use_frozen_perturbed_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE",
        default=True,
    )
    use_coord_lane_cuda_graphs = _env_flag_enabled(
        "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS",
        default=True,
    )
    assignment_key = tuple(sorted(assignment_c.items()))
    rng_devices = []
    if device.type == "cuda" and torch.cuda.is_available():
        rng_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    measured: list[float] = []
    for lanes in batches:
        if not lanes:
            continue
        target_names = {lane.name for lane in lanes}
        context_assignment = {
            name: fmt
            for name, fmt in assignment_c.items()
            if name not in target_names
        }
        cache_entries = [
            L3NeighborhoodEntry(
                name=name,
                current_format=assignment_c.get(name, "BF16"),
                formats=tuple(
                    sorted({
                        assignment_c.get(name, "BF16"),
                        *[
                            lane.fmt
                            for lane in lanes
                            if lane.name == name
                        ],
                    })
                ),
                margin=0.0,
                l2_current_cost=0.0,
            )
            for name in sorted(target_names)
        ]
        group_quant_cache = (
            build_quant_weight_cache(
                model,
                cache_entries,
                list({id(spec): spec for spec in specs_by_name.values()}.values()),
            )
            if use_prequant_cache
            else None
        )
        lane_depths = [layer_depth(lane.name) for lane in lanes]
        replay_layer_idx = (
            min(depth for depth in lane_depths if depth is not None)
            if (
                replay_cache is not None
                and lane_depths
                and all(depth is not None for depth in lane_depths)
            )
            else None
        )
        use_replay_cache = (
            replay_cache is not None
            and replay_layer_idx is not None
            and 0 <= replay_layer_idx < len(replay_cache.layers)
            and _env_flag_enabled(
                "PRISMAQUANT_COORD_REPLAY_CACHE",
                default=True,
            )
        )
        if use_replay_cache:
            target_hooks = None
            kl_totals = [0.0 for _lane in lanes]
            batch_count = (
                int(calib_ids.size(0))
                if isinstance(calib_ids, torch.Tensor)
                else 0
            )
            base_batch = batch_count
            rng_cm = torch.random.fork_rng(devices=rng_devices)
            try:
                with rng_cm:
                    torch.manual_seed(0)
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.manual_seed_all(0)
                    target_hooks = _DepthGroupTargetHooks(
                        model,
                        assignment_c,
                        specs_by_name,
                        lanes,
                        base_batch=base_batch,
                        quant_weight_cache=group_quant_cache,
                    )
                    target_hooks.install()
                    lane_key = tuple(
                        (lane.name, lane.fmt, lane.baseline_index, lane.is_baseline)
                        for lane in lanes
                    )

                    def _replay_forward():
                        return _lane_replay_cache_logits(
                            replay_cache,
                            int(replay_layer_idx),
                            lane_count=len(lanes),
                            base_batch=base_batch,
                            target_names=target_names,
                        )

                    logits = _COORD_LANE_CUDA_GRAPH_REGISTRY.run(
                        "coord-lane-replay",
                        (
                            "replay",
                            id(model),
                            id(replay_cache),
                            assignment_key,
                            cal_hash,
                            int(replay_layer_idx),
                            int(len(lanes)),
                            int(base_batch),
                            lane_key,
                            tuple(sorted(target_names)),
                        ),
                        _replay_forward,
                        enabled=use_coord_lane_cuda_graphs,
                        device=device,
                        keepalive=(
                            replay_cache,
                            target_hooks,
                            group_quant_cache,
                        ),
                    )
                    logits = _extract_logits(logits)
                    if logits.dim() >= 3:
                        logits = logits[:, -1:, :]
                    chunks = _split_lanes(logits.detach(), base_batch, len(lanes))
                    if chunks is None:
                        raise RuntimeError(
                            "lane-batched coord replay logits did not preserve lane "
                            f"batching: shape={tuple(logits.shape)} "
                            f"base_batch={base_batch} lanes={len(lanes)}"
                        )
                    for idx, chunk in enumerate(chunks):
                        for batch_index, teacher in enumerate(ref_log_probs):
                            teacher = teacher.to(chunk.device)
                            if teacher.dim() >= 3:
                                teacher = teacher[:, -1:, :]
                            kl_totals[idx] += float(
                                kl_divergence(
                                    chunk[batch_index:batch_index + 1],
                                    teacher,
                                ).item()
                            )
                missing_targets = set(target_hooks.missing if target_hooks else [])
                if missing_targets:
                    raise RuntimeError(
                        "target module missing or unsupported for lane-batched KL: "
                        + ", ".join(sorted(missing_targets))
                    )
                measured.extend(
                    total / max(batch_count, 1)
                    for total in kl_totals
                )
            finally:
                if target_hooks is not None:
                    target_hooks.remove()
            continue

        cache_dir = Path(tempfile.mkdtemp(
            prefix="prismaquant_coord_lanes_",
            dir=tmp_parent,
        ))
        context_hooks = PerturbedActivationCache(
            model,
            context_assignment,
            cache_dir,
            input_rows=0,
            cal_hash=cal_hash,
            profile=profile,
        )
        frozen_context = (
            context_hooks.frozen_weight_cache()
            if use_frozen_perturbed_cache
            else nullcontext(context_hooks)
        )
        target_hooks = None
        kl_totals = [0.0 for _lane in lanes]
        batch_count = 0
        frozen_context_entered = False
        rng_cm = torch.random.fork_rng(devices=rng_devices)
        try:
            frozen_context.__enter__()
            frozen_context_entered = True
            context_hooks.install()
            with rng_cm:
                torch.manual_seed(0)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                for batch_index, (args, kwargs) in enumerate(
                    iter_calibration_forwards(calib_ids, device)
                ):
                    base_batch = _first_tensor_batch_size(args, kwargs)
                    if target_hooks is None:
                        target_hooks = _DepthGroupTargetHooks(
                            model,
                            assignment_c,
                            specs_by_name,
                            lanes,
                            base_batch=base_batch,
                            quant_weight_cache=group_quant_cache,
                        )
                        target_hooks.install()
                    rep_args, rep_kwargs = _repeat_inputs_for_lanes(
                        args,
                        kwargs,
                        len(lanes),
                    )
                    lane_key = tuple(
                        (lane.name, lane.fmt, lane.baseline_index, lane.is_baseline)
                        for lane in lanes
                    )

                    def _full_forward(*call_args, **call_kwargs):
                        return _extract_logits(model(*call_args, **call_kwargs))

                    logits = _COORD_LANE_CUDA_GRAPH_REGISTRY.run(
                        "coord-lane-full",
                        (
                            "full",
                            id(model),
                            assignment_key,
                            cal_hash,
                            int(len(lanes)),
                            int(base_batch),
                            lane_key,
                            tuple(sorted(target_names)),
                        ),
                        _full_forward,
                        *rep_args,
                        enabled=use_coord_lane_cuda_graphs,
                        device=device,
                        keepalive=(
                            context_hooks,
                            target_hooks,
                            group_quant_cache,
                        ),
                        **rep_kwargs,
                    )
                    if logits.dim() >= 3:
                        logits = logits[:, -1:, :]
                    chunks = _split_lanes(logits.detach(), base_batch, len(lanes))
                    if chunks is None:
                        raise RuntimeError(
                            "lane-batched coord KL logits did not preserve lane "
                            f"batching: shape={tuple(logits.shape)} "
                            f"base_batch={base_batch} lanes={len(lanes)}"
                        )
                    teacher = ref_log_probs[batch_index]
                    if teacher.dim() >= 3:
                        teacher = teacher[:, -1:, :]
                    for idx, chunk in enumerate(chunks):
                        kl_totals[idx] += float(
                            kl_divergence(chunk, teacher).item()
                        )
                    batch_count += 1
            missing_targets = set(target_hooks.missing if target_hooks else [])
            if missing_targets:
                raise RuntimeError(
                    "target module missing or unsupported for lane-batched KL: "
                    + ", ".join(sorted(missing_targets))
                )
            measured.extend(
                total / max(batch_count, 1)
                for total in kl_totals
            )
        finally:
            if target_hooks is not None:
                target_hooks.remove()
            if context_hooks.installed:
                context_hooks.remove()
            if frozen_context_entered:
                frozen_context.__exit__(None, None, None)
            shutil.rmtree(cache_dir, ignore_errors=True)

    if len(measured) != len(candidate_flips):
        raise RuntimeError(
            "lane-batched coord KL produced "
            f"{len(measured)} results for {len(candidate_flips)} candidates"
        )
    return measured


@torch.no_grad()
def measure_propagated_costs(
    model: nn.Module,
    assignment: Mapping[str, str],
    neighborhood: list[L3NeighborhoodEntry],
    calibration_data,
    specs: list[fr.FormatSpec],
    *,
    work_root: str | Path | None = None,
    profile=None,
    max_lanes_per_batch: int = 16,
    tail_only: bool = True,
    cache_tail_layer_inputs: bool = True,
    output_mse_names: list[str] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, dict[str, dict]]:
    """Measure paired end-KL and downstream output-MSE for L3 candidates.

    Each non-BF16 candidate lane is paired with a target-specific BF16 lane in
    the same model call, while all non-target modules run under the converged
    L2 assignment. Depth groups are microbatched by lane count so memory stays
    bounded.
    """
    if not neighborhood:
        return {}

    specs_by_name: dict[str, fr.FormatSpec] = {}
    for spec in specs:
        specs_by_name[spec.name] = spec
        specs_by_name[fr.canonical_format_name(spec.name)] = spec
    assignment_c = _canonical_assignment(assignment)
    results: dict[str, dict[str, dict]] = {
        entry.name: {
            "BF16": {
                "propagated_end_kl": 0.0,
                "downstream_output_mse": 0.0,
                "paired_baseline": "target_bf16_under_l2_assignment",
            }
        }
        for entry in neighborhood
        if "BF16" in entry.formats
    }
    ordered_names = _ordered_quantizable_names(model, set(assignment_c))
    all_output_names = ordered_names if output_mse_names is None else output_mse_names
    device = next(model.parameters()).device
    requested_max_lanes_per_batch = max(int(max_lanes_per_batch), 1)
    max_lanes_per_batch = _adjust_l3_max_lanes_for_memory(
        requested_max_lanes_per_batch,
        calibration_data,
        device,
    )
    if (
        progress_callback is not None
        and max_lanes_per_batch != requested_max_lanes_per_batch
    ):
        progress_callback({
            "event": "lane_batch_memory_adjusted",
            "requested_max_lanes_per_batch": requested_max_lanes_per_batch,
            "max_lanes_per_batch": max_lanes_per_batch,
        })
    cal_hash = calibration_data_hash(calibration_data)
    tmp_parent = str(work_root) if work_root is not None else None

    depth_groups = _group_neighborhood_by_depth(neighborhood)
    _decoder_base, decoder_layers = _decoder_stack(model)
    tail_call_cache: dict[int, list[tuple[tuple, dict, int]]] = {}
    use_prequant_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_PREQUANT_CACHE",
        default=True,
    )
    use_frozen_perturbed_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE",
        default=True,
    )
    use_cuda_graphs = _env_flag_enabled(
        "PRISMAQUANT_L3_CUDA_GRAPHS",
        default=True,
    )
    tail_graph_cache = _TailCudaGraphCache(enabled=use_cuda_graphs)
    if (
        bool(tail_only)
        and bool(cache_tail_layer_inputs)
        and decoder_layers is not None
    ):
        needed_depths = {
            layer_depth(group_entries[0].name)
            for _group_key, group_entries in depth_groups
            if group_entries
        }
        needed_depths = {
            depth
            for depth in needed_depths
            if depth is not None and 0 <= depth < len(decoder_layers)
        }
        if needed_depths:
            cache_dir = Path(tempfile.mkdtemp(
                prefix="prismaquant_l3_baseline_context_",
                dir=tmp_parent,
            ))
            context_hooks = PerturbedActivationCache(
                model,
                assignment_c,
                cache_dir,
                input_rows=0,
                cal_hash=cal_hash,
                profile=profile,
            )
            frozen_context = (
                context_hooks.frozen_weight_cache()
                if use_frozen_perturbed_cache
                else nullcontext(context_hooks)
            )
            with frozen_context:
                context_hooks.install()
                try:
                    all_layer_calls = _capture_all_layer_calls(
                        model,
                        decoder_layers,
                        needed_depths,
                        calibration_data,
                        device,
                    )
                    tail_call_cache = {
                        depth: all_layer_calls.get(depth, [])
                        for depth in needed_depths
                    }
                finally:
                    context_hooks.remove()
                    shutil.rmtree(cache_dir, ignore_errors=True)
    for group_index, (group_key, group_entries) in enumerate(depth_groups, start=1):
        group_depth = layer_depth(group_entries[0].name) if group_entries else None
        use_tail_group = (
            bool(tail_only)
            and group_depth is not None
            and decoder_layers is not None
            and 0 <= group_depth < len(decoder_layers)
        )
        group_start = time.monotonic()
        group_lane_count = sum(
            len(_lane_specs_for_entries([entry]))
            for entry in group_entries
        )
        if progress_callback is not None:
            progress_callback({
                "event": "depth_group_start",
                "group": group_key,
                "group_index": group_index,
                "group_count": len(depth_groups),
                "entry_count": len(group_entries),
                "lane_count": group_lane_count,
                "mode": "tail-only" if use_tail_group else "full-forward",
            })
        group_quant_cache = (
            build_quant_weight_cache(model, group_entries, specs)
            if use_prequant_cache
            else None
        )
        for lanes in _lane_microbatches_for_entries(
            group_entries,
            max_lanes_per_batch,
        ):
            if not lanes:
                continue
            target_names = {lane.name for lane in lanes}
            context_assignment = {
                name: fmt
                for name, fmt in assignment_c.items()
                if name not in target_names
            }
            downstream_names = [
                name for name in _downstream_names_for_group(ordered_names, target_names)
                if name in set(all_output_names)
            ]
            tail_graph_safe = (
                use_tail_group
                and use_cuda_graphs
                and not _output_mse_names_reach_tail(downstream_names, group_depth)
            )
            execution_lanes = (
                _pad_lanes_for_cuda_graph(lanes)
                if tail_graph_safe
                else lanes
            )
            cache_dir = Path(tempfile.mkdtemp(
                prefix="prismaquant_l3_context_",
                dir=tmp_parent,
            ))
            context_hooks = PerturbedActivationCache(
                model,
                context_assignment,
                cache_dir,
                input_rows=0,
                cal_hash=cal_hash,
                profile=profile,
            )
            frozen_context = (
                context_hooks.frozen_weight_cache()
                if use_frozen_perturbed_cache
                else nullcontext(context_hooks)
            )
            frozen_context_entered = False
            target_hooks = None
            output_mse = None
            try:
                frozen_context.__enter__()
                frozen_context_entered = True
                context_hooks.install()
                first_batch = True
                kl_totals = [0.0 for _ in lanes]
                batch_count = 0
                cached_calls = (
                    tail_call_cache.get(group_depth, [])
                    if use_tail_group and cache_tail_layer_inputs
                    else None
                )
                if not cached_calls:
                    cached_calls = None
                call_iter = (
                    cached_calls
                    if cached_calls is not None
                    else iter_calibration_forwards(calibration_data, device)
                )
                try:
                    for call_item in call_iter:
                        if cached_calls is not None:
                            args, kwargs, base_batch = _move_cached_layer_call(
                                call_item,
                                device,
                            )
                        else:
                            args, kwargs = call_item
                            base_batch = _first_tensor_batch_size(args, kwargs)
                        if first_batch:
                            target_hooks = _DepthGroupTargetHooks(
                                model,
                                assignment_c,
                                specs_by_name,
                                execution_lanes,
                                base_batch=base_batch,
                                quant_weight_cache=group_quant_cache,
                            )
                            target_hooks.install()
                            output_mse = _LaneOutputMSE(
                                model,
                                downstream_names,
                                execution_lanes,
                                base_batch=base_batch,
                            )
                            output_mse.install()
                            first_batch = False

                        if use_tail_group:
                            layer = decoder_layers[group_depth]
                            if cached_calls is not None:
                                layer_args, layer_kwargs = args, kwargs
                            else:
                                layer_args, layer_kwargs = _capture_layer_call(
                                    model,
                                    layer,
                                    args,
                                    kwargs,
                                )
                            rep_args, rep_kwargs = _repeat_layer_call_for_lanes(
                                layer_args,
                                layer_kwargs,
                                len(execution_lanes),
                                base_batch,
                            )
                            layer_output = layer(*rep_args, **rep_kwargs)
                            hidden = _first_tensor_output(layer_output)
                            if hidden is None:
                                raise RuntimeError(
                                    "tail-only L3 decoder layer returned no tensor"
                                )
                            logits = tail_forward_from_layer(
                                model,
                                group_depth,
                                rep_args,
                                rep_kwargs,
                                hidden,
                                cuda_graph_cache=(
                                    tail_graph_cache if tail_graph_safe else None
                                ),
                                lane_count=len(execution_lanes),
                            )
                        else:
                            rep_args, rep_kwargs = _repeat_inputs_for_lanes(
                                args,
                                kwargs,
                                len(execution_lanes),
                            )
                            logits = _extract_logits(model(*rep_args, **rep_kwargs))

                        chunks = _split_lanes(
                            logits.detach(),
                            base_batch,
                            len(execution_lanes),
                        )
                        if chunks is None:
                            raise RuntimeError(
                                "L3 propagated-cost logits did not preserve lane "
                                f"batching: shape={tuple(logits.shape)} "
                                f"base_batch={base_batch} "
                                f"lanes={len(execution_lanes)}"
                            )
                        for idx, lane in enumerate(lanes):
                            if lane.is_baseline or lane.baseline_index is None:
                                continue
                            teacher = F.log_softmax(
                                chunks[lane.baseline_index].float(),
                                dim=-1,
                            )
                            kl_totals[idx] += float(
                                kl_divergence(chunks[idx], teacher).item()
                            )
                        batch_count += 1
                        if output_mse is not None:
                            output_mse.mark_batch()
                finally:
                    if target_hooks is not None:
                        target_hooks.remove()
                    if output_mse is not None:
                        output_mse.remove()

                missing_targets = set(target_hooks.missing if target_hooks else [])
                for idx, lane in enumerate(lanes):
                    if lane.is_baseline:
                        continue
                    per_name = results.setdefault(lane.name, {})
                    if lane.name in missing_targets:
                        per_name[lane.fmt] = {
                            "error": "target module missing or unsupported for L3"
                        }
                        continue
                    per_name[lane.fmt] = {
                        "propagated_end_kl": kl_totals[idx] / max(batch_count, 1),
                        "downstream_output_mse": (
                            output_mse.value_for_lane(idx)
                            if output_mse is not None
                            else 0.0
                        ),
                        "paired_baseline": "target_bf16_under_l2_assignment",
                    }
            finally:
                context_hooks.remove()
                if frozen_context_entered:
                    frozen_context.__exit__(None, None, None)
                shutil.rmtree(cache_dir, ignore_errors=True)
        if progress_callback is not None:
            progress_callback({
                "event": "depth_group_end",
                "group": group_key,
                "group_index": group_index,
                "group_count": len(depth_groups),
                "entry_count": len(group_entries),
                "lane_count": group_lane_count,
                "elapsed_seconds": time.monotonic() - group_start,
            })

    tail_graph_cache.clear()
    return results
