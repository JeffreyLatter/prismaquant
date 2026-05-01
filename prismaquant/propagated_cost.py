"""L3 propagated-cost utilities.

This module owns the final-pass "L3 polish" path: select a small allocator
neighborhood around the converged L2 assignment, measure propagated costs for
that neighborhood, and re-solve only those measured choices while freezing the
rest of the L2 assignment.
"""
from __future__ import annotations

import math
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
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


def _is_l3_unsupported_target(stats_entry: Mapping) -> bool:
    """Return True for probe entries whose live module is not L3-hookable."""
    return _stats_indicates_packed_expert(dict(stats_entry))


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


def _candidate_total_bits(candidate: Candidate) -> float:
    return 8.0 * float(candidate.memory_bytes)


def _greedy_l3_under_budget(
    open_cands: Mapping[str, list[Candidate]],
    current_assignment: Mapping[str, str],
    remaining_bits: float,
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
    attempts = []
    for name in names:
        current = chosen[name]
        for cand in open_cands[name]:
            if cand.fmt == current.fmt:
                continue
            improvement = current.predicted_dloss - cand.predicted_dloss
            attempts.append((improvement, name, cand))
    attempts.sort(
        key=lambda item: (
            -item[0],
            _candidate_total_bits(item[2]),
            item[1],
            item[2].fmt,
        )
    )

    stats = {
        "attempts": 0,
        "accepted": 0,
        "rejected_not_better": 0,
        "rejected_budget": 0,
        "start_bits": used_bits,
        "end_bits": None,
        "remaining_bits": float(remaining_bits),
    }
    swapped_names: set[str] = set()
    for improvement, name, cand in attempts:
        if name in swapped_names:
            continue
        stats["attempts"] += 1
        if improvement <= 1e-12:
            stats["rejected_not_better"] += 1
            continue
        current = chosen[name]
        current_bits = _candidate_total_bits(current)
        cand_bits = _candidate_total_bits(cand)
        next_bits = used_bits - current_bits + cand_bits
        if used_bits > remaining_bits + 1e-6:
            if cand_bits >= current_bits - 1e-6:
                stats["rejected_budget"] += 1
                continue
        elif next_bits > remaining_bits + 1e-6:
            stats["rejected_budget"] += 1
            continue
        chosen[name] = cand
        used_bits = next_bits
        swapped_names.add(name)
        stats["accepted"] += 1

    stats["end_bits"] = used_bits
    assignment = {name: chosen[name].fmt for name in names}
    return assignment, chosen, stats


def _minimum_bpp_result_if_within_precision(
    open_stats: Mapping[str, Mapping],
    open_cands: Mapping[str, list[Candidate]],
    open_target_bits: float,
    bit_precision: float,
) -> tuple[dict[str, str], dict[str, Candidate]] | None:
    total_params = sum(int(open_stats[name]["n_params"]) for name in open_cands)
    if total_params <= 0:
        return {}, {}
    chosen = {
        name: min(open_cands[name], key=lambda c: (c.bits_per_param, c.fmt))
        for name in open_cands
    }
    min_bpp = sum(
        chosen[name].bits_per_param * int(open_stats[name]["n_params"])
        for name in open_cands
    ) / float(total_params)
    if open_target_bits < min_bpp and min_bpp - open_target_bits <= bit_precision + 1e-12:
        return (
            {name: cand.fmt for name, cand in chosen.items()},
            dict(chosen),
        )
    return None


def solve_frozen_l3_neighborhood(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    l3_candidates: Mapping[str, list[Candidate]],
    specs: list[fr.FormatSpec],
    *,
    target_bits: float,
    bit_precision: float,
    return_metadata: bool = False,
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
        result = (dict(assignment), {})
        if return_metadata:
            return (*result, {"frozen_dp_precision_used": "none"})
        return result

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
            min_result = _minimum_bpp_result_if_within_precision(
                open_stats,
                open_cands,
                open_target_bits,
                fallback_precision,
            )
            if min_result is not None:
                result = min_result
                precision_used = fallback_precision
                dp_attempts[-1]["result"] = "minimum_bpp_within_precision"
                print(
                    "[l3] frozen DP precision "
                    f"{fallback_precision:g}: minimum-bpp assignment within "
                    "precision",
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
        )
        result = (open_assignment, chosen)
        precision_used = "greedy"
        print(
            "[l3] frozen DP greedy: "
            f"attempts={greedy_stats['attempts']} "
            f"accepted={greedy_stats['accepted']} "
            f"rejected_not_better={greedy_stats['rejected_not_better']} "
            f"rejected_budget={greedy_stats['rejected_budget']}",
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


def _split_lanes(tensor: torch.Tensor, base_batch: int, lane_count: int):
    if tensor.dim() == 0 or tensor.size(0) != base_batch * lane_count:
        return None
    return tensor.split(base_batch, dim=0)


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
    ):
        self.model = model
        self.assignment = _canonical_assignment(assignment)
        self.specs_by_name = specs_by_name
        self.lanes = lanes
        self.base_batch = int(base_batch)
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
                w_hat = spec.quantize_dequantize(weight.clone())
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
) -> list[_LaneSpec]:
    lanes: list[_LaneSpec] = []
    for entry in entries:
        candidate_fmts = [fmt for fmt in entry.formats if fmt != "BF16"]
        if not candidate_fmts:
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
) -> list[list[_LaneSpec]]:
    batches: list[list[_LaneSpec]] = []
    current: list[_LaneSpec] = []
    max_lanes = max(int(max_lanes_per_batch), 1)
    for entry in entries:
        entry_lanes = _lane_specs_for_entries([entry])
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
    max_lanes_per_batch: int = 8,
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

    specs_by_name = {s.name: s for s in specs}
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
    all_output_names = output_mse_names or ordered_names
    device = next(model.parameters()).device
    cal_hash = calibration_data_hash(calibration_data)
    tmp_parent = str(work_root) if work_root is not None else None

    depth_groups = _group_neighborhood_by_depth(neighborhood)
    for group_index, (group_key, group_entries) in enumerate(depth_groups, start=1):
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
            })
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
            context_hooks.install()
            target_hooks = None
            output_mse = None
            try:
                first_batch = True
                kl_totals = [0.0 for _ in lanes]
                batch_count = 0
                try:
                    for args, kwargs in iter_calibration_forwards(calibration_data, device):
                        base_batch = _first_tensor_batch_size(args, kwargs)
                        if first_batch:
                            target_hooks = _DepthGroupTargetHooks(
                                model,
                                assignment_c,
                                specs_by_name,
                                lanes,
                                base_batch=base_batch,
                            )
                            target_hooks.install()
                            output_mse = _LaneOutputMSE(
                                model,
                                downstream_names,
                                lanes,
                                base_batch=base_batch,
                            )
                            output_mse.install()
                            first_batch = False
                        rep_args, rep_kwargs = _repeat_inputs_for_lanes(
                            args,
                            kwargs,
                            len(lanes),
                        )
                        logits = _extract_logits(model(*rep_args, **rep_kwargs))
                        chunks = _split_lanes(logits.detach(), base_batch, len(lanes))
                        if chunks is None:
                            raise RuntimeError(
                                "L3 propagated-cost logits did not preserve lane "
                                f"batching: shape={tuple(logits.shape)} "
                                f"base_batch={base_batch} lanes={len(lanes)}"
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

    return results
