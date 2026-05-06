"""Adjoint-sketch L3 allocator utilities.

The measured L3 path directly perturbs one candidate at a time and observes the
final KL.  This module supports a cheaper surrogate: each decision unit stores
a low-rank adjoint sketch for every format, plus an optional diagonal floor.
The resulting objective is PSD by construction:

    0.5 / R * ||sum_i a[i, fmt_i]||^2 + sum_i diagonal[i, fmt_i]

where ``R`` is the sketch rank and ``a`` is the adjoint projection of the local
quantization error through downstream Jacobian products.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import format_registry as fr
from .allocator_solver import Candidate, _shape_from_stats


SCHEMA = "prismaquant.adjoint_l3.v1"


@dataclass(frozen=True)
class AdjointFormatOption:
    """One format choice for one adjoint-sketch decision unit."""

    name: str
    fmt: str
    sketch: tuple[float, ...]
    diagonal_cost: float
    bits_per_param: float
    memory_bytes: int

    @property
    def bits_total(self) -> float:
        return float(self.memory_bytes) * 8.0


@dataclass(frozen=True)
class AdjointUnit:
    """A fused or unfused allocator decision unit."""

    name: str
    options: tuple[AdjointFormatOption, ...]


@dataclass(frozen=True)
class AdjointSolveResult:
    assignment: dict[str, str]
    objective: float
    diagonal_cost: float
    low_rank_cost: float
    lagrangian_objective: float
    bits_total: float
    passes: int
    moves: int
    rank: int
    changed_units: int | None = None


def load_adjoint_l3_payload(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    validate_adjoint_l3_payload(payload)
    return payload


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def retune_adjoint_diagonal_costs(
    payload: Mapping,
    *,
    diagonal_floor_frac: float | None = None,
    mse_diagonal_floor_frac: float | None = None,
) -> dict:
    """Return a payload copy with solve-time diagonal floors recomputed.

    Newer measured artifacts store enough components to vary the diagonal
    regularizer without remeasuring sketches:

        diagonal = d * adjoint_self_cost + m * mse_floor_scale * output_delta_mse

    This makes the empirically important MSE floor a cheap solve-time sweep
    knob instead of a collector-time one-off.
    """
    validate_adjoint_l3_payload(payload)
    if diagonal_floor_frac is None and mse_diagonal_floor_frac is None:
        return dict(payload)

    out = copy.deepcopy(dict(payload))
    meta = out.setdefault("meta", {})
    if diagonal_floor_frac is None:
        diagonal_floor_frac = float(meta.get("diagonal_floor_frac", 1.0))
    if mse_diagonal_floor_frac is None:
        mse_diagonal_floor_frac = float(meta.get("mse_diagonal_floor_frac", 0.0))
    diagonal_floor_frac = max(float(diagonal_floor_frac), 0.0)
    mse_diagonal_floor_frac = max(float(mse_diagonal_floor_frac), 0.0)

    mse_floor_scale = meta.get("mse_floor_scale")
    if mse_floor_scale is None and mse_diagonal_floor_frac > 0.0:
        ratios = []
        for unit in out["units"].values():
            for entry in unit["formats"].values():
                self_cost = entry.get("adjoint_self_cost")
                mse = entry.get("output_delta_mse")
                if self_cost is not None and mse is not None:
                    self_cost = float(self_cost)
                    mse = float(mse)
                    if self_cost > 0.0 and mse > 0.0:
                        ratios.append(self_cost / mse)
        if not ratios:
            raise ValueError(
                "cannot apply --mse-diagonal-floor-frac because artifact lacks "
                "mse_floor_scale and positive output_delta_mse entries"
            )
        mse_floor_scale = _median(ratios)
    mse_floor_scale = float(mse_floor_scale or 0.0)

    rank = int(out["rank"])
    for unit_name, unit in out["units"].items():
        for fmt, entry in unit["formats"].items():
            canonical = fr.canonical_format_name(fmt)
            sketch = entry.get("sketch", [])
            self_cost = entry.get("adjoint_self_cost")
            if self_cost is None:
                self_cost = 0.5 / float(rank) * sum(float(v) * float(v) for v in sketch)
            output_mse = entry.get("output_delta_mse")
            if output_mse is None:
                if mse_diagonal_floor_frac > 0.0 and canonical != "BF16":
                    raise ValueError(
                        f"adjoint L3 entry {unit_name!r}/{fmt!r} lacks "
                        "output_delta_mse for MSE floor retuning"
                    )
                output_mse = 0.0
            if canonical == "BF16":
                diagonal_cost = 0.0
                mse_floor_cost = 0.0
            else:
                mse_floor_cost = (
                    mse_diagonal_floor_frac
                    * mse_floor_scale
                    * float(output_mse)
                )
                diagonal_cost = diagonal_floor_frac * float(self_cost) + mse_floor_cost
            entry["diagonal_cost"] = float(diagonal_cost)
            entry["mse_floor_cost"] = float(mse_floor_cost)
            entry["solve_diagonal_floor_frac"] = float(diagonal_floor_frac)
            entry["solve_mse_diagonal_floor_frac"] = float(mse_diagonal_floor_frac)

    meta["solve_diagonal_floor_frac"] = float(diagonal_floor_frac)
    meta["solve_mse_diagonal_floor_frac"] = float(mse_diagonal_floor_frac)
    meta["solve_mse_floor_scale"] = float(mse_floor_scale)
    return out


def validate_adjoint_l3_payload(payload: Mapping) -> None:
    """Validate the portable JSON artifact shape.

    Expected payload shape:

    {
      "schema": "prismaquant.adjoint_l3.v1",
      "rank": R,
      "units": {
        "module.name": {
          "formats": {
            "NVFP4": {"sketch": [...], "diagonal_cost": 0.0},
            "BF16": {"sketch": [...], "diagonal_cost": 0.0}
          }
        }
      }
    }
    """
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported adjoint L3 schema: {payload.get('schema')!r}")
    rank = int(payload.get("rank", 0) or 0)
    if rank <= 0:
        raise ValueError("adjoint L3 payload requires positive rank")
    units = payload.get("units")
    if not isinstance(units, Mapping) or not units:
        raise ValueError("adjoint L3 payload requires non-empty units")

    for name, unit in units.items():
        if not isinstance(name, str) or not name:
            raise ValueError("adjoint L3 unit names must be non-empty strings")
        formats = unit.get("formats") if isinstance(unit, Mapping) else None
        if not isinstance(formats, Mapping) or not formats:
            raise ValueError(f"adjoint L3 unit {name!r} has no formats")
        for fmt, entry in formats.items():
            if not isinstance(fmt, str) or not fmt:
                raise ValueError(f"adjoint L3 unit {name!r} has invalid format key")
            if not isinstance(entry, Mapping):
                raise ValueError(f"adjoint L3 entry {name!r}/{fmt!r} is not a map")
            sketch = entry.get("sketch")
            if not isinstance(sketch, Sequence) or isinstance(sketch, (str, bytes)):
                raise ValueError(f"adjoint L3 entry {name!r}/{fmt!r} has no sketch")
            if len(sketch) != rank:
                raise ValueError(
                    f"adjoint L3 entry {name!r}/{fmt!r} sketch rank "
                    f"{len(sketch)} != payload rank {rank}"
                )
            for value in sketch:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(
                        f"adjoint L3 entry {name!r}/{fmt!r} has non-finite sketch"
                    )
            diagonal_cost = float(entry.get("diagonal_cost", 0.0))
            if not math.isfinite(diagonal_cost) or diagonal_cost < 0.0:
                raise ValueError(
                    f"adjoint L3 entry {name!r}/{fmt!r} has invalid diagonal_cost"
                )


def _stats_shape_and_bits(
    stats_entry: Mapping,
    spec: fr.FormatSpec,
) -> tuple[float, int]:
    shape = _shape_from_stats(dict(stats_entry))
    memory_bytes = spec.memory_bytes_for_shape(shape)
    bits_per_param = spec.effective_bits_for_shape(shape)
    return bits_per_param, memory_bytes


def _option_from_entry(
    name: str,
    fmt: str,
    entry: Mapping,
    stats: Mapping[str, Mapping] | None,
    specs_by_name: Mapping[str, fr.FormatSpec] | None,
    *,
    require_memory: bool,
) -> AdjointFormatOption:
    canonical = fr.canonical_format_name(fmt)
    sketch = tuple(float(v) for v in entry["sketch"])
    diagonal_cost = max(float(entry.get("diagonal_cost", 0.0)), 0.0)

    bits_per_param = entry.get("bits_per_param")
    memory_bytes = entry.get("memory_bytes")
    if bits_per_param is not None and memory_bytes is not None:
        bits_per_param = float(bits_per_param)
        memory_bytes = int(memory_bytes)
    elif not require_memory:
        bits_per_param = 0.0
        memory_bytes = 0
    else:
        if stats is None or specs_by_name is None:
            raise ValueError(
                f"adjoint L3 entry {name!r}/{fmt!r} lacks memory fields and "
                "stats/specs were not supplied"
            )
        if name not in stats:
            raise KeyError(f"adjoint L3 unit {name!r} is missing from stats")
        if canonical not in specs_by_name:
            raise KeyError(f"format {canonical!r} is not available for adjoint L3")
        bits_per_param, memory_bytes = _stats_shape_and_bits(
            stats[name],
            specs_by_name[canonical],
        )

    return AdjointFormatOption(
        name=name,
        fmt=canonical,
        sketch=sketch,
        diagonal_cost=diagonal_cost,
        bits_per_param=float(bits_per_param),
        memory_bytes=int(memory_bytes),
    )


def adjoint_units_from_payload(
    payload: Mapping,
    *,
    stats: Mapping[str, Mapping] | None = None,
    formats: Sequence[fr.FormatSpec] | None = None,
    require_memory: bool = True,
) -> tuple[AdjointUnit, ...]:
    validate_adjoint_l3_payload(payload)
    specs_by_name = None
    if formats is not None:
        specs_by_name = {fr.canonical_format_name(spec.name): spec for spec in formats}

    units: list[AdjointUnit] = []
    for name, unit_payload in payload["units"].items():
        options = []
        for fmt, entry in unit_payload["formats"].items():
            options.append(_option_from_entry(
                name,
                fmt,
                entry,
                stats,
                specs_by_name,
                require_memory=require_memory,
            ))
        options.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
        units.append(AdjointUnit(name=name, options=tuple(options)))
    units.sort(key=lambda unit: unit.name)
    return tuple(units)


def option_unary_cost(
    option: AdjointFormatOption,
    rank: int,
    *,
    low_rank_weight: float = 1.0,
    diagonal_weight: float = 1.0,
) -> float:
    """Return the additive cost implied by one option alone."""
    low_rank = 0.5 / float(rank) * sum(v * v for v in option.sketch)
    return (
        float(diagonal_weight) * option.diagonal_cost
        + float(low_rank_weight) * low_rank
    )


def build_adjoint_l3_candidates(
    stats: Mapping[str, Mapping],
    payload: Mapping,
    formats: Sequence[fr.FormatSpec],
    *,
    low_rank_weight: float = 1.0,
    diagonal_weight: float = 1.0,
) -> dict[str, list[Candidate]]:
    """Build additive Candidate costs from an adjoint-sketch artifact.

    This is the compatibility mode for existing DP allocators.  It ignores
    cross terms and uses only each format's unary ``0.5/R * ||a||^2 + diagonal``
    cost.  Use ``solve_low_rank_lagrangian`` when the low-rank cross terms
    should influence the assignment directly.
    """
    rank = int(payload["rank"])
    units = adjoint_units_from_payload(payload, stats=stats, formats=formats)
    out: dict[str, list[Candidate]] = {}
    for unit in units:
        cands = []
        for option in unit.options:
            cands.append(Candidate(
                fmt=option.fmt,
                bits_per_param=option.bits_per_param,
                memory_bytes=option.memory_bytes,
                predicted_dloss=max(
                    option_unary_cost(
                        option,
                        rank,
                        low_rank_weight=low_rank_weight,
                        diagonal_weight=diagonal_weight,
                    ),
                    0.0,
                ),
            ))
        out[unit.name] = cands
    return out


def adjoint_payload_to_propagated_costs(
    payload: Mapping,
    *,
    low_rank_weight: float = 1.0,
    diagonal_weight: float = 1.0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Convert adjoint unary costs into the legacy L3 cost dictionary shape."""
    rank = int(payload["rank"])
    units = adjoint_units_from_payload(payload, require_memory=False)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for unit in units:
        out[unit.name] = {}
        for option in unit.options:
            unary = option_unary_cost(
                option,
                rank,
                low_rank_weight=low_rank_weight,
                diagonal_weight=diagonal_weight,
            )
            out[unit.name][option.fmt] = {
                "propagated_end_kl": max(unary, 0.0),
                "adjoint_l3_unary_cost": max(unary, 0.0),
                "adjoint_l3_diagonal_cost": option.diagonal_cost,
                "adjoint_l3_low_rank_self_cost": (
                    0.5 / float(rank) * sum(v * v for v in option.sketch)
                ),
            }
    return out


def adjoint_payload_to_l3_resume_payload(
    payload: Mapping,
    *,
    formats: Sequence[str] | None = None,
    meta: Mapping | None = None,
    low_rank_weight: float = 1.0,
    diagonal_weight: float = 1.0,
) -> dict:
    """Build a pickle-compatible payload for ``--resume-l3-costs``."""
    costs = adjoint_payload_to_propagated_costs(
        payload,
        low_rank_weight=low_rank_weight,
        diagonal_weight=diagonal_weight,
    )
    out_meta = {
        "source_schema": SCHEMA,
        "rank": int(payload["rank"]),
        "normalization": payload.get("normalization", "0.5/rank"),
    }
    if meta:
        out_meta.update(dict(meta))
    return {
        "costs": costs,
        "cost_history": [costs],
        "formats": list(formats or []),
        "meta": out_meta,
    }


def _options_by_format(
    units: Sequence[AdjointUnit],
) -> dict[str, dict[str, AdjointFormatOption]]:
    return {
        unit.name: {option.fmt: option for option in unit.options}
        for unit in units
    }


def group_adjoint_units_by_profile(
    units: Sequence[AdjointUnit],
    profile,
) -> tuple[tuple[AdjointUnit, ...], dict[str, tuple[str, ...]]]:
    """Aggregate profile-fused siblings into single decision units.

    The runtime/export path requires all members of a fused group to share one
    serving format.  Solving the low-rank objective on raw linears can therefore
    spend an impossible budget.  This helper preserves the same PSD objective by
    summing member sketches and diagonal terms for each shared-format option.
    """
    grouped: dict[str, list[AdjointUnit]] = defaultdict(list)
    for unit in units:
        group_key = None
        if profile is not None:
            try:
                group_key = profile.fused_sibling_group(unit.name)
            except Exception:
                group_key = None
        grouped[str(group_key or unit.name)].append(unit)

    out: list[AdjointUnit] = []
    members_by_group: dict[str, tuple[str, ...]] = {}
    for group_name, members in sorted(grouped.items()):
        members = sorted(members, key=lambda unit: unit.name)
        members_by_group[group_name] = tuple(unit.name for unit in members)
        option_maps = [_options_by_format((unit,))[unit.name] for unit in members]
        common_formats = set(option_maps[0])
        for option_map in option_maps[1:]:
            common_formats.intersection_update(option_map)
        if not common_formats:
            raise ValueError(f"fused group {group_name!r} has no common formats")

        group_options: list[AdjointFormatOption] = []
        for fmt in sorted(common_formats, key=lambda value: (fr.get_format(value).effective_bits, value)):
            child_options = [option_map[fmt] for option_map in option_maps]
            rank = len(child_options[0].sketch)
            sketch = tuple(
                sum(option.sketch[idx] for option in child_options)
                for idx in range(rank)
            )
            diagonal_cost = sum(option.diagonal_cost for option in child_options)
            memory_bytes = sum(option.memory_bytes for option in child_options)
            # Best-effort aggregate bpp for reporting/candidate compatibility.
            inferred_params = 0.0
            for option in child_options:
                if option.bits_per_param > 0.0:
                    inferred_params += option.bits_total / option.bits_per_param
            bits_per_param = (
                (float(memory_bytes) * 8.0 / inferred_params)
                if inferred_params > 0.0 else 0.0
            )
            group_options.append(AdjointFormatOption(
                name=group_name,
                fmt=fmt,
                sketch=sketch,
                diagonal_cost=float(diagonal_cost),
                bits_per_param=float(bits_per_param),
                memory_bytes=int(memory_bytes),
            ))
        out.append(AdjointUnit(name=group_name, options=tuple(group_options)))
    return tuple(out), members_by_group


def expand_grouped_assignment(
    assignment: Mapping[str, str],
    group_members: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Broadcast a grouped assignment back to raw unit names."""
    out: dict[str, str] = {}
    for name, fmt in assignment.items():
        canonical = fr.canonical_format_name(fmt)
        members = group_members.get(name)
        if members:
            for member in members:
                out[str(member)] = canonical
        else:
            out[str(name)] = canonical
    return out


def _initial_assignment(
    units: Sequence[AdjointUnit],
    rank: int,
    lambda_penalty: float,
    initial_assignment: Mapping[str, str] | None,
    *,
    low_rank_weight: float,
    diagonal_weight: float,
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for unit in units:
        options = {option.fmt: option for option in unit.options}
        if initial_assignment is not None and unit.name in initial_assignment:
            fmt = fr.canonical_format_name(initial_assignment[unit.name])
            if fmt in options:
                assignment[unit.name] = fmt
                continue
        best = min(
            unit.options,
            key=lambda option: (
                option_unary_cost(
                    option,
                    rank,
                    low_rank_weight=low_rank_weight,
                    diagonal_weight=diagonal_weight,
                )
                + float(lambda_penalty) * option.bits_total,
                option.bits_total,
                option.fmt,
            ),
        )
        assignment[unit.name] = best.fmt
    return assignment


def _normalise_reference_assignment(
    units: Sequence[AdjointUnit],
    reference_assignment: Mapping[str, str] | None,
) -> dict[str, str]:
    if reference_assignment is None:
        return {}
    unit_names = {unit.name for unit in units}
    return {
        str(name): fr.canonical_format_name(str(fmt))
        for name, fmt in reference_assignment.items()
        if str(name) in unit_names
    }


def _assignment_change_count(
    units: Sequence[AdjointUnit],
    assignment: Mapping[str, str],
    reference_assignment: Mapping[str, str] | None,
) -> int:
    reference = _normalise_reference_assignment(units, reference_assignment)
    if not reference:
        return 0
    changed = 0
    for unit in units:
        ref_fmt = reference.get(unit.name)
        if ref_fmt is None:
            continue
        fmt = fr.canonical_format_name(assignment[unit.name])
        if fmt != ref_fmt:
            changed += 1
    return changed


def _trial_change_count(
    changed_count: int,
    reference_assignment: Mapping[str, str],
    current_by_name: Mapping[str, str],
    moves: Sequence[tuple[str, str]],
) -> int:
    out = int(changed_count)
    for name, next_fmt in moves:
        ref_fmt = reference_assignment.get(name)
        if ref_fmt is None:
            continue
        current_changed = current_by_name[name] != ref_fmt
        next_changed = fr.canonical_format_name(next_fmt) != ref_fmt
        out += int(next_changed) - int(current_changed)
    return out


def _reference_options_by_name(
    units: Sequence[AdjointUnit],
    reference_assignment: Mapping[str, str] | None,
) -> dict[str, AdjointFormatOption]:
    if not reference_assignment:
        return {}
    by_format = _options_by_format(units)
    out: dict[str, AdjointFormatOption] = {}
    for unit in units:
        ref_fmt = reference_assignment.get(unit.name)
        if ref_fmt is None:
            continue
        ref_fmt = fr.canonical_format_name(ref_fmt)
        option = by_format[unit.name].get(ref_fmt)
        if option is not None:
            out[unit.name] = option
    return out


def _is_reference_downgrade(
    reference_options: Mapping[str, AdjointFormatOption],
    unit_name: str,
    option: AdjointFormatOption,
) -> bool:
    reference = reference_options.get(unit_name)
    if reference is None:
        return False
    return option.bits_total < reference.bits_total - 1e-6


def _coerce_reference_downgrades(
    assignment: dict[str, str],
    reference_options: Mapping[str, AdjointFormatOption],
    units: Sequence[AdjointUnit],
) -> dict[str, str]:
    by_format = _options_by_format(units)
    out = dict(assignment)
    for unit in units:
        reference = reference_options.get(unit.name)
        if reference is None:
            continue
        current = by_format[unit.name].get(out[unit.name])
        if current is not None and current.bits_total < reference.bits_total - 1e-6:
            out[unit.name] = reference.fmt
    return out


def score_adjoint_assignment(
    units: Sequence[AdjointUnit],
    rank: int,
    assignment: Mapping[str, str],
    *,
    low_rank_weight: float = 1.0,
    diagonal_weight: float = 1.0,
) -> tuple[float, float, float, float, tuple[float, ...]]:
    """Return objective, diagonal, low-rank, bits, and sketch sum."""
    by_format = _options_by_format(units)
    sketch_sum = [0.0 for _ in range(rank)]
    diagonal = 0.0
    bits_total = 0.0
    for unit in units:
        fmt = fr.canonical_format_name(assignment[unit.name])
        option = by_format[unit.name][fmt]
        diagonal += option.diagonal_cost
        bits_total += option.bits_total
        for idx, value in enumerate(option.sketch):
            sketch_sum[idx] += value
    low_rank = 0.5 / float(rank) * sum(v * v for v in sketch_sum)
    objective = float(diagonal_weight) * diagonal + float(low_rank_weight) * low_rank
    return objective, diagonal, low_rank, bits_total, tuple(sketch_sum)


def solve_low_rank_lagrangian(
    units: Sequence[AdjointUnit],
    rank: int,
    *,
    lambda_penalty: float = 0.0,
    initial_assignment: Mapping[str, str] | None = None,
    reference_assignment: Mapping[str, str] | None = None,
    max_changed_units: int | None = None,
    change_penalty: float = 0.0,
    forbid_reference_downgrades: bool = False,
    low_rank_weight: float = 1.0,
    diagonal_weight: float = 1.0,
    max_passes: int = 16,
    improvement_tol: float = 1e-12,
    pairwise: bool = True,
) -> AdjointSolveResult:
    """Coordinate-descent solve for the PSD low-rank adjoint objective.

    ``lambda_penalty`` is charged per total bit.  Sweeping this value gives the
    usual budget frontier; positive values prefer smaller artifacts.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    if not units:
        raise ValueError("at least one adjoint unit is required")
    reference = _normalise_reference_assignment(units, reference_assignment)
    if max_changed_units is not None:
        if int(max_changed_units) < 0:
            raise ValueError("max_changed_units must be non-negative")
        if not reference:
            raise ValueError("max_changed_units requires a reference_assignment")
        max_changed_units = int(max_changed_units)
    change_penalty = max(float(change_penalty), 0.0)
    reference_options = _reference_options_by_name(units, reference)
    if forbid_reference_downgrades and not reference_options:
        raise ValueError("forbid_reference_downgrades requires a reference_assignment")
    if initial_assignment is None and reference and (
        max_changed_units is not None
        or change_penalty > 0.0
        or forbid_reference_downgrades
    ):
        initial_assignment = reference
    assignment = _initial_assignment(
        units,
        rank,
        lambda_penalty,
        initial_assignment,
        low_rank_weight=low_rank_weight,
        diagonal_weight=diagonal_weight,
    )
    if forbid_reference_downgrades:
        assignment = _coerce_reference_downgrades(
            assignment,
            reference_options,
            units,
        )
    (
        objective,
        diagonal,
        low_rank,
        bits_total,
        sketch_sum,
    ) = score_adjoint_assignment(
        units,
        rank,
        assignment,
        low_rank_weight=low_rank_weight,
        diagonal_weight=diagonal_weight,
    )
    changed_count = _assignment_change_count(units, assignment, reference)
    if max_changed_units is not None and changed_count > max_changed_units:
        raise ValueError(
            "initial assignment exceeds max_changed_units "
            f"({changed_count} > {max_changed_units})"
        )
    current_lagrangian = (
        objective
        + float(lambda_penalty) * bits_total
        + change_penalty * float(changed_count)
    )
    sketch_sum_list = list(sketch_sum)
    by_format = _options_by_format(units)
    moves = 0
    passes_done = 0

    max_iterations = max(int(max_passes), 0) * max(len(units), 1)
    for iteration in range(max_iterations):
        passes_done = iteration // max(len(units), 1) + 1
        best_move = None
        best_values = (
            current_lagrangian,
            objective,
            diagonal,
            low_rank,
            bits_total,
            tuple(sketch_sum_list),
            changed_count,
        )
        for unit in units:
            current = by_format[unit.name][assignment[unit.name]]
            current_assignment = {unit.name: assignment[unit.name]}
            for option in unit.options:
                if option.fmt == current.fmt:
                    continue
                if (
                    forbid_reference_downgrades
                    and _is_reference_downgrade(reference_options, unit.name, option)
                ):
                    continue
                trial_changes = _trial_change_count(
                    changed_count,
                    reference,
                    current_assignment,
                    ((unit.name, option.fmt),),
                )
                if (
                    max_changed_units is not None
                    and trial_changes > max_changed_units
                ):
                    continue
                trial_sketch = [
                    sketch_sum_list[idx] - current.sketch[idx] + option.sketch[idx]
                    for idx in range(rank)
                ]
                trial_diagonal = diagonal - current.diagonal_cost + option.diagonal_cost
                trial_bits = bits_total - current.bits_total + option.bits_total
                trial_low_rank = 0.5 / float(rank) * sum(v * v for v in trial_sketch)
                trial_objective = (
                    float(diagonal_weight) * trial_diagonal
                    + float(low_rank_weight) * trial_low_rank
                )
                trial_lagrangian = (
                    trial_objective
                    + float(lambda_penalty) * trial_bits
                    + change_penalty * float(trial_changes)
                )
                if trial_lagrangian < best_values[0] - float(improvement_tol):
                    best_move = (unit.name, option.fmt)
                    best_values = (
                        trial_lagrangian,
                        trial_objective,
                        trial_diagonal,
                        trial_low_rank,
                        trial_bits,
                        tuple(trial_sketch),
                        trial_changes,
                    )
        if best_move is None and pairwise:
            unit_count = len(units)
            for left_idx in range(unit_count):
                left = units[left_idx]
                left_current = by_format[left.name][assignment[left.name]]
                for right_idx in range(left_idx + 1, unit_count):
                    right = units[right_idx]
                    right_current = by_format[right.name][assignment[right.name]]
                    current_assignment = {
                        left.name: assignment[left.name],
                        right.name: assignment[right.name],
                    }
                    for left_option in left.options:
                        if left_option.fmt == left_current.fmt:
                            continue
                        if (
                            forbid_reference_downgrades
                            and _is_reference_downgrade(
                                reference_options,
                                left.name,
                                left_option,
                            )
                        ):
                            continue
                        for right_option in right.options:
                            if right_option.fmt == right_current.fmt:
                                continue
                            if (
                                forbid_reference_downgrades
                                and _is_reference_downgrade(
                                    reference_options,
                                    right.name,
                                    right_option,
                                )
                            ):
                                continue
                            trial_changes = _trial_change_count(
                                changed_count,
                                reference,
                                current_assignment,
                                (
                                    (left.name, left_option.fmt),
                                    (right.name, right_option.fmt),
                                ),
                            )
                            if (
                                max_changed_units is not None
                                and trial_changes > max_changed_units
                            ):
                                continue
                            trial_sketch = [
                                sketch_sum_list[idx]
                                - left_current.sketch[idx]
                                - right_current.sketch[idx]
                                + left_option.sketch[idx]
                                + right_option.sketch[idx]
                                for idx in range(rank)
                            ]
                            trial_diagonal = (
                                diagonal
                                - left_current.diagonal_cost
                                - right_current.diagonal_cost
                                + left_option.diagonal_cost
                                + right_option.diagonal_cost
                            )
                            trial_bits = (
                                bits_total
                                - left_current.bits_total
                                - right_current.bits_total
                                + left_option.bits_total
                                + right_option.bits_total
                            )
                            trial_low_rank = (
                                0.5 / float(rank) * sum(v * v for v in trial_sketch)
                            )
                            trial_objective = (
                                float(diagonal_weight) * trial_diagonal
                                + float(low_rank_weight) * trial_low_rank
                            )
                            trial_lagrangian = (
                                trial_objective
                                + float(lambda_penalty) * trial_bits
                                + change_penalty * float(trial_changes)
                            )
                            if (
                                trial_lagrangian
                                < best_values[0] - float(improvement_tol)
                            ):
                                best_move = (
                                    (left.name, left_option.fmt),
                                    (right.name, right_option.fmt),
                                )
                                best_values = (
                                    trial_lagrangian,
                                    trial_objective,
                                    trial_diagonal,
                                    trial_low_rank,
                                    trial_bits,
                                    tuple(trial_sketch),
                                    trial_changes,
                                )
        if best_move is None:
            break
        if isinstance(best_move[0], tuple):
            for name, fmt in best_move:
                assignment[name] = fmt
        else:
            name, fmt = best_move
            assignment[name] = fmt
        (
            current_lagrangian,
            objective,
            diagonal,
            low_rank,
            bits_total,
            sketch_sum,
            changed_count,
        ) = best_values
        sketch_sum_list = list(sketch_sum)
        moves += 1

    return AdjointSolveResult(
        assignment=dict(assignment),
        objective=float(objective),
        diagonal_cost=float(diagonal),
        low_rank_cost=float(low_rank),
        lagrangian_objective=float(current_lagrangian),
        bits_total=float(bits_total),
        passes=passes_done,
        moves=moves,
        rank=rank,
        changed_units=changed_count if reference else None,
    )


def polish_low_rank_to_budget(
    units: Sequence[AdjointUnit],
    rank: int,
    result: AdjointSolveResult,
    *,
    target_bits_total: float,
    reference_assignment: Mapping[str, str] | None = None,
    max_changed_units: int | None = None,
    forbid_reference_downgrades: bool = False,
    low_rank_weight: float = 1.0,
    diagonal_weight: float = 1.0,
    max_moves: int | None = None,
    improvement_tol: float = 1e-12,
) -> AdjointSolveResult:
    """Repair and polish a low-rank solution against a hard bit budget.

    If the seed is over budget, take the least damaging bit-reducing move until
    it is feasible.  Once feasible, spend any remaining budget only on moves
    that reduce the PSD objective.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    reference = _normalise_reference_assignment(units, reference_assignment)
    if max_changed_units is not None:
        if int(max_changed_units) < 0:
            raise ValueError("max_changed_units must be non-negative")
        if not reference:
            raise ValueError("max_changed_units requires a reference_assignment")
        max_changed_units = int(max_changed_units)
    reference_options = _reference_options_by_name(units, reference)
    if forbid_reference_downgrades and not reference_options:
        raise ValueError("forbid_reference_downgrades requires a reference_assignment")
    by_format = _options_by_format(units)
    assignment = {
        unit.name: fr.canonical_format_name(result.assignment[unit.name])
        for unit in units
    }
    if forbid_reference_downgrades:
        assignment = _coerce_reference_downgrades(
            assignment,
            reference_options,
            units,
        )
    (
        objective,
        diagonal,
        low_rank,
        bits_total,
        sketch_sum,
    ) = score_adjoint_assignment(
        units,
        rank,
        assignment,
        low_rank_weight=low_rank_weight,
        diagonal_weight=diagonal_weight,
    )
    sketch_sum_list = list(sketch_sum)
    changed_count = _assignment_change_count(units, assignment, reference)
    if max_changed_units is not None and changed_count > max_changed_units:
        raise ValueError(
            "initial assignment exceeds max_changed_units "
            f"({changed_count} > {max_changed_units})"
        )
    moves = 0
    move_limit = max_moves if max_moves is not None else len(units) * 4

    def _trial_values(unit: AdjointUnit, option: AdjointFormatOption):
        current = by_format[unit.name][assignment[unit.name]]
        if (
            forbid_reference_downgrades
            and _is_reference_downgrade(reference_options, unit.name, option)
        ):
            return None
        trial_changes = _trial_change_count(
            changed_count,
            reference,
            {unit.name: current.fmt},
            ((unit.name, option.fmt),),
        )
        if max_changed_units is not None and trial_changes > max_changed_units:
            return None
        trial_sketch = [
            sketch_sum_list[idx] - current.sketch[idx] + option.sketch[idx]
            for idx in range(rank)
        ]
        trial_diagonal = diagonal - current.diagonal_cost + option.diagonal_cost
        trial_bits = bits_total - current.bits_total + option.bits_total
        trial_low_rank = 0.5 / float(rank) * sum(v * v for v in trial_sketch)
        trial_objective = (
            float(diagonal_weight) * trial_diagonal
            + float(low_rank_weight) * trial_low_rank
        )
        return (
            trial_objective,
            trial_diagonal,
            trial_low_rank,
            trial_bits,
            tuple(trial_sketch),
            trial_changes,
        )

    while moves < move_limit and bits_total > float(target_bits_total) + 1e-6:
        best = None
        for unit in units:
            current = by_format[unit.name][assignment[unit.name]]
            for option in unit.options:
                if option.fmt == current.fmt:
                    continue
                if option.bits_total >= current.bits_total - 1e-6:
                    continue
                values = _trial_values(unit, option)
                if values is None:
                    continue
                saved_bits = bits_total - values[3]
                if saved_bits <= 1e-6:
                    continue
                objective_delta = values[0] - objective
                if values[3] > float(target_bits_total) + 1e-6:
                    key = (
                        0,
                        objective_delta / saved_bits,
                        objective_delta,
                        -saved_bits,
                        unit.name,
                        option.fmt,
                    )
                else:
                    key = (
                        1,
                        float(target_bits_total) - values[3],
                        objective_delta,
                        values[0],
                        unit.name,
                        option.fmt,
                    )
                if best is None or key < best[0]:
                    best = (key, unit.name, option.fmt, values)
        if best is None:
            break
        _, name, fmt, values = best
        assignment[name] = fmt
        objective, diagonal, low_rank, bits_total, sketch_sum, changed_count = values
        sketch_sum_list = list(sketch_sum)
        moves += 1

    while moves < move_limit:
        best = None
        for unit in units:
            current = by_format[unit.name][assignment[unit.name]]
            for option in unit.options:
                if option.fmt == current.fmt:
                    continue
                values = _trial_values(unit, option)
                if values is None:
                    continue
                if values[3] > float(target_bits_total) + 1e-6:
                    continue
                objective_delta = values[0] - objective
                if objective_delta >= -float(improvement_tol):
                    continue
                key = (
                    objective_delta,
                    values[3],
                    unit.name,
                    option.fmt,
                )
                if best is None or key < best[0]:
                    best = (key, unit.name, option.fmt, values)
        if best is None:
            break
        _, name, fmt, values = best
        assignment[name] = fmt
        objective, diagonal, low_rank, bits_total, sketch_sum, changed_count = values
        sketch_sum_list = list(sketch_sum)
        moves += 1

    return AdjointSolveResult(
        assignment=dict(assignment),
        objective=float(objective),
        diagonal_cost=float(diagonal),
        low_rank_cost=float(low_rank),
        lagrangian_objective=float(objective),
        bits_total=float(bits_total),
        passes=result.passes,
        moves=result.moves + moves,
        rank=rank,
        changed_units=changed_count if reference else None,
    )


def solve_low_rank_budget_sweep(
    units: Sequence[AdjointUnit],
    rank: int,
    *,
    target_bits_total: float,
    lambdas: Sequence[float] | None = None,
    initial_assignments: Sequence[Mapping[str, str] | None] | None = None,
    reference_assignment: Mapping[str, str] | None = None,
    max_changed_units: int | None = None,
    change_penalty: float = 0.0,
    forbid_reference_downgrades: bool = False,
    low_rank_weight: float = 1.0,
    diagonal_weight: float = 1.0,
    max_passes: int = 16,
) -> AdjointSolveResult:
    """Pick the best solution at or below ``target_bits_total`` from a sweep."""
    if lambdas is None:
        lambdas = (
            0.0,
            1e-12,
            3e-12,
            1e-11,
            3e-11,
            1e-10,
            3e-10,
            1e-9,
            3e-9,
            1e-8,
            3e-8,
            1e-7,
            3e-7,
            1e-6,
        )
    if initial_assignments is None:
        initial_assignments = (None,)
    else:
        initial_assignments = tuple(initial_assignments) or (None,)
    reference = _normalise_reference_assignment(units, reference_assignment)
    if max_changed_units is not None:
        if not reference:
            raise ValueError("max_changed_units requires a reference_assignment")
        if not any(item == reference for item in initial_assignments if item is not None):
            initial_assignments = (reference, *initial_assignments)
    if forbid_reference_downgrades and not reference:
        raise ValueError("forbid_reference_downgrades requires a reference_assignment")
    if forbid_reference_downgrades and not any(
        item == reference for item in initial_assignments if item is not None
    ):
        initial_assignments = (reference, *initial_assignments)
    feasible: list[AdjointSolveResult] = []
    all_results: list[AdjointSolveResult] = []
    for initial_assignment in initial_assignments:
        for lambda_penalty in lambdas:
            try:
                result = solve_low_rank_lagrangian(
                    units,
                    rank,
                    lambda_penalty=lambda_penalty,
                    initial_assignment=initial_assignment,
                    reference_assignment=reference,
                    max_changed_units=max_changed_units,
                    change_penalty=change_penalty,
                    forbid_reference_downgrades=forbid_reference_downgrades,
                    low_rank_weight=low_rank_weight,
                    diagonal_weight=diagonal_weight,
                    max_passes=max_passes,
                )
                result = polish_low_rank_to_budget(
                    units,
                    rank,
                    result,
                    target_bits_total=target_bits_total,
                    reference_assignment=reference,
                    max_changed_units=max_changed_units,
                    forbid_reference_downgrades=forbid_reference_downgrades,
                    low_rank_weight=low_rank_weight,
                    diagonal_weight=diagonal_weight,
                )
            except ValueError:
                continue
            all_results.append(result)
            if result.bits_total <= float(target_bits_total) + 1e-6:
                feasible.append(result)
    if feasible:
        return min(feasible, key=lambda result: (
            result.objective,
            result.changed_units if result.changed_units is not None else 0,
            -result.bits_total,
        ))
    if not all_results:
        raise ValueError("no adjoint L3 sweep starts satisfied the trust-region constraints")
    return min(all_results, key=lambda result: result.bits_total)


def result_to_json_dict(
    result: AdjointSolveResult,
    *,
    assignment: Mapping[str, str] | None = None,
    meta: Mapping | None = None,
) -> dict:
    out_assignment = dict(assignment) if assignment is not None else result.assignment
    return {
        "schema": "prismaquant.adjoint_l3.solve_result.v1",
        "assignment": out_assignment,
        "objective": result.objective,
        "diagonal_cost": result.diagonal_cost,
        "low_rank_cost": result.low_rank_cost,
        "lagrangian_objective": result.lagrangian_objective,
        "bits_total": result.bits_total,
        "passes": result.passes,
        "moves": result.moves,
        "rank": result.rank,
        **({"changed_units": result.changed_units} if result.changed_units is not None else {}),
        **({"meta": dict(meta)} if meta else {}),
    }


def build_move_report(
    units: Sequence[AdjointUnit],
    rank: int,
    assignment: Mapping[str, str],
    reference_assignment: Mapping[str, str],
    *,
    group_members: Mapping[str, Sequence[str]] | None = None,
    low_rank_weight: float = 1.0,
    diagonal_weight: float = 1.0,
) -> list[dict]:
    """Explain changed solve-unit decisions against a reference assignment.

    ``delta_objective_vs_revert`` is computed in the final assignment context:
    it is ``objective(final) - objective(final with this unit reverted)``.
    Negative values mean the new choice helps the surrogate objective; positive
    values are budget/coupling tradeoffs that are locally costly.
    """
    by_format = _options_by_format(units)
    current_assignment = {
        unit.name: fr.canonical_format_name(assignment[unit.name])
        for unit in units
    }
    reference = collapse_assignment_to_solve_units(
        reference_assignment,
        units,
        group_members,
    )
    current = score_adjoint_assignment(
        units,
        rank,
        current_assignment,
        low_rank_weight=low_rank_weight,
        diagonal_weight=diagonal_weight,
    )
    rows = []
    for unit in units:
        old_fmt = reference.get(unit.name)
        if old_fmt is None:
            continue
        old_fmt = fr.canonical_format_name(old_fmt)
        new_fmt = current_assignment[unit.name]
        if old_fmt == new_fmt or old_fmt not in by_format[unit.name]:
            continue
        reverted = dict(current_assignment)
        reverted[unit.name] = old_fmt
        trial = score_adjoint_assignment(
            units,
            rank,
            reverted,
            low_rank_weight=low_rank_weight,
            diagonal_weight=diagonal_weight,
        )
        old_option = by_format[unit.name][old_fmt]
        new_option = by_format[unit.name][new_fmt]
        rows.append({
            "name": unit.name,
            "members": list(group_members.get(unit.name, ())) if group_members else [unit.name],
            "from_format": old_fmt,
            "to_format": new_fmt,
            "delta_objective_vs_revert": float(current[0] - trial[0]),
            "objective_if_reverted": float(trial[0]),
            "delta_diagonal_vs_revert": float(current[1] - trial[1]),
            "delta_low_rank_vs_revert": float(current[2] - trial[2]),
            "delta_bits": float(new_option.bits_total - old_option.bits_total),
            "from_bits": float(old_option.bits_total),
            "to_bits": float(new_option.bits_total),
        })
    rows.sort(key=lambda row: (
        row["delta_objective_vs_revert"],
        row["delta_bits"],
        row["name"],
    ))
    return rows


def _load_pickle(path: str | Path):
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def _load_probe_stats(path: str | Path) -> dict:
    payload = _load_pickle(path)
    if isinstance(payload, Mapping) and isinstance(payload.get("stats"), Mapping):
        return dict(payload["stats"])
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError(f"probe file {path} does not contain a stats mapping")


def _format_from_assignment_value(value) -> str:
    if isinstance(value, str):
        return fr.canonical_format_name(value)
    if isinstance(value, int):
        if value == 16:
            return "BF16"
        raise ValueError(f"cannot infer format from integer config {value!r}")
    if not isinstance(value, Mapping):
        raise ValueError(f"cannot infer format from assignment value {value!r}")
    for spec in fr.list_formats():
        cfg = spec.autoround_config()
        keys = {"bits", "group_size", "data_type", "act_bits", "act_data_type"}
        if all(cfg.get(key) == value.get(key) for key in keys if key in cfg or key in value):
            return fr.canonical_format_name(spec.name)
    bits = int(value.get("bits", 0) or 0)
    data_type = str(value.get("data_type", ""))
    act_bits = int(value.get("act_bits", 16) or 16)
    group_size = int(value.get("group_size", 0) or 0)
    if bits == 16 and data_type == "float":
        return "BF16"
    if bits == 4 and data_type == "fp4_e2m1" and act_bits < 16:
        return "NVFP4"
    if bits == 8 and data_type == "fp8_e4m3" and act_bits < 16 and group_size == 32:
        return "MXFP8_E4M3"
    raise ValueError(f"cannot infer registered format from assignment value {value!r}")


def _load_assignment_json(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, Mapping) and isinstance(payload.get("assignment"), Mapping):
        payload = payload["assignment"]
    if not isinstance(payload, Mapping):
        raise ValueError(f"assignment JSON {path} is not a mapping")
    return {
        str(name): _format_from_assignment_value(fmt)
        for name, fmt in payload.items()
    }


def collapse_assignment_to_solve_units(
    assignment: Mapping[str, str],
    solve_units: Sequence[AdjointUnit],
    group_members: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """Map a raw/full assignment onto the units currently being solved.

    For grouped/fused units, member assignments are reduced to one seed format.
    Mixed member formats are legal here because this is only a starting point;
    choose the majority format and let the grouped solver enforce a shared
    final decision.
    """
    solve_names = {unit.name for unit in solve_units}
    out: dict[str, str] = {}
    if group_members:
        for group_name, members in group_members.items():
            if group_name not in solve_names:
                continue
            counts: dict[str, int] = defaultdict(int)
            for member in members:
                if member in assignment:
                    counts[fr.canonical_format_name(assignment[member])] += 1
            if counts:
                out[group_name] = min(
                    counts,
                    key=lambda fmt: (-counts[fmt], fr.get_format(fmt).effective_bits, fmt),
                )
    for unit in solve_units:
        if unit.name in assignment:
            out[unit.name] = fr.canonical_format_name(assignment[unit.name])
    return out


def _parse_float_list(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("float list must contain at least one value")
    return tuple(out)


def _parse_formats(value: str) -> list[fr.FormatSpec]:
    return [fr.get_format(part.strip()) for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve an adjoint-sketch L3 allocation artifact."
    )
    parser.add_argument("--adjoint-costs", required=True, help="Adjoint L3 JSON file")
    parser.add_argument("--probe", required=True, help="Probe pickle with stats")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model path for profile detection when --fused-groups is set",
    )
    parser.add_argument(
        "--formats",
        default="NVFP4,MXFP8_E4M3,BF16",
        help="Comma-separated legal format names",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--lambda-penalty",
        type=float,
        default=0.0,
        help="Lagrange penalty per total bit",
    )
    parser.add_argument(
        "--lambdas",
        default=None,
        help=(
            "Comma-separated lambda grid for --target-total-bits or "
            "--target-full-bpp sweeps"
        ),
    )
    parser.add_argument(
        "--target-total-bits",
        type=float,
        default=None,
        help="Optional target for a fixed lambda sweep",
    )
    parser.add_argument(
        "--diagonal-floor-frac",
        type=float,
        default=None,
        help="Optional solve-time override for adjoint_self_cost diagonal floor",
    )
    parser.add_argument(
        "--mse-diagonal-floor-frac",
        type=float,
        default=None,
        help="Optional solve-time override for output-MSE diagonal floor",
    )
    parser.add_argument(
        "--target-full-bpp",
        type=float,
        default=None,
        help=(
            "Full-model bpp target. Requires --base-assignment when some "
            "stats entries are outside the adjoint artifact; fixed entries "
            "keep their base formats and the remaining bit budget is solved."
        ),
    )
    parser.add_argument(
        "--base-assignment",
        default=None,
        help="Full assignment JSON used to compute fixed bits for --target-full-bpp",
    )
    parser.add_argument(
        "--seed-assignment",
        action="append",
        default=[],
        help=(
            "Optional assignment JSON used as a coordinate-descent start. "
            "May be passed multiple times; solve-result JSONs are accepted."
        ),
    )
    parser.add_argument(
        "--seed-from-base",
        action="store_true",
        help="Also use --base-assignment as a coordinate-descent start",
    )
    parser.add_argument(
        "--trust-reference-assignment",
        default=None,
        help=(
            "Assignment JSON used to count changed solve units for "
            "--max-changed-units/--change-penalty. Defaults to "
            "--base-assignment when available."
        ),
    )
    parser.add_argument(
        "--max-changed-units",
        type=int,
        default=None,
        help="Optional trust-region cap on solve-unit changes vs reference.",
    )
    parser.add_argument(
        "--change-penalty",
        type=float,
        default=0.0,
        help="Optional Lagrangian penalty for each changed solve unit.",
    )
    parser.add_argument(
        "--forbid-reference-downgrades",
        action="store_true",
        help=(
            "When a trust reference is provided, disallow formats with fewer "
            "bits than the reference choice for that solve unit."
        ),
    )
    parser.add_argument(
        "--full-assignment-output",
        default=None,
        help=(
            "Optional solve-result JSON whose assignment is the base assignment "
            "overlaid with the solved adjoint-unit choices"
        ),
    )
    parser.add_argument(
        "--move-report-output",
        default=None,
        help="Optional JSON report explaining solve-unit changes vs --base-assignment",
    )
    parser.add_argument("--max-passes", type=int, default=16)
    parser.add_argument(
        "--fused-groups",
        action="store_true",
        help="Solve profile-fused siblings as single shared-format units",
    )
    parser.add_argument(
        "--additive-costs-output",
        default=None,
        help="Optional path for legacy propagated_costs-compatible JSON",
    )
    parser.add_argument(
        "--legacy-pickle-output",
        default=None,
        help="Optional pickle payload compatible with --resume-l3-costs",
    )
    args = parser.parse_args(argv)

    payload = load_adjoint_l3_payload(args.adjoint_costs)
    payload = retune_adjoint_diagonal_costs(
        payload,
        diagonal_floor_frac=args.diagonal_floor_frac,
        mse_diagonal_floor_frac=args.mse_diagonal_floor_frac,
    )
    stats = _load_probe_stats(args.probe)
    formats = _parse_formats(args.formats)
    units = adjoint_units_from_payload(payload, stats=stats, formats=formats)
    rank = int(payload["rank"])
    solve_units = units
    group_members: dict[str, tuple[str, ...]] | None = None
    profile_name = None
    if args.fused_groups:
        if args.model:
            from .model_profiles import detect_profile

            profile = detect_profile(args.model)
        else:
            from .model_profiles import DefaultProfile

            profile = DefaultProfile()
        profile_name = getattr(profile, "name", type(profile).__name__)
        solve_units, group_members = group_adjoint_units_by_profile(units, profile)

    base_assignment = _load_assignment_json(args.base_assignment) if args.base_assignment else None
    fixed_bits = 0.0
    total_params = 0
    target_total_bits = args.target_total_bits
    payload_meta = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else {}
    target_meta = {
        key: payload_meta[key]
        for key in (
            "direction_mode",
            "objective_metric",
            "curvature",
            "fisher_temperature",
            "fisher_token_scope",
            "fisher_probe_distribution",
            "fisher_probes_per_sample",
            "fisher_seed",
            "solve_diagonal_floor_frac",
            "solve_mse_diagonal_floor_frac",
            "solve_mse_floor_scale",
        )
        if key in payload_meta
    }
    raw_unit_names = {unit.name for unit in units}
    all_base_names: list[str] = []
    fixed_names: list[str] = []
    if base_assignment is not None:
        all_base_names = sorted(set(stats) & set(base_assignment))
        if not all_base_names:
            raise ValueError("base assignment and probe stats have no shared names")
        fixed_names = [name for name in all_base_names if name not in raw_unit_names]
        total_params = sum(
            int(stats[name].get("n_params", 0) or 0) for name in all_base_names
        )
        from .propagated_cost import assignment_bit_total

        specs_by_name = {fr.canonical_format_name(spec.name): spec for spec in formats}
        fixed_assignment = {
            name: base_assignment[name]
            for name in fixed_names
        }
        fixed_bits = assignment_bit_total(stats, fixed_assignment, specs_by_name)

    if args.target_full_bpp is not None:
        if args.target_total_bits is not None:
            raise ValueError("use only one of --target-total-bits or --target-full-bpp")
        if base_assignment is None:
            raise ValueError("--target-full-bpp requires --base-assignment")
        target_total_bits = (
            float(args.target_full_bpp) * float(total_params) - float(fixed_bits)
        )
        if target_total_bits < 0.0:
            raise ValueError(
                f"target-full-bpp={args.target_full_bpp} leaves negative "
                f"adjoint-unit budget after fixed bits ({target_total_bits})"
            )
        target_meta.update({
            "target_full_bpp": float(args.target_full_bpp),
            "computed_target_total_bits": float(target_total_bits),
            "base_assignment": str(args.base_assignment),
            "fixed_entry_count": len(fixed_names),
            "total_param_count": int(total_params),
            "fixed_bits": float(fixed_bits),
        })
    if args.full_assignment_output and base_assignment is None:
        raise ValueError("--full-assignment-output requires --base-assignment")
    if args.move_report_output and base_assignment is None:
        raise ValueError("--move-report-output requires --base-assignment")
    lambda_grid = _parse_float_list(args.lambdas)
    initial_assignments: list[Mapping[str, str] | None] = [None]
    seed_paths: list[str] = list(args.seed_assignment or [])
    if args.seed_from_base:
        if base_assignment is None:
            raise ValueError("--seed-from-base requires --base-assignment")
        seed_paths.append(args.base_assignment)
    for seed_path in seed_paths:
        seed_assignment = _load_assignment_json(seed_path)
        collapsed = collapse_assignment_to_solve_units(
            seed_assignment,
            solve_units,
            group_members,
        )
        if collapsed:
            initial_assignments.append(collapsed)
    reference_assignment = None
    reference_path = args.trust_reference_assignment
    if reference_path is None and base_assignment is not None:
        reference_path = args.base_assignment
    if reference_path is not None:
        reference_assignment = collapse_assignment_to_solve_units(
            _load_assignment_json(reference_path),
            solve_units,
            group_members,
        )
    if (
        args.max_changed_units is not None
        or float(args.change_penalty) > 0.0
        or args.forbid_reference_downgrades
    ) and not reference_assignment:
        raise ValueError(
            "--max-changed-units/--change-penalty/"
            "--forbid-reference-downgrades require "
            "--trust-reference-assignment or --base-assignment"
        )

    if target_total_bits is None:
        result = solve_low_rank_lagrangian(
            solve_units,
            rank,
            lambda_penalty=args.lambda_penalty,
            initial_assignment=initial_assignments[-1] if len(initial_assignments) > 1 else None,
            reference_assignment=reference_assignment,
            max_changed_units=args.max_changed_units,
            change_penalty=args.change_penalty,
            forbid_reference_downgrades=args.forbid_reference_downgrades,
            max_passes=args.max_passes,
        )
    else:
        result = solve_low_rank_budget_sweep(
            solve_units,
            rank,
            target_bits_total=target_total_bits,
            lambdas=lambda_grid,
            initial_assignments=initial_assignments,
            reference_assignment=reference_assignment,
            max_changed_units=args.max_changed_units,
            change_penalty=args.change_penalty,
            forbid_reference_downgrades=args.forbid_reference_downgrades,
            max_passes=args.max_passes,
        )
    output_assignment = result.assignment
    meta = dict(target_meta) if target_meta else None
    if target_total_bits is not None:
        if meta is None:
            meta = {}
        meta.update({
            "target_solved_bits_total": float(target_total_bits),
            "achieved_solved_bits_total": float(result.bits_total),
            "target_feasible": bool(result.bits_total <= float(target_total_bits) + 1e-6),
        })
    if group_members is not None:
        output_assignment = expand_grouped_assignment(result.assignment, group_members)
        fused_meta = {
            "fused_groups": True,
            "profile": profile_name,
            "raw_unit_count": len(units),
            "grouped_unit_count": len(solve_units),
            "fused_group_count": sum(
                1 for members in group_members.values() if len(members) > 1
            ),
            "group_members": {
                key: list(value) for key, value in sorted(group_members.items())
            },
        }
        if meta is None:
            meta = fused_meta
        else:
            meta.update(fused_meta)

    full_assignment = None
    if base_assignment is not None:
        full_assignment = dict(base_assignment)
        full_assignment.update(output_assignment)
        if meta is None:
            meta = {}
        if total_params > 0:
            full_bits_total = float(fixed_bits) + float(result.bits_total)
            meta.update({
                "solved_full_bits_total": full_bits_total,
                "solved_full_bpp": full_bits_total / float(total_params),
                "full_assignment_entry_count": len(full_assignment),
            })
    if meta is None:
        meta = {}
    if lambda_grid is not None:
        meta["lambda_grid"] = list(lambda_grid)
    if len(initial_assignments) > 1:
        meta["seed_assignment_count"] = len(initial_assignments) - 1
    if reference_assignment:
        meta["trust_reference_assignment"] = str(reference_path)
        if args.max_changed_units is not None:
            meta["max_changed_units"] = int(args.max_changed_units)
        if float(args.change_penalty) > 0.0:
            meta["change_penalty"] = float(args.change_penalty)
        if args.forbid_reference_downgrades:
            meta["forbid_reference_downgrades"] = True

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result_to_json_dict(result, assignment=output_assignment, meta=meta),
            indent=2,
        )
        + "\n"
    )

    if args.full_assignment_output:
        full_output = Path(args.full_assignment_output)
        full_output.parent.mkdir(parents=True, exist_ok=True)
        full_meta = dict(meta or {})
        full_meta["assignment_scope"] = "base_plus_adjoint_overlay"
        full_output.write_text(
            json.dumps(
                result_to_json_dict(
                    result,
                    assignment=full_assignment,
                    meta=full_meta,
                ),
                indent=2,
            )
            + "\n"
        )

    if args.move_report_output:
        move_report = build_move_report(
            solve_units,
            rank,
            result.assignment,
            base_assignment,
            group_members=group_members,
        )
        report_path = Path(args.move_report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "schema": "prismaquant.adjoint_l3.move_report.v1",
                    "move_count": len(move_report),
                    "moves": move_report,
                },
                indent=2,
            )
            + "\n"
        )

    if args.additive_costs_output:
        additive = adjoint_payload_to_propagated_costs(payload)
        additive_path = Path(args.additive_costs_output)
        additive_path.parent.mkdir(parents=True, exist_ok=True)
        additive_path.write_text(json.dumps(additive, indent=2) + "\n")

    if args.legacy_pickle_output:
        legacy = adjoint_payload_to_l3_resume_payload(
            payload,
            formats=[spec.name for spec in formats],
        )
        legacy_path = Path(args.legacy_pickle_output)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        with legacy_path.open("wb") as fh:
            pickle.dump(legacy, fh)

    print(
        f"[adjoint-l3] wrote {output} "
        f"objective={result.objective:.8g} bits_total={result.bits_total:.0f} "
        f"moves={result.moves} passes={result.passes}"
        + (
            f" changed_units={result.changed_units}"
            if result.changed_units is not None else ""
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
