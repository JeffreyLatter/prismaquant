"""interaction_refine.py — sparse interaction-aware refinement near the knee.

The main allocator remains additive and cheap. This module adds a bounded
second stage:

  1. Collapse serving-tied tensors into refinement units
  2. Select the most important units near a base assignment
  3. Refine them with sparse pairwise interaction terms under the same budget

This follows the spirit of recent interaction-aware MPQ work without turning
the whole problem into a dense quadratic program over every layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import prod

import numpy as np

from .allocator import Candidate, _shape_from_stats, _group_by_profile


@dataclass(frozen=True)
class UnitOption:
    fmt: str
    bits_total: float
    predicted_dloss: float


@dataclass
class RefinementUnit:
    key: str
    members: tuple[str, ...]
    base_fmt: str
    base_member_fmts: tuple[tuple[str, str], ...]
    options: tuple[UnitOption, ...]

    @property
    def option_map(self) -> dict[str, UnitOption]:
        return {opt.fmt: opt for opt in self.options}


def _block_group_for_name(name: str, present: set[str]) -> tuple[str, ...] | None:
    parts = name.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    prefix = ".".join(parts[:3])
    leaf = parts[-1]
    if parts[3] == "self_attn" and leaf in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        members = tuple(
            sorted(
                f"{prefix}.self_attn.{proj}"
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
                if f"{prefix}.self_attn.{proj}" in present
            )
        )
        return members if len(members) > 1 else None
    if parts[3] == "mlp" and leaf in {"gate_proj", "up_proj", "down_proj"}:
        members = tuple(
            sorted(
                f"{prefix}.mlp.{proj}"
                for proj in ("gate_proj", "up_proj", "down_proj")
                if f"{prefix}.mlp.{proj}" in present
            )
        )
        return members if len(members) > 1 else None
    return None


def _layer_group_for_name(name: str, present: set[str]) -> tuple[str, ...] | None:
    parts = name.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    prefix = ".".join(parts[:3]) + "."
    members = tuple(sorted(n for n in present if n.startswith(prefix)))
    return members if len(members) > 1 else None


_SIBLING_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (".self_attn.", ("q_proj", "k_proj", "v_proj")),
    (".mlp.", ("gate_proj", "up_proj")),
    (".mlp.shared_expert.", ("gate_proj", "up_proj")),
    (".linear_attn.", ("in_proj_qkv", "in_proj_z")),
    (".linear_attn.", ("in_proj_a", "in_proj_b")),
)


def _name_pattern_siblings(name: str, present: set[str]) -> tuple[str, ...] | None:
    """Heuristic sibling detector by name pattern. Covers the serving-
    fused families (q/k/v, gate/up, in_proj_qkvz/ba) across all Qwen/
    LLaMA/DeltaNet variants we ship. Used when a profile-based classifier
    isn't available (e.g. bare-name test inputs)."""
    for parent_marker, leaves in _SIBLING_PATTERNS:
        idx = name.rfind(parent_marker)
        if idx < 0:
            continue
        parent = name[:idx + len(parent_marker)]
        leaf = name[idx + len(parent_marker):]
        if leaf not in leaves:
            continue
        members = tuple(sorted(
            f"{parent}{cand}" for cand in leaves if f"{parent}{cand}" in present
        ))
        if len(members) > 1:
            return members
    return None


def _unit_groups(names: list[str], unit_scope: str = "sibling") -> list[tuple[str, ...]]:
    present = set(names)
    # Build the sibling-key → [names] map once. `_group_by_profile` uses
    # the profile's `fused_sibling_group` classifier (derived from the
    # vLLM class's `packed_modules_mapping`), which is the same key the
    # native exporter uses for joint NVFP4 globals — so a refinement
    # unit groups exactly what vLLM fuses at serve time. We fall back
    # to the pattern detector below when the profile can't classify a
    # name (happens with bare-name test inputs, or legacy Linears that
    # don't appear in a vLLM packed_modules_mapping).
    from .model_profiles import DefaultProfile
    profile = DefaultProfile()
    sibling_key_to_names = _group_by_profile(list(present), profile)
    name_to_fusion = {
        name: tuple(sorted(members))
        for members in sibling_key_to_names.values()
        for name in members
    }

    groups: dict[tuple[str, ...], tuple[str, ...]] = {}
    for name in names:
        if ".__fused__." in name:
            key = (name,)
        else:
            key = None
            if unit_scope == "layer":
                key = _layer_group_for_name(name, present)
            if unit_scope in {"block", "hybrid"}:
                key = _block_group_for_name(name, present)
            if unit_scope == "layer" and key is None:
                key = _layer_group_for_name(name, present)
            if key is None:
                sibs = name_to_fusion.get(name)
                if sibs is not None and len(sibs) > 1:
                    key = sibs
            if key is None:
                sibs = _name_pattern_siblings(name, present)
                if sibs is not None:
                    key = sibs
            if key is None:
                key = (name,)
            else:
                key = tuple(sorted(set(key)))
        groups[key] = tuple(sorted(set(key)))
    return sorted(groups.values())


def build_refinement_units(
    stats: dict,
    candidates: dict[str, list[Candidate]],
    assignment: dict[str, str],
    unit_scope: str = "sibling",
) -> list[RefinementUnit]:
    units = []
    for members in _unit_groups(list(assignment.keys()), unit_scope=unit_scope):
        base_fmts = {assignment[m] for m in members}
        base_member_fmts = tuple((member, assignment[member]) for member in members)
        heterogeneous_base = len(base_fmts) != 1
        base_fmt = "__base__" if heterogeneous_base else next(iter(base_fmts))
        fmt_sets = [{cand.fmt for cand in candidates[m]} for m in members if m in candidates]
        if not fmt_sets:
            continue
        shared = set.intersection(*fmt_sets)
        options = []
        if heterogeneous_base:
            bits_total = 0.0
            predicted = 0.0
            for member in members:
                cand = next(c for c in candidates[member] if c.fmt == assignment[member])
                n_params = stats[member]["n_params"]
                bits_total += cand.bits_per_param * n_params
                predicted += cand.predicted_dloss
            options.append(UnitOption(fmt="__base__", bits_total=bits_total, predicted_dloss=predicted))
        for fmt in shared:
            bits_total = 0.0
            predicted = 0.0
            for member in members:
                shape = _shape_from_stats(stats[member])
                n_params = stats[member]["n_params"]
                cand = next(c for c in candidates[member] if c.fmt == fmt)
                bits_total += cand.bits_per_param * n_params
                predicted += cand.predicted_dloss
            options.append(UnitOption(fmt=fmt, bits_total=bits_total, predicted_dloss=predicted))
        options.sort(key=lambda opt: (opt.bits_total, opt.predicted_dloss, opt.fmt))
        if not options:
            continue
        key = "|".join(members)
        units.append(
                RefinementUnit(
                    key=key,
                    members=members,
                    base_fmt=base_fmt,
                    base_member_fmts=base_member_fmts,
                    options=tuple(options),
                )
            )
    return units


def select_critical_units(units: list[RefinementUnit], top_n: int) -> list[RefinementUnit]:
    scored = []
    for unit in units:
        opt_map = unit.option_map
        base = opt_map[unit.base_fmt]
        cheapest = min(unit.options, key=lambda opt: (opt.bits_total, opt.predicted_dloss))
        gain = max(cheapest.predicted_dloss - base.predicted_dloss, 0.0)
        scored.append((gain, base.predicted_dloss, unit.key, unit))
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return [row[-1] for row in scored[:top_n]]


def neighborhood_options(unit: RefinementUnit, radius: int = 1) -> tuple[UnitOption, ...]:
    opts = list(unit.options)
    idx = next((i for i, opt in enumerate(opts) if opt.fmt == unit.base_fmt), 0)
    lo = max(0, idx - radius)
    hi = min(len(opts), idx + radius + 1)
    return tuple(opts[lo:hi])


def base_assignment_for_units(units: list[RefinementUnit]) -> dict[str, str]:
    return {unit.key: unit.base_fmt for unit in units}


def expand_unit_assignment(units: list[RefinementUnit], choices: dict[str, str]) -> dict[str, str]:
    out = {}
    for unit in units:
        fmt = choices.get(unit.key, unit.base_fmt)
        if fmt == "__base__":
            for member, member_fmt in unit.base_member_fmts:
                out[member] = member_fmt
        else:
            for member in unit.members:
                out[member] = fmt
    return out


def objective_delta(
    choices: dict[str, str],
    units: list[RefinementUnit],
    unary: dict[str, dict[str, float]],
    pairwise: dict[tuple[str, str, str, str], float],
) -> float:
    total = 0.0
    for unit in units:
        total += unary.get(unit.key, {}).get(choices.get(unit.key, unit.base_fmt), 0.0)
    for left, right in combinations(sorted(choices), 2):
        lfmt = choices[left]
        rfmt = choices[right]
        total += _pairwise_value(pairwise, left, lfmt, right, rfmt)
    return total


def make_pair_key(left_unit: str, left_fmt: str, right_unit: str, right_fmt: str):
    if left_unit <= right_unit:
        return (left_unit, left_fmt, right_unit, right_fmt)
    return (right_unit, right_fmt, left_unit, left_fmt)


def _pairwise_value(
    pairwise: dict[tuple[str, str, str, str], float],
    left_unit: str,
    left_fmt: str,
    right_unit: str,
    right_fmt: str,
) -> float:
    return float(pairwise.get(make_pair_key(left_unit, left_fmt, right_unit, right_fmt), 0.0))


def psd_project_quadratic(
    units: list[RefinementUnit],
    unary: dict[str, dict[str, float]],
    pairwise: dict[tuple[str, str, str, str], float],
    *,
    allowed: dict[str, tuple[UnitOption, ...]] | None = None,
    min_eigenvalue: float = 0.0,
    shrink_to_diagonal: float = 0.0,
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str, str, str], float], dict]:
    """Project sparse pair residuals to a PSD quadratic over option variables.

    The local refiner represents a one-hot choice per unit and scores pair
    residuals as ``sum_{i<j} pairwise(choice_i, choice_j)``.  For projection we
    lift that to ``x.T @ Q @ x`` where off-diagonal entries are half the stored
    pairwise residual, clamp negative eigenvalues, then fold diagonal terms back
    into unary option costs.  The projection is used as a stabilizing surrogate;
    final candidates are still validated by measured KL.
    """
    option_maps = _candidate_choice_maps(units, allowed)
    option_keys: list[tuple[str, str]] = []
    option_index: dict[tuple[str, str], int] = {}
    for unit in sorted(units, key=lambda item: item.key):
        for fmt in sorted(option_maps[unit.key]):
            key = (unit.key, fmt)
            option_index[key] = len(option_keys)
            option_keys.append(key)

    dim = len(option_keys)
    if dim == 0:
        return dict(unary), dict(pairwise), {
            "enabled": True,
            "dimension": 0,
            "reason": "empty_option_space",
        }

    q = np.zeros((dim, dim), dtype=np.float64)
    loaded_entries = 0
    skipped_entries = 0
    for key, value in pairwise.items():
        if len(key) != 4:
            skipped_entries += 1
            continue
        left_unit, left_fmt, right_unit, right_fmt = key
        left_idx = option_index.get((left_unit, left_fmt))
        right_idx = option_index.get((right_unit, right_fmt))
        if left_idx is None or right_idx is None or left_idx == right_idx:
            skipped_entries += 1
            continue
        q[left_idx, right_idx] += float(value) / 2.0
        q[right_idx, left_idx] += float(value) / 2.0
        loaded_entries += 1

    q = (q + q.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(q)
    min_eigenvalue = max(float(min_eigenvalue), 0.0)
    clipped = np.maximum(eigvals, min_eigenvalue)
    projected = (eigvecs * clipped) @ eigvecs.T
    projected = (projected + projected.T) / 2.0

    shrink = min(max(float(shrink_to_diagonal), 0.0), 1.0)
    if shrink > 0.0:
        projected = (1.0 - shrink) * projected + shrink * np.diag(np.diag(projected))
        projected = (projected + projected.T) / 2.0

    projected_unary = {
        unit_key: {fmt: float(value) for fmt, value in fmt_costs.items()}
        for unit_key, fmt_costs in unary.items()
    }
    for idx, (unit_key, fmt) in enumerate(option_keys):
        projected_unary.setdefault(unit_key, {})
        projected_unary[unit_key][fmt] = (
            float(projected_unary[unit_key].get(fmt, 0.0))
            + float(projected[idx, idx])
        )

    projected_pairwise: dict[tuple[str, str, str, str], float] = {}
    for left_idx, (left_unit, left_fmt) in enumerate(option_keys):
        for right_idx in range(left_idx + 1, dim):
            right_unit, right_fmt = option_keys[right_idx]
            if left_unit == right_unit:
                continue
            value = float(2.0 * projected[left_idx, right_idx])
            if abs(value) <= 1e-15:
                continue
            projected_pairwise[make_pair_key(left_unit, left_fmt, right_unit, right_fmt)] = value

    original_min = float(eigvals[0]) if eigvals.size else 0.0
    projected_eigvals = np.linalg.eigvalsh(projected)
    meta = {
        "enabled": True,
        "dimension": int(dim),
        "input_pairwise_entries": int(len(pairwise)),
        "loaded_pairwise_entries": int(loaded_entries),
        "skipped_pairwise_entries": int(skipped_entries),
        "output_pairwise_entries": int(len(projected_pairwise)),
        "negative_eigenvalues_clipped": int(np.sum(eigvals < min_eigenvalue - 1e-12)),
        "min_eigenvalue_before": original_min,
        "min_eigenvalue_after": (
            float(projected_eigvals[0]) if projected_eigvals.size else 0.0
        ),
        "max_eigenvalue_before": (
            float(eigvals[-1]) if eigvals.size else 0.0
        ),
        "max_eigenvalue_after": (
            float(projected_eigvals[-1]) if projected_eigvals.size else 0.0
        ),
        "min_eigenvalue_floor": float(min_eigenvalue),
        "shrink_to_diagonal": float(shrink),
        "input_frobenius_norm": float(np.linalg.norm(q, ord="fro")),
        "output_frobenius_norm": float(np.linalg.norm(projected, ord="fro")),
    }
    return projected_unary, projected_pairwise, meta


def _bits_total_for_choices(
    choices: dict[str, str],
    unit_map: dict[str, RefinementUnit],
    fixed_bits_total: float,
) -> float:
    total = fixed_bits_total
    for unit_key, fmt in choices.items():
        total += unit_map[unit_key].option_map[fmt].bits_total
    return total


def _candidate_choice_maps(units: list[RefinementUnit], allowed: dict[str, tuple[UnitOption, ...]] | None):
    out = {}
    for unit in units:
        opts = allowed[unit.key] if allowed and unit.key in allowed else unit.options
        out[unit.key] = {opt.fmt: opt for opt in opts}
    return out


def _initial_choice_map(
    units: list[RefinementUnit],
    option_maps: dict[str, dict[str, UnitOption]],
    initial_choices: dict[str, str] | None,
) -> dict[str, str]:
    current = {unit.key: unit.base_fmt for unit in units}
    if initial_choices:
        for unit in units:
            fmt = initial_choices.get(unit.key)
            if fmt in option_maps[unit.key]:
                current[unit.key] = fmt
    return current


def _exact_local_refine(
    units: list[RefinementUnit],
    unary: dict[str, dict[str, float]],
    pairwise: dict[tuple[str, str, str, str], float],
    target_total_bits: float,
    fixed_bits_total: float,
    option_maps: dict[str, dict[str, UnitOption]],
    initial_choices: dict[str, str] | None,
) -> dict:
    unit_map = {unit.key: unit for unit in units}
    initial = _initial_choice_map(units, option_maps, initial_choices)
    initial_bits = _bits_total_for_choices(initial, unit_map, fixed_bits_total)
    if initial_bits > target_total_bits + 1e-6:
        raise ValueError("initial refinement state exceeds target budget")

    ordered_units = sorted(
        units,
        key=lambda unit: (len(option_maps[unit.key]), unit.key),
        reverse=True,
    )
    option_lists = [
        tuple(option_maps[unit.key].values())
        for unit in ordered_units
    ]
    state_count = prod(max(len(options), 1) for options in option_lists)
    suffix_min_bits = [0.0 for _unit in range(len(ordered_units) + 1)]
    for idx in range(len(ordered_units) - 1, -1, -1):
        suffix_min_bits[idx] = suffix_min_bits[idx + 1] + min(
            opt.bits_total for opt in option_lists[idx]
        )
    suffix_min_unary = [0.0 for _unit in range(len(ordered_units) + 1)]
    for idx in range(len(ordered_units) - 1, -1, -1):
        unit = ordered_units[idx]
        suffix_min_unary[idx] = suffix_min_unary[idx + 1] + min(
            float(unary.get(unit.key, {}).get(opt.fmt, 0.0))
            for opt in option_lists[idx]
        )
    pair_min_by_index: dict[tuple[int, int], float] = {}
    for left_idx, right_idx in combinations(range(len(ordered_units)), 2):
        left = ordered_units[left_idx]
        right = ordered_units[right_idx]
        pair_min_by_index[(left_idx, right_idx)] = min(
            _pairwise_value(pairwise, left.key, left_opt.fmt, right.key, right_opt.fmt)
            for left_opt in option_lists[left_idx]
            for right_opt in option_lists[right_idx]
        )
    suffix_min_unassigned_pairwise = [
        0.0 for _unit in range(len(ordered_units) + 1)
    ]
    for idx in range(len(ordered_units) - 1, -1, -1):
        suffix_min_unassigned_pairwise[idx] = (
            suffix_min_unassigned_pairwise[idx + 1]
            + sum(
                pair_min_by_index[(idx, right_idx)]
                for right_idx in range(idx + 1, len(ordered_units))
            )
        )

    best_choices = dict(initial)
    best_obj = objective_delta(initial, units, unary, pairwise)
    best_bits = initial_bits
    states_evaluated = 0
    nodes_visited = 0
    states_pruned = 0

    assigned: dict[str, str] = {}
    assigned_by_index: dict[int, str] = {}

    def _objective_lower_bound(index: int, obj_total: float) -> float:
        lower = (
            float(obj_total)
            + suffix_min_unary[index]
            + suffix_min_unassigned_pairwise[index]
        )
        for prev_idx, prev_fmt in assigned_by_index.items():
            prev_unit = ordered_units[prev_idx]
            for next_idx in range(index, len(ordered_units)):
                next_unit = ordered_units[next_idx]
                lower += min(
                    _pairwise_value(
                        pairwise,
                        prev_unit.key,
                        prev_fmt,
                        next_unit.key,
                        option.fmt,
                    )
                    for option in option_lists[next_idx]
                )
        return lower

    def _search(index: int, bits_total: float, obj_total: float) -> None:
        nonlocal best_choices, best_obj, best_bits, states_evaluated
        nonlocal nodes_visited, states_pruned
        nodes_visited += 1
        if bits_total + suffix_min_bits[index] > target_total_bits + 1e-6:
            states_pruned += 1
            return
        if _objective_lower_bound(index, obj_total) >= best_obj - 1e-12:
            states_pruned += 1
            return
        if index == len(ordered_units):
            states_evaluated += 1
            if obj_total + 1e-12 < best_obj:
                best_obj = obj_total
                best_bits = bits_total
                best_choices = dict(assigned)
            return

        unit = ordered_units[index]
        for option in option_lists[index]:
            next_bits = bits_total + option.bits_total
            if next_bits + suffix_min_bits[index + 1] > target_total_bits + 1e-6:
                continue
            delta = float(unary.get(unit.key, {}).get(option.fmt, 0.0))
            for prev_key, prev_fmt in assigned.items():
                delta += _pairwise_value(
                    pairwise,
                    prev_key,
                    prev_fmt,
                    unit.key,
                    option.fmt,
                )
            assigned[unit.key] = option.fmt
            assigned_by_index[index] = option.fmt
            _search(index + 1, next_bits, obj_total + delta)
            assigned_by_index.pop(index, None)
            assigned.pop(unit.key, None)

    _search(0, fixed_bits_total, 0.0)
    return {
        "choices": best_choices,
        "objective_delta": best_obj,
        "bits_total": best_bits,
        "bits_per_param": None,
        "solver": "exact",
        "states_evaluated": states_evaluated,
        "state_count": state_count,
        "nodes_visited": nodes_visited,
        "states_pruned": states_pruned,
    }


def _lagrangian_score(
    obj_total: float,
    bits_total: float,
    *,
    total_params: float,
    bpp_penalty: float,
) -> float:
    if total_params <= 0:
        raise ValueError("total_params must be positive for lagrangian refinement")
    return float(obj_total) + float(bpp_penalty) * (float(bits_total) / float(total_params))


def _exact_lagrangian_refine(
    units: list[RefinementUnit],
    unary: dict[str, dict[str, float]],
    pairwise: dict[tuple[str, str, str, str], float],
    fixed_bits_total: float,
    option_maps: dict[str, dict[str, UnitOption]],
    initial_choices: dict[str, str] | None,
    *,
    total_params: float,
    bpp_penalty: float,
) -> dict:
    unit_map = {unit.key: unit for unit in units}
    initial = _initial_choice_map(units, option_maps, initial_choices)
    ordered_units = sorted(
        units,
        key=lambda unit: (len(option_maps[unit.key]), unit.key),
        reverse=True,
    )
    option_lists = [
        tuple(option_maps[unit.key].values())
        for unit in ordered_units
    ]
    state_count = prod(max(len(options), 1) for options in option_lists)

    bit_penalty_per_bit = float(bpp_penalty) / float(total_params)
    suffix_min_adjusted_unary = [0.0 for _unit in range(len(ordered_units) + 1)]
    for idx in range(len(ordered_units) - 1, -1, -1):
        unit = ordered_units[idx]
        suffix_min_adjusted_unary[idx] = suffix_min_adjusted_unary[idx + 1] + min(
            float(unary.get(unit.key, {}).get(opt.fmt, 0.0))
            + bit_penalty_per_bit * float(opt.bits_total)
            for opt in option_lists[idx]
        )
    pair_min_by_index: dict[tuple[int, int], float] = {}
    for left_idx, right_idx in combinations(range(len(ordered_units)), 2):
        left = ordered_units[left_idx]
        right = ordered_units[right_idx]
        pair_min_by_index[(left_idx, right_idx)] = min(
            _pairwise_value(pairwise, left.key, left_opt.fmt, right.key, right_opt.fmt)
            for left_opt in option_lists[left_idx]
            for right_opt in option_lists[right_idx]
        )
    suffix_min_unassigned_pairwise = [
        0.0 for _unit in range(len(ordered_units) + 1)
    ]
    for idx in range(len(ordered_units) - 1, -1, -1):
        suffix_min_unassigned_pairwise[idx] = (
            suffix_min_unassigned_pairwise[idx + 1]
            + sum(
                pair_min_by_index[(idx, right_idx)]
                for right_idx in range(idx + 1, len(ordered_units))
            )
        )

    initial_bits = _bits_total_for_choices(initial, unit_map, fixed_bits_total)
    initial_obj = objective_delta(initial, units, unary, pairwise)
    best_choices = dict(initial)
    best_obj = initial_obj
    best_bits = initial_bits
    best_score = _lagrangian_score(
        initial_obj,
        initial_bits,
        total_params=total_params,
        bpp_penalty=bpp_penalty,
    )
    states_evaluated = 0
    nodes_visited = 0
    states_pruned = 0

    assigned: dict[str, str] = {}
    assigned_by_index: dict[int, str] = {}

    def _score_lower_bound(index: int, obj_total: float, bits_total: float) -> float:
        lower = (
            float(obj_total)
            + bit_penalty_per_bit * float(bits_total)
            + suffix_min_adjusted_unary[index]
            + suffix_min_unassigned_pairwise[index]
        )
        for prev_idx, prev_fmt in assigned_by_index.items():
            prev_unit = ordered_units[prev_idx]
            for next_idx in range(index, len(ordered_units)):
                next_unit = ordered_units[next_idx]
                lower += min(
                    _pairwise_value(
                        pairwise,
                        prev_unit.key,
                        prev_fmt,
                        next_unit.key,
                        option.fmt,
                    )
                    for option in option_lists[next_idx]
                )
        return lower

    def _search(index: int, bits_total: float, obj_total: float) -> None:
        nonlocal best_choices, best_obj, best_bits, best_score
        nonlocal states_evaluated, nodes_visited, states_pruned
        nodes_visited += 1
        if _score_lower_bound(index, obj_total, bits_total) >= best_score - 1e-12:
            states_pruned += 1
            return
        if index == len(ordered_units):
            states_evaluated += 1
            score = _lagrangian_score(
                obj_total,
                bits_total,
                total_params=total_params,
                bpp_penalty=bpp_penalty,
            )
            if score + 1e-12 < best_score:
                best_score = score
                best_obj = obj_total
                best_bits = bits_total
                best_choices = dict(assigned)
            return

        unit = ordered_units[index]
        for option in option_lists[index]:
            delta = float(unary.get(unit.key, {}).get(option.fmt, 0.0))
            for prev_key, prev_fmt in assigned.items():
                delta += _pairwise_value(
                    pairwise,
                    prev_key,
                    prev_fmt,
                    unit.key,
                    option.fmt,
                )
            assigned[unit.key] = option.fmt
            assigned_by_index[index] = option.fmt
            _search(index + 1, bits_total + option.bits_total, obj_total + delta)
            assigned_by_index.pop(index, None)
            assigned.pop(unit.key, None)

    _search(0, fixed_bits_total, 0.0)
    return {
        "choices": best_choices,
        "objective_delta": best_obj,
        "bits_total": best_bits,
        "bits_per_param": best_bits / float(total_params),
        "solver": "lagrangian_exact",
        "lagrangian_objective": best_score,
        "lagrangian_bpp_penalty": float(bpp_penalty),
        "states_evaluated": states_evaluated,
        "state_count": state_count,
        "nodes_visited": nodes_visited,
        "states_pruned": states_pruned,
    }


def sparse_local_refine(
    units: list[RefinementUnit],
    unary: dict[str, dict[str, float]],
    pairwise: dict[tuple[str, str, str, str], float],
    target_total_bits: float,
    fixed_bits_total: float,
    allowed: dict[str, tuple[UnitOption, ...]] | None = None,
    max_passes: int = 8,
    initial_choices: dict[str, str] | None = None,
    exact_max_states: int = 2_000_000,
    solver_mode: str = "budget",
    lagrangian_bpp_penalty: float = 0.0,
    total_params: float | None = None,
) -> dict:
    unit_map = {unit.key: unit for unit in units}
    option_maps = _candidate_choice_maps(units, allowed)
    state_count = prod(max(len(options), 1) for options in option_maps.values())
    mode = str(solver_mode or "budget").lower()
    if mode in {"lagrangian", "qubo"}:
        if total_params is None or float(total_params) <= 0:
            raise ValueError("total_params must be positive for lagrangian refinement")
        if exact_max_states > 0 and state_count <= int(exact_max_states):
            return _exact_lagrangian_refine(
                units,
                unary,
                pairwise,
                fixed_bits_total,
                option_maps,
                initial_choices,
                total_params=float(total_params),
                bpp_penalty=float(lagrangian_bpp_penalty),
            )
        current = _initial_choice_map(units, option_maps, initial_choices)
        current_bits = _bits_total_for_choices(current, unit_map, fixed_bits_total)
        current_obj = objective_delta(current, units, unary, pairwise)
        current_score = _lagrangian_score(
            current_obj,
            current_bits,
            total_params=float(total_params),
            bpp_penalty=float(lagrangian_bpp_penalty),
        )
        for _pass in range(max_passes):
            best = None
            for unit in units:
                for fmt in option_maps[unit.key]:
                    if fmt == current[unit.key]:
                        continue
                    trial = dict(current)
                    trial[unit.key] = fmt
                    bits = _bits_total_for_choices(trial, unit_map, fixed_bits_total)
                    obj = objective_delta(trial, units, unary, pairwise)
                    score = _lagrangian_score(
                        obj,
                        bits,
                        total_params=float(total_params),
                        bpp_penalty=float(lagrangian_bpp_penalty),
                    )
                    if score + 1e-12 < current_score and (best is None or score < best[0]):
                        best = (score, obj, bits, trial)
            for left, right in combinations(units, 2):
                for lfmt in option_maps[left.key]:
                    for rfmt in option_maps[right.key]:
                        if lfmt == current[left.key] and rfmt == current[right.key]:
                            continue
                        trial = dict(current)
                        trial[left.key] = lfmt
                        trial[right.key] = rfmt
                        bits = _bits_total_for_choices(trial, unit_map, fixed_bits_total)
                        obj = objective_delta(trial, units, unary, pairwise)
                        score = _lagrangian_score(
                            obj,
                            bits,
                            total_params=float(total_params),
                            bpp_penalty=float(lagrangian_bpp_penalty),
                        )
                        if score + 1e-12 < current_score and (best is None or score < best[0]):
                            best = (score, obj, bits, trial)
            if best is None:
                break
            current_score, current_obj, current_bits, current = best
        return {
            "choices": current,
            "objective_delta": current_obj,
            "bits_total": current_bits,
            "bits_per_param": current_bits / float(total_params),
            "solver": "lagrangian_greedy_local",
            "lagrangian_objective": current_score,
            "lagrangian_bpp_penalty": float(lagrangian_bpp_penalty),
            "states_evaluated": None,
            "state_count": state_count,
        }
    if mode != "budget":
        raise ValueError(f"unknown sparse local refine solver_mode={solver_mode!r}")
    if exact_max_states > 0 and state_count <= int(exact_max_states):
        return _exact_local_refine(
            units,
            unary,
            pairwise,
            target_total_bits,
            fixed_bits_total,
            option_maps,
            initial_choices,
        )
    current = _initial_choice_map(units, option_maps, initial_choices)
    current_bits = _bits_total_for_choices(current, unit_map, fixed_bits_total)
    if current_bits > target_total_bits + 1e-6:
        raise ValueError("initial refinement state exceeds target budget")
    current_obj = objective_delta(current, units, unary, pairwise)

    for _pass in range(max_passes):
        best = None

        # Single-unit moves.
        for unit in units:
            for fmt in option_maps[unit.key]:
                if fmt == current[unit.key]:
                    continue
                trial = dict(current)
                trial[unit.key] = fmt
                bits = _bits_total_for_choices(trial, unit_map, fixed_bits_total)
                if bits > target_total_bits + 1e-6:
                    continue
                obj = objective_delta(trial, units, unary, pairwise)
                if obj + 1e-12 < current_obj and (best is None or obj < best[0]):
                    best = (obj, bits, trial)

        # Pair moves capture most of the useful interaction space while
        # remaining cheap for N≈16-32 critical units.
        for left, right in combinations(units, 2):
            for lfmt in option_maps[left.key]:
                for rfmt in option_maps[right.key]:
                    if lfmt == current[left.key] and rfmt == current[right.key]:
                        continue
                    trial = dict(current)
                    trial[left.key] = lfmt
                    trial[right.key] = rfmt
                    bits = _bits_total_for_choices(trial, unit_map, fixed_bits_total)
                    if bits > target_total_bits + 1e-6:
                        continue
                    obj = objective_delta(trial, units, unary, pairwise)
                    if obj + 1e-12 < current_obj and (best is None or obj < best[0]):
                        best = (obj, bits, trial)

        if best is None:
            break
        current_obj, current_bits, current = best

    return {
        "choices": current,
        "objective_delta": current_obj,
        "bits_total": current_bits,
        "bits_per_param": None,
        "solver": "greedy_local",
        "states_evaluated": None,
        "state_count": state_count,
    }
