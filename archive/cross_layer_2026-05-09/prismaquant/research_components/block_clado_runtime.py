"""Executable Block-CLADO research runtime.

This module ports the archived Block-CLADO measure/solve path onto current
PrismaQuant infrastructure:

* decision units come from :mod:`prismaquant.decision_units`;
* KL replay uses :func:`prismaquant.kl_measurement.measure_assignment_kl`;
* rendered weights flow through ``ProductionWeightCache`` when supplied.

The runtime remains a research component.  It can generate and validate
candidate assignments, but it does not promote them into export by itself.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from prismaquant import decision_units as du
from prismaquant import format_registry as fr
from prismaquant.allocator_solver import promote_fused, promote_moe_pair
from prismaquant.build_rtn_cache import cache_reference_log_probs, stage_multimodal
from prismaquant.calibration_data import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.kl_measurement import measure_assignment_kl
from prismaquant.layer_config import canonicalize_format
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.sensitivity_probe import load_calibration
from prismaquant.source_prefetch import prefetch_safetensors_checkpoint


SCHEMA = du.SCHEMA
SWEEP_SCHEMA = "prismaquant.block_clado.sweep.v1"
BUDGET_SCHEMA = "prismaquant.block_clado.budget.v1"
KNEEDLE_SCHEMA = "prismaquant.block_clado.kneedle.v1"
KNEEDLE_SUMMARY_SCHEMA = "prismaquant.block_clado.kneedle_summary.v1"
STRUCTURE_SCOPES = ("none", "runtime", "subblock", "layer")


@dataclass(frozen=True)
class BlockSolution:
    block_id: str
    assignment: dict[str, str]
    cost: float
    bits_total: float


@dataclass(frozen=True)
class GlobalSolveResult:
    assignment: dict[str, str]
    cost_total: float
    bits_total: float
    bpp: float
    lambda_used: float | None
    per_block_costs: dict[str, float]
    per_block_bits: dict[str, float]


def unit_is_bf16_pinned(
    unit: du.DecisionUnit,
    pin_to_bf16: Sequence[str] = ("lm_head",),
) -> bool:
    pin_tokens = [str(token) for token in pin_to_bf16 if str(token)]
    if not pin_tokens:
        return False
    for candidate in (unit.name, *unit.member_qnames):
        parts = str(candidate).split(".")
        if any(token in parts for token in pin_tokens):
            return True
    return False


def apply_bf16_pins_to_units(
    blocks: Mapping[str, Sequence[du.DecisionUnit]],
    singletons: Sequence[du.DecisionUnit],
    *,
    pin_to_bf16: Sequence[str] = ("lm_head",),
) -> tuple[dict[str, list[du.DecisionUnit]], list[du.DecisionUnit]]:
    def _pin(unit: du.DecisionUnit) -> du.DecisionUnit:
        if not unit_is_bf16_pinned(unit, pin_to_bf16):
            return unit
        bf16_options = tuple(
            opt for opt in unit.options
            if fr.canonical_format_name(opt.fmt) == "BF16"
        )
        if not bf16_options:
            return unit
        return du.DecisionUnit(
            name=unit.name,
            block_id=unit.block_id,
            member_qnames=unit.member_qnames,
            options=bf16_options,
        )

    return (
        {
            str(block_id): [_pin(unit) for unit in units]
            for block_id, units in blocks.items()
        },
        [_pin(unit) for unit in singletons],
    )


def enumerate_block_pairs(
    block_units: Sequence[du.DecisionUnit],
) -> list[tuple[str, str]]:
    names = [unit.name for unit in block_units]
    return [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]


def _pair_key(fmt_a: str, fmt_b: str) -> str:
    return f"{fr.canonical_format_name(fmt_a)}__{fr.canonical_format_name(fmt_b)}"


def units_and_pairs_to_payload(
    *,
    blocks: Mapping[str, Sequence[du.DecisionUnit]],
    singletons: Sequence[du.DecisionUnit],
    pairs_by_block: Mapping[str, Sequence[du.BlockPair]],
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out_blocks: dict[str, Any] = {}
    for block_id, units in blocks.items():
        unit_payload: dict[str, Any] = {}
        for unit in units:
            unit_payload[unit.name] = {
                "members": list(unit.member_qnames),
                "options": {
                    cost.fmt: {
                        "omega_ii": float(cost.omega_ii),
                        "bits_per_param": float(cost.bits_per_param),
                        "memory_bytes": int(cost.memory_bytes),
                    }
                    for cost in unit.options
                },
            }
        pair_payload = []
        for pair in pairs_by_block.get(str(block_id), ()):
            pair_payload.append({
                "unit_a": pair.unit_a,
                "unit_b": pair.unit_b,
                "omega_ij": {
                    _pair_key(fmt_a, fmt_b): float(value)
                    for (fmt_a, fmt_b), value in pair.omega_ij.items()
                },
            })
        out_blocks[str(block_id)] = {
            "units": unit_payload,
            "pairs": pair_payload,
        }

    singleton_payload: dict[str, Any] = {}
    for unit in singletons:
        singleton_payload[unit.name] = {
            "block_id": unit.block_id,
            "members": list(unit.member_qnames),
            "options": {
                cost.fmt: {
                    "omega_ii": float(cost.omega_ii),
                    "bits_per_param": float(cost.bits_per_param),
                    "memory_bytes": int(cost.memory_bytes),
                }
                for cost in unit.options
            },
        }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "blocks": out_blocks,
        "singletons": singleton_payload,
    }
    if meta:
        payload["meta"] = dict(meta)
    return payload


def expand_unit_assignment(
    unit_assignment: Mapping[str, str],
    units: Sequence[du.DecisionUnit],
    *,
    pin_to_bf16: Sequence[str] = ("lm_head",),
    omit_bf16_pinned: bool = True,
) -> dict[str, str]:
    by_name = {unit.name: unit for unit in units}
    out: dict[str, str] = {}
    for unit_name, fmt in unit_assignment.items():
        canonical = fr.canonical_format_name(str(fmt))
        unit = by_name.get(str(unit_name))
        if unit is None:
            if any(
                token in str(unit_name).split(".")
                for token in pin_to_bf16
                if str(token)
            ):
                if omit_bf16_pinned:
                    continue
                canonical = "BF16"
            out[str(unit_name)] = canonical
            continue
        if unit_is_bf16_pinned(unit, pin_to_bf16):
            if omit_bf16_pinned:
                continue
            canonical = "BF16"
        for member in unit.member_qnames:
            out[member] = canonical
    return out


def _bf16_assignment(units: Sequence[du.DecisionUnit]) -> dict[str, str]:
    return {
        member: "BF16"
        for unit in units
        for member in unit.member_qnames
    }


def _center_histogram(per_unit_center: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fmt in per_unit_center.values():
        counts[str(fmt)] = counts.get(str(fmt), 0) + 1
    return counts


def _format_sort_key(fmt: str) -> tuple[float, str]:
    canonical = fr.canonical_format_name(str(fmt))
    try:
        spec = fr.get_format(canonical)
        bits = float(spec.effective_bits)
    except Exception:
        bits = 16.0 if canonical == "BF16" else math.inf
    return bits, canonical


def _complete_format_rank(
    format_rank: Mapping[str, int] | None,
    formats: Iterable[str],
) -> dict[str, int]:
    """Return a canonical low-to-high precision rank for promotion guards."""

    needed = {fr.canonical_format_name(str(fmt)) for fmt in formats}
    out: dict[str, int] = {}
    if format_rank:
        for fmt, rank in format_rank.items():
            out[fr.canonical_format_name(str(fmt))] = int(rank)
    missing = sorted(needed.difference(out), key=_format_sort_key)
    if missing and out:
        all_names = set(out).union(needed)
        return {
            fmt: idx
            for idx, fmt in enumerate(sorted(all_names, key=_format_sort_key))
        }
    start = max(out.values(), default=-1) + 1
    for offset, fmt in enumerate(missing):
        out[fmt] = start + offset
    if not out:
        out["BF16"] = 0
    return out


def _format_rank_from_specs(
    formats: Sequence[fr.FormatSpec],
) -> dict[str, int]:
    specs = sorted(
        {fr.canonical_format_name(spec.name): spec for spec in formats}.values(),
        key=lambda spec: (float(spec.effective_bits), spec.name),
    )
    return {fr.canonical_format_name(spec.name): idx for idx, spec in enumerate(specs)}


def _format_rank_from_units(
    units: Sequence[du.DecisionUnit],
    format_rank: Mapping[str, int] | None = None,
) -> dict[str, int]:
    names = [
        opt.fmt
        for unit in units
        for opt in unit.options
    ]
    return _complete_format_rank(format_rank, names)


def _format_rank_from_payload(
    payload: Mapping[str, Any],
    format_rank: Mapping[str, int] | None = None,
) -> dict[str, int]:
    names: list[str] = []
    meta_formats = (payload.get("meta") or {}).get("formats") or ()
    names.extend(str(fmt) for fmt in meta_formats)
    for block in (payload.get("blocks") or {}).values():
        for unit in (block.get("units") or {}).values():
            names.extend(str(fmt) for fmt in (unit.get("options") or {}).keys())
    for unit in (payload.get("singletons") or {}).values():
        names.extend(str(fmt) for fmt in (unit.get("options") or {}).keys())
    return _complete_format_rank(format_rank, names)


def _option_by_format(unit: du.DecisionUnit) -> dict[str, du.FormatCost]:
    return {
        fr.canonical_format_name(opt.fmt): opt
        for opt in unit.options
    }


def _unit_param_count(unit: du.DecisionUnit) -> int:
    for opt in unit.options:
        if float(opt.bits_per_param) > 0:
            return int(round(float(opt.memory_bytes) * 8.0 / opt.bits_per_param))
    return 0


def _pair_lookup(
    pairs: Sequence[du.BlockPair],
) -> dict[tuple[str, str], du.BlockPair]:
    out: dict[tuple[str, str], du.BlockPair] = {}
    for pair in pairs:
        out[(pair.unit_a, pair.unit_b)] = pair
        out[(pair.unit_b, pair.unit_a)] = pair
    return out


def _pair_cost(
    pair: du.BlockPair | None,
    unit_a: str,
    fmt_a: str,
    unit_b: str,
    fmt_b: str,
) -> float:
    if pair is None:
        return 0.0
    key = (
        fr.canonical_format_name(fmt_a),
        fr.canonical_format_name(fmt_b),
    )
    if pair.unit_a == unit_a and pair.unit_b == unit_b:
        return float(pair.omega_ij.get(key, 0.0))
    rev = (key[1], key[0])
    return float(pair.omega_ij.get(rev, 0.0))


def _qname_subblock(qname: str) -> str:
    block_id = du.block_id_from_qname(qname)
    if not block_id or block_id == qname or not qname.startswith(block_id + "."):
        return qname
    tail = qname[len(block_id) + 1:]
    if tail.startswith("self_attn."):
        return f"{block_id}.self_attn"
    if tail.startswith("linear_attn."):
        return f"{block_id}.linear_attn"
    if tail.startswith("mlp.") or ".experts." in tail or ".moe." in tail:
        return f"{block_id}.mlp"
    return qname


def _runtime_structure_key(unit: du.DecisionUnit, profile=None) -> str:
    group_keys: list[str] = []
    fused_fn = getattr(profile, "fused_sibling_group", None)
    packed_fn = getattr(profile, "packed_expert_format_group", None)
    for member in unit.member_qnames:
        group_key = None
        if callable(fused_fn):
            try:
                group_key = fused_fn(member)
            except Exception:
                group_key = None
        if group_key is None and callable(packed_fn):
            try:
                group_key = packed_fn(member)
            except Exception:
                group_key = None
        if group_key is not None:
            group_keys.append(str(group_key))
    if not group_keys:
        return unit.name
    unique = sorted(set(group_keys))
    return unique[0] if len(unique) == 1 else unit.name


def _structure_key(
    unit: du.DecisionUnit,
    *,
    scope: str,
    profile=None,
) -> str:
    if scope == "none":
        return unit.name
    if scope == "runtime":
        return _runtime_structure_key(unit, profile)
    if scope == "subblock":
        subblocks = {_qname_subblock(member) for member in unit.member_qnames}
        return next(iter(subblocks)) if len(subblocks) == 1 else unit.name
    if scope == "layer":
        blocks = {du.block_id_from_qname(member) for member in unit.member_qnames}
        blocks.discard("")
        return next(iter(blocks)) if len(blocks) == 1 else unit.name
    raise ValueError(
        f"unsupported CLADO structure scope {scope!r}; "
        f"expected one of {', '.join(STRUCTURE_SCOPES)}"
    )


def _merge_structural_units(
    *,
    group_name: str,
    block_id: str,
    units: Sequence[du.DecisionUnit],
    pair_by_unit: Mapping[tuple[str, str], du.BlockPair],
) -> du.DecisionUnit:
    if len(units) == 1 and units[0].name == group_name:
        return units[0]

    option_maps = {unit.name: _option_by_format(unit) for unit in units}
    shared_formats = set.intersection(
        *(set(options) for options in option_maps.values())
    )
    if not shared_formats:
        raise ValueError(
            f"cannot coarsen CLADO structure group {group_name!r}; "
            "member units have no common legal format"
        )

    n_params = sum(_unit_param_count(unit) for unit in units)
    merged_options: list[du.FormatCost] = []
    for fmt in sorted(shared_formats, key=_format_sort_key):
        omega = 0.0
        memory_bytes = 0
        for unit in units:
            opt = option_maps[unit.name][fmt]
            omega += float(opt.omega_ii)
            memory_bytes += int(opt.memory_bytes)
        for idx, unit_a in enumerate(units):
            for unit_b in units[idx + 1:]:
                omega += _pair_cost(
                    pair_by_unit.get((unit_a.name, unit_b.name)),
                    unit_a.name,
                    fmt,
                    unit_b.name,
                    fmt,
                )
        merged_options.append(du.FormatCost(
            fmt=fmt,
            omega_ii=float(omega),
            bits_per_param=(
                8.0 * float(memory_bytes) / max(float(n_params), 1.0)
            ),
            memory_bytes=int(memory_bytes),
        ))

    members = tuple(sorted({
        member
        for unit in units
        for member in unit.member_qnames
    }))
    return du.DecisionUnit(
        name=group_name,
        block_id=block_id,
        member_qnames=members,
        options=tuple(merged_options),
    )


def _coarsen_block_units(
    *,
    block_id: str,
    units: Sequence[du.DecisionUnit],
    pairs: Sequence[du.BlockPair],
    scope: str,
    profile=None,
) -> tuple[list[du.DecisionUnit], list[du.BlockPair], dict[str, Any]]:
    groups: dict[str, list[du.DecisionUnit]] = {}
    group_order: list[str] = []
    for unit in units:
        key = _structure_key(unit, scope=scope, profile=profile)
        if key not in groups:
            group_order.append(key)
            groups[key] = []
        groups[key].append(unit)

    pair_by_unit = _pair_lookup(pairs)
    unit_to_group = {
        unit.name: key
        for key, grouped_units in groups.items()
        for unit in grouped_units
    }
    merged_units = [
        _merge_structural_units(
            group_name=group_name,
            block_id=block_id,
            units=groups[group_name],
            pair_by_unit=pair_by_unit,
        )
        for group_name in group_order
    ]
    merged_by_name = {unit.name: unit for unit in merged_units}

    merged_pairs: list[du.BlockPair] = []
    for idx, unit_a in enumerate(merged_units):
        child_a = groups[unit_a.name]
        for unit_b in merged_units[idx + 1:]:
            child_b = groups[unit_b.name]
            omega_ij: dict[tuple[str, str], float] = {}
            for opt_a in unit_a.options:
                for opt_b in unit_b.options:
                    value = 0.0
                    for child_unit_a in child_a:
                        for child_unit_b in child_b:
                            value += _pair_cost(
                                pair_by_unit.get((
                                    child_unit_a.name,
                                    child_unit_b.name,
                                )),
                                child_unit_a.name,
                                opt_a.fmt,
                                child_unit_b.name,
                                opt_b.fmt,
                            )
                    omega_ij[(opt_a.fmt, opt_b.fmt)] = float(value)
            merged_pairs.append(du.BlockPair(
                unit_a=unit_a.name,
                unit_b=unit_b.name,
                block_id=block_id,
                omega_ij=omega_ij,
            ))

    merged_group_count = sum(1 for grouped in groups.values() if len(grouped) > 1)
    changed = (
        len(merged_units) != len(units)
        or any(unit_to_group.get(unit.name) != unit.name for unit in units)
    )
    return merged_units, merged_pairs, {
        "input_units": len(units),
        "output_units": len(merged_units),
        "merged_groups": int(merged_group_count),
        "changed": bool(changed),
        "groups": {
            group_name: [unit.name for unit in grouped_units]
            for group_name, grouped_units in groups.items()
            if len(grouped_units) > 1
        },
        "output_unit_names": sorted(merged_by_name),
    }


def coarsen_payload_to_structure(
    payload: Mapping[str, Any],
    *,
    scope: str = "runtime",
    profile=None,
) -> dict[str, Any]:
    """Lift a CLADO payload onto profile/model structural units.

    Fresh CLADO measurement already discovers profile-backed decision units.
    Older archived payloads may still expose raw Linears.  Coarsening lets the
    solver optimize over the same structural units used by the current model
    graph: runtime-coupled groups, attention/MLP subblocks, or whole layers.
    The pairwise surrogate is preserved by folding intra-group pair terms into
    the merged unary cost and summing inter-group pair terms.
    """

    scope = str(scope or "none")
    if scope not in STRUCTURE_SCOPES:
        raise ValueError(
            f"unsupported CLADO structure scope {scope!r}; "
            f"expected one of {', '.join(STRUCTURE_SCOPES)}"
        )
    if scope == "none":
        return dict(payload)

    blocks, singletons, pairs_by_block = du.parse_payload(payload)
    out_blocks: dict[str, list[du.DecisionUnit]] = {}
    out_pairs: dict[str, list[du.BlockPair]] = {}
    block_reports: dict[str, Any] = {}
    changed = False
    for block_id, units in blocks.items():
        merged_units, merged_pairs, report = _coarsen_block_units(
            block_id=str(block_id),
            units=units,
            pairs=pairs_by_block.get(str(block_id), ()),
            scope=scope,
            profile=profile,
        )
        out_blocks[str(block_id)] = merged_units
        out_pairs[str(block_id)] = merged_pairs
        block_reports[str(block_id)] = report
        changed = changed or bool(report["changed"])

    out_singletons = list(singletons)
    meta = dict(payload.get("meta") or {})
    meta["structure_scope"] = scope
    meta["structure_coarsened"] = bool(changed)
    meta["structure_coarsening"] = {
        "scope": scope,
        "profile": getattr(profile, "name", None),
        "input_block_count": len(blocks),
        "input_singleton_count": len(singletons),
        "output_block_count": len(out_blocks),
        "output_singleton_count": len(out_singletons),
        "input_unit_count": sum(len(units) for units in blocks.values()) + len(singletons),
        "output_unit_count": sum(len(units) for units in out_blocks.values()) + len(out_singletons),
        "blocks": block_reports,
    }
    return units_and_pairs_to_payload(
        blocks=out_blocks,
        singletons=out_singletons,
        pairs_by_block=out_pairs,
        meta=meta,
    )


def _payload_for_structure_scope(
    payload: Mapping[str, Any],
    *,
    structure_scope: str = "none",
    profile=None,
) -> dict[str, Any]:
    if str(structure_scope or "none") == "none":
        return dict(payload)
    return coarsen_payload_to_structure(
        payload,
        scope=structure_scope,
        profile=profile,
    )


def _payload_unit_names(payload: Mapping[str, Any]) -> set[str]:
    blocks, singletons, _pairs = du.parse_payload(payload)
    names = {
        unit.name
        for unit_list in blocks.values()
        for unit in unit_list
    }
    names.update(unit.name for unit in singletons)
    return names


def legalize_assignment_for_runtime(
    assignment: Mapping[str, str],
    *,
    profile=None,
    format_rank: Mapping[str, int] | None = None,
) -> dict[str, str]:
    """Apply the same serving-format coupling guards used by the allocator.

    CLADO proposals are measured and exported as complete assignments, so a
    single Linear move must not create a mixed q/k/v, gate/up, or packed-MoE
    serving group.  This helper canonicalizes formats and promotes coupled
    members to the highest-ranked format when needed.
    """

    out = {
        str(name): fr.canonical_format_name(str(fmt))
        for name, fmt in assignment.items()
    }
    if profile is None:
        return out
    rank = _complete_format_rank(format_rank, out.values())
    out = promote_fused(out, rank, profile=profile)
    out = promote_moe_pair(out, rank, profile=profile)
    return out


def _expand_unit_assignment_members(
    unit_assignment: Mapping[str, str],
    units: Sequence[du.DecisionUnit],
) -> dict[str, str]:
    by_name = {unit.name: unit for unit in units}
    out: dict[str, str] = {}
    for unit_name, fmt in unit_assignment.items():
        canonical = fr.canonical_format_name(str(fmt))
        unit = by_name.get(str(unit_name))
        if unit is None:
            out[str(unit_name)] = canonical
            continue
        for member in unit.member_qnames:
            out[member] = canonical
    return out


def _unit_assignment_is_runtime_legal(
    unit_assignment: Mapping[str, str],
    units: Sequence[du.DecisionUnit],
    *,
    profile=None,
    format_rank: Mapping[str, int] | None = None,
) -> bool:
    if profile is None:
        return True
    expanded = _expand_unit_assignment_members(unit_assignment, units)
    legalized = legalize_assignment_for_runtime(
        expanded,
        profile=profile,
        format_rank=format_rank,
    )
    return legalized == expanded


def _center_assignment_for_units(
    units_by_name: Mapping[str, du.DecisionUnit],
    singletons_by_name: Mapping[str, du.DecisionUnit],
    center_assignment: Mapping[str, str] | None,
    pin_to_bf16: Sequence[str] = ("lm_head",),
) -> tuple[dict[str, str], dict[str, str]]:
    base: dict[str, str] = {}
    per_unit: dict[str, str] = {}
    all_units = list(units_by_name.values()) + list(singletons_by_name.values())
    for unit in all_units:
        chosen: str | None = None
        if unit_is_bf16_pinned(unit, pin_to_bf16):
            chosen = "BF16"
        elif center_assignment:
            for member in unit.member_qnames:
                if member in center_assignment:
                    chosen = canonicalize_format(center_assignment[member])
                    break
        if chosen is None:
            chosen = "BF16"
        per_unit[unit.name] = chosen
        for member in unit.member_qnames:
            base[member] = chosen
    return base, per_unit


def _override_assignment(
    base: Mapping[str, str],
    unit: du.DecisionUnit,
    fmt: str,
) -> dict[str, str]:
    out = dict(base)
    canonical = fr.canonical_format_name(str(fmt))
    for member in unit.member_qnames:
        out[member] = canonical
    return out


def _override_pair(
    base: Mapping[str, str],
    unit_a: du.DecisionUnit,
    fmt_a: str,
    unit_b: du.DecisionUnit,
    fmt_b: str,
) -> dict[str, str]:
    out = dict(base)
    canonical_a = fr.canonical_format_name(str(fmt_a))
    canonical_b = fr.canonical_format_name(str(fmt_b))
    for member in unit_a.member_qnames:
        out[member] = canonical_a
    for member in unit_b.member_qnames:
        out[member] = canonical_b
    return out


def _prefetch_assignment_if_available(
    production_weight_cache,
    assignment: Mapping[str, str],
    *,
    require: bool,
    max_workers: int = 4,
) -> None:
    if production_weight_cache is None:
        return
    prefetch = getattr(production_weight_cache, "prefetch_assignment", None)
    if callable(prefetch):
        prefetch(
            assignment,
            require=bool(require),
            max_workers=max(1, int(max_workers)),
            progress=False,
        )


def collect_block_clado(
    model,
    calib_ids: torch.Tensor,
    formats: Sequence[fr.FormatSpec],
    *,
    profile=None,
    target_profile: str | None = None,
    work_root: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    skip_pairs: bool = False,
    center_assignment: Mapping[str, str] | None = None,
    use_frozen_weight_cache: bool = False,
    production_weight_cache=None,
    prefetch_production_cache: bool = True,
    production_cache_prefetch_workers: int = 4,
    pin_to_bf16: Sequence[str] = ("lm_head",),
    kl_scope: str = "last_token",
    include_activation_quant: bool = True,
    use_cuda_graphs: bool | None = None,
) -> dict[str, Any]:
    """Measure a Block-CLADO payload using live PrismaQuant APIs."""

    if not isinstance(calib_ids, torch.Tensor) or calib_ids.dim() != 2:
        raise ValueError("calib_ids must be a 2D tensor [samples, seqlen]")

    spec_by_name = {fr.canonical_format_name(spec.name): spec for spec in formats}
    if "BF16" not in spec_by_name:
        spec_by_name["BF16"] = fr.get_format("BF16")
    specs_sorted = [spec_by_name[name] for name in sorted(spec_by_name)]
    format_rank = _format_rank_from_specs(specs_sorted)
    if center_assignment is not None:
        center_assignment = legalize_assignment_for_runtime(
            center_assignment,
            profile=profile,
            format_rank=format_rank,
        )

    blocks, singletons, n_params_by_unit = du.discover_units(
        model,
        profile,
        specs_sorted,
        target_profile=target_profile,
    )
    blocks, singletons = apply_bf16_pins_to_units(
        blocks,
        singletons,
        pin_to_bf16=pin_to_bf16,
    )
    units_by_name = {
        unit.name: unit
        for unit_list in blocks.values()
        for unit in unit_list
    }
    singletons_by_name = {unit.name: unit for unit in singletons}
    all_units = list(units_by_name.values()) + list(singletons_by_name.values())
    if not all_units:
        raise RuntimeError("no quantizable units discovered in model")

    device = next(model.parameters()).device
    if work_root is None:
        work_root_path = Path(tempfile.mkdtemp(prefix="prismaquant_block_clado_"))
        cleanup_work_root = True
    else:
        work_root_path = Path(work_root)
        work_root_path.mkdir(parents=True, exist_ok=True)
        cleanup_work_root = False

    start = time.time()
    try:
        ref_log_probs = cache_reference_log_probs(
            model,
            calib_ids,
            device,
            kl_scope=kl_scope,
        )
        base, per_unit_center = _center_assignment_for_units(
            units_by_name,
            singletons_by_name,
            center_assignment,
            pin_to_bf16,
        )
        if production_weight_cache is not None and prefetch_production_cache:
            _prefetch_assignment_if_available(
                production_weight_cache,
                base,
                require=True,
                max_workers=production_cache_prefetch_workers,
            )
        center_kl = 0.0
        if center_assignment is not None and any(
            fmt != "BF16" for fmt in per_unit_center.values()
        ):
            center_kl = float(measure_assignment_kl(
                model,
                base,
                calib_ids,
                ref_log_probs,
                work_root=work_root_path,
                profile=profile,
                use_frozen_weight_cache=use_frozen_weight_cache,
                production_weight_cache=production_weight_cache,
                rng_seed=0,
                kl_scope=kl_scope,
                include_activation_quant=include_activation_quant,
                use_cuda_graphs=use_cuda_graphs,
            ))
            if progress_callback is not None:
                progress_callback({"event": "center_kl", "kl": center_kl})

        omega_ii: dict[tuple[str, str], float] = {}
        n_unary = sum(
            sum(
                1
                for opt in unit.options
                if opt.fmt != per_unit_center.get(unit.name, "BF16")
            )
            for unit in all_units
        )
        unary_done = 0
        for unit in all_units:
            center_fmt = per_unit_center.get(unit.name, "BF16")
            for opt in unit.options:
                if opt.fmt == center_fmt:
                    omega_ii[(unit.name, opt.fmt)] = 0.0
                    continue
                assignment = _override_assignment(base, unit, opt.fmt)
                if production_weight_cache is not None and prefetch_production_cache:
                    _prefetch_assignment_if_available(
                        production_weight_cache,
                        assignment,
                        require=True,
                        max_workers=production_cache_prefetch_workers,
                    )
                kl = measure_assignment_kl(
                    model,
                    assignment,
                    calib_ids,
                    ref_log_probs,
                    work_root=work_root_path,
                    profile=profile,
                    use_frozen_weight_cache=use_frozen_weight_cache,
                    production_weight_cache=production_weight_cache,
                    rng_seed=0,
                    kl_scope=kl_scope,
                    include_activation_quant=include_activation_quant,
                    use_cuda_graphs=use_cuda_graphs,
                )
                omega_ii[(unit.name, opt.fmt)] = float(kl) - center_kl
                unary_done += 1
                if progress_callback is not None:
                    progress_callback({
                        "event": "unary_done",
                        "unit": unit.name,
                        "format": opt.fmt,
                        "kl": omega_ii[(unit.name, opt.fmt)],
                        "completed": unary_done,
                        "total": n_unary,
                    })

        measured_blocks: dict[str, list[du.DecisionUnit]] = {}
        for block_id, unit_list in blocks.items():
            measured_units = []
            for unit in unit_list:
                measured_units.append(du.DecisionUnit(
                    name=unit.name,
                    block_id=unit.block_id,
                    member_qnames=unit.member_qnames,
                    options=tuple(
                        du.FormatCost(
                            fmt=opt.fmt,
                            omega_ii=float(omega_ii[(unit.name, opt.fmt)]),
                            bits_per_param=opt.bits_per_param,
                            memory_bytes=opt.memory_bytes,
                        )
                        for opt in unit.options
                    ),
                ))
            measured_blocks[str(block_id)] = measured_units
        measured_singletons = [
            du.DecisionUnit(
                name=unit.name,
                block_id=unit.block_id,
                member_qnames=unit.member_qnames,
                options=tuple(
                    du.FormatCost(
                        fmt=opt.fmt,
                        omega_ii=float(omega_ii[(unit.name, opt.fmt)]),
                        bits_per_param=opt.bits_per_param,
                        memory_bytes=opt.memory_bytes,
                    )
                    for opt in unit.options
                ),
            )
            for unit in singletons
        ]
        measured_units_by_name = {
            unit.name: unit
            for unit_list in measured_blocks.values()
            for unit in unit_list
        }

        pairs_by_block: dict[str, list[du.BlockPair]] = {}
        actual_pair_measurements = 0
        if not skip_pairs:
            n_pairs_total = 0
            for unit_list in measured_blocks.values():
                for unit_a_name, unit_b_name in enumerate_block_pairs(unit_list):
                    unit_a = measured_units_by_name[unit_a_name]
                    unit_b = measured_units_by_name[unit_b_name]
                    center_a = per_unit_center.get(unit_a.name, "BF16")
                    center_b = per_unit_center.get(unit_b.name, "BF16")
                    for opt_a in unit_a.options:
                        if opt_a.fmt == center_a:
                            continue
                        for opt_b in unit_b.options:
                            if opt_b.fmt == center_b:
                                continue
                            n_pairs_total += 1

            pair_done = 0
            for block_id, unit_list in measured_blocks.items():
                pair_list: list[du.BlockPair] = []
                for unit_a_name, unit_b_name in enumerate_block_pairs(unit_list):
                    unit_a = measured_units_by_name[unit_a_name]
                    unit_b = measured_units_by_name[unit_b_name]
                    center_a = per_unit_center.get(unit_a.name, "BF16")
                    center_b = per_unit_center.get(unit_b.name, "BF16")
                    omega_ij: dict[tuple[str, str], float] = {}
                    for opt_a in unit_a.options:
                        for opt_b in unit_b.options:
                            if opt_a.fmt == center_a or opt_b.fmt == center_b:
                                omega_ij[(opt_a.fmt, opt_b.fmt)] = 0.0
                                continue
                            assignment = _override_pair(
                                base,
                                unit_a,
                                opt_a.fmt,
                                unit_b,
                                opt_b.fmt,
                            )
                            if (
                                production_weight_cache is not None
                                and prefetch_production_cache
                            ):
                                _prefetch_assignment_if_available(
                                    production_weight_cache,
                                    assignment,
                                    require=True,
                                    max_workers=production_cache_prefetch_workers,
                                )
                            kl_ab = measure_assignment_kl(
                                model,
                                assignment,
                                calib_ids,
                                ref_log_probs,
                                work_root=work_root_path,
                                profile=profile,
                                use_frozen_weight_cache=use_frozen_weight_cache,
                                production_weight_cache=production_weight_cache,
                                rng_seed=0,
                                kl_scope=kl_scope,
                                include_activation_quant=include_activation_quant,
                                use_cuda_graphs=use_cuda_graphs,
                            )
                            omega_a = float(omega_ii[(unit_a.name, opt_a.fmt)])
                            omega_b = float(omega_ii[(unit_b.name, opt_b.fmt)])
                            omega_ij[(opt_a.fmt, opt_b.fmt)] = (
                                float(kl_ab) - omega_a - omega_b - center_kl
                            )
                            pair_done += 1
                            actual_pair_measurements += 1
                            if progress_callback is not None:
                                progress_callback({
                                    "event": "pair_done",
                                    "block_id": block_id,
                                    "unit_a": unit_a.name,
                                    "unit_b": unit_b.name,
                                    "fmt_a": opt_a.fmt,
                                    "fmt_b": opt_b.fmt,
                                    "kl_ab": float(kl_ab),
                                    "omega_ij": omega_ij[(opt_a.fmt, opt_b.fmt)],
                                    "completed": pair_done,
                                    "total": n_pairs_total,
                                })
                    pair_list.append(du.BlockPair(
                        unit_a=unit_a.name,
                        unit_b=unit_b.name,
                        block_id=str(block_id),
                        omega_ij=omega_ij,
                    ))
                pairs_by_block[str(block_id)] = pair_list
        else:
            for block_id in measured_blocks:
                pairs_by_block[str(block_id)] = []

        elapsed = time.time() - start
        meta = {
            "elapsed_seconds": float(elapsed),
            "n_calib_samples": int(calib_ids.size(0)),
            "calib_seqlen": int(calib_ids.size(1)),
            "formats": [spec.name for spec in specs_sorted],
            "target_profile": target_profile,
            "objective_metric": "teacher_forward_kl_four_term",
            "loss": "teacher_student_kl",
            "kl_scope": kl_scope,
            "block_count": len(measured_blocks),
            "singleton_count": len(measured_singletons),
            "n_unary_measurements": int(n_unary),
            "n_pair_measurements": int(actual_pair_measurements),
            "skip_pairs": bool(skip_pairs),
            "centered": bool(center_assignment is not None),
            "center_kl": float(center_kl),
            "center_format_histogram": _center_histogram(per_unit_center),
            "format_rank": dict(format_rank),
            "model_profile": getattr(profile, "name", None),
            "production_cache_used": bool(production_weight_cache is not None),
            "production_cache_prefetch": bool(
                production_weight_cache is not None and prefetch_production_cache
            ),
            "include_activation_quant": bool(include_activation_quant),
            "n_params_by_unit": dict(n_params_by_unit),
        }
        return units_and_pairs_to_payload(
            blocks=measured_blocks,
            singletons=measured_singletons,
            pairs_by_block=pairs_by_block,
            meta=meta,
        )
    finally:
        if cleanup_work_root:
            shutil.rmtree(work_root_path, ignore_errors=True)


def score_block_assignment(
    units: Sequence[du.DecisionUnit],
    assignment: Mapping[str, str],
    pairs: Sequence[du.BlockPair],
) -> tuple[float, float]:
    by_unit_format = {
        unit.name: {opt.fmt: opt for opt in unit.options}
        for unit in units
    }
    cost = 0.0
    bits = 0.0
    for unit in units:
        opt = by_unit_format[unit.name][assignment[unit.name]]
        cost += opt.omega_ii
        bits += opt.bits_total
    for pair in pairs:
        fmt_a = assignment[pair.unit_a]
        fmt_b = assignment[pair.unit_b]
        omega = pair.omega_ij.get((fmt_a, fmt_b))
        if omega is None:
            omega = pair.omega_ij.get((fmt_b, fmt_a), 0.0)
        cost += float(omega)
    return float(cost), float(bits)


def enumerate_block_states(
    units: Sequence[du.DecisionUnit],
    pairs: Sequence[du.BlockPair],
    *,
    max_states: int | None = 65_536,
    profile=None,
    format_rank: Mapping[str, int] | None = None,
) -> list[BlockSolution]:
    if not units:
        return []
    runtime_format_rank = _format_rank_from_units(units, format_rank)
    total_combinations = 1
    for unit in units:
        total_combinations *= max(len(unit.options), 1)
    if max_states is not None and total_combinations > int(max_states):
        raise ValueError(
            f"block has {total_combinations} format tuples > max {max_states}; "
            "reduce format menu or split the block"
        )

    block_id = units[0].block_id
    all_states: list[BlockSolution] = []

    def recurse(idx: int, partial: dict[str, str]) -> None:
        if idx == len(units):
            if not _unit_assignment_is_runtime_legal(
                partial,
                units,
                profile=profile,
                format_rank=runtime_format_rank,
            ):
                return
            cost, bits = score_block_assignment(units, partial, pairs)
            all_states.append(BlockSolution(
                block_id=block_id,
                assignment=dict(partial),
                cost=cost,
                bits_total=bits,
            ))
            return
        unit = units[idx]
        for option in unit.options:
            partial[unit.name] = option.fmt
            recurse(idx + 1, partial)
            del partial[unit.name]

    recurse(0, {})
    if profile is not None and not all_states:
        raise ValueError(
            f"block {block_id!r} has no runtime-legal CLADO states after "
            "applying fused-sibling and packed-expert format constraints"
        )
    return _pareto_states(all_states)


def enumerate_singleton_states(
    unit: du.DecisionUnit,
    *,
    profile=None,
    format_rank: Mapping[str, int] | None = None,
) -> list[BlockSolution]:
    runtime_format_rank = _format_rank_from_units((unit,), format_rank)
    states: list[BlockSolution] = []
    for option in unit.options:
        assignment = {unit.name: option.fmt}
        if not _unit_assignment_is_runtime_legal(
            assignment,
            (unit,),
            profile=profile,
            format_rank=runtime_format_rank,
        ):
            continue
        states.append(BlockSolution(
            block_id=unit.block_id,
            assignment=assignment,
            cost=float(option.omega_ii),
            bits_total=float(option.bits_total),
        ))
    if profile is not None and not states:
        raise ValueError(
            f"singleton {unit.name!r} has no runtime-legal CLADO states after "
            "applying format constraints"
        )
    return _pareto_states(states)


def _pareto_states(states: Sequence[BlockSolution]) -> list[BlockSolution]:
    ordered = sorted(states, key=lambda state: (state.bits_total, state.cost))
    pareto: list[BlockSolution] = []
    best_cost = math.inf
    for state in ordered:
        if state.cost < best_cost - 1e-12:
            pareto.append(state)
            best_cost = state.cost
    return pareto


def build_block_states(
    payload: Mapping[str, Any],
    *,
    max_states_per_block: int | None = 65_536,
    profile=None,
    format_rank: Mapping[str, int] | None = None,
) -> dict[str, list[BlockSolution]]:
    blocks, singletons, pairs_by_block = du.parse_payload(payload)
    runtime_format_rank = _format_rank_from_payload(payload, format_rank)
    block_states: dict[str, list[BlockSolution]] = {}
    for block_id, units in blocks.items():
        block_states[str(block_id)] = enumerate_block_states(
            units,
            pairs_by_block.get(str(block_id), ()),
            max_states=max_states_per_block,
            profile=profile,
            format_rank=runtime_format_rank,
        )
    for unit in singletons:
        block_states[unit.block_id] = enumerate_singleton_states(
            unit,
            profile=profile,
            format_rank=runtime_format_rank,
        )
    return block_states


def _per_block_lambda_pick(
    block_states: Sequence[BlockSolution],
    lambda_penalty: float,
) -> BlockSolution:
    best: BlockSolution | None = None
    best_score = math.inf
    for state in block_states:
        score = state.cost + float(lambda_penalty) * state.bits_total
        if score < best_score - 1e-12:
            best = state
            best_score = score
    if best is None:
        raise ValueError("empty block state set")
    return best


def solve_lagrangian(
    block_states: Mapping[str, Sequence[BlockSolution]],
    *,
    lambda_penalty: float,
) -> GlobalSolveResult:
    assignment: dict[str, str] = {}
    cost_total = 0.0
    bits_total = 0.0
    per_block_costs: dict[str, float] = {}
    per_block_bits: dict[str, float] = {}
    for block_id, states in block_states.items():
        pick = _per_block_lambda_pick(states, lambda_penalty)
        assignment.update(pick.assignment)
        cost_total += pick.cost
        bits_total += pick.bits_total
        per_block_costs[str(block_id)] = pick.cost
        per_block_bits[str(block_id)] = pick.bits_total
    return GlobalSolveResult(
        assignment=assignment,
        cost_total=float(cost_total),
        bits_total=float(bits_total),
        bpp=0.0,
        lambda_used=float(lambda_penalty),
        per_block_costs=per_block_costs,
        per_block_bits=per_block_bits,
    )


def lambda_sweep(
    block_states: Mapping[str, Sequence[BlockSolution]],
    *,
    lambda_min: float = 1e-12,
    lambda_max: float = 1e-3,
    n_lambdas: int = 41,
    log_scale: bool = True,
) -> list[GlobalSolveResult]:
    if n_lambdas <= 0:
        raise ValueError("n_lambdas must be positive")
    if lambda_min < 0 or lambda_max <= lambda_min:
        raise ValueError("require 0 <= lambda_min < lambda_max")
    if log_scale and lambda_min <= 0:
        lambda_min = max(lambda_min, 1e-30)
    lambdas: list[float] = []
    if log_scale:
        log_lo = math.log10(lambda_min)
        log_hi = math.log10(lambda_max)
        for idx in range(n_lambdas):
            t = idx / max(n_lambdas - 1, 1)
            lambdas.append(10.0 ** (log_lo + t * (log_hi - log_lo)))
    else:
        for idx in range(n_lambdas):
            t = idx / max(n_lambdas - 1, 1)
            lambdas.append(lambda_min + t * (lambda_max - lambda_min))

    seen: set[tuple[float, float]] = set()
    results: list[GlobalSolveResult] = []
    for lam in lambdas:
        result = solve_lagrangian(block_states, lambda_penalty=lam)
        key = (round(result.bits_total, 6), round(result.cost_total, 9))
        if key in seen:
            continue
        seen.add(key)
        results.append(result)
    return sorted(results, key=lambda result: (result.bits_total, result.cost_total))


def solve_budget(
    block_states: Mapping[str, Sequence[BlockSolution]],
    *,
    bits_budget: float,
    bit_precision_bits: float = 1.0,
) -> GlobalSolveResult | None:
    if not block_states:
        raise ValueError("no block states supplied")
    if bits_budget <= 0:
        raise ValueError("bits_budget must be positive")
    bits_min = sum(min(s.bits_total for s in states) for states in block_states.values())
    if bits_min > bits_budget + 1e-6:
        return None

    bin_width = max(float(bit_precision_bits), 1e-9)
    bins = max(int(math.ceil(bits_budget / bin_width)) + 1, 2)
    inf = math.inf
    block_ids = list(block_states.keys())
    state_lists = [list(block_states[bid]) for bid in block_ids]
    dp_prev = [inf] * bins
    dp_prev[0] = 0.0
    backpointers: list[list[tuple[int, int] | None]] = []
    for state_list in state_lists:
        next_cost = [inf] * bins
        bp_layer: list[tuple[int, int] | None] = [None] * bins
        for prev_bin, prev_cost in enumerate(dp_prev):
            if prev_cost == inf:
                continue
            for choice_idx, state in enumerate(state_list):
                next_bin = prev_bin + int(math.floor(state.bits_total / bin_width))
                if next_bin >= bins:
                    continue
                candidate = prev_cost + state.cost
                if candidate < next_cost[next_bin] - 1e-12:
                    next_cost[next_bin] = candidate
                    bp_layer[next_bin] = (prev_bin, choice_idx)
        dp_prev = next_cost
        backpointers.append(bp_layer)

    best_bin = -1
    best_cost = inf
    for idx, cost in enumerate(dp_prev):
        if cost < best_cost - 1e-12:
            best_bin = idx
            best_cost = cost
    if best_bin < 0 or best_cost == inf:
        return None

    assignment: dict[str, str] = {}
    cost_total = 0.0
    bits_total = 0.0
    per_block_costs: dict[str, float] = {}
    per_block_bits: dict[str, float] = {}
    cur_bin = best_bin
    for level in range(len(state_lists) - 1, -1, -1):
        entry = backpointers[level][cur_bin]
        if entry is None:
            return None
        prev_bin, choice_idx = entry
        state = state_lists[level][choice_idx]
        block_id = block_ids[level]
        assignment.update(state.assignment)
        cost_total += state.cost
        bits_total += state.bits_total
        per_block_costs[str(block_id)] = state.cost
        per_block_bits[str(block_id)] = state.bits_total
        cur_bin = prev_bin

    return GlobalSolveResult(
        assignment=assignment,
        cost_total=float(cost_total),
        bits_total=float(bits_total),
        bpp=0.0,
        lambda_used=None,
        per_block_costs=per_block_costs,
        per_block_bits=per_block_bits,
    )


def fill_bpp(result: GlobalSolveResult, total_params: int) -> GlobalSolveResult:
    if total_params <= 0:
        return result
    return GlobalSolveResult(
        assignment=result.assignment,
        cost_total=result.cost_total,
        bits_total=result.bits_total,
        bpp=result.bits_total / float(total_params),
        lambda_used=result.lambda_used,
        per_block_costs=result.per_block_costs,
        per_block_bits=result.per_block_bits,
    )


def _normalise(values: Sequence[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) <= 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def kneedle_pick(points: Sequence[tuple[float, float]]) -> tuple[int, float, bool]:
    if len(points) < 3:
        return len(points) // 2, 0.0, True
    pts = sorted(points, key=lambda xy: xy[0])
    xs = _normalise([p[0] for p in pts])
    ys_raw = _normalise([p[1] for p in pts])
    ys = [1.0 - value for value in ys_raw]
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    denom = max(((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5, 1e-12)
    best_score = -math.inf
    best_idx = 0
    for idx, (x, y) in enumerate(zip(xs, ys)):
        score = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        if score > best_score:
            best_score = score
            best_idx = idx
    endpoint = best_idx in {0, len(pts) - 1}
    if endpoint:
        best_idx = len(pts) // 2
    sorted_to_orig = [points.index(point) for point in pts]
    return sorted_to_orig[best_idx], float(best_score), bool(endpoint)


def expand_sweep_row_to_linear_assignment(
    payload: Mapping[str, Any],
    unit_assignment: Mapping[str, str],
    *,
    profile=None,
    format_rank: Mapping[str, int] | None = None,
) -> dict[str, str]:
    blocks, singletons, _pairs = du.parse_payload(payload)
    units: list[du.DecisionUnit] = []
    for unit_list in blocks.values():
        units.extend(unit_list)
    units.extend(singletons)
    expanded = expand_unit_assignment(unit_assignment, units)
    if profile is None:
        return expanded
    runtime_format_rank = _format_rank_from_payload(payload, format_rank)
    legalized = legalize_assignment_for_runtime(
        expanded,
        profile=profile,
        format_rank=runtime_format_rank,
    )
    if legalized != expanded:
        changed = sorted(
            name for name in set(expanded).union(legalized)
            if expanded.get(name) != legalized.get(name)
        )
        sample = ", ".join(changed[:8])
        raise ValueError(
            "CLADO candidate assignment violates runtime format constraints "
            f"for {len(changed)} Linear(s): {sample}. Regenerate the sweep "
            "with the same profile so fused siblings are tested as a group."
        )
    return expanded


def sweep_payload(
    payload: Mapping[str, Any],
    *,
    lambda_min: float = 1e-12,
    lambda_max: float = 1e-3,
    n_lambdas: int = 41,
    max_states_per_block: int | None = 65_536,
    profile=None,
    format_rank: Mapping[str, int] | None = None,
    structure_scope: str = "none",
) -> dict[str, Any]:
    payload = _payload_for_structure_scope(
        payload,
        structure_scope=structure_scope,
        profile=profile,
    )
    block_states = build_block_states(
        payload,
        max_states_per_block=max_states_per_block,
        profile=profile,
        format_rank=format_rank,
    )
    total_params = du.total_param_count(payload)
    rows = [
        {
            "lambda": result.lambda_used,
            "bits_total": result.bits_total,
            "bpp": (
                result.bits_total / float(total_params)
                if total_params else 0.0
            ),
            "cost_total": result.cost_total,
            "assignment": result.assignment,
        }
        for result in lambda_sweep(
            block_states,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            n_lambdas=n_lambdas,
        )
    ]
    return {
        "schema": SWEEP_SCHEMA,
        "rows": rows,
        "total_params": int(total_params),
        "meta": {
            "structure_scope": structure_scope,
            "structure_coarsening": (payload.get("meta") or {}).get(
                "structure_coarsening"
            ),
        },
    }


def budget_payload(
    payload: Mapping[str, Any],
    *,
    target_bpp: float,
    bit_precision_bits: float = 1.0,
    max_states_per_block: int | None = 65_536,
    profile=None,
    format_rank: Mapping[str, int] | None = None,
    structure_scope: str = "none",
) -> dict[str, Any]:
    payload = _payload_for_structure_scope(
        payload,
        structure_scope=structure_scope,
        profile=profile,
    )
    block_states = build_block_states(
        payload,
        max_states_per_block=max_states_per_block,
        profile=profile,
        format_rank=format_rank,
    )
    total_params = du.total_param_count(payload)
    result = solve_budget(
        block_states,
        bits_budget=float(target_bpp) * float(total_params),
        bit_precision_bits=bit_precision_bits,
    )
    if result is None:
        raise RuntimeError("infeasible budget")
    result = fill_bpp(result, total_params)
    return {
        "schema": BUDGET_SCHEMA,
        "bits_total": result.bits_total,
        "bpp": result.bpp,
        "cost_total": result.cost_total,
        "assignment": result.assignment,
        "meta": {
            "structure_scope": structure_scope,
            "structure_coarsening": (payload.get("meta") or {}).get(
                "structure_coarsening"
            ),
        },
    }


def kneedle_payloads(
    payload: Mapping[str, Any],
    sweep: Mapping[str, Any],
    *,
    n_neighbors: int = 2,
    profile=None,
    format_rank: Mapping[str, int] | None = None,
    structure_scope: str = "none",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _payload_for_structure_scope(
        payload,
        structure_scope=structure_scope,
        profile=profile,
    )
    sweep_scope = (sweep.get("meta") or {}).get("structure_scope")
    if sweep_scope is not None and str(sweep_scope) != str(structure_scope):
        raise ValueError(
            "kneedle structure scope does not match sweep: "
            f"sweep={sweep_scope!r} requested={structure_scope!r}. "
            "Rerun sweep with the same structure scope."
        )
    rows = list(sweep.get("rows") or ())
    if not rows:
        raise RuntimeError("sweep file has no rows")
    positive_rows = [row for row in rows if float(row["cost_total"]) > 0.0]
    if len(positive_rows) < 3:
        positive_rows = rows
    known_unit_names = _payload_unit_names(payload)
    points = [
        (float(row["bpp"]), float(row["cost_total"]))
        for row in positive_rows
    ]
    idx, score, endpoint = kneedle_pick(points)
    chosen = positive_rows[idx]
    sweep_sorted = sorted(rows, key=lambda row: float(row["bpp"]))
    chosen_bpp = float(chosen["bpp"])
    chosen_sorted_idx = min(
        range(len(sweep_sorted)),
        key=lambda i: abs(float(sweep_sorted[i]["bpp"]) - chosen_bpp),
    )
    neighbour_indices = list(range(
        max(chosen_sorted_idx - max(int(n_neighbors), 0), 0),
        min(chosen_sorted_idx + max(int(n_neighbors), 0) + 1, len(sweep_sorted)),
    ))
    candidates: list[dict[str, Any]] = []
    summary_candidates: list[dict[str, Any]] = []
    for sort_idx in neighbour_indices:
        row = sweep_sorted[sort_idx]
        label = (
            "kneedle"
            if sort_idx == chosen_sorted_idx
            else f"neighbor_bpp_{float(row['bpp']):.4f}".replace(".", "p")
        )
        if str(structure_scope) != "none":
            row_unit_names = set(str(name) for name in row["assignment"])
            unknown = sorted(row_unit_names.difference(known_unit_names))
            if unknown:
                sample = ", ".join(unknown[:8])
                raise ValueError(
                    "sweep assignment keys do not match the requested "
                    f"CLADO structure scope {structure_scope!r}; sample "
                    f"unknown unit(s): {sample}. Rerun sweep with the same "
                    "structure scope, or use --structure-scope none for an "
                    "archived flat sweep."
                )
        assignment = expand_sweep_row_to_linear_assignment(
            payload,
            row["assignment"],
            profile=profile,
            format_rank=format_rank,
        )
        candidate = {
            "schema": KNEEDLE_SCHEMA,
            "label": label,
            "bpp": float(row["bpp"]),
            "bits_total": float(row["bits_total"]),
            "surrogate_cost": float(row["cost_total"]),
            "lambda": float(row["lambda"]),
            "assignment": assignment,
        }
        candidates.append(candidate)
        summary_candidates.append({
            "label": label,
            "bpp": float(row["bpp"]),
            "surrogate_cost": float(row["cost_total"]),
        })
    summary = {
        "schema": KNEEDLE_SUMMARY_SCHEMA,
        "kneedle_score": float(score),
        "endpoint_fallback": bool(endpoint),
        "chosen": {
            "bpp": float(chosen["bpp"]),
            "surrogate_cost": float(chosen["cost_total"]),
            "lambda": float(chosen["lambda"]),
        },
        "candidates": summary_candidates,
        "frontier_size_used": len(positive_rows),
        "frontier_size_total": len(rows),
        "meta": {
            "structure_scope": structure_scope,
            "structure_coarsening": (payload.get("meta") or {}).get(
                "structure_coarsening"
            ),
        },
    }
    return summary, candidates


def _load_assignment(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, Mapping) and isinstance(payload.get("assignment"), Mapping):
        return {
            str(key): canonicalize_format(value)
            for key, value in payload["assignment"].items()
        }
    if isinstance(payload, Mapping):
        return {
            str(key): canonicalize_format(value)
            for key, value in payload.items()
        }
    raise ValueError(f"unsupported assignment shape: {path}")


def _load_production_weight_cache(path: str | Path | None):
    if not path:
        return None
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def _profile_from_optional_model(model_path: str | None):
    if not model_path:
        return None
    try:
        return detect_profile(model_path)
    except Exception:
        return DefaultProfile()


def _progress_printer(event: dict[str, Any]) -> None:
    kind = event.get("event")
    if kind == "center_kl":
        print(f"[block-clado] center KL={float(event['kl']):.6g}", flush=True)
    elif kind == "unary_done":
        print(
            "[block-clado] unary "
            f"{int(event['completed'])}/{int(event['total'])} "
            f"{event['unit']}@{event['format']} "
            f"KL={float(event['kl']):.6g}",
            flush=True,
        )
    elif kind == "pair_done":
        print(
            "[block-clado] pair "
            f"{int(event['completed'])}/{int(event['total'])} "
            f"{event['unit_a']}@{event['fmt_a']} x "
            f"{event['unit_b']}@{event['fmt_b']} "
            f"KL_ab={float(event['kl_ab']):.6g} "
            f"omega_ij={float(event['omega_ij']):.6g}",
            flush=True,
        )


def _add_common_measure_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--formats", default="NVFP4,MXFP8_E4M3,BF16")
    parser.add_argument("--target-profile", default=None)
    parser.add_argument("--n-calib-samples", type=int, default=2)
    parser.add_argument("--calib-seqlen", type=int, default=128)
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional calibration source accepted by sensitivity_probe "
        "(HF dataset id, .jsonl, or .txt). When omitted, preserves the "
        "historical wikitext-2 windowed loader.",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--skip-pairs", action="store_true")
    parser.add_argument("--center-assignment", default=None)
    parser.add_argument("--production-weight-cache", default=None)
    parser.add_argument("--production-cache-dir-override", default=None)
    parser.add_argument(
        "--production-cache-lru-gb",
        type=float,
        default=4.0,
        help="Resident tensor budget for disk-backed production cache use.",
    )
    parser.add_argument(
        "--production-cache-prefetch-workers",
        type=int,
        default=4,
        help="Worker count for required rendered-weight prefetch.",
    )
    parser.add_argument("--use-frozen-weight-cache", action="store_true")
    parser.add_argument("--no-production-cache-prefetch", action="store_true")
    parser.add_argument(
        "--source-prefetch",
        choices=("off", "auto", "require"),
        default="auto",
        help="Prefetch local BF16 source safetensors before model load.",
    )
    parser.add_argument(
        "--source-prefetch-max-gb",
        type=float,
        default=0.0,
        help="Resident byte budget for source safetensors prefetch. 0 derives "
        "the budget from available memory minus --source-prefetch-headroom-gb.",
    )
    parser.add_argument(
        "--source-prefetch-headroom-gb",
        type=float,
        default=16.0,
    )
    parser.add_argument("--source-prefetch-workers", type=int, default=2)
    parser.add_argument("--no-activation-quant", action="store_true")
    parser.add_argument(
        "--kl-scope",
        choices=("last_token", "full_sequence"),
        default="last_token",
    )


def _device_from_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _cmd_measure(args) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_str = _device_from_arg(args.device)
    dtype = _dtype_from_name(args.dtype)
    staged, cleanup = stage_multimodal(args.model)
    try:
        local_only = bool(args.local_files_only or Path(staged).exists())
        tokenizer = AutoTokenizer.from_pretrained(
            staged,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        if args.dataset:
            calib_ids = load_calibration(
                tokenizer,
                args.dataset,
                args.n_calib_samples,
                args.calib_seqlen,
            )
        else:
            calib_ids = load_wikitext_calibration_windowed(
                tokenizer,
                args.n_calib_samples,
                args.calib_seqlen,
                split=args.calib_split,
                seed=args.calib_seed,
            )
        source_prefetch_stats = prefetch_safetensors_checkpoint(
            staged,
            mode=args.source_prefetch,
            max_resident_bytes=(
                int(float(args.source_prefetch_max_gb) * 1024**3)
                if float(args.source_prefetch_max_gb) > 0
                else None
            ),
            headroom_gb=float(args.source_prefetch_headroom_gb),
            workers=int(args.source_prefetch_workers),
            progress=True,
            log_prefix="[block-clado/source]",
        )
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": local_only,
        }
        if args.device_map:
            load_kwargs["device_map"] = args.device_map
        elif device_str == "cuda":
            load_kwargs["device_map"] = device_str
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        if not load_kwargs.get("device_map") and device_str != "cuda":
            model.to(device_str)
        model.eval()
        try:
            profile = detect_profile(args.model)
        except Exception:
            profile = DefaultProfile()
        center_assignment = (
            _load_assignment(args.center_assignment)
            if args.center_assignment else None
        )
        production_cache = _load_production_weight_cache(args.production_weight_cache)
        if (
            production_cache is not None
            and args.production_cache_dir_override
            and hasattr(production_cache, "relocate")
        ):
            production_cache.relocate(args.production_cache_dir_override)
        if (
            production_cache is not None
            and args.production_cache_lru_gb
            and float(args.production_cache_lru_gb) > 0
            and hasattr(production_cache, "enable_lru")
        ):
            production_cache.enable_lru(
                int(float(args.production_cache_lru_gb) * 1024**3)
            )
        specs = [
            fr.get_format(part.strip())
            for part in args.formats.split(",")
            if part.strip()
        ]
        payload = collect_block_clado(
            model,
            calib_ids,
            specs,
            profile=profile,
            target_profile=args.target_profile,
            work_root=args.work_dir,
            progress_callback=_progress_printer,
            skip_pairs=bool(args.skip_pairs),
            center_assignment=center_assignment,
            use_frozen_weight_cache=bool(args.use_frozen_weight_cache),
            production_weight_cache=production_cache,
            prefetch_production_cache=not bool(args.no_production_cache_prefetch),
            production_cache_prefetch_workers=int(
                args.production_cache_prefetch_workers
            ),
            kl_scope=args.kl_scope,
            include_activation_quant=not bool(args.no_activation_quant),
        )
        payload.setdefault("meta", {})
        payload["meta"].update({
            "dataset": args.dataset,
            "calib_split": args.calib_split,
            "calib_seed": int(args.calib_seed),
            "production_cache_lru_gb": float(args.production_cache_lru_gb),
            "production_cache_prefetch_workers": int(
                args.production_cache_prefetch_workers
            ),
            "source_prefetch": source_prefetch_stats,
        })
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        meta = payload.get("meta", {})
        print(
            f"[block-clado] wrote {out_path} "
            f"blocks={meta.get('block_count')} "
            f"singletons={meta.get('singleton_count')} "
            f"unary_meas={meta.get('n_unary_measurements')} "
            f"pair_meas={meta.get('n_pair_measurements')} "
            f"elapsed={float(meta.get('elapsed_seconds', 0.0)):.1f}s",
            flush=True,
        )
        return 0
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cmd_sweep(args) -> int:
    payload = du.load_payload(args.payload)
    profile = _profile_from_optional_model(getattr(args, "model", None))
    sweep = sweep_payload(
        payload,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        n_lambdas=args.n_lambdas,
        max_states_per_block=args.max_states_per_block,
        profile=profile,
        structure_scope=args.structure_scope,
    )
    Path(args.output).write_text(json.dumps(sweep, indent=2) + "\n")
    print(
        f"[block-clado] lambda sweep wrote {len(sweep['rows'])} "
        f"frontier points to {args.output}",
        flush=True,
    )
    return 0


def _cmd_budget(args) -> int:
    payload = du.load_payload(args.payload)
    profile = _profile_from_optional_model(getattr(args, "model", None))
    result = budget_payload(
        payload,
        target_bpp=args.target_bpp,
        bit_precision_bits=args.bit_precision_bits,
        max_states_per_block=args.max_states_per_block,
        profile=profile,
        structure_scope=args.structure_scope,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[block-clado] budget solve wrote bpp={float(result['bpp']):.4f} "
        f"cost={float(result['cost_total']):.6g} to {args.output}",
        flush=True,
    )
    return 0


def _cmd_kneedle(args) -> int:
    payload = du.load_payload(args.payload)
    profile = _profile_from_optional_model(getattr(args, "model", None))
    sweep = json.loads(Path(args.sweep).read_text())
    summary, candidates = kneedle_payloads(
        payload,
        sweep,
        n_neighbors=args.n_neighbors,
        profile=profile,
        structure_scope=args.structure_scope,
    )
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        label = str(candidate["label"])
        (out_root / f"{label}.json").write_text(
            json.dumps(candidate, indent=2) + "\n"
        )
        for entry in summary["candidates"]:
            if entry["label"] == label:
                entry["path"] = str(out_root / f"{label}.json")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    chosen = summary["chosen"]
    print(
        f"[block-clado] kneedle bpp={float(chosen['bpp']):.4f} "
        f"cost={float(chosen['surrogate_cost']):.6g} "
        f"score={float(summary['kneedle_score']):.4f}",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Executable Block-CLADO research component"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="measure a Block-CLADO payload")
    _add_common_measure_args(measure)
    measure.set_defaults(func=_cmd_measure)

    sweep = sub.add_parser("sweep", help="lambda sweep over a payload")
    sweep.add_argument("--payload", required=True)
    sweep.add_argument(
        "--model",
        default=None,
        help="Optional model/checkpoint path used to detect fused-sibling and "
        "packed-expert format constraints for legacy payloads.",
    )
    sweep.add_argument("--lambda-min", type=float, default=1e-12)
    sweep.add_argument("--lambda-max", type=float, default=1e-3)
    sweep.add_argument("--n-lambdas", type=int, default=41)
    sweep.add_argument("--max-states-per-block", type=int, default=65_536)
    sweep.add_argument(
        "--structure-scope",
        choices=STRUCTURE_SCOPES,
        default="runtime",
        help="Coarsen legacy payloads onto model-structure units before "
        "solving. runtime merges profile-coupled serving groups; subblock "
        "merges attention/MLP groups; layer merges whole transformer layers.",
    )
    sweep.add_argument("--output", required=True)
    sweep.set_defaults(func=_cmd_sweep)

    budget = sub.add_parser("budget", help="exact budget solve over a payload")
    budget.add_argument("--payload", required=True)
    budget.add_argument(
        "--model",
        default=None,
        help="Optional model/checkpoint path used to detect fused-sibling and "
        "packed-expert format constraints for legacy payloads.",
    )
    budget.add_argument("--target-bpp", type=float, required=True)
    budget.add_argument("--bit-precision-bits", type=float, default=1.0)
    budget.add_argument("--max-states-per-block", type=int, default=65_536)
    budget.add_argument(
        "--structure-scope",
        choices=STRUCTURE_SCOPES,
        default="runtime",
        help="Coarsen legacy payloads onto model-structure units before "
        "solving. Use none to preserve the archived flat unit space.",
    )
    budget.add_argument("--output", required=True)
    budget.set_defaults(func=_cmd_budget)

    knee = sub.add_parser("kneedle", help="write kneedle candidate JSONs")
    knee.add_argument("--payload", required=True)
    knee.add_argument("--sweep", required=True)
    knee.add_argument("--output-dir", required=True)
    knee.add_argument(
        "--model",
        default=None,
        help="Optional model/checkpoint path used to detect fused-sibling and "
        "packed-expert format constraints for legacy payloads.",
    )
    knee.add_argument("--n-neighbors", type=int, default=2)
    knee.add_argument(
        "--structure-scope",
        choices=STRUCTURE_SCOPES,
        default="runtime",
        help="Must match the structure scope used for the sweep.",
    )
    knee.set_defaults(func=_cmd_kneedle)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
