"""SMRF archive candidate generator.

SMRF's useful surviving role is candidate generation: build a compact archive
of assignment candidates from additive propagated/end-KL costs, then let the
shared real-KL validator decide whether any candidate should be promoted.  This
module ports that candidate-generation layer onto the current decision-unit
payload shape.  It does not measure KL and it does not replace the production
assignment by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prismaquant import decision_units as du
from prismaquant import format_registry as fr


ARCHIVE_SCHEMA = "prismaquant.smrf.archive.v1"
CANDIDATE_MANIFEST_SCHEMA = "prismaquant.smrf.candidates.v1"


@dataclass(frozen=True)
class SmrfCandidate:
    unit_assignment: dict[str, str]
    assignment: dict[str, str]
    achieved_bpp: float
    bits_total: float
    surrogate_loss: float
    source: str
    rank: int | None = None
    lambda_penalty: float | None = None
    label: str | None = None

    @property
    def assignment_hash(self) -> str:
        return assignment_hash(self.assignment)

    def to_row(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "assignment_hash": self.assignment_hash,
            "unit_assignment": dict(sorted(self.unit_assignment.items())),
            "assignment": dict(sorted(self.assignment.items())),
            "achieved_bpp": float(self.achieved_bpp),
            "bits_total": float(self.bits_total),
            "surrogate_loss": float(self.surrogate_loss),
            "source": self.source,
            "rank": self.rank,
            "lambda_penalty": self.lambda_penalty,
            "format_counts": format_counts(self.assignment),
            "block_format_counts": block_format_counts(self.assignment),
            "assignment_entries": len(self.assignment),
            "unit_entries": len(self.unit_assignment),
        }


def assignment_hash(assignment: Mapping[str, str]) -> str:
    blob = json.dumps(
        {str(k): fr.canonical_format_name(v) for k, v in sorted(assignment.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def format_counts(assignment: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fmt in assignment.values():
        canonical = fr.canonical_format_name(str(fmt))
        counts[canonical] = counts.get(canonical, 0) + 1
    return counts


def block_format_counts(assignment: Mapping[str, str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for qname, fmt in assignment.items():
        block_id = du.block_id_from_qname(str(qname))
        canonical = fr.canonical_format_name(str(fmt))
        row = counts.setdefault(block_id, {})
        row[canonical] = row.get(canonical, 0) + 1
    return dict(sorted((block, dict(sorted(row.items()))) for block, row in counts.items()))


def _all_units(payload: Mapping[str, Any]) -> list[du.DecisionUnit]:
    blocks, singletons, _pairs = du.parse_payload(payload)
    units: list[du.DecisionUnit] = []
    for block_id in sorted(blocks):
        units.extend(blocks[block_id])
    units.extend(singletons)
    units.sort(key=lambda unit: unit.name)
    return units


def _extract_stats(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    stats = payload.get("stats")
    if isinstance(stats, Mapping):
        return {str(k): v for k, v in stats.items() if isinstance(v, Mapping)}
    return {str(k): v for k, v in payload.items() if isinstance(v, Mapping)}


def _extract_costs(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    costs = payload.get("costs")
    if isinstance(costs, Mapping):
        return {str(k): v for k, v in costs.items() if isinstance(v, Mapping)}
    return {str(k): v for k, v in payload.items() if isinstance(v, Mapping)}


def _format_specs(format_names: Sequence[str] | None) -> list[fr.FormatSpec]:
    requested = format_names or ("NVFP4", "MXFP8_E4M3", "FP8_E4M3", "BF16")
    seen: set[str] = set()
    specs: list[fr.FormatSpec] = []
    for raw in requested:
        canonical = fr.canonical_format_name(str(raw).strip().upper())
        if not canonical or canonical in seen:
            continue
        specs.append(fr.get_format(canonical))
        seen.add(canonical)
    return specs


def _load_pickle_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected pickle containing a mapping")
    return dict(payload)


def _profile_for_model(model_path: str | Path | None):
    if model_path is None:
        from prismaquant.model_profiles import DefaultProfile

        return DefaultProfile()
    from prismaquant.model_profiles import detect_profile

    return detect_profile(str(model_path))


def _profile_excludes_qname(profile: Any, qname: str) -> bool:
    """Whether profile policy keeps this recipe qname out of quantization."""
    name = str(qname)
    is_pinned = getattr(profile, "is_pinned_name", None)
    if callable(is_pinned) and bool(is_pinned(name)):
        return True
    pinned_names = getattr(profile, "pinned_names", None)
    if callable(pinned_names):
        for raw_pin in pinned_names():
            pin = str(raw_pin)
            pin_module = pin[:-7] if pin.endswith(".weight") else pin
            if name == pin_module or name.endswith("." + pin_module):
                return True
    passthrough_prefixes = getattr(profile, "source_passthrough_prefixes", None)
    if callable(passthrough_prefixes):
        for raw_prefix in passthrough_prefixes():
            prefix = str(raw_prefix)
            if not prefix:
                continue
            stripped = prefix.rstrip(".")
            if name == stripped or name.startswith(prefix):
                return True
    return False


def _filter_profile_mutable(
    stats: Mapping[str, Mapping[str, Any]],
    costs: Mapping[str, Any],
    profile: Any,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any], dict[str, Any]]:
    excluded = sorted(
        name for name in set(stats) | set(costs)
        if _profile_excludes_qname(profile, str(name))
    )
    if not excluded:
        return dict(stats), dict(costs), {
            "profile_excluded_qnames": 0,
            "profile_excluded_sample": [],
        }
    excluded_set = set(excluded)
    return (
        {str(name): value for name, value in stats.items() if name not in excluded_set},
        {str(name): value for name, value in costs.items() if name not in excluded_set},
        {
            "profile_excluded_qnames": len(excluded),
            "profile_excluded_sample": excluded[:8],
        },
    )


def _candidate_members(
    stats: Mapping[str, Mapping[str, Any]],
    unit_name: str,
) -> tuple[str, ...]:
    entry = stats.get(str(unit_name))
    if isinstance(entry, Mapping):
        members = entry.get("_fused_siblings")
        if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
            return tuple(str(member) for member in members)
    return (str(unit_name),)


def _filter_profile_mutable_candidate_units(
    stats: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Any]],
    profile: Any,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[Any]], dict[str, Any]]:
    excluded = sorted(
        str(unit_name)
        for unit_name in candidates
        if _profile_excludes_qname(profile, str(unit_name))
        or any(_profile_excludes_qname(profile, member) for member in _candidate_members(stats, str(unit_name)))
    )
    if not excluded:
        return dict(stats), {str(name): list(cands) for name, cands in candidates.items()}, {
            "profile_excluded_units": 0,
            "profile_excluded_unit_sample": [],
        }
    excluded_set = set(excluded)
    return (
        {str(name): value for name, value in stats.items() if name not in excluded_set},
        {
            str(name): list(cands)
            for name, cands in candidates.items()
            if str(name) not in excluded_set
        },
        {
            "profile_excluded_units": len(excluded),
            "profile_excluded_unit_sample": excluded[:8],
        },
    )


def _filter_serving_legal_candidates(
    stats: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Any]],
    *,
    target_profile: str | None,
) -> dict[str, list[Any]]:
    if not target_profile:
        return {str(name): list(cands) for name, cands in candidates.items()}
    from prismaquant.serving_profiles import check_serving_format

    out: dict[str, list[Any]] = {}
    for name, cands in candidates.items():
        members = _candidate_members(stats, str(name))
        kept = []
        for cand in cands:
            legal = True
            for member in members:
                decision = check_serving_format(target_profile, member, cand.fmt)
                if not decision.legal and decision.detail.startswith("unknown target profile"):
                    raise ValueError(decision.detail)
                if not decision.legal:
                    legal = False
                    break
            if legal:
                kept.append(cand)
        if kept:
            out[str(name)] = kept
    return out


def _decision_payload_from_candidates(
    *,
    stats: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Any]],
    specs: Sequence[fr.FormatSpec],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    blocks: dict[str, dict[str, Any]] = {}
    singletons: dict[str, Any] = {}
    for unit_name in sorted(candidates):
        stats_entry = stats.get(unit_name)
        if not isinstance(stats_entry, Mapping):
            continue
        members = stats_entry.get("_fused_siblings")
        if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
            members = (unit_name,)
        members = tuple(str(member) for member in members)
        options = {
            fr.canonical_format_name(cand.fmt): {
                "omega_ii": float(cand.predicted_dloss),
                "bits_per_param": float(cand.bits_per_param),
                "memory_bytes": int(cand.memory_bytes),
            }
            for cand in candidates[unit_name]
        }
        if not options:
            continue
        unit_payload = {
            "members": list(members),
            "options": options,
        }
        block_id = du.block_id_from_qname(members[0] if members else unit_name)
        if block_id != (members[0] if members else unit_name):
            block = blocks.setdefault(str(block_id), {"units": {}, "pairs": []})
            block["units"][str(unit_name)] = unit_payload
        else:
            unit_payload["block_id"] = str(block_id)
            singletons[str(unit_name)] = unit_payload

    payload_meta = dict(meta)
    payload_meta.setdefault("formats", [spec.name for spec in specs])
    payload_meta["unit_count"] = (
        sum(len(block["units"]) for block in blocks.values()) + len(singletons)
    )
    return {
        "schema": du.SCHEMA,
        "meta": payload_meta,
        "blocks": blocks,
        "singletons": singletons,
    }


def decision_unit_payload_from_probe_costs(
    probe_payload: Mapping[str, Any],
    cost_payload: Mapping[str, Any],
    *,
    model_path: str | Path | None = None,
    formats: Sequence[str] | None = None,
    target_profile: str | None = None,
    aggregate_siblings: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a SMRF decision-unit payload from existing probe/cost artifacts.

    This is the compatibility bridge for current 4B experiments: reuse the
    allocator's measured per-format local costs and legality filters, then
    package the resulting choices as decision units for SMRF archive search.
    If the selected profile exposes q/k/v, gate/up, MoE, or other serving
    groups, those groups become single SMRF units.
    """

    from prismaquant.allocator_candidates import (
        aggregate_fused_siblings,
        build_candidates,
    )

    specs = _format_specs(formats or cost_payload.get("formats"))
    profile = _profile_for_model(model_path)
    stats, costs, profile_filter_meta = _filter_profile_mutable(
        _extract_stats(probe_payload),
        _extract_costs(cost_payload),
        profile,
    )
    candidates = build_candidates(
        stats,
        costs,
        specs,
        source_manifest=None,
        target_profile=target_profile,
    )
    if aggregate_siblings:
        stats, _costs, candidates = aggregate_fused_siblings(
            stats,
            costs,
            specs,
            candidates,
            profile=profile,
        )
    stats, candidates, unit_filter_meta = _filter_profile_mutable_candidate_units(
        stats,
        candidates,
        profile,
    )
    candidates = _filter_serving_legal_candidates(
        stats,
        candidates,
        target_profile=target_profile,
    )

    payload_meta = {
        "source": "probe_cost_allocator_bridge",
        "model_path": str(model_path) if model_path is not None else None,
        "model_profile": getattr(profile, "name", ""),
        "formats": [spec.name for spec in specs],
        "target_profile": target_profile,
        "aggregate_siblings": bool(aggregate_siblings),
        "cost_objective": "allocator_cost_entry_predicted_dloss",
        **profile_filter_meta,
        **unit_filter_meta,
    }
    if meta:
        payload_meta.update(dict(meta))
    return _decision_payload_from_candidates(
        stats=stats,
        candidates=candidates,
        specs=specs,
        meta=payload_meta,
    )


def _extract_l3_costs(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    costs = payload.get("costs")
    if isinstance(costs, Mapping):
        return {str(k): v for k, v in costs.items() if isinstance(v, Mapping)}
    return {str(k): v for k, v in payload.items() if isinstance(v, Mapping)}


def _load_assignment_mapping(path: str | Path) -> dict[str, str]:
    from prismaquant.layer_config import canonicalize_format

    payload = json.loads(Path(path).read_text())
    raw = payload.get("assignment") if isinstance(payload, Mapping) else None
    if raw is None and isinstance(payload, Mapping):
        raw = payload
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: expected assignment or layer_config mapping")
    return {
        str(name): canonicalize_format(value)
        for name, value in raw.items()
        if str(name).strip()
    }


def _parse_labeled_assignment(value: str) -> tuple[str, dict[str, str]]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip() or Path(raw_path).stem
        path = Path(raw_path)
    else:
        path = Path(value)
        label = path.stem
    return label, _load_assignment_mapping(path)


def decision_unit_payload_from_l3_costs(
    probe_payload: Mapping[str, Any],
    l3_payload: Mapping[str, Any],
    baseline_assignment: Mapping[str, str],
    *,
    model_path: str | Path | None = None,
    formats: Sequence[str] | None = None,
    target_profile: str | None = None,
    aggregate_siblings: bool = True,
    include_frozen_baseline: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a full-bpp SMRF payload from measured L3 propagated costs.

    Measured L3 entries become mutable SMRF options. Unmeasured linears are
    retained as single-option units at the supplied baseline assignment so the
    archive bpp is reported over the same quantizable parameter set as normal
    assignments. This keeps L3V candidate generation sparse without mixing L3
    propagated costs with local L2 costs.
    """

    from prismaquant.allocator_candidates import (
        aggregate_fused_siblings,
        check_stats_format_applicability,
    )
    from prismaquant.allocator_solver import Candidate, _shape_from_stats

    specs = _format_specs(formats or l3_payload.get("formats"))
    specs_by_name = {spec.name: spec for spec in specs}
    profile = _profile_for_model(model_path)
    stats, l3_costs, profile_filter_meta = _filter_profile_mutable(
        _extract_stats(probe_payload),
        _extract_l3_costs(l3_payload),
        profile,
    )
    baseline = {
        str(name): fr.canonical_format_name(fmt)
        for name, fmt in baseline_assignment.items()
        if str(name) in stats
    }

    candidate_stats: dict[str, Mapping[str, Any]] = {}
    candidate_costs: dict[str, dict[str, dict[str, float]]] = {}
    candidates: dict[str, list[Any]] = {}

    names = sorted(set(baseline) | (set(l3_costs) & set(baseline)))
    measured_units = 0
    frozen_units = 0
    for name in names:
        stats_entry = dict(stats[name])
        shape = _shape_from_stats(stats_entry)
        per_name = l3_costs.get(name)
        row: list[Any] = []
        cost_row: dict[str, dict[str, float]] = {}
        if isinstance(per_name, Mapping):
            for raw_fmt, entry in per_name.items():
                fmt = fr.canonical_format_name(str(raw_fmt))
                spec = specs_by_name.get(fmt)
                if spec is None or not isinstance(entry, Mapping):
                    continue
                if "error" in entry:
                    continue
                value = entry.get("propagated_end_kl")
                if value is None:
                    value = entry.get("predicted_dloss")
                if value is None:
                    continue
                verdict = check_stats_format_applicability(
                    stats_entry,
                    spec,
                    qname=name,
                    target_profile=target_profile,
                )
                if not verdict.legal:
                    continue
                memory_bytes = int(spec.memory_bytes_for_shape(shape))
                omega = max(float(value), 0.0)
                row.append(Candidate(
                    fmt=fmt,
                    bits_per_param=float(spec.effective_bits_for_shape(shape)),
                    memory_bytes=memory_bytes,
                    predicted_dloss=omega,
                ))
                cost_row[fmt] = {
                    "predicted_dloss": omega,
                    "propagated_end_kl": omega,
                }
        if row:
            measured_units += 1
        elif include_frozen_baseline:
            fmt = baseline.get(name, "BF16")
            spec = specs_by_name.get(fmt)
            if spec is None:
                continue
            verdict = check_stats_format_applicability(
                stats_entry,
                spec,
                qname=name,
                target_profile=target_profile,
            )
            if not verdict.legal:
                continue
            memory_bytes = int(spec.memory_bytes_for_shape(shape))
            row.append(Candidate(
                fmt=fmt,
                bits_per_param=float(spec.effective_bits_for_shape(shape)),
                memory_bytes=memory_bytes,
                predicted_dloss=0.0,
            ))
            cost_row[fmt] = {
                "predicted_dloss": 0.0,
                "propagated_end_kl": 0.0,
            }
            frozen_units += 1
        if not row:
            continue
        candidate_stats[name] = stats_entry
        candidate_costs[name] = cost_row
        candidates[name] = row

    if aggregate_siblings:
        candidate_stats, _costs, candidates = aggregate_fused_siblings(
            dict(candidate_stats),
            candidate_costs,
            specs,
            candidates,
            profile=profile,
        )
    candidate_stats, candidates, unit_filter_meta = _filter_profile_mutable_candidate_units(
        candidate_stats,
        candidates,
        profile,
    )
    candidates = _filter_serving_legal_candidates(
        candidate_stats,
        candidates,
        target_profile=target_profile,
    )

    payload_meta = {
        "source": "l3_propagated_cost_bridge",
        "model_path": str(model_path) if model_path is not None else None,
        "model_profile": getattr(profile, "name", ""),
        "formats": [spec.name for spec in specs],
        "target_profile": target_profile,
        "aggregate_siblings": bool(aggregate_siblings),
        "include_frozen_baseline": bool(include_frozen_baseline),
        "cost_objective": "l3_propagated_end_kl",
        "measured_units_pre_aggregation": int(measured_units),
        "frozen_units_pre_aggregation": int(frozen_units),
        **profile_filter_meta,
        **unit_filter_meta,
    }
    if meta:
        payload_meta.update(dict(meta))
    return _decision_payload_from_candidates(
        stats=candidate_stats,
        candidates=candidates,
        specs=specs,
        meta=payload_meta,
    )


def decision_unit_payload_from_l3_cost_files(
    *,
    probe: str | Path,
    l3_costs: str | Path,
    baseline_assignment: str | Path,
    model_path: str | Path | None = None,
    formats: Sequence[str] | None = None,
    target_profile: str | None = None,
    aggregate_siblings: bool = True,
    include_frozen_baseline: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return decision_unit_payload_from_l3_costs(
        _load_pickle_mapping(probe),
        _load_pickle_mapping(l3_costs),
        _load_assignment_mapping(baseline_assignment),
        model_path=model_path,
        formats=formats,
        target_profile=target_profile,
        aggregate_siblings=aggregate_siblings,
        include_frozen_baseline=include_frozen_baseline,
        meta={
            "probe": str(probe),
            "l3_costs": str(l3_costs),
            "baseline_assignment": str(baseline_assignment),
            **dict(meta or {}),
        },
    )


def decision_unit_payload_from_probe_cost_files(
    *,
    probe: str | Path,
    costs: str | Path,
    model_path: str | Path | None = None,
    formats: Sequence[str] | None = None,
    target_profile: str | None = None,
    aggregate_siblings: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return decision_unit_payload_from_probe_costs(
        _load_pickle_mapping(probe),
        _load_pickle_mapping(costs),
        model_path=model_path,
        formats=formats,
        target_profile=target_profile,
        aggregate_siblings=aggregate_siblings,
        meta={
            "probe": str(probe),
            "costs": str(costs),
            **dict(meta or {}),
        },
    )


def _unit_param_count(unit: du.DecisionUnit) -> int:
    ref = next((opt for opt in unit.options if opt.fmt == "BF16"), None)
    if ref is None:
        ref = unit.options[0]
    if ref.bits_per_param <= 0:
        return 0
    return int(round(float(ref.memory_bytes) * 8.0 / float(ref.bits_per_param)))


def _total_params(units: Sequence[du.DecisionUnit]) -> int:
    return sum(_unit_param_count(unit) for unit in units)


def _expand_unit_assignment(
    unit_assignment: Mapping[str, str],
    units_by_name: Mapping[str, du.DecisionUnit],
) -> dict[str, str]:
    expanded: dict[str, str] = {}
    for unit_name, fmt in unit_assignment.items():
        canonical = fr.canonical_format_name(str(fmt))
        unit = units_by_name.get(str(unit_name))
        if unit is None:
            expanded[str(unit_name)] = canonical
            continue
        for member in unit.member_qnames:
            expanded[str(member)] = canonical
    return expanded


def _candidate_from_unit_assignment(
    unit_assignment: Mapping[str, str],
    *,
    units_by_name: Mapping[str, du.DecisionUnit],
    total_params: int,
    source: str,
    rank: int | None = None,
    lambda_penalty: float | None = None,
    label: str | None = None,
) -> SmrfCandidate:
    bits_total = 0.0
    surrogate_loss = 0.0
    canonical_assignment: dict[str, str] = {}
    for unit_name, raw_fmt in unit_assignment.items():
        fmt = fr.canonical_format_name(str(raw_fmt))
        unit = units_by_name[str(unit_name)]
        options = {opt.fmt: opt for opt in unit.options}
        if fmt not in options:
            raise ValueError(f"{unit_name}: missing option for format {fmt}")
        option = options[fmt]
        canonical_assignment[str(unit_name)] = fmt
        bits_total += option.bits_total
        surrogate_loss += float(option.omega_ii)
    expanded = _expand_unit_assignment(canonical_assignment, units_by_name)
    return SmrfCandidate(
        unit_assignment=canonical_assignment,
        assignment=expanded,
        achieved_bpp=float(bits_total) / max(float(total_params), 1.0),
        bits_total=float(bits_total),
        surrogate_loss=float(surrogate_loss),
        source=source,
        rank=rank,
        lambda_penalty=lambda_penalty,
        label=label,
    )


def _option_bpp_contribution(opt: du.FormatCost, total_params: int) -> float:
    return float(opt.bits_total) / max(float(total_params), 1.0)


def _lambda_grid(
    units: Sequence[du.DecisionUnit],
    n_lambdas: int,
    *,
    total_params: int | None = None,
) -> list[float]:
    if n_lambdas <= 1:
        return [0.0]
    total = int(total_params if total_params is not None else _total_params(units))
    slopes: list[float] = []
    for unit in units:
        options = list(unit.options)
        for left_idx, left in enumerate(options):
            left_bpp = _option_bpp_contribution(left, total)
            for right in options[left_idx + 1:]:
                right_bpp = _option_bpp_contribution(right, total)
                dbpp = abs(left_bpp - right_bpp)
                dloss = abs(float(left.omega_ii) - float(right.omega_ii))
                if dbpp <= 0.0 or dloss <= 0.0:
                    continue
                slope = dloss / dbpp
                if math.isfinite(slope) and slope > 0.0:
                    slopes.append(float(slope))
    unique = sorted(set(round(value, 12) for value in slopes if value > 0.0))
    if not unique:
        return [0.0]
    slots = max(int(n_lambdas) - 1, 1)
    selected: set[float] = set()
    if len(unique) >= slots:
        if slots == 1:
            selected.add(unique[len(unique) // 2])
        else:
            for idx in range(slots):
                src_idx = round(idx * (len(unique) - 1) / max(slots - 1, 1))
                selected.add(unique[int(src_idx)])
    else:
        selected.update(unique)
        fill = slots - len(selected)
        low = max(min(unique) / 10.0, 1e-12)
        high = max(max(unique) * 10.0, low)
        if fill > 0:
            if fill == 1 or high <= low:
                selected.add(math.sqrt(low * high))
            else:
                log_low = math.log(low)
                log_high = math.log(high)
                for idx in range(fill):
                    t = idx / max(fill - 1, 1)
                    selected.add(math.exp(log_low + t * (log_high - log_low)))
    return [0.0, *sorted(selected)]


def solve_lagrangian_assignments(
    payload: Mapping[str, Any],
    *,
    n_lambdas: int = 21,
    bpp_min: float = 0.0,
    bpp_max: float = 16.0,
) -> list[SmrfCandidate]:
    units = _all_units(payload)
    units_by_name = {unit.name: unit for unit in units}
    total_params = _total_params(units)
    rows: list[SmrfCandidate] = []
    for idx, lambda_penalty in enumerate(
        _lambda_grid(units, int(n_lambdas), total_params=total_params)
    ):
        assignment: dict[str, str] = {}
        for unit in units:
            chosen = min(
                unit.options,
                key=lambda opt: (
                    float(opt.omega_ii)
                    + float(lambda_penalty) * _option_bpp_contribution(opt, total_params),
                    _option_bpp_contribution(opt, total_params),
                    opt.fmt,
                ),
            )
            assignment[unit.name] = chosen.fmt
        candidate = _candidate_from_unit_assignment(
            assignment,
            units_by_name=units_by_name,
            total_params=total_params,
            source="lambda_sweep",
            rank=idx,
            lambda_penalty=float(lambda_penalty),
        )
        if float(bpp_min) <= candidate.achieved_bpp <= float(bpp_max):
            rows.append(candidate)
    return _dedupe_candidates(rows)


def solve_pareto_archive_assignments(
    payload: Mapping[str, Any],
    *,
    bpp_min: float = 0.0,
    bpp_max: float = 16.0,
    bit_precision_bpp: float = 0.001,
    beam_per_bin: int = 4,
) -> tuple[list[SmrfCandidate], dict[str, Any]]:
    """Return cached-cost Pareto-DP candidates, retaining variants per bit bin."""

    units = _all_units(payload)
    if not units:
        return [], {"enabled": True, "reason": "no_units"}
    units_by_name = {unit.name: unit for unit in units}
    total_params = _total_params(units)
    if total_params <= 0:
        return [], {"enabled": True, "reason": "zero_total_params"}
    beam = max(int(beam_per_bin), 1)
    bin_width = max(float(bit_precision_bpp) * float(total_params), 1.0)
    min_bits_by_unit = {
        unit.name: min(float(opt.bits_total) for opt in unit.options)
        for unit in units
    }

    # bin -> list[(cost, bits_delta, assignment)]
    states: dict[int, list[tuple[float, float, dict[str, str]]]] = {0: [(0.0, 0.0, {})]}
    for unit in units:
        next_states: dict[int, list[tuple[float, float, dict[str, str]]]] = {}
        unit_min_bits = min_bits_by_unit[unit.name]
        for _bin, state_rows in states.items():
            for cost, bits_delta, assignment in state_rows:
                for option in unit.options:
                    opt_delta = max(float(option.bits_total) - unit_min_bits, 0.0)
                    next_delta = bits_delta + opt_delta
                    next_bin = int(round(next_delta / bin_width))
                    next_assignment = dict(assignment)
                    next_assignment[unit.name] = option.fmt
                    bucket = next_states.setdefault(next_bin, [])
                    bucket.append((
                        cost + float(option.omega_ii),
                        next_delta,
                        next_assignment,
                    ))
        states = {
            idx: _prune_state_bucket(rows, beam)
            for idx, rows in next_states.items()
        }

    floor_bits = sum(min_bits_by_unit.values())
    rows: list[SmrfCandidate] = []
    for bucket_rows in states.values():
        for rank, (cost, bits_delta, assignment) in enumerate(bucket_rows):
            candidate = _candidate_from_unit_assignment(
                assignment,
                units_by_name=units_by_name,
                total_params=total_params,
                source="pareto_dp",
                rank=rank,
            )
            bits_total = float(floor_bits + bits_delta)
            if abs(candidate.bits_total - bits_total) > max(bin_width, 1.0):
                # Actual bits are authoritative; this diagnostic catches only
                # impossible state bookkeeping errors.
                raise AssertionError("SMRF DP state bits drifted from assignment bits")
            if float(bpp_min) <= candidate.achieved_bpp <= float(bpp_max):
                rows.append(candidate)
    rows = _dedupe_candidates(rows)
    rows.sort(key=lambda row: (row.achieved_bpp, row.surrogate_loss, row.assignment_hash))
    retained_states = sum(len(bucket) for bucket in states.values())
    max_bucket_size = max((len(bucket) for bucket in states.values()), default=0)
    return rows, {
        "enabled": True,
        "open_items": len(units),
        "bins": len(states),
        "beam_per_bin": beam,
        "bit_precision_bpp": float(bit_precision_bpp),
        "generated_candidates": len(rows),
        "retained_states": int(retained_states),
        "max_bucket_size": int(max_bucket_size),
    }


def _prune_state_bucket(
    rows: Sequence[tuple[float, float, dict[str, str]]],
    limit: int,
) -> list[tuple[float, float, dict[str, str]]]:
    best_by_hash: dict[str, tuple[float, float, dict[str, str]]] = {}
    for row in rows:
        digest = assignment_hash(row[2])
        previous = best_by_hash.get(digest)
        if previous is None or (row[0], row[1]) < (previous[0], previous[1]):
            best_by_hash[digest] = row
    ordered = sorted(
        best_by_hash.values(),
        key=lambda row: (float(row[0]), float(row[1]), sorted(row[2].items())),
    )
    return list(ordered[: max(int(limit), 1)])


def _dedupe_candidates(rows: Sequence[SmrfCandidate]) -> list[SmrfCandidate]:
    best: dict[str, SmrfCandidate] = {}
    for row in rows:
        previous = best.get(row.assignment_hash)
        if previous is None or (
            row.surrogate_loss,
            row.achieved_bpp,
            _source_priority(row.source),
            row.source,
        ) < (
            previous.surrogate_loss,
            previous.achieved_bpp,
            _source_priority(previous.source),
            previous.source,
        ):
            best[row.assignment_hash] = row
    return list(best.values())


def _source_priority(source: str) -> int:
    if source == "pareto_dp":
        return 0
    if source == "lambda_sweep":
        return 1
    if source == "included_assignment":
        return 2
    return 3


def surrogate_frontier(rows: Sequence[SmrfCandidate]) -> list[SmrfCandidate]:
    ordered = sorted(rows, key=lambda row: (row.achieved_bpp, row.surrogate_loss))
    frontier: list[SmrfCandidate] = []
    best_loss = float("inf")
    for row in ordered:
        if row.surrogate_loss < best_loss - 1e-12 or not frontier:
            frontier.append(row)
            best_loss = row.surrogate_loss
    return frontier


def _kneedle_index(rows: Sequence[SmrfCandidate]) -> int:
    if not rows:
        return -1
    if len(rows) < 3:
        return min(range(len(rows)), key=lambda idx: (rows[idx].surrogate_loss, rows[idx].achieved_bpp))
    xs = [row.achieved_bpp for row in rows]
    ys = [row.surrogate_loss for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin or ymax <= ymin:
        return min(range(len(rows)), key=lambda idx: (ys[idx], xs[idx]))
    scores = []
    for x, y in zip(xs, ys, strict=True):
        x_norm = (x - xmin) / (xmax - xmin)
        y_norm = (y - ymin) / (ymax - ymin)
        scores.append(y_norm - (1.0 - x_norm))
    return min(range(len(scores)), key=lambda idx: (scores[idx], ys[idx], xs[idx]))


def select_validation_candidates(
    rows: Sequence[SmrfCandidate],
    *,
    limit: int = 9,
) -> list[SmrfCandidate]:
    frontier = surrogate_frontier(rows)
    if not frontier:
        return []
    requested = max(int(limit), 0)
    if requested == 0 or requested >= len(frontier):
        return list(frontier)
    selected: dict[str, SmrfCandidate] = {}
    for row in (
        frontier[0],
        frontier[-1],
        min(frontier, key=lambda item: (item.surrogate_loss, item.achieved_bpp)),
    ):
        selected[row.assignment_hash] = row
    knee_idx = _kneedle_index(frontier)
    if knee_idx >= 0:
        selected[frontier[knee_idx].assignment_hash] = frontier[knee_idx]
    remaining = max(requested - len(selected), 0)
    if remaining > 0:
        if remaining == 1:
            indices = [len(frontier) // 2]
        else:
            indices = [
                round(idx * (len(frontier) - 1) / max(remaining - 1, 1))
                for idx in range(remaining)
            ]
        for idx in indices:
            selected[frontier[int(idx)].assignment_hash] = frontier[int(idx)]
            if len(selected) >= requested:
                break
    out = list(selected.values())
    out.sort(key=lambda row: (row.achieved_bpp, row.surrogate_loss, row.assignment_hash))
    return out[:requested]


def _candidate_from_expanded_assignment(
    assignment: Mapping[str, str],
    *,
    units_by_name: Mapping[str, du.DecisionUnit],
    total_params: int,
    source: str,
    label: str | None = None,
) -> SmrfCandidate:
    unit_assignment: dict[str, str] = {}
    expanded = {
        str(name): fr.canonical_format_name(fmt)
        for name, fmt in assignment.items()
    }
    for unit in units_by_name.values():
        direct = expanded.get(unit.name)
        member_formats = {
            expanded[member]
            for member in unit.member_qnames
            if member in expanded
        }
        if direct is not None:
            member_formats.add(direct)
        if len(member_formats) != 1:
            detail = ", ".join(
                f"{member}={expanded.get(member, '<missing>')}"
                for member in unit.member_qnames
            )
            raise ValueError(
                f"{label or source}: assignment does not provide one coherent "
                f"format for SMRF unit {unit.name!r}: {detail}"
            )
        unit_assignment[unit.name] = next(iter(member_formats))
    return _candidate_from_unit_assignment(
        unit_assignment,
        units_by_name=units_by_name,
        total_params=total_params,
        source=source,
        label=label,
    )


def _with_hamming_from_anchor(
    rows: Sequence[SmrfCandidate],
    anchor: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    canonical_anchor = (
        {
            str(name): fr.canonical_format_name(fmt)
            for name, fmt in anchor.items()
        }
        if anchor is not None
        else None
    )
    for row in rows:
        record = row.to_row()
        if canonical_anchor is not None:
            record["hamming_from_anchor"] = sum(
                1
                for name, fmt in row.assignment.items()
                if canonical_anchor.get(name) != fmt
            )
        out.append(record)
    return out


def _frontier_diagnostics(rows: Sequence[SmrfCandidate]) -> dict[str, Any]:
    frontier = surrogate_frontier(rows)
    bpp_gaps = [
        float(frontier[idx].achieved_bpp - frontier[idx - 1].achieved_bpp)
        for idx in range(1, len(frontier))
    ]
    loss_inversions = sum(
        1
        for idx in range(1, len(frontier))
        if frontier[idx].surrogate_loss > frontier[idx - 1].surrogate_loss + 1e-12
    )
    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row.source] = source_counts.get(row.source, 0) + 1
    return {
        "surrogate_frontier_points": len(frontier),
        "source_counts": dict(sorted(source_counts.items())),
        "max_bpp_gap": max(bpp_gaps, default=0.0),
        "mean_bpp_gap": (
            sum(bpp_gaps) / len(bpp_gaps)
            if bpp_gaps
            else 0.0
        ),
        "surrogate_loss_inversions": int(loss_inversions),
        "bpp_min_generated": min((row.achieved_bpp for row in rows), default=None),
        "bpp_max_generated": max((row.achieved_bpp for row in rows), default=None),
    }


def generate_archive_payload(
    payload: Mapping[str, Any],
    *,
    bpp_min: float = 4.0,
    bpp_max: float = 8.0,
    n_lambdas: int = 21,
    bit_precision_bpp: float = 0.001,
    beam_per_bin: int = 4,
    validation_candidates: int = 9,
    include_assignments: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    units = _all_units(payload)
    units_by_name = {unit.name: unit for unit in units}
    total_params = _total_params(units)
    lambda_rows = solve_lagrangian_assignments(
        payload,
        n_lambdas=n_lambdas,
        bpp_min=bpp_min,
        bpp_max=bpp_max,
    )
    dp_rows, dp_meta = solve_pareto_archive_assignments(
        payload,
        bpp_min=bpp_min,
        bpp_max=bpp_max,
        bit_precision_bpp=bit_precision_bpp,
        beam_per_bin=beam_per_bin,
    )
    generated = _dedupe_candidates([*lambda_rows, *dp_rows])
    generated.sort(key=lambda row: (row.achieved_bpp, row.surrogate_loss, row.assignment_hash))
    selected = select_validation_candidates(
        generated,
        limit=validation_candidates,
    )
    included_rows: list[SmrfCandidate] = []
    for label, assignment in (include_assignments or {}).items():
        included = _candidate_from_expanded_assignment(
            assignment,
            units_by_name=units_by_name,
            total_params=total_params,
            source="included_assignment",
            label=str(label),
        )
        included_rows.append(included)
    for row in included_rows:
        selected.append(row)
    anchor = included_rows[0].assignment if included_rows else None
    return {
        "schema": ARCHIVE_SCHEMA,
        "meta": {
            "source_schema": payload.get("schema"),
            "mode": "smrf_archive_candidate_generation",
            "bpp_min": float(bpp_min),
            "bpp_max": float(bpp_max),
            "n_lambdas": int(n_lambdas),
            "bit_precision_bpp": float(bit_precision_bpp),
            "beam_per_bin": int(beam_per_bin),
            "validation_candidates": int(validation_candidates),
            "dp": dp_meta,
            "n_generated": len(generated),
            "n_selected": len(selected),
            "n_included_assignments": len(included_rows),
            "diagnostics": _frontier_diagnostics(generated),
            "quality_note": (
                "SMRF rows are candidates only; promotion requires real KL "
                "validation through prismaquant.validate_assignments_kl."
            ),
        },
        "generated": _with_hamming_from_anchor(generated, anchor),
        "surrogate_frontier": _with_hamming_from_anchor(surrogate_frontier(generated), anchor),
        "validation_candidates": _with_hamming_from_anchor(selected, anchor),
    }


def write_candidate_archive(
    archive: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "smrf_archive.json"
    archive_path.write_text(json.dumps(archive, indent=2, sort_keys=True) + "\n")

    candidates_dir = out / "assignments"
    candidates_dir.mkdir(exist_ok=True)
    manifest_rows = []
    for idx, row in enumerate(archive.get("validation_candidates") or []):
        digest = str(row["assignment_hash"])
        requested_label = row.get("label")
        if requested_label:
            safe_label = "".join(
                ch if ch.isalnum() or ch in {"_", "-", "."} else "_"
                for ch in str(requested_label)
            ).strip("._-")
            label = f"{safe_label or 'included'}_bpp_{float(row['achieved_bpp']):.4f}_{digest}"
        else:
            label = f"smrf_{idx:03d}_bpp_{float(row['achieved_bpp']):.4f}_{digest}"
        path = candidates_dir / f"{label}.json"
        path.write_text(json.dumps({
            "schema": "prismaquant.smrf.assignment.v1",
            "label": label,
            "source": row.get("source"),
            "assignment_hash": digest,
            "achieved_bpp": float(row["achieved_bpp"]),
            "surrogate_loss": float(row["surrogate_loss"]),
            "assignment": dict(row["assignment"]),
        }, indent=2, sort_keys=True) + "\n")
        manifest_rows.append({
            "label": label,
            "path": str(path),
            "assignment_hash": digest,
            "achieved_bpp": float(row["achieved_bpp"]),
            "surrogate_loss": float(row["surrogate_loss"]),
            "source": row.get("source"),
            "hamming_from_anchor": row.get("hamming_from_anchor"),
        })
    manifest = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "archive": str(archive_path),
        "candidates": manifest_rows,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def generate_archive_candidates(
    payload: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    archive = generate_archive_payload(payload, **kwargs)
    if output_dir is not None:
        manifest = write_candidate_archive(archive, output_dir)
        archive = dict(archive)
        archive["manifest"] = manifest
    return archive


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected JSON object")
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SMRF candidate assignments from a decision-unit cost payload."
    )
    parser.add_argument(
        "--payload",
        help="Existing decision-unit payload JSON. Mutually exclusive with --probe/--costs.",
    )
    parser.add_argument(
        "--probe",
        help="Probe pickle used to build a SMRF payload from allocator costs.",
    )
    parser.add_argument(
        "--costs",
        help="Cost pickle used to build a SMRF payload from allocator costs.",
    )
    parser.add_argument(
        "--l3-costs",
        help=(
            "L3 propagated-cost pickle used to build a SMRF payload from "
            "propagated_end_kl values."
        ),
    )
    parser.add_argument(
        "--baseline-assignment",
        help=(
            "Assignment or layer_config JSON used with --l3-costs. Unmeasured "
            "linears are frozen to this baseline so bpp remains full-model."
        ),
    )
    parser.add_argument(
        "--include-validation-assignment",
        action="append",
        default=[],
        help=(
            "Additional label=assignment.json row to include in the validation "
            "manifest, normally a matched standard-PQ baseline. The assignment "
            "must provide one coherent format for each SMRF decision unit."
        ),
    )
    parser.add_argument("--model", help="HF model path for profile detection.")
    parser.add_argument("--formats", default=None)
    parser.add_argument("--target-profile", default=None)
    parser.add_argument(
        "--no-aggregate-siblings",
        action="store_true",
        help="Do not group profile-defined fused siblings into one SMRF unit.",
    )
    parser.add_argument(
        "--no-include-frozen-baseline",
        action="store_true",
        help="With --l3-costs, omit unmeasured baseline-frozen linears.",
    )
    parser.add_argument(
        "--write-payload",
        help="Optional path for the generated decision-unit payload JSON.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bpp-min", type=float, default=4.0)
    parser.add_argument("--bpp-max", type=float, default=8.0)
    parser.add_argument("--n-lambdas", type=int, default=21)
    parser.add_argument("--bit-precision-bpp", type=float, default=0.001)
    parser.add_argument("--beam-per-bin", type=int, default=4)
    parser.add_argument("--validation-candidates", type=int, default=9)
    args = parser.parse_args(argv)

    if args.payload:
        if args.probe or args.costs or args.l3_costs or args.baseline_assignment:
            raise ValueError(
                "--payload cannot be combined with --probe/--costs/--l3-costs"
            )
        payload = _load_json(args.payload)
    else:
        format_names = (
            [item.strip() for item in str(args.formats).split(",") if item.strip()]
            if args.formats
            else None
        )
        if args.l3_costs:
            if not args.probe or not args.baseline_assignment:
                raise ValueError(
                    "--l3-costs requires --probe and --baseline-assignment"
                )
            if args.costs:
                raise ValueError("--costs and --l3-costs are mutually exclusive")
            payload = decision_unit_payload_from_l3_cost_files(
                probe=args.probe,
                l3_costs=args.l3_costs,
                baseline_assignment=args.baseline_assignment,
                model_path=args.model,
                formats=format_names,
                target_profile=args.target_profile,
                aggregate_siblings=not bool(args.no_aggregate_siblings),
                include_frozen_baseline=not bool(args.no_include_frozen_baseline),
            )
        else:
            if not args.probe or not args.costs:
                raise ValueError(
                    "provide either --payload, --probe/--costs, or "
                    "--probe/--l3-costs/--baseline-assignment"
                )
            payload = decision_unit_payload_from_probe_cost_files(
                probe=args.probe,
                costs=args.costs,
                model_path=args.model,
                formats=format_names,
                target_profile=args.target_profile,
                aggregate_siblings=not bool(args.no_aggregate_siblings),
            )
    if args.write_payload:
        payload_path = Path(args.write_payload)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    included = {
        label: assignment
        for label, assignment in (
            _parse_labeled_assignment(value)
            for value in (args.include_validation_assignment or [])
        )
    }
    archive = generate_archive_payload(
        payload,
        bpp_min=args.bpp_min,
        bpp_max=args.bpp_max,
        n_lambdas=args.n_lambdas,
        bit_precision_bpp=args.bit_precision_bpp,
        beam_per_bin=args.beam_per_bin,
        validation_candidates=args.validation_candidates,
        include_assignments=included,
    )
    manifest = write_candidate_archive(archive, args.output_dir)
    print(
        "[smrf] generated "
        f"{archive['meta']['n_generated']} archive row(s), "
        f"{archive['meta']['n_selected']} validation candidate(s)",
        flush=True,
    )
    print(
        "[smrf] units "
        f"blocks={len(payload.get('blocks') or {})} "
        f"singletons={len(payload.get('singletons') or {})}",
        flush=True,
    )
    if args.write_payload:
        print(f"[smrf] payload -> {args.write_payload}", flush=True)
    print(f"[smrf] manifest -> {Path(args.output_dir) / 'manifest.json'}", flush=True)
    if manifest["candidates"]:
        best = min(manifest["candidates"], key=lambda row: row["surrogate_loss"])
        print(
            "[smrf] best surrogate "
            f"{best['label']} bpp={best['achieved_bpp']:.6f} "
            f"loss={best['surrogate_loss']:.8g}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
