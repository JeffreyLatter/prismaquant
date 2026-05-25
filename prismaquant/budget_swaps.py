"""Budget-neutral empirical swap candidate builder.

The propagated-sensitivity path answers "where do extra bits help?"  This
module builds the next empirical question: "which low-risk demotions can pay
for those promotions under the same bit budget?"
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from prismaquant import format_registry as fr
from prismaquant.mse_promotion import (
    _cost_entry,
    _finite_float,
    _group_key,
    _n_params,
    _stats_indicates_packed_expert,
    layer_number,
    semantic_category,
)


@dataclass(frozen=True)
class SwapUnit:
    key: str
    category: str
    layer: str
    members: tuple[str, ...]
    current_formats: dict[str, int]
    current_bits: float
    promotion_target_formats: dict[str, str]
    promotion_bits_added: float
    promotion_output_mse_removed: float
    propagated_kl: float
    propagated_kl_per_added_bit: float | None
    demotion_target_formats: dict[str, str]
    demotion_bits_saved: float
    demotion_output_mse_added: float
    demotion_predicted_dloss_added: float

    @property
    def promotion_score(self) -> float:
        if self.propagated_kl_per_added_bit is not None:
            return float(self.propagated_kl_per_added_bit)
        return self.promotion_output_mse_removed / max(self.promotion_bits_added, 1e-30)

    @property
    def demotion_risk_per_saved_bit(self) -> float:
        primary = self.demotion_output_mse_added
        if primary <= 0.0:
            primary = self.demotion_predicted_dloss_added
        return primary / max(self.demotion_bits_saved, 1e-30)

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "category": self.category,
            "layer": self.layer,
            "members": list(self.members),
            "member_count": len(self.members),
            "current_formats": dict(self.current_formats),
            "current_bits": float(self.current_bits),
            "promotion_target_formats": dict(self.promotion_target_formats),
            "promotion_bits_added": float(self.promotion_bits_added),
            "promotion_output_mse_removed": float(self.promotion_output_mse_removed),
            "propagated_kl": float(self.propagated_kl),
            "propagated_kl_per_added_bit": self.propagated_kl_per_added_bit,
            "demotion_target_formats": dict(self.demotion_target_formats),
            "demotion_bits_saved": float(self.demotion_bits_saved),
            "demotion_output_mse_added": float(self.demotion_output_mse_added),
            "demotion_predicted_dloss_added": float(
                self.demotion_predicted_dloss_added
            ),
            "promotion_score": float(self.promotion_score),
            "demotion_risk_per_saved_bit": float(self.demotion_risk_per_saved_bit),
        }


@dataclass(frozen=True)
class BudgetNeutralSwap:
    key: str
    promotion_unit: SwapUnit
    demotion_units: tuple[SwapUnit, ...]
    override: dict[str, str]
    bits_added: float
    bits_saved: float
    net_bits_delta: float
    promoted_propagated_kl: float
    demotion_local_risk: float

    @property
    def score(self) -> float:
        return (
            self.promoted_propagated_kl
            - self.demotion_local_risk
            - max(self.net_bits_delta, 0.0)
        )

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "promotion_unit": self.promotion_unit.to_json(),
            "demotion_units": [unit.to_json() for unit in self.demotion_units],
            "override": dict(sorted(self.override.items())),
            "bits_added": float(self.bits_added),
            "bits_saved": float(self.bits_saved),
            "net_bits_delta": float(self.net_bits_delta),
            "promoted_propagated_kl": float(self.promoted_propagated_kl),
            "demotion_local_risk": float(self.demotion_local_risk),
            "score": float(self.score),
        }


def build_budget_neutral_swaps(
    assignment: Mapping[str, str],
    *,
    costs: Mapping[str, object],
    stats: Mapping[str, Mapping],
    propagated_report: Mapping[str, object] | None,
    formats: Sequence[str],
    categories: Sequence[str],
    profile=None,
    group_by: str = "serving_unit",
    promotion_target_format: str = "next_higher",
    max_promotions: int = 32,
    max_demotions_per_swap: int = 4,
    max_swaps: int = 64,
    demotion_start_window: int = 8,
    max_net_bpp_increase: float = 0.0,
) -> dict[str, object]:
    """Build auditable swap candidates whose bit delta is <= budget guard.

    Candidate swaps are only proposals.  They are meant to be measured with
    full end-KL before becoming allocation policy.
    """
    formats_c = tuple(_canonical(fmt) for fmt in formats)
    assignment_c = {
        str(name): _canonical(fmt)
        for name, fmt in assignment.items()
        if str(name) in stats
    }
    wanted_categories = {
        str(category).strip()
        for category in categories
        if str(category).strip()
    }
    params = sum(_n_params(stats[name]) for name in assignment_c)
    row_by_key = _propagated_rows_by_key(propagated_report)
    units = build_swap_units(
        assignment_c,
        costs=costs,
        stats=stats,
        row_by_key=row_by_key,
        formats=formats_c,
        categories=wanted_categories,
        profile=profile,
        group_by=group_by,
        promotion_target_format=promotion_target_format,
    )
    promotions = [
        unit for unit in units
        if unit.promotion_bits_added > 0.0 and unit.promotion_target_formats
    ]
    promotions.sort(
        key=lambda unit: (
            -unit.promotion_score,
            -unit.propagated_kl,
            unit.promotion_bits_added,
            unit.key,
        )
    )
    demotions = [
        unit for unit in units
        if unit.demotion_bits_saved > 0.0 and unit.demotion_target_formats
    ]
    demotions.sort(
        key=lambda unit: (
            unit.demotion_risk_per_saved_bit,
            unit.propagated_kl_per_added_bit
            if unit.propagated_kl_per_added_bit is not None
            else 0.0,
            -unit.demotion_bits_saved,
            unit.key,
        )
    )

    allowed_net_bits = max(float(max_net_bpp_increase), 0.0) * max(float(params), 1.0)
    swaps: list[BudgetNeutralSwap] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for promotion in promotions[: max(int(max_promotions), 0)]:
        for start in range(min(max(int(demotion_start_window), 1), len(demotions))):
            chosen: list[SwapUnit] = []
            used_members = set(promotion.members)
            saved = 0.0
            risk = 0.0
            for demotion in demotions[start:]:
                if demotion.key == promotion.key:
                    continue
                if any(member in used_members for member in demotion.members):
                    continue
                chosen.append(demotion)
                used_members.update(demotion.members)
                saved += demotion.demotion_bits_saved
                risk += (
                    demotion.demotion_output_mse_added
                    if demotion.demotion_output_mse_added > 0.0
                    else demotion.demotion_predicted_dloss_added
                )
                if saved + allowed_net_bits >= promotion.promotion_bits_added:
                    break
                if len(chosen) >= max(int(max_demotions_per_swap), 1):
                    break
            if not chosen:
                continue
            net = promotion.promotion_bits_added - saved
            if net > allowed_net_bits + 1e-6:
                continue
            demotion_keys = tuple(unit.key for unit in chosen)
            seen_key = (promotion.key, demotion_keys)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            override = dict(promotion.promotion_target_formats)
            for demotion in chosen:
                override.update(demotion.demotion_target_formats)
            swaps.append(
                BudgetNeutralSwap(
                    key=f"{promotion.key}::paid_by::{'+'.join(demotion_keys)}",
                    promotion_unit=promotion,
                    demotion_units=tuple(chosen),
                    override=override,
                    bits_added=float(promotion.promotion_bits_added),
                    bits_saved=float(saved),
                    net_bits_delta=float(net),
                    promoted_propagated_kl=float(promotion.propagated_kl),
                    demotion_local_risk=float(risk),
                )
            )
            if len(swaps) >= max(int(max_swaps), 0):
                break
        if len(swaps) >= max(int(max_swaps), 0):
            break

    swaps.sort(
        key=lambda swap: (
            -swap.score,
            swap.net_bits_delta,
            -swap.promoted_propagated_kl,
            swap.key,
        )
    )
    return {
        "schema": "prismaquant.budget_neutral_swaps.v1",
        "group_by": str(group_by),
        "categories": sorted(wanted_categories),
        "formats": list(formats_c),
        "promotion_target_format": _promotion_target_label(
            promotion_target_format
        ),
        "params": int(params),
        "max_net_bpp_increase": float(max_net_bpp_increase),
        "unit_count": len(units),
        "promotion_candidate_count": len(promotions),
        "demotion_candidate_count": len(demotions),
        "swap_count": len(swaps),
        "top_promotions": [unit.to_json() for unit in promotions[:50]],
        "top_demotions": [unit.to_json() for unit in demotions[:50]],
        "swaps": [swap.to_json() for swap in swaps],
    }


def build_swap_units(
    assignment: Mapping[str, str],
    *,
    costs: Mapping[str, object],
    stats: Mapping[str, Mapping],
    row_by_key: Mapping[str, Mapping[str, object]],
    formats: Sequence[str],
    categories: set[str],
    profile=None,
    group_by: str = "serving_unit",
    promotion_target_format: str = "next_higher",
) -> list[SwapUnit]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, fmt in assignment.items():
        category = semantic_category(name)
        if categories and category not in categories:
            continue
        if name not in stats:
            continue
        if _stats_indicates_packed_expert(stats.get(name, {})):
            continue
        grouped[_group_key(name, group_by, profile=profile)].append(name)

    units: list[SwapUnit] = []
    target_fmt = (
        None
        if _is_next_higher_target(promotion_target_format)
        else _canonical(promotion_target_format)
    )
    for key, members_unsorted in sorted(grouped.items()):
        members = tuple(sorted(members_unsorted))
        current_formats = Counter(_canonical(assignment[name]) for name in members)
        category_counts = Counter(semantic_category(name) for name in members)
        layer_counts = Counter(layer_number(name) for name in members)
        current_bits = sum(_bits_for(stats[name], assignment[name]) for name in members)

        promotion_target_formats: dict[str, str] = {}
        promotion_bits_added = 0.0
        promotion_output_mse_removed = 0.0
        for name in members:
            current_fmt = _canonical(assignment[name])
            promote_fmt = (
                _next_higher_format(
                    stats[name],
                    costs=costs,
                    name=name,
                    current_fmt=current_fmt,
                    formats=formats,
                )
                if target_fmt is None
                else target_fmt
            )
            if promote_fmt is None or current_fmt == promote_fmt:
                continue
            added = _bits_for(stats[name], promote_fmt) - _bits_for(stats[name], current_fmt)
            if added <= 0.0:
                continue
            if (
                promote_fmt != "BF16"
                and _cost_entry(costs, name, promote_fmt) is None
            ):
                continue
            promotion_target_formats[name] = promote_fmt
            promotion_bits_added += added
            promotion_output_mse_removed += max(
                _entry_output_mse(costs, name, current_fmt)
                - _entry_output_mse(costs, name, promote_fmt),
                0.0,
            )

        demotion_target_formats: dict[str, str] = {}
        demotion_bits_saved = 0.0
        demotion_output_mse_added = 0.0
        demotion_predicted_dloss_added = 0.0
        for name in members:
            current_fmt = _canonical(assignment[name])
            lower_fmt = _next_lower_format(
                stats[name],
                costs=costs,
                name=name,
                current_fmt=current_fmt,
                formats=formats,
            )
            if lower_fmt is None:
                continue
            saved = _bits_for(stats[name], current_fmt) - _bits_for(stats[name], lower_fmt)
            if saved <= 0.0:
                continue
            demotion_target_formats[name] = lower_fmt
            demotion_bits_saved += saved
            demotion_output_mse_added += max(
                _entry_output_mse(costs, name, lower_fmt)
                - _entry_output_mse(costs, name, current_fmt),
                0.0,
            )
            demotion_predicted_dloss_added += max(
                _entry_predicted_dloss(costs, name, lower_fmt)
                - _entry_predicted_dloss(costs, name, current_fmt),
                0.0,
            )

        row = row_by_key.get(key, {})
        propagated_kl = _finite_float(row.get("propagated_kl"))
        propagated_per_bit_value = row.get("propagated_kl_per_added_bit")
        propagated_per_bit = (
            None
            if propagated_per_bit_value is None
            else _finite_float(propagated_per_bit_value)
        )
        units.append(
            SwapUnit(
                key=key,
                category=category_counts.most_common(1)[0][0],
                layer=layer_counts.most_common(1)[0][0],
                members=members,
                current_formats=dict(sorted(current_formats.items())),
                current_bits=float(current_bits),
                promotion_target_formats=dict(sorted(promotion_target_formats.items())),
                promotion_bits_added=float(promotion_bits_added),
                promotion_output_mse_removed=float(promotion_output_mse_removed),
                propagated_kl=float(propagated_kl),
                propagated_kl_per_added_bit=propagated_per_bit,
                demotion_target_formats=dict(sorted(demotion_target_formats.items())),
                demotion_bits_saved=float(demotion_bits_saved),
                demotion_output_mse_added=float(demotion_output_mse_added),
                demotion_predicted_dloss_added=float(demotion_predicted_dloss_added),
            )
        )
    return units


def _propagated_rows_by_key(
    report: Mapping[str, object] | None,
) -> dict[str, Mapping[str, object]]:
    if not isinstance(report, Mapping):
        return {}
    out: dict[str, Mapping[str, object]] = {}
    for row in report.get("rows", ()):
        if isinstance(row, Mapping) and row.get("key"):
            out[str(row["key"])] = row
    return out


def _next_lower_format(
    stats_entry: Mapping,
    *,
    costs: Mapping[str, object],
    name: str,
    current_fmt: str,
    formats: Sequence[str],
) -> str | None:
    current_bits = _bits_for(stats_entry, current_fmt)
    candidates: list[tuple[float, str]] = []
    for fmt in formats:
        fmt_c = _canonical(fmt)
        bits = _bits_for(stats_entry, fmt_c)
        if bits >= current_bits - 1e-9:
            continue
        if _cost_entry(costs, name, fmt_c) is None:
            continue
        candidates.append((bits, fmt_c))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _next_higher_format(
    stats_entry: Mapping,
    *,
    costs: Mapping[str, object],
    name: str,
    current_fmt: str,
    formats: Sequence[str],
) -> str | None:
    current_bits = _bits_for(stats_entry, current_fmt)
    candidates: list[tuple[float, str]] = []
    for fmt in formats:
        fmt_c = _canonical(fmt)
        bits = _bits_for(stats_entry, fmt_c)
        if bits <= current_bits + 1e-9:
            continue
        if fmt_c != "BF16" and _cost_entry(costs, name, fmt_c) is None:
            continue
        candidates.append((bits, fmt_c))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def _bits_for(stats_entry: Mapping, fmt: str) -> float:
    spec = fr.get_format(_canonical(fmt))
    return float(8 * spec.memory_bytes_for_shape(_shape_from_stats(stats_entry)))


def _shape_from_stats(stats_entry: Mapping) -> tuple[int, ...]:
    value = stats_entry.get("_shape")
    if isinstance(value, Sequence) and value:
        return tuple(int(dim) for dim in value)
    if stats_entry.get("shape") is not None:
        return tuple(int(dim) for dim in stats_entry["shape"])
    if stats_entry.get("out_features") and stats_entry.get("in_features"):
        return (int(stats_entry["out_features"]), int(stats_entry["in_features"]))
    if stats_entry.get("n_params"):
        return (int(stats_entry["n_params"]),)
    raise ValueError(f"could not infer shape from stats entry: {stats_entry!r}")


def _entry_output_mse(costs: Mapping[str, object], name: str, fmt: str) -> float:
    entry = _cost_entry(costs, name, _canonical(fmt))
    return 0.0 if entry is None else _finite_float(entry.get("output_mse"))


def _entry_predicted_dloss(costs: Mapping[str, object], name: str, fmt: str) -> float:
    entry = _cost_entry(costs, name, _canonical(fmt))
    return 0.0 if entry is None else _finite_float(entry.get("predicted_dloss"))


def _canonical(fmt: object) -> str:
    return fr.canonical_format_name(str(fmt).strip().upper())


def _is_next_higher_target(value: object) -> bool:
    return str(value).strip().lower().replace("-", "_") in {
        "next",
        "next_higher",
        "next_higher_format",
    }


def _promotion_target_label(value: object) -> str:
    if _is_next_higher_target(value):
        return "next_higher"
    return _canonical(value)
