"""MSE-driven assignment promotion utilities.

This is a post-allocation policy: start with an existing per-Linear assignment,
then spend a bounded amount of extra bit budget on groups whose current format
accounts for the most measured local output MSE per added bit.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from prismaquant import format_registry as fr
from prismaquant.kl_measurement import assignment_bit_total


_LAYER_RE = re.compile(r"(?:^|[.])layers[.](\d+)[.]")
_VISUAL_BLOCK_RE = re.compile(r"(?:^|[.])visual[.]blocks[.](\d+)[.]")


@dataclass(frozen=True)
class PromotionCandidate:
    key: str
    category: str
    layer: str
    members: tuple[str, ...]
    current_formats: dict[str, int]
    target_format: str
    output_mse_removed: float
    weight_mse_param_removed: float
    predicted_dloss_removed: float
    bits_delta: float
    bpp_delta: float
    score: float
    missing_cost_count: int = 0
    non_finite_count: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "category": self.category,
            "layer": self.layer,
            "members": list(self.members),
            "member_count": len(self.members),
            "current_formats": dict(self.current_formats),
            "target_format": self.target_format,
            "output_mse_removed": float(self.output_mse_removed),
            "weight_mse_param_removed": float(self.weight_mse_param_removed),
            "predicted_dloss_removed": float(self.predicted_dloss_removed),
            "bits_delta": float(self.bits_delta),
            "bpp_delta": float(self.bpp_delta),
            # +inf is a valid ordering score (a blown-up measurement ranks
            # first) but is not valid JSON; emit null + the explicit flag.
            "score": float(self.score) if math.isfinite(self.score) else None,
            "non_finite": bool(self.non_finite_count > 0),
            "non_finite_count": int(self.non_finite_count),
            "missing_cost_count": int(self.missing_cost_count),
        }


def semantic_category(name: str) -> str:
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


def layer_config_from_assignment(assignment: Mapping[str, str]) -> dict[str, dict]:
    return {
        str(name): fr.get_format(_canonical(fmt)).autoround_config()
        for name, fmt in sorted(assignment.items())
    }


def build_mse_promotion_assignment(
    assignment: Mapping[str, str],
    *,
    costs: Mapping[str, object],
    stats: Mapping[str, Mapping],
    categories: Sequence[str],
    target_format: str = "BF16",
    max_bpp_delta: float | None = None,
    target_bpp: float | None = None,
    group_by: str = "layer_category",
    metric: str = "output_mse_per_bit",
    profile=None,
) -> dict[str, object]:
    """Return promoted assignment and an auditable report."""
    assignment_c = {
        str(name): _canonical(fmt)
        for name, fmt in assignment.items()
        if str(name) in stats
    }
    specs = _specs_for_assignment(assignment_c, extra=(target_format,))
    params = _total_params(stats, assignment_c)
    base_bits = assignment_bit_total(stats, assignment_c, specs)
    base_bpp = base_bits / max(float(params), 1.0)
    if target_bpp is not None:
        allowed_bpp_delta = max(float(target_bpp) - float(base_bpp), 0.0)
    elif max_bpp_delta is not None:
        allowed_bpp_delta = max(float(max_bpp_delta), 0.0)
    else:
        allowed_bpp_delta = float("inf")
    allowed_bits_delta = allowed_bpp_delta * float(params)

    wanted_categories = {str(category).strip() for category in categories if str(category).strip()}
    target_fmt = _canonical(target_format)
    candidates = build_promotion_candidates(
        assignment_c,
        costs=costs,
        stats=stats,
        categories=wanted_categories,
        target_format=target_fmt,
        group_by=group_by,
        metric=metric,
        params=params,
        profile=profile,
    )

    promoted = dict(assignment_c)
    selected: list[PromotionCandidate] = []
    skipped_budget: list[PromotionCandidate] = []
    spent_bits = 0.0
    for candidate in candidates:
        if candidate.bits_delta <= 0.0:
            continue
        if spent_bits + candidate.bits_delta > allowed_bits_delta + 1e-6:
            skipped_budget.append(candidate)
            continue
        for member in candidate.members:
            promoted[member] = target_fmt
        selected.append(candidate)
        spent_bits += candidate.bits_delta

    promoted_bits = assignment_bit_total(stats, promoted, specs)
    selected_output_mse = sum(candidate.output_mse_removed for candidate in selected)
    selected_predicted = sum(candidate.predicted_dloss_removed for candidate in selected)
    selected_weight = sum(candidate.weight_mse_param_removed for candidate in selected)
    baseline_output_mse = _assignment_output_mse(assignment_c, costs)
    baseline_predicted = _assignment_predicted_dloss(assignment_c, costs)
    report = {
        "schema": "prismaquant.mse_promotion.v1",
        "target_format": target_fmt,
        "categories": sorted(wanted_categories),
        "group_by": str(group_by),
        "metric": str(metric),
        "params": int(params),
        "base_bits": float(base_bits),
        "base_bpp": float(base_bpp),
        "target_bpp": None if target_bpp is None else float(target_bpp),
        "max_bpp_delta": (
            None if math.isinf(allowed_bpp_delta) else float(allowed_bpp_delta)
        ),
        "promoted_bits": float(promoted_bits),
        "promoted_bpp": float(promoted_bits / max(float(params), 1.0)),
        "actual_bpp_delta": float((promoted_bits - base_bits) / max(float(params), 1.0)),
        "baseline_output_mse_sum": float(baseline_output_mse),
        "selected_output_mse_removed": float(selected_output_mse),
        "selected_output_mse_removed_pct": _pct(selected_output_mse, baseline_output_mse),
        "baseline_predicted_dloss_sum": float(baseline_predicted),
        "selected_predicted_dloss_removed": float(selected_predicted),
        "selected_weight_mse_param_removed": float(selected_weight),
        "selected_group_count": len(selected),
        "selected_member_count": sum(len(candidate.members) for candidate in selected),
        "candidate_count": len(candidates),
        "budget_skipped_count": len(skipped_budget),
        "base_format_counts": dict(sorted(Counter(assignment_c.values()).items())),
        "promoted_format_counts": dict(sorted(Counter(promoted.values()).items())),
        "selected": [candidate.to_json() for candidate in selected],
        "budget_skipped": [candidate.to_json() for candidate in skipped_budget[:50]],
        "top_candidates": [candidate.to_json() for candidate in candidates[:50]],
        "category_summary": _category_summary(selected, baseline_output_mse),
    }
    return {"assignment": promoted, "report": report}


def build_promotion_candidates(
    assignment: Mapping[str, str],
    *,
    costs: Mapping[str, object],
    stats: Mapping[str, Mapping],
    categories: set[str],
    target_format: str,
    group_by: str,
    metric: str,
    params: int,
    profile=None,
) -> list[PromotionCandidate]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, current_fmt in assignment.items():
        category = semantic_category(name)
        if categories and category not in categories:
            continue
        if _canonical(current_fmt) == target_format:
            continue
        if name not in stats:
            continue
        if _stats_indicates_packed_expert(stats.get(name, {})):
            continue
        key = _group_key(name, group_by, profile=profile)
        grouped[key].append(name)

    candidates: list[PromotionCandidate] = []
    for key, members in sorted(grouped.items()):
        output_mse = 0.0
        weight_mse_param = 0.0
        predicted_dloss = 0.0
        bits_delta = 0.0
        missing_cost = 0
        non_finite_counts: Counter[str] = Counter()
        format_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        layer_counts: Counter[str] = Counter()
        for member in sorted(members):
            current_fmt = _canonical(assignment[member])
            format_counts[current_fmt] += 1
            category_counts[semantic_category(member)] += 1
            layer_counts[layer_number(member)] += 1
            entry = _cost_entry(costs, member, current_fmt)
            if entry is None:
                missing_cost += 1
            else:
                target_entry = _cost_entry(costs, member, target_format)
                output_mse += _promotion_gain(
                    entry, target_entry, "output_mse", non_finite_counts,
                )
                weight_mse_param += (
                    _promotion_gain(
                        entry, target_entry, "weight_mse", non_finite_counts,
                    )
                    * float(_n_params(stats.get(member, {})))
                )
                predicted_dloss += _promotion_gain(
                    entry, target_entry, "predicted_dloss", non_finite_counts,
                )
            bits_delta += _bits_delta(stats[member], current_fmt, target_format)
        if bits_delta <= 0.0:
            continue
        bpp_delta = bits_delta / max(float(params), 1.0)
        if non_finite_counts.get(_METRIC_NUMERATOR_FIELD.get(metric, ""), 0) > 0:
            # A non-finite measured cost means the measurement blew up on
            # this group: that is the CATASTROPHIC end of the ranking, not
            # the free end. Coercing it to 0.0 sorted it dead last
            # (priority inversion, audit 2026-07-02 §3.4).
            score = float("inf")
        else:
            score = _candidate_score(
                metric,
                output_mse=output_mse,
                predicted_dloss=predicted_dloss,
                weight_mse_param=weight_mse_param,
                bits_delta=bits_delta,
            )
        if math.isnan(score):
            continue
        candidates.append(
            PromotionCandidate(
                key=key,
                category=category_counts.most_common(1)[0][0],
                layer=layer_counts.most_common(1)[0][0],
                members=tuple(sorted(members)),
                current_formats=dict(sorted(format_counts.items())),
                target_format=target_format,
                output_mse_removed=float(output_mse),
                weight_mse_param_removed=float(weight_mse_param),
                predicted_dloss_removed=float(predicted_dloss),
                bits_delta=float(bits_delta),
                bpp_delta=float(bpp_delta),
                score=float(score),
                missing_cost_count=int(missing_cost),
                non_finite_count=int(sum(non_finite_counts.values())),
            )
        )
    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            -candidate.output_mse_removed,
            candidate.bits_delta,
            candidate.key,
        )
    )
    return candidates


def build_promotion_candidate_report(
    assignment: Mapping[str, str],
    *,
    costs: Mapping[str, object],
    stats: Mapping[str, Mapping],
    categories: Sequence[str],
    target_format: str = "BF16",
    group_by: str = "layer_category",
    metric: str = "output_mse_per_bit",
    profile=None,
) -> dict[str, object]:
    """Build auditable promotion candidates without selecting a budget.

    The returned candidates are the same objects consumed by
    ``build_mse_promotion_assignment``.  ``current_format_overrides`` is keyed
    by candidate key and is the paired-KL candidate lane: the target group keeps
    its current assignment while the paired baseline lane promotes the same
    group to ``target_format``.
    """
    assignment_c = {
        str(name): _canonical(fmt)
        for name, fmt in assignment.items()
        if str(name) in stats
    }
    specs = _specs_for_assignment(assignment_c, extra=(target_format,))
    params = _total_params(stats, assignment_c)
    base_bits = assignment_bit_total(stats, assignment_c, specs)
    wanted_categories = {
        str(category).strip()
        for category in categories
        if str(category).strip()
    }
    target_fmt = _canonical(target_format)
    candidates = build_promotion_candidates(
        assignment_c,
        costs=costs,
        stats=stats,
        categories=wanted_categories,
        target_format=target_fmt,
        group_by=group_by,
        metric=metric,
        params=params,
        profile=profile,
    )
    overrides = {
        candidate.key: {
            member: assignment_c[member]
            for member in candidate.members
        }
        for candidate in candidates
    }
    return {
        "schema": "prismaquant.mse_promotion.candidates.v1",
        "assignment": assignment_c,
        "target_format": target_fmt,
        "categories": sorted(wanted_categories),
        "group_by": str(group_by),
        "metric": str(metric),
        "params": int(params),
        "base_bits": float(base_bits),
        "base_bpp": float(base_bits / max(float(params), 1.0)),
        "candidates": candidates,
        "current_format_overrides": overrides,
    }


def _group_key(name: str, group_by: str, *, profile=None) -> str:
    category = semantic_category(name)
    layer = layer_number(name)
    mode = str(group_by)
    if mode == "name":
        return str(name)
    if mode in {"serving_unit", "fused_unit"}:
        fused = _fused_sibling_group(profile, name)
        if fused:
            return f"fused:{fused}"
        return f"tensor:{name}"
    if mode == "category":
        return category
    if mode == "layer_category":
        return f"{category}.layer_{layer}"
    raise ValueError(
        "group_by must be one of: name, serving_unit, fused_unit, "
        "layer_category, category; "
        f"got {group_by!r}"
    )


def _fused_sibling_group(profile, name: str) -> str | None:
    if profile is None:
        return None
    group_fn = getattr(profile, "fused_sibling_group", None)
    if group_fn is None:
        return None
    try:
        group = group_fn(str(name))
    except Exception:
        return None
    return str(group) if group else None


# Which accumulated field feeds each metric's numerator (a non-finite
# measured value in that field makes the group's score +inf = rank first).
_METRIC_NUMERATOR_FIELD = {
    "output_mse_per_bit": "output_mse",
    "output_mse": "output_mse",
    "predicted_dloss_per_bit": "predicted_dloss",
    "weight_mse_per_bit": "weight_mse",
}


def _measured_value(value: object) -> tuple[float, bool, bool]:
    """Return ``(finite_value, present, non_finite)`` for a measured field.

    ``present`` is False when the field is absent or unparseable. A present
    but non-finite value (inf/NaN) is a blown-up measurement: callers must
    rank the group as maximal priority, never coerce it to 0.0.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0, False, False
    if math.isfinite(out):
        return out, True, False
    return 0.0, True, True


def _promotion_gain(
    entry: Mapping,
    target_entry: Mapping | None,
    field: str,
    non_finite_counts: Counter,
) -> float:
    """Measured error removed by promoting one member for one cost field.

    When the target format's cost entry was measured, the benefit is the
    ``current - target`` delta, clamped >= 0 (scoring the full current-format
    error overstates the benefit of lossy targets and can invert the
    ranking — audit 2026-07-02 §3.4b). Only when the target entry (or the
    field within it) is genuinely absent — e.g. BF16 rows are typically not
    in the cost payload because BF16 is lossless passthrough — do we fall
    back to the historical current-only score. Non-finite current values are
    flagged (rank-first) and excluded from the finite sum; a non-finite
    target measurement is treated as absent.
    """
    current, present, non_finite = _measured_value(entry.get(field))
    if non_finite:
        non_finite_counts[field] += 1
        return 0.0
    if not present:
        return 0.0
    if target_entry is not None:
        target, t_present, t_non_finite = _measured_value(
            target_entry.get(field)
        )
        if t_present and not t_non_finite:
            return max(current - target, 0.0)
    return current


def _candidate_score(
    metric: str,
    *,
    output_mse: float,
    predicted_dloss: float,
    weight_mse_param: float,
    bits_delta: float,
) -> float:
    denom = max(float(bits_delta), 1e-30)
    if metric == "output_mse_per_bit":
        return float(output_mse) / denom
    if metric == "output_mse":
        return float(output_mse)
    if metric == "predicted_dloss_per_bit":
        return float(predicted_dloss) / denom
    if metric == "weight_mse_per_bit":
        return float(weight_mse_param) / denom
    raise ValueError(
        "metric must be one of: output_mse_per_bit, output_mse, "
        f"predicted_dloss_per_bit, weight_mse_per_bit; got {metric!r}"
    )


def _category_summary(
    selected: Sequence[PromotionCandidate],
    baseline_output_mse: float,
) -> list[dict[str, object]]:
    by_category: dict[str, list[PromotionCandidate]] = defaultdict(list)
    for candidate in selected:
        by_category[candidate.category].append(candidate)
    rows = []
    for category, candidates in sorted(by_category.items()):
        output_mse = sum(candidate.output_mse_removed for candidate in candidates)
        bits = sum(candidate.bits_delta for candidate in candidates)
        rows.append({
            "category": category,
            "group_count": len(candidates),
            "member_count": sum(len(candidate.members) for candidate in candidates),
            "output_mse_removed": float(output_mse),
            "output_mse_removed_pct": _pct(output_mse, baseline_output_mse),
            "bits_delta": float(bits),
            "score_output_mse_per_bit": float(output_mse / max(bits, 1e-30)),
        })
    rows.sort(key=lambda row: -float(row["output_mse_removed"]))
    return rows


def _specs_for_assignment(
    assignment: Mapping[str, str],
    *,
    extra: Sequence[str] = (),
) -> dict[str, fr.FormatSpec]:
    specs: dict[str, fr.FormatSpec] = {}
    for fmt in set(assignment.values()) | {str(item) for item in extra}:
        spec = fr.get_format(_canonical(fmt))
        specs[spec.name] = spec
        specs[_canonical(spec.name)] = spec
        for alias in fr.aliases_for(spec.name):
            specs[alias] = spec
    return specs


def _bits_delta(stats_entry: Mapping, current_fmt: str, target_fmt: str) -> float:
    current = fr.get_format(_canonical(current_fmt))
    target = fr.get_format(_canonical(target_fmt))
    shape = _shape_from_stats(stats_entry)
    return float(
        8 * target.memory_bytes_for_shape(shape)
        - 8 * current.memory_bytes_for_shape(shape)
    )


def _shape_from_stats(stats_entry: Mapping) -> tuple[int, ...]:
    memory_shapes = stats_entry.get("_shape")
    if isinstance(memory_shapes, Sequence) and memory_shapes:
        return tuple(int(v) for v in memory_shapes)
    if stats_entry.get("shape") is not None:
        return tuple(int(v) for v in stats_entry["shape"])
    if stats_entry.get("out_features") and stats_entry.get("in_features"):
        return (int(stats_entry["out_features"]), int(stats_entry["in_features"]))
    if stats_entry.get("n_params"):
        return (int(stats_entry["n_params"]),)
    raise ValueError(f"could not infer shape from stats entry: {stats_entry!r}")


def _cost_entry(costs: Mapping[str, object], name: str, fmt: str) -> Mapping | None:
    per_name = costs.get(name)
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


def _stats_indicates_packed_expert(stats_entry: Mapping) -> bool:
    return bool(
        stats_entry.get("_packed_experts_module")
        or stats_entry.get("_packed_param")
        or int(stats_entry.get("num_experts", 0) or 0) > 0
    )


def _assignment_output_mse(
    assignment: Mapping[str, str],
    costs: Mapping[str, object],
) -> float:
    total = 0.0
    for name, fmt in assignment.items():
        if _canonical(fmt) == "BF16":
            continue
        entry = _cost_entry(costs, name, _canonical(fmt))
        if entry is not None:
            total += _finite_float(entry.get("output_mse"))
    return float(total)


def _assignment_predicted_dloss(
    assignment: Mapping[str, str],
    costs: Mapping[str, object],
) -> float:
    total = 0.0
    for name, fmt in assignment.items():
        if _canonical(fmt) == "BF16":
            continue
        entry = _cost_entry(costs, name, _canonical(fmt))
        if entry is not None:
            total += _finite_float(entry.get("predicted_dloss"))
    return float(total)


def _n_params(stats_entry: Mapping) -> int:
    if stats_entry.get("n_params"):
        return int(stats_entry["n_params"])
    shape = _shape_from_stats(stats_entry)
    out = 1
    for dim in shape:
        out *= int(dim)
    return int(out)


def _total_params(stats: Mapping[str, Mapping], assignment: Mapping[str, str]) -> int:
    return sum(_n_params(stats[name]) for name in assignment if name in stats)


def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _canonical(fmt: object) -> str:
    return fr.canonical_format_name(str(fmt).strip().upper())


def _pct(num: float, den: float) -> float | None:
    if not math.isfinite(den) or abs(den) <= 1e-30:
        return None
    return float(100.0 * float(num) / float(den))
