"""Solver primitives for PrismaQuant allocation.

This module owns the multi-choice knapsack, candidate scoring, and
serve-time coupling promotions.  ``allocator.py`` keeps the CLI and
re-exports these symbols for backwards compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import format_registry as fr


@dataclass
class Candidate:
    fmt: str
    bits_per_param: float
    memory_bytes: int
    predicted_dloss: float


def _shape_from_stats(entry: dict) -> tuple[int, ...]:
    out_features = int(entry.get("out_features", 0) or 0)
    in_features = int(entry.get("in_features", 0) or 0)
    num_experts = int(entry.get("num_experts", 0) or 0)
    if num_experts > 0 and out_features > 0 and in_features > 0:
        return (num_experts, out_features, in_features)
    if out_features > 0 and in_features > 0:
        return (out_features, in_features)
    n_params = int(entry.get("n_params", 0) or 0)
    return (n_params,)


def predicted_dloss(h_trace: float, weight_mse: float,
                    gain: float = 1.0) -> float:
    """Per-(layer, format) predicted loss under the diagonal-Fisher model."""
    return 0.5 * float(h_trace) * float(weight_mse) * float(gain)


def _group_by_profile(names, profile) -> dict[str, list[str]]:
    """Group Linear names by the profile's fused-sibling key."""
    groups: dict[str, list[str]] = {}
    for name in names:
        key = profile.fused_sibling_group(name) if profile is not None else None
        if key is None:
            continue
        groups.setdefault(key, []).append(name)
    return groups


def _packed_groups_by_profile(names, profile) -> dict[str, list[str]]:
    """Group Linear names by the profile's packed-MoE serving unit key."""
    groups: dict[str, list[str]] = {}
    group_fn = getattr(profile, "packed_expert_format_group", None)
    if not callable(group_fn):
        return groups
    for name in names:
        try:
            key = group_fn(name)
        except Exception:
            key = None
        if key is None:
            continue
        groups.setdefault(key, []).append(name)
    return groups


def _promote_group_components(
    assignment: dict[str, str],
    format_rank: dict[str, int],
    groups: list[list[str]],
) -> dict[str, str]:
    """Promote connected serving-unit components to one shared format."""
    out = dict(assignment)
    parent = {name: name for name in out}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for group in groups:
        members = [m for m in group if m in out]
        if len(members) < 2:
            continue
        first = members[0]
        for member in members[1:]:
            union(first, member)

    components: dict[str, list[str]] = {}
    for name in out:
        components.setdefault(find(name), []).append(name)

    for members in components.values():
        if len(members) < 2:
            continue
        best_fmt = max((out[member] for member in members), key=lambda fmt: format_rank[fmt])
        best_rank = format_rank[best_fmt]
        for member in members:
            if format_rank[out[member]] < best_rank:
                out[member] = best_fmt
    return out


def promote_serving_units(
    assignment: dict[str, str],
    format_rank: dict[str, int],
    *,
    profile=None,
    include_fused: bool = True,
    include_moe: bool = True,
) -> dict[str, str]:
    """Promote all serving-coupled units in one order-independent pass."""
    if profile is None:
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    groups: list[list[str]] = []
    if include_fused:
        groups.extend(_group_by_profile(assignment.keys(), profile).values())
    if include_moe:
        groups.extend(_packed_groups_by_profile(assignment.keys(), profile).values())
    return _promote_group_components(assignment, format_rank, groups)


def fused_siblings(name: str, profile=None) -> tuple[tuple[str, ...], str] | None:
    """Legacy scalar sibling lookup kept for backward compatibility."""
    if profile is None:
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    key = profile.fused_sibling_group(name)
    if key is None:
        return None
    return (name,), key


def promote_moe_pair(
    assignment: dict[str, str],
    format_rank: dict[str, int],
    *,
    profile=None,
) -> dict[str, str]:
    """Promote packed MoE projections that must share one serving format."""
    if profile is None:
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    return promote_serving_units(
        assignment,
        format_rank,
        profile=profile,
        include_fused=False,
        include_moe=True,
    )


def promote_fused(assignment: dict[str, str],
                  format_rank: dict[str, int],
                  profile=None) -> dict[str, str]:
    """Promote each fused-sibling group to one shared serving format."""
    if profile is None:
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    out = promote_serving_units(
        assignment,
        format_rank,
        profile=profile,
        include_fused=True,
        include_moe=False,
    )
    groups = _group_by_profile(assignment.keys(), profile)
    for members_present in groups.values():
        if len(members_present) < 2:
            continue
        ranks = [format_rank[out[m]] for m in members_present]
        best = max(ranks)
        best_fmt = next(k for k, v in format_rank.items() if v == best)
        for m in members_present:
            if format_rank[out[m]] < best:
                out[m] = best_fmt

    for group_key, members in groups.items():
        if len(members) < 2:
            continue
        fmts = {out[m] for m in members}
        if len(fmts) > 1:
            detail = ", ".join(f"{m}={out[m]}" for m in members)
            raise AssertionError(
                f"promote_fused post-check failed for group {group_key!r}: "
                f"siblings have mixed formats after promotion - {detail}. "
                "This produces an unservable artifact."
            )
    return out


def _charged_bins(d_avg_bits: float, bit_precision: float) -> int:
    """Discretize a candidate's average-bit delta into DP bins.

    Conservative by construction: any strictly positive delta is charged at
    least one bin. ``round()`` alone charged 0 bins for deltas below
    ``0.5 * bit_precision``, handing sub-half-bin units free one-directional
    format upgrades the tightening loop could never correct (the achieved
    bits violate the solver's own overshoot tolerance). The maximum
    over-charge is one bin = ``bit_precision`` average bits per unit —
    bounded by the existing precision knob, no new constant. Genuinely-zero
    deltas still cost 0.
    """
    dbins = int(round(d_avg_bits / bit_precision))
    if dbins == 0 and d_avg_bits > 0.0:
        return 1
    return dbins


def solve_allocation(stats: dict, candidates: dict[str, list[Candidate]],
                     target_bits: float, bit_precision: float = 0.001
                     ) -> tuple[dict[str, str], dict[str, Candidate]] | None:
    """Solve multi-choice knapsack in average-bits-per-parameter units."""
    import numpy as np

    names = list(candidates.keys())
    total_params = sum(stats[n]["n_params"] for n in names)
    if total_params == 0:
        return {}

    baselines = {n: min(cs, key=lambda c: c.bits_per_param)
                 for n, cs in candidates.items()}
    min_bits = sum(baselines[n].bits_per_param * stats[n]["n_params"]
                   for n in names) / total_params

    if target_bits < min_bits - 1e-6:
        return None

    excess = target_bits - min_bits
    n_bins = int(round(excess / bit_precision)) + 2

    INF_NEG = -1e30
    dp = np.full(n_bins, INF_NEG, dtype=np.float64)
    dp[0] = 0.0
    choice: list[np.ndarray] = []

    for name in names:
        baseline = baselines[name]
        cs = candidates[name]
        params = stats[name]["n_params"]
        fraction = params / total_params
        baseline_loss = baseline.predicted_dloss
        options = []
        for idx, c in enumerate(cs):
            d_avg_bits = (c.bits_per_param - baseline.bits_per_param) * fraction
            dbins = _charged_bins(d_avg_bits, bit_precision)
            if dbins < 0 or dbins >= n_bins:
                continue
            dgain = baseline_loss - c.predicted_dloss
            options.append((dbins, dgain, idx))
        if not options:
            options = [(0, 0.0, cs.index(baseline))]

        opt_dbins = np.asarray([o[0] for o in options], dtype=np.int32)
        opt_dgain = np.asarray([o[1] for o in options], dtype=np.float64)
        opt_idx = np.asarray([o[2] for o in options], dtype=np.int32)

        new_dp = np.full(n_bins, INF_NEG, dtype=np.float64)
        new_choice = np.full(n_bins, -1, dtype=np.int32)

        for db, dg, idx in zip(opt_dbins, opt_dgain, opt_idx):
            if db == 0:
                candidate_vals = dp + dg
                target_slice = new_dp
                mask = candidate_vals > target_slice
                new_dp = np.where(mask, candidate_vals, new_dp)
                new_choice = np.where(mask, idx, new_choice)
            else:
                candidate_vals = dp[:-db] + dg
                target_slice = new_dp[db:]
                mask = candidate_vals > target_slice
                target_slice[:] = np.where(mask, candidate_vals, target_slice)
                new_choice[db:] = np.where(mask, idx, new_choice[db:])
        dp = new_dp
        choice.append(new_choice)

    if not np.isfinite(dp).any() or dp.max() == INF_NEG:
        return None
    best_b = int(np.argmax(dp))

    assignment: dict[str, str] = {}
    chosen_cands: dict[str, Candidate] = {}
    cur = best_b
    for layer_idx in range(len(names) - 1, -1, -1):
        idx_chosen = int(choice[layer_idx][cur])
        name = names[layer_idx]
        cs = candidates[name]
        if idx_chosen < 0:
            idx_chosen = 0
        chosen = cs[idx_chosen]
        assignment[name] = chosen.fmt
        chosen_cands[name] = chosen
        baseline = baselines[name]
        params = stats[name]["n_params"]
        fraction = params / total_params
        d_avg_bits = (chosen.bits_per_param
                      - baseline.bits_per_param) * fraction
        # Must mirror the forward charge exactly, or the backtrack walks the
        # wrong DP column for the remaining units.
        cur -= _charged_bins(d_avg_bits, bit_precision)
        if cur < 0:
            cur = 0
    return assignment, chosen_cands


def _candidate_for_assignment(
    name: str,
    fmt: str,
    candidates: dict[str, list[Candidate]],
) -> Candidate | None:
    """Resolve the scored candidate for one assignment entry."""
    cands_for_name = candidates.get(name, [])
    for cand in cands_for_name:
        if cand.fmt == fmt:
            return cand
    return None


def compute_achieved(stats: dict, assignment: dict[str, str],
                     format_specs: dict[str, fr.FormatSpec],
                     candidates: dict[str, list[Candidate]] | None = None,
                     ) -> tuple[float, float]:
    """Return ``(avg_bits, total_predicted_dloss)`` for an assignment."""
    total_params = sum(stats[n]["n_params"] for n in assignment)
    total_bits = 0.0
    total_predicted_dloss = 0.0
    cs = candidates or {}
    for n in assignment:
        fmt = assignment[n]
        chosen_cand = _candidate_for_assignment(n, fmt, cs)
        if chosen_cand is not None:
            total_bits += 8.0 * chosen_cand.memory_bytes
            total_predicted_dloss += float(
                getattr(chosen_cand, "predicted_dloss", 0.0))
            continue
        memory_map = stats[n].get("_memory_bytes_by_format")
        if memory_map is not None and fmt in memory_map:
            total_bits += 8.0 * memory_map[fmt]
        else:
            shape = _shape_from_stats(stats[n])
            total_bits += (
                format_specs[fmt].effective_bits_for_shape(shape)
                * stats[n]["n_params"]
            )
    return total_bits / max(total_params, 1), total_predicted_dloss


def compute_assignment_predicted_dloss(
    assignment: dict[str, str],
    candidates: dict[str, list[Candidate]],
) -> float:
    """Sum predicted loss for a concrete assignment."""
    total = 0.0
    for name, fmt in assignment.items():
        chosen = _candidate_for_assignment(name, fmt, candidates)
        if chosen is None:
            raise AssertionError(
                f"assignment {name!r} picked fmt={fmt!r}, but no candidate "
                "exists to price its predicted loss"
            )
        total += chosen.predicted_dloss
    return total


def solve_with_promotion(
    stats: dict,
    candidates: dict[str, list[Candidate]],
    target_bits: float,
    format_specs: dict[str, fr.FormatSpec],
    format_rank: dict[str, int],
    bit_precision: float,
    *,
    no_fused_promote: bool = False,
    overshoot_tolerance: float = 0.01,
    max_iters: int = 40,
    stall_threshold: float = 1e-4,
    stall_grace: int = 3,
    profile=None,
) -> tuple[dict[str, str] | None, float]:
    """Solve, promote coupled tensors, and retry if promotion exceeds budget."""
    tightened = float(target_bits)
    last_assign: dict[str, str] | None = None
    last_achieved = float("nan")
    prev_overshoot = float("inf")
    stall_count = 0
    for _iteration in range(max_iters):
        result = solve_allocation(stats, candidates, tightened, bit_precision)
        if result is None:
            return last_assign, last_achieved
        assign, chosen_cands = result
        del chosen_cands
        assign = promote_serving_units(
            assign,
            format_rank,
            profile=profile,
            include_fused=not no_fused_promote,
            include_moe=True,
        )
        achieved, _ = compute_achieved(
            stats, assign, format_specs,
            candidates=candidates,
        )
        last_assign = assign
        last_achieved = achieved
        overshoot = achieved - target_bits
        if overshoot <= overshoot_tolerance:
            return assign, achieved

        if abs(prev_overshoot - overshoot) < stall_threshold:
            stall_count += 1
            if stall_count >= stall_grace:
                return assign, achieved
        else:
            stall_count = 0
        prev_overshoot = overshoot

        tightened -= overshoot / 2.0
        if tightened <= 0:
            break
    return last_assign, last_achieved
