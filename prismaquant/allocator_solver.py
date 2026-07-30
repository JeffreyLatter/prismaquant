"""Solver primitives for PrismaQuant allocation.

This module owns the multi-choice knapsack, candidate scoring, and
serve-time coupling promotions.  ``allocator.py`` keeps the CLI and
re-exports these symbols for backwards compatibility.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from . import format_registry as fr


class PackedExpertRoleUnknown(RuntimeError):
    """A role split was requested for an expert projection the profile cannot
    name a role for (see ``allocator_candidates._RoleSplitProfile``).

    Raised (not returned as None) because the split is opt-in and already
    hard-gated on a serving profile that declares per-role expert schemes: the
    operator asked for gate_up/down units, so quietly falling back to
    layer-uniform units for the leaves the profile does not recognize would
    ship a DIFFERENT allocation than the one requested, with no signal — and a
    partial fallback is worse than none (some layers split, some not). The
    remedy is one line of declarative spec, so failing loud costs one run and
    fixes the architecture permanently.

    Defined here rather than beside its raiser because both the aggregation
    path (``allocator_candidates``) and the promotion path (this module) must
    catch-and-re-raise it, and ``allocator_candidates`` imports from this
    module — so this is the only home that does not close an import cycle.
    ``allocator_candidates`` re-exports it.
    """

# Read once at import (coarse offline path; no per-call getenv).
_SOLVER_TRACE = os.environ.get("PRISMAQUANT_SOLVER_TRACE", "") not in ("", "0")


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
        except PackedExpertRoleUnknown:
            # A profile that cannot name a packed expert's role under
            # --packed-role-split is a declaration gap, not a "this row has no
            # group" answer: swallowing it here would silently promote the two
            # roles as one unit, i.e. ship a DIFFERENT allocation than the
            # operator asked for. Aggregation raises this first today, so this
            # is defence in depth for any future path that promotes without
            # aggregating.
            raise
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
    """Return ``(avg_bits, total_predicted_dloss)`` for an assignment.

    A priced row (a name present in ``candidates``) whose assigned format has
    no candidate is a HARD ERROR, not a zero-cost row. It means promotion put
    that Linear on a format its own candidate set never offered — the format
    is illegal for it, and pricing the move at 0 Δloss made the unpriced row
    look free. That mattered the moment ``solve_with_promotion`` started
    ratcheting on this Δloss: the contaminated (too-low) sum is exactly what
    the min-Δloss ratchet would prefer. ``compute_assignment_predicted_dloss``
    already refuses the same input, so failing here only moves the same error
    to where the miscosting happens.

    Names ABSENT from ``candidates`` (and the ``candidates=None`` byte-only
    call form) stay on the unpriced byte path: they are not DP rows, so they
    contribute bytes and no Δloss.
    """
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
        if n in cs:
            raise AssertionError(
                f"assignment {n!r} picked fmt={fmt!r}, but no candidate "
                "exists to price its predicted loss (available: "
                f"{sorted(c.fmt for c in cs[n])}). Serving-unit promotion "
                "moved a priced Linear onto a format its candidate set never "
                "offered; scoring it at zero Δloss would bias the solver's "
                "min-Δloss ratchet toward exactly this unpriced state."
            )
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
    diagnostics: dict | None = None,
) -> tuple[dict[str, str] | None, float]:
    """Solve, promote coupled tensors, and retry if promotion exceeds budget.

    Termination contract: the returned
    assignment is always FEASIBLE — ``achieved <= target_bits +
    overshoot_tolerance`` — and, among the feasible iterates seen, the
    one with MINIMUM total predicted Δloss (ties broken toward larger
    achieved bits). Δloss is the solver's actual objective; density is
    not a proxy for it — more bits is not monotonically better (5.5 bpp
    has beaten 6.0 bpp on served PPL), and promotion can flip a serving
    group into a denser-but-worse format. When no
    iterate is feasible within ``max_iters`` the rung is INFEASIBLE and
    ``(None, nan)`` is returned so callers exclude it from the Pareto curve
    and byte-budget bisection. Three silent fallbacks are replaced:

      - ``solve_allocation`` returning None (tightened below the format
        floor) used to return the previous, massively over-target iterate;
      - any undershoot, however deep, used to be accepted immediately
        (rung 6.0 returning achieved 4.95 with 25x worse loss);
      - the stall exit used to return an iterate still far above target.

    Honest scope: only the ``--target-bits`` emit path shipped those
    over-target iterates silently. The byte-budget selector priced them
    at their true (larger) disk size, so its feasibility calls were
    correct — the rung label was just wrong, skewing the Pareto curve's
    x-axis rather than fabricating feasibility.

    The search runs in two phases on the *tightened* DP target:

    1. Damped descent (the old loop's cost profile — the first eval at the
       full target is the only expensive DP; every subsequent eval has a
       smaller bin table): tighten by ``overshoot/2`` until an iterate is
       feasible or the DP floor is reached.
    2. Bracket bisection between the first feasible tightened value and the
       last infeasible one, ratcheting the minimum-Δloss feasible
       assignment.
       Promotion is a coarse step function (one packed-MoE serving group
       flipping format moves the average by tenths of a bit), so
       achieved(tightened) is locally non-monotone; the ratchet keeps
       correctness anyway, and infeasibility is only declared when the
       whole bracket is exhausted with no feasible iterate.

    ``stall_threshold`` is reused as the plateau epsilon for the descent
    acceleration; ``stall_grace`` is retained for call compatibility but
    unused (both phases shrink their search state every evaluation, so
    they cannot stall).

    Pass ``diagnostics`` (an empty dict) to receive the numbers an INFEASIBLE
    verdict needs to be actionable — they are otherwise computed and thrown
    away. The dict is filled IN PLACE, on every return path:

      ``target_bits``, ``min_bits`` (the DP baseline: cheapest-candidate
      average bits, i.e. the format floor), ``overshoot_tolerance``,
      ``evals``, ``feasible``, ``achieved_bits`` / ``predicted_dloss`` (the
      returned iterate, ``None`` when INFEASIBLE), ``closest_achieved_bits``
      / ``closest_tightened_target`` (the least-overshooting iterate seen —
      "how close did it get"), and ``floor_achieved_bits`` (what the DP floor
      itself promoted to, the number that distinguishes "target below the
      format floor" from "serving-group promotion overshoots").
    """
    del stall_grace  # legacy fixed-point knob; see docstring
    stall_eps = float(stall_threshold)
    target = float(target_bits)
    diag = diagnostics if diagnostics is not None else {}
    diag.update({
        "target_bits": target,
        "min_bits": None,
        "overshoot_tolerance": float(overshoot_tolerance),
        "evals": 0,
        "feasible": False,
        "achieved_bits": None,
        "predicted_dloss": None,
        "closest_achieved_bits": None,
        "closest_tightened_target": None,
        "floor_achieved_bits": None,
    })
    names = list(candidates.keys())
    total_params = sum(stats[n]["n_params"] for n in names)
    if total_params <= 0:
        return None, float("nan")
    min_bits = sum(
        min(cs, key=lambda c: c.bits_per_param).bits_per_param
        * stats[n]["n_params"]
        for n, cs in candidates.items()
    ) / total_params
    diag["min_bits"] = float(min_bits)
    if target < min_bits - 1e-6:
        return None, float("nan")

    best_assign: dict[str, str] | None = None
    best_achieved = float("nan")
    best_dloss = float("inf")

    def _evaluate(t: float) -> float | None:
        """Solve+promote at tightened ``t``; ratchet if feasible.

        Returns achieved bits, or None when the DP is infeasible at ``t``.
        The ratchet keeps the feasible iterate with minimum total predicted
        Δloss (the solve objective), tie-broken toward higher achieved bits.
        """
        nonlocal best_assign, best_achieved, best_dloss
        t0 = time.time() if _SOLVER_TRACE else 0.0
        diag["evals"] += 1
        result = solve_allocation(stats, candidates, t, bit_precision)
        if result is None:
            if _SOLVER_TRACE:
                print(f"[solver] target={target:.4f} eval t={t:.4f} -> "
                      f"DP infeasible ({time.time() - t0:.1f}s)", flush=True)
            return None
        assign, chosen_cands = result
        del chosen_cands
        assign = promote_serving_units(
            assign,
            format_rank,
            profile=profile,
            include_fused=not no_fused_promote,
            include_moe=True,
        )
        achieved, predicted = compute_achieved(
            stats, assign, format_specs,
            candidates=candidates,
        )
        if _SOLVER_TRACE:
            print(f"[solver] target={target:.4f} eval t={t:.4f} -> "
                  f"achieved={achieved:.4f} dloss={predicted:.3e} "
                  f"({time.time() - t0:.1f}s)",
                  flush=True)
        if abs(t - min_bits) <= 1e-9:
            diag["floor_achieved_bits"] = float(achieved)
        if achieved - target > overshoot_tolerance:
            closest = diag["closest_achieved_bits"]
            if closest is None or achieved < closest:
                diag["closest_achieved_bits"] = float(achieved)
                diag["closest_tightened_target"] = float(t)
        if achieved - target <= overshoot_tolerance and (
                best_assign is None
                or predicted < best_dloss
                or (predicted == best_dloss and achieved > best_achieved)):
            best_assign = assign
            best_achieved = achieved
            best_dloss = predicted
            diag["feasible"] = True
            diag["achieved_bits"] = float(achieved)
            diag["predicted_dloss"] = float(predicted)
        return achieved

    # Phase 1: damped descent until the first feasible iterate. Promotion
    # makes achieved(tightened) a plateau-and-cliff staircase (a packed-MoE
    # layer only leaves a format when EVERY one of its rows does), so a pure
    # overshoot/2 step can crawl across a plateau for dozens of expensive
    # DP evals. Accelerate: whenever an eval fails to reduce the overshoot
    # meaningfully, double the step.
    evals = 0
    tightened = target
    hi = target  # lowest tightened known to promote OVER the target
    lo = None    # highest tightened known to promote under it (feasible)
    step = None
    prev_overshoot = float("inf")
    while evals < max_iters:
        evals += 1
        achieved = _evaluate(tightened)
        if achieved is None:
            # Below the DP floor: all-baseline promotes to itself, which is
            # feasible for any target >= min_bits, so probe the floor next
            # unless we already did.
            if tightened <= min_bits + 1e-9:
                break
            hi = min(hi, tightened)
            tightened = min_bits
            continue
        if achieved - target <= overshoot_tolerance:
            if achieved >= target - overshoot_tolerance:
                # Within the band on both sides — no denser feasible
                # iterate exists; return the ratcheted min-Δloss one.
                return best_assign, best_achieved
            lo = tightened
            break
        hi = tightened
        if tightened <= min_bits + 1e-9:
            # The floor solve itself promotes over the target: no tighter
            # DP exists, so re-running the identical solve until max_iters
            # cannot help. The rung is infeasible unless a feasible iterate
            # was already ratcheted.
            break
        overshoot = achieved - target
        if step is None:
            step = overshoot / 2.0
        elif prev_overshoot - overshoot < max(step / 4.0, stall_eps):
            # Plateau: tightening by `step` bought back less than a quarter
            # of it — the promoted outcome is pinned by serving-group
            # atomicity. Jump exponentially further instead of crawling.
            step *= 2.0
        else:
            step = overshoot / 2.0
        prev_overshoot = overshoot
        tightened -= step
        if tightened <= min_bits:
            tightened = min_bits

    # Phase 2: bisect (lo, hi) toward the target. The bisection PROPOSES
    # progressively denser feasible iterates; the ratchet in _evaluate
    # KEEPS whichever feasible iterate has the minimum predicted Δloss.
    if lo is not None:
        while evals < max_iters and hi - lo > max(bit_precision, 1e-9):
            evals += 1
            mid = 0.5 * (lo + hi)
            achieved = _evaluate(mid)
            if achieved is not None and achieved - target <= overshoot_tolerance:
                if achieved >= target - overshoot_tolerance:
                    # In-band: densest possible; keep the min-Δloss ratchet.
                    return best_assign, best_achieved
                lo = mid
            else:
                hi = mid

    if best_assign is None:
        if _SOLVER_TRACE:
            print(f"[solver] target={target:.4f} INFEASIBLE after "
                  f"{diag['evals']} evals: floor={min_bits:.4f} "
                  f"floor_achieved={diag['floor_achieved_bits']} "
                  f"closest={diag['closest_achieved_bits']}", flush=True)
        return None, float("nan")
    return best_assign, best_achieved
