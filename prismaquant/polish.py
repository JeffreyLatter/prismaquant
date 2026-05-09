"""Measured single-flip polish for per-Linear assignments.

Takes a per-Linear starting assignment, the model + calibration set, and a
list of fused-sibling decision units.  For each pass, evaluates every single-
unit format flip and accepts the one with the lowest measured KL — provided
that KL strictly improves on the current state by more than a noise floor.

Properties:

* Monotone non-regressing on the measured KL: each accepted move strictly
  improves the metric we ship on.
* Fused-sibling aware: a flip applies to all members of a fused group at
  once, preserving serving compatibility.
* No surrogate involved: this is pure measured KL gating.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from prismaquant import decision_units as du
from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    stage_multimodal,
)
from prismaquant.calibration_data import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.kl_measurement import measure_assignment_kl
from prismaquant.model_profiles import DefaultProfile, detect_profile


@dataclass
class PolishStep:
    """One accepted (or rejected) move during coord-descent polish."""

    pass_index: int
    accepted: bool
    unit: str
    from_fmt: str
    to_fmt: str
    kl_before: float
    kl_after: float
    candidates_evaluated: int


@dataclass
class PolishResult:
    initial_kl: float
    final_kl: float
    final_assignment: dict[str, str]
    steps: list[PolishStep] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    n_kl_measurements: int = 0
    diagnostics: dict = field(default_factory=dict)


def _override_unit(
    assignment: Mapping[str, str],
    unit: du.DecisionUnit,
    fmt: str,
) -> dict[str, str]:
    """Return a new assignment with all members of ``unit`` set to ``fmt``."""
    out = dict(assignment)
    canonical = fr.canonical_format_name(str(fmt))
    for member in unit.member_qnames:
        out[member] = canonical
    return out


def _current_unit_format(
    assignment: Mapping[str, str],
    unit: du.DecisionUnit,
) -> str | None:
    """Return the format applied to the first member; None if missing."""
    for member in unit.member_qnames:
        fmt = assignment.get(member)
        if fmt is not None:
            return fr.canonical_format_name(str(fmt))
    return None


def _assignment_bits(
    units: Sequence[du.DecisionUnit],
    assignment: Mapping[str, str],
) -> float:
    """Return the total bits implied by ``assignment`` across ``units``."""
    total = 0.0
    for unit in units:
        fmt = _current_unit_format(assignment, unit)
        if fmt is None:
            continue
        for opt in unit.options:
            if opt.fmt == fmt:
                total += float(opt.bits_total)
                break
    return total


def _surrogate_candidate_priority(
    units: Sequence[du.DecisionUnit],
    pairs_by_block: Mapping[str, Sequence[du.BlockPair]],
    current_assignment: Mapping[str, str],
) -> dict[tuple[str, str], float]:
    """Return surrogate-predicted ΔCost for every (unit, target_fmt) flip.

    For unit i with current format c_i and trial format f, the predicted
    delta in surrogate cost is::

        ΔΩ_i = Ω_ii(i, f) − Ω_ii(i, c_i)
             + Σ_{j ≠ i, in block_i} [Ω_ij(i, j; f, c_j) − Ω_ij(i, j; c_i, c_j)]

    More-negative ΔΩ_i predicts larger improvement.  We only compute pair
    deltas for *intra-block* edges since that's the only Ω_ij we measured.
    """
    by_unit_format = {
        unit.name: {opt.fmt: opt for opt in unit.options}
        for unit in units
    }
    current_fmts: dict[str, str] = {}
    for unit in units:
        for member in unit.member_qnames:
            if member in current_assignment:
                current_fmts[unit.name] = fr.canonical_format_name(
                    str(current_assignment[member])
                )
                break
    # Build a quick lookup of the block each unit is in.
    units_by_block: dict[str, list[du.DecisionUnit]] = {}
    for unit in units:
        units_by_block.setdefault(unit.block_id, []).append(unit)

    priority: dict[tuple[str, str], float] = {}
    for unit in units:
        cur_fmt = current_fmts.get(unit.name)
        if cur_fmt is None:
            continue
        cur_opt = by_unit_format[unit.name].get(cur_fmt)
        if cur_opt is None:
            continue
        block_pairs = pairs_by_block.get(unit.block_id, ())
        for opt in unit.options:
            if opt.fmt == cur_fmt:
                continue
            delta = float(opt.omega_ii) - float(cur_opt.omega_ii)
            for pair in block_pairs:
                if unit.name not in (pair.unit_a, pair.unit_b):
                    continue
                other_name = pair.unit_b if pair.unit_a == unit.name else pair.unit_a
                other_fmt = current_fmts.get(other_name)
                if other_fmt is None:
                    continue
                if pair.unit_a == unit.name:
                    new_omega = pair.omega_ij.get((opt.fmt, other_fmt))
                    if new_omega is None:
                        new_omega = pair.omega_ij.get((other_fmt, opt.fmt))
                    cur_omega = pair.omega_ij.get((cur_fmt, other_fmt))
                    if cur_omega is None:
                        cur_omega = pair.omega_ij.get((other_fmt, cur_fmt))
                else:
                    new_omega = pair.omega_ij.get((other_fmt, opt.fmt))
                    if new_omega is None:
                        new_omega = pair.omega_ij.get((opt.fmt, other_fmt))
                    cur_omega = pair.omega_ij.get((other_fmt, cur_fmt))
                    if cur_omega is None:
                        cur_omega = pair.omega_ij.get((cur_fmt, other_fmt))
                if new_omega is None:
                    new_omega = 0.0
                if cur_omega is None:
                    cur_omega = 0.0
                delta += float(new_omega) - float(cur_omega)
            priority[(unit.name, opt.fmt)] = float(delta)
    return priority


def coord_descent_polish(
    model,
    calib_ids: torch.Tensor,
    ref_log_probs,
    *,
    units: Sequence[du.DecisionUnit],
    starting_assignment: Mapping[str, str],
    profile=None,
    work_root: str | Path | None = None,
    noise_floor: float = 1e-5,
    max_passes: int = 8,
    bits_budget: float | None = None,
    bits_tolerance: float = 0.0,
    pairs_by_block: Mapping[str, Sequence[du.BlockPair]] | None = None,
    steepest_first: bool = False,
    use_frozen_weight_cache: bool = False,
    production_weight_cache=None,
    pinned_units: set[str] | None = None,
    direction: str = "bottom_up",
    kl_budget: float | None = None,
    delta_quantize: bool | None = None,
    kl_scope: str | None = "last_token",
    include_activation_quant: bool = True,
    weight_session_spill_to_disk: bool = False,
    weight_session_snapshot_dir: str | Path | None = None,
    restore_bf16_on_exit: bool = False,
    progress_callback=None,
) -> PolishResult:
    """Polish a starting assignment via real-KL-gated single-flip moves.

    Two directions are supported (Buchbinder et al. 2012 "double greedy"
    formulation, adapted for constrained submodular maximization):

    bottom_up (default):
      Minimize KL subject to ``bits ≤ bits_budget``.  Start at a low-bpp
      feasible state (typically all-NVFP4 + lm_head=BF16).  Accept a flip
      iff ``current_kl - trial_kl > noise_floor`` and the trial does not
      bust ``bits_budget + bits_tolerance``.  Each move has KL going
      DOWN; bpp generally increases (with occasional decreasing
      "rebalance" moves).

    top_down:
      Minimize bits subject to ``KL ≤ kl_budget``.  Start at a high-bpp
      lossy-but-low-KL state (typically all-BF16).  Accept a flip iff
      ``current_bits - trial_bits > 0`` (bits go DOWN by at least one
      candidate's worth) and ``trial_kl <= kl_budget``.  Among the
      bits-decreasing budget-respecting moves, pick the one with the
      largest ``bits_decrease`` for the smallest ``kl_increase`` —
      specifically the one with the lowest ``trial_kl``.

    Running both directions traces the Pareto frontier from opposite
    ends.  Where the two assignments agree on a unit's format at
    similar (bpp, KL) operating points, you have high confidence;
    disagreement localizes Linears with non-trivial interactions
    (1-flip greedy stops at a local optimum on either side).

    Cost: O(passes × units × (formats - 1)) KL measurements per
    direction.  For Qwen 0.6B with 113 fused units × 2 non-current
    formats × 4 passes ≈ 900 measurements; per-call latency dominates.

    References:
      - Buchbinder, Feldman, Naor, Schwartz, "A tight linear time
        (1/2)-approximation for unconstrained submodular maximization,"
        FOCS 2012.
      - Sviridenko, "A note on maximizing a submodular set function
        subject to a knapsack constraint," ORL 2004.
    """
    direction = direction.lower().strip()
    if direction not in ("bottom_up", "top_down"):
        raise ValueError(
            f"direction must be 'bottom_up' or 'top_down', got {direction!r}"
        )
    # Defensive: if any DecisionUnit's members aren't in the starting
    # assignment, polish will silently miss it and bits accounting will
    # undercount.  Detect early so we don't ship wrong numbers.  This
    # guarded a real, hard-to-find bug where a payload file used the
    # wrong key for ``members`` and parse_payload fell back to using
    # the unit name as a sole pseudo-member; the entire fused-sibling
    # decision space disappeared from polish.
    # Use ALL not ANY: a fused-sibling unit with one member present and
    # another missing would silently undercount bits for the missing
    # member, which is the same class of bug the original assertion was
    # added to catch (Gemini 2026-05-06 review).
    _missing_units = []
    for _u in units:
        _members = list(getattr(_u, "member_qnames", []) or [])
        if not _members:
            continue
        if not all(m in starting_assignment for m in _members):
            _missing_units.append(_u.name)
    if _missing_units:
        raise ValueError(
            f"{len(_missing_units)} DecisionUnits have ≥1 member_qname "
            f"missing from starting_assignment — polish would silently "
            f"undercount bits for the missing member(s).  Likely causes: "
            f"the payload's unit entries are missing the 'members' field "
            f"(parse_payload falls back to [unit_name]); or fused-sibling "
            f"members were dropped during assignment construction.  "
            f"Sample: {_missing_units[:5]}"
        )
    if work_root is None:
        work_root = Path(tempfile.mkdtemp(prefix="prismaquant_polish_"))
        cleanup_work_root = True
    else:
        work_root = Path(work_root)
        work_root.mkdir(parents=True, exist_ok=True)
        cleanup_work_root = False

    # When delta_quantize_polish is requested, polish opens a
    # WeightSession at the start, materializes the assignment ONCE on
    # model.params, and per trial only swaps the diffed unit's weight
    # in place (with revert-on-reject).  The hooks installed by
    # PerturbedActivationCache skip their clone+restore because the
    # PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT flag is set.  Net effect:
    # ~50 GB of GPU/UMA pressure removed on 27B-class polish.
    #
    # Selection precedence: explicit kwarg > env var > default(False).
    # The env var is preserved for back-compat; new callers should pass
    # delta_quantize=True.  We restore the prior value of
    # PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT in the finally block so
    # subsequent in-process evaluators (L3 measurement, validation,
    # etc.) see the same environment they would absent this call.
    import os as _os
    if delta_quantize is None:
        use_delta_quant = (
            production_weight_cache is not None
            and _os.environ.get(
                "PRISMAQUANT_DELTA_QUANTIZE_POLISH", "",
            ).strip().lower() in {"1", "true", "yes", "on"}
        )
    else:
        use_delta_quant = bool(delta_quantize) and (
            production_weight_cache is not None
        )
    weight_session = None
    _ext_wm_prev = _os.environ.get("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT")
    if use_delta_quant:
        from prismaquant.weight_session import WeightSession
        _os.environ["PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT"] = "1"

    start = time.time()
    n_measurements = 0
    try:
        current = {
            str(k): fr.canonical_format_name(str(v))
            for k, v in starting_assignment.items()
        }
        # MED-4: validate that fused-sibling members share a format in the
        # starting assignment.  Polish itself only flips whole units (so it
        # *maintains* coherence after the first move), but if the caller
        # passes in an already-incoherent assignment we'd silently start
        # measuring a state that violates the vLLM fused-Linear contract.
        for unit in units:
            members = list(getattr(unit, "member_qnames", []) or [])
            if len(members) <= 1:
                continue
            fmts = {current.get(m) for m in members if m in current}
            fmts.discard(None)
            if len(fmts) > 1:
                raise ValueError(
                    f"starting_assignment violates sibling consistency for "
                    f"DecisionUnit {unit.name!r}: members {members} carry "
                    f"formats {sorted(fmts)}"
                )
        if use_delta_quant:
            weight_session = WeightSession(
                model,
                production_weight_cache=production_weight_cache,
                snapshot_dir=(
                    str(weight_session_snapshot_dir)
                    if weight_session_snapshot_dir is not None
                    else (
                        str(work_root / "weight_session_snapshots")
                        if weight_session_spill_to_disk else None
                    )
                ),
            )
            weight_session.initialize(current, units)
            if progress_callback is not None:
                progress_callback({
                    "event": "weight_session_initialized",
                    "n_bf16_snapshots": weight_session.n_bf16_snapshots,
                    "diagnostics": weight_session.diagnostics(),
                })
        current_kl = measure_assignment_kl(
            model, current, calib_ids, ref_log_probs,
            work_root=work_root, profile=profile,
            use_frozen_weight_cache=use_frozen_weight_cache,
            production_weight_cache=production_weight_cache, rng_seed=0,
            kl_scope=kl_scope,
            include_activation_quant=include_activation_quant,
            stream_ref_log_probs=kl_scope == "full_sequence",
        )
        n_measurements += 1
        initial_kl = float(current_kl)
        steps: list[PolishStep] = []
        if progress_callback is not None:
            progress_callback({
                "event": "starting", "kl": float(initial_kl),
            })

        # Per-unit format → bits lookup, so we can quickly compute the bits
        # delta of a candidate flip without rebuilding the whole sum.
        unit_by_name = {unit.name: unit for unit in units}
        option_bits: dict[tuple[str, str], float] = {}
        for unit in units:
            for opt in unit.options:
                option_bits[(unit.name, opt.fmt)] = float(opt.bits_total)

        current_bits = _assignment_bits(units, current)
        if bits_budget is None:
            budget_bits = float("inf")
        else:
            budget_bits = float(bits_budget) + float(bits_tolerance)
        if progress_callback is not None and bits_budget is not None:
            progress_callback({
                "event": "budget_set",
                "starting_bits": float(current_bits),
                "budget_bits": float(budget_bits),
            })

        for pass_idx in range(max_passes):
            best_move: tuple[du.DecisionUnit, str] | None = None
            if direction == "bottom_up":
                # Sentinel: any move with KL strictly below current beats this.
                best_kl_after = float(current_kl)
                best_bits_after = float(current_bits)
            else:
                # Top-down: any feasible bits-decreasing move with KL ≤ budget
                # beats `inf`; ties broken by larger bit decrease.
                best_kl_after = float("inf")
                best_bits_after = float(current_bits)
            candidates_this_pass = 0

            # Order candidates by surrogate priority when steepest_first is
            # enabled and a pairs_by_block payload was supplied.  Otherwise
            # sweep lexicographically (greedy-best).
            pinned = set(pinned_units or set())

            def _is_pinned(unit: du.DecisionUnit) -> bool:
                if unit.name in pinned:
                    return True
                for member in getattr(unit, "member_qnames", []) or []:
                    if member in pinned:
                        return True
                return False

            order: list[tuple[du.DecisionUnit, du.FormatCost]]
            if steepest_first and pairs_by_block is not None:
                priority = _surrogate_candidate_priority(
                    units, pairs_by_block, current,
                )
                order = []
                for unit in units:
                    if _is_pinned(unit):
                        continue
                    cur_fmt = _current_unit_format(current, unit)
                    if cur_fmt is None:
                        continue
                    for option in unit.options:
                        if option.fmt == cur_fmt:
                            continue
                        order.append((unit, option))
                order.sort(key=lambda pair: priority.get((pair[0].name, pair[1].fmt), 0.0))
            else:
                order = []
                for unit in sorted(units, key=lambda u: u.name):
                    if _is_pinned(unit):
                        continue
                    cur_fmt = _current_unit_format(current, unit)
                    if cur_fmt is None:
                        continue
                    for option in unit.options:
                        if option.fmt == cur_fmt:
                            continue
                        order.append((unit, option))

            for unit, option in order:
                cur_fmt = _current_unit_format(current, unit)
                if cur_fmt is None:
                    continue
                cur_bits = option_bits.get((unit.name, cur_fmt), 0.0)
                trial_bits = current_bits - cur_bits + option_bits.get(
                    (unit.name, option.fmt), 0.0,
                )
                if direction == "bottom_up":
                    # Bottom-up: KL decreases, bits ≤ budget_bits.
                    if trial_bits > budget_bits + 1e-9:
                        continue
                else:
                    # Top-down: bits decrease, KL ≤ kl_budget.
                    if trial_bits >= current_bits - 1e-9:
                        continue
                trial = _override_unit(current, unit, option.fmt)
                # DELTA: stage the unit's format change directly on
                # model.params via WeightSession.  PerturbedActivation
                # Cache's hooks see the new weights already in place
                # (PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT=1) so they
                # skip the per-module clone+restore.  We always revert
                # after the trial — only the accept path persists.
                n_staged = 0
                if weight_session is not None:
                    n_staged = weight_session.stage_unit(unit, option.fmt)
                try:
                    trial_kl = measure_assignment_kl(
                        model, trial, calib_ids, ref_log_probs,
                        work_root=work_root, profile=profile,
                        use_frozen_weight_cache=use_frozen_weight_cache,
                        production_weight_cache=production_weight_cache, rng_seed=0,
                        kl_scope=kl_scope,
                        include_activation_quant=include_activation_quant,
                        stream_ref_log_probs=kl_scope == "full_sequence",
                    )
                finally:
                    if weight_session is not None and n_staged:
                        weight_session.revert_unit_last(n_staged)
                n_measurements += 1
                candidates_this_pass += 1
                if progress_callback is not None:
                    progress_callback({
                        "event": "trial",
                        "pass": pass_idx,
                        "unit": unit.name,
                        "from": cur_fmt,
                        "to": option.fmt,
                        "current_kl": float(current_kl),
                        "trial_kl": float(trial_kl),
                        "trial_bits": float(trial_bits),
                        "best_so_far": float(best_kl_after),
                        "n_measurements": n_measurements,
                    })
                if direction == "bottom_up":
                    if trial_kl < best_kl_after - float(noise_floor):
                        best_kl_after = float(trial_kl)
                        best_bits_after = float(trial_bits)
                        best_move = (unit, option.fmt)
                        if steepest_first:
                            break
                else:
                    # Top-down: must respect kl_budget; among feasible
                    # bits-decreasing moves, prefer the one with smallest
                    # KL (smallest quality cost for an equivalent bit
                    # save) — the textbook "min KL increase per bit
                    # saved" greedy rule.  Ties broken by larger bit
                    # decrease.
                    if kl_budget is not None and trial_kl > kl_budget + float(
                        noise_floor
                    ):
                        continue
                    bits_dec = current_bits - trial_bits
                    best_bits_dec = current_bits - best_bits_after
                    is_better = (
                        trial_kl < best_kl_after - float(noise_floor)
                        or (
                            abs(trial_kl - best_kl_after) <= float(noise_floor)
                            and bits_dec > best_bits_dec + 1e-9
                        )
                    )
                    if is_better:
                        best_kl_after = float(trial_kl)
                        best_bits_after = float(trial_bits)
                        best_move = (unit, option.fmt)
                        if steepest_first:
                            break
            if best_move is None:
                if progress_callback is not None:
                    progress_callback({
                        "event": "pass_no_improvement",
                        "pass": pass_idx,
                        "candidates_evaluated": candidates_this_pass,
                    })
                break
            unit, fmt = best_move
            cur_fmt = _current_unit_format(current, unit) or "BF16"
            current = _override_unit(current, unit, fmt)
            current_bits = best_bits_after
            # DELTA: re-apply the accepted move on the live model and
            # commit so the staged change persists across subsequent
            # trials (no revert).  The next pass's trials see the
            # updated current_assignment in model.params.
            if weight_session is not None:
                n_committed = weight_session.stage_unit(unit, fmt)
                weight_session.commit_unit_last(n_committed)
            step = PolishStep(
                pass_index=pass_idx,
                accepted=True,
                unit=unit.name,
                from_fmt=cur_fmt,
                to_fmt=fmt,
                kl_before=float(current_kl),
                kl_after=float(best_kl_after),
                candidates_evaluated=candidates_this_pass,
            )
            steps.append(step)
            if progress_callback is not None:
                progress_callback({
                    "event": "accept_move",
                    "pass": pass_idx,
                    "unit": unit.name,
                    "from": cur_fmt,
                    "to": fmt,
                    "kl_before": float(current_kl),
                    "kl_after": float(best_kl_after),
                    "improvement": float(current_kl - best_kl_after),
                })
            current_kl = best_kl_after

        return PolishResult(
            initial_kl=float(initial_kl),
            final_kl=float(current_kl),
            final_assignment=dict(current),
            steps=steps,
            elapsed_seconds=float(time.time() - start),
            n_kl_measurements=int(n_measurements),
            diagnostics={
                "kl_scope": kl_scope,
                "include_activation_quant": bool(include_activation_quant),
                "weight_session": (
                    weight_session.diagnostics()
                    if weight_session is not None else None
                ),
            },
        )
    finally:
        if use_delta_quant and restore_bf16_on_exit and weight_session is not None:
            try:
                restore_assignment = {
                    member: "BF16"
                    for unit in units
                    for member in unit.member_qnames
                }
                n_restored = weight_session.apply_assignment(restore_assignment)
                if progress_callback is not None:
                    progress_callback({
                        "event": "weight_session_restored",
                        "n_changed": int(n_restored),
                        "diagnostics": weight_session.diagnostics(),
                    })
            except Exception as exc:
                if progress_callback is not None:
                    progress_callback({
                        "event": "weight_session_restore_error",
                        "error": str(exc),
                    })
        # Restore PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT to its prior
        # value so we do not poison subsequent in-process callers
        # (L3 measurement, validation, model.eval loops) that expect
        # PerturbedActivationCache to install weights normally.
        if use_delta_quant:
            if _ext_wm_prev is None:
                _os.environ.pop("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", None)
            else:
                _os.environ["PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT"] = (
                    _ext_wm_prev
                )
        if cleanup_work_root:
            shutil.rmtree(work_root, ignore_errors=True)


def units_from_payload(payload_path: str | Path) -> list[du.DecisionUnit]:
    """Load DecisionUnits from a decision-unit payload."""
    with Path(payload_path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    blocks, singletons, _pairs = du.parse_payload(payload)
    units: list[du.DecisionUnit] = []
    for unit_list in blocks.values():
        units.extend(unit_list)
    units.extend(singletons)
    return units


def units_and_pairs_from_payload(
    payload_path: str | Path,
) -> tuple[list[du.DecisionUnit], dict[str, list[du.BlockPair]]]:
    """Load DecisionUnits AND intra-block pairs from a decision-unit payload.

    Returns ``(units, pairs_by_block)`` for use with ``steepest_first``.
    """
    with Path(payload_path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    blocks, singletons, pairs = du.parse_payload(payload)
    units: list[du.DecisionUnit] = []
    for unit_list in blocks.values():
        units.extend(unit_list)
    units.extend(singletons)
    return units, dict(pairs)


def _device_from_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _progress_printer(event: dict) -> None:
    kind = event.get("event")
    if kind == "starting":
        print(f"[polish] start KL={event['kl']:.6f}", flush=True)
    elif kind == "accept_move":
        print(
            f"[polish] pass {event['pass']} accept "
            f"{event['unit']} {event['from']}→{event['to']} "
            f"KL {event['kl_before']:.6f} → {event['kl_after']:.6f} "
            f"(Δ {event['improvement']:+.6f})",
            flush=True,
        )
    elif kind == "pass_no_improvement":
        print(
            f"[polish] pass {event['pass']} no improvement "
            f"after {event['candidates_evaluated']} candidates",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coord-descent polish")
    parser.add_argument("--model", required=True)
    parser.add_argument("--payload", required=True,
                        help="Decision-unit payload JSON")
    parser.add_argument("--starting-assignment", required=True,
                        help="Per-Linear assignment JSON (kneedle output)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-calib-samples", type=int, default=2)
    parser.add_argument("--calib-seqlen", type=int, default=128)
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--kl-scope",
        choices=["last_token", "full_sequence"],
        default="last_token",
        help="KL reduction scope.  last_token is the scalable default.",
    )
    parser.add_argument(
        "--no-activation-quant",
        action="store_true",
        help="Disable production activation quantization during polish KL.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-passes", type=int, default=8)
    parser.add_argument("--noise-floor", type=float, default=1e-5)
    parser.add_argument(
        "--bits-budget",
        type=float,
        default=None,
        help=(
            "Total-bits budget for polish moves.  Without this flag polish "
            "may creep toward higher precision monotonically.  Defaults to "
            "the bits of the starting assignment when --bits-budget-mode "
            "is 'starting'."
        ),
    )
    parser.add_argument(
        "--bits-budget-mode",
        choices=["explicit", "starting", "unconstrained"],
        default="starting",
        help=(
            "How to set the polish bits budget when --bits-budget is not "
            "given.  'starting' uses the starting assignment's bits "
            "(default).  'unconstrained' lets polish creep upward freely."
        ),
    )
    parser.add_argument("--bits-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--steepest-first",
        action="store_true",
        help=(
            "Order candidate flips by the surrogate's predicted ΔΩ and "
            "accept the first measured improvement.  Faster than greedy-"
            "best when the surrogate ranks moves accurately around the "
            "current assignment."
        ),
    )
    parser.add_argument(
        "--use-frozen-weight-cache",
        action="store_true",
        help=(
            "Pre-quantize the centered base assignment once and reuse "
            "cached weights across polish trials.  Big speedup for non-"
            "BF16 starting points; OOM-prone at LLM scale."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    units, pairs_by_block = units_and_pairs_from_payload(args.payload)
    if not units:
        raise RuntimeError(f"no decision units in {args.payload}")
    print(
        f"[polish] loaded {len(units)} decision units, "
        f"{sum(len(p) for p in pairs_by_block.values())} pairs from {args.payload}",
        flush=True,
    )

    starting_payload = json.loads(Path(args.starting_assignment).read_text())
    if isinstance(starting_payload, dict) and "assignment" in starting_payload:
        starting_assignment = starting_payload["assignment"]
    else:
        starting_assignment = starting_payload
    print(f"[polish] starting assignment has {len(starting_assignment)} entries",
          flush=True)

    device_str = _device_from_arg(args.device)
    dtype = _dtype_from_name(args.dtype)
    staged, cleanup = stage_multimodal(args.model)
    work_root = Path(tempfile.mkdtemp(prefix="prismaquant_polish_run_"))
    try:
        local_only = bool(args.local_files_only or Path(staged).exists())
        tokenizer = AutoTokenizer.from_pretrained(
            staged, trust_remote_code=True, local_files_only=local_only,
        )
        calib_ids = load_wikitext_calibration_windowed(
            tokenizer, args.n_calib_samples, args.calib_seqlen,
            split=args.calib_split, seed=args.calib_seed,
        )
        load_kwargs = {
            "torch_dtype": dtype, "trust_remote_code": True,
            "local_files_only": local_only,
        }
        if device_str == "cuda":
            load_kwargs["device_map"] = "cuda"
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        if device_str != "cuda":
            model.to(device_str)
        model.eval()
        try:
            profile = detect_profile(args.model)
        except Exception:
            profile = DefaultProfile()
        device = next(model.parameters()).device
        ref_log_probs = cache_reference_log_probs(
            model, calib_ids, device, kl_scope=args.kl_scope,
        )

        # Resolve bits budget
        if args.bits_budget is not None:
            bits_budget = float(args.bits_budget)
        elif args.bits_budget_mode == "starting":
            bits_budget = _assignment_bits(units, starting_assignment)
        elif args.bits_budget_mode == "unconstrained":
            bits_budget = None
        else:
            raise ValueError(f"unsupported bits_budget_mode: {args.bits_budget_mode}")
        if bits_budget is not None:
            print(
                f"[polish] budget = {bits_budget:.0f} bits "
                f"(tolerance ±{args.bits_tolerance:.0f})",
                flush=True,
            )

        result = coord_descent_polish(
            model, calib_ids, ref_log_probs,
            units=units,
            starting_assignment=starting_assignment,
            profile=profile,
            work_root=work_root,
            noise_floor=args.noise_floor,
            max_passes=args.max_passes,
            bits_budget=bits_budget,
            bits_tolerance=args.bits_tolerance,
            pairs_by_block=pairs_by_block,
            steepest_first=bool(args.steepest_first),
            use_frozen_weight_cache=bool(args.use_frozen_weight_cache),
            kl_scope=args.kl_scope,
            include_activation_quant=not bool(args.no_activation_quant),
            progress_callback=_progress_printer,
        )
        out_payload = {
            "schema": "prismaquant.coord_descent_polish.v1",
            "initial_kl": result.initial_kl,
            "final_kl": result.final_kl,
            "improvement": result.initial_kl - result.final_kl,
            "n_steps_accepted": len(result.steps),
            "n_kl_measurements": result.n_kl_measurements,
            "elapsed_seconds": result.elapsed_seconds,
            "steps": [
                {
                    "pass_index": s.pass_index,
                    "unit": s.unit,
                    "from_fmt": s.from_fmt,
                    "to_fmt": s.to_fmt,
                    "kl_before": s.kl_before,
                    "kl_after": s.kl_after,
                    "candidates_evaluated": s.candidates_evaluated,
                }
                for s in result.steps
            ],
            "final_assignment": result.final_assignment,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out_payload, indent=2) + "\n")
        print(
            f"[polish] done initial_kl={result.initial_kl:.6f} "
            f"final_kl={result.final_kl:.6f} "
            f"Δ={result.initial_kl - result.final_kl:+.6f} "
            f"steps={len(result.steps)} measurements={result.n_kl_measurements} "
            f"elapsed={result.elapsed_seconds:.1f}s",
            flush=True,
        )
        print(f"[polish] wrote {args.output}", flush=True)
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
        shutil.rmtree(work_root, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
