"""End-to-end iterated Block-CLADO refinement.

Wraps the full pipeline:

    initial measure (BF16-centered)
    ↓
    λ-sweep + frontier validate + pick best by real KL
    ↓
    coord-descent polish (real-KL gated)
    ↓ (sandwich)
    re-measure block-CLADO centered at polished assignment
    ↓
    λ-sweep + frontier validate + pick best
    ↓
    polish again
    ↓ ...
    until best assignment is stable across iterations.

Each iteration costs ~1× a full block-CLADO measurement plus a polish run.
For Qwen 0.6B that's roughly 5-10 minutes per iteration; for 4B it's
~30-60 minutes; for 27B it's a few hours.

This module orchestrates the existing pieces (measure_block_clado,
block_clado solver, validate_block_clado, coord_descent_polish) — it does
not re-implement them.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch

from prismaquant import block_clado as bc
from prismaquant import coord_descent_polish as cdp
from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    stage_multimodal,
)
from prismaquant.iterate_perturbed_allocation import measure_assignment_kl
from prismaquant.measure_adjoint_l3 import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.measure_block_clado import (
    collect_block_clado,
    discover_blocks,
)
from prismaquant.measure_output_fisher import collect_output_fisher
from prismaquant.model_profiles import DefaultProfile, detect_profile

_VALIDATION_CHECKPOINT_SCHEMA = "prismaquant.block_clado.validation_checkpoint.v1"


@dataclass
class IterationResult:
    iteration: int
    centered_at: str  # "BF16" or "iter_{n-1}_polish"
    payload_path: Path
    sweep_path: Path
    kneedle_label: str
    kneedle_bpp: float
    kneedle_surrogate_cost: float
    best_validated_kl: float
    best_validated_bpp: float
    best_validated_assignment: dict[str, str]
    polished_kl: float
    polished_assignment: dict[str, str]
    polish_steps: int
    elapsed_seconds: float


def assignment_hash(assignment: dict[str, str]) -> str:
    """Stable hash for change detection."""
    items = sorted(assignment.items())
    s = "|".join(f"{k}:{v}" for k, v in items)
    h = 0
    for c in s:
        h = (h * 33 + ord(c)) & 0xFFFFFFFF
    return f"{h:08x}"


def _stable_json_sha256(payload) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def _validation_candidate_key(row: Mapping) -> str:
    return _stable_json_sha256({
        "bpp": float(row["bpp"]),
        "surrogate_cost": float(row["cost_total"]),
        "assignment": sorted((str(k), str(v)) for k, v in row["assignment"].items()),
    })


def _validation_center_key(center_assignment: Mapping[str, str], center_kl: float) -> str:
    return _stable_json_sha256({
        "kind": "center_baseline",
        "center_kl": float(center_kl),
        "assignment": sorted((str(k), str(v)) for k, v in center_assignment.items()),
    })


def _validation_checkpoint_signature(
    *,
    payload: Mapping,
    sorted_rows: Sequence[Mapping],
    indices: Sequence[int],
    center_assignment: Mapping[str, str] | None,
    center_kl: float,
    include_activation_quant: bool,
    validation_delta_quantize: bool,
    production_weight_cache,
) -> dict:
    candidate_keys = [
        _validation_candidate_key(sorted_rows[i])
        for i in indices
    ]
    center_key = (
        _validation_center_key(center_assignment, center_kl)
        if center_assignment is not None and center_kl > 0.0 else None
    )
    signature_payload = {
        "payload_hash": _stable_json_sha256(payload),
        "candidate_keys": candidate_keys,
        "center_key": center_key,
        "include_activation_quant": bool(include_activation_quant),
        "validation_delta_quantize": bool(validation_delta_quantize),
        "production_cache": bool(production_weight_cache is not None),
    }
    return {
        "sha256": _stable_json_sha256(signature_payload),
        "payload": signature_payload,
    }


def _load_validation_checkpoint(
    path: Path,
    signature: Mapping,
) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    if raw.get("schema") != _VALIDATION_CHECKPOINT_SCHEMA:
        return {}
    if raw.get("signature", {}).get("sha256") != signature.get("sha256"):
        return {}
    rows: dict[str, dict] = {}
    for item in raw.get("rows", []):
        key = str(item.get("selection_key", ""))
        row = item.get("row")
        if key and isinstance(row, dict):
            rows[key] = dict(row)
    return rows


def _write_validation_checkpoint(
    path: Path,
    *,
    signature: Mapping,
    rows_by_key: Mapping[str, Mapping],
) -> None:
    payload = {
        "schema": _VALIDATION_CHECKPOINT_SCHEMA,
        "signature": dict(signature),
        "updated_at": time.time(),
        "rows": [
            {"selection_key": key, "row": dict(row)}
            for key, row in sorted(rows_by_key.items())
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_assignment_json(path: str | Path) -> dict[str, str]:
    """Load a per-Linear assignment from the common candidate JSON shapes."""
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        for key in ("assignment", "chosen_assignment", "final_assignment"):
            value = raw.get(key)
            if isinstance(value, dict):
                return {str(k): str(v) for k, v in value.items()}
        selection = raw.get("selection")
        if isinstance(selection, dict):
            chosen = selection.get("chosen")
            if isinstance(chosen, dict) and isinstance(chosen.get("assignment"), dict):
                return {str(k): str(v) for k, v in chosen["assignment"].items()}
        if raw and all(isinstance(k, str) for k in raw):
            return {str(k): str(v) for k, v in raw.items()}
    raise ValueError(f"unsupported assignment JSON shape: {path}")


def assignment_for_units(
    assignment: dict[str, str] | None,
    units: Sequence[bc.DecisionUnit],
) -> dict[str, str]:
    """Normalize a partial per-Linear assignment over the current units."""

    def _canonical_assignment_fmt(value: str) -> str:
        raw = str(value).strip()
        canonical = fr.canonical_format_name(raw)
        if canonical in fr.REGISTRY:
            return canonical
        upper = fr.canonical_format_name(raw.upper())
        if upper in fr.REGISTRY:
            return upper
        for known in (*fr.REGISTRY.keys(), *fr.FORMAT_ALIASES.keys()):
            if known.casefold() == raw.casefold():
                return fr.canonical_format_name(known)
        return canonical

    source = assignment or {}
    out: dict[str, str] = {}
    for unit in units:
        chosen = None
        for member in unit.member_qnames:
            if member in source:
                chosen = _canonical_assignment_fmt(str(source[member]))
                break
        if chosen is None:
            chosen = "BF16"
        legal = {
            fr.canonical_format_name(str(opt.fmt))
            for opt in unit.options
        }
        if chosen not in legal:
            chosen = "BF16" if "BF16" in legal else next(iter(sorted(legal)))
        for member in unit.member_qnames:
            out[member] = chosen
    return out


def bf16_assignment_for_units(units: Sequence[bc.DecisionUnit]) -> dict[str, str]:
    return {
        member: "BF16"
        for unit in units
        for member in unit.member_qnames
    }


@contextmanager
def _temporary_env(name: str, value: str):
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def _production_cache_key_variants(qname: str, fmt: str):
    yield (qname, fmt)
    if qname.endswith(".weight"):
        yield (qname[:-len(".weight")], fmt)
    if qname.startswith("model.language_model."):
        yield ("model." + qname[len("model.language_model."):], fmt)
    elif qname.startswith("model."):
        yield ("model.language_model." + qname[len("model."):], fmt)


def _prefetch_assignment_delta(
    production_weight_cache,
    current: Mapping[str, str],
    target: Mapping[str, str],
) -> tuple[int, int]:
    """Prefetch cache-backed tensors needed to move current -> target.

    The production cache is already LRU-bounded; this only gives the cache a
    small look-ahead before WeightSession applies assignment deltas.  It avoids
    the previous startup-only prefetch policy, which left validation to
    synchronously torch.load each candidate's changed weights.
    """
    if production_weight_cache is None or not hasattr(production_weight_cache, "prefetch"):
        return 0, 0
    cache_weights = getattr(production_weight_cache, "weights", {}) or {}
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for qname, raw_fmt in target.items():
        fmt = fr.canonical_format_name(str(raw_fmt))
        prev = fr.canonical_format_name(str(current.get(qname, "BF16")))
        if fmt == prev or fmt == "BF16":
            continue
        chosen = None
        for key in _production_cache_key_variants(str(qname), fmt):
            if key in cache_weights:
                chosen = key
                break
        if chosen is None:
            continue
        if chosen not in seen:
            seen.add(chosen)
            keys.append(chosen)
    if not keys:
        return 0, 0
    return len(keys), int(production_weight_cache.prefetch(keys))


def run_iteration(
    *,
    model,
    calib_ids: torch.Tensor,
    ref_log_probs,
    profile,
    formats,
    work_root: Path,
    iter_idx: int,
    center_assignment: dict[str, str] | None,
    center_label: str,
    output_root: Path,
    n_neighbors_validate: int = 4,
    polish_max_passes: int = 8,
    polish_noise_floor: float = 1e-5,
    polish_budget_creep: float = 0.05,
    polish_steepest_first: bool = False,
    skip_polish: bool = False,
    use_frozen_weight_cache: bool = False,
    measure_method: str = "four_term",
    production_weight_cache=None,
    calib_microbatch: int = 1,
    output_fisher_reduction_device: str = "auto",
    output_fisher_logit_scope: str = "full_sequence",
    include_activation_quant: bool = True,
    validation_delta_quantize: bool = True,
    weight_session_snapshot_dir: str | Path | None = None,
    log_callback=None,
) -> IterationResult:
    """One iteration: measure → sweep → kneedle → validate → polish."""
    iter_dir = output_root / f"iter_{iter_idx}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    log = log_callback or (lambda **kw: None)
    start = time.time()

    log(event="iter_start", iter=iter_idx, centered_at=center_label)

    # ---- measure
    # When method=='output_fisher' and we're at the BF16 center (iter 0
    # or any time center_assignment is unset/all-BF16), use the analytic
    # Fisher form which is much faster.  At non-trivial centers fall back
    # to four-term since OF doesn't support sandwich centering yet.
    method_used = measure_method
    if measure_method == "output_fisher":
        # OF now supports sandwich centering as well — pass through.
        log(event="measure_start", iter=iter_idx, method="output_fisher",
            centered=(center_assignment is not None))

        def _of_progress(event: dict) -> None:
            kind = str(event.get("event", ""))
            if kind == "perturbation_done":
                completed = int(event.get("completed", 0))
                total = int(event.get("total", 0))
                if completed % 10 != 0 and completed != total:
                    return
            elif kind == "block_pairs_done":
                block_id = str(event.get("block_id", ""))
                if "layers." in block_id:
                    try:
                        if int(block_id.rsplit(".", 1)[-1]) % 8 != 0:
                            return
                    except Exception:
                        pass
            log(
                event=f"output_fisher_{kind}",
                iter=iter_idx,
                **{k: v for k, v in event.items() if k != "event"},
            )

        payload = collect_output_fisher(
            model, calib_ids, formats,
            profile=profile,
            cache_dir=str(iter_dir / "of_cache"),
            keep_disk_cache=False,
            skip_pairs=False,
            center_assignment=center_assignment,
            use_frozen_weight_cache=use_frozen_weight_cache,
            include_activation_quant=include_activation_quant,
            production_weight_cache=production_weight_cache,
            calib_microbatch=calib_microbatch,
            reduction_device=output_fisher_reduction_device,
            logit_scope=output_fisher_logit_scope,
            progress_callback=_of_progress,
        )
        payload_path = iter_dir / "block_clado.json"
        payload_path.write_text(json.dumps(payload, indent=2) + "\n")
        log(event="measure_done", iter=iter_idx,
            method="output_fisher",
            elapsed=payload["meta"]["elapsed_seconds"],
            center_kl=payload["meta"].get("center_kl", 0.0))
    else:
        log(event="measure_start", iter=iter_idx, method="four_term")
        payload = collect_block_clado(
            model, calib_ids, formats,
            profile=profile, work_root=work_root,
            skip_pairs=False,
            center_assignment=center_assignment,
            use_frozen_weight_cache=use_frozen_weight_cache,
            production_weight_cache=production_weight_cache,
        )
        payload_path = iter_dir / "block_clado.json"
        payload_path.write_text(json.dumps(payload, indent=2) + "\n")
        log(event="measure_done", iter=iter_idx,
            method="four_term",
            elapsed=payload["meta"]["elapsed_seconds"],
            center_kl=payload["meta"].get("center_kl", 0.0))
    log(event="measure_done", iter=iter_idx,
        elapsed=payload["meta"]["elapsed_seconds"],
        center_kl=payload["meta"].get("center_kl", 0.0))

    # ---- sweep
    block_states = bc.build_block_states(payload)
    total_params = bc.total_param_count(payload)
    blocks_back, singletons_back, pairs_back = bc.parse_payload(payload)
    units = []
    for unit_list in blocks_back.values():
        units.extend(unit_list)
    units.extend(singletons_back)
    sweep_results = bc.lambda_sweep(
        block_states, lambda_min=1e-12, lambda_max=1e-3, n_lambdas=61,
    )
    sweep_rows = [
        {
            "lambda": r.lambda_used,
            "bits_total": r.bits_total,
            "bpp": r.bits_total / float(total_params) if total_params else 0.0,
            "cost_total": r.cost_total,
            "assignment": r.assignment,
        }
        for r in sweep_results
    ]
    sweep_path = iter_dir / "lambda_sweep.json"
    sweep_path.write_text(json.dumps({
        "schema": "prismaquant.block_clado.sweep.v1",
        "rows": sweep_rows,
        "total_params": int(total_params),
    }, indent=2) + "\n")
    log(event="sweep_done", iter=iter_idx, points=len(sweep_rows))

    # ---- kneedle + neighbours expansion (validate cone around the elbow)
    # Filter to physically meaningful frontier: predicted_kl = center_kl
    # + cost_total > 0.  For BF16-centered (center_kl=0) this collapses
    # to cost_total > 0 (the original behavior); for sandwich-centered
    # (center_kl>0) it correctly admits negative-cost rows representing
    # predicted improvements over the centered state.
    center_kl = bc.center_kl_from_payload(payload)
    feasible_rows = [
        r for r in sweep_rows
        if (center_kl + r["cost_total"]) > 1e-9
    ]
    if len(feasible_rows) < 3:
        feasible_rows = sweep_rows
    points = [(float(r["bpp"]), float(r["cost_total"])) for r in feasible_rows]
    knee_idx, knee_score, knee_endpoint = bc.kneedle_pick(points)
    sorted_rows = sorted(feasible_rows, key=lambda r: r["bpp"])
    knee_bpp_target = feasible_rows[knee_idx]["bpp"]
    knee_in_sorted = min(
        range(len(sorted_rows)),
        key=lambda i: abs(sorted_rows[i]["bpp"] - knee_bpp_target),
    )
    indices = list(range(
        max(knee_in_sorted - n_neighbors_validate, 0),
        min(knee_in_sorted + n_neighbors_validate + 1, len(sorted_rows)),
    ))

    # ---- validate cone with real KL
    validation: list[dict] = []
    validation_checkpoint_path = iter_dir / "validation_checkpoint.json"
    validation_checkpoint_signature = _validation_checkpoint_signature(
        payload=payload,
        sorted_rows=sorted_rows,
        indices=indices,
        center_assignment=(
            assignment_for_units(center_assignment, units)
            if center_assignment is not None else None
        ),
        center_kl=center_kl,
        include_activation_quant=include_activation_quant,
        validation_delta_quantize=validation_delta_quantize,
        production_weight_cache=production_weight_cache,
    )
    validation_rows_by_key = _load_validation_checkpoint(
        validation_checkpoint_path,
        validation_checkpoint_signature,
    )
    if validation_rows_by_key:
        log(
            event="validation_checkpoint_loaded",
            iter=iter_idx,
            path=str(validation_checkpoint_path),
            rows=len(validation_rows_by_key),
        )

    def _save_validation_checkpoint() -> None:
        _write_validation_checkpoint(
            validation_checkpoint_path,
            signature=validation_checkpoint_signature,
            rows_by_key=validation_rows_by_key,
        )

    validation_weight_session = None
    validate_env_cm = (
        _temporary_env("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
        if validation_delta_quantize and production_weight_cache is not None
        else nullcontext()
    )
    with validate_env_cm:
        if validation_delta_quantize and production_weight_cache is not None:
            from prismaquant.weight_session import WeightSession

            validation_weight_session = WeightSession(
                model,
                production_weight_cache=production_weight_cache,
                snapshot_dir=(
                    str(weight_session_snapshot_dir)
                    if weight_session_snapshot_dir is not None else None
                ),
            )
            base_for_validation = (
                assignment_for_units(center_assignment, units)
                if center_assignment is not None
                else bf16_assignment_for_units(units)
            )
            n_prefetch, n_loaded = _prefetch_assignment_delta(
                production_weight_cache,
                bf16_assignment_for_units(units),
                base_for_validation,
            )
            validation_weight_session.initialize(base_for_validation, units)
            log(
                event="validation_weight_session_initialized",
                iter=iter_idx,
                n_prefetch_keys=n_prefetch,
                n_prefetch_loaded=n_loaded,
                diagnostics=validation_weight_session.diagnostics(),
            )

        try:
            for i in indices:
                r = sorted_rows[i]
                assignment = bc.expand_sweep_row_to_linear_assignment(
                    payload, r["assignment"],
                )
                selection_key = _validation_candidate_key(r)
                cached_row = validation_rows_by_key.get(selection_key)
                if cached_row is not None:
                    cached_validation_row = dict(cached_row)
                    cached_validation_row["is_kneedle"] = (i == knee_in_sorted)
                    validation.append(cached_validation_row)
                    log(event="validation_checkpoint_row_skipped",
                        iter=iter_idx,
                        bpp=cached_validation_row.get("bpp"),
                        real_kl=cached_validation_row.get("real_kl"),
                        is_kneedle=cached_validation_row.get("is_kneedle", False))
                    continue
                measure_frozen_cache = use_frozen_weight_cache
                if validation_weight_session is not None:
                    current_assignment = validation_weight_session.current_assignment()
                    n_prefetch, n_loaded = _prefetch_assignment_delta(
                        production_weight_cache,
                        current_assignment,
                        assignment,
                    )
                    n_changed = validation_weight_session.apply_assignment(assignment)
                    measure_frozen_cache = False
                    log(
                        event="validation_delta_assignment_applied",
                        iter=iter_idx,
                        bpp=r["bpp"],
                        n_changed=n_changed,
                        n_prefetch_keys=n_prefetch,
                        n_prefetch_loaded=n_loaded,
                    )
                kl = measure_assignment_kl(
                    model, assignment, calib_ids, ref_log_probs,
                    work_root=work_root, profile=profile,
                    use_frozen_weight_cache=measure_frozen_cache,
                    production_weight_cache=production_weight_cache, rng_seed=0,
                    include_activation_quant=include_activation_quant,
                )
                validation_row = {
                    "bpp": r["bpp"],
                    "surrogate_cost": r["cost_total"],
                    "real_kl": float(kl),
                    "is_kneedle": (i == knee_in_sorted),
                    "assignment": assignment,
                }
                validation.append(validation_row)
                validation_rows_by_key[selection_key] = dict(validation_row)
                _save_validation_checkpoint()
                log(event="validate_done", iter=iter_idx,
                    bpp=r["bpp"], real_kl=float(kl),
                    is_kneedle=(i == knee_in_sorted))
        finally:
            if validation_weight_session is not None:
                try:
                    n_restored = validation_weight_session.apply_assignment(
                        bf16_assignment_for_units(units),
                    )
                    log(
                        event="validation_weight_session_restored",
                        iter=iter_idx,
                        n_changed=n_restored,
                        diagnostics=validation_weight_session.diagnostics(),
                    )
                finally:
                    validation_weight_session = None

    if center_assignment is not None and center_kl > 0.0:
        center_complete = assignment_for_units(center_assignment, units)
        center_key = _validation_center_key(center_complete, center_kl)
        cached_center = validation_rows_by_key.get(center_key)
        if cached_center is not None:
            validation.append(dict(cached_center))
            log(event="validation_checkpoint_row_skipped", iter=iter_idx,
                bpp=cached_center.get("bpp"),
                real_kl=cached_center.get("real_kl"),
                is_center_baseline=True)
        else:
            center_bits = cdp._assignment_bits(units, center_complete)
            center_bpp = center_bits / float(total_params) if total_params else 0.0
            center_row = {
                "bpp": center_bpp,
                "surrogate_cost": 0.0,
                "real_kl": float(center_kl),
                "is_kneedle": False,
                "is_center_baseline": True,
                "assignment": center_complete,
            }
            validation.append(center_row)
            validation_rows_by_key[center_key] = dict(center_row)
            _save_validation_checkpoint()
            log(event="validate_center_baseline", iter=iter_idx,
                bpp=center_bpp, real_kl=float(center_kl))

    (iter_dir / "validation.json").write_text(json.dumps({
        "schema": "prismaquant.block_clado.iter.validation.v1",
        "kneedle_index": int(knee_in_sorted),
        "kneedle_score": float(knee_score),
        "endpoint_fallback": bool(knee_endpoint),
        "rows": validation,
    }, indent=2) + "\n")

    best_validated = min(validation, key=lambda v: v["real_kl"])
    log(event="best_validated", iter=iter_idx,
        bpp=best_validated["bpp"], real_kl=best_validated["real_kl"])

    # ---- polish the best validated assignment.  Allow modest budget
    # creep (default 5%) so polish can take Pareto-beneficial precision
    # upgrades on a small number of high-impact layers, but not all the
    # way to BF16-everywhere.
    log(event="polish_start", iter=iter_idx)
    if skip_polish:
        log(event="polish_skipped", iter=iter_idx)
        polish_result = cdp.PolishResult(
            initial_kl=float(best_validated["real_kl"]),
            final_kl=float(best_validated["real_kl"]),
            final_assignment=dict(best_validated["assignment"]),
        )
    else:
        starting_bits = cdp._assignment_bits(units, best_validated["assignment"])
        polish_budget = starting_bits * (1.0 + polish_budget_creep)
        def _polish_progress(event):
            kind = event.get("event")
            if kind in {"accept_move", "pass_no_improvement", "budget_set", "starting"}:
                log(event=f"polish_{kind}", iter=iter_idx, **{
                    k: v for k, v in event.items() if k != "event"
                })

        polish_result = cdp.coord_descent_polish(
            model, calib_ids, ref_log_probs,
            units=units,
            starting_assignment=best_validated["assignment"],
            profile=profile,
            work_root=work_root,
            noise_floor=polish_noise_floor,
            max_passes=polish_max_passes,
            bits_budget=polish_budget,
            pairs_by_block=dict(pairs_back),
            steepest_first=polish_steepest_first,
            use_frozen_weight_cache=use_frozen_weight_cache,
            production_weight_cache=production_weight_cache,
            include_activation_quant=include_activation_quant,
            delta_quantize=production_weight_cache is not None,
            weight_session_spill_to_disk=weight_session_snapshot_dir is not None,
            weight_session_snapshot_dir=weight_session_snapshot_dir,
            restore_bf16_on_exit=True,
            progress_callback=_polish_progress,
        )
    log(event="polish_done", iter=iter_idx,
        initial_kl=polish_result.initial_kl,
        final_kl=polish_result.final_kl,
        n_steps=len(polish_result.steps),
        n_meas=polish_result.n_kl_measurements)

    polish_path = iter_dir / "polish.json"
    polish_path.write_text(json.dumps({
        "schema": "prismaquant.coord_descent_polish.v1",
        "initial_kl": polish_result.initial_kl,
        "final_kl": polish_result.final_kl,
        "improvement": polish_result.initial_kl - polish_result.final_kl,
        "n_steps_accepted": len(polish_result.steps),
        "n_kl_measurements": polish_result.n_kl_measurements,
        "elapsed_seconds": polish_result.elapsed_seconds,
        "steps": [
            {
                "pass_index": s.pass_index,
                "unit": s.unit,
                "from_fmt": s.from_fmt,
                "to_fmt": s.to_fmt,
                "kl_before": s.kl_before,
                "kl_after": s.kl_after,
            }
            for s in polish_result.steps
        ],
        "final_assignment": polish_result.final_assignment,
    }, indent=2) + "\n")

    return IterationResult(
        iteration=iter_idx,
        centered_at=center_label,
        payload_path=payload_path,
        sweep_path=sweep_path,
        kneedle_label=f"frontier_bpp_{feasible_rows[knee_idx]['bpp']:.4f}",
        kneedle_bpp=float(feasible_rows[knee_idx]["bpp"]),
        kneedle_surrogate_cost=float(feasible_rows[knee_idx]["cost_total"]),
        best_validated_kl=float(best_validated["real_kl"]),
        best_validated_bpp=float(best_validated["bpp"]),
        best_validated_assignment=dict(best_validated["assignment"]),
        polished_kl=float(polish_result.final_kl),
        polished_assignment=dict(polish_result.final_assignment),
        polish_steps=len(polish_result.steps),
        elapsed_seconds=float(time.time() - start),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Iterated Block-CLADO refinement")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--formats", default="NVFP4,MXFP8_E4M3,BF16")
    parser.add_argument("--n-calib-samples", type=int, default=2)
    parser.add_argument("--calib-seqlen", type=int, default=128)
    parser.add_argument("--calib-microbatch", type=int, default=1)
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help=(
            "Transformers attention backend for model load. Accepts built-ins "
            "such as sdpa/flash_attention_2 and Kernel Hub backends such as "
            "kernels-community/flash-attn2."
        ),
    )
    parser.add_argument("--n-neighbors-validate", type=int, default=4)
    parser.add_argument("--polish-max-passes", type=int, default=8)
    parser.add_argument("--polish-noise-floor", type=float, default=1e-5)
    parser.add_argument(
        "--polish-budget-creep",
        type=float,
        default=0.05,
        help=(
            "Polish bits-budget tolerance as a fraction of the starting "
            "bits.  0.0 = strict (no precision creep); default 0.05 lets "
            "polish make ~5%% Pareto-beneficial precision upgrades."
        ),
    )
    parser.add_argument(
        "--polish-steepest-first",
        action="store_true",
        help=(
            "Order polish candidates by surrogate ΔΩ; accept the first "
            "real-KL improvement.  Faster than greedy-best when the "
            "surrogate ranks moves accurately around the current point."
        ),
    )
    parser.add_argument(
        "--use-frozen-weight-cache",
        action="store_true",
        help=(
            "Pre-quantize centered base assignment once and reuse cached "
            "weights across measurements.  Big speedup on sandwich runs "
            "for small/medium models; OOM-prone at LLM scale."
        ),
    )
    parser.add_argument(
        "--no-activation-quant",
        action="store_true",
        help="Skip activation quantization in surrogate validation and polish KL.",
    )
    parser.add_argument(
        "--skip-polish",
        action="store_true",
        help=(
            "Skip the coord-descent polish stage at every iteration.  "
            "Useful for fast surrogate-only sweeps where polish is the "
            "dominant cost; the iterate output then equals best-validated."
        ),
    )
    parser.add_argument(
        "--measure-method",
        choices=["four_term", "output_fisher"],
        default="four_term",
        help=(
            "Surrogate measurement method. 'output_fisher' uses the analytic "
            "per-token Fisher and supports sandwich centering at a supplied "
            "or previous polished assignment."
        ),
    )
    parser.add_argument(
        "--output-fisher-reduction-device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help=(
            "Device for output-Fisher full-vocab reductions when "
            "--measure-method=output_fisher. 'cuda' improves utilization on "
            "small/medium calibration stacks; 'auto' uses CUDA only when the "
            "estimated full-vocab tensor stack fits a conservative memory budget."
        ),
    )
    parser.add_argument(
        "--output-fisher-logit-scope",
        choices=["full_sequence", "last_token"],
        default="full_sequence",
        help=(
            "Output-Fisher KL scope. Use last_token for long calibration "
            "windows until full-sequence teacher/center streaming is wired."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--production-weight-cache",
        default=None,
        help=(
            "Path to a pickled ProductionWeightCache (e.g. from "
            "prismaquant.build_production_cache).  When supplied, four-term "
            "/ output-Fisher measurements, cone validation, and polish all "
            "use production-faithful δw (GPTQ + scale_sweep on NVFP4) "
            "instead of bare RTN."
        ),
    )
    parser.add_argument(
        "--production-cache-dir-override",
        default=None,
        help="Relocate a loaded disk-streamed production cache to this shard directory.",
    )
    parser.add_argument(
        "--production-cache-lru-gb",
        type=float,
        default=16.0,
        help="LRU budget for lazily loaded production-cache shards. Use 0 to disable.",
    )
    parser.add_argument(
        "--production-cache-prefetch",
        choices=["none", "initial_center", "all"],
        default="initial_center",
        help=(
            "Production-cache prefetch policy. 'initial_center' prefetches "
            "cached low-bit weights used by --initial-center-assignment; "
            "'all' materializes every shard within the LRU budget; 'none' "
            "only lazy-loads on demand."
        ),
    )
    parser.add_argument(
        "--validation-delta-quantize",
        dest="validation_delta_quantize",
        action="store_true",
        default=True,
        help=(
            "Validate frontier candidates by materializing assignment deltas "
            "with WeightSession instead of rebuilding a whole frozen-weight "
            "cache per candidate. Enabled by default when a production cache "
            "is supplied."
        ),
    )
    parser.add_argument(
        "--no-validation-delta-quantize",
        dest="validation_delta_quantize",
        action="store_false",
        help="Disable WeightSession-backed validation and use the legacy hook path.",
    )
    parser.add_argument(
        "--weight-session-snapshot-dir",
        default=os.environ.get("PRISMAQUANT_WEIGHT_SESSION_SNAPSHOT_DIR"),
        help=(
            "Optional shared directory for BF16 WeightSession snapshots used "
            "by output-Fisher, validation delta materialization, and polish."
        ),
    )
    parser.add_argument(
        "--initial-center-assignment",
        default=None,
        help=(
            "Optional per-Linear assignment JSON to use as the first "
            "sandwich center. This lets Block-CLADO refine around an "
            "externally measured allocator/frontier candidate instead of "
            "spending iteration 0 centered at BF16."
        ),
    )
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = _dtype_from_name(args.dtype)
    staged, cleanup = stage_multimodal(args.model)
    work_root = Path(tempfile.mkdtemp(prefix="prismaquant_iter_bc_"))
    summary_rows: list[IterationResult] = []
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
            "attn_implementation": args.attn_implementation,
        }
        if device == "cuda":
            load_kwargs["device_map"] = "cuda"
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        if device != "cuda":
            model.to(device)
        model.eval()
        try:
            profile = detect_profile(args.model)
        except Exception:
            profile = DefaultProfile()
        model_device = next(model.parameters()).device
        ref_log_probs = cache_reference_log_probs(model, calib_ids, model_device)
        formats = [fr.get_format(name.strip()) for name in args.formats.split(",") if name.strip()]

        production_weight_cache = None
        if args.production_weight_cache:
            import pickle
            with open(args.production_weight_cache, "rb") as fh:
                production_weight_cache = pickle.load(fh)
            if args.production_cache_dir_override:
                production_weight_cache.relocate(args.production_cache_dir_override)
            if getattr(production_weight_cache, "cache_dir", None) is not None:
                verify = production_weight_cache.verify_files()
                if verify.get("missing"):
                    raise RuntimeError(
                        "production cache has missing shard files after relocation; "
                        f"sample={verify['missing'][:5]}"
                    )
            lru_gb = float(args.production_cache_lru_gb)
            if lru_gb > 0.0:
                production_weight_cache.enable_lru(int(lru_gb * (1024 ** 3)))
            print(
                f"[iter] loaded production cache with "
                f"{len(production_weight_cache)} entries",
                flush=True,
            )

        def log(**kw):
            payload = dict(kw)
            print(f"[iter] {json.dumps(payload, default=str)}", flush=True)

        center_assignment: dict[str, str] | None = None
        center_label = "BF16"
        if args.initial_center_assignment:
            center_assignment = load_assignment_json(args.initial_center_assignment)
            center_label = f"initial:{Path(args.initial_center_assignment).stem}"
            print(
                f"[iter] initial center assignment {args.initial_center_assignment} "
                f"({len(center_assignment)} entries)",
                flush=True,
            )
        if production_weight_cache is not None:
            prefetch_mode = str(args.production_cache_prefetch)
            if prefetch_mode == "all":
                n_loaded = production_weight_cache.prefetch()
                print(f"[iter] production cache prefetched all: {n_loaded}", flush=True)
            elif prefetch_mode == "initial_center" and center_assignment:
                n_keys, n_loaded = _prefetch_assignment_delta(
                    production_weight_cache,
                    {},
                    center_assignment,
                )
                print(
                    "[iter] production cache prefetched initial_center: "
                    f"{n_loaded} loaded / {n_keys} keys",
                    flush=True,
                )
        prev_polish_hash: str | None = None
        best_overall: IterationResult | None = None
        for iter_idx in range(int(args.max_iterations)):
            result = run_iteration(
                model=model,
                calib_ids=calib_ids,
                ref_log_probs=ref_log_probs,
                profile=profile,
                formats=formats,
                work_root=work_root,
                iter_idx=iter_idx,
                center_assignment=center_assignment,
                center_label=center_label,
                output_root=output_root,
                n_neighbors_validate=args.n_neighbors_validate,
                polish_max_passes=args.polish_max_passes,
                polish_noise_floor=args.polish_noise_floor,
                polish_budget_creep=args.polish_budget_creep,
                polish_steepest_first=bool(args.polish_steepest_first),
                skip_polish=bool(args.skip_polish),
                use_frozen_weight_cache=bool(args.use_frozen_weight_cache),
                measure_method=str(args.measure_method),
                production_weight_cache=production_weight_cache,
                calib_microbatch=int(args.calib_microbatch),
                output_fisher_reduction_device=str(args.output_fisher_reduction_device),
                output_fisher_logit_scope=str(args.output_fisher_logit_scope),
                include_activation_quant=not bool(args.no_activation_quant),
                validation_delta_quantize=bool(args.validation_delta_quantize),
                weight_session_snapshot_dir=args.weight_session_snapshot_dir,
                log_callback=log,
            )
            summary_rows.append(result)
            if best_overall is None or result.polished_kl < best_overall.polished_kl - 1e-9:
                best_overall = result
                log(event="best_overall_updated",
                    iter=iter_idx,
                    polished_kl=result.polished_kl,
                    bpp=result.best_validated_bpp)
            log(event="iter_summary",
                iter=iter_idx,
                kneedle_bpp=result.kneedle_bpp,
                best_validated_kl=result.best_validated_kl,
                polished_kl=result.polished_kl,
                polish_steps=result.polish_steps,
                elapsed_seconds=result.elapsed_seconds)
            polish_hash = assignment_hash(result.polished_assignment)
            if prev_polish_hash is not None and polish_hash == prev_polish_hash:
                log(event="converged", iter=iter_idx)
                break
            prev_polish_hash = polish_hash
            center_assignment = result.polished_assignment
            center_label = f"iter_{iter_idx}_polish"

        # Summary
        summary = {
            "schema": "prismaquant.block_clado.iter.summary.v1",
            "iterations": [
                {
                    "iteration": r.iteration,
                    "centered_at": r.centered_at,
                    "kneedle_bpp": r.kneedle_bpp,
                    "kneedle_surrogate_cost": r.kneedle_surrogate_cost,
                    "best_validated_kl": r.best_validated_kl,
                    "best_validated_bpp": r.best_validated_bpp,
                    "polished_kl": r.polished_kl,
                    "polish_steps": r.polish_steps,
                    "elapsed_seconds": r.elapsed_seconds,
                }
                for r in summary_rows
            ],
            "best_overall": (
                {
                    "iteration": best_overall.iteration,
                    "polished_kl": best_overall.polished_kl,
                    "best_validated_bpp": best_overall.best_validated_bpp,
                    "polish_steps": best_overall.polish_steps,
                }
                if best_overall is not None else None
            ),
        }
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if best_overall is not None:
            (output_root / "best_assignment.json").write_text(json.dumps({
                "schema": "prismaquant.block_clado.best.v1",
                "iteration": best_overall.iteration,
                "polished_kl": best_overall.polished_kl,
                "best_validated_bpp": best_overall.best_validated_bpp,
                "assignment": best_overall.polished_assignment,
            }, indent=2) + "\n")
        print(f"[iter] wrote {output_root / 'summary.json'}", flush=True)
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
