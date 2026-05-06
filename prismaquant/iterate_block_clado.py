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
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
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
        payload = collect_output_fisher(
            model, calib_ids, formats,
            profile=profile,
            cache_dir=str(iter_dir / "of_cache"),
            keep_disk_cache=False,
            skip_pairs=False,
            center_assignment=center_assignment,
            use_frozen_weight_cache=use_frozen_weight_cache,
            include_activation_quant=True,
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
    for i in indices:
        r = sorted_rows[i]
        assignment = bc.expand_sweep_row_to_linear_assignment(payload, r["assignment"])
        kl = measure_assignment_kl(
            model, assignment, calib_ids, ref_log_probs,
            work_root=work_root, profile=profile,
            use_frozen_weight_cache=use_frozen_weight_cache, rng_seed=0,
        )
        validation.append({
            "bpp": r["bpp"],
            "surrogate_cost": r["cost_total"],
            "real_kl": float(kl),
            "is_kneedle": (i == knee_in_sorted),
            "assignment": assignment,
        })
        log(event="validate_done", iter=iter_idx,
            bpp=r["bpp"], real_kl=float(kl),
            is_kneedle=(i == knee_in_sorted))

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
    units = []
    blocks_back, singletons_back, pairs_back = bc.parse_payload(payload)
    for unit_list in blocks_back.values():
        units.extend(unit_list)
    units.extend(singletons_back)
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
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
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
            "Surrogate measurement method.  'output_fisher' uses the "
            "analytic per-token Fisher and is much cheaper on the BF16-"
            "centered iter 0; sandwich iter 1+ falls back to four_term "
            "automatically (Fisher form doesn't support non-zero centers)."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
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

        def log(**kw):
            payload = dict(kw)
            print(f"[iter] {json.dumps(payload, default=str)}", flush=True)

        center_assignment: dict[str, str] | None = None
        center_label = "BF16"
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
