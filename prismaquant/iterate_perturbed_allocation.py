#!/usr/bin/env python3
"""Iterate allocation against perturbed activation caches.

Each iteration builds an activation cache under the previous allocation,
measures a fresh W*A* cost table from that cache, smooths costs with a
per-(layer, format) EMA, and re-solves the allocator. The loop stops when the
weighted assignment change is small or when max_iters is reached.
"""
from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from collections.abc import Callable, Mapping
from numbers import Real
from pathlib import Path

import torch

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    aggregate_fused_siblings,
    build_candidates,
    expand_fused_sibling_assignment,
)
from prismaquant.allocator_solver import solve_allocation
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    kl_divergence,
    load_wikitext_calibration,
)
from prismaquant.measure_quant_cost import ActivationIndex, run_cost_pass
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    calibration_data_hash,
    capture_perturbed_activation_cache,
    load_text_model_under_work_root,
)
from prismaquant.propagated_cost import (
    build_l3_candidates,
    measure_propagated_costs,
    select_l3_neighborhood,
    solve_frozen_l3_neighborhood,
)
from prismaquant.schemas import validate_cost_payload, validate_probe_payload


def assignment_hash(assignment: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(name), str(fmt)) for name, fmt in assignment.items()))


def _is_numeric_metric(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def smooth_cost_history(
    history: list[dict],
    *,
    decay: float = 0.5,
) -> dict:
    """Geometric EMA over numeric cost metrics, newest weighted highest."""
    if not history:
        return {}
    out: dict[str, dict] = {}
    names = sorted({name for table in history for name in table})
    for name in names:
        fmt_names = sorted({
            fmt
            for table in history
            for fmt in table.get(name, {})
        })
        out[name] = {}
        for fmt in fmt_names:
            weighted: dict[str, float] = {}
            weights: dict[str, float] = {}
            bool_fields: dict[str, list[bool]] = {}
            newest_meta: dict = {}
            for age, table in enumerate(reversed(history)):
                entry = table.get(name, {}).get(fmt)
                if not isinstance(entry, dict) or "error" in entry:
                    continue
                if not newest_meta:
                    newest_meta = {
                        k: v for k, v in entry.items()
                        if not _is_numeric_metric(v) and k != "error"
                    }
                w = float(decay) ** age
                for key, value in entry.items():
                    if key == "error":
                        continue
                    if isinstance(value, bool):
                        bool_fields.setdefault(key, []).append(value)
                    elif _is_numeric_metric(value):
                        weighted[key] = weighted.get(key, 0.0) + float(value) * w
                        weights[key] = weights.get(key, 0.0) + w
            if not weights and not newest_meta and not bool_fields:
                out[name][fmt] = {"error": "no usable cost entries"}
                continue
            smoothed = dict(newest_meta)
            for key, total in weighted.items():
                smoothed[key] = total / max(weights[key], 1e-12)
            for key, values in bool_fields.items():
                if key == "output_mse_measured":
                    smoothed[key] = all(values)
                else:
                    smoothed[key] = values[0]
            out[name][fmt] = smoothed
    return out


def average_cost_tables(left: dict, right: dict) -> dict:
    return smooth_cost_history([left, right], decay=1.0)


def cost_value(name: str, fmt: str, costs: dict, stats: dict | None = None) -> float:
    entry = costs.get(name, {}).get(fmt, {})
    if not isinstance(entry, dict) or "error" in entry:
        return 0.0
    if stats is not None and name in stats:
        from prismaquant.allocator_candidates import cost_entry_predicted_dloss

        return float(cost_entry_predicted_dloss(stats[name], entry))
    if "predicted_dloss" in entry:
        return float(entry["predicted_dloss"])
    if "output_mse" in entry:
        return float(entry["output_mse"])
    return float(entry.get("weight_mse", 0.0))


def weighted_hamming_fraction(
    old: Mapping[str, str],
    new: Mapping[str, str],
    costs: dict,
    *,
    stats: dict | None = None,
    eps: float = 1e-12,
) -> float:
    numerator = 0.0
    for name in sorted(set(old) | set(new)):
        old_fmt = old.get(name)
        new_fmt = new.get(name)
        if old_fmt is None or new_fmt is None or old_fmt == new_fmt:
            continue
        numerator += abs(
            cost_value(name, old_fmt, costs, stats)
            - cost_value(name, new_fmt, costs, stats)
        )
    denom = 0.0
    for name, fmt in new.items():
        denom += cost_value(name, fmt, costs, stats)
    return numerator / max(denom, eps)


def resolve_two_cycle(
    prevprev: Mapping[str, str],
    prev: Mapping[str, str],
    current: Mapping[str, str],
    previous_costs: dict,
    current_costs: dict,
    solve_fn: Callable[[dict], dict[str, str]],
    kl_fn: Callable[[Mapping[str, str]], float],
) -> tuple[dict[str, str], str]:
    """Resolve A_k == A_{k-2} by cost averaging, then KL tie-break if stuck."""
    if assignment_hash(prevprev) != assignment_hash(current):
        return dict(current), "none"
    averaged = average_cost_tables(previous_costs, current_costs)
    resolved = dict(solve_fn(averaged))
    endpoints = {assignment_hash(prev), assignment_hash(current)}
    if assignment_hash(resolved) not in endpoints:
        return resolved, "averaged-costs"
    prev_kl = float(kl_fn(prev))
    current_kl = float(kl_fn(current))
    if prev_kl <= current_kl:
        return dict(prev), "kl-prev"
    return dict(current), "kl-current"


def _format_from_config_entry(entry) -> str:
    if isinstance(entry, str):
        return fr.canonical_format_name(entry)
    if isinstance(entry, int):
        if entry == 16:
            return "BF16"
        raise ValueError(f"cannot infer format from integer config {entry!r}")
    if not isinstance(entry, dict):
        raise ValueError(f"cannot infer format from config entry {entry!r}")
    for spec in fr.list_formats():
        cfg = spec.autoround_config()
        keys = {"bits", "group_size", "data_type", "act_bits", "act_data_type"}
        if all(cfg.get(k) == entry.get(k) for k in keys if k in cfg or k in entry):
            return spec.name
    bits = int(entry.get("bits", 0) or 0)
    data_type = str(entry.get("data_type", ""))
    act_bits = int(entry.get("act_bits", 16) or 16)
    group_size = int(entry.get("group_size", 0) or 0)
    if bits == 16 and data_type == "float":
        return "BF16"
    if bits == 4 and data_type == "fp4_e2m1" and act_bits < 16:
        return "NVFP4"
    if bits == 8 and data_type == "fp8_e4m3" and act_bits < 16 and group_size == 32:
        return "MXFP8"
    raise ValueError(f"cannot infer registered format from config entry {entry!r}")


def load_assignment_config(path: str | Path) -> dict[str, str]:
    with open(path) as f:
        payload = json.load(f)
    return {str(name): _format_from_config_entry(entry) for name, entry in payload.items()}


def default_initial_assignment(
    stats: dict,
    costs: dict,
    specs: list[fr.FormatSpec],
) -> dict[str, str]:
    available = {s.name for s in specs}
    base = "NVFP4" if "NVFP4" in available else min(specs, key=lambda s: s.effective_bits).name
    fallback = "BF16" if "BF16" in available else max(specs, key=lambda s: s.effective_bits).name
    out: dict[str, str] = {}
    for name in stats:
        entry = costs.get(name, {}).get(base)
        if isinstance(entry, dict) and "error" not in entry:
            out[name] = base
        else:
            out[name] = fallback
    return out


def solve_from_costs(
    stats: dict,
    costs: dict,
    specs: list[fr.FormatSpec],
    *,
    target_bits: float,
    bit_precision: float,
    profile=None,
) -> dict[str, str]:
    candidates = build_candidates(stats, costs, specs)
    stats_alloc, _costs_alloc, candidates = aggregate_fused_siblings(
        stats,
        costs,
        specs,
        candidates,
        profile=profile,
    )
    result = solve_allocation(stats_alloc, candidates, target_bits, bit_precision)
    if result is None:
        raise RuntimeError(f"infeasible target_bits={target_bits}")
    assignment, _chosen = result
    return expand_fused_sibling_assignment(assignment, stats_alloc)


def write_layer_config(assignment: Mapping[str, str], output_path: str | Path) -> None:
    payload = {
        name: fr.get_format(fmt).autoround_config()
        for name, fmt in sorted(assignment.items())
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


def build_l3_polish_summary(
    *,
    selected,
    l3_costs: dict,
    before_assignment: Mapping[str, str],
    after_assignment: Mapping[str, str],
    kl_before: float,
    kl_after: float,
    accepted: bool = True,
) -> dict:
    flips = []
    for name in sorted(set(before_assignment) | set(after_assignment)):
        old = before_assignment.get(name)
        new = after_assignment.get(name)
        if old == new:
            continue
        per_name = l3_costs.get(name, {})
        flips.append(
            {
                "name": name,
                "from": old,
                "to": new,
                "from_l3_cost": (
                    per_name.get(old, {}).get("propagated_end_kl")
                    if old is not None else None
                ),
                "to_l3_cost": (
                    per_name.get(new, {}).get("propagated_end_kl")
                    if new is not None else None
                ),
            }
        )
    regression = bool(float(kl_after) > float(kl_before))
    return {
        "enabled": True,
        "accepted": bool(accepted),
        "selected_count": len(selected),
        "measured_count": len(l3_costs),
        "kl_before": float(kl_before),
        "kl_after": float(kl_after),
        "regression": regression,
        "flip_count": len(flips),
        "flips": flips,
        "selected": [
            {
                "name": entry.name,
                "current_format": entry.current_format,
                "formats": list(entry.formats),
                "margin": entry.margin,
                "l2_current_cost": entry.l2_current_cost,
                "reasons": list(entry.reasons),
            }
            for entry in selected
        ],
    }


@torch.no_grad()
def measure_assignment_kl(
    model,
    assignment: Mapping[str, str],
    calib_ids: torch.Tensor,
    ref_log_probs,
    *,
    work_root: str | Path,
    profile=None,
) -> float:
    cache_dir = Path(tempfile.mkdtemp(prefix="prismaquant_kl_hooks_", dir=str(work_root)))
    cal_hash = calibration_data_hash(calib_ids)
    hooks = PerturbedActivationCache(
        model,
        assignment,
        cache_dir,
        input_rows=0,
        cal_hash=cal_hash,
        profile=profile,
    )
    device = next(model.parameters()).device
    values = []
    hooks.install()
    try:
        for i in range(calib_ids.size(0)):
            batch = calib_ids[i:i + 1].to(device)
            logits = model(batch).logits[:, -1:, :]
            teacher = ref_log_probs[i][:, -1:, :]
            values.append(float(kl_divergence(logits, teacher).item()))
    finally:
        hooks.remove()
    return sum(values) / max(len(values), 1)


def _dtype_from_name(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--initial-costs", required=True)
    ap.add_argument("--formats", default="NVFP4,MXFP8,BF16")
    ap.add_argument("--target-bits", type=float, required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--initial-config")
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--convergence-frac", type=float, default=0.01)
    ap.add_argument("--ema-decay", type=float, default=0.5)
    ap.add_argument("--input-rows", type=int, default=256)
    ap.add_argument("--n-calib-samples", type=int, default=8)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--bit-precision", type=float, default=0.001)
    ap.add_argument("--cost-mode", default="auto", choices=["auto", "unbatched", "batched"])
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--h-detail-dir")
    ap.add_argument("--l3-polish", action="store_true",
                    help="Run one propagated-cost polish pass after L2 convergence.")
    ap.add_argument("--l3-per-iter", action="store_true",
                    help="Reserved for future thorough runs; currently unsupported.")
    ap.add_argument("--l3-uncertainty-rel-tol", type=float, default=0.10)
    ap.add_argument("--l3-min-fraction", type=float, default=0.05)
    ap.add_argument("--l3-max-fraction", type=float, default=0.10)
    ap.add_argument("--l3-safety-fraction", type=float, default=0.02)
    ap.add_argument("--l3-max-lanes-per-batch", type=int, default=8)
    args = ap.parse_args(argv)

    if args.l3_per_iter:
        raise SystemExit("--l3-per-iter is reserved; use --l3-polish final pass")

    work_root = Path(args.work_dir)
    output_root = Path(args.output_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    with open(args.initial_costs, "rb") as f:
        cost_payload = pickle.load(f)
    validate_probe_payload(probe, args.probe)
    validate_cost_payload(cost_payload, args.initial_costs)
    stats = probe["stats"]
    current_costs = cost_payload["costs"]
    specs = [fr.get_format(s.strip()) for s in args.formats.split(",") if s.strip()]
    specs = sorted(specs, key=lambda s: s.effective_bits)

    from transformers import AutoTokenizer
    from prismaquant.model_profiles import detect_profile, DefaultProfile

    profile = DefaultProfile()
    try:
        profile = detect_profile(args.model)
    except Exception:
        pass

    dtype = _dtype_from_name(args.dtype)
    model = load_text_model_under_work_root(
        args.model,
        device=args.device,
        dtype=dtype,
        work_root=work_root,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    calib_ids = load_wikitext_calibration(
        tokenizer,
        args.n_calib_samples,
        args.calib_seqlen,
    )
    ref_log_probs = cache_reference_log_probs(model, calib_ids, next(model.parameters()).device)

    if args.initial_config:
        assignment = load_assignment_config(args.initial_config)
    else:
        assignment = default_initial_assignment(stats, current_costs, specs)

    cost_history: list[dict] = []
    latest_smoothed_costs = current_costs
    assignment_history: list[dict[str, str]] = [dict(assignment)]
    iteration_log: list[dict] = []

    for iteration in range(1, int(args.max_iters) + 1):
        cache_dir = output_root / f"activation_cache_iter_{iteration:02d}"
        capture_manifest = capture_perturbed_activation_cache(
            model,
            assignment,
            calib_ids,
            cache_dir,
            input_rows=args.input_rows,
            profile=profile,
        )
        act_cache = ActivationIndex(cache_dir, stats.keys())
        target_names = set(stats.keys())
        missing_act = [n for n in target_names if n not in act_cache]
        cost_path = output_root / f"costs_iter_{iteration:02d}.pkl"
        measured_costs = run_cost_pass(
            model,
            act_cache,
            target_names,
            missing_act,
            specs,
            args.model,
            args.probe,
            args.device,
            dtype,
            args.cost_mode,
            args.chunk_size,
            str(cost_path),
            h_detail_dir=args.h_detail_dir,
        )
        cost_history.append(measured_costs)
        smoothed_costs = smooth_cost_history(cost_history, decay=args.ema_decay)
        latest_smoothed_costs = smoothed_costs

        def _solve(cost_table: dict) -> dict[str, str]:
            return solve_from_costs(
                stats,
                cost_table,
                specs,
                target_bits=args.target_bits,
                bit_precision=args.bit_precision,
                profile=profile,
            )

        next_assignment = _solve(smoothed_costs)
        cycle_mode = "none"
        if len(assignment_history) >= 2:
            maybe, cycle_mode = resolve_two_cycle(
                assignment_history[-2],
                assignment_history[-1],
                next_assignment,
                cost_history[-2] if len(cost_history) >= 2 else measured_costs,
                measured_costs,
                _solve,
                lambda a: measure_assignment_kl(
                    model,
                    a,
                    calib_ids,
                    ref_log_probs,
                    work_root=work_root,
                    profile=profile,
                ),
            )
            next_assignment = maybe

        flip_frac = weighted_hamming_fraction(
            assignment,
            next_assignment,
            smoothed_costs,
            stats=stats,
        )
        iteration_log.append(
            {
                "iteration": iteration,
                "cache": capture_manifest,
                "cost_path": str(cost_path),
                "flip_fraction": flip_frac,
                "cycle_mode": cycle_mode,
                "assignment_hash": list(assignment_hash(next_assignment)),
            }
        )
        assignment = dict(next_assignment)
        assignment_history.append(dict(assignment))
        if flip_frac <= args.convergence_frac:
            break

    l3_summary_path = None
    if args.l3_polish:
        l2_assignment = dict(assignment)
        kl_before = measure_assignment_kl(
            model,
            l2_assignment,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
        )
        selected = select_l3_neighborhood(
            stats,
            latest_smoothed_costs,
            l2_assignment,
            specs,
            uncertainty_rel_tol=args.l3_uncertainty_rel_tol,
            min_fraction=args.l3_min_fraction,
            max_fraction=args.l3_max_fraction,
            safety_fraction=args.l3_safety_fraction,
        )
        l3_costs = measure_propagated_costs(
            model,
            l2_assignment,
            selected,
            calib_ids,
            specs,
            work_root=work_root,
            profile=profile,
            max_lanes_per_batch=args.l3_max_lanes_per_batch,
        )
        l3_cost_path = output_root / "l3_propagated_costs.pkl"
        with open(l3_cost_path, "wb") as f:
            pickle.dump(
                {
                    "costs": l3_costs,
                    "formats": [s.name for s in specs],
                    "meta": {
                        "model": args.model,
                        "probe": args.probe,
                        "paired_baseline": "target_bf16_under_l2_assignment",
                        "selected_count": len(selected),
                    },
                },
                f,
            )
        l3_candidates = build_l3_candidates(stats, l3_costs, specs)
        current_fmt_by_name = {
            name: l2_assignment[name]
            for name in l3_candidates
            if name in l2_assignment
        }
        l3_candidates = {
            name: cands
            for name, cands in l3_candidates.items()
            if any(c.fmt == current_fmt_by_name.get(name) for c in cands)
        }
        if l3_candidates:
            polished_assignment, _chosen = solve_frozen_l3_neighborhood(
                stats,
                l2_assignment,
                l3_candidates,
                specs,
                target_bits=args.target_bits,
                bit_precision=args.bit_precision,
            )
        else:
            polished_assignment = dict(l2_assignment)
        kl_after = measure_assignment_kl(
            model,
            polished_assignment,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
        )
        l3_summary = build_l3_polish_summary(
            selected=selected,
            l3_costs=l3_costs,
            before_assignment=l2_assignment,
            after_assignment=polished_assignment,
            kl_before=kl_before,
            kl_after=kl_after,
        )
        l3_summary["cost_path"] = str(l3_cost_path)
        l3_summary_path = output_root / "l3_polish_summary.json"
        with open(l3_summary_path, "w") as f:
            json.dump(l3_summary, f, indent=2)
        assignment = dict(polished_assignment)

    final_assignment_path = output_root / "final_assignment.json"
    with open(final_assignment_path, "w") as f:
        json.dump(dict(sorted(assignment.items())), f, indent=2)
    final_layer_config = output_root / "final_layer_config.json"
    write_layer_config(assignment, final_layer_config)
    summary = {
        "iterations": iteration_log,
        "final_assignment": str(final_assignment_path),
        "final_layer_config": str(final_layer_config),
        "target_bits": args.target_bits,
    }
    if l3_summary_path is not None:
        summary["l3_polish_summary"] = str(l3_summary_path)
    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
