#!/usr/bin/env python3
"""Iterate allocation against perturbed activation caches.

Each iteration builds an activation cache under the previous allocation,
measures a fresh W*A* cost table from that cache, smooths costs with a
per-(layer, format) EMA, and re-solves the allocator. The loop stops when the
weighted assignment change is small or when max_iters is reached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tempfile
import time
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
    L3UnsupportedTargetError,
    assignment_bit_total,
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


def _emit(message: str) -> None:
    print(message, flush=True)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _assignment_digest(assignment: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(assignment.items())), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _cuda_reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    peak = float(torch.cuda.max_memory_allocated()) / float(1024 ** 3)
    torch.cuda.reset_peak_memory_stats()
    return peak


def _phase_start(label: str) -> float:
    _cuda_reset_peak()
    _emit(f"{label}: start")
    return time.monotonic()


def _phase_end(label: str, start: float) -> dict:
    elapsed = time.monotonic() - start
    peak_gb = _cuda_peak_gb()
    suffix = f", cuda_peak={peak_gb:.2f}GB" if peak_gb is not None else ""
    _emit(f"{label}: done in {elapsed:.1f}s{suffix}")
    return {"elapsed_seconds": elapsed, "cuda_peak_gb": peak_gb}


def _directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def _format_histogram(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
    target_bits: float,
) -> dict:
    counts: dict[str, int] = {}
    for fmt in assignment.values():
        canonical = fr.canonical_format_name(fmt)
        counts[canonical] = counts.get(canonical, 0) + 1
    specs_by_name = {s.name: s for s in specs}
    total_params = sum(
        int(stats[name].get("n_params", 0) or 0)
        for name in assignment
        if name in stats
    )
    known_assignment = {
        name: fmt
        for name, fmt in assignment.items()
        if fr.canonical_format_name(fmt) in specs_by_name
    }
    total_bits = assignment_bit_total(stats, known_assignment, specs_by_name)
    achieved = total_bits / float(total_params) if total_params > 0 else 0.0
    return {
        "counts": dict(sorted(counts.items())),
        "total": len(assignment),
        "achieved_bpp": achieved,
        "target_bpp": float(target_bits),
    }


def _format_histogram_delta(previous: Mapping[str, int], current: Mapping[str, int]) -> str:
    parts = []
    for fmt in sorted(set(previous) | set(current)):
        delta = int(current.get(fmt, 0)) - int(previous.get(fmt, 0))
        if delta == 0:
            continue
        parts.append(f"{fmt} {delta:+d}")
    return ", ".join(parts) if parts else "no count changes"


def _ema_weights(history_len: int, decay: float) -> list[float]:
    return [float(decay) ** age for age in range(history_len)]


def _top_cost_changes(
    previous: dict | None,
    current: dict,
    stats: dict,
    *,
    top_k: int = 5,
) -> list[dict]:
    if previous is None:
        return []
    changes = []
    names = sorted(set(previous) | set(current))
    for name in names:
        fmts = sorted(set(previous.get(name, {})) | set(current.get(name, {})))
        for fmt in fmts:
            if fmt not in previous.get(name, {}) or fmt not in current.get(name, {}):
                continue
            old = cost_value(name, fmt, previous, stats)
            new = cost_value(name, fmt, current, stats)
            changes.append({
                "name": name,
                "format": fmt,
                "old": old,
                "new": new,
                "delta": new - old,
            })
    changes.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return changes[:top_k]


def _flip_details(
    old: Mapping[str, str],
    new: Mapping[str, str],
    costs: dict,
    stats: dict,
    *,
    top_k: int | None = 5,
) -> list[dict]:
    flips = []
    for name in sorted(set(old) | set(new)):
        old_fmt = old.get(name)
        new_fmt = new.get(name)
        if old_fmt is None or new_fmt is None or old_fmt == new_fmt:
            continue
        old_cost = cost_value(name, old_fmt, costs, stats)
        new_cost = cost_value(name, new_fmt, costs, stats)
        flips.append({
            "name": name,
            "from": old_fmt,
            "to": new_fmt,
            "old_cost": old_cost,
            "new_cost": new_cost,
            "delta": new_cost - old_cost,
        })
    flips.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return flips if top_k is None else flips[:top_k]


def _weighted_hamming_detail(
    old: Mapping[str, str],
    new: Mapping[str, str],
    costs: dict,
    stats: dict,
    *,
    eps: float = 1e-12,
) -> dict:
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
    denominator = sum(
        cost_value(name, fmt, costs, stats)
        for name, fmt in new.items()
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "ratio": numerator / max(denominator, eps),
    }


def _top_l2_formats(
    name: str,
    costs: dict,
    stats: dict,
    *,
    top_k: int = 2,
) -> list[dict]:
    values = []
    for fmt in sorted(costs.get(name, {})):
        entry = costs.get(name, {}).get(fmt)
        if not isinstance(entry, dict) or "error" in entry:
            continue
        values.append({
            "format": fmt,
            "cost": cost_value(name, fmt, costs, stats),
        })
    values.sort(key=lambda item: item["cost"])
    return values[:top_k]


def _normalised_disagreement(l2_cost: float, l3_cost: float) -> float:
    denom = max(abs(l2_cost), abs(l3_cost), 1e-12)
    return abs(l3_cost - l2_cost) / denom


def _phase_elapsed(timing: Mapping | None) -> float:
    if not isinstance(timing, Mapping):
        return 0.0
    return float(timing.get("elapsed_seconds", 0.0) or 0.0)


def _max_peak_gb(*timings: Mapping | None) -> float | None:
    values = [
        float(timing["cuda_peak_gb"])
        for timing in timings
        if isinstance(timing, Mapping) and timing.get("cuda_peak_gb") is not None
    ]
    return max(values) if values else None


def _pct(part: float, total: float) -> float:
    return (float(part) / float(total) * 100.0) if total > 1e-12 else 0.0


def _gb(num_bytes: int) -> float:
    return float(num_bytes) / float(1024 ** 3)


def _fmt_gb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}GB"


def build_l3_polish_summary(
    *,
    selected,
    l3_costs: dict,
    before_assignment: Mapping[str, str],
    after_assignment: Mapping[str, str],
    kl_before: float,
    kl_after: float,
    elapsed_seconds: float = 0.0,
    frozen_dp_precision_used=None,
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
        "l3_enabled": True,
        "enabled": True,
        "accepted": bool(accepted),
        "selected_count": len(selected),
        "measured_count": len(l3_costs),
        "kl_before": float(kl_before),
        "kl_after": float(kl_after),
        "regression": regression,
        "elapsed_seconds": float(elapsed_seconds),
        "frozen_dp_precision_used": frozen_dp_precision_used,
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
    total_wall_start = time.monotonic()
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
    ap.add_argument("--l3-uncertainty-rel-tol", type=float, default=0.10)
    ap.add_argument("--l3-min-fraction", type=float, default=0.05)
    ap.add_argument("--l3-max-fraction", type=float, default=0.10)
    ap.add_argument("--l3-safety-fraction", type=float, default=0.02)
    ap.add_argument("--l3-max-lanes-per-batch", type=int, default=8)
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-Linear per-format costs each iteration.")
    args = ap.parse_args(argv)

    work_root = Path(args.work_dir)
    output_root = Path(args.output_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    iteration_trace_path = output_root / "iteration_trace.jsonl"
    l3_trace_path = output_root / "l3_polish_trace.jsonl"
    iteration_trace_path.write_text("")
    l3_trace_path.write_text("")

    probe_load_start = _phase_start("[init] probe load")
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
    probe_load_timing = _phase_end("[init] probe load", probe_load_start)

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
    convergence_reached = False
    previous_histogram_counts: dict[str, int] | None = None
    previous_measured_linears: int | None = None
    previous_assignment_hash: str | None = None
    l2_iteration_walls: list[float] = []
    cache_size_total_bytes = 0
    l2_peak_gb: float | None = None

    for iteration in range(1, int(args.max_iters) + 1):
        _emit(f"[l2] === iteration {iteration}/{int(args.max_iters)} ===")
        cache_dir = output_root / f"activation_cache_iter_{iteration:02d}"
        _emit(f"[l2] iteration {iteration}: building perturbed cache at {cache_dir}")
        cache_start = _phase_start(
            f"[l2] iteration {iteration}: perturbed-cache build at {cache_dir}"
        )
        capture_manifest = capture_perturbed_activation_cache(
            model,
            assignment,
            calib_ids,
            cache_dir,
            input_rows=args.input_rows,
            profile=profile,
        )
        cache_timing = _phase_end(
            f"[l2] iteration {iteration}: perturbed-cache build",
            cache_start,
        )
        cache_bytes = _directory_size_bytes(cache_dir)
        cache_size_total_bytes += cache_bytes
        _emit(
            f"[l2] iteration {iteration}: activation cache size "
            f"{_human_bytes(cache_bytes)}"
        )
        act_cache = ActivationIndex(cache_dir, stats.keys())
        target_names = set(stats.keys())
        missing_act = [n for n in target_names if n not in act_cache]
        cost_path = output_root / f"costs_iter_{iteration:02d}.pkl"
        previous_costs = cost_history[-1] if cost_history else None
        cost_start = _phase_start(f"[l2] iteration {iteration}: cost step")
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
        cost_timing = _phase_end(
            f"[l2] iteration {iteration}: cost step",
            cost_start,
        )
        measured_linears = len(measured_costs)
        if (
            previous_measured_linears is None
            or measured_linears != previous_measured_linears
        ):
            _emit(
                f"[l2] iteration {iteration}: cost step done in "
                f"{cost_timing['elapsed_seconds']:.1f}s, measured "
                f"{measured_linears} Linears"
            )
        previous_measured_linears = measured_linears
        if args.verbose:
            for name in sorted(measured_costs):
                for fmt in sorted(measured_costs.get(name, {})):
                    value = cost_value(name, fmt, measured_costs, stats)
                    _emit(
                        f"[l2] iteration {iteration}: cost "
                        f"{name} {fmt}={value:.6g}"
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

        solve_start = _phase_start(f"[l2] iteration {iteration}: allocator solve")
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
        solve_timing = _phase_end(
            f"[l2] iteration {iteration}: allocator solve",
            solve_start,
        )

        hamming = _weighted_hamming_detail(
            assignment,
            next_assignment,
            smoothed_costs,
            stats,
        )
        flip_frac = hamming["ratio"]
        histogram = _format_histogram(stats, next_assignment, specs, args.target_bits)
        histogram_counts = histogram["counts"]
        top_changes = _top_cost_changes(
            previous_costs,
            measured_costs,
            stats,
            top_k=5,
        )
        flips = _flip_details(
            assignment,
            next_assignment,
            smoothed_costs,
            stats,
            top_k=5,
        )
        ema_weights = _ema_weights(len(cost_history), args.ema_decay)
        assignment_hash_short = _assignment_digest(next_assignment)
        if previous_histogram_counts is None:
            _emit(
                f"[l2] iteration {iteration}: format histogram "
                f"{histogram_counts} total={histogram['total']} "
                f"achieved_bpp={histogram['achieved_bpp']:.4f} "
                f"target_bpp={histogram['target_bpp']:.4f}"
            )
        elif histogram_counts == previous_histogram_counts:
            _emit(f"[l2] iteration {iteration}: format histogram: unchanged")
        else:
            _emit(
                f"[l2] iteration {iteration}: format histogram delta: "
                f"{_format_histogram_delta(previous_histogram_counts, histogram_counts)}"
            )
        previous_histogram_counts = dict(histogram_counts)
        if top_changes:
            rendered = [
                f"({item['name']}, {item['format']}) old={item['old']:.4g} "
                f"new={item['new']:.4g} delta={item['delta']:.4g}"
                for item in top_changes
            ]
            _emit(f"[l2] iteration {iteration}: top cost changes {rendered}")
        else:
            _emit(f"[l2] iteration {iteration}: top cost changes []")
        if flips:
            rendered_flips = [
                f"({item['name']}, {item['from']}->{item['to']})"
                for item in flips
            ]
            _emit(f"[l2] iteration {iteration}: top flips {rendered_flips}")
        else:
            _emit(f"[l2] iteration {iteration}: top flips []")
        _emit(
            f"[l2] iteration {iteration}: EMA weights newest-first "
            f"{[round(w, 6) for w in ema_weights]}"
        )
        if assignment_hash_short != previous_assignment_hash:
            _emit(
                f"[l2] iteration {iteration}: assignment_hash="
                f"{assignment_hash_short}"
            )
        previous_assignment_hash = assignment_hash_short
        if cycle_mode != "none":
            _emit(f"[l2] iteration {iteration}: cycle_mode={cycle_mode}")
        if flip_frac > 1e-6:
            _emit(
                f"[l2] iteration {iteration}: weighted_hamming "
                f"numerator={hamming['numerator']:.6g}, "
                f"denominator={hamming['denominator']:.6g}, "
                f"ratio={flip_frac:.4f}, threshold={args.convergence_frac:.4f}"
            )
        verdict = "converged" if flip_frac <= args.convergence_frac else "continuing"
        _emit(f"[l2] iteration {iteration}: {verdict}")
        iteration_wall = (
            _phase_elapsed(cache_timing)
            + _phase_elapsed(cost_timing)
            + _phase_elapsed(solve_timing)
        )
        l2_iteration_walls.append(iteration_wall)
        iter_peak = _max_peak_gb(cache_timing, cost_timing, solve_timing)
        if iter_peak is not None:
            l2_peak_gb = iter_peak if l2_peak_gb is None else max(l2_peak_gb, iter_peak)
        iteration_log.append(
            {
                "iteration": iteration,
                "cache": capture_manifest,
                "cost_path": str(cost_path),
                "flip_fraction": flip_frac,
                "cycle_mode": cycle_mode,
                "assignment_hash": list(assignment_hash(next_assignment)),
                "iteration_wall_seconds": iteration_wall,
            }
        )
        _append_jsonl(
            iteration_trace_path,
            {
                "iteration": iteration,
                "cache_dir": str(cache_dir),
                "cache_size_bytes": cache_bytes,
                "cost_path": str(cost_path),
                "format_histogram": histogram,
                "top_cost_changes": top_changes,
                "top_flips": flips,
                "ema_weights_newest_first": ema_weights,
                "hamming": hamming,
                "cycle_mode": cycle_mode,
                "assignment_hash": assignment_hash_short,
                "timing": {
                    "cache": cache_timing,
                    "cost": cost_timing,
                    "solve": solve_timing,
                    "iteration_wall_seconds": iteration_wall,
                    "memory_peak_gb": iter_peak,
                },
            },
        )
        assignment = dict(next_assignment)
        assignment_history.append(dict(assignment))
        if verdict == "converged":
            convergence_reached = True
            break

    l3_summary_path = None
    l3_cost_path = None
    l3_flip_count = 0
    l3_kl_improvement_pct = None
    l3_timing_breakdown: dict[str, float] = {}
    l3_wall = 0.0
    l3_peak_gb: float | None = None
    if args.l3_polish:
        _emit("[l3] === polish ===")
        # Per-iteration L3 can reuse this path later; only final polish is
        # exposed because each propagated measurement is a full forward pass.
        l2_assignment = dict(assignment)
        kl_before_start = _phase_start("[l3] validation KL before")
        kl_before = measure_assignment_kl(
            model,
            l2_assignment,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
        )
        kl_before_timing = _phase_end(
            "[l3] validation KL before",
            kl_before_start,
        )
        selection_start = _phase_start("[l3] selection")
        try:
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
        except L3UnsupportedTargetError as exc:
            _emit(f"[l3] ERROR: packed experts present in L3 selection: {exc}")
            raise
        selection_timing = _phase_end("[l3] selection", selection_start)
        n_uncertain = sum(1 for entry in selected if "uncertain" in entry.reasons)
        n_safety = sum(1 for entry in selected if "high_l2_cost" in entry.reasons)
        n_total = len(set(stats) & set(l2_assignment))
        _emit(
            f"[l3] selection details: uncertain {n_uncertain} "
            f"(margin threshold {args.l3_uncertainty_rel_tol:.2f}), "
            f"safety includes {n_safety}, total selected {len(selected)} "
            f"from {n_total} Linears"
        )
        _emit(f"[l3] selecting neighborhood: {len(selected)} candidates")
        for entry in selected:
            if "uncertain" not in entry.reasons:
                continue
            top2 = _top_l2_formats(entry.name, latest_smoothed_costs, stats)
            _emit(
                f"[l3] uncertain {entry.name}: top2={top2} "
                f"margin={entry.margin:.6g} reasons={list(entry.reasons)}"
            )
        l3_measure_start = time.monotonic()
        l3_lane_count = sum(
            len([fmt for fmt in entry.formats if fmt != "BF16"]) + 1
            for entry in selected
        )
        avg_formats = (
            sum(len(entry.formats) for entry in selected) / float(len(selected))
            if selected else 0.0
        )
        _emit(
            f"[l3] measuring propagated costs: {len(selected)} candidates x "
            f"{avg_formats:.2f} formats = {l3_lane_count} total lanes"
        )

        def _l3_progress(event: dict) -> None:
            if event.get("event") == "depth_group_start":
                _emit(
                    f"[l3] depth group {event['group_index']}/"
                    f"{event['group_count']} {event['group']}: start "
                    f"entries={event['entry_count']} lanes={event['lane_count']}"
                )
            elif event.get("event") == "depth_group_end":
                _emit(
                    f"[l3] depth group {event['group_index']}/"
                    f"{event['group_count']} {event['group']}: done in "
                    f"{event['elapsed_seconds']:.1f}s"
                )

        _cuda_reset_peak()
        l3_costs = measure_propagated_costs(
            model,
            l2_assignment,
            selected,
            calib_ids,
            specs,
            work_root=work_root,
            profile=profile,
            max_lanes_per_batch=args.l3_max_lanes_per_batch,
            progress_callback=_l3_progress,
        )
        l3_elapsed_seconds = time.monotonic() - l3_measure_start
        l3_peak_gb = _cuda_peak_gb()
        suffix = f", cuda_peak={l3_peak_gb:.2f}GB" if l3_peak_gb is not None else ""
        _emit(f"[l3] measurement done in {l3_elapsed_seconds:.1f}s{suffix}")
        l3_measure_timing = {
            "elapsed_seconds": l3_elapsed_seconds,
            "cuda_peak_gb": l3_peak_gb,
        }
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
        amplifications = []
        for name in sorted(l3_costs):
            for fmt in sorted(l3_costs.get(name, {})):
                entry = l3_costs[name][fmt]
                if not isinstance(entry, dict):
                    continue
                if "error" in entry:
                    _emit(f"[l3] metric {name} {fmt}: error={entry['error']}")
                    continue
                l3_value = float(entry.get("propagated_end_kl", 0.0))
                output_mse = float(entry.get("downstream_output_mse", 0.0))
                l2_value = cost_value(name, fmt, latest_smoothed_costs, stats)
                disagreement = _normalised_disagreement(l2_value, l3_value)
                _emit(
                    f"[l3] metric {name} {fmt}: propagated_end_kl="
                    f"{l3_value:.6g}, downstream_output_mse={output_mse:.6g}, "
                    f"l3_l2_disagreement={disagreement:.6g}"
                )
                if abs(l2_value) > 1e-12 and fmt != "BF16":
                    amplifications.append({
                        "name": name,
                        "format": fmt,
                        "l2_cost": l2_value,
                        "l3_cost": l3_value,
                        "amplification": l3_value / l2_value,
                    })
        amplifications.sort(
            key=lambda item: item["amplification"],
            reverse=True,
        )
        for item in amplifications[:10]:
            _emit(
                f"[l3] amplification ({item['name']}, {item['format']}) "
                f"L2_cost={item['l2_cost']:.6g} L3_cost={item['l3_cost']:.6g} "
                f"amplification={item['amplification']:.6g}"
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
        total_params = sum(
            int(stats[name].get("n_params", 0) or 0)
            for name in set(stats) & set(l2_assignment)
        )
        open_names = set(l3_candidates)
        frozen_assignment = {
            name: l2_assignment[name]
            for name in sorted((set(stats) & set(l2_assignment)) - open_names)
        }
        frozen_bits = assignment_bit_total(
            stats,
            frozen_assignment,
            {s.name: s for s in specs},
        )
        target_total_bits = float(args.target_bits) * float(total_params)
        open_params = sum(
            int(stats[name].get("n_params", 0) or 0)
            for name in open_names
        )
        remaining_bpp = (
            (target_total_bits - frozen_bits) / float(open_params)
            if open_params > 0 else 0.0
        )
        frozen_bpp_total = (
            frozen_bits / float(total_params)
            if total_params > 0 else 0.0
        )
        _emit(
            f"[l3] solving frozen DP: open={len(open_names)}, "
            f"frozen={len(frozen_assignment)}, remaining_bpp={remaining_bpp:.4f}"
        )
        _emit(
            f"[l3] frozen DP detail: frozen {len(frozen_assignment)} "
            f"({frozen_bpp_total:.4f} bpp total), open {len(open_names)}, "
            f"remaining_bpp {remaining_bpp:.4f}"
        )
        frozen_solve_start = _phase_start("[l3] frozen DP solve")
        if l3_candidates:
            polished_assignment, _chosen, frozen_dp_meta = solve_frozen_l3_neighborhood(
                stats,
                l2_assignment,
                l3_candidates,
                specs,
                target_bits=args.target_bits,
                bit_precision=args.bit_precision,
                return_metadata=True,
            )
        else:
            polished_assignment = dict(l2_assignment)
            frozen_dp_meta = {"frozen_dp_precision_used": "none"}
        frozen_solve_timing = _phase_end(
            "[l3] frozen DP solve",
            frozen_solve_start,
        )
        l3_all_flips = []
        for name in sorted(set(l2_assignment) | set(polished_assignment)):
            old_fmt = l2_assignment.get(name)
            new_fmt = polished_assignment.get(name)
            if old_fmt is None or new_fmt is None or old_fmt == new_fmt:
                continue
            per_name = l3_costs.get(name, {})
            old_cost = per_name.get(old_fmt, {}).get("propagated_end_kl")
            new_cost = per_name.get(new_fmt, {}).get("propagated_end_kl")
            if old_cost is None or new_cost is None:
                delta = 0.0
            else:
                delta = float(new_cost) - float(old_cost)
            l3_all_flips.append({
                "name": name,
                "from": old_fmt,
                "to": new_fmt,
                "old_cost": old_cost,
                "new_cost": new_cost,
                "delta": delta,
            })
        l3_flip_count = len(l3_all_flips)
        _emit(f"[l3] polish: {l3_flip_count} flips out of {len(selected)}")
        for item in sorted(
            l3_all_flips,
            key=lambda row: abs(row["delta"]),
            reverse=True,
        )[:20]:
            _emit(
                f"[l3] flip {item['name']} {item['from']}->{item['to']} "
                f"L3_delta={item['delta']:.6g}"
            )
        kl_after_start = _phase_start("[l3] validation KL after")
        kl_after = measure_assignment_kl(
            model,
            polished_assignment,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
        )
        kl_after_timing = _phase_end("[l3] validation KL after", kl_after_start)
        regression = bool(float(kl_after) > float(kl_before))
        l3_kl_improvement_pct = (
            (float(kl_before) - float(kl_after)) / abs(float(kl_before)) * 100.0
            if abs(float(kl_before)) > 1e-12 else 0.0
        )
        _emit(
            f"[l3] validating: KL_before={kl_before:.4e}, "
            f"KL_after={kl_after:.4e}, regression={str(regression).lower()}, "
            f"improvement={l3_kl_improvement_pct:.2f}%"
        )
        validation_wall = (
            _phase_elapsed(kl_before_timing)
            + _phase_elapsed(kl_after_timing)
        )
        l3_timing_breakdown = {
            "selection": _phase_elapsed(selection_timing),
            "measurement": _phase_elapsed(l3_measure_timing),
            "dp": _phase_elapsed(frozen_solve_timing),
            "validation": validation_wall,
        }
        l3_wall = sum(l3_timing_breakdown.values())
        l3_peak_gb = _max_peak_gb(
            kl_before_timing,
            selection_timing,
            l3_measure_timing,
            frozen_solve_timing,
            kl_after_timing,
        )
        l3_summary = build_l3_polish_summary(
            selected=selected,
            l3_costs=l3_costs,
            before_assignment=l2_assignment,
            after_assignment=polished_assignment,
            kl_before=kl_before,
            kl_after=kl_after,
            elapsed_seconds=l3_elapsed_seconds,
            frozen_dp_precision_used=frozen_dp_meta["frozen_dp_precision_used"],
        )
        l3_summary["cost_path"] = str(l3_cost_path)
        l3_summary["timing"] = {
            "kl_before": kl_before_timing,
            "selection": selection_timing,
            "measurement": l3_measure_timing,
            "frozen_solve": frozen_solve_timing,
            "kl_after": kl_after_timing,
            "l3_wall_seconds": l3_wall,
        }
        l3_summary_path = output_root / "l3_polish_summary.json"
        with open(l3_summary_path, "w") as f:
            json.dump(l3_summary, f, indent=2)
        for entry in selected:
            per_name = l3_costs.get(entry.name, {})
            propagated = {}
            downstream = {}
            disagreements = {}
            for fmt, cost_entry in per_name.items():
                if not isinstance(cost_entry, dict) or "error" in cost_entry:
                    continue
                l3_value = float(cost_entry.get("propagated_end_kl", 0.0))
                propagated[fmt] = l3_value
                downstream[fmt] = float(cost_entry.get("downstream_output_mse", 0.0))
                l2_value = cost_value(entry.name, fmt, latest_smoothed_costs, stats)
                disagreements[fmt] = _normalised_disagreement(l2_value, l3_value)
            _append_jsonl(
                l3_trace_path,
                {
                    "name": entry.name,
                    "format_candidates": list(entry.formats),
                    "propagated_end_kl": propagated,
                    "downstream_output_mse": downstream,
                    "l2_disagreement": disagreements,
                    "accepted_flip": (
                        l2_assignment.get(entry.name)
                        != polished_assignment.get(entry.name)
                    ),
                    "from": l2_assignment.get(entry.name),
                    "to": polished_assignment.get(entry.name),
                    "timing": l3_summary["timing"],
                },
            )
        assignment = dict(polished_assignment)

    final_assignment_path = output_root / "final_assignment.json"
    with open(final_assignment_path, "w") as f:
        json.dump(dict(sorted(assignment.items())), f, indent=2)
    final_layer_config = output_root / "final_layer_config.json"
    write_layer_config(assignment, final_layer_config)
    total_wall = time.monotonic() - total_wall_start
    l2_wall = sum(l2_iteration_walls)
    baseline_wall = max(total_wall - l2_wall - l3_wall, 0.0)
    additional_iter_wall = sum(l2_iteration_walls[1:])
    single_iter_wall = l2_iteration_walls[0] if l2_iteration_walls else 0.0
    marginal_cost = {
        "total_wall_seconds": total_wall,
        "l2_wall_seconds": l2_wall,
        "l2_wall_pct": _pct(l2_wall, total_wall),
        "l2_iteration_wall_seconds": list(l2_iteration_walls),
        "additional_iter_wall_seconds": additional_iter_wall,
        "additional_iter_extra_pct": _pct(additional_iter_wall, single_iter_wall),
        "l3_wall_seconds": l3_wall,
        "l3_wall_pct": _pct(l3_wall, total_wall),
        "l3_timing_breakdown": dict(l3_timing_breakdown),
        "baseline_wall_seconds": baseline_wall,
        "l2_cuda_peak_gb": l2_peak_gb,
        "l3_cuda_peak_gb": l3_peak_gb,
        "perturbed_cache_disk_gb": _gb(cache_size_total_bytes),
        "perturbed_cache_count": len(iteration_log),
    }
    summary = {
        "iterations": iteration_log,
        "final_assignment": str(final_assignment_path),
        "final_layer_config": str(final_layer_config),
        "target_bits": args.target_bits,
        "converged": convergence_reached,
        "probe_load_timing": probe_load_timing,
        "iteration_trace": str(iteration_trace_path),
        "marginal_cost": marginal_cost,
    }
    if l3_summary_path is not None:
        summary["l3_polish_summary"] = str(l3_summary_path)
        summary["l3_polish_trace"] = str(l3_trace_path)
    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _emit("===== summary =====")
    _emit(f"L2 iterations: {len(iteration_log)}")
    _emit(f"Converged: {str(convergence_reached).lower()}")
    if args.l3_polish:
        improvement = (
            f"{l3_kl_improvement_pct:.2f}%"
            if l3_kl_improvement_pct is not None else "n/a"
        )
        _emit(f"L3 flips: {l3_flip_count}")
        _emit(f"KL improvement: {improvement}")
    _emit(f"Total wall time: {total_wall:.1f}s")
    _emit("== marginal cost ==")
    _emit(f"total wall: {total_wall:.1f}s")
    _emit(
        f"  L2 (perturbed-X iteration): {l2_wall:.1f}s "
        f"({_pct(l2_wall, total_wall):.1f}% of wall)"
    )
    iter_parts = [
        f"iter{idx}={seconds:.1f}s"
        for idx, seconds in enumerate(l2_iteration_walls, start=1)
    ]
    _emit(f"    iteration breakdown: [{', '.join(iter_parts)}]")
    _emit(
        f"    marginal cost of additional iters (vs single-pass): "
        f"+{additional_iter_wall:.1f}s "
        f"({_pct(additional_iter_wall, single_iter_wall):.1f}% extra)"
    )
    _emit(
        f"  L3 polish: {l3_wall:.1f}s ({_pct(l3_wall, total_wall):.1f}% "
        f"of wall) - selection={l3_timing_breakdown.get('selection', 0.0):.1f}s, "
        f"measurement={l3_timing_breakdown.get('measurement', 0.0):.1f}s, "
        f"DP={l3_timing_breakdown.get('dp', 0.0):.1f}s, "
        f"validation={l3_timing_breakdown.get('validation', 0.0):.1f}s"
    )
    _emit(f"  baseline (probe/cost loading, model setup): {baseline_wall:.1f}s")
    _emit(f"memory peak: L2={_fmt_gb(l2_peak_gb)}, L3={_fmt_gb(l3_peak_gb)}")
    _emit(
        f"disk: perturbed-X caches total {_gb(cache_size_total_bytes):.3f}GB "
        f"across {len(iteration_log)} iters"
    )
    _emit(f"Summary manifest: {output_root / 'summary.json'}")
    _emit(f"Iteration trace: {iteration_trace_path}")
    if l3_summary_path is not None:
        _emit(f"L3 summary: {l3_summary_path}")
        _emit(f"L3 trace: {l3_trace_path}")
        _emit(f"L3 costs: {l3_cost_path}")
    _emit("===================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
