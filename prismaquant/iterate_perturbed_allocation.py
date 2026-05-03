#!/usr/bin/env python3
"""Iterate allocation against perturbed activation caches.

Each iteration builds an activation cache under the previous allocation,
measures a fresh W*A* cost table from that cache, smooths costs with a
per-(layer, format) EMA, and re-solves the allocator. The loop stops when the
weighted assignment change is small or when max_iters is reached.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import tempfile
import time
from contextlib import nullcontext
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from pathlib import Path

import torch

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    aggregate_fused_siblings,
    build_candidates,
    expand_fused_sibling_assignment,
)
from prismaquant.allocator_solver import Candidate, promote_fused, solve_allocation
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    kl_divergence,
    load_wikitext_calibration,
)
from prismaquant.measure_quant_cost import ActivationIndex, run_cost_pass
from prismaquant.memory_management import (
    cuda_memory_info,
    phase_boundary_memory_cleanup,
    report_graph_memory,
)
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    calibration_data_hash,
    capture_perturbed_activation_cache,
    load_text_model_under_work_root,
)
from prismaquant.layer_state_cache import LayerHiddenStateCache
from prismaquant.interaction_refine import (
    RefinementUnit,
    UnitOption,
    make_pair_key,
    sparse_local_refine,
)
from prismaquant.propagated_cost import (
    CUDAGraphRegistry,
    FrozenBudgetError,
    L3NeighborhoodEntry,
    L3UnsupportedTargetError,
    _env_cuda_graphs_enabled_for_call_count,
    assignment_bit_total,
    build_global_l3_neighborhood,
    build_l3_candidates,
    layer_depth,
    measure_lane_batched_kl_deltas,
    measure_override_paired_kl_deltas,
    measure_propagated_costs,
    select_l3_neighborhood,
    solve_frozen_l3_neighborhood,
)
from prismaquant.schemas import validate_cost_payload, validate_probe_payload


_KL_CUDA_GRAPH_REGISTRY = CUDAGraphRegistry(
    label="assignment-kl",
    max_entries=4,
    max_entries_env="PRISMAQUANT_KL_CUDA_GRAPH_CACHE_SIZE",
    verbose_env="PRISMAQUANT_KL_CUDA_GRAPHS_VERBOSE",
)


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


def _hamming_count(
    left: Mapping[str, str],
    right: Mapping[str, str],
) -> int:
    return sum(
        1
        for name in set(left) | set(right)
        if left.get(name) != right.get(name)
    )


def _baseline_for_allocated_stats(
    baseline_assignment: Mapping[str, str],
    stats_alloc: Mapping[str, Mapping],
) -> dict[str, str]:
    baseline: dict[str, str] = {}
    for name, entry in stats_alloc.items():
        members = entry.get("_fused_siblings") if isinstance(entry, Mapping) else None
        if members:
            fmts = [
                fr.canonical_format_name(baseline_assignment[member])
                for member in members
                if member in baseline_assignment
            ]
            if fmts:
                baseline[name] = fmts[0]
            continue
        if name in baseline_assignment:
            baseline[name] = fr.canonical_format_name(baseline_assignment[name])
    return baseline


def solve_allocation_with_hamming_cap(
    stats: dict,
    candidates: dict[str, list[Candidate]],
    target_bits: float,
    bit_precision: float = 0.001,
    *,
    baseline_assignment: Mapping[str, str],
    max_flips: int,
) -> tuple[dict[str, str], dict[str, Candidate]] | None:
    """Solve allocation with a cap on Hamming distance from a baseline."""
    import numpy as np

    names = list(candidates.keys())
    total_params = sum(stats[n]["n_params"] for n in names)
    if total_params == 0:
        return {}, {}

    max_flips = max(0, int(max_flips))
    baseline_fmt = {
        name: fr.canonical_format_name(fmt)
        for name, fmt in baseline_assignment.items()
    }
    baselines = {
        name: min(cs, key=lambda c: c.bits_per_param)
        for name, cs in candidates.items()
    }
    min_bits = sum(
        baselines[n].bits_per_param * stats[n]["n_params"]
        for n in names
    ) / total_params
    if target_bits < min_bits - 1e-6:
        return None

    excess = target_bits - min_bits
    n_bins = int(round(excess / bit_precision)) + 2
    flip_bins = max_flips + 1
    INF_NEG = -1e30
    dp = np.full((n_bins, flip_bins), INF_NEG, dtype=np.float64)
    dp[0, 0] = 0.0
    choice: list[np.ndarray] = []

    for name in names:
        baseline = baselines[name]
        cs = candidates[name]
        params = stats[name]["n_params"]
        fraction = params / total_params
        baseline_loss = baseline.predicted_dloss
        options = []
        for idx, cand in enumerate(cs):
            d_avg_bits = (cand.bits_per_param - baseline.bits_per_param) * fraction
            dbins = int(round(d_avg_bits / bit_precision))
            if dbins < 0 or dbins >= n_bins:
                continue
            flip = int(
                name in baseline_fmt
                and fr.canonical_format_name(cand.fmt) != baseline_fmt[name]
            )
            if flip > max_flips:
                continue
            dgain = baseline_loss - cand.predicted_dloss
            options.append((dbins, flip, dgain, idx))
        if not options:
            idx = cs.index(baseline)
            flip = int(
                name in baseline_fmt
                and fr.canonical_format_name(baseline.fmt) != baseline_fmt[name]
            )
            if flip <= max_flips:
                options = [(0, flip, 0.0, idx)]

        new_dp = np.full((n_bins, flip_bins), INF_NEG, dtype=np.float64)
        new_choice = np.full((n_bins, flip_bins), -1, dtype=np.int32)
        for dbins, flip, dgain, idx in options:
            src_b = slice(0, n_bins - dbins) if dbins else slice(None)
            dst_b = slice(dbins, None)
            src_f = slice(0, flip_bins - flip) if flip else slice(None)
            dst_f = slice(flip, None)
            candidate_vals = dp[src_b, src_f] + dgain
            if candidate_vals.size == 0:
                continue
            target_slice = new_dp[dst_b, dst_f]
            mask = candidate_vals > target_slice
            target_slice[:] = np.where(mask, candidate_vals, target_slice)
            choice_slice = new_choice[dst_b, dst_f]
            choice_slice[:] = np.where(mask, idx, choice_slice)
        dp = new_dp
        choice.append(new_choice)

    if dp.max() <= INF_NEG / 2:
        return None
    best_b, best_f = np.unravel_index(int(np.argmax(dp)), dp.shape)
    assignment: dict[str, str] = {}
    chosen_cands: dict[str, Candidate] = {}
    cur_b = int(best_b)
    cur_f = int(best_f)
    for layer_idx in range(len(names) - 1, -1, -1):
        name = names[layer_idx]
        idx_chosen = int(choice[layer_idx][cur_b, cur_f])
        cs = candidates[name]
        if idx_chosen < 0:
            idx_chosen = 0
        chosen = cs[idx_chosen]
        assignment[name] = chosen.fmt
        chosen_cands[name] = chosen
        baseline = baselines[name]
        params = stats[name]["n_params"]
        fraction = params / total_params
        d_avg_bits = (chosen.bits_per_param - baseline.bits_per_param) * fraction
        cur_b -= int(round(d_avg_bits / bit_precision))
        flip = int(
            name in baseline_fmt
            and fr.canonical_format_name(chosen.fmt) != baseline_fmt[name]
        )
        cur_f -= flip
        if cur_b < 0:
            cur_b = 0
        if cur_f < 0:
            cur_f = 0
    return assignment, chosen_cands


def solve_from_costs_with_cap(
    stats: dict,
    costs: dict,
    specs: list[fr.FormatSpec],
    *,
    target_bits: float,
    bit_precision: float,
    baseline_assignment: Mapping[str, str],
    max_flips: int,
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
    baseline_alloc = _baseline_for_allocated_stats(
        baseline_assignment,
        stats_alloc,
    )
    result = solve_allocation_with_hamming_cap(
        stats_alloc,
        candidates,
        target_bits,
        bit_precision,
        baseline_assignment=baseline_alloc,
        max_flips=max_flips,
    )
    if result is None:
        raise RuntimeError(
            f"infeasible target_bits={target_bits} with max_flips={max_flips}"
        )
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


def _format_rank(specs: list[fr.FormatSpec]) -> dict[str, int]:
    return {
        spec.name: idx
        for idx, spec in enumerate(sorted(specs, key=lambda s: s.effective_bits))
    }


def _fused_sibling_members(
    assignment: Mapping[str, str],
    name: str,
    profile=None,
) -> tuple[str, ...]:
    if profile is None or name not in assignment:
        return (name,)
    try:
        group_key = profile.fused_sibling_group(name)
    except Exception:
        group_key = None
    if group_key is None:
        return (name,)
    members = []
    for candidate in assignment:
        try:
            if profile.fused_sibling_group(candidate) == group_key:
                members.append(candidate)
        except Exception:
            continue
    members = sorted(set(members))
    return tuple(members) if members else (name,)


def _enforce_fused_assignment_coherence(
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
    profile=None,
) -> dict[str, str]:
    if profile is None:
        return dict(assignment)
    return promote_fused(dict(assignment), _format_rank(specs), profile=profile)


def _emit(message: str) -> None:
    print(message, flush=True)


def _make_l3_progress(
    emit: Callable[[str], None],
    *,
    prefix: str = "[l3]",
) -> Callable[[dict], None]:
    def _l3_progress(event: dict) -> None:
        if event.get("event") == "depth_group_start":
            emit(
                f"{prefix} depth group {event['group_index']}/"
                f"{event['group_count']} {event['group']}: start "
                f"entries={event['entry_count']} lanes={event['lane_count']} "
                f"mode={event.get('mode', 'unknown')}"
            )
        elif event.get("event") == "depth_group_end":
            emit(
                f"{prefix} depth group {event['group_index']}/"
                f"{event['group_count']} {event['group']}: done in "
                f"{event['elapsed_seconds']:.1f}s"
            )

    return _l3_progress


def _l3_output_mse_names(args) -> list[str] | None:
    """Return downstream-MSE targets for PrismaClade L3 measurement.

    Downstream output-MSE is useful diagnostics, but current L3 allocation and
    coordinate descent consume propagated end-KL. Keep the metric opt-in on the
    PrismaClade path so default runtime measures the actual decision signal.
    """
    return None if bool(getattr(args, "l3_output_mse", False)) else []


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


def _phase_boundary_cleanup(label: str) -> None:
    phase_boundary_memory_cleanup(label)


def _report_graph_memory(label: str) -> None:
    report_graph_memory(label)


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


@dataclass
class BudgetRuntime:
    work_root: Path
    output_root: Path
    stats: dict
    current_costs: dict
    specs: list[fr.FormatSpec]
    profile: object
    model: object
    calib_ids: torch.Tensor
    l3_calib_ids: torch.Tensor
    ref_log_probs: object
    dtype: torch.dtype
    probe_load_timing: dict


@dataclass
class AnchorResult:
    anchor_bpp: float
    output_dir: Path
    l2_assignment: dict[str, str]
    l2_kl: float
    l3_selected: list[L3NeighborhoodEntry]
    l3_candidates: dict
    l3_costs: dict
    latest_smoothed_costs: dict
    l3_cost_history: list[dict] | None = None
    l3_resumed: bool = False


@dataclass
class BudgetResult:
    target_bpp: float
    anchor_bpp: float
    distance_from_anchor: float
    anchor_stale: bool
    achieved_bpp: float
    predicted_dloss: float
    l2_kl: float
    validation_kl: float
    accepted: bool
    regression: bool
    flips_accepted: int
    format_histogram: dict
    assignment: dict[str, str]
    layer_config_path: str
    assignment_path: str = ""
    l3_iterations: int = 0
    coord_descent_fired: bool = False
    coord_descent_flips: int = 0

    def as_record(self) -> dict:
        return {
            "target_bpp": self.target_bpp,
            "anchor_bpp": self.anchor_bpp,
            "distance_from_anchor": self.distance_from_anchor,
            "anchor_stale": self.anchor_stale,
            "achieved_bpp": self.achieved_bpp,
            "predicted_dloss": self.predicted_dloss,
            "l2_kl": self.l2_kl,
            "validation_kl": self.validation_kl,
            "accepted": self.accepted,
            "regression": self.regression,
            "flips_accepted": self.flips_accepted,
            "format_histogram": self.format_histogram,
            "assignment_path": self.assignment_path,
            "layer_config_path": self.layer_config_path,
            "l3_iterations": self.l3_iterations,
            "coord_descent_fired": self.coord_descent_fired,
            "coord_descent_flips": self.coord_descent_flips,
        }


@dataclass
class L3PolishRun:
    initial_assignment: dict[str, str]
    assignment: dict[str, str]
    proposed_assignment: dict[str, str]
    initial_kl: float
    final_kl: float
    proposed_kl: float
    accepted: bool
    regression: bool
    iterations: list[dict]
    selected: list[L3NeighborhoodEntry]
    selected_history: list[list[L3NeighborhoodEntry]]
    l3_cost_history: list[dict]
    l3_costs: dict
    smoothed_l3_costs: dict
    frozen_dp_meta: dict
    measurement_timing: dict
    dp_timing: dict
    validation_timing: dict
    coord_descent_fired: bool
    coord_descent_flips: int
    coord_descent_passes: int
    cycle_detected: bool
    validation_scout: dict | None = None


@dataclass(frozen=True)
class L3ResumeCosts:
    path: Path
    costs: dict
    cost_history: list[dict]
    payload: dict
    mismatches: tuple[str, ...] = ()


class L3ResumeMetadataError(RuntimeError):
    pass


def _move_tensor_tree_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        non_blocking = bool(value.device.type == "cpu" and value.is_pinned())
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            key: _move_tensor_tree_to_device(child, device)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree_to_device(child, device) for child in value)
    if isinstance(value, list):
        return [_move_tensor_tree_to_device(child, device) for child in value]
    return value


def _prepare_kl_tensor_inputs(calib_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not isinstance(calib_ids, torch.Tensor):
        return calib_ids
    if torch.device(device).type == "cuda" and calib_ids.device != device:
        if (
            calib_ids.device.type == "cpu"
            and torch.cuda.is_available()
            and not calib_ids.is_pinned()
        ):
            try:
                calib_ids = calib_ids.pin_memory()
            except RuntimeError:
                pass
        non_blocking = bool(calib_ids.device.type == "cpu" and calib_ids.is_pinned())
        return calib_ids.to(device, non_blocking=non_blocking)
    return calib_ids


def _prepare_ref_log_probs_for_kl(ref_log_probs, device: torch.device):
    if torch.device(device).type != "cuda":
        return ref_log_probs
    return _move_tensor_tree_to_device(ref_log_probs, device)


def _calibration_metadata(args) -> dict:
    return {
        "kl_calib_split": str(getattr(args, "calib_split", "validation")),
        "kl_calib_seed": int(getattr(args, "calib_seed", 4242)),
        "n_calib_samples": int(getattr(args, "n_calib_samples", 0)),
        "calib_seqlen": int(getattr(args, "calib_seqlen", 0)),
        "l3_calib_split": str(getattr(args, "l3_calib_split", "train")),
        "l3_calib_seed": int(getattr(args, "l3_calib_seed", 42)),
        "l3_n_calib_samples": int(getattr(args, "l3_n_calib_samples", 0)),
        "l3_calib_seqlen": int(getattr(args, "l3_calib_seqlen", 0)),
    }


def _warn_endpoint_fallback(context: str, metadata: Mapping | None) -> None:
    if isinstance(metadata, Mapping) and metadata.get("endpoint_fallback"):
        _emit(
            f"[{context}] warning: no interior validated knee found; "
            "using endpoint fallback"
        )


def _l3_resume_cli_force(args) -> bool:
    return bool(
        getattr(args, "resume_l3_ignore_mismatch", False)
        or getattr(args, "force_resume_l3_costs", False)
    )


def _normalise_resume_anchor_key(value: str | float) -> str:
    text = str(value).strip()
    for prefix in ("anchor_bpp_", "anchor_", "bpp_", "target_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return _bpp_label(float(text))


def _parse_resume_l3_costs(value: str | None) -> tuple[Path | None, dict[str, Path]]:
    if not value:
        return None, {}
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) == 1 and "=" not in parts[0]:
        return Path(parts[0]), {}
    by_anchor: dict[str, Path] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(
                "--resume-l3-costs entries must be either one path or "
                "anchor_bpp_X.XX=/path pairs"
            )
        key, path_text = part.split("=", 1)
        by_anchor[_normalise_resume_anchor_key(key)] = Path(path_text)
    return None, by_anchor


def _resume_l3_cost_path_for_anchor(
    args,
    anchor_bpp: float,
    *,
    single_output: bool = False,
) -> Path | None:
    label = _bpp_label(anchor_bpp)
    by_anchor = getattr(args, "_resume_l3_costs_by_anchor", {}) or {}
    if label in by_anchor:
        return Path(by_anchor[label])
    resume_dir = getattr(args, "resume_l3_costs_dir", None)
    if resume_dir:
        root = Path(resume_dir)
        candidates = [
            root / f"anchor_bpp_{label}" / "l3_propagated_costs.pkl",
            root / f"anchor_{label}" / "l3_propagated_costs.pkl",
        ]
        if single_output:
            candidates.insert(0, root / "l3_propagated_costs.pkl")
        for candidate in candidates:
            if candidate.exists():
                return candidate
    single = getattr(args, "_resume_l3_costs_single", None)
    if single is not None:
        return Path(single)
    return None


def _format_set_from_payload(payload: Mapping) -> set[str]:
    meta = payload.get("meta", {}) if isinstance(payload, Mapping) else {}
    formats = None
    if isinstance(meta, Mapping):
        formats = meta.get("formats") or meta.get("format_set")
    if formats is None and isinstance(payload, Mapping):
        formats = payload.get("formats")
    return {
        fr.canonical_format_name(str(fmt))
        for fmt in (formats or [])
    }


def _expected_l3_resume_meta(
    args,
    runtime: BudgetRuntime,
    anchor_bpp: float,
) -> dict:
    return {
        "model": str(getattr(args, "model", "")),
        "anchor_bpp": float(anchor_bpp),
        "formats": [s.name for s in runtime.specs],
        "format_set": {
            fr.canonical_format_name(s.name)
            for s in runtime.specs
        },
        "calib_hash": calibration_data_hash(runtime.l3_calib_ids),
    }


def _l3_resume_metadata_mismatches(
    payload: Mapping,
    *,
    expected: Mapping,
) -> list[str]:
    meta = payload.get("meta", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(meta, Mapping):
        meta = {}
    mismatches: list[str] = []
    actual_model = meta.get("model") or meta.get("model_path")
    if actual_model != expected["model"]:
        mismatches.append(
            f"model expected {expected['model']!r} got {actual_model!r}"
        )
    actual_anchor = meta.get("anchor_bpp")
    try:
        anchor_matches = abs(float(actual_anchor) - float(expected["anchor_bpp"])) <= 1e-9
    except (TypeError, ValueError):
        anchor_matches = False
    if not anchor_matches:
        mismatches.append(
            f"anchor_bpp expected {expected['anchor_bpp']:.2f} got {actual_anchor!r}"
        )
    actual_formats = _format_set_from_payload(payload)
    if actual_formats != set(expected["format_set"]):
        mismatches.append(
            "formats expected "
            f"{sorted(expected['format_set'])!r} got {sorted(actual_formats)!r}"
        )
    actual_hash = (
        meta.get("l3_calib_hash")
        or meta.get("calib_hash")
        or meta.get("calibration_hash")
    )
    if actual_hash != expected["calib_hash"]:
        mismatches.append(
            f"calib_hash expected {expected['calib_hash']!r} got {actual_hash!r}"
        )
    return mismatches


def _load_l3_resume_costs(
    path: Path,
    *,
    args,
    runtime: BudgetRuntime,
    anchor_bpp: float,
    prefix: str,
) -> L3ResumeCosts:
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, Mapping) or "costs" not in payload:
        raise L3ResumeMetadataError(
            f"L3 resume file {path} does not contain a 'costs' payload"
        )
    expected = _expected_l3_resume_meta(args, runtime, anchor_bpp)
    mismatches = _l3_resume_metadata_mismatches(payload, expected=expected)
    if mismatches and not _l3_resume_cli_force(args):
        raise L3ResumeMetadataError(
            f"L3 resume metadata mismatch for {path}: "
            + "; ".join(mismatches)
            + ". Refusing to resume; pass --no-resume-on-mismatch to force."
        )
    if mismatches:
        _emit(
            f"{prefix} WARNING: forcing L3 resume from {path} despite "
            f"metadata mismatch: {'; '.join(mismatches)}"
        )
    cost_history = payload.get("cost_history")
    if not isinstance(cost_history, list) or not all(
        isinstance(item, dict) for item in cost_history
    ):
        cost_history = [payload["costs"]]
    return L3ResumeCosts(
        path=path,
        costs=dict(payload["costs"]),
        cost_history=list(cost_history),
        payload=dict(payload),
        mismatches=tuple(mismatches),
    )


def _l3_cost_payload_meta(
    args,
    runtime: BudgetRuntime,
    *,
    anchor_bpp: float,
    selected_count: int,
    extra: Mapping | None = None,
) -> dict:
    meta = {
        "model": str(getattr(args, "model", "")),
        "probe": str(getattr(args, "probe", "")),
        "anchor_bpp": float(anchor_bpp),
        "formats": [s.name for s in runtime.specs],
        "format_set": sorted(
            fr.canonical_format_name(s.name)
            for s in runtime.specs
        ),
        "calib_hash": calibration_data_hash(runtime.l3_calib_ids),
        "l3_calib_hash": calibration_data_hash(runtime.l3_calib_ids),
        "kl_calib_hash": calibration_data_hash(runtime.calib_ids),
        **_calibration_metadata(args),
        "selected_count": int(selected_count),
    }
    if extra:
        meta.update(dict(extra))
    return meta


def parse_target_bits_list(raw: str) -> list[float]:
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("--target-bits-list must contain at least one value")
    return values


def plan_target_bit_clusters(
    targets: list[float],
    tolerance: float,
    *,
    anchor: float | None = None,
) -> list[dict]:
    if tolerance < 0:
        raise ValueError("target bit share tolerance must be non-negative")
    remaining = list(sorted(targets))
    clusters: list[dict] = []
    anchors: list[float] = []
    if anchor is not None:
        anchors.append(float(anchor))
    while remaining:
        if anchors:
            center = anchors.pop(0)
        else:
            center = remaining[len(remaining) // 2]
        members = [v for v in remaining if abs(v - center) <= tolerance + 1e-12]
        if not members:
            nearest = min(remaining, key=lambda v: abs(v - center))
            center = nearest
            members = [nearest]
        clusters.append({"anchor": float(center), "targets": members})
        remaining = [v for v in remaining if v not in set(members)]
    clusters.sort(key=lambda item: min(item["targets"]))
    return clusters


def widen_l3_neighborhood_formats(
    entries: list[L3NeighborhoodEntry],
    specs: list[fr.FormatSpec],
) -> list[L3NeighborhoodEntry]:
    formats = tuple(spec.name for spec in specs)
    return [
        L3NeighborhoodEntry(
            name=entry.name,
            current_format=entry.current_format,
            formats=formats,
            margin=entry.margin,
            l2_current_cost=entry.l2_current_cost,
            reasons=tuple(dict.fromkeys((*entry.reasons, "all_formats"))),
        )
        for entry in entries
    ]


def _predicted_dloss_for_chosen(chosen: Mapping, stats: Mapping) -> float:
    total_params = sum(int(stats[n].get("n_params", 0) or 0) for n in chosen if n in stats)
    if total_params <= 0:
        return 0.0
    return sum(
        float(c.predicted_dloss) * int(stats[name].get("n_params", 0) or 0)
        / float(total_params)
        for name, c in chosen.items()
        if name in stats
    )


def _predicted_dloss_for_assignment(
    assignment: Mapping[str, str],
    costs: Mapping,
    stats: Mapping,
) -> float:
    total_params = sum(
        int(stats[name].get("n_params", 0) or 0)
        for name in assignment
        if name in stats
    )
    if total_params <= 0:
        return 0.0
    total = 0.0
    for name, fmt in assignment.items():
        if name not in stats:
            continue
        total += (
            cost_value(name, fmt, costs, stats)
            * int(stats[name].get("n_params", 0) or 0)
        )
    return total / float(total_params)


def _solve_global_l3_assignment(
    stats: dict,
    candidates: dict,
    target_bpp: float,
    bit_precision: float,
) -> tuple[dict[str, str], dict]:
    scoped_stats = {name: stats[name] for name in candidates if name in stats}
    result = solve_allocation(scoped_stats, candidates, target_bpp, bit_precision)
    if result is None:
        raise RuntimeError(f"infeasible target_bits={target_bpp}")
    return result


def _assignment_average_bpp(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
) -> float:
    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
    known_assignment = {
        name: fr.canonical_format_name(fmt)
        for name, fmt in assignment.items()
        if name in stats and fr.canonical_format_name(fmt) in specs_by_name
    }
    total_params = sum(
        int(stats[name].get("n_params", 0) or 0)
        for name in known_assignment
    )
    if total_params <= 0:
        return 0.0
    total_bits = assignment_bit_total(stats, known_assignment, specs_by_name)
    return total_bits / float(total_params)


def _within_assignment_budget(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
    target_bpp: float,
    tolerance_bpp: float,
) -> bool:
    achieved = _assignment_average_bpp(stats, assignment, specs)
    return achieved <= float(target_bpp) + max(float(tolerance_bpp), 0.0) + 1e-12


def _filter_l3_candidates_for_assignment(
    l3_candidates: Mapping[str, list[Candidate]],
    assignment: Mapping[str, str],
) -> dict[str, list[Candidate]]:
    current_fmt_by_name = {
        name: fr.canonical_format_name(assignment[name])
        for name in l3_candidates
        if name in assignment
    }
    return {
        name: list(cands)
        for name, cands in l3_candidates.items()
        if _is_l3_mutable_target(name)
        if any(
            fr.canonical_format_name(c.fmt) == current_fmt_by_name.get(name)
            for c in cands
        )
    }


def _is_l3_mutable_target(name: str) -> bool:
    return not (name == "lm_head" or name.endswith(".lm_head"))


def _filter_l3_neighborhood_entries(
    selected: list[L3NeighborhoodEntry],
) -> list[L3NeighborhoodEntry]:
    return [entry for entry in selected if _is_l3_mutable_target(entry.name)]


def _select_l3_neighborhood_for_assignment(
    args,
    stats: dict,
    costs: dict,
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
    *,
    measure_all_formats: bool = False,
) -> list[L3NeighborhoodEntry]:
    if getattr(args, "l3_mode", "selective") == "global":
        selected = build_global_l3_neighborhood(stats, costs, assignment, specs)
    else:
        selected = select_l3_neighborhood(
            stats,
            costs,
            assignment,
            specs,
            uncertainty_rel_tol=getattr(args, "l3_uncertainty_rel_tol", 0.10),
            min_fraction=getattr(args, "l3_min_fraction", 0.05),
            max_fraction=getattr(args, "l3_max_fraction", 0.30),
            safety_fraction=getattr(args, "l3_safety_fraction", 0.02),
        )
    if measure_all_formats:
        selected = widen_l3_neighborhood_formats(selected, specs)
    return _filter_l3_neighborhood_entries(selected)


def _resolve_l3_measure_all_formats(args) -> bool:
    requested = getattr(args, "l3_measure_all_formats", None)
    return bool(requested) if requested is not None else False


def _l3_scope_metadata(args) -> dict:
    return {
        "l3_mode": getattr(args, "l3_mode", "selective"),
        "measure_all_formats": bool(
            getattr(args, "_l3_measure_all_formats", False)
        ),
        "l3_uncertainty_rel_tol": float(
            getattr(args, "l3_uncertainty_rel_tol", 0.10)
        ),
        "l3_min_fraction": float(getattr(args, "l3_min_fraction", 0.05)),
        "l3_max_fraction": float(getattr(args, "l3_max_fraction", 0.30)),
        "l3_safety_fraction": float(getattr(args, "l3_safety_fraction", 0.02)),
        "l3_iter_max": int(getattr(args, "l3_iter_max", 3)),
        "l3_hamming_cap_init": int(getattr(args, "l3_hamming_cap_init", 8)),
        "l3_hamming_cap_max": int(getattr(args, "l3_hamming_cap_max", 64)),
        "l3_validation_scout": bool(getattr(args, "l3_validation_scout", False)),
        "l3_interaction_refine": bool(getattr(args, "l3_interaction_refine", False)),
    }


def _choose_current_l3_candidates(
    assignment: Mapping[str, str],
    l3_candidates: Mapping[str, list[Candidate]],
) -> dict[str, Candidate]:
    chosen: dict[str, Candidate] = {}
    for name, cands in l3_candidates.items():
        current = fr.canonical_format_name(assignment.get(name, "BF16"))
        by_fmt = {fr.canonical_format_name(c.fmt): c for c in cands}
        if current in by_fmt:
            chosen[name] = by_fmt[current]
    return chosen


def _solve_l3_candidates_with_hamming_cap(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    l3_candidates: Mapping[str, list[Candidate]],
    specs: list[fr.FormatSpec],
    *,
    target_bits: float,
    bit_precision: float,
    max_flips: int,
    budget_tolerance: float = 0.0,
    profile=None,
) -> tuple[dict[str, str], dict[str, Candidate], dict]:
    if not l3_candidates:
        return dict(assignment), {}, {
            "frozen_dp_precision_used": "none",
            "hamming_cap": int(max_flips),
            "hamming_cap_infeasible": False,
        }

    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
    all_names = set(stats) & set(assignment)
    open_names = set(l3_candidates) & all_names
    frozen_assignment = {
        name: assignment[name]
        for name in sorted(all_names - open_names)
    }
    total_params = sum(
        int(stats[name].get("n_params", 0) or 0)
        for name in all_names
    )
    open_params = sum(
        int(stats[name].get("n_params", 0) or 0)
        for name in open_names
    )
    if total_params <= 0 or open_params <= 0:
        return dict(assignment), {}, {
            "frozen_dp_precision_used": "none",
            "hamming_cap": int(max_flips),
            "hamming_cap_infeasible": False,
        }

    target_total_bits = float(target_bits) * float(total_params)
    frozen_bits = assignment_bit_total(stats, frozen_assignment, specs_by_name)
    remaining_bits = target_total_bits - frozen_bits
    if remaining_bits < -1e-6:
        return dict(assignment), _choose_current_l3_candidates(
            assignment,
            l3_candidates,
        ), {
            "frozen_dp_precision_used": "none",
            "hamming_cap": int(max_flips),
            "hamming_cap_infeasible": True,
            "remaining_bits": float(remaining_bits),
        }

    open_target_bits = remaining_bits / float(open_params)
    open_stats = {name: dict(stats[name]) for name in sorted(open_names)}
    open_cands = {name: list(l3_candidates[name]) for name in sorted(open_names)}
    open_baseline = {
        name: assignment[name]
        for name in open_cands
        if name in assignment
    }
    result = solve_allocation_with_hamming_cap(
        open_stats,
        open_cands,
        open_target_bits,
        bit_precision,
        baseline_assignment=open_baseline,
        max_flips=max_flips,
    )
    meta = {
        "frozen_dp_precision_used": float(bit_precision),
        "global_dp": len(open_names) == len(all_names),
        "hamming_cap": int(max_flips),
        "hamming_cap_infeasible": result is None,
        "frozen_dp_budget_tolerance": float(budget_tolerance),
        "frozen_dp_budget_tolerance_bits": (
            max(0.0, float(budget_tolerance)) * target_total_bits
        ),
    }
    if result is None:
        return dict(assignment), _choose_current_l3_candidates(
            assignment,
            l3_candidates,
        ), meta
    open_assignment, chosen = result
    merged = dict(assignment)
    merged.update(open_assignment)
    merged = _enforce_fused_assignment_coherence(merged, specs, profile=profile)
    if not _within_assignment_budget(
        stats,
        merged,
        specs,
        target_bits,
        budget_tolerance,
    ):
        meta["fused_coherence_budget_rejected"] = True
        return dict(assignment), _choose_current_l3_candidates(
            assignment,
            l3_candidates,
        ), meta
    meta["fused_coherence_budget_rejected"] = False
    return merged, chosen, meta


def _l3_entry_by_canonical_format(
    per_name: Mapping,
) -> dict[str, Mapping]:
    out: dict[str, Mapping] = {}
    for raw_fmt, entry in per_name.items():
        if not isinstance(entry, Mapping) or "error" in entry:
            continue
        out[fr.canonical_format_name(str(raw_fmt))] = entry
    return out


def _l3_propagated_end_kl(entry: Mapping | None) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("propagated_end_kl")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_bits_total(candidate: Candidate) -> float:
    return 8.0 * float(candidate.memory_bytes)


def _l3_candidate_units(
    assignment: Mapping[str, str],
    l3_candidates: Mapping[str, list[Candidate]],
) -> list[RefinementUnit]:
    units: list[RefinementUnit] = []
    for name in sorted(l3_candidates):
        base_fmt = fr.canonical_format_name(assignment.get(name, "BF16"))
        options_by_fmt: dict[str, UnitOption] = {}
        for cand in l3_candidates.get(name, []):
            fmt = fr.canonical_format_name(cand.fmt)
            options_by_fmt[fmt] = UnitOption(
                fmt=fmt,
                bits_total=_candidate_bits_total(cand),
                predicted_dloss=float(cand.predicted_dloss),
            )
        if base_fmt not in options_by_fmt or len(options_by_fmt) < 2:
            continue
        options = tuple(
            sorted(
                options_by_fmt.values(),
                key=lambda opt: (opt.bits_total, opt.predicted_dloss, opt.fmt),
            )
        )
        units.append(
            RefinementUnit(
                key=name,
                members=(name,),
                base_fmt=base_fmt,
                base_member_fmts=((name, base_fmt),),
                options=options,
            )
        )
    return units


def _l3_option_cost(unit: RefinementUnit, fmt: str) -> float:
    return float(unit.option_map[fr.canonical_format_name(fmt)].predicted_dloss)


def _l3_interaction_unary(
    units: list[RefinementUnit],
) -> dict[str, dict[str, float]]:
    unary: dict[str, dict[str, float]] = {}
    for unit in units:
        base_cost = _l3_option_cost(unit, unit.base_fmt)
        unary[unit.key] = {
            opt.fmt: float(opt.predicted_dloss) - base_cost
            for opt in unit.options
        }
    return unary


def _l3_interaction_allowed_options(
    unit: RefinementUnit,
    *,
    radius: int,
    preferred_fmt: str | None,
) -> tuple[UnitOption, ...]:
    opts = list(unit.options)
    base_idx = next((i for i, opt in enumerate(opts) if opt.fmt == unit.base_fmt), 0)
    radius = max(int(radius), 0)
    selected = {
        opt.fmt: opt
        for opt in opts[max(0, base_idx - radius):base_idx + radius + 1]
    }
    selected[unit.base_fmt] = unit.option_map[unit.base_fmt]
    if preferred_fmt is not None:
        preferred = fr.canonical_format_name(preferred_fmt)
        if preferred in unit.option_map:
            selected[preferred] = unit.option_map[preferred]
    return tuple(
        sorted(
            selected.values(),
            key=lambda opt: (opt.bits_total, opt.predicted_dloss, opt.fmt),
        )
    )


def _l3_select_interaction_units(
    units: list[RefinementUnit],
    candidate_assignment: Mapping[str, str],
    *,
    top_units: int,
) -> list[RefinementUnit]:
    scored = []
    for unit in units:
        candidate_fmt = fr.canonical_format_name(
            candidate_assignment.get(unit.key, unit.base_fmt)
        )
        base_cost = _l3_option_cost(unit, unit.base_fmt)
        if candidate_fmt in unit.option_map:
            candidate_cost = _l3_option_cost(unit, candidate_fmt)
        else:
            candidate_cost = base_cost
        span = max(
            abs(float(opt.predicted_dloss) - base_cost)
            for opt in unit.options
        )
        changed = candidate_fmt != unit.base_fmt
        score = max(abs(candidate_cost - base_cost), span)
        scored.append((0 if changed else 1, -score, unit.key, unit))
    scored.sort()
    return [row[-1] for row in scored[:max(int(top_units), 0)]]


def _assignment_bits_total_from_specs(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
) -> float:
    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
    return assignment_bit_total(
        stats,
        {
            name: fmt
            for name, fmt in assignment.items()
            if name in stats
        },
        specs_by_name,
    )


def _tracked_param_count(stats: Mapping[str, Mapping]) -> int:
    return sum(int(entry.get("n_params", 0) or 0) for entry in stats.values())


def _gpu_memory_gb_for_model(model) -> tuple[float | None, float | None]:
    if model is None or not torch.cuda.is_available():
        return None, None
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda")
    if device.type != "cuda":
        return None, None
    info = cuda_memory_info(device)
    if info is None:
        return None, None
    free_bytes, total_bytes = info
    scale = float(1024 ** 3)
    return float(free_bytes) / scale, float(total_bytes) / scale


def _host_available_memory_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024.0 * 1024.0)
    except OSError:
        return None
    return None


def _memory_aware_enabled(args) -> bool:
    return bool(getattr(args, "memory_aware_l3", True))


def _memory_class(stats: Mapping[str, Mapping]) -> str:
    params_b = _tracked_param_count(stats) / 1.0e9
    if params_b >= 20.0:
        return "xxlarge"
    if params_b >= 6.0:
        return "xlarge"
    if params_b >= 3.0:
        return "large"
    if params_b >= 1.0:
        return "medium"
    return "small"


def _memory_aware_caps(stats: Mapping[str, Mapping]) -> dict[str, int]:
    klass = _memory_class(stats)
    if klass == "xxlarge":
        return {
            "n_calib_samples": 2,
            "calib_seqlen": 128,
            "l3_n_calib_samples": 2,
            "l3_calib_seqlen": 128,
            "input_rows": 64,
            "chunk_size": 64,
            "l3_max_lanes_per_batch": 8,
            "l3_interaction_max_lanes_per_batch": 4,
            "l3_validation_scout_max_candidates": 8,
        }
    if klass == "xlarge":
        return {
            "n_calib_samples": 3,
            "calib_seqlen": 192,
            "l3_n_calib_samples": 3,
            "l3_calib_seqlen": 192,
            "input_rows": 96,
            "chunk_size": 96,
            "l3_max_lanes_per_batch": 12,
            "l3_interaction_max_lanes_per_batch": 8,
            "l3_validation_scout_max_candidates": 12,
        }
    if klass == "large":
        return {
            "n_calib_samples": 2,
            "calib_seqlen": 128,
            "l3_n_calib_samples": 2,
            "l3_calib_seqlen": 128,
            "input_rows": 64,
            "chunk_size": 64,
            "l3_max_lanes_per_batch": 8,
            "l3_interaction_max_lanes_per_batch": 6,
            "l3_validation_scout_max_candidates": 16,
        }
    if klass == "medium":
        return {
            "l3_max_lanes_per_batch": 32,
            "l3_interaction_max_lanes_per_batch": 16,
            "l3_validation_scout_max_candidates": 32,
        }
    return {
        "l3_max_lanes_per_batch": 64,
        "l3_interaction_max_lanes_per_batch": 32,
        "l3_validation_scout_max_candidates": 64,
    }


def _apply_memory_aware_runtime_policy(args, stats: Mapping[str, Mapping], model) -> None:
    if not _memory_aware_enabled(args):
        return
    caps = _memory_aware_caps(stats)
    changes: list[str] = []
    resume_l3_costs = bool(
        getattr(args, "resume_l3_costs", None)
        or getattr(args, "resume_l3_costs_dir", None)
    )
    resume_locked_calibration = {
        "l3_n_calib_samples",
        "l3_calib_seqlen",
    }
    for attr, cap in caps.items():
        if resume_l3_costs and attr in resume_locked_calibration:
            continue
        current = int(getattr(args, attr, cap))
        if cap > 0 and current > cap:
            setattr(args, attr, cap)
            changes.append(f"{attr} {current}->{cap}")
    if changes:
        free_gb, total_gb = _gpu_memory_gb_for_model(model)
        host_free_gb = _host_available_memory_gb()
        params_b = _tracked_param_count(stats) / 1.0e9
        mem_text = "unknown"
        if free_gb is not None and total_gb is not None:
            mem_text = f"free={free_gb:.1f}GB total={total_gb:.1f}GB"
        if host_free_gb is not None:
            mem_text = f"{mem_text} host_available={host_free_gb:.1f}GB"
        _emit(
            "[memory-aware] applying L3 safety caps "
            f"class={_memory_class(stats)} tracked_params={params_b:.2f}B "
            f"{mem_text}: {', '.join(changes)}"
        )


def _memory_aware_lane_cap(
    args,
    stats: Mapping[str, Mapping],
    model,
    requested: int,
    *,
    phase: str,
) -> int:
    requested = max(int(requested), 1)
    if not _memory_aware_enabled(args):
        return requested
    caps = _memory_aware_caps(stats)
    if phase == "interaction":
        cap = caps.get("l3_interaction_max_lanes_per_batch", requested)
    elif phase == "scout":
        cap = caps.get("l3_max_lanes_per_batch", requested)
    else:
        cap = caps.get("l3_max_lanes_per_batch", requested)
    free_gb, _total_gb = _gpu_memory_gb_for_model(model)
    host_free_gb = _host_available_memory_gb()
    if free_gb is not None:
        if free_gb < 12.0:
            cap = min(cap, 2)
        elif free_gb < 24.0:
            cap = min(cap, 4)
        elif free_gb < 40.0:
            cap = min(cap, 8)
        elif free_gb < 70.0:
            cap = min(cap, 16)
    if host_free_gb is not None:
        if host_free_gb < 8.0:
            cap = min(cap, 1)
        elif host_free_gb < 16.0:
            cap = min(cap, 2)
        elif host_free_gb < 32.0:
            cap = min(cap, 4)
        elif host_free_gb < 64.0:
            cap = min(cap, 8)
    return max(min(requested, int(cap)), 1)


def _l3_interaction_enabled(args) -> bool:
    return bool(getattr(args, "l3_interaction_refine", False))


def run_l3_interaction_refine(
    args,
    model,
    base_assignment: Mapping[str, str],
    candidate_assignment: Mapping[str, str],
    l3_candidates: Mapping[str, list[Candidate]],
    stats: Mapping[str, Mapping],
    specs: list[fr.FormatSpec],
    target_bpp: float,
    calib_ids: torch.Tensor,
    *,
    work_root: Path,
    profile=None,
    emit: Callable[[str], None] = _emit,
    prefix: str = "[l3]",
) -> tuple[dict[str, str], dict]:
    meta = {
        "enabled": _l3_interaction_enabled(args),
        "attempted": False,
        "selected_units": 0,
        "pair_measurements": 0,
        "changed": False,
        "skipped_reason": None,
        "solver_mode": str(getattr(args, "l3_interaction_solver", "budget")),
        "lagrangian_bpp_penalty": float(
            getattr(args, "l3_interaction_lagrangian_bpp_penalty", 0.0)
        ),
        "max_seconds": float(getattr(args, "l3_interaction_max_seconds", 600.0)),
    }
    if not meta["enabled"]:
        meta["skipped_reason"] = "disabled"
        return dict(candidate_assignment), meta

    units = _l3_candidate_units(base_assignment, l3_candidates)
    if len(units) < 2:
        meta["skipped_reason"] = "too_few_units"
        return dict(candidate_assignment), meta

    total_params = sum(
        int(stats[name].get("n_params", 0) or 0)
        for name in set(stats) & set(base_assignment)
    )
    if total_params <= 0:
        meta["skipped_reason"] = "empty_stats"
        return dict(candidate_assignment), meta
    target_total_bits = float(target_bpp) * float(total_params)
    budget_tolerance_bits = (
        max(float(getattr(args, "bit_precision", 0.001)), 0.0)
        * float(total_params)
    )
    base_total_bits = _assignment_bits_total_from_specs(
        stats,
        base_assignment,
        specs,
    )
    candidate_total_bits = _assignment_bits_total_from_specs(
        stats,
        candidate_assignment,
        specs,
    )
    effective_target_total_bits = max(
        target_total_bits + budget_tolerance_bits,
        candidate_total_bits,
    )
    meta["base_total_bits"] = float(base_total_bits)
    meta["candidate_total_bits"] = float(candidate_total_bits)
    meta["target_total_bits"] = float(target_total_bits)
    meta["budget_tolerance_bits"] = float(budget_tolerance_bits)

    selected_units = _l3_select_interaction_units(
        units,
        candidate_assignment,
        top_units=int(getattr(args, "l3_interaction_top_units", 8)),
    )
    if len(selected_units) < 2:
        meta["skipped_reason"] = "too_few_selected_units"
        return dict(candidate_assignment), meta

    radius = int(getattr(args, "l3_interaction_neighbor_radius", 1))
    allowed = {
        unit.key: _l3_interaction_allowed_options(
            unit,
            radius=radius,
            preferred_fmt=candidate_assignment.get(unit.key),
        )
        for unit in selected_units
    }
    selected_keys = {unit.key for unit in selected_units}
    fixed_bits_total = _assignment_bits_total_from_specs(
        stats,
        {
            name: fmt
            for name, fmt in candidate_assignment.items()
            if name not in selected_keys
        },
        specs,
    )
    unary = _l3_interaction_unary(selected_units)
    unit_map = {unit.key: unit for unit in selected_units}

    overrides: list[dict[str, str]] = []
    override_meta: list[tuple[str, str, str, str, bool]] = []
    measured_seen: set[tuple[tuple[str, str], ...]] = set()
    max_pairs = max(int(getattr(args, "l3_interaction_max_pairs", 0)), 0)
    for left, right in combinations(selected_units, 2):
        combos: list[tuple[str, str, bool]] = []
        combos.append((left.base_fmt, right.base_fmt, True))
        for left_opt in allowed[left.key]:
            for right_opt in allowed[right.key]:
                if left_opt.fmt == left.base_fmt and right_opt.fmt == right.base_fmt:
                    continue
                combos.append((left_opt.fmt, right_opt.fmt, False))
        for left_fmt, right_fmt, is_base in combos:
            override = {
                left.key: fr.canonical_format_name(left_fmt),
                right.key: fr.canonical_format_name(right_fmt),
            }
            key = tuple(sorted(override.items()))
            if key in measured_seen:
                continue
            measured_seen.add(key)
            overrides.append(override)
            override_meta.append((left.key, left_fmt, right.key, right_fmt, is_base))
            if max_pairs > 0 and len(overrides) >= max_pairs:
                break
        if max_pairs > 0 and len(overrides) >= max_pairs:
            break

    if not overrides:
        meta["skipped_reason"] = "empty_pair_frontier"
        return dict(candidate_assignment), meta

    meta.update(
        {
            "attempted": True,
            "selected_units": len(selected_units),
            "pair_measurements": len(overrides),
        }
    )
    emit(
        f"{prefix} interaction refine: selected {len(selected_units)} units; "
        f"measuring {len(overrides)} paired overrides"
    )
    pair_measure_start = time.monotonic()

    class _InteractionTimeBudgetExceeded(RuntimeError):
        pass

    def _interaction_progress(event: dict) -> None:
        elapsed = time.monotonic() - pair_measure_start
        max_seconds = float(getattr(args, "l3_interaction_max_seconds", 600.0))
        if (
            max_seconds > 0.0
            and event.get("event") == "paired_override_chunk_start"
            and elapsed >= max_seconds
        ):
            raise _InteractionTimeBudgetExceeded(
                f"interaction refine exceeded {max_seconds:.1f}s before "
                f"chunk {event.get('chunk_index')}"
            )
        kind = event.get("event")
        if kind == "paired_override_chunk_start":
            emit(
                f"{prefix} interaction refine: pair chunk "
                f"{event['chunk_index']}/{event['chunk_count']} start "
                f"overrides={event['override_count']} lanes={event['lane_count']}"
            )
        elif kind == "paired_override_chunk_end":
            emit(
                f"{prefix} interaction refine: pair chunk "
                f"{event['chunk_index']}/{event['chunk_count']} done "
                f"overrides={event['override_count']} "
                f"batches={event['batch_count']} "
                f"elapsed={float(event['elapsed_seconds']):.1f}s"
            )

    requested_pair_lanes = min(
        int(getattr(args, "l3_max_lanes_per_batch", 64)),
        int(getattr(args, "l3_interaction_max_lanes_per_batch", 16)),
    )
    pair_lanes = _memory_aware_lane_cap(
        args,
        stats,
        model,
        requested_pair_lanes,
        phase="interaction",
    )
    if pair_lanes < requested_pair_lanes:
        emit(
            f"{prefix} interaction refine: memory-aware lane cap "
            f"{requested_pair_lanes}->{pair_lanes}"
        )
    try:
        pair_kls = measure_override_paired_kl_deltas(
            model,
            base_assignment,
            overrides,
            calib_ids,
            work_root=work_root,
            max_lanes_per_batch=pair_lanes,
            profile=profile,
            progress_callback=_interaction_progress,
        )
    except _InteractionTimeBudgetExceeded as exc:
        pair_measure_seconds = time.monotonic() - pair_measure_start
        meta["pair_measurement_seconds"] = float(pair_measure_seconds)
        meta["stopped_reason"] = "time_budget_exceeded"
        meta["skipped_reason"] = "time_budget_exceeded"
        emit(f"{prefix} interaction refine: {exc}; returning additive candidate")
        return dict(candidate_assignment), meta
    pair_measure_seconds = time.monotonic() - pair_measure_start
    meta["pair_measurement_seconds"] = float(pair_measure_seconds)
    emit(
        f"{prefix} interaction refine: measured {len(overrides)} paired "
        f"overrides in {pair_measure_seconds:.1f}s"
    )

    base_residual_by_pair: dict[tuple[str, str], float] = {}
    residual_rows = []
    for (left_key, left_fmt, right_key, right_fmt, is_base), pair_kl in zip(
        override_meta,
        pair_kls,
    ):
        left_unit = unit_map[left_key]
        right_unit = unit_map[right_key]
        residual = (
            float(pair_kl)
            - _l3_option_cost(left_unit, left_fmt)
            - _l3_option_cost(right_unit, right_fmt)
        )
        pair_id = tuple(sorted((left_key, right_key)))
        if is_base:
            base_residual_by_pair[pair_id] = residual
        residual_rows.append(
            (left_key, left_fmt, right_key, right_fmt, residual, is_base)
        )

    pairwise: dict[tuple[str, str, str, str], float] = {}
    for left_key, left_fmt, right_key, right_fmt, residual, is_base in residual_rows:
        if is_base:
            continue
        pair_id = tuple(sorted((left_key, right_key)))
        base_residual = base_residual_by_pair.get(pair_id, 0.0)
        pairwise[make_pair_key(left_key, left_fmt, right_key, right_fmt)] = (
            residual - base_residual
        )

    try:
        refined = sparse_local_refine(
            selected_units,
            unary,
            pairwise,
            target_total_bits=effective_target_total_bits,
            fixed_bits_total=fixed_bits_total,
            allowed=allowed,
            max_passes=int(getattr(args, "l3_interaction_max_passes", 4)),
            initial_choices={
                unit.key: fr.canonical_format_name(
                    candidate_assignment.get(unit.key, unit.base_fmt)
                )
                for unit in selected_units
            },
            exact_max_states=(
                int(getattr(args, "l3_interaction_exact_max_states", 2_000_000))
                if bool(getattr(args, "l3_interaction_exact", True))
                else 0
            ),
            solver_mode=str(getattr(args, "l3_interaction_solver", "budget")),
            lagrangian_bpp_penalty=float(
                getattr(args, "l3_interaction_lagrangian_bpp_penalty", 0.0)
            ),
            total_params=float(total_params),
        )
    except ValueError as exc:
        meta["skipped_reason"] = str(exc)
        emit(f"{prefix} interaction refine: skipped {exc}")
        return dict(candidate_assignment), meta

    refined_assignment = dict(candidate_assignment)
    for unit in selected_units:
        fmt = refined["choices"].get(unit.key, unit.base_fmt)
        refined_assignment[unit.key] = fmt
    pre_coherence_assignment = dict(refined_assignment)
    refined_assignment = _enforce_fused_assignment_coherence(
        refined_assignment,
        specs,
        profile=profile,
    )
    if (
        assignment_hash(refined_assignment) != assignment_hash(pre_coherence_assignment)
        and not _within_assignment_budget(
            stats,
            refined_assignment,
            specs,
            target_bpp,
            max(float(getattr(args, "bit_precision", 0.001)), 0.0),
        )
    ):
        meta["skipped_reason"] = "fused_coherence_budget_rejected"
        emit(
            f"{prefix} interaction refine: skipped because fused-sibling "
            "coherence would exceed the target budget"
        )
        return dict(candidate_assignment), meta
    meta.update(
        {
            "objective_delta": float(refined["objective_delta"]),
            "bits_total": float(refined["bits_total"]),
            "bits_per_param": refined.get("bits_per_param"),
            "solver": refined.get("solver"),
            "lagrangian_objective": refined.get("lagrangian_objective"),
            "lagrangian_bpp_penalty": refined.get("lagrangian_bpp_penalty"),
            "states_evaluated": refined.get("states_evaluated"),
            "state_count": refined.get("state_count"),
            "nodes_visited": refined.get("nodes_visited"),
            "states_pruned": refined.get("states_pruned"),
            "changed": assignment_hash(refined_assignment)
            != assignment_hash(candidate_assignment),
        }
    )
    emit(
        f"{prefix} interaction refine: objective_delta="
        f"{float(refined['objective_delta']):.4e} "
        f"solver={refined.get('solver', 'unknown')} "
        f"bits_per_param={refined.get('bits_per_param')} "
        f"changed={str(meta['changed']).lower()}"
    )
    return refined_assignment, meta


def _coord_descent_ranked_candidates(
    assignment: Mapping[str, str],
    l3_costs: Mapping,
    specs: list[fr.FormatSpec],
) -> list[tuple[float, str, str]]:
    spec_names = [fr.canonical_format_name(spec.name) for spec in specs]
    ranked: list[tuple[float, str, str]] = []
    if not isinstance(l3_costs, Mapping):
        return ranked
    for name in sorted(assignment):
        current_fmt = fr.canonical_format_name(assignment[name])
        per_name = l3_costs.get(name, {})
        if not isinstance(per_name, Mapping):
            continue
        by_fmt = _l3_entry_by_canonical_format(per_name)
        current_cost = _l3_propagated_end_kl(by_fmt.get(current_fmt))
        if current_cost is None:
            continue
        for candidate_fmt in spec_names:
            if candidate_fmt == current_fmt:
                continue
            candidate_cost = _l3_propagated_end_kl(by_fmt.get(candidate_fmt))
            if candidate_cost is None:
                continue
            ranked.append((candidate_cost - current_cost, name, candidate_fmt))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked


def _rank_bucket(rank: int) -> str:
    if rank < 32:
        return "rank_000_031"
    if rank < 128:
        return "rank_032_127"
    if rank < 512:
        return "rank_128_511"
    return "rank_512_plus"


def _module_leaf(name: str) -> str:
    return str(name).split(".")[-1]


def _module_family(name: str) -> str:
    if ".self_attn." in name:
        return "self_attn"
    if ".mlp." in name:
        return "mlp"
    return "other"


def _depth_bucket(name: str) -> str:
    depth = layer_depth(name)
    if depth is None:
        return "depth_other"
    start = (int(depth) // 4) * 4
    return f"depth_{start:02d}_{start + 3:02d}"


def _candidate_bucket(
    *,
    rank: int,
    name: str,
    current_fmt: str,
    candidate_fmt: str,
) -> str:
    return "|".join(
        (
            _rank_bucket(rank),
            _depth_bucket(name),
            _module_family(name),
            _module_leaf(name),
            f"{current_fmt}->{candidate_fmt}",
        )
    )


def _search_telemetry_path(args) -> Path | None:
    raw = getattr(args, "_search_telemetry_path", None)
    if raw is None:
        raw = getattr(args, "search_telemetry_jsonl", None)
    if not raw:
        return None
    return Path(raw)


def _append_search_telemetry(args, payload: Mapping) -> None:
    path = _search_telemetry_path(args)
    if path is None:
        return
    _append_jsonl(path, dict(payload))


def _candidate_trial_assignment(
    assignment: Mapping[str, str],
    name: str,
    candidate_fmt: str,
    *,
    profile=None,
) -> dict[str, str]:
    trial = dict(assignment)
    fmt = fr.canonical_format_name(candidate_fmt)
    for member in _fused_sibling_members(trial, name, profile):
        trial[member] = fmt
    return trial


def _candidate_override(
    assignment: Mapping[str, str],
    name: str,
    candidate_fmt: str,
    *,
    profile=None,
) -> dict[str, str]:
    trial = _candidate_trial_assignment(
        assignment,
        name,
        candidate_fmt,
        profile=profile,
    )
    return {
        key: fmt
        for key, fmt in trial.items()
        if fr.canonical_format_name(assignment.get(key, "")) != fmt
    }


def _assignment_bits_bpp(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
) -> tuple[float, float]:
    bits = _assignment_bits_total_from_specs(stats, assignment, specs)
    total_params = sum(
        int(stats[name].get("n_params", 0) or 0)
        for name in set(stats) & set(assignment)
    )
    bpp = bits / float(total_params) if total_params > 0 else 0.0
    return bits, bpp


def _l3_validation_scout_enabled(args) -> bool:
    return bool(getattr(args, "l3_validation_scout", False))


def _l3_validation_scout_candidates(
    args,
    assignment: Mapping[str, str],
    l3_costs: Mapping,
    specs: list[fr.FormatSpec],
    stats: Mapping[str, Mapping],
    target_bpp: float,
    *,
    profile=None,
) -> list[dict]:
    ranked = _coord_descent_ranked_candidates(assignment, l3_costs, specs)
    max_predicted_delta = float(
        getattr(args, "l3_validation_scout_max_predicted_delta", 0.0)
    )
    bpp_slack = max(float(getattr(args, "l3_validation_scout_bpp_slack", 0.25)), 0.0)
    rows: list[dict] = []
    for rank, (predicted_delta, name, candidate_fmt) in enumerate(ranked):
        current_fmt = fr.canonical_format_name(assignment.get(name, ""))
        candidate_fmt = fr.canonical_format_name(candidate_fmt)
        if candidate_fmt == current_fmt:
            continue
        if float(predicted_delta) > max_predicted_delta:
            continue
        trial = _candidate_trial_assignment(
            assignment,
            name,
            candidate_fmt,
            profile=profile,
        )
        if not _within_assignment_budget(
            stats,
            trial,
            specs,
            target_bpp,
            bpp_slack,
        ):
            continue
        bits, bpp = _assignment_bits_bpp(stats, trial, specs)
        rows.append(
            {
                "rank": rank,
                "predicted_delta": float(predicted_delta),
                "name": name,
                "from_format": current_fmt,
                "to_format": candidate_fmt,
                "bucket": _candidate_bucket(
                    rank=rank,
                    name=name,
                    current_fmt=current_fmt,
                    candidate_fmt=candidate_fmt,
                ),
                "trial_bits": float(bits),
                "trial_bpp": float(bpp),
            }
        )
    return rows


def _select_validation_scout_candidates(args, rows: list[dict]) -> list[dict]:
    max_candidates = max(int(getattr(args, "l3_validation_scout_max_candidates", 64)), 0)
    if max_candidates <= 0 or not rows:
        return []
    sample_per_bucket = max(
        int(getattr(args, "l3_validation_scout_sample_per_bucket", 0)),
        0,
    )
    rank_buckets = sorted({_rank_bucket(int(row["rank"])) for row in rows})
    reserve = (
        min(max_candidates, sample_per_bucket * max(len(rank_buckets) - 1, 0))
        if sample_per_bucket > 0 else 0
    )
    top_k = min(max(max_candidates - reserve, 0), len(rows))
    selected = list(rows[:top_k])
    if sample_per_bucket <= 0 or len(selected) >= max_candidates:
        return selected[:max_candidates]

    selected_keys = {(row["name"], row["to_format"]) for row in selected}
    dominant_bucket = _rank_bucket(int(rows[0]["rank"]))
    by_bucket: dict[str, list[dict]] = {}
    for row in rows[top_k:]:
        bucket = _rank_bucket(int(row["rank"]))
        if bucket == dominant_bucket:
            continue
        by_bucket.setdefault(bucket, []).append(row)
    rng = random.Random(int(getattr(args, "l3_validation_scout_seed", 0)))
    for bucket in sorted(by_bucket):
        if len(selected) >= max_candidates:
            break
        candidates = [
            row for row in by_bucket[bucket]
            if (row["name"], row["to_format"]) not in selected_keys
        ]
        if not candidates:
            continue
        take = min(sample_per_bucket, len(candidates), max_candidates - len(selected))
        for row in rng.sample(candidates, take):
            selected.append(row)
            selected_keys.add((row["name"], row["to_format"]))
    selected.sort(key=lambda row: (int(row["rank"]), row["name"], row["to_format"]))
    return selected


def run_l3_validation_scout(
    args,
    model,
    current_assignment: Mapping[str, str],
    current_kl: float,
    l3_costs: Mapping,
    specs: list[fr.FormatSpec],
    target_bpp: float,
    calib_ids: torch.Tensor,
    ref_log_probs,
    *,
    stats: Mapping[str, Mapping],
    work_root: Path,
    profile=None,
    emit: Callable[[str], None] | None = _emit,
    prefix: str = "[l3]",
    checkpoint_callback: Callable[[Mapping[str, str], float, dict], None] | None = None,
) -> tuple[dict[str, str], float, dict]:
    meta = {
        "enabled": _l3_validation_scout_enabled(args),
        "rounds_completed": 0,
        "candidates_evaluated": 0,
        "hits": 0,
        "flips_committed": 0,
        "elapsed_seconds": 0.0,
        "best_gain_kl": 0.0,
        "stopped_reason": "disabled",
    }
    if not meta["enabled"]:
        return dict(current_assignment), float(current_kl), meta

    attempted_by_assignment: dict[str, set[tuple[str, str]]] = {}
    assignment = dict(current_assignment)
    kl = float(current_kl)
    max_rounds = max(int(getattr(args, "l3_validation_scout_rounds", 1)), 0)
    requested_lanes_per_batch = max(
        int(getattr(args, "l3_max_lanes_per_batch", 64)),
        1,
    )
    lanes_per_batch = _memory_aware_lane_cap(
        args,
        stats,
        model,
        requested_lanes_per_batch,
        phase="scout",
    )
    scout_lane_cap = max(
        int(getattr(args, "l3_validation_scout_max_lanes_per_batch", 1)),
        1,
    )
    lanes_per_batch = min(lanes_per_batch, scout_lane_cap)
    min_improvement = max(
        float(getattr(args, "l3_validation_scout_min_improvement", 0.0)),
        0.0,
    )
    commit_best = bool(getattr(args, "l3_validation_scout_commit_best", True))
    overall_start = time.monotonic()
    bits_before, bpp_before = _assignment_bits_bpp(stats, assignment, specs)

    if emit is not None:
        emit(
            f"{prefix} validation scout: start rounds={max_rounds} "
            f"max_candidates={getattr(args, 'l3_validation_scout_max_candidates', 64)} "
            f"sample_per_bucket={getattr(args, 'l3_validation_scout_sample_per_bucket', 0)}"
        )

    for round_idx in range(1, max_rounds + 1):
        assignment_digest = _assignment_digest(assignment)
        attempted_for_assignment = attempted_by_assignment.setdefault(
            assignment_digest,
            set(),
        )
        all_candidate_rows = _l3_validation_scout_candidates(
            args,
            assignment,
            l3_costs,
            specs,
            stats,
            target_bpp,
            profile=profile,
        )
        candidate_rows = [
            row for row in all_candidate_rows
            if (row["name"], row["to_format"]) not in attempted_for_assignment
        ]
        selected = _select_validation_scout_candidates(args, candidate_rows)
        if _memory_aware_enabled(args):
            scout_cap = _memory_aware_caps(stats).get(
                "l3_validation_scout_max_candidates",
                len(selected),
            )
            if len(selected) > scout_cap:
                selected = selected[:scout_cap]
        if not selected:
            meta["stopped_reason"] = (
                "no_untried_candidates" if all_candidate_rows else "no_candidates"
            )
            break
        meta["rounds_completed"] = round_idx
        _append_search_telemetry(
            args,
            {
                "event": "validation_scout_round_start",
                "candidate_source": "l3_ranked_unary",
                "prefix": prefix,
                "round": round_idx,
                "target_bpp": float(target_bpp),
                "eligible_candidates": len(candidate_rows),
                "eligible_candidates_total": len(all_candidate_rows),
                "selected_candidates": len(selected),
                "assignment_hash_before": assignment_digest,
                "bpp_before": float(bpp_before),
            },
        )
        if emit is not None:
            emit(
                f"{prefix} validation scout: round {round_idx} "
                f"evaluating {len(selected)} / {len(candidate_rows)} candidates"
            )

        round_best: tuple[float, dict] | None = None
        chunk_index = 0
        for start in range(0, len(selected), lanes_per_batch):
            chunk_index += 1
            chunk = selected[start:start + lanes_per_batch]
            flips = [(row["name"], row["to_format"]) for row in chunk]
            chunk_start = time.monotonic()
            if emit is not None:
                emit(
                    f"{prefix} validation scout: round {round_idx} "
                    f"chunk {chunk_index} start lanes={len(chunk)}"
                )
            graph_env_name = "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS"
            old_graph_env = os.environ.get(graph_env_name)
            if not bool(getattr(args, "l3_validation_scout_cuda_graphs", False)):
                os.environ[graph_env_name] = "0"
            try:
                needs_group_serial = any(
                    len(_candidate_override(
                        assignment,
                        row["name"],
                        row["to_format"],
                        profile=profile,
                    )) > 1
                    for row in chunk
                )
                if needs_group_serial:
                    measured_kls = [
                        measure_assignment_kl(
                            model,
                            _candidate_trial_assignment(
                                assignment,
                                row["name"],
                                row["to_format"],
                                profile=profile,
                            ),
                            calib_ids,
                            ref_log_probs,
                            work_root=work_root,
                            profile=profile,
                        )
                        for row in chunk
                    ]
                else:
                    measured_kls = measure_lane_batched_kl_deltas(
                        model,
                        assignment,
                        flips,
                        calib_ids,
                        ref_log_probs,
                        work_root=work_root,
                        max_lanes_per_batch=lanes_per_batch,
                        profile=profile,
                    )
            finally:
                if old_graph_env is None:
                    os.environ.pop(graph_env_name, None)
                else:
                    os.environ[graph_env_name] = old_graph_env
                _phase_boundary_cleanup(
                    f"{prefix} validation_scout_round_{round_idx}_chunk_{chunk_index}"
                )
            elapsed = time.monotonic() - chunk_start
            seconds_per_candidate = elapsed / max(len(chunk), 1)
            for row, trial_kl_raw in zip(chunk, measured_kls):
                trial_kl = float(trial_kl_raw)
                gain = max(0.0, kl - trial_kl)
                hit = bool(gain > min_improvement + 1e-12)
                meta["candidates_evaluated"] += 1
                if hit:
                    meta["hits"] += 1
                    meta["best_gain_kl"] = max(float(meta["best_gain_kl"]), gain)
                trial = _candidate_trial_assignment(
                    assignment,
                    row["name"],
                    row["to_format"],
                    profile=profile,
                )
                trial_bits, trial_bpp = _assignment_bits_bpp(stats, trial, specs)
                _append_search_telemetry(
                    args,
                    {
                        "event": "validation_scout_candidate",
                        "candidate_source": "l3_ranked_unary",
                        "prefix": prefix,
                        "round": round_idx,
                        "chunk_index": chunk_index,
                        "target_bpp": float(target_bpp),
                        "eligible_candidates": len(candidate_rows),
                        "eligible_candidates_total": len(all_candidate_rows),
                        "selected_candidates": len(selected),
                        "rank": int(row["rank"]),
                        "bucket": row["bucket"],
                        "name": row["name"],
                        "from_format": row["from_format"],
                        "to_format": row["to_format"],
                        "predicted_delta_kl": float(row["predicted_delta"]),
                        "kl_before": float(kl),
                        "kl_after": trial_kl,
                        "actual_delta_kl": trial_kl - kl,
                        "gain_kl": gain,
                        "hit": hit,
                        "seconds": seconds_per_candidate,
                        "assignment_hash_before": _assignment_digest(assignment),
                        "assignment_hash_after": _assignment_digest(trial),
                        "bits_before": float(bits_before),
                        "bits_after": float(trial_bits),
                        "bpp_before": float(bpp_before),
                        "bpp_after": float(trial_bpp),
                    },
                )
                attempted_for_assignment.add((row["name"], row["to_format"]))
                if round_best is None or trial_kl < round_best[0]:
                    round_best = (trial_kl, row)
            if emit is not None:
                best_text = ""
                if round_best is not None:
                    best_text = (
                        f" best={round_best[1]['name']}->"
                        f"{round_best[1]['to_format']} "
                        f"kl={round_best[0]:.4e}"
                    )
                emit(
                    f"{prefix} validation scout: round {round_idx} "
                    f"chunk {chunk_index} done elapsed={elapsed:.1f}s{best_text}"
                )

        if round_best is None:
            meta["stopped_reason"] = "empty_measurement"
            break
        best_kl, best_row = round_best
        best_gain = kl - float(best_kl)
        if not commit_best:
            meta["stopped_reason"] = "measured_only"
            break
        if best_gain <= min_improvement + 1e-12:
            meta["stopped_reason"] = "no_validated_improvement"
            break
        old_fmt = assignment.get(best_row["name"], "")
        assignment = _candidate_trial_assignment(
            assignment,
            best_row["name"],
            best_row["to_format"],
            profile=profile,
        )
        old_kl = kl
        kl = float(best_kl)
        bits_before, bpp_before = _assignment_bits_bpp(stats, assignment, specs)
        meta["flips_committed"] += 1
        meta["stopped_reason"] = "max_rounds"
        commit_record = {
            "event": "validation_scout_commit",
            "candidate_source": "l3_ranked_unary",
            "prefix": prefix,
            "round": round_idx,
            "target_bpp": float(target_bpp),
            "name": best_row["name"],
            "from_format": old_fmt,
            "to_format": best_row["to_format"],
            "kl_before": float(old_kl),
            "kl_after": float(kl),
            "gain_kl": float(best_gain),
            "assignment_hash_after": _assignment_digest(assignment),
            "bpp_after": float(bpp_before),
        }
        _append_search_telemetry(args, commit_record)
        if checkpoint_callback is not None:
            try:
                checkpoint_callback(dict(assignment), float(kl), dict(commit_record))
            except Exception as exc:
                if emit is not None:
                    emit(
                        f"{prefix} validation scout: checkpoint failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if emit is not None:
            emit(
                f"{prefix} validation scout COMMIT: {best_row['name']}."
                f"{old_fmt} -> {best_row['to_format']} "
                f"kl {old_kl:.4e} -> {kl:.4e} "
                f"(gain={best_gain:.4e})"
            )

    meta["elapsed_seconds"] = time.monotonic() - overall_start
    if emit is not None:
        emit(
            f"{prefix} validation scout: done rounds={meta['rounds_completed']} "
            f"evaluated={meta['candidates_evaluated']} hits={meta['hits']} "
            f"commits={meta['flips_committed']} "
            f"halted={meta['stopped_reason']} KL={kl:.4e}"
        )
    return assignment, kl, meta


def _coord_descent_lane_batch_count(
    ranked_candidates: list[tuple[float, str, str]],
    assignment: Mapping[str, str],
    stats: Mapping[str, Mapping],
    specs: list[fr.FormatSpec],
    target_bpp: float,
    bit_precision: float,
    max_lanes_per_batch: int,
    *,
    start_index: int = 0,
    profile=None,
) -> int:
    lanes_per_batch = max(int(max_lanes_per_batch), 1)
    batch_count = 0
    lanes_in_batch = 0
    start_index = max(int(start_index), 0)
    for _predicted_delta, name, candidate_fmt in ranked_candidates[start_index:]:
        if candidate_fmt == fr.canonical_format_name(assignment.get(name, "")):
            continue
        trial = _candidate_trial_assignment(
            assignment,
            name,
            candidate_fmt,
            profile=profile,
        )
        if not _within_assignment_budget(
            stats,
            trial,
            specs,
            target_bpp,
            bit_precision,
        ):
            continue
        lanes_in_batch += 1
        if lanes_in_batch >= lanes_per_batch:
            batch_count += 1
            lanes_in_batch = 0
    if lanes_in_batch:
        batch_count += 1
    return batch_count


def _coord_lane_batch_enabled() -> bool:
    value = os.environ.get("PRISMAQUANT_COORD_LANE_BATCH")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_flag_enabled(name: str, *, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _coord_replay_cache_enabled() -> bool:
    value = os.environ.get("PRISMAQUANT_COORD_REPLAY_CACHE")
    if value is None:
        return False
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _coord_replay_cache_dtype(model) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _populate_coord_replay_cache(
    model,
    assignment: Mapping[str, str],
    calib_ids: torch.Tensor,
) -> LayerHiddenStateCache | None:
    try:
        cache = LayerHiddenStateCache(model)
    except (AttributeError, TypeError, ValueError):
        return None
    device = next(model.parameters()).device
    dtype = _coord_replay_cache_dtype(model)
    rng_devices = []
    if device.type == "cuda" and torch.cuda.is_available():
        rng_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(0)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        cache.populate(
            assignment,
            calib_ids,
            device=str(device),
            dtype=dtype,
        )
    return cache


def coordinate_descent_polish(
    model,
    current_assignment: Mapping[str, str],
    l3_costs: Mapping,
    specs: list[fr.FormatSpec],
    target_bpp: float,
    calib_ids: torch.Tensor,
    ref_log_probs,
    *,
    stats: Mapping[str, Mapping],
    work_root: Path,
    profile=None,
    bit_precision: float = 0.001,
    max_passes: int = 1,
    early_stop_streak: int = 50,
    current_kl: float | None = None,
    return_metadata: bool = False,
    emit: Callable[[str], None] | None = None,
    anchor_label: str | None = None,
    max_lanes_per_batch: int = 64,
) -> tuple[dict[str, str], float] | tuple[dict[str, str], float, dict]:
    assignment = dict(current_assignment)
    if current_kl is None:
        current_kl = measure_assignment_kl(
            model,
            assignment,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
        )
    current_kl = float(current_kl)
    passes_completed = 0
    flips_committed = 0
    measurements = 0
    failed_streak = 0
    halted = "max_passes" if int(max_passes) > 0 else "max_passes"
    anchor_text = f"anchor {anchor_label}" if anchor_label is not None else "anchor n/a"
    lane_batch_enabled = _coord_lane_batch_enabled()
    replay_cache_enabled = lane_batch_enabled and _coord_replay_cache_enabled()
    replay_cache: LayerHiddenStateCache | None = None
    replay_cache_populates = 0
    replay_cache_active = False

    hooks: PerturbedActivationCache | None = None
    cached_names: set[str] = set()
    cache_cm = nullcontext()
    if int(max_passes) > 0 and not lane_batch_enabled:
        Path(work_root).mkdir(parents=True, exist_ok=True)
        cache_dir = Path(
            tempfile.mkdtemp(
                prefix="prismaquant_coord_kl_hooks_",
                dir=str(work_root),
            )
        )
        hooks = PerturbedActivationCache(
            model,
            assignment,
            cache_dir,
            input_rows=0,
            cal_hash=calibration_data_hash(calib_ids),
            profile=profile,
        )
        cached_names = {
            param_plan.name
            for plan in hooks.plans
            for param_plan in plan.params
        }
        if cached_names and _env_flag_enabled(
            "PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE",
            default=True,
        ):
            cache_cm = hooks.frozen_weight_cache()

    with cache_cm:
        if hooks is not None and cached_names:
            hooks.install()
        try:
            pass_limit = max(0, int(max_passes))
            for _pass_idx in range(pass_limit):
                passes_completed += 1
                pass_index = passes_completed
                pass_start = time.monotonic()
                pass_measurements_start = measurements
                pass_flips_start = flips_committed
                pass_improved = False
                ranked_candidates = _coord_descent_ranked_candidates(
                    assignment,
                    l3_costs,
                    specs,
                )

                def _emit_pass_done(reason: str | None = None) -> None:
                    if emit is None:
                        return
                    pass_measurements = measurements - pass_measurements_start
                    pass_flips = flips_committed - pass_flips_start
                    pass_halt_reason = halted if reason is None else reason
                    emit(
                        f"[coord] {anchor_text} pass {pass_index} done: "
                        f"tried_{pass_measurements}_flips accepted_{pass_flips} "
                        f"halted={pass_halt_reason} "
                        f"elapsed={time.monotonic() - pass_start:.0f}s"
                    )

                if emit is not None:
                    if ranked_candidates:
                        best_pred, best_name, best_fmt = ranked_candidates[0]
                        emit(
                            f"[coord] {anchor_text}: L3-ranked "
                            f"{len(ranked_candidates)} candidates; best="
                            f"{best_name}->{best_fmt} predicted_delta={best_pred:.6g}"
                        )
                    else:
                        emit(f"[coord] {anchor_text}: L3-ranked 0 candidates")
                if not ranked_candidates:
                    halted = "no_improvement"
                    _emit_pass_done()
                    break

                if lane_batch_enabled:
                    pass_batch_index = 0
                    replay_cache = None
                    if (
                        replay_cache_enabled
                        and any(
                            layer_depth(name) is not None
                            for _delta, name, _fmt in ranked_candidates
                        )
                    ):
                        replay_cache = _populate_coord_replay_cache(
                            model,
                            assignment,
                            calib_ids,
                        )
                        if replay_cache is not None:
                            replay_cache_populates += 1
                            replay_cache_active = True
                    cursor = 0
                    while cursor < len(ranked_candidates):
                        batch: list[tuple[str, str]] = []
                        batch_trials: list[dict[str, str]] = []
                        next_positions: list[int] = []
                        scan = cursor
                        while (
                            scan < len(ranked_candidates)
                            and len(batch) < max(int(max_lanes_per_batch), 1)
                        ):
                            _predicted_delta, name, candidate_fmt = ranked_candidates[scan]
                            scan += 1
                            if candidate_fmt == fr.canonical_format_name(
                                assignment.get(name, "")
                            ):
                                continue
                            trial = _candidate_trial_assignment(
                                assignment,
                                name,
                                candidate_fmt,
                                profile=profile,
                            )
                            if not _within_assignment_budget(
                                stats,
                                trial,
                                specs,
                                target_bpp,
                                bit_precision,
                            ):
                                continue
                            batch.append((name, candidate_fmt))
                            batch_trials.append(trial)
                            next_positions.append(scan)

                        if not batch:
                            cursor = scan
                            continue

                        pass_batch_index += 1
                        display_batch_total = pass_batch_index
                        batch_start_kl = float(current_kl)
                        needs_group_serial = any(
                            len(_candidate_override(
                                assignment,
                                name,
                                candidate_fmt,
                                profile=profile,
                            )) > 1
                            for name, candidate_fmt in batch
                        )
                        if needs_group_serial:
                            batch_kls = [
                                measure_assignment_kl(
                                    model,
                                    trial,
                                    calib_ids,
                                    ref_log_probs,
                                    work_root=work_root,
                                    profile=profile,
                                )
                                for trial in batch_trials
                            ]
                        else:
                            batch_kls = measure_lane_batched_kl_deltas(
                                model,
                                assignment,
                                batch,
                                calib_ids,
                                ref_log_probs,
                                work_root=work_root,
                                max_lanes_per_batch=max_lanes_per_batch,
                                profile=profile,
                                replay_cache=replay_cache,
                            )
                        batch_results = [
                            (name, candidate_fmt, trial, float(trial_kl))
                            for (name, candidate_fmt), trial, trial_kl
                            in zip(batch, batch_trials, batch_kls)
                        ]
                        best_batch_name, best_batch_fmt, best_batch_kl = min(
                            (
                                (name, candidate_fmt, trial_kl)
                                for name, candidate_fmt, _trial, trial_kl
                                in batch_results
                            ),
                            key=lambda item: item[2],
                        )
                        best_batch_delta = best_batch_kl - batch_start_kl
                        committed_in_batch = False
                        for idx, (name, candidate_fmt, trial, trial_kl) in enumerate(
                            batch_results
                        ):
                            measurements += 1
                            if trial_kl < current_kl - 1e-12:
                                old_fmt = assignment.get(name, "")
                                old_kl = float(current_kl)
                                assignment = dict(trial)
                                current_kl = trial_kl
                                flips_committed += 1
                                failed_streak = 0
                                pass_improved = True
                                committed_in_batch = True
                                cursor = next_positions[idx]
                                if emit is not None:
                                    display_batch_total = (
                                        pass_batch_index
                                        + _coord_descent_lane_batch_count(
                                            ranked_candidates,
                                            assignment,
                                            stats,
                                            specs,
                                            target_bpp,
                                            bit_precision,
                                            max_lanes_per_batch,
                                            start_index=cursor,
                                            profile=profile,
                                        )
                                    )
                                    emit(
                                        f"[coord] {anchor_text} COMMIT: "
                                        f"{name}.{old_fmt} -> {candidate_fmt}, "
                                        f"kl {old_kl:.4e} -> {current_kl:.4e} "
                                        f"(delta={current_kl - old_kl:+.4e}), "
                                        f"pass {pass_index} batch {pass_batch_index}/"
                                        f"{display_batch_total}"
                                    )
                                if replay_cache is not None:
                                    replay_cache.invalidate()
                                    if any(
                                        layer_depth(candidate_name) is not None
                                        for _delta, candidate_name, _fmt in ranked_candidates[
                                            next_positions[idx]:
                                        ]
                                    ):
                                        rebuilt = _populate_coord_replay_cache(
                                            model,
                                            assignment,
                                            calib_ids,
                                        )
                                        replay_cache = rebuilt
                                        if replay_cache is not None:
                                            replay_cache_populates += 1
                                            replay_cache_active = True
                                break
                            failed_streak += 1
                            if (
                                int(early_stop_streak) > 0
                                and failed_streak >= int(early_stop_streak)
                            ):
                                halted = "streak"
                                break
                        if emit is not None:
                            if halted == "streak":
                                display_batch_total = pass_batch_index
                            elif not committed_in_batch:
                                display_batch_total = (
                                    pass_batch_index
                                    + _coord_descent_lane_batch_count(
                                        ranked_candidates,
                                        assignment,
                                        stats,
                                        specs,
                                        target_bpp,
                                        bit_precision,
                                        max_lanes_per_batch,
                                        start_index=scan,
                                        profile=profile,
                                    )
                                )
                            emit(
                                f"[coord] {anchor_text} pass {pass_index} "
                                f"batch {pass_batch_index}/{display_batch_total}: "
                                f"elapsed={time.monotonic() - pass_start:.0f}s "
                                f"n_lanes={len(batch)} "
                                f"best_in_batch={best_batch_name}->{best_batch_fmt} "
                                f"delta={best_batch_delta:+.3e} "
                                f"cumul_accepted={flips_committed - pass_flips_start} "
                                f"cumul_best_kl={current_kl:.4e}"
                            )
                        if halted == "streak":
                            break
                        if not committed_in_batch:
                            cursor = scan
                    if halted == "streak":
                        _emit_pass_done()
                        break
                else:
                    for _predicted_delta, name, candidate_fmt in ranked_candidates:
                        if candidate_fmt == fr.canonical_format_name(assignment.get(name, "")):
                            continue
                        trial = _candidate_trial_assignment(
                            assignment,
                            name,
                            candidate_fmt,
                            profile=profile,
                        )
                        if not _within_assignment_budget(
                            stats,
                            trial,
                            specs,
                            target_bpp,
                            bit_precision,
                        ):
                            continue
                        override = _candidate_override(
                            assignment,
                            name,
                            candidate_fmt,
                            profile=profile,
                        )
                        if (
                            hooks is not None
                            and len(override) == 1
                            and name in cached_names
                        ):
                            with hooks.override({name: candidate_fmt}):
                                trial_kl = measure_assignment_kl(
                                    model,
                                    trial,
                                    calib_ids,
                                    ref_log_probs,
                                    work_root=work_root,
                                    profile=profile,
                                    perturbed_cache=hooks,
                                )
                        else:
                            trial_kl = measure_assignment_kl(
                                model,
                                trial,
                                calib_ids,
                                ref_log_probs,
                                work_root=work_root,
                                profile=profile,
                            )
                        measurements += 1
                        if float(trial_kl) < current_kl - 1e-12:
                            assignment = dict(trial)
                            current_kl = float(trial_kl)
                            if (
                                hooks is not None
                                and len(override) == 1
                                and name in cached_names
                            ):
                                hooks.set_frozen_weight_format(name, candidate_fmt)
                            flips_committed += 1
                            failed_streak = 0
                            pass_improved = True
                        else:
                            failed_streak += 1
                            if (
                                int(early_stop_streak) > 0
                                and failed_streak >= int(early_stop_streak)
                            ):
                                halted = "streak"
                                break

                if halted == "streak":
                    _emit_pass_done()
                    break
                if not pass_improved:
                    halted = "no_improvement"
                    _emit_pass_done()
                    break
                pass_done_reason = "max_passes" if pass_index >= pass_limit else "continue"
                _emit_pass_done(pass_done_reason)
            else:
                if int(max_passes) <= 0:
                    halted = "max_passes"
        finally:
            if hooks is not None and hooks.installed:
                hooks.remove()

    coherent_assignment = _enforce_fused_assignment_coherence(
        assignment,
        specs,
        profile=profile,
    )
    if assignment_hash(coherent_assignment) != assignment_hash(assignment):
        assignment = coherent_assignment
        current_kl = float(
            measure_assignment_kl(
                model,
                assignment,
                calib_ids,
                ref_log_probs,
                work_root=work_root,
                profile=profile,
            )
        )
    else:
        assignment = coherent_assignment
    meta = {
        "passes_completed": passes_completed,
        "flips_committed": flips_committed,
        "measurements": measurements,
        "fired": bool(max_passes > 0),
        "halted": halted,
        "early_stop_streak": int(early_stop_streak),
        "failed_streak": int(failed_streak),
        "lane_batched": bool(lane_batch_enabled),
        "replay_cache_enabled": bool(replay_cache_enabled),
        "replay_cache_active": bool(replay_cache_active),
        "replay_cache_populates": int(replay_cache_populates),
        "max_lanes_per_batch": int(max_lanes_per_batch),
    }
    if emit is not None:
        emit(
            f"[coord] {anchor_text}: tried {measurements} flips, "
            f"accepted {flips_committed}, halted ({halted})"
        )
    if return_metadata:
        return assignment, current_kl, meta
    return assignment, current_kl


def run_iterated_l3_polish(
    args,
    model,
    initial_assignment: Mapping[str, str],
    initial_kl: float,
    stats: dict,
    l2_costs: dict,
    specs: list[fr.FormatSpec],
    target_bpp: float,
    calib_ids: torch.Tensor,
    l3_calib_ids: torch.Tensor,
    ref_log_probs,
    *,
    work_root: Path,
    profile=None,
    emit: Callable[[str], None] = _emit,
    prefix: str = "[l3]",
    measure_all_formats: bool = False,
    initial_selected: list[L3NeighborhoodEntry] | None = None,
    initial_l3_costs: dict | None = None,
    resume_l3_cost_history: list[dict] | None = None,
    coord_anchor_label: str | None = None,
    scout_checkpoint_callback: Callable[[Mapping[str, str], float, dict], None] | None = None,
) -> L3PolishRun:
    """Generate L3 candidates with additive costs, then gate by measured KL.

    Propagated L3 costs are a surrogate for fast discrete search; they do not
    claim joint optimality. Every proposed assignment is measured on the KL
    calibration slice and accepted only when it is non-regressive under that
    empirical objective. Validation scout and coordinate descent use the same
    measured-KL gate for local corrections around the surrogate proposal.
    """
    l3_assignment = dict(initial_assignment)
    current_kl = float(initial_kl)
    proposed_assignment = dict(l3_assignment)
    proposed_kl = float(current_kl)
    l3_cost_history: list[dict] = []
    selected_history: list[list[L3NeighborhoodEntry]] = []
    iteration_records: list[dict] = []
    seen_hashes = {assignment_hash(l3_assignment)}
    hamming_cap = max(0, int(getattr(args, "l3_hamming_cap_init", 8)))
    hamming_cap_max = max(hamming_cap, int(getattr(args, "l3_hamming_cap_max", 64)))
    max_iters = max(1, int(getattr(args, "l3_iter_max", 3)))
    latest_smoothed_l3: dict = {}
    latest_l3_costs: dict = {}
    latest_selected: list[L3NeighborhoodEntry] = []
    frozen_dp_meta: dict = {"frozen_dp_precision_used": "none"}
    latest_interaction_meta: dict = {"enabled": False, "attempted": False}
    measurement_seconds = 0.0
    dp_seconds = 0.0
    validation_seconds = 0.0
    measurement_peak: float | None = None
    dp_peak: float | None = None
    validation_peak: float | None = None
    cycle_detected = False
    termination = "max_iters"

    for iteration in range(1, max_iters + 1):
        if iteration == 1 and initial_selected is not None:
            selected = list(initial_selected)
        else:
            selected = _select_l3_neighborhood_for_assignment(
                args,
                stats,
                l2_costs,
                l3_assignment,
                specs,
                measure_all_formats=measure_all_formats,
            )
        latest_selected = list(selected)
        selected_history.append(list(selected))
        if not selected:
            termination = "empty_neighborhood"
            break

        if resume_l3_cost_history is not None:
            if iteration > len(resume_l3_cost_history):
                termination = "resume_costs_exhausted"
                break
            l3_costs_iter = resume_l3_cost_history[iteration - 1]
            measure_elapsed = 0.0
            measure_peak = None
            emit(
                f"{prefix} iteration {iteration}: using resumed L3 costs; "
                f"skipping measurement"
            )
        elif iteration == 1 and initial_l3_costs is not None:
            l3_costs_iter = initial_l3_costs
            measure_elapsed = 0.0
            measure_peak = None
        else:
            lane_count = sum(
                len([fmt for fmt in entry.formats if fmt != "BF16"]) + 1
                for entry in selected
            )
            emit(
                f"{prefix} iteration {iteration}: measuring "
                f"{len(selected)} Linears x {lane_count} lanes "
                f"cap={hamming_cap}"
            )
            _cuda_reset_peak()
            measure_start = time.monotonic()
            l3_costs_iter = measure_propagated_costs(
                model,
                l3_assignment,
                selected,
                l3_calib_ids,
                specs,
                work_root=work_root,
                profile=profile,
                max_lanes_per_batch=getattr(args, "l3_max_lanes_per_batch", 16),
                tail_only=getattr(args, "l3_tail_only", True),
                output_mse_names=_l3_output_mse_names(args),
                progress_callback=_make_l3_progress(emit, prefix=prefix),
            )
            measure_elapsed = time.monotonic() - measure_start
            measure_peak = _cuda_peak_gb()
            suffix = (
                f", cuda_peak={measure_peak:.2f}GB"
                if measure_peak is not None else ""
            )
            emit(
                f"{prefix} iteration {iteration}: measurement done in "
                f"{measure_elapsed:.1f}s{suffix}"
            )
        _report_graph_memory(f"{prefix} iteration {iteration} after_l3_measurement")
        latest_l3_costs = l3_costs_iter
        l3_cost_history.append(l3_costs_iter)
        latest_smoothed_l3 = smooth_cost_history(
            l3_cost_history,
            decay=getattr(args, "ema_decay", 0.5),
        )
        measurement_seconds += float(measure_elapsed)
        if measure_peak is not None:
            measurement_peak = (
                measure_peak
                if measurement_peak is None else max(measurement_peak, measure_peak)
            )

        l3_candidates = _filter_l3_candidates_for_assignment(
            build_l3_candidates(stats, latest_smoothed_l3, specs),
            l3_assignment,
        )
        _cuda_reset_peak()
        dp_start = time.monotonic()
        candidate_assignment, _chosen, frozen_dp_meta = (
            _solve_l3_candidates_with_hamming_cap(
                stats,
                l3_assignment,
                l3_candidates,
                specs,
                target_bits=target_bpp,
                bit_precision=getattr(args, "bit_precision", 0.001),
                max_flips=hamming_cap,
                budget_tolerance=getattr(args, "frozen_dp_budget_tolerance", 0.0),
                profile=profile,
            )
        )
        additive_candidate_assignment = dict(candidate_assignment)
        candidate_assignment, interaction_meta = run_l3_interaction_refine(
            args,
            model,
            l3_assignment,
            candidate_assignment,
            l3_candidates,
            stats,
            specs,
            target_bpp,
            l3_calib_ids,
            work_root=work_root,
            profile=profile,
            emit=emit,
            prefix=prefix,
        )
        latest_interaction_meta = dict(interaction_meta)
        interaction_changed = (
            assignment_hash(candidate_assignment)
            != assignment_hash(additive_candidate_assignment)
        )
        dp_elapsed = time.monotonic() - dp_start
        dp_peak_iter = _cuda_peak_gb()
        dp_seconds += float(dp_elapsed)
        if dp_peak_iter is not None:
            dp_peak = dp_peak_iter if dp_peak is None else max(dp_peak, dp_peak_iter)

        _report_graph_memory(f"{prefix} iteration {iteration} before_validation")
        validation_start = time.monotonic()
        candidate_kl = measure_assignment_kl(
            model,
            candidate_assignment,
            calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
        )
        if interaction_changed:
            additive_kl = measure_assignment_kl(
                model,
                additive_candidate_assignment,
                calib_ids,
                ref_log_probs,
                work_root=work_root,
                profile=profile,
            )
            latest_interaction_meta["additive_candidate_kl"] = float(additive_kl)
            latest_interaction_meta["interaction_candidate_kl"] = float(candidate_kl)
            if float(additive_kl) + 1e-12 < float(candidate_kl):
                emit(
                    f"{prefix} interaction refine: validation rollback to "
                    f"additive KL={float(additive_kl):.4e} "
                    f"interaction_KL={float(candidate_kl):.4e}"
                )
                candidate_assignment = dict(additive_candidate_assignment)
                candidate_kl = float(additive_kl)
                latest_interaction_meta["validation_rollback_to_additive"] = True
            else:
                latest_interaction_meta["validation_rollback_to_additive"] = False
        validation_elapsed = time.monotonic() - validation_start
        validation_peak_iter = _cuda_peak_gb()
        _report_graph_memory(f"{prefix} iteration {iteration} after_validation")
        validation_seconds += float(validation_elapsed)
        if validation_peak_iter is not None:
            validation_peak = (
                validation_peak_iter
                if validation_peak is None else max(validation_peak, validation_peak_iter)
            )
        candidate_kl = float(candidate_kl)
        candidate_hamming = _hamming_count(l3_assignment, candidate_assignment)
        proposed_assignment = dict(candidate_assignment)
        proposed_kl = candidate_kl
        allowed_regression = (
            max(float(getattr(args, "l3_regression_tolerance", 0.0)), 0.0)
            * abs(current_kl)
        )
        accepted_step = bool(candidate_kl <= current_kl + allowed_regression + 1e-12)
        candidate_hash = assignment_hash(candidate_assignment)
        record = {
            "iteration": iteration,
            "hamming_cap": int(hamming_cap),
            "hamming": int(candidate_hamming),
            "kl_before": float(current_kl),
            "candidate_kl": float(candidate_kl),
            "accepted": accepted_step,
            "assignment_hash": list(candidate_hash),
            "cycle": False,
            "selected_count": len(selected),
            "interaction_refine": latest_interaction_meta,
        }
        iteration_records.append(record)
        emit(
            f"{prefix} iteration {iteration}: candidate_hamming="
            f"{candidate_hamming} KL_before={current_kl:.4e} "
            f"KL_candidate={candidate_kl:.4e} "
            f"accepted={str(accepted_step).lower()}"
        )
        if not accepted_step:
            termination = "rollback"
            break
        if candidate_hamming == 0:
            current_kl = float(candidate_kl)
            l3_assignment = dict(candidate_assignment)
            termination = "fixed_point"
            break
        if candidate_hash in seen_hashes:
            record["cycle"] = True
            cycle_detected = True
            termination = "cycle"
            break
        l3_assignment = dict(candidate_assignment)
        current_kl = float(candidate_kl)
        seen_hashes.add(candidate_hash)
        hamming_cap = min(
            max(hamming_cap * 2, hamming_cap + 1),
            hamming_cap_max,
            max(len(selected), 1),
        )
    else:
        termination = "max_iters"

    _phase_boundary_cleanup(f"{prefix} l3_to_coord")
    coord_meta = {
        "fired": False,
        "flips_committed": 0,
        "passes_completed": 0,
        "measurements": 0,
        "halted": "disabled",
        "early_stop_streak": int(getattr(args, "coord_descent_early_stop_streak", 50)),
        "failed_streak": 0,
        "lane_batched": _coord_lane_batch_enabled(),
        "max_lanes_per_batch": int(getattr(args, "l3_max_lanes_per_batch", 64)),
    }
    scout_meta = {
        "enabled": False,
        "rounds_completed": 0,
        "candidates_evaluated": 0,
        "hits": 0,
        "flips_committed": 0,
        "elapsed_seconds": 0.0,
        "stopped_reason": "disabled",
    }
    if _l3_validation_scout_enabled(args):
        scout_start = time.monotonic()
        l3_assignment, current_kl, scout_meta = run_l3_validation_scout(
            args,
            model,
            l3_assignment,
            current_kl,
            latest_smoothed_l3 or latest_l3_costs,
            specs,
            target_bpp,
            calib_ids,
            ref_log_probs,
            stats=stats,
            work_root=work_root,
            profile=profile,
            emit=emit,
            prefix=prefix,
            checkpoint_callback=scout_checkpoint_callback,
        )
        validation_seconds += time.monotonic() - scout_start
        if scout_meta.get("flips_committed"):
            proposed_assignment = dict(l3_assignment)
            proposed_kl = float(current_kl)

    if bool(getattr(args, "l3_coord_descent_fallback", True)):
        coord_passes = max(0, int(getattr(args, "l3_coord_descent_max_passes", 1)))
        coord_start = time.monotonic()
        l3_assignment, current_kl, coord_meta = coordinate_descent_polish(
            model,
            l3_assignment,
            latest_smoothed_l3 or latest_l3_costs,
            specs,
            target_bpp,
            calib_ids,
            ref_log_probs,
            stats=stats,
            work_root=work_root,
            profile=profile,
            bit_precision=getattr(args, "bit_precision", 0.001),
            max_passes=coord_passes,
            early_stop_streak=getattr(args, "coord_descent_early_stop_streak", 50),
            current_kl=current_kl,
            return_metadata=True,
            emit=emit,
            anchor_label=coord_anchor_label,
            max_lanes_per_batch=getattr(args, "l3_max_lanes_per_batch", 64),
        )
        coord_elapsed = time.monotonic() - coord_start
        _report_graph_memory(f"{prefix} after_coord_descent")
        validation_seconds += float(coord_elapsed)
        if coord_meta["flips_committed"]:
            proposed_assignment = dict(l3_assignment)
            proposed_kl = float(current_kl)
        emit(
            f"{prefix} coordinate descent: passes="
            f"{coord_meta['passes_completed']} flips="
            f"{coord_meta['flips_committed']} tried="
            f"{coord_meta['measurements']} halted={coord_meta.get('halted')} "
            f"KL={current_kl:.4e}"
        )

    coherent_l3_assignment = _enforce_fused_assignment_coherence(
        l3_assignment,
        specs,
        profile=profile,
    )
    if assignment_hash(coherent_l3_assignment) != assignment_hash(l3_assignment):
        l3_assignment = coherent_l3_assignment
        current_kl = float(
            measure_assignment_kl(
                model,
                l3_assignment,
                calib_ids,
                ref_log_probs,
                work_root=work_root,
                profile=profile,
            )
        )
        proposed_assignment = dict(l3_assignment)
        proposed_kl = float(current_kl)

    final_changed = assignment_hash(l3_assignment) != assignment_hash(initial_assignment)
    allowed_final_regression = (
        max(float(getattr(args, "l3_regression_tolerance", 0.0)), 0.0)
        * abs(float(initial_kl))
    )
    accepted = bool(
        final_changed
        and float(current_kl) <= float(initial_kl) + allowed_final_regression + 1e-12
    )
    regression = bool(float(proposed_kl) > float(initial_kl))
    measurement_timing = {
        "elapsed_seconds": measurement_seconds,
        "cuda_peak_gb": measurement_peak,
    }
    dp_timing = {"elapsed_seconds": dp_seconds, "cuda_peak_gb": dp_peak}
    validation_timing = {
        "elapsed_seconds": validation_seconds,
        "cuda_peak_gb": validation_peak,
    }
    if iteration_records:
        iteration_records[-1]["termination"] = termination
    return L3PolishRun(
        initial_assignment=dict(initial_assignment),
        assignment=dict(l3_assignment),
        proposed_assignment=dict(proposed_assignment),
        initial_kl=float(initial_kl),
        final_kl=float(current_kl),
        proposed_kl=float(proposed_kl),
        accepted=accepted,
        regression=regression,
        iterations=iteration_records,
        selected=list(latest_selected),
        selected_history=selected_history,
        l3_cost_history=l3_cost_history,
        l3_costs=latest_l3_costs,
        smoothed_l3_costs=latest_smoothed_l3 or latest_l3_costs,
        frozen_dp_meta=frozen_dp_meta,
        measurement_timing=measurement_timing,
        dp_timing=dp_timing,
        validation_timing=validation_timing,
        coord_descent_fired=bool(coord_meta["fired"]),
        coord_descent_flips=int(coord_meta["flips_committed"]),
        coord_descent_passes=int(coord_meta["passes_completed"]),
        cycle_detected=cycle_detected,
        validation_scout=scout_meta,
    )


DEFAULT_KNEE_KL_NOISE_FLOOR = 1e-5


def _knee_kl_noise_floor(args=None) -> float:
    return max(
        float(
            getattr(
                args,
                "knee_kl_noise_floor",
                DEFAULT_KNEE_KL_NOISE_FLOOR,
            )
        ),
        0.0,
    )


def _normalise(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def segmented_kneedle_point(points: list[tuple[float, float]]) -> tuple[float, float, float, bool]:
    pts = sorted(points)
    if len(pts) < 3:
        mid = pts[len(pts) // 2]
        return mid[0], mid[1], 0.0, True
    xs = _normalise([p[0] for p in pts])
    ys_raw = _normalise([p[1] for p in pts])
    ys = [1.0 - y for y in ys_raw]
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    denom = max(((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5, 1e-12)
    scores = []
    for idx, (x, y) in enumerate(zip(xs, ys)):
        score = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        scores.append((score, idx))
    score, idx = max(scores)
    endpoint = idx in {0, len(pts) - 1}
    if endpoint:
        idx = len(pts) // 2
    return pts[idx][0], pts[idx][1], float(score), endpoint


def _largest_adjacent_interval(points: list[float], center: float) -> tuple[float, float] | None:
    ordered = sorted(points)
    if len(ordered) < 2:
        return None
    center_idx = min(range(len(ordered)), key=lambda i: abs(ordered[i] - center))
    intervals = []
    if center_idx > 0:
        intervals.append((ordered[center_idx] - ordered[center_idx - 1], ordered[center_idx - 1], ordered[center_idx]))
    if center_idx + 1 < len(ordered):
        intervals.append((ordered[center_idx + 1] - ordered[center_idx], ordered[center_idx], ordered[center_idx + 1]))
    if not intervals:
        return None
    _width, left, right = max(intervals, key=lambda item: item[0])
    return left, right


def adaptive_segmented_kneedle(
    evaluator: Callable[[float], float],
    bpp_min: float,
    bpp_max: float,
    *,
    tolerance: float,
    max_evaluations: int,
) -> tuple[float, list[tuple[float, float]], dict]:
    evaluated: dict[float, float] = {}

    def eval_once(x: float) -> None:
        key = round(float(x), 8)
        if key not in evaluated:
            evaluated[key] = float(evaluator(float(x)))

    midpoint = (float(bpp_min) + float(bpp_max)) / 2.0
    for point in (float(bpp_min), midpoint, float(bpp_max)):
        eval_once(point)
    while len(evaluated) < int(max_evaluations):
        points = sorted(evaluated.items())
        knee_bpp, _kl, score, endpoint = segmented_kneedle_point(points)
        interval = _largest_adjacent_interval(list(evaluated), knee_bpp)
        if interval is None or interval[1] - interval[0] <= float(tolerance):
            break
        eval_once((interval[0] + interval[1]) / 2.0)
    points = sorted(evaluated.items())
    knee_bpp, knee_kl, score, endpoint = segmented_kneedle_point(points)
    return knee_bpp, points, {
        "mode": "kneedle",
        "validation_kl": knee_kl,
        "kneedle_score": score,
        "endpoint_fallback": endpoint,
    }


def threshold_knee_search(
    evaluator: Callable[[float], float],
    bpp_min: float,
    bpp_max: float,
    *,
    threshold_kl: float,
    tolerance: float,
    max_evaluations: int,
) -> tuple[float, list[tuple[float, float]], dict]:
    evaluated: dict[float, float] = {}

    def eval_once(x: float) -> float:
        key = round(float(x), 8)
        if key not in evaluated:
            evaluated[key] = float(evaluator(float(x)))
        return evaluated[key]

    high_kl = eval_once(float(bpp_max))
    if high_kl > float(threshold_kl):
        raise RuntimeError("no feasible bpp meets threshold")
    low = float(bpp_min)
    high = float(bpp_max)
    eval_once(low)
    while high - low > float(tolerance) and len(evaluated) < int(max_evaluations):
        mid = (low + high) / 2.0
        mid_kl = eval_once(mid)
        if mid_kl <= float(threshold_kl):
            high = mid
        else:
            low = mid
    return high, sorted(evaluated.items()), {
        "mode": "threshold",
        "threshold_kl": float(threshold_kl),
        "validation_kl": evaluated[round(high, 8)],
    }


def knee_candidate_from_points(points: list[tuple[float, float]]) -> dict | None:
    if not points:
        return None
    knee_bpp, knee_kl, score, endpoint = segmented_kneedle_point(points)
    return {
        "mode": "kneedle_on_evaluated_points",
        "chosen_bpp": float(knee_bpp),
        "validation_kl": float(knee_kl),
        "kneedle_score": float(score),
        "endpoint_fallback": bool(endpoint),
    }


def kneedle_leave_one_out_diagnostic(
    points: list[tuple[float, float]],
    chosen_bpp: float,
    chosen_kl: float,
    *,
    tolerance_bpp: float,
    kl_noise_floor: float = DEFAULT_KNEE_KL_NOISE_FLOOR,
) -> dict:
    pts = sorted((float(bpp), float(kl)) for bpp, kl in points)
    tolerance_bpp = max(float(tolerance_bpp), 0.0)
    kl_noise_floor = max(float(kl_noise_floor), 0.0)
    if len(pts) < 4:
        return {
            "enabled": False,
            "stable": True,
            "reason": "requires_at_least_4_points",
            "points": len(pts),
            "max_bpp_shift": 0.0,
            "max_kl_shift": 0.0,
            "tolerance_bpp": float(tolerance_bpp),
            "kl_noise_floor": float(kl_noise_floor),
            "trials": [],
        }

    trials = []
    for idx, removed in enumerate(pts):
        subset = pts[:idx] + pts[idx + 1:]
        loo_bpp, loo_kl, loo_score, loo_endpoint = segmented_kneedle_point(subset)
        trials.append(
            {
                "removed_bpp": float(removed[0]),
                "removed_kl": float(removed[1]),
                "chosen_bpp": float(loo_bpp),
                "chosen_kl": float(loo_kl),
                "bpp_shift": abs(float(loo_bpp) - float(chosen_bpp)),
                "kl_shift": abs(float(loo_kl) - float(chosen_kl)),
                "kneedle_score": float(loo_score),
                "endpoint_fallback": bool(loo_endpoint),
            }
        )
    max_bpp_shift = max(float(row["bpp_shift"]) for row in trials)
    max_kl_shift = max(float(row["kl_shift"]) for row in trials)
    stable = bool(
        max_bpp_shift <= tolerance_bpp + 1e-12
        and max_kl_shift <= kl_noise_floor + 1e-12
    )
    return {
        "enabled": True,
        "stable": stable,
        "points": len(pts),
        "max_bpp_shift": float(max_bpp_shift),
        "max_kl_shift": float(max_kl_shift),
        "tolerance_bpp": float(tolerance_bpp),
        "kl_noise_floor": float(kl_noise_floor),
        "trials": trials,
    }


def _warn_kneedle_instability(context: str, metadata: Mapping | None) -> None:
    if not isinstance(metadata, Mapping):
        return
    diagnostic = metadata.get("leave_one_out")
    if not isinstance(diagnostic, Mapping):
        return
    if diagnostic.get("enabled") and not diagnostic.get("stable", True):
        _emit(
            f"[{context}] warning: leave-one-out kneedle unstable "
            f"(max_bpp_shift={float(diagnostic.get('max_bpp_shift', 0.0)):.4g}, "
            f"max_kl_shift={float(diagnostic.get('max_kl_shift', 0.0)):.4g})"
        )
        guard = metadata.get("selection_guard")
        if isinstance(guard, Mapping) and guard.get("applied"):
            _emit(
                f"[{context}] warning: selected best measured-KL frontier point "
                f"target={float(guard.get('guarded_chosen_target_bpp', 0.0)):.4g} "
                f"KL={float(guard.get('guarded_chosen_validation_kl', 0.0)):.4g} "
                "because the geometric knee was unstable"
            )


def _linspace_bpp(lo: float, hi: float, count: int) -> list[float]:
    count = max(int(count), 1)
    if count == 1:
        return [round((float(lo) + float(hi)) / 2.0, 8)]
    return [
        round(float(lo) + (float(hi) - float(lo)) * idx / float(count - 1), 8)
        for idx in range(count)
    ]


def _nondominated_budget_results(
    results: list[BudgetResult],
    *,
    kl_noise_floor: float = DEFAULT_KNEE_KL_NOISE_FLOOR,
) -> list[BudgetResult]:
    ordered = sorted(
        results,
        key=lambda result: (
            float(result.achieved_bpp),
            float(result.validation_kl),
            float(result.target_bpp),
        ),
    )
    frontier: list[BudgetResult] = []
    best_kl = float("inf")
    kl_noise_floor = max(float(kl_noise_floor), 0.0)
    for result in ordered:
        kl = float(result.validation_kl)
        if kl < best_kl - kl_noise_floor - 1e-12:
            frontier.append(result)
            best_kl = kl
    return frontier


def _result_attr(result: BudgetResult, attr: str) -> float:
    return float(getattr(result, attr))


def _bracketing_result_pair(
    results: list[BudgetResult],
    value: float,
    *,
    attr: str,
) -> tuple[BudgetResult, BudgetResult] | None:
    ordered = sorted(results, key=lambda result: (_result_attr(result, attr), result.target_bpp))
    if len(ordered) < 2:
        return None
    value = float(value)
    for left, right in zip(ordered, ordered[1:]):
        if (
            _result_attr(left, attr) <= value + 1e-12
            and value <= _result_attr(right, attr) + 1e-12
        ):
            return left, right
    if value < _result_attr(ordered[0], attr):
        return ordered[0], ordered[1]
    return ordered[-2], ordered[-1]


def _neighbor_result_pairs(
    results: list[BudgetResult],
    center: BudgetResult,
    *,
    attr: str = "target_bpp",
) -> list[tuple[BudgetResult, BudgetResult]]:
    ordered = sorted(results, key=lambda result: (_result_attr(result, attr), result.achieved_bpp))
    center_digest = _assignment_digest(center.assignment)
    center_index = None
    for idx, result in enumerate(ordered):
        if result is center or (
            round(result.target_bpp, 8) == round(center.target_bpp, 8)
            and _assignment_digest(result.assignment) == center_digest
        ):
            center_index = idx
            break
    if center_index is None:
        return []
    pairs: list[tuple[BudgetResult, BudgetResult]] = []
    if center_index > 0:
        pairs.append((ordered[center_index - 1], ordered[center_index]))
    if center_index + 1 < len(ordered):
        pairs.append((ordered[center_index], ordered[center_index + 1]))
    return pairs


def _result_pair_width(
    pair: tuple[BudgetResult, BudgetResult] | None,
    *,
    attr: str,
) -> float | None:
    if pair is None:
        return None
    return abs(_result_attr(pair[1], attr) - _result_attr(pair[0], attr))


def _next_target_between_results(
    pair: tuple[BudgetResult, BudgetResult] | None,
    evaluated_targets: set[float],
    *,
    bpp_min: float,
    bpp_max: float,
) -> float | None:
    if pair is None:
        return None
    target_interval = (
        float(pair[0].target_bpp),
        float(pair[1].target_bpp),
    )
    candidate = _next_midpoint_target(
        target_interval,
        evaluated_targets,
        bpp_min=bpp_min,
        bpp_max=bpp_max,
    )
    if candidate is not None:
        return candidate
    achieved_interval = (
        float(pair[0].achieved_bpp),
        float(pair[1].achieved_bpp),
    )
    return _next_midpoint_target(
        achieved_interval,
        evaluated_targets,
        bpp_min=bpp_min,
        bpp_max=bpp_max,
    )


def _widest_pair(
    pairs: list[tuple[BudgetResult, BudgetResult]],
) -> tuple[BudgetResult, BudgetResult] | None:
    if not pairs:
        return None
    return max(
        pairs,
        key=lambda pair: (
            _result_pair_width(pair, attr="achieved_bpp") or 0.0,
            _result_pair_width(pair, attr="target_bpp") or 0.0,
        ),
    )


def _largest_adjacent_gap(values: list[float]) -> tuple[float, float] | None:
    ordered = sorted({round(float(value), 8) for value in values})
    if len(ordered) < 2:
        return None
    left, right = max(
        zip(ordered, ordered[1:]),
        key=lambda pair: (pair[1] - pair[0], -pair[0]),
    )
    return float(left), float(right)


def _next_midpoint_target(
    interval: tuple[float, float] | None,
    evaluated_targets: set[float],
    *,
    bpp_min: float,
    bpp_max: float,
) -> float | None:
    if interval is None:
        return None
    left, right = sorted((float(interval[0]), float(interval[1])))
    left = max(float(bpp_min), left)
    right = min(float(bpp_max), right)
    if right - left <= 1e-12:
        return None
    fractions = (0.5, 1.0 / 3.0, 2.0 / 3.0, 0.25, 0.75)
    for frac in fractions:
        candidate = round(left + (right - left) * frac, 8)
        if candidate <= float(bpp_min) - 1e-12:
            continue
        if candidate >= float(bpp_max) + 1e-12:
            continue
        if candidate not in evaluated_targets:
            return candidate
    return None


def adaptive_validated_frontier_kneedle(
    evaluator: Callable[[float], BudgetResult],
    bpp_min: float,
    bpp_max: float,
    *,
    tolerance: float,
    max_evaluations: int,
    initial_points: int = 9,
    initial_results: list[BudgetResult] | None = None,
    kl_noise_floor: float = DEFAULT_KNEE_KL_NOISE_FLOOR,
    unstable_policy: str = "best-kl",
) -> tuple[BudgetResult, list[BudgetResult], dict]:
    """Search the kneedle on the real-KL Pareto envelope.

    Target bpps are only probes. Selection and refinement use achieved bpp and
    validation KL, then discard empirically dominated points before applying
    kneedle geometry.
    """
    evaluated: dict[float, BudgetResult] = {}
    max_evaluations = max(int(max_evaluations), 3)
    initial_points = min(max(int(initial_points), 3), max_evaluations)
    bpp_min = float(bpp_min)
    bpp_max = float(bpp_max)
    tolerance = max(float(tolerance), 0.0)
    kl_noise_floor = max(float(kl_noise_floor), 0.0)
    unstable_policy = str(unstable_policy or "best-kl")
    seeded = 0
    for result in initial_results or []:
        key = round(float(result.target_bpp), 8)
        previous = evaluated.get(key)
        if previous is None or (
            float(result.validation_kl),
            float(result.achieved_bpp),
        ) < (
            float(previous.validation_kl),
            float(previous.achieved_bpp),
        ):
            evaluated[key] = result
        seeded += 1

    def eval_once(x: float) -> BudgetResult:
        target = min(max(float(x), bpp_min), bpp_max)
        key = round(target, 8)
        if key not in evaluated:
            evaluated[key] = evaluator(target)
        return evaluated[key]

    for point in _linspace_bpp(bpp_min, bpp_max, initial_points):
        if len(evaluated) >= max_evaluations:
            break
        eval_once(point)

    termination = "max_evaluations"
    knee_score = 0.0
    endpoint = True
    bracket_width: float | None = None
    target_bracket_width: float | None = None
    knee_bpp: float | None = None
    knee_kl: float | None = None
    chosen: BudgetResult | None = None

    while len(evaluated) < max_evaluations:
        results = sorted(evaluated.values(), key=lambda result: result.target_bpp)
        frontier = _nondominated_budget_results(
            results,
            kl_noise_floor=kl_noise_floor,
        )
        evaluated_targets = set(evaluated)
        next_target: float | None = None

        if len(frontier) >= 3:
            knee_bpp, knee_kl, knee_score, endpoint = segmented_kneedle_point(
                [
                    (float(result.achieved_bpp), float(result.validation_kl))
                    for result in frontier
                ]
            )
            chosen = min(
                frontier,
                key=lambda result: (
                    abs(float(result.achieved_bpp) - float(knee_bpp)),
                    abs(float(result.validation_kl) - float(knee_kl)),
                ),
            )
            bracket_pair = _bracketing_result_pair(
                frontier,
                float(knee_bpp),
                attr="achieved_bpp",
            )
            if endpoint:
                bracket_pair = _widest_pair(
                    _neighbor_result_pairs(frontier, chosen, attr="achieved_bpp")
                    or _neighbor_result_pairs(results, chosen, attr="target_bpp")
                ) or bracket_pair
            bracket_width = _result_pair_width(bracket_pair, attr="achieved_bpp")
            target_bracket_width = _result_pair_width(bracket_pair, attr="target_bpp")
            if (
                bracket_pair is None
                or (
                    bracket_width is not None
                    and bracket_width <= tolerance + 1e-12
                    and (
                        target_bracket_width is None
                        or target_bracket_width <= tolerance + 1e-12
                        or len(evaluated) >= max_evaluations
                    )
                )
            ):
                termination = "tolerance"
                break
            next_target = _next_target_between_results(
                bracket_pair,
                evaluated_targets,
                bpp_min=bpp_min,
                bpp_max=bpp_max,
            )
        else:
            best = min(
                results,
                key=lambda result: (
                    float(result.validation_kl),
                    float(result.achieved_bpp),
                ),
            )
            interval = _largest_adjacent_interval(
                [float(result.target_bpp) for result in results],
                float(best.target_bpp),
            )
            if interval is None:
                interval = _largest_adjacent_gap(
                    [float(result.target_bpp) for result in results]
                )
            bracket_width = (
                None if interval is None else float(interval[1]) - float(interval[0])
            )
            if interval is None or bracket_width <= tolerance + 1e-12:
                termination = "too_few_frontier_points"
                break
            next_target = _next_midpoint_target(
                interval,
                evaluated_targets,
                bpp_min=bpp_min,
                bpp_max=bpp_max,
            )

        if next_target is None:
            interval = _largest_adjacent_gap(
                [float(result.target_bpp) for result in evaluated.values()]
            )
            next_target = _next_midpoint_target(
                interval,
                evaluated_targets,
                bpp_min=bpp_min,
                bpp_max=bpp_max,
            )
        if next_target is None:
            termination = "no_unevaluated_midpoint"
            break
        eval_once(next_target)

    results = sorted(evaluated.values(), key=lambda result: result.target_bpp)
    frontier = _nondominated_budget_results(
        results,
        kl_noise_floor=kl_noise_floor,
    )
    if len(frontier) >= 3:
        knee_bpp, knee_kl, knee_score, endpoint = segmented_kneedle_point(
            [
                (float(result.achieved_bpp), float(result.validation_kl))
                for result in frontier
            ]
        )
        chosen = min(
            frontier,
            key=lambda result: (
                abs(float(result.achieved_bpp) - float(knee_bpp)),
                abs(float(result.validation_kl) - float(knee_kl)),
            ),
        )
        bracket_pair = _bracketing_result_pair(
            frontier,
            float(knee_bpp),
            attr="achieved_bpp",
        )
        if endpoint:
            bracket_pair = _widest_pair(
                _neighbor_result_pairs(frontier, chosen, attr="achieved_bpp")
                or _neighbor_result_pairs(results, chosen, attr="target_bpp")
            ) or bracket_pair
        bracket_width = _result_pair_width(bracket_pair, attr="achieved_bpp")
        target_bracket_width = _result_pair_width(bracket_pair, attr="target_bpp")
    else:
        chosen = min(
            frontier or results,
            key=lambda result: (
                float(result.validation_kl),
                float(result.achieved_bpp),
            ),
        )
        knee_bpp = float(chosen.achieved_bpp)
        knee_kl = float(chosen.validation_kl)
        knee_score = 0.0
        endpoint = True
        bracket_pair = _widest_pair(
            _neighbor_result_pairs(results, chosen, attr="target_bpp")
        )
        bracket_width = _result_pair_width(bracket_pair, attr="achieved_bpp")
        target_bracket_width = _result_pair_width(bracket_pair, attr="target_bpp")

    geometric_chosen = chosen
    loo_diagnostic = kneedle_leave_one_out_diagnostic(
        [
            (float(result.achieved_bpp), float(result.validation_kl))
            for result in frontier
        ],
        float(geometric_chosen.achieved_bpp),
        float(geometric_chosen.validation_kl),
        tolerance_bpp=float(tolerance),
        kl_noise_floor=float(kl_noise_floor),
    )
    selection_guard = {
        "policy": unstable_policy,
        "applied": False,
        "reason": "stable_or_disabled",
    }
    if (
        unstable_policy == "best-kl"
        and loo_diagnostic.get("enabled")
        and not loo_diagnostic.get("stable", True)
        and frontier
    ):
        best_kl = min(
            frontier,
            key=lambda result: (
                float(result.validation_kl),
                float(result.achieved_bpp),
                float(result.target_bpp),
            ),
        )
        if best_kl is not geometric_chosen:
            chosen = best_kl
            selection_guard = {
                "policy": unstable_policy,
                "applied": True,
                "reason": "leave_one_out_unstable",
                "geometric_chosen_target_bpp": float(geometric_chosen.target_bpp),
                "geometric_chosen_achieved_bpp": float(geometric_chosen.achieved_bpp),
                "geometric_chosen_validation_kl": float(geometric_chosen.validation_kl),
                "guarded_chosen_target_bpp": float(chosen.target_bpp),
                "guarded_chosen_achieved_bpp": float(chosen.achieved_bpp),
                "guarded_chosen_validation_kl": float(chosen.validation_kl),
            }

    return chosen, results, {
        "mode": "validated_frontier_kneedle",
        "chosen_target_bpp": float(chosen.target_bpp),
        "chosen_achieved_bpp": float(chosen.achieved_bpp),
        "validation_kl": float(chosen.validation_kl),
        "kneedle_score": float(knee_score),
        "endpoint_fallback": bool(endpoint),
        "termination": termination,
        "bracket_width_bpp": (
            float(bracket_width) if bracket_width is not None else None
        ),
        "target_bracket_width_bpp": (
            float(target_bracket_width)
            if target_bracket_width is not None else None
        ),
        "knee_tolerance_bpp": float(tolerance),
        "knee_kl_noise_floor": float(kl_noise_floor),
        "leave_one_out": loo_diagnostic,
        "selection_guard": selection_guard,
        "initial_points": int(initial_points),
        "seeded_evaluations": int(seeded),
        "evaluations": len(results),
        "frontier_points": [result.as_record() for result in frontier],
        "evaluated_points": [result.as_record() for result in results],
        "quality_note": (
            "Kneedle is computed after filtering empirically dominated points "
            "by achieved bpp and measured validation KL."
        ),
    }


def _surrogate_grid_points(args) -> list[float]:
    lo = float(args.knee_bpp_min)
    hi = float(args.knee_bpp_max)
    step = getattr(args, "knee_surrogate_bpp_step", None)
    if step is not None and float(step) > 0:
        values = []
        x = lo
        while x <= hi + 1e-12:
            values.append(round(float(x), 8))
            x += float(step)
        if abs(values[-1] - hi) > 1e-8:
            values.append(round(hi, 8))
        return sorted(set(values))
    count = max(int(getattr(args, "knee_surrogate_points", 41)), 3)
    if count == 1:
        return [round((lo + hi) / 2.0, 8)]
    return [
        round(lo + (hi - lo) * idx / float(count - 1), 8)
        for idx in range(count)
    ]


def _load_l3_cost_payload_unchecked(path: Path) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, Mapping) or "costs" not in payload:
        raise L3ResumeMetadataError(
            f"L3 resume file {path} does not contain a 'costs' payload"
        )
    cost_history = payload.get("cost_history")
    if not isinstance(cost_history, list) or not all(
        isinstance(item, dict) for item in cost_history
    ):
        cost_history = [payload["costs"]]
    return {
        "path": str(path),
        "costs": dict(payload["costs"]),
        "cost_history": list(cost_history),
        "formats": list(payload.get("formats") or []),
        "meta": dict(payload.get("meta") or {}),
    }


def _assignment_from_anchor_or_default(
    args,
    *,
    resume_path: Path,
    anchor_bpp: float,
    stats: dict,
    costs: dict,
    specs: list[fr.FormatSpec],
) -> tuple[dict[str, str], Path | None, str]:
    if args.initial_config:
        path = Path(args.initial_config)
        return load_assignment_config(path), path, "initial_config"
    label = _bpp_label(anchor_bpp)
    candidate = resume_path.parent / f"l2_assignment_bpp_{label}.json"
    if candidate.exists():
        return load_assignment_config(candidate), candidate, "anchor_l2_assignment"
    return default_initial_assignment(stats, costs, specs), None, "default_initial"


def _candidate_loss_sum(chosen: Mapping[str, Candidate]) -> float:
    return sum(float(c.predicted_dloss) for c in chosen.values())


def _nondominated_surrogate_points(points: list[dict]) -> list[dict]:
    ordered = sorted(
        points,
        key=lambda row: (
            float(row["achieved_bpp"]),
            float(row["surrogate_loss"]),
            float(row["target_bpp"]),
        ),
    )
    frontier = []
    best_loss = float("inf")
    for row in ordered:
        loss = float(row["surrogate_loss"])
        if loss < best_loss - 1e-12:
            frontier.append(row)
            best_loss = loss
    return frontier


def _write_surrogate_frontier_outputs(
    output_root: Path,
    *,
    points: list[dict],
    frontier: list[dict],
    knee: dict,
    selected_indices: list[int],
    metadata: Mapping,
) -> tuple[Path, Path]:
    csv_path = output_root / "surrogate_frontier.csv"
    json_path = output_root / "surrogate_frontier.json"
    fields = [
        "target_bpp",
        "achieved_bpp",
        "surrogate_loss",
        "assignment_hash",
        "hamming_from_anchor",
        "solver",
        "format_histogram",
        "assignment_path",
        "layer_config_path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in frontier:
            record = dict(row)
            record["format_histogram"] = json.dumps(
                record.get("format_histogram", {}),
                sort_keys=True,
            )
            writer.writerow({field: record.get(field) for field in fields})
    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": dict(metadata),
                "knee": dict(knee),
                "selected_indices": list(selected_indices),
                "frontier": frontier,
                "all_unique_points": points,
            },
            f,
            indent=2,
        )
    return csv_path, json_path


def _spread_indices(count: int, selected_count: int) -> set[int]:
    count = int(count)
    selected_count = max(int(selected_count), 0)
    if count <= 0 or selected_count <= 0:
        return set()
    if selected_count == 1:
        return {0}
    if selected_count >= count:
        return set(range(count))
    return {
        int(round(idx * (count - 1) / float(selected_count - 1)))
        for idx in range(selected_count)
    }


def _cap_spread_candidates(
    candidates: list[dict],
    limit: int,
    *,
    required_key: str | None = None,
) -> list[dict]:
    if limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    required_indices = {
        idx
        for idx, row in enumerate(candidates)
        if required_key is not None and row.get(required_key)
    }
    keep = set(required_indices)
    for idx in sorted(_spread_indices(len(candidates), limit)):
        if len(keep) >= limit:
            break
        keep.add(idx)
    if len(keep) < limit:
        for idx in range(len(candidates)):
            if len(keep) >= limit:
                break
            keep.add(idx)
    return [candidates[idx] for idx in sorted(keep)]


def run_surrogate_knee_search(
    args,
    *,
    stats: dict,
    current_costs: dict,
    specs: list[fr.FormatSpec],
    output_root: Path,
) -> int:
    anchor_hint = (
        args.target_bits_anchor
        if args.target_bits_anchor is not None
        else (float(args.knee_bpp_min) + float(args.knee_bpp_max)) / 2.0
    )
    resume_path = _resume_l3_cost_path_for_anchor(
        args,
        float(anchor_hint),
        single_output=True,
    )
    if resume_path is None:
        raise RuntimeError(
            "--knee-surrogate requires --resume-l3-costs or "
            "--resume-l3-costs-dir containing an anchor L3 cost table"
        )
    payload = _load_l3_cost_payload_unchecked(Path(resume_path))
    meta = payload.get("meta", {})
    anchor_bpp = float(meta.get("anchor_bpp", anchor_hint))
    anchor_assignment, anchor_assignment_path, anchor_source = (
        _assignment_from_anchor_or_default(
            args,
            resume_path=Path(resume_path),
            anchor_bpp=anchor_bpp,
            stats=stats,
            costs=current_costs,
            specs=specs,
        )
    )
    l3_candidates = _filter_l3_candidates_for_assignment(
        build_l3_candidates(stats, payload["costs"], specs),
        anchor_assignment,
    )
    if not l3_candidates:
        raise RuntimeError("surrogate knee search found no L3 candidates")

    output_root.mkdir(parents=True, exist_ok=True)
    grid = _surrogate_grid_points(args)
    points_by_hash: dict[tuple, dict] = {}
    failures: list[dict] = []
    for target_bpp in grid:
        try:
            assignment, chosen, solver_meta = solve_frozen_l3_neighborhood(
                stats,
                anchor_assignment,
                l3_candidates,
                specs,
                target_bits=float(target_bpp),
                bit_precision=float(args.bit_precision),
                budget_tolerance=float(args.frozen_dp_budget_tolerance),
                return_metadata=True,
            )
        except FrozenBudgetError as exc:
            failures.append({
                "target_bpp": float(target_bpp),
                "error": str(exc),
            })
            continue
        histogram = _format_histogram(stats, assignment, specs, float(target_bpp))
        digest = assignment_hash(assignment)
        row = {
            "target_bpp": float(target_bpp),
            "achieved_bpp": float(histogram["achieved_bpp"]),
            "surrogate_loss": float(_candidate_loss_sum(chosen)),
            "assignment_hash": _assignment_digest(assignment),
            "hamming_from_anchor": _hamming_count(anchor_assignment, assignment),
            "solver": str(solver_meta.get("frozen_dp_precision_used", "unknown")),
            "format_histogram": histogram,
            "solver_meta": dict(solver_meta),
        }
        previous = points_by_hash.get(digest)
        if previous is None or (
            float(row["surrogate_loss"]),
            abs(float(row["target_bpp"]) - float(row["achieved_bpp"])),
        ) < (
            float(previous["surrogate_loss"]),
            abs(float(previous["target_bpp"]) - float(previous["achieved_bpp"])),
        ):
            row["_assignment"] = assignment
            points_by_hash[digest] = row

    unique_points = list(points_by_hash.values())
    frontier = _nondominated_surrogate_points(unique_points)
    if len(frontier) < 3:
        raise RuntimeError(
            f"surrogate frontier has only {len(frontier)} point(s); "
            "widen --knee-bpp-min/--knee-bpp-max or reduce bit precision"
        )
    knee_bpp, knee_loss, knee_score, endpoint = segmented_kneedle_point(
        [
            (float(row["achieved_bpp"]), float(row["surrogate_loss"]))
            for row in frontier
        ]
    )
    knee_index = min(
        range(len(frontier)),
        key=lambda idx: abs(float(frontier[idx]["achieved_bpp"]) - knee_bpp),
    )
    neighbor_count = max(int(getattr(args, "knee_surrogate_neighbors", 1)), 0)
    selected_indices = sorted(
        {
            idx
            for idx in range(
                max(0, knee_index - neighbor_count),
                min(len(frontier), knee_index + neighbor_count + 1),
            )
        }
    )
    validation_candidate_count = int(
        getattr(args, "knee_surrogate_validation_candidates", 0) or 0
    )
    selected_indices = sorted(
        set(selected_indices)
        | _spread_indices(len(frontier), validation_candidate_count)
    )
    selected_dir = output_root / "surrogate_selected"
    for idx in selected_indices:
        row = frontier[idx]
        label = f"{idx:03d}_bpp_{float(row['achieved_bpp']):.4f}"
        assignment = dict(row.pop("_assignment"))
        assignment_path, layer_config_path = _write_named_assignment_artifacts(
            assignment,
            assignment_path=selected_dir / f"assignment_{label}.json",
            layer_config_path=selected_dir / f"layer_config_{label}.json",
        )
        row["assignment_path"] = str(assignment_path)
        row["layer_config_path"] = str(layer_config_path)
        row["_assignment"] = assignment
    knee_assignment = dict(frontier[knee_index]["_assignment"])
    knee_assignment_path, knee_layer_config_path = _write_named_assignment_artifacts(
        knee_assignment,
        assignment_path=output_root / "knee_surrogate_assignment.json",
        layer_config_path=output_root / "final_layer_config_knee_surrogate.json",
    )
    for row in unique_points:
        row.pop("_assignment", None)
    knee = {
        "mode": "surrogate_kneedle",
        "frontier_index": int(knee_index),
        "achieved_bpp": float(frontier[knee_index]["achieved_bpp"]),
        "target_bpp": float(frontier[knee_index]["target_bpp"]),
        "surrogate_loss": float(knee_loss),
        "kneedle_score": float(knee_score),
        "endpoint_fallback": bool(endpoint),
        "assignment": str(knee_assignment_path),
        "layer_config": str(knee_layer_config_path),
        "format_histogram": frontier[knee_index]["format_histogram"],
    }
    _warn_endpoint_fallback("knee-surrogate", knee)
    for row in frontier:
        row.pop("_assignment", None)
    csv_path, json_path = _write_surrogate_frontier_outputs(
        output_root,
        points=sorted(unique_points, key=lambda r: float(r["achieved_bpp"])),
        frontier=frontier,
        knee=knee,
        selected_indices=selected_indices,
        metadata={
            "mode": "surrogate_knee_search",
            "resume_l3_costs": str(resume_path),
            "anchor_bpp": float(anchor_bpp),
            "anchor_assignment": (
                str(anchor_assignment_path) if anchor_assignment_path else None
            ),
            "anchor_assignment_source": anchor_source,
            "grid": grid,
            "failures": failures,
            "l3_candidate_count": len(l3_candidates),
            "formats": [s.name for s in specs],
            "bit_precision": float(args.bit_precision),
            "frozen_dp_budget_tolerance": float(args.frozen_dp_budget_tolerance),
            **_calibration_metadata(args),
            "quality_note": (
                "Surrogate losses are computed from one cached L3 cost table. "
                "They rank candidate assignments but are not held-out KL."
            ),
        },
    )
    summary = {
        "pareto": {
            "mode": "surrogate_knee_search",
            "csv": str(csv_path),
            "json": str(json_path),
            "frontier_points": len(frontier),
            "unique_points": len(unique_points),
        },
        "knee": knee,
    }
    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _emit("===== summary =====")
    _emit(
        f"Surrogate knee: achieved_bpp={knee['achieved_bpp']:.4f} "
        f"target_bpp={knee['target_bpp']:.4f} "
        f"surrogate_loss={knee['surrogate_loss']:.6g}"
    )
    _emit(f"Knee assignment: {knee_assignment_path}")
    _emit(f"Surrogate frontier: {json_path}")
    _emit("===================")
    return 0


def _bpp_label(value: float) -> str:
    return f"{float(value):.2f}"


def _write_pareto_outputs(
    output_root: Path,
    results: list[BudgetResult],
    *,
    metadata: Mapping | None = None,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "pareto_curve.csv"
    json_path = output_root / "pareto_curve.json"
    fields = [
        "target_bpp",
        "anchor_bpp",
        "distance_from_anchor",
        "anchor_stale",
        "achieved_bpp",
        "predicted_dloss",
        "l2_kl",
        "validation_kl",
        "accepted",
        "regression",
        "flips_accepted",
        "format_histogram",
        "assignment_path",
        "layer_config_path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda r: r.target_bpp):
            record = result.as_record()
            record["format_histogram"] = json.dumps(
                record["format_histogram"],
                sort_keys=True,
            )
            writer.writerow({field: record[field] for field in fields})
    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": dict(metadata or {}),
                "points": [
                    result.as_record()
                    for result in sorted(results, key=lambda r: r.target_bpp)
                ],
            },
            f,
            indent=2,
        )
    return csv_path, json_path


def _write_budget_artifacts(
    output_root: Path,
    label: str,
    assignment: Mapping[str, str],
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    assignment_path = output_root / f"final_assignment_bpp_{label}.json"
    layer_config_path = output_root / f"final_layer_config_bpp_{label}.json"
    with open(assignment_path, "w") as f:
        json.dump(dict(sorted(assignment.items())), f, indent=2)
    write_layer_config(assignment, layer_config_path)
    return assignment_path, layer_config_path


def _write_knee_checkpoint_point(
    output_root: Path,
    result: BudgetResult,
    *,
    source: str,
    metadata: Mapping | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "knee_checkpoint_frontier.json"
    existing_points: list[dict] = []
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                payload = json.load(f)
            if isinstance(payload, Mapping):
                existing = payload.get("points")
                if isinstance(existing, list):
                    existing_points = [
                        row for row in existing if isinstance(row, dict)
                    ]
        except (OSError, json.JSONDecodeError, TypeError):
            existing_points = []

    row = result.as_record()
    row.update(
        {
            "source": source,
            "assignment_hash": _assignment_digest(result.assignment),
            "checkpoint_metadata": dict(metadata or {}),
        }
    )
    by_hash: dict[str, dict] = {}
    for candidate in [*existing_points, row]:
        digest = str(
            candidate.get("assignment_hash")
            or candidate.get("layer_config_path")
            or candidate.get("assignment_path")
            or len(by_hash)
        )
        previous = by_hash.get(digest)
        if previous is None or (
            float(candidate.get("validation_kl", float("inf"))),
            float(candidate.get("achieved_bpp", float("inf"))),
        ) < (
            float(previous.get("validation_kl", float("inf"))),
            float(previous.get("achieved_bpp", float("inf"))),
        ):
            by_hash[digest] = candidate

    points = sorted(
        by_hash.values(),
        key=lambda candidate: (
            float(candidate.get("target_bpp", 0.0)),
            float(candidate.get("validation_kl", float("inf"))),
        ),
    )
    with open(checkpoint_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "mode": "knee_checkpoint_frontier",
                    "quality_note": (
                        "Each point is a real validation-KL assignment checkpoint "
                        "written as soon as it is available."
                    ),
                },
                "points": points,
            },
            f,
            indent=2,
        )
    return checkpoint_path


def _write_named_assignment_artifacts(
    assignment: Mapping[str, str],
    *,
    assignment_path: Path,
    layer_config_path: Path,
) -> tuple[Path, Path]:
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    with open(assignment_path, "w") as f:
        json.dump(dict(sorted(assignment.items())), f, indent=2)
    write_layer_config(assignment, layer_config_path)
    return assignment_path, layer_config_path


def _resolve_seed_artifact_path(raw_path, source_json: Path) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path)
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    candidates: list[Path] = []
    source_dir = source_json.parent
    if not path.is_absolute():
        candidates.append(source_dir / path)
    parts = path.parts
    for marker in (("work", "out"), ("run", "out")):
        for idx in range(max(len(parts) - 1, 0)):
            if tuple(parts[idx:idx + 2]) == marker:
                rel = Path(*parts[idx + 2:])
                candidates.append(source_dir / rel)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _inferred_assignment_paths(path: Path) -> list[Path]:
    names = []
    for old, new in (
        ("final_layer_config", "final_assignment"),
        ("layer_config", "assignment"),
    ):
        if old in path.name:
            names.append(path.with_name(path.name.replace(old, new, 1)))
    return names


def _load_seed_assignment(row: Mapping, source_json: Path) -> dict[str, str] | None:
    raw_paths = [
        row.get("assignment_path"),
        row.get("assignment"),
        row.get("source_path"),
        row.get("layer_config_path"),
        row.get("layer_config"),
    ]
    candidate_paths: list[Path] = []
    for raw_path in raw_paths:
        path = _resolve_seed_artifact_path(raw_path, source_json)
        if path is None:
            continue
        candidate_paths.extend(_inferred_assignment_paths(path))
        candidate_paths.append(path)
    seen: set[Path] = set()
    for path in candidate_paths:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            return load_assignment_config(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def _seed_rows_from_payload(payload: Mapping) -> list[Mapping]:
    rows: list[Mapping] = []
    for key in ("validated_candidates", "frontier", "points"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, Mapping))
    knee = payload.get("knee")
    if isinstance(knee, Mapping):
        rows.append(knee)
    return rows


def _budget_result_from_seed_row(
    row: Mapping,
    source_json: Path,
    runtime: BudgetRuntime,
) -> BudgetResult | None:
    if row.get("validation_kl") is None:
        return None
    assignment = _load_seed_assignment(row, source_json)
    if not assignment:
        return None
    coherent_assignment = _enforce_fused_assignment_coherence(
        assignment,
        runtime.specs,
        profile=getattr(runtime, "profile", None),
    )
    if assignment_hash(coherent_assignment) != assignment_hash(assignment):
        return None
    assignment_path = _resolve_seed_artifact_path(
        row.get("assignment_path") or row.get("assignment"),
        source_json,
    )
    target_bpp = row.get("target_bpp")
    if target_bpp is None:
        target_bpp = row.get("achieved_bpp")
    if target_bpp is None:
        target_bpp = row.get("chosen_bpp")
    target_bpp = float(target_bpp if target_bpp is not None else 0.0)
    histogram = _format_histogram(
        runtime.stats,
        assignment,
        runtime.specs,
        target_bpp,
    )
    achieved_bpp = float(row.get("achieved_bpp", histogram["achieved_bpp"]))
    if achieved_bpp <= 0.0:
        achieved_bpp = float(histogram["achieved_bpp"])
    if target_bpp <= 0.0:
        target_bpp = achieved_bpp
    histogram["achieved_bpp"] = achieved_bpp
    histogram["target_bpp"] = target_bpp
    validation_kl = float(row["validation_kl"])
    l2_kl = float(row.get("l2_kl", validation_kl))
    return BudgetResult(
        target_bpp=target_bpp,
        anchor_bpp=float(row.get("anchor_bpp", target_bpp)),
        distance_from_anchor=float(row.get("distance_from_anchor", 0.0)),
        anchor_stale=bool(row.get("anchor_stale", False)),
        achieved_bpp=achieved_bpp,
        predicted_dloss=float(row.get("predicted_dloss", 0.0)),
        l2_kl=l2_kl,
        validation_kl=validation_kl,
        accepted=bool(row.get("accepted", True)),
        regression=bool(row.get("regression", validation_kl > l2_kl)),
        flips_accepted=int(row.get("flips_accepted", 0) or 0),
        format_histogram=histogram,
        assignment=assignment,
        assignment_path=str(assignment_path or ""),
        layer_config_path=str(
            _resolve_seed_artifact_path(
                row.get("layer_config_path") or row.get("layer_config"),
                source_json,
            )
            or ""
        ),
        l3_iterations=int(row.get("l3_iterations", 0) or 0),
        coord_descent_fired=bool(row.get("coord_descent_fired", False)),
        coord_descent_flips=int(row.get("coord_descent_flips", 0) or 0),
    )


def _load_knee_seed_results_from_file(
    path: Path,
    runtime: BudgetRuntime,
    *,
    seen_files: set[Path],
) -> tuple[list[BudgetResult], dict]:
    path = path.expanduser()
    if not path.exists():
        raise RuntimeError(f"knee seed frontier does not exist: {path}")
    path = path.resolve()
    if path in seen_files:
        return [], {"path": str(path), "skipped": "already_loaded"}
    seen_files.add(path)
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"knee seed frontier is not a JSON object: {path}")

    results: list[BudgetResult] = []
    referenced = payload.get("validated_frontier")
    referenced_meta = None
    if isinstance(referenced, str) and referenced:
        ref_path = _resolve_seed_artifact_path(referenced, path)
        if ref_path is not None and ref_path.exists():
            ref_results, referenced_meta = _load_knee_seed_results_from_file(
                ref_path,
                runtime,
                seen_files=seen_files,
            )
            results.extend(ref_results)

    rows = _seed_rows_from_payload(payload)
    skipped = 0
    for row in rows:
        result = _budget_result_from_seed_row(row, path, runtime)
        if result is None:
            skipped += 1
            continue
        results.append(result)
    return results, {
        "path": str(path),
        "rows": len(rows),
        "loaded": len(results),
        "skipped_rows": skipped,
        "referenced": referenced_meta,
    }


def _load_knee_seed_results(args, runtime: BudgetRuntime) -> tuple[list[BudgetResult], dict]:
    paths = [Path(p) for p in (getattr(args, "knee_seed_frontier", None) or [])]
    auto_resume_paths = [
        runtime.output_root / "pareto_curve.json",
        runtime.output_root / "knee_checkpoint_frontier.json",
    ]
    seen_path_text = {str(path.expanduser()) for path in paths}
    for path in auto_resume_paths:
        if path.exists() and str(path.expanduser()) not in seen_path_text:
            paths.append(path)
            seen_path_text.add(str(path.expanduser()))
    if not paths:
        return [], {"files": [], "loaded": 0}
    by_assignment: dict[str, BudgetResult] = {}
    files = []
    seen_files: set[Path] = set()
    for path in paths:
        results, meta = _load_knee_seed_results_from_file(
            path,
            runtime,
            seen_files=seen_files,
        )
        files.append(meta)
        for result in results:
            digest = _assignment_digest(result.assignment)
            previous = by_assignment.get(digest)
            if previous is None or (
                float(result.validation_kl),
                float(result.achieved_bpp),
            ) < (
                float(previous.validation_kl),
                float(previous.achieved_bpp),
            ):
                by_assignment[digest] = result
    seeds = sorted(
        by_assignment.values(),
        key=lambda result: (
            float(result.achieved_bpp),
            float(result.validation_kl),
            float(result.target_bpp),
        ),
    )
    return seeds, {"files": files, "loaded": len(seeds)}


def _remeasure_knee_seed_results(
    args,
    runtime: BudgetRuntime,
    seed_results: list[BudgetResult],
) -> dict:
    enabled = bool(getattr(args, "knee_seed_remeasure", True))
    meta = {
        "enabled": enabled,
        "points": 0,
        "unique_assignments": 0,
        "updated": 0,
        "max_abs_delta": 0.0,
        "rows": [],
    }
    if not enabled or not seed_results:
        return meta

    measured_by_digest: dict[str, float] = {}
    for result in seed_results:
        digest = _assignment_digest(result.assignment)
        old_kl = float(result.validation_kl)
        if digest not in measured_by_digest:
            measured_by_digest[digest] = float(
                measure_assignment_kl(
                    runtime.model,
                    result.assignment,
                    runtime.calib_ids,
                    runtime.ref_log_probs,
                    work_root=runtime.work_root,
                    profile=runtime.profile,
                )
            )
        measured_kl = float(measured_by_digest[digest])
        result.validation_kl = measured_kl
        result.l2_kl = measured_kl
        result.regression = False
        label = f"seed_{_bpp_label(result.achieved_bpp)}_{digest[:8]}"
        assignment_path, layer_config_path = _write_budget_artifacts(
            runtime.output_root,
            label,
            result.assignment,
        )
        result.assignment_path = str(assignment_path)
        result.layer_config_path = str(layer_config_path)
        _write_knee_checkpoint_point(
            runtime.output_root,
            result,
            source="seed_remeasurement",
            metadata={
                "old_validation_kl": old_kl,
                "new_validation_kl": measured_kl,
                "assignment_hash": digest,
            },
        )
        delta = measured_kl - old_kl
        meta["points"] += 1
        if abs(delta) > 1e-12:
            meta["updated"] += 1
        meta["max_abs_delta"] = max(float(meta["max_abs_delta"]), abs(delta))
        meta["rows"].append(
            {
                "target_bpp": float(result.target_bpp),
                "achieved_bpp": float(result.achieved_bpp),
                "assignment_hash": digest,
                "old_validation_kl": old_kl,
                "new_validation_kl": measured_kl,
                "delta_validation_kl": delta,
            }
        )

    meta["unique_assignments"] = len(measured_by_digest)
    _emit(
        "[knee] remeasured seed frontier: "
        f"{meta['points']} point(s), {meta['unique_assignments']} unique assignment(s), "
        f"{meta['updated']} changed, max_abs_delta={meta['max_abs_delta']:.6g}"
    )
    return meta


def _build_anchor_from_seed_result(
    args,
    runtime: BudgetRuntime,
    seed: BudgetResult,
    *,
    measure_all_formats: bool,
) -> AnchorResult | None:
    resume_path = _resume_l3_cost_path_for_anchor(
        args,
        float(seed.target_bpp),
    )
    if resume_path is None:
        return None
    label = _bpp_label(seed.achieved_bpp)
    output_dir = runtime.output_root / f"seed_anchor_bpp_{label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    resumed_l3 = _load_l3_resume_costs(
        Path(resume_path),
        args=args,
        runtime=runtime,
        anchor_bpp=float(seed.target_bpp),
        prefix=f"[knee][seed-anchor] {label}:",
    )
    selected = _select_l3_neighborhood_for_assignment(
        args,
        runtime.stats,
        runtime.current_costs,
        seed.assignment,
        runtime.specs,
        measure_all_formats=measure_all_formats,
    )
    l3_candidates = build_l3_candidates(
        runtime.stats,
        resumed_l3.costs,
        runtime.specs,
    )
    l3_candidates = _filter_l3_candidates_for_assignment(
        l3_candidates,
        seed.assignment,
    )
    if not l3_candidates:
        return None
    _emit(
        f"[knee] seed anchor {label}: using prevalidated "
        f"KL={seed.validation_kl:.6g} from achieved_bpp={seed.achieved_bpp:.4f}"
    )
    return AnchorResult(
        anchor_bpp=float(seed.achieved_bpp),
        output_dir=output_dir,
        l2_assignment=dict(seed.assignment),
        l2_kl=float(seed.validation_kl),
        l3_selected=list(selected),
        l3_candidates=l3_candidates,
        l3_costs=resumed_l3.costs,
        latest_smoothed_costs=runtime.current_costs,
        l3_cost_history=resumed_l3.cost_history,
        l3_resumed=True,
    )


def run_anchor_budget(
    args,
    runtime: BudgetRuntime,
    anchor_bpp: float,
    *,
    measure_all_formats: bool = False,
) -> AnchorResult:
    anchor_label = _bpp_label(anchor_bpp)
    anchor_dir = runtime.output_root / f"anchor_bpp_{anchor_label}"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    iteration_trace_path = anchor_dir / "iteration_trace.jsonl"
    iteration_trace_path.write_text("")
    _emit(f"[multi] === anchor {anchor_label} bpp ===")
    if args.initial_config:
        assignment = load_assignment_config(args.initial_config)
    else:
        assignment = default_initial_assignment(
            runtime.stats,
            runtime.current_costs,
            runtime.specs,
        )

    cost_history: list[dict] = []
    latest_smoothed_costs = runtime.current_costs
    assignment_history: list[dict[str, str]] = [dict(assignment)]
    convergence_reached = False
    for iteration in range(1, int(args.max_iters) + 1):
        iter_start = time.monotonic()
        _emit(f"[multi][l2] anchor {anchor_label}: iteration {iteration}")
        cache_dir = anchor_dir / f"activation_cache_iter_{iteration:02d}"
        capture_manifest = capture_perturbed_activation_cache(
            runtime.model,
            assignment,
            runtime.calib_ids,
            cache_dir,
            input_rows=args.input_rows,
            profile=runtime.profile,
        )
        cache_bytes = _directory_size_bytes(cache_dir)
        act_cache = ActivationIndex(cache_dir, runtime.stats.keys())
        target_names = set(runtime.stats.keys())
        missing_act = [n for n in target_names if n not in act_cache]
        cost_path = anchor_dir / f"costs_iter_{iteration:02d}.pkl"
        # PRISMAQUANT_L2_CUDA_GRAPHS is skipped for the mixed-shape L2 cost
        # table loop; see the single-budget path comment for the same boundary.
        measured_costs = run_cost_pass(
            runtime.model,
            act_cache,
            target_names,
            missing_act,
            runtime.specs,
            args.model,
            args.probe,
            args.device,
            runtime.dtype,
            args.cost_mode,
            args.chunk_size,
            str(cost_path),
            h_detail_dir=args.h_detail_dir,
        )
        cost_history.append(measured_costs)
        latest_smoothed_costs = smooth_cost_history(
            cost_history,
            decay=args.ema_decay,
        )
        def _solve(cost_table: dict) -> dict[str, str]:
            return solve_from_costs(
                runtime.stats,
                cost_table,
                runtime.specs,
                target_bits=anchor_bpp,
                bit_precision=args.bit_precision,
                profile=runtime.profile,
            )

        next_assignment = _solve(latest_smoothed_costs)
        cycle_mode = "none"
        if len(assignment_history) >= 2:
            next_assignment, cycle_mode = resolve_two_cycle(
                assignment_history[-2],
                assignment_history[-1],
                next_assignment,
                cost_history[-2] if len(cost_history) >= 2 else measured_costs,
                measured_costs,
                _solve,
                lambda candidate_assignment: measure_assignment_kl(
                    runtime.model,
                    candidate_assignment,
                    runtime.calib_ids,
                    runtime.ref_log_probs,
                    work_root=runtime.work_root,
                    profile=runtime.profile,
                ),
            )
        hamming = _weighted_hamming_detail(
            assignment,
            next_assignment,
            latest_smoothed_costs,
            runtime.stats,
        )
        histogram = _format_histogram(
            runtime.stats,
            next_assignment,
            runtime.specs,
            anchor_bpp,
        )
        iter_wall = time.monotonic() - iter_start
        _append_jsonl(
            iteration_trace_path,
            {
                "iteration": iteration,
                "cache_dir": str(cache_dir),
                "cache_size_bytes": cache_bytes,
                "cache": capture_manifest,
                "cost_path": str(cost_path),
                "format_histogram": histogram,
                "hamming": hamming,
                "cycle_mode": cycle_mode,
                "iteration_wall_seconds": iter_wall,
            },
        )
        assignment = dict(next_assignment)
        assignment_history.append(dict(assignment))
        _report_graph_memory(f"[multi][l2] anchor {anchor_label} iter {iteration} after")
        if hamming["ratio"] <= float(args.convergence_frac):
            convergence_reached = True
            break

    _emit(
        f"[multi][l2] anchor {anchor_label}: "
        f"{'converged' if convergence_reached else 'max-iters'}"
    )
    l2_assignment_path, l2_layer_config_path = _write_named_assignment_artifacts(
        assignment,
        assignment_path=anchor_dir / f"l2_assignment_bpp_{anchor_label}.json",
        layer_config_path=anchor_dir / f"l2_layer_config_bpp_{anchor_label}.json",
    )
    _emit(
        f"[multi][l2] anchor {anchor_label}: wrote L2 seed "
        f"{l2_assignment_path} and {l2_layer_config_path}"
    )
    _phase_boundary_cleanup(f"anchor_{anchor_label}_l2_to_l3")
    _report_graph_memory(f"anchor_{anchor_label}_l2_to_l3")
    _report_graph_memory(f"[multi][l3] anchor {anchor_label} before_validation")
    kl_before = measure_assignment_kl(
        runtime.model,
        assignment,
        runtime.calib_ids,
        runtime.ref_log_probs,
        work_root=runtime.work_root,
        profile=runtime.profile,
    )
    _report_graph_memory(f"[multi][l3] anchor {anchor_label} after_validation")
    selected = _select_l3_neighborhood_for_assignment(
        args,
        runtime.stats,
        latest_smoothed_costs,
        assignment,
        runtime.specs,
        measure_all_formats=measure_all_formats,
    )
    resume_path = _resume_l3_cost_path_for_anchor(args, anchor_bpp)
    resumed_l3: L3ResumeCosts | None = None
    if resume_path is not None:
        resumed_l3 = _load_l3_resume_costs(
            resume_path,
            args=args,
            runtime=runtime,
            anchor_bpp=anchor_bpp,
            prefix=f"[multi][l3] anchor {anchor_label}:",
        )
        _emit(
            f"[multi][l3] anchor {anchor_label}: loaded L3 costs from "
            f"{resumed_l3.path}; skipping measurement"
        )
        l3_costs = resumed_l3.costs
    else:
        _emit(
            f"[multi][l3] anchor {anchor_label}: measuring {len(selected)} "
            f"Linears ({'all formats' if measure_all_formats else 'filtered formats'})"
        )
        l3_costs = measure_propagated_costs(
            runtime.model,
            assignment,
            selected,
            runtime.l3_calib_ids,
            runtime.specs,
            work_root=runtime.work_root,
            profile=runtime.profile,
            max_lanes_per_batch=args.l3_max_lanes_per_batch,
            tail_only=args.l3_tail_only,
            output_mse_names=_l3_output_mse_names(args),
            progress_callback=_make_l3_progress(
                _emit,
                prefix=f"[multi][l3] anchor {anchor_label}:",
            ),
        )
        _report_graph_memory(f"[multi][l3] anchor {anchor_label} after_l3_measurement")
    l3_cost_path = anchor_dir / "l3_propagated_costs.pkl"
    meta_extra = {
        "l3_mode": getattr(args, "l3_mode", "selective"),
        "measure_all_formats": bool(measure_all_formats),
        "resumed_from": str(resumed_l3.path) if resumed_l3 is not None else None,
    }
    with open(l3_cost_path, "wb") as f:
        pickle.dump(
            {
                "costs": l3_costs,
                "cost_history": (
                    resumed_l3.cost_history
                    if resumed_l3 is not None
                    else [l3_costs]
                ),
                "formats": [s.name for s in runtime.specs],
                "meta": _l3_cost_payload_meta(
                    args,
                    runtime,
                    anchor_bpp=anchor_bpp,
                    selected_count=len(selected),
                    extra=meta_extra,
                ),
            },
            f,
        )
    l3_candidates = build_l3_candidates(runtime.stats, l3_costs, runtime.specs)
    l3_candidates = _filter_l3_candidates_for_assignment(
        l3_candidates,
        assignment,
    )
    _phase_boundary_cleanup(f"anchor_{anchor_label}_complete")
    return AnchorResult(
        anchor_bpp=float(anchor_bpp),
        output_dir=anchor_dir,
        l2_assignment=dict(assignment),
        l2_kl=float(kl_before),
        l3_selected=list(selected),
        l3_candidates=l3_candidates,
        l3_costs=l3_costs,
        latest_smoothed_costs=latest_smoothed_costs,
        l3_cost_history=(
            resumed_l3.cost_history
            if resumed_l3 is not None
            else [l3_costs]
        ),
        l3_resumed=bool(resumed_l3 is not None),
    )


def solve_target_from_anchor(
    args,
    runtime: BudgetRuntime,
    anchor: AnchorResult,
    target_bpp: float,
) -> BudgetResult:
    target_label = _bpp_label(target_bpp)

    def _checkpoint_scout_assignment(
        assignment: Mapping[str, str],
        validation_kl: float,
        commit_record: Mapping,
    ) -> None:
        checkpoint_label = (
            f"{target_label}_scout_"
            f"{int(commit_record.get('round', 0)):02d}_"
            f"{_assignment_digest(assignment)[:8]}"
        )
        assignment_path, layer_config_path = _write_budget_artifacts(
            runtime.output_root,
            checkpoint_label,
            assignment,
        )
        histogram = _format_histogram(
            runtime.stats,
            assignment,
            runtime.specs,
            target_bpp,
        )
        predicted = _predicted_dloss_for_assignment(
            assignment,
            anchor.latest_smoothed_costs,
            runtime.stats,
        )
        checkpoint_result = BudgetResult(
            target_bpp=float(target_bpp),
            anchor_bpp=float(anchor.anchor_bpp),
            distance_from_anchor=abs(float(target_bpp) - float(anchor.anchor_bpp)),
            anchor_stale=bool(
                abs(float(target_bpp) - float(anchor.anchor_bpp)) > 1e-12
            ),
            achieved_bpp=float(histogram["achieved_bpp"]),
            predicted_dloss=float(predicted),
            l2_kl=float(anchor.l2_kl),
            validation_kl=float(validation_kl),
            accepted=True,
            regression=bool(float(validation_kl) > float(anchor.l2_kl)),
            flips_accepted=_hamming_count(anchor.l2_assignment, assignment),
            format_histogram=histogram,
            assignment=dict(assignment),
            assignment_path=str(assignment_path),
            layer_config_path=str(layer_config_path),
        )
        checkpoint_path = _write_knee_checkpoint_point(
            runtime.output_root,
            checkpoint_result,
            source="validation_scout_commit",
            metadata=commit_record,
        )
        _emit(
            f"[knee] checkpoint target={target_label} "
            f"achieved={checkpoint_result.achieved_bpp:.4f} "
            f"KL={checkpoint_result.validation_kl:.6g} -> {checkpoint_path}"
        )

    polish = run_iterated_l3_polish(
        args,
        runtime.model,
        anchor.l2_assignment,
        anchor.l2_kl,
        runtime.stats,
        anchor.latest_smoothed_costs,
        runtime.specs,
        target_bpp,
        runtime.calib_ids,
        runtime.l3_calib_ids,
        runtime.ref_log_probs,
        work_root=runtime.work_root,
        profile=runtime.profile,
        prefix=f"[multi][l3] target {target_label}:",
        measure_all_formats=bool(getattr(args, "_l3_measure_all_formats", False)),
        initial_selected=anchor.l3_selected,
        initial_l3_costs=anchor.l3_costs,
        resume_l3_cost_history=(
            anchor.l3_cost_history if anchor.l3_resumed else None
        ),
        coord_anchor_label=_bpp_label(anchor.anchor_bpp),
        scout_checkpoint_callback=_checkpoint_scout_assignment,
    )
    accepted = bool(polish.accepted)
    if accepted:
        final_assignment = dict(polish.assignment)
        validation_kl = float(polish.final_kl)
    else:
        final_assignment = solve_from_costs(
            runtime.stats,
            anchor.latest_smoothed_costs,
            runtime.specs,
            target_bits=target_bpp,
            bit_precision=args.bit_precision,
            profile=runtime.profile,
        )
        if dict(final_assignment) == dict(polish.assignment):
            validation_kl = float(polish.proposed_kl)
        else:
            _emit(
                f"[multi][l3] target {target_label}: validating fallback "
                "target assignment"
            )
            validation_kl = float(
                measure_assignment_kl(
                    runtime.model,
                    final_assignment,
                    runtime.calib_ids,
                    runtime.ref_log_probs,
                    work_root=runtime.work_root,
                    profile=runtime.profile,
                )
            )
    predicted = _predicted_dloss_for_assignment(
        final_assignment,
        anchor.latest_smoothed_costs,
        runtime.stats,
    )
    histogram = _format_histogram(
        runtime.stats,
        final_assignment,
        runtime.specs,
        target_bpp,
    )
    flips = sum(
        1
        for name in set(anchor.l2_assignment) | set(final_assignment)
        if anchor.l2_assignment.get(name) != final_assignment.get(name)
    )
    assignment_path, layer_config_path = _write_budget_artifacts(
        runtime.output_root,
        target_label,
        final_assignment,
    )
    regression = bool(polish.regression)
    return BudgetResult(
        target_bpp=float(target_bpp),
        anchor_bpp=float(anchor.anchor_bpp),
        distance_from_anchor=abs(float(target_bpp) - float(anchor.anchor_bpp)),
        anchor_stale=bool(abs(float(target_bpp) - float(anchor.anchor_bpp)) > 1e-12),
        achieved_bpp=float(histogram["achieved_bpp"]),
        predicted_dloss=float(predicted),
        l2_kl=float(anchor.l2_kl),
        validation_kl=float(validation_kl),
        accepted=accepted,
        regression=regression,
        flips_accepted=int(flips),
        format_histogram=histogram,
        assignment=dict(final_assignment),
        assignment_path=str(assignment_path),
        layer_config_path=str(layer_config_path),
        l3_iterations=len(polish.iterations),
        coord_descent_fired=bool(polish.coord_descent_fired),
        coord_descent_flips=int(polish.coord_descent_flips),
    )


def run_single_budget(
    args,
    target_bits: float,
    reusable_anchor: AnchorResult | None = None,
) -> BudgetResult:
    runtime = args._runtime
    if reusable_anchor is None:
        reusable_anchor = run_anchor_budget(
            args,
            runtime,
            float(target_bits),
            measure_all_formats=bool(getattr(args, "_l3_measure_all_formats", False)),
        )
    return solve_target_from_anchor(args, runtime, reusable_anchor, float(target_bits))


def run_multi_budget(args, runtime: BudgetRuntime) -> int:
    targets = parse_target_bits_list(args.target_bits_list)
    measure_all_formats = _resolve_l3_measure_all_formats(args)
    args._runtime = runtime
    args._l3_measure_all_formats = measure_all_formats
    clusters = plan_target_bit_clusters(
        targets,
        float(args.target_bits_share_tolerance),
        anchor=args.target_bits_anchor,
    )
    results: list[BudgetResult] = []
    for cluster in clusters:
        anchor = run_anchor_budget(
            args,
            runtime,
            float(cluster["anchor"]),
            measure_all_formats=measure_all_formats,
        )
        for target in cluster["targets"]:
            result = run_single_budget(args, float(target), reusable_anchor=anchor)
            results.append(result)
            _phase_boundary_cleanup(
                f"target_{_bpp_label(float(target))}_complete"
            )
            l2_kl = float(result.l2_kl)
            l3_kl = float(result.validation_kl)
            _emit(
                f"[multi] target {result.target_bpp:.2f}: "
                f"anchor={result.anchor_bpp:.2f} "
                f"L2_KL={l2_kl:.4g} "
                f"L3_KL={l3_kl:.4g} "
                f"delta={l3_kl - l2_kl:+.4g} "
                f"iters={result.l3_iterations} "
                f"coord={result.coord_descent_flips} "
                f"accepted={str(result.accepted).lower()}"
            )
            if not result.accepted and l3_kl > l2_kl * 1.05:
                _emit(
                    "[multi] WARNING: L3 polish regressed L2 by >5% — "
                    "likely non-additive cost interaction; consider tighter L3 scope"
                )
        _phase_boundary_cleanup(
            f"anchor_{_bpp_label(float(cluster['anchor']))}_to_next_anchor"
        )
    csv_path, json_path = _write_pareto_outputs(
        runtime.output_root,
        results,
        metadata={
            "mode": "target_bits_list",
            "targets": targets,
            "share_tolerance": float(args.target_bits_share_tolerance),
            "clusters": clusters,
            **_l3_scope_metadata(args),
            **_calibration_metadata(args),
            "search_telemetry": str(_search_telemetry_path(args)),
        },
    )
    summary = {
        "pareto": {
            "mode": "target_bits_list",
            "targets": targets,
            "share_tolerance": float(args.target_bits_share_tolerance),
            "clusters": clusters,
            "csv": str(csv_path),
            "json": str(json_path),
            "search_telemetry": str(_search_telemetry_path(args)),
            **_l3_scope_metadata(args),
            **_calibration_metadata(args),
            "points": [result.as_record() for result in results],
        }
    }
    with open(runtime.output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _emit("===== summary =====")
    _emit(f"Pareto CSV: {csv_path}")
    _emit(f"Pareto JSON: {json_path}")
    _emit("===================")
    return 0


def run_knee_search(args, runtime: BudgetRuntime) -> int:
    args._runtime = runtime
    args._l3_measure_all_formats = _resolve_l3_measure_all_formats(args)
    kl_noise_floor = _knee_kl_noise_floor(args)
    anchors: list[AnchorResult] = []
    seed_anchors_by_digest: dict[str, AnchorResult] = {}
    results_by_bpp: dict[float, BudgetResult] = {}
    seed_results, seed_meta = _load_knee_seed_results(args, runtime)
    seed_results = [
        result for result in seed_results
        if float(args.knee_bpp_min) - 1e-12
        <= float(result.achieved_bpp)
        <= float(args.knee_bpp_max) + 1e-12
    ]
    seed_meta["remeasure"] = _remeasure_knee_seed_results(
        args,
        runtime,
        seed_results,
    )
    for result in seed_results:
        key = round(float(result.target_bpp), 8)
        previous = results_by_bpp.get(key)
        if previous is None or (
            float(result.validation_kl),
            float(result.achieved_bpp),
        ) < (
            float(previous.validation_kl),
            float(previous.achieved_bpp),
        ):
            results_by_bpp[key] = result
    if seed_results:
        _emit(
            f"[knee] loaded {len(seed_results)} prevalidated seed point(s) "
            "for real-KL frontier search"
        )

    def _nearest_anchor(bpp: float) -> AnchorResult | None:
        eligible = [
            anchor
            for anchor in anchors
            if abs(anchor.anchor_bpp - float(bpp))
            <= float(args.target_bits_share_tolerance) + 1e-12
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda a: abs(a.anchor_bpp - float(bpp)))

    def _nearest_seed_result(bpp: float) -> BudgetResult | None:
        eligible = [
            result for result in seed_results
            if abs(float(result.achieved_bpp) - float(bpp))
            <= float(args.target_bits_share_tolerance) + 1e-12
        ]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda result: (
                abs(float(result.achieved_bpp) - float(bpp)),
                float(result.validation_kl),
            ),
        )

    def _seed_anchor_for_target(bpp: float) -> AnchorResult | None:
        seed = _nearest_seed_result(float(bpp))
        if seed is None:
            return None
        digest = _assignment_digest(seed.assignment)
        anchor = seed_anchors_by_digest.get(digest)
        if anchor is not None:
            return anchor
        anchor = _build_anchor_from_seed_result(
            args,
            runtime,
            seed,
            measure_all_formats=bool(args._l3_measure_all_formats),
        )
        if anchor is None:
            return None
        seed_anchors_by_digest[digest] = anchor
        anchors.append(anchor)
        return anchor

    def _evaluate_result(bpp: float) -> BudgetResult:
        key = round(float(bpp), 8)
        if key in results_by_bpp:
            return results_by_bpp[key]
        anchor = _nearest_anchor(float(bpp))
        if anchor is None:
            anchor = _seed_anchor_for_target(float(bpp))
        if anchor is None:
            anchor = run_anchor_budget(
                args,
                runtime,
                float(bpp),
                measure_all_formats=bool(args._l3_measure_all_formats),
            )
            anchors.append(anchor)
        result = run_single_budget(args, float(bpp), reusable_anchor=anchor)
        results_by_bpp[key] = result
        _write_knee_checkpoint_point(
            runtime.output_root,
            result,
            source="completed_budget_eval",
            metadata={"target_bpp": float(bpp)},
        )
        _write_pareto_outputs(
            runtime.output_root,
            list(results_by_bpp.values()),
            metadata={
                "mode": "knee_search_checkpoint",
                "knee_max_evaluations": int(args.knee_max_evaluations),
                "knee_tolerance": float(args.knee_tolerance),
                "knee_kl_noise_floor": float(kl_noise_floor),
                "knee_seed_frontier": seed_meta,
                **_l3_scope_metadata(args),
                **_calibration_metadata(args),
            },
        )
        _phase_boundary_cleanup(f"knee_eval_{_bpp_label(float(bpp))}_complete")
        return result

    def _evaluate_kl(bpp: float) -> float:
        return float(_evaluate_result(float(bpp)).validation_kl)

    if args.knee_mode == "threshold":
        if args.knee_threshold_kl is None:
            raise RuntimeError("--knee-threshold-kl is required for threshold mode")
        chosen_bpp, points, knee_meta = threshold_knee_search(
            _evaluate_kl,
            float(args.knee_bpp_min),
            float(args.knee_bpp_max),
            threshold_kl=float(args.knee_threshold_kl),
            tolerance=float(args.knee_tolerance),
            max_evaluations=int(args.knee_max_evaluations),
        )
        knee_meta["evaluated_points"] = points
        chosen_key = round(float(chosen_bpp), 8)
        if chosen_key not in results_by_bpp:
            _evaluate_result(float(chosen_bpp))
        chosen = results_by_bpp[chosen_key]
        results = list(results_by_bpp.values())
    else:
        chosen, results, knee_meta = adaptive_validated_frontier_kneedle(
            _evaluate_result,
            float(args.knee_bpp_min),
            float(args.knee_bpp_max),
            tolerance=float(args.knee_tolerance),
            max_evaluations=int(args.knee_max_evaluations),
            initial_points=int(args.knee_initial_points),
            initial_results=list(results_by_bpp.values()),
            kl_noise_floor=float(kl_noise_floor),
            unstable_policy=str(getattr(args, "knee_unstable_policy", "best-kl")),
        )
        _warn_endpoint_fallback("knee", knee_meta)
        _warn_kneedle_instability("knee", knee_meta)
    csv_path, json_path = _write_pareto_outputs(
        runtime.output_root,
        results,
        metadata={
            "mode": "knee_search",
            "knee": knee_meta,
            "knee_max_evaluations": int(args.knee_max_evaluations),
            "knee_tolerance": float(args.knee_tolerance),
            "knee_kl_noise_floor": float(kl_noise_floor),
            "knee_seed_frontier": seed_meta,
            **_l3_scope_metadata(args),
            **_calibration_metadata(args),
        },
    )
    knee_assignment_path = runtime.output_root / "knee_assignment.json"
    with open(knee_assignment_path, "w") as f:
        json.dump(dict(sorted(chosen.assignment.items())), f, indent=2)
    knee_config_path = runtime.output_root / "final_layer_config_knee.json"
    write_layer_config(chosen.assignment, knee_config_path)
    summary = {
        "pareto": {
            "mode": "knee_search",
            "csv": str(csv_path),
            "json": str(json_path),
            "knee_max_evaluations": int(args.knee_max_evaluations),
            "knee_tolerance": float(args.knee_tolerance),
            "knee_seed_frontier": seed_meta,
            **_l3_scope_metadata(args),
            **_calibration_metadata(args),
            "points": [result.as_record() for result in results],
        },
        "knee": {
            **knee_meta,
            "chosen_bpp": float(chosen.target_bpp),
            "l2_kl": float(chosen.l2_kl),
            "validation_kl": float(chosen.validation_kl),
            "assignment": str(knee_assignment_path),
            "layer_config": str(knee_config_path),
        },
    }
    with open(runtime.output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _emit("===== summary =====")
    _emit(
        f"Knee bpp: target={chosen.target_bpp:.2f} "
        f"achieved={chosen.achieved_bpp:.4f} KL={chosen.validation_kl:.6g}"
    )
    _emit(f"Knee assignment: {knee_assignment_path}")
    _emit(f"Pareto CSV: {csv_path}")
    _emit("===================")
    return 0


def _load_assignment_json(path: str | Path) -> dict[str, str]:
    with open(path) as f:
        payload = json.load(f)
    return {
        str(name): fr.canonical_format_name(str(fmt))
        for name, fmt in payload.items()
    }


def _candidate_from_assignment(
    *,
    assignment: Mapping[str, str],
    stats: Mapping[str, Mapping],
    specs: list[fr.FormatSpec],
    source: str,
    source_path: str | None = None,
    target_bpp: float | None = None,
    surrogate_loss: float | None = None,
) -> dict:
    histogram = _format_histogram(
        stats,
        assignment,
        specs,
        float(target_bpp) if target_bpp is not None else 0.0,
    )
    return {
        "source": source,
        "source_path": source_path,
        "target_bpp": (float(target_bpp) if target_bpp is not None else None),
        "achieved_bpp": float(histogram["achieved_bpp"]),
        "surrogate_loss": (
            float(surrogate_loss) if surrogate_loss is not None else None
        ),
        "assignment_hash": _assignment_digest(assignment),
        "format_histogram": histogram,
        "_assignment": dict(assignment),
    }


def _nondominated_validated_points(
    points: list[dict],
    *,
    kl_noise_floor: float = DEFAULT_KNEE_KL_NOISE_FLOOR,
) -> list[dict]:
    ordered = sorted(
        points,
        key=lambda row: (
            float(row["achieved_bpp"]),
            float(row["validation_kl"]),
            str(row["assignment_hash"]),
        ),
    )
    frontier = []
    best_kl = float("inf")
    kl_noise_floor = max(float(kl_noise_floor), 0.0)
    for row in ordered:
        kl = float(row["validation_kl"])
        if kl < best_kl - kl_noise_floor - 1e-12:
            frontier.append(row)
            best_kl = kl
    return frontier


def _write_validated_frontier_outputs(
    output_root: Path,
    *,
    candidates: list[dict],
    frontier: list[dict],
    knee: dict,
    metadata: Mapping,
) -> tuple[Path, Path]:
    csv_path = output_root / "validated_frontier.csv"
    json_path = output_root / "validated_frontier.json"
    fields = [
        "source",
        "target_bpp",
        "achieved_bpp",
        "validation_kl",
        "surrogate_loss",
        "assignment_hash",
        "format_histogram",
        "assignment_path",
        "layer_config_path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in frontier:
            record = dict(row)
            record["format_histogram"] = json.dumps(
                record.get("format_histogram", {}),
                sort_keys=True,
            )
            writer.writerow({field: record.get(field) for field in fields})
    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": dict(metadata),
                "knee": dict(knee),
                "frontier": frontier,
                "validated_candidates": candidates,
            },
            f,
            indent=2,
        )
    return csv_path, json_path


def run_validated_surrogate_knee_search(args, runtime: BudgetRuntime) -> int:
    kl_noise_floor = _knee_kl_noise_floor(args)
    unstable_policy = str(getattr(args, "knee_unstable_policy", "best-kl"))
    frontier_path = runtime.output_root / "surrogate_frontier.json"
    if not frontier_path.exists():
        raise RuntimeError(
            "validated surrogate knee search requires surrogate_frontier.json; "
            "run_surrogate_knee_search first"
        )
    with open(frontier_path) as f:
        surrogate = json.load(f)
    candidates_by_hash: dict[str, dict] = {}
    for row in surrogate.get("frontier", []):
        assignment_path = row.get("assignment_path")
        if not assignment_path:
            continue
        assignment = _load_assignment_json(assignment_path)
        candidate = _candidate_from_assignment(
            assignment=assignment,
            stats=runtime.stats,
            specs=runtime.specs,
            source="surrogate_frontier",
            source_path=str(assignment_path),
            target_bpp=row.get("target_bpp"),
            surrogate_loss=row.get("surrogate_loss"),
        )
        candidate["assignment_path"] = str(assignment_path)
        candidate["layer_config_path"] = row.get("layer_config_path")
        candidates_by_hash.setdefault(candidate["assignment_hash"], candidate)

    knee_assignment_path = surrogate.get("knee", {}).get("assignment")
    if knee_assignment_path:
        assignment = _load_assignment_json(knee_assignment_path)
        candidate = _candidate_from_assignment(
            assignment=assignment,
            stats=runtime.stats,
            specs=runtime.specs,
            source="surrogate_knee",
            source_path=str(knee_assignment_path),
            target_bpp=surrogate.get("knee", {}).get("target_bpp"),
            surrogate_loss=surrogate.get("knee", {}).get("surrogate_loss"),
        )
        candidate["is_surrogate_knee"] = True
        candidate["assignment_path"] = str(knee_assignment_path)
        candidate["layer_config_path"] = surrogate.get("knee", {}).get("layer_config")
        existing = candidates_by_hash.get(candidate["assignment_hash"])
        if existing is None:
            candidates_by_hash[candidate["assignment_hash"]] = candidate
        else:
            existing["is_surrogate_knee"] = True
            if existing.get("source") == "surrogate_frontier":
                existing["source"] = "surrogate_knee"
                existing["source_path"] = str(knee_assignment_path)
                existing["target_bpp"] = candidate.get("target_bpp")
                existing["surrogate_loss"] = candidate.get("surrogate_loss")
                existing["assignment_path"] = str(knee_assignment_path)
                existing["layer_config_path"] = candidate.get("layer_config_path")

    for path in getattr(args, "knee_include_assignment", None) or []:
        assignment = _load_assignment_json(path)
        candidate = _candidate_from_assignment(
            assignment=assignment,
            stats=runtime.stats,
            specs=runtime.specs,
            source="included_assignment",
            source_path=str(path),
        )
        candidates_by_hash.setdefault(candidate["assignment_hash"], candidate)

    sort_key = lambda row: (
        float(row["achieved_bpp"]),
        (
            row.get("surrogate_loss")
            if row.get("surrogate_loss") is not None
            else float("inf")
        ),
        str(row["assignment_hash"]),
    )
    generated_candidates = sorted(
        [
            row
            for row in candidates_by_hash.values()
            if row.get("source") != "included_assignment"
        ],
        key=sort_key,
    )
    included_candidates = sorted(
        [
            row
            for row in candidates_by_hash.values()
            if row.get("source") == "included_assignment"
        ],
        key=sort_key,
    )
    validate_limit = int(
        getattr(args, "knee_surrogate_validation_candidates", 0) or 0
    )
    if validate_limit > 0:
        generated_candidates = _cap_spread_candidates(
            generated_candidates,
            validate_limit,
            required_key="is_surrogate_knee",
        )
    candidates = sorted(
        generated_candidates + included_candidates,
        key=sort_key,
    )
    if len(candidates) < 3:
        raise RuntimeError(
            f"validated surrogate knee search has only {len(candidates)} "
            "candidate(s); increase --knee-surrogate-validation-candidates"
        )
    validated_dir = runtime.output_root / "validated_selected"
    for index, row in enumerate(candidates):
        assignment = dict(row["_assignment"])
        _emit(
            f"[knee][validate] {index + 1}/{len(candidates)} "
            f"source={row['source']} bpp={row['achieved_bpp']:.4f}"
        )
        validation_kl = measure_assignment_kl(
            runtime.model,
            assignment,
            runtime.calib_ids,
            runtime.ref_log_probs,
            work_root=runtime.work_root,
            profile=runtime.profile,
            use_frozen_weight_cache=False,
        )
        row["validation_kl"] = float(validation_kl)
        label = f"{index:03d}_bpp_{float(row['achieved_bpp']):.4f}"
        if not row.get("assignment_path"):
            assignment_path, layer_config_path = _write_named_assignment_artifacts(
                assignment,
                assignment_path=validated_dir / f"assignment_{label}.json",
                layer_config_path=validated_dir / f"layer_config_{label}.json",
            )
            row["assignment_path"] = str(assignment_path)
            row["layer_config_path"] = str(layer_config_path)
        row.pop("_assignment", None)

    validated_frontier = _nondominated_validated_points(
        candidates,
        kl_noise_floor=kl_noise_floor,
    )
    if len(validated_frontier) < 3:
        # Still write all data; choose the best KL if monotone dominance leaves
        # too few points for a geometric knee.
        chosen = min(candidates, key=lambda row: float(row["validation_kl"]))
        knee_score = 0.0
        endpoint = True
    else:
        knee_bpp, knee_kl, knee_score, endpoint = segmented_kneedle_point(
            [
                (float(row["achieved_bpp"]), float(row["validation_kl"]))
                for row in validated_frontier
            ]
        )
        chosen = min(
            validated_frontier,
            key=lambda row: (
                abs(float(row["achieved_bpp"]) - float(knee_bpp)),
                abs(float(row["validation_kl"]) - float(knee_kl)),
            ),
        )
    geometric_chosen = chosen
    loo_diagnostic = kneedle_leave_one_out_diagnostic(
        [
            (float(row["achieved_bpp"]), float(row["validation_kl"]))
            for row in validated_frontier
        ],
        float(geometric_chosen["achieved_bpp"]),
        float(geometric_chosen["validation_kl"]),
        tolerance_bpp=float(getattr(args, "knee_tolerance", 0.1)),
        kl_noise_floor=float(kl_noise_floor),
    )
    selection_guard = {
        "policy": unstable_policy,
        "applied": False,
        "reason": "stable_or_disabled",
    }
    if (
        unstable_policy == "best-kl"
        and loo_diagnostic.get("enabled")
        and not loo_diagnostic.get("stable", True)
        and validated_frontier
    ):
        best_kl = min(
            validated_frontier,
            key=lambda row: (
                float(row["validation_kl"]),
                float(row["achieved_bpp"]),
                float(row.get("target_bpp") or row["achieved_bpp"]),
            ),
        )
        if best_kl is not geometric_chosen:
            chosen = best_kl
            selection_guard = {
                "policy": unstable_policy,
                "applied": True,
                "reason": "leave_one_out_unstable",
                "geometric_chosen_target_bpp": geometric_chosen.get("target_bpp"),
                "geometric_chosen_achieved_bpp": float(geometric_chosen["achieved_bpp"]),
                "geometric_chosen_validation_kl": float(geometric_chosen["validation_kl"]),
                "guarded_chosen_target_bpp": chosen.get("target_bpp"),
                "guarded_chosen_achieved_bpp": float(chosen["achieved_bpp"]),
                "guarded_chosen_validation_kl": float(chosen["validation_kl"]),
            }
    assignment = _load_assignment_json(chosen["assignment_path"])
    knee_assignment_path, knee_layer_config_path = _write_named_assignment_artifacts(
        assignment,
        assignment_path=runtime.output_root / "validated_knee_assignment.json",
        layer_config_path=runtime.output_root / "final_layer_config_validated_knee.json",
    )
    knee = {
        "mode": "validated_surrogate_kneedle",
        "source": chosen["source"],
        "achieved_bpp": float(chosen["achieved_bpp"]),
        "target_bpp": chosen.get("target_bpp"),
        "validation_kl": float(chosen["validation_kl"]),
        "surrogate_loss": chosen.get("surrogate_loss"),
        "kneedle_score": float(knee_score),
        "endpoint_fallback": bool(endpoint),
        "knee_kl_noise_floor": float(kl_noise_floor),
        "leave_one_out": loo_diagnostic,
        "selection_guard": selection_guard,
        "assignment": str(knee_assignment_path),
        "layer_config": str(knee_layer_config_path),
        "format_histogram": chosen["format_histogram"],
    }
    _warn_endpoint_fallback("validated-knee", knee)
    _warn_kneedle_instability("validated-knee", knee)
    csv_path, json_path = _write_validated_frontier_outputs(
        runtime.output_root,
        candidates=candidates,
        frontier=validated_frontier,
        knee=knee,
        metadata={
            "mode": "validated_surrogate_knee_search",
            "surrogate_frontier": str(frontier_path),
            "validation_candidate_count": len(candidates),
            "validation_calib_samples": int(args.n_calib_samples),
            "validation_calib_seqlen": int(args.calib_seqlen),
            "knee_kl_noise_floor": float(kl_noise_floor),
            "included_assignments": list(
                getattr(args, "knee_include_assignment", None) or []
            ),
            **_calibration_metadata(args),
            "quality_note": (
                "Kneedle is selected from real assignment KL measured on this "
                "validation calibration set; surrogate losses are retained "
                "only as candidate-generation diagnostics."
            ),
        },
    )
    with open(runtime.output_root / "validated_summary.json", "w") as f:
        json.dump(
            {
                "knee": knee,
                "validated_frontier": str(json_path),
                "validated_frontier_csv": str(csv_path),
            },
            f,
            indent=2,
        )
    _emit("===== validated summary =====")
    _emit(
        f"Validated knee: bpp={knee['achieved_bpp']:.4f} "
        f"KL={knee['validation_kl']:.6g} source={knee['source']}"
    )
    _emit(f"Validated knee config: {knee_layer_config_path}")
    _emit(f"Validated frontier: {json_path}")
    _emit("=============================")
    return 0


def run_quality_equivalent_search(args, runtime: BudgetRuntime) -> int:
    """Find the smallest bpp matching a higher-bit reference KL."""
    args._runtime = runtime
    reference_bpp = float(args.quality_equivalent_bits)
    bpp_min = float(args.quality_equivalent_bpp_min)
    args._l3_measure_all_formats = _resolve_l3_measure_all_formats(args)
    anchors: list[AnchorResult] = []
    results_by_bpp: dict[float, BudgetResult] = {}

    def _nearest_anchor(bpp: float) -> AnchorResult | None:
        eligible = [
            anchor
            for anchor in anchors
            if abs(anchor.anchor_bpp - float(bpp))
            <= float(args.target_bits_share_tolerance) + 1e-12
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda a: abs(a.anchor_bpp - float(bpp)))

    def _evaluate_result(bpp: float) -> BudgetResult:
        key = round(float(bpp), 8)
        if key in results_by_bpp:
            return results_by_bpp[key]
        anchor = _nearest_anchor(float(bpp))
        if anchor is None:
            anchor = run_anchor_budget(
                args,
                runtime,
                float(bpp),
                measure_all_formats=bool(args._l3_measure_all_formats),
            )
            anchors.append(anchor)
        result = run_single_budget(args, float(bpp), reusable_anchor=anchor)
        results_by_bpp[key] = result
        _phase_boundary_cleanup(
            f"quality_equiv_eval_{_bpp_label(float(bpp))}_complete"
        )
        return result

    def _evaluate_kl(bpp: float) -> float:
        return float(_evaluate_result(float(bpp)).validation_kl)

    reference = _evaluate_result(reference_bpp)
    threshold_kl = (
        float(reference.validation_kl)
        + float(args.quality_equivalent_kl_slack)
    )
    chosen_bpp, points, threshold_meta = threshold_knee_search(
        _evaluate_kl,
        bpp_min,
        reference_bpp,
        threshold_kl=threshold_kl,
        tolerance=float(args.quality_equivalent_bpp_tolerance),
        max_evaluations=int(args.quality_equivalent_max_evaluations),
    )
    chosen_key = round(float(chosen_bpp), 8)
    if chosen_key not in results_by_bpp:
        _evaluate_result(float(chosen_bpp))
    chosen = results_by_bpp[chosen_key]
    results = list(results_by_bpp.values())
    knee_meta = knee_candidate_from_points(points)
    csv_path, json_path = _write_pareto_outputs(
        runtime.output_root,
        results,
        metadata={
            "mode": "quality_equivalent",
            "reference_bpp": reference_bpp,
            "reference_validation_kl": float(reference.validation_kl),
            "kl_slack": float(args.quality_equivalent_kl_slack),
            "threshold_kl": threshold_kl,
            "threshold": threshold_meta,
            "evaluated_points": points,
            "knee_candidate": knee_meta,
            **_l3_scope_metadata(args),
            **_calibration_metadata(args),
            "search_telemetry": str(_search_telemetry_path(args)),
        },
    )
    assignment_path = runtime.output_root / "quality_equivalent_assignment.json"
    with open(assignment_path, "w") as f:
        json.dump(dict(sorted(chosen.assignment.items())), f, indent=2)
    config_path = runtime.output_root / "final_layer_config_quality_equivalent.json"
    write_layer_config(chosen.assignment, config_path)
    summary = {
        "pareto": {
            "mode": "quality_equivalent",
            "csv": str(csv_path),
            "json": str(json_path),
            "search_telemetry": str(_search_telemetry_path(args)),
            **_l3_scope_metadata(args),
            **_calibration_metadata(args),
            "points": [result.as_record() for result in results],
        },
        "quality_equivalent": {
            "reference_bpp": reference_bpp,
            "reference_validation_kl": float(reference.validation_kl),
            "kl_slack": float(args.quality_equivalent_kl_slack),
            "threshold_kl": threshold_kl,
            "chosen_bpp": float(chosen.target_bpp),
            "chosen_achieved_bpp": float(chosen.achieved_bpp),
            "validation_kl": float(chosen.validation_kl),
            "threshold": threshold_meta,
            "evaluated_points": points,
            "knee_candidate": knee_meta,
            "assignment": str(assignment_path),
            "layer_config": str(config_path),
        },
    }
    with open(runtime.output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _emit("===== summary =====")
    _emit(
        "Quality-equivalent target: "
        f"{reference_bpp:.2f} bpp KL={reference.validation_kl:.4g} "
        f"+ slack={float(args.quality_equivalent_kl_slack):.4g} "
        f"=> threshold={threshold_kl:.4g}"
    )
    _emit(
        f"Quality-equivalent bpp: {chosen.target_bpp:.2f} "
        f"(achieved={chosen.achieved_bpp:.4f}, KL={chosen.validation_kl:.4g})"
    )
    if knee_meta is not None:
        _emit(
            "Evaluated-point knee candidate: "
            f"{knee_meta['chosen_bpp']:.2f} bpp "
            f"(KL={knee_meta['validation_kl']:.4g})"
        )
    _emit(f"Quality-equivalent assignment: {assignment_path}")
    _emit(f"Pareto CSV: {csv_path}")
    _emit("===================")
    return 0


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
    accepted: bool | None = None,
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
    accepted_value = (not regression) if accepted is None else bool(accepted)
    return {
        "l3_enabled": True,
        "enabled": True,
        "accepted": accepted_value,
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
    perturbed_cache: PerturbedActivationCache | None = None,
    use_frozen_weight_cache: bool = True,
    rng_seed: int | None = 0,
) -> float:
    device = next(model.parameters()).device
    calib_ids = _prepare_kl_tensor_inputs(calib_ids, device)
    ref_log_probs = _prepare_ref_log_probs_for_kl(ref_log_probs, device)
    if use_frozen_weight_cache and not _env_flag_enabled(
        "PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE",
        default=True,
    ):
        use_frozen_weight_cache = False
    hooks = perturbed_cache
    if hooks is None:
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
    values = []
    use_cuda_graphs = _env_cuda_graphs_enabled_for_call_count(
        "PRISMAQUANT_KL_CUDA_GRAPHS",
        default="auto",
        call_count=int(calib_ids.size(0)),
        min_calls=16,
    )
    graph_key = (
        id(model),
        assignment_hash(assignment),
        bool(use_frozen_weight_cache),
        rng_seed,
    )
    cache_cm = nullcontext()
    if use_frozen_weight_cache and hooks._frozen_weight_cache is None:
        cache_cm = hooks.frozen_weight_cache()
    rng_devices = []
    if rng_seed is not None and device.type == "cuda" and torch.cuda.is_available():
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    rng_cm = (
        torch.random.fork_rng(devices=rng_devices)
        if rng_seed is not None else nullcontext()
    )
    installed_here = not hooks.installed
    with cache_cm:
        materialized_cm = nullcontext()
        if (
            use_cuda_graphs
            and use_frozen_weight_cache
            and device.type == "cuda"
            and torch.cuda.is_available()
            and hooks._frozen_weight_cache is not None
        ):
            materialized_cm = hooks.materialized_frozen_weights()
        with materialized_cm:
            if installed_here:
                hooks.install()
            try:
                with rng_cm:
                    if rng_seed is not None:
                        torch.manual_seed(int(rng_seed))
                        if device.type == "cuda" and torch.cuda.is_available():
                            torch.cuda.manual_seed_all(int(rng_seed))
                    for i in range(calib_ids.size(0)):
                        batch = calib_ids[i:i + 1].to(device)
                        def _forward(batch_ids):
                            return model(batch_ids).logits[:, -1:, :].clone()

                        # With PRISMAQUANT_GRAPH_OUTPUT_CLONE=0 this returns
                        # the graph's static logits tensor. It is consumed by
                        # KL immediately and not retained across replays.
                        logits = _KL_CUDA_GRAPH_REGISTRY.run(
                            "assignment-kl-forward",
                            graph_key,
                            _forward,
                            batch,
                            enabled=use_cuda_graphs,
                            device=device,
                            keepalive=(hooks,),
                        )
                        teacher = ref_log_probs[i][:, -1:, :]
                        values.append(float(kl_divergence(logits, teacher).item()))
            finally:
                if installed_here:
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
    ap.add_argument("--target-bits", type=float)
    ap.add_argument("--target-bits-list")
    ap.add_argument("--target-bits-share-tolerance", type=float, default=0.25)
    ap.add_argument("--target-bits-anchor", type=float)
    ap.add_argument("--knee-search", action="store_true")
    ap.add_argument("--knee-bpp-min", type=float, default=4.0)
    ap.add_argument("--knee-bpp-max", type=float, default=8.0)
    ap.add_argument("--knee-tolerance", type=float, default=0.1)
    ap.add_argument("--knee-mode", choices=["kneedle", "threshold"], default="kneedle")
    ap.add_argument("--knee-threshold-kl", type=float)
    ap.add_argument(
        "--knee-kl-noise-floor",
        type=float,
        default=DEFAULT_KNEE_KL_NOISE_FLOOR,
        help=(
            "Measured-KL improvement required before a higher-bpp point "
            "extends the nondominated knee frontier."
        ),
    )
    ap.add_argument(
        "--knee-unstable-policy",
        choices=["best-kl", "warn"],
        default="best-kl",
        help=(
            "Policy when leave-one-out diagnostics show an unstable geometric "
            "knee. best-kl selects the lowest measured-KL frontier point; "
            "warn keeps the geometric knee and only records the warning."
        ),
    )
    ap.add_argument("--knee-max-evaluations", type=int, default=12)
    ap.add_argument(
        "--knee-initial-points",
        type=int,
        default=9,
        help=(
            "Initial evenly spaced real-KL probes for --knee-mode kneedle. "
            "The search then refines the nondominated achieved-bpp frontier."
        ),
    )
    ap.add_argument(
        "--knee-seed-frontier",
        "--knee-resume-frontier",
        action="append",
        default=[],
        help=(
            "JSON file with already validated knee/pareto points to seed "
            "normal --knee-search. Supports validated_frontier.json, "
            "validated_summary.json, and pareto_curve.json."
        ),
    )
    ap.add_argument(
        "--knee-seed-remeasure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remeasure seeded frontier assignments on the current calibration "
            "before using them in kneedle selection."
        ),
    )
    ap.add_argument(
        "--knee-surrogate",
        action="store_true",
        help=(
            "Legacy side path: generate candidate configs from one cached L3 "
            "cost table. The normal kneedle path uses measured validation KL."
        ),
    )
    ap.add_argument(
        "--knee-surrogate-points",
        type=int,
        default=41,
        help="Number of evenly spaced CPU surrogate frontier points to solve.",
    )
    ap.add_argument(
        "--knee-surrogate-bpp-step",
        type=float,
        help="Optional fixed bpp grid step for --knee-surrogate.",
    )
    ap.add_argument(
        "--knee-surrogate-neighbors",
        type=int,
        default=1,
        help="Write layer configs for this many frontier neighbors around the knee.",
    )
    ap.add_argument(
        "--knee-surrogate-validate",
        action="store_true",
        help=(
            "After generating the surrogate frontier, load the model and "
            "validate selected candidates with real KL before choosing the knee."
        ),
    )
    ap.add_argument(
        "--allow-unsafe-surrogate-knee",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--knee-surrogate-validation-candidates",
        type=int,
        default=9,
        help=(
            "Number of evenly spread surrogate frontier candidates to write "
            "and validate when --knee-surrogate-validate is set."
        ),
    )
    ap.add_argument(
        "--knee-include-assignment",
        action="append",
        default=[],
        help=(
            "Additional assignment JSON to validate during "
            "--knee-surrogate-validate. May be repeated."
        ),
    )
    ap.add_argument(
        "--quality-equivalent-bits",
        "--quality-equivalent-to-bits",
        dest="quality_equivalent_bits",
        type=float,
        help=(
            "Find the smallest bpp whose validated KL is no worse than the "
            "validated KL of this reference bpp plus --quality-equivalent-kl-slack."
        ),
    )
    ap.add_argument("--quality-equivalent-bpp-min", type=float, default=4.0)
    ap.add_argument("--quality-equivalent-bpp-tolerance", type=float, default=0.1)
    ap.add_argument("--quality-equivalent-max-evaluations", type=int, default=12)
    ap.add_argument("--quality-equivalent-kl-slack", type=float, default=0.0)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--initial-config")
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--convergence-frac", type=float, default=0.01)
    ap.add_argument("--ema-decay", type=float, default=0.5)
    ap.add_argument("--input-rows", type=int, default=256)
    ap.add_argument("--n-calib-samples", type=int, default=8)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument(
        "--calib-split",
        default="validation",
        help=(
            "Wikitext split used for measured KL scoring. Defaults to "
            "validation so validation_kl is disjoint from L3 cost calibration."
        ),
    )
    ap.add_argument(
        "--calib-seed",
        type=int,
        default=4242,
        help="Stable random seed for measured KL calibration windows.",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument(
        "--memory-aware-l3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Clamp calibration, lane batching, and validation-scout breadth "
            "based on tracked model size and current CUDA memory."
        ),
    )
    ap.add_argument("--bit-precision", type=float, default=0.001)
    ap.add_argument("--cost-mode", default="auto", choices=["auto", "unbatched", "batched"])
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--h-detail-dir")
    ap.add_argument("--l3-polish", action="store_true",
                    help="Run propagated-cost L3 polish after L2 convergence.")
    ap.add_argument(
        "--l3-mode",
        default="selective",
        choices=["selective", "global"],
        help="L3 polish mode: bounded neighborhood or all eligible Linears.",
    )
    ap.add_argument("--l3-uncertainty-rel-tol", type=float, default=0.10)
    ap.add_argument("--l3-min-fraction", type=float, default=0.05)
    ap.add_argument("--l3-max-fraction", type=float, default=0.30)
    ap.add_argument("--l3-safety-fraction", type=float, default=0.02)
    ap.add_argument("--l3-max-lanes-per-batch", type=int, default=64)
    ap.add_argument(
        "--l3-tail-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use decoder tail-only L3 propagation when the model supports it.",
    )
    ap.add_argument(
        "--l3-output-mse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also collect downstream output-MSE diagnostics during L3. "
            "Disabled by default because allocation uses propagated end-KL."
        ),
    )
    ap.add_argument(
        "--l3-measure-all-formats",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Measure every available L3 format instead of the L2-neighbor "
            "filter. Defaults to auto: enabled for wide multi-target spans."
        ),
    )
    ap.add_argument(
        "--resume-l3-costs",
        help=(
            "Load cached L3 costs from one pickle, or comma-separated "
            "anchor_bpp_X.XX=/path/l3_propagated_costs.pkl pairs."
        ),
    )
    ap.add_argument(
        "--resume-l3-costs-dir",
        help=(
            "Directory containing l3_propagated_costs.pkl or "
            "anchor_bpp_X.XX/l3_propagated_costs.pkl files."
        ),
    )
    ap.add_argument(
        "--no-resume-on-mismatch",
        dest="resume_l3_ignore_mismatch",
        action="store_true",
        help="Force L3 resume despite metadata mismatch, with a warning.",
    )
    ap.add_argument(
        "--force-resume-l3-costs",
        dest="force_resume_l3_costs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--l3-n-calib-samples", type=int, default=4)
    ap.add_argument("--l3-calib-seqlen", type=int, default=256)
    ap.add_argument(
        "--l3-calib-split",
        default="train",
        help="Wikitext split used for L3 propagated-cost candidate generation.",
    )
    ap.add_argument(
        "--l3-calib-seed",
        type=int,
        default=42,
        help="Stable random seed for L3 calibration windows.",
    )
    ap.add_argument(
        "--l3-regression-tolerance",
        type=float,
        default=0.0,
        help=(
            "Relative KL regression allowed during L3 acceptance. The default "
            "0.0 keeps the polish loop strictly monotonic under measured KL."
        ),
    )
    ap.add_argument(
        "--l3-iter-max",
        type=int,
        default=3,
        help="Max iterations of L3 trust-region polish loop.",
    )
    ap.add_argument(
        "--l3-hamming-cap-init",
        type=int,
        default=8,
        help="Initial Hamming-distance cap on L3 DP per iteration.",
    )
    ap.add_argument(
        "--l3-hamming-cap-max",
        type=int,
        default=64,
        help="Maximum Hamming cap after expansion on successful L3 steps.",
    )
    ap.add_argument(
        "--l3-coord-descent-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run coordinate-descent greedy after iterated L3.",
    )
    ap.add_argument(
        "--l3-coord-descent-max-passes",
        type=int,
        default=1,
        help="Max coordinate-descent passes after iterated L3.",
    )
    ap.add_argument(
        "--l3-validation-scout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run a bounded validation-backed one-flip scout after L3. "
            "This logs per-candidate outcomes for marginal-value stopping."
        ),
    )
    ap.add_argument(
        "--l3-validation-scout-rounds",
        type=int,
        default=1,
        help="Maximum validation-scout rounds; each round can commit one best flip.",
    )
    ap.add_argument(
        "--l3-validation-scout-max-candidates",
        type=int,
        default=64,
        help="Maximum validation-scout candidates per round.",
    )
    ap.add_argument(
        "--l3-validation-scout-max-lanes-per-batch",
        type=int,
        default=1,
        help=(
            "Maximum simultaneous KL lanes for validation scout. The default "
            "keeps scout probes restartable and memory-stable during knee search."
        ),
    )
    ap.add_argument(
        "--l3-validation-scout-cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow CUDA graph capture inside validation scout. Disabled by "
            "default because scout shapes vary across candidates."
        ),
    )
    ap.add_argument(
        "--l3-validation-scout-sample-per-bucket",
        type=int,
        default=0,
        help="Additional stable-random validation-scout samples per rank bucket.",
    )
    ap.add_argument(
        "--l3-validation-scout-seed",
        type=int,
        default=0,
        help="Stable random seed for validation-scout exploration samples.",
    )
    ap.add_argument(
        "--l3-validation-scout-bpp-slack",
        type=float,
        default=0.25,
        help="Allowed bpp above target for validation-scout trial flips.",
    )
    ap.add_argument(
        "--l3-validation-scout-max-predicted-delta",
        type=float,
        default=0.0,
        help=(
            "Largest L3-predicted KL delta eligible for validation scout; "
            "0 means only predicted non-worsening flips."
        ),
    )
    ap.add_argument(
        "--l3-validation-scout-min-improvement",
        type=float,
        default=0.0,
        help="Minimum KL gain required for a validation-scout hit/commit.",
    )
    ap.add_argument(
        "--l3-validation-scout-commit-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Commit the best validated validation-scout flip in each round.",
    )
    ap.add_argument(
        "--search-telemetry-jsonl",
        help=(
            "Path for candidate-search telemetry JSONL. Defaults to "
            "<output-dir>/search_telemetry.jsonl."
        ),
    )
    ap.add_argument(
        "--l3-interaction-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Measure sparse pairwise L3 interaction residuals near the DP "
            "proposal and run a bounded local quadratic refiner."
        ),
    )
    ap.add_argument(
        "--l3-interaction-top-units",
        type=int,
        default=8,
        help="Maximum Linear units included in sparse L3 interaction refinement.",
    )
    ap.add_argument(
        "--l3-interaction-neighbor-radius",
        type=int,
        default=1,
        help="Format-neighborhood radius around each selected unit's current format.",
    )
    ap.add_argument(
        "--l3-interaction-max-pairs",
        type=int,
        default=0,
        help=(
            "Cap paired override measurements for interaction refinement; "
            "0 means no explicit cap."
        ),
    )
    ap.add_argument(
        "--l3-interaction-max-seconds",
        type=float,
        default=600.0,
        help=(
            "Wall-time budget for sparse interaction-refine pair measurement. "
            "Checked between chunks; <=0 disables the time cap."
        ),
    )
    ap.add_argument(
        "--l3-interaction-max-lanes-per-batch",
        type=int,
        default=16,
        help=(
            "Maximum live lanes for paired interaction measurement. Pairwise "
            "overrides have a higher transient memory footprint than unary L3, "
            "so this defaults lower than --l3-max-lanes-per-batch."
        ),
    )
    ap.add_argument(
        "--l3-interaction-max-passes",
        type=int,
        default=4,
        help="Max local-search passes for the sparse quadratic L3 refiner.",
    )
    ap.add_argument(
        "--l3-interaction-solver",
        choices=("budget", "lagrangian", "qubo"),
        default="budget",
        help=(
            "Objective for sparse interaction refinement. 'budget' minimizes "
            "KL under the target bit budget. 'lagrangian'/'qubo' minimize "
            "KL + lambda*bpp and can move along the Pareto frontier."
        ),
    )
    ap.add_argument(
        "--l3-interaction-lagrangian-bpp-penalty",
        type=float,
        default=0.0,
        help=(
            "Lambda for --l3-interaction-solver=lagrangian/qubo, in KL units "
            "per additional bpp."
        ),
    )
    ap.add_argument(
        "--l3-interaction-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exactly solve measured sparse interaction subproblems when the "
            "option-state count is below --l3-interaction-exact-max-states."
        ),
    )
    ap.add_argument(
        "--l3-interaction-exact-max-states",
        type=int,
        default=2_000_000,
        help=(
            "Maximum option combinations to enumerate exactly in the sparse "
            "interaction refiner; larger subproblems fall back to greedy local search."
        ),
    )
    ap.add_argument(
        "--coord-descent-early-stop-streak",
        type=int,
        default=50,
        help="Halt coord descent after this many consecutive non-improving flips",
    )
    ap.add_argument(
        "--frozen-dp-budget-tolerance",
        type=float,
        default=0.05,
        help="Fraction of total target bits allowed as L3 frozen-DP slack.",
    )
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-Linear per-format costs each iteration.")
    args = ap.parse_args(argv)
    budget_mode_count = sum(
        [
            args.target_bits is not None,
            args.target_bits_list is not None,
            bool(args.knee_search),
            args.quality_equivalent_bits is not None,
        ]
    )
    if budget_mode_count != 1:
        ap.error(
            "Specify exactly one of --target-bits, --target-bits-list, "
            "--knee-search, or --quality-equivalent-bits."
        )
    if (
        args.knee_search
        and args.knee_mode == "threshold"
        and args.knee_threshold_kl is None
    ):
        ap.error("--knee-threshold-kl is required with --knee-mode threshold.")
    if args.target_bits_list is not None:
        try:
            parse_target_bits_list(args.target_bits_list)
        except ValueError as exc:
            ap.error(str(exc))
    if args.target_bits_share_tolerance < 0:
        ap.error("--target-bits-share-tolerance must be non-negative.")
    if args.knee_search:
        if args.knee_bpp_min >= args.knee_bpp_max:
            ap.error("--knee-bpp-min must be less than --knee-bpp-max.")
        if args.knee_max_evaluations < 3:
            ap.error("--knee-max-evaluations must be at least 3.")
        if args.knee_initial_points < 3:
            ap.error("--knee-initial-points must be at least 3.")
        if args.knee_surrogate and args.knee_surrogate_points < 3:
            ap.error("--knee-surrogate-points must be at least 3.")
        if (
            args.knee_surrogate
            and args.knee_surrogate_bpp_step is not None
            and args.knee_surrogate_bpp_step <= 0
        ):
            ap.error("--knee-surrogate-bpp-step must be positive.")
        if args.knee_surrogate_validation_candidates < 0:
            ap.error("--knee-surrogate-validation-candidates must be non-negative.")
    if args.quality_equivalent_bits is not None:
        if args.quality_equivalent_bpp_min >= args.quality_equivalent_bits:
            ap.error(
                "--quality-equivalent-bpp-min must be less than "
                "--quality-equivalent-bits."
            )
        if args.quality_equivalent_bpp_tolerance <= 0:
            ap.error("--quality-equivalent-bpp-tolerance must be positive.")
        if args.quality_equivalent_max_evaluations < 3:
            ap.error("--quality-equivalent-max-evaluations must be at least 3.")
        if args.quality_equivalent_kl_slack < 0:
            ap.error("--quality-equivalent-kl-slack must be non-negative.")
    if (
        args.target_bits_list is not None
        or args.knee_search
        or args.quality_equivalent_bits is not None
    ) and not args.l3_polish:
        ap.error(
            "--target-bits-list, --knee-search, and "
            "--quality-equivalent-bits require --l3-polish."
        )
    if (args.resume_l3_costs or args.resume_l3_costs_dir) and not args.l3_polish:
        ap.error("--resume-l3-costs requires --l3-polish.")
    try:
        resume_single, resume_by_anchor = _parse_resume_l3_costs(
            args.resume_l3_costs
        )
    except ValueError as exc:
        ap.error(str(exc))
    args._resume_l3_costs_single = resume_single
    args._resume_l3_costs_by_anchor = resume_by_anchor
    work_root = Path(args.work_dir)
    output_root = Path(args.output_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    iteration_trace_path = output_root / "iteration_trace.jsonl"
    l3_trace_path = output_root / "l3_polish_trace.jsonl"
    search_telemetry_path = (
        Path(args.search_telemetry_jsonl)
        if args.search_telemetry_jsonl
        else output_root / "search_telemetry.jsonl"
    )
    args._search_telemetry_path = search_telemetry_path
    iteration_trace_path.write_text("")
    l3_trace_path.write_text("")
    search_telemetry_path.write_text("")

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
    if args.knee_search and args.knee_surrogate:
        if not args.knee_surrogate_validate and not args.allow_unsafe_surrogate_knee:
            raise RuntimeError(
                "--knee-surrogate writes a surrogate-only knee candidate. "
                "Pass --knee-surrogate-validate to select the knee by real KL "
                "or --allow-unsafe-surrogate-knee for diagnostic/internal use."
            )
        run_surrogate_knee_search(
            args,
            stats=stats,
            current_costs=current_costs,
            specs=specs,
            output_root=output_root,
        )
        if not args.knee_surrogate_validate:
            return 0

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
    model_device = next(model.parameters()).device
    _apply_memory_aware_runtime_policy(args, stats, model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    calib_ids = load_wikitext_calibration(
        tokenizer,
        args.n_calib_samples,
        args.calib_seqlen,
        split=args.calib_split,
        seed=args.calib_seed,
    )
    calib_ids = _prepare_kl_tensor_inputs(calib_ids, model_device)
    l3_calib_ids = calib_ids
    if args.l3_polish and (
        args.l3_n_calib_samples != args.n_calib_samples
        or args.l3_calib_seqlen != args.calib_seqlen
        or args.l3_calib_split != args.calib_split
        or int(args.l3_calib_seed) != int(args.calib_seed)
    ):
        l3_calib_ids = load_wikitext_calibration(
            tokenizer,
            args.l3_n_calib_samples,
            args.l3_calib_seqlen,
            split=args.l3_calib_split,
            seed=args.l3_calib_seed,
        )
        l3_calib_ids = _prepare_kl_tensor_inputs(l3_calib_ids, model_device)
    ref_log_probs = cache_reference_log_probs(model, calib_ids, model_device)
    ref_log_probs = _prepare_ref_log_probs_for_kl(ref_log_probs, model_device)
    runtime = BudgetRuntime(
        work_root=work_root,
        output_root=output_root,
        stats=stats,
        current_costs=current_costs,
        specs=specs,
        profile=profile,
        model=model,
        calib_ids=calib_ids,
        l3_calib_ids=l3_calib_ids,
        ref_log_probs=ref_log_probs,
        dtype=dtype,
        probe_load_timing=probe_load_timing,
    )
    args._runtime = runtime
    if args.target_bits_list is not None:
        return run_multi_budget(args, runtime)
    if args.knee_search and args.knee_surrogate_validate:
        return run_validated_surrogate_knee_search(args, runtime)
    if args.knee_search:
        return run_knee_search(args, runtime)
    if args.quality_equivalent_bits is not None:
        return run_quality_equivalent_search(args, runtime)

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
        # PRISMAQUANT_L2_CUDA_GRAPHS is not wired into the L2 cost kernel path
        # here: run_cost_pass measures many per-Linear/per-chunk shapes and
        # returns host scalar tables, so safe graphing needs a separate
        # per-shape output-buffer API rather than wrapping this mixed loop.
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
        _report_graph_memory(f"[l2] iteration {iteration} after")
        if verdict == "converged":
            convergence_reached = True
            break

    _phase_boundary_cleanup("single_budget_l2_to_l3")
    _report_graph_memory("single_budget_l2_to_l3")
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
        # exposed because each propagated measurement is still expensive even
        # when tail-only propagation is available.
        l2_assignment = dict(assignment)
        _report_graph_memory("[l3] validation KL before start")
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
        _report_graph_memory("[l3] validation KL before done")
        selection_start = _phase_start("[l3] selection")
        n_total = len(set(stats) & set(l2_assignment))
        try:
            if args.l3_mode == "global":
                selected = build_global_l3_neighborhood(
                    stats,
                    latest_smoothed_costs,
                    l2_assignment,
                    specs,
                )
            else:
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
        selected = _filter_l3_neighborhood_entries(selected)
        selection_timing = _phase_end("[l3] selection", selection_start)
        n_uncertain = sum(1 for entry in selected if "uncertain" in entry.reasons)
        n_safety = sum(1 for entry in selected if "high_l2_cost" in entry.reasons)
        if args.l3_mode == "global":
            _emit(
                f"[l3] global mode: measuring {len(selected)} eligible "
                f"Linears from {n_total} total Linears"
            )
        else:
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
        l3_lane_count = sum(
            len([fmt for fmt in entry.formats if fmt != "BF16"]) + 1
            for entry in selected
        )
        avg_formats = (
            sum(len(entry.formats) for entry in selected) / float(len(selected))
            if selected else 0.0
        )
        resume_path = _resume_l3_cost_path_for_anchor(
            args,
            float(args.target_bits),
            single_output=True,
        )
        resumed_l3: L3ResumeCosts | None = None
        if resume_path is not None:
            resumed_l3 = _load_l3_resume_costs(
                resume_path,
                args=args,
                runtime=runtime,
                anchor_bpp=float(args.target_bits),
                prefix="[l3]",
            )
            _emit(
                f"[l3] loaded L3 costs from {resumed_l3.path}; "
                "skipping measurement"
            )
        else:
            _emit(
                f"[l3] measuring propagated costs: {len(selected)} candidates x "
                f"{avg_formats:.2f} formats = {l3_lane_count} total lanes"
            )

        polish_run = run_iterated_l3_polish(
            args,
            model,
            l2_assignment,
            kl_before,
            stats,
            latest_smoothed_costs,
            specs,
            args.target_bits,
            calib_ids,
            l3_calib_ids,
            ref_log_probs,
            work_root=work_root,
            profile=profile,
            initial_selected=selected,
            initial_l3_costs=(resumed_l3.costs if resumed_l3 is not None else None),
            resume_l3_cost_history=(
                resumed_l3.cost_history if resumed_l3 is not None else None
            ),
            coord_anchor_label=_bpp_label(args.target_bits),
        )
        l3_costs = polish_run.smoothed_l3_costs
        l3_elapsed_seconds = float(polish_run.measurement_timing["elapsed_seconds"])
        l3_peak_gb = polish_run.measurement_timing.get("cuda_peak_gb")
        l3_measure_timing = dict(polish_run.measurement_timing)
        l3_cost_path = output_root / "l3_propagated_costs.pkl"
        with open(l3_cost_path, "wb") as f:
            pickle.dump(
                {
                    "costs": l3_costs,
                    "cost_history": polish_run.l3_cost_history,
                    "formats": [s.name for s in specs],
                    "meta": _l3_cost_payload_meta(
                        args,
                        runtime,
                        anchor_bpp=float(args.target_bits),
                        selected_count=len(selected),
                        extra={
                            "paired_baseline": "target_bf16_under_l2_assignment",
                            "l3_mode": args.l3_mode,
                            "total_count": len(selected),
                            "model_linear_count": n_total,
                            "tail_only": bool(args.l3_tail_only),
                            "l3_max_lanes_per_batch": int(args.l3_max_lanes_per_batch),
                            "l3_n_calib_samples": int(args.l3_n_calib_samples),
                            "l3_calib_seqlen": int(args.l3_calib_seqlen),
                            "l3_iter_max": int(args.l3_iter_max),
                            "l3_iterations": len(polish_run.iterations),
                            "l3_hamming_cap_init": int(args.l3_hamming_cap_init),
                            "l3_hamming_cap_max": int(args.l3_hamming_cap_max),
                            "coord_descent_fired": bool(polish_run.coord_descent_fired),
                            "coord_descent_flips": int(polish_run.coord_descent_flips),
                            "resumed_from": (
                                str(resumed_l3.path)
                                if resumed_l3 is not None else None
                            ),
                        },
                    ),
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
        polished_assignment = dict(polish_run.proposed_assignment)
        accepted_assignment = dict(polish_run.assignment)
        kl_after = float(polish_run.proposed_kl)
        final_kl = float(polish_run.final_kl)
        frozen_dp_meta = dict(polish_run.frozen_dp_meta)
        frozen_solve_timing = dict(polish_run.dp_timing)
        kl_after_timing = dict(polish_run.validation_timing)
        polish_accepted = bool(polish_run.accepted)
        regression = bool(polish_run.regression)
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
        l3_kl_improvement_pct = (
            (float(kl_before) - float(final_kl)) / abs(float(kl_before)) * 100.0
            if abs(float(kl_before)) > 1e-12 else 0.0
        )
        _emit(
            f"[l3] validating: KL_before={kl_before:.4e}, "
            f"KL_after={final_kl:.4e}, regression={str(regression).lower()}, "
            f"improvement={l3_kl_improvement_pct:.2f}%"
        )
        if not polish_accepted:
            delta_pct = (
                (float(kl_after) - float(kl_before)) / abs(float(kl_before)) * 100.0
                if abs(float(kl_before)) > 1e-12 else 0.0
            )
            _emit(
                f"[l3] regression detected (kl_before={kl_before:.4e}, "
                f"kl_after={kl_after:.4e}, delta=+{delta_pct:.2f}%); "
                "REJECTING polish, falling back to L2 assignment."
            )
        l3_flip_count = sum(
            1
            for name in set(l2_assignment) | set(accepted_assignment)
            if l2_assignment.get(name) != accepted_assignment.get(name)
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
            accepted=polish_accepted,
        )
        l3_summary["l3_mode"] = args.l3_mode
        l3_summary["total_count"] = len(selected)
        l3_summary["model_linear_count"] = n_total
        l3_summary["accepted_assignment"] = (
            "polished" if polish_accepted else "l2"
        )
        l3_summary["accepted_flip_count"] = int(l3_flip_count)
        l3_summary["regression_tolerance"] = float(args.l3_regression_tolerance)
        l3_summary["cost_path"] = str(l3_cost_path)
        l3_summary["final_kl"] = float(final_kl)
        l3_summary["l3_iterations"] = len(polish_run.iterations)
        l3_summary["l3_iteration_trace"] = polish_run.iterations
        l3_summary["l3_hamming_cap_init"] = int(args.l3_hamming_cap_init)
        l3_summary["l3_hamming_cap_max"] = int(args.l3_hamming_cap_max)
        l3_summary["l3_cycle_detected"] = bool(polish_run.cycle_detected)
        l3_summary["coord_descent_fired"] = bool(polish_run.coord_descent_fired)
        l3_summary["coord_descent_flips"] = int(polish_run.coord_descent_flips)
        l3_summary["coord_descent_passes"] = int(polish_run.coord_descent_passes)
        l3_summary["validation_scout"] = polish_run.validation_scout
        l3_summary["frozen_dp_attempts"] = frozen_dp_meta.get("frozen_dp_attempts")
        l3_summary["frozen_dp_greedy"] = frozen_dp_meta.get("frozen_dp_greedy")
        l3_summary["frozen_dp_budget_tolerance"] = frozen_dp_meta.get(
            "frozen_dp_budget_tolerance",
            args.frozen_dp_budget_tolerance,
        )
        l3_summary["frozen_dp_budget_tolerance_bits"] = frozen_dp_meta.get(
            "frozen_dp_budget_tolerance_bits",
        )
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
                    "l3_mode": args.l3_mode,
                    "format_candidates": list(entry.formats),
                    "propagated_end_kl": propagated,
                    "downstream_output_mse": downstream,
                    "l2_disagreement": disagreements,
                    "accepted_flip": (
                        l2_assignment.get(entry.name)
                        != accepted_assignment.get(entry.name)
                    ),
                    "from": l2_assignment.get(entry.name),
                    "to": accepted_assignment.get(entry.name),
                    "proposed_to": polished_assignment.get(entry.name),
                    "l3_iterations": len(polish_run.iterations),
                    "coord_descent_flips": int(polish_run.coord_descent_flips),
                    "timing": l3_summary["timing"],
                },
            )
        assignment = dict(accepted_assignment)

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
        "search_telemetry": str(search_telemetry_path),
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
        proposed_flips = l3_summary.get("flip_count") if l3_summary_path else None
        if proposed_flips is not None:
            _emit(f"L3 flips: {l3_flip_count} accepted / {proposed_flips} proposed")
        else:
            _emit(f"L3 flips: {l3_flip_count}")
        if l3_summary_path:
            _emit(
                f"L3 iterations: {l3_summary.get('l3_iterations', 0)} "
                f"coord_flips={l3_summary.get('coord_descent_flips', 0)}"
            )
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
