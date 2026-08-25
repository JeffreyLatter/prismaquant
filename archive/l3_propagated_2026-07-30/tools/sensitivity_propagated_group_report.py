#!/usr/bin/env python3
"""Measure propagated KL for MSE-promotion groups.

This is a report-only diagnostic.  It does not select or write a promoted
assignment; it compares each target group at BF16 against the same group at its
current production-rendered format under the fixed surrounding assignment.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant.calibration_data import _dtype_from_name
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.kl_measurement import measure_override_paired_kl_deltas
from prismaquant.layer_config import load_assignment
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.mse_promotion import build_promotion_candidate_report
from prismaquant.perturbed_x_cache import load_text_model_under_work_root
from prismaquant.sensitivity_probe import load_calibration


def _load_pickle_mapping(path: str | Path, key: str) -> Mapping[str, object]:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a mapping")
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} did not contain mapping key {key!r}")
    return value


def _load_costs(path: str | Path) -> Mapping[str, object]:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a mapping")
    costs = payload.get("costs")
    if isinstance(costs, Mapping):
        return costs
    return payload


def _load_production_cache(
    path: str | Path,
    *,
    cache_dir_override: str | None,
    lru_gb: float,
):
    with Path(path).open("rb") as fh:
        cache = pickle.load(fh)
    if cache_dir_override and hasattr(cache, "relocate"):
        cache.relocate(cache_dir_override)
    if float(lru_gb) > 0 and hasattr(cache, "enable_lru"):
        cache.enable_lru(int(float(lru_gb) * 1024**3))
    return cache


def _parse_categories(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _finite_ratio(num: float, den: float) -> float | None:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1e-30:
        return None
    return float(num / den)


def _prefetch_production_cache(cache, assignment: Mapping[str, str], args) -> dict | None:
    if cache is None or args.production_cache_prefetch == "off":
        return None
    if args.production_cache_prefetch == "file-pages":
        if not hasattr(cache, "prefetch_assignment_file_pages"):
            raise RuntimeError(
                "production cache does not support file-page prefetch"
            )
        return cache.prefetch_assignment_file_pages(
            assignment,
            mode="require",
            max_resident_bytes=(
                int(float(args.production_cache_file_prefetch_max_gb) * 1024**3)
                if float(args.production_cache_file_prefetch_max_gb) > 0
                else None
            ),
            headroom_gb=float(args.production_cache_file_prefetch_headroom_gb),
            max_workers=int(args.production_cache_prefetch_workers),
            progress=True,
            log_prefix="[group-sensitivity/prod-cache-files]",
        )
    if args.production_cache_prefetch == "load":
        if not hasattr(cache, "prefetch_assignment"):
            raise RuntimeError("production cache does not support assignment preload")
        return cache.prefetch_assignment(
            assignment,
            max_resident_bytes=(
                int(float(args.production_cache_load_max_gb) * 1024**3)
                if float(args.production_cache_load_max_gb) > 0
                else None
            ),
            max_workers=int(args.production_cache_prefetch_workers),
            require=True,
            progress=True,
            log_prefix="[group-sensitivity/prod-cache]",
        )
    raise ValueError(f"unknown production prefetch mode {args.production_cache_prefetch!r}")


def _progress(event: dict) -> None:
    name = event.get("event", "event")
    if name.endswith("_start"):
        print(
            "[group-sensitivity] "
            f"{name} {event.get('chunk_index')}/{event.get('chunk_count')} "
            f"lanes={event.get('lane_count')} overrides={event.get('override_count')}",
            flush=True,
        )
    elif name.endswith("_end"):
        print(
            "[group-sensitivity] "
            f"{name} {event.get('chunk_index')}/{event.get('chunk_count')} "
            f"batches={event.get('batch_count')} "
            f"dt={float(event.get('elapsed_seconds', 0.0)):.1f}s",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure production-cache-faithful propagated KL for semantic "
            "MSE-promotion groups."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-assignment", required=True)
    parser.add_argument("--costs", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--production-weight-cache", required=True)
    parser.add_argument("--production-cache-dir-override", default=None)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--dataset",
        default="/home/rob/dq-runs/calibration/diverse-v1.jsonl",
    )
    parser.add_argument("--n-calib-samples", type=int, default=8)
    parser.add_argument("--calib-seqlen", type=int, default=512)
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument(
        "--categories",
        default="linear_attn,self_attn,shared_expert",
        help="Comma-separated semantic categories to measure.",
    )
    parser.add_argument("--target-format", default="BF16")
    parser.add_argument(
        "--group-by",
        default="serving_unit",
        choices=("name", "serving_unit", "fused_unit", "layer_category", "category"),
        help="Serving-unit probes respect fused siblings by default.",
    )
    parser.add_argument(
        "--metric",
        default="output_mse_per_bit",
        choices=(
            "output_mse_per_bit",
            "output_mse",
            "predicted_dloss_per_bit",
            "weight_mse_per_bit",
        ),
    )
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-lanes-per-batch", type=int, default=8)
    parser.add_argument("--no-tail-only", action="store_true")
    parser.add_argument("--no-cache-tail-layer-inputs", action="store_true")
    parser.add_argument("--no-activation-quant", action="store_true")
    parser.add_argument(
        "--frozen-context-cache",
        action="store_true",
        help=(
            "Opt into building the GPU frozen context cache. Default off for "
            "large production-cache diagnostics to avoid whole-assignment "
            "materialization."
        ),
    )
    parser.add_argument("--allow-rtn-fallback", action="store_true")
    parser.add_argument(
        "--production-cache-prefetch",
        default="file-pages",
        choices=("off", "file-pages", "load"),
    )
    parser.add_argument("--production-cache-lru-gb", type=float, default=16.0)
    parser.add_argument("--production-cache-prefetch-workers", type=int, default=4)
    parser.add_argument("--production-cache-file-prefetch-max-gb", type=float, default=0.0)
    parser.add_argument("--production-cache-file-prefetch-headroom-gb", type=float, default=24.0)
    parser.add_argument("--production-cache-load-max-gb", type=float, default=0.0)
    args = parser.parse_args(argv)

    require_cuda_hot_path("sensitivity_propagated_group_report", args.device)

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    work_root = (
        Path(args.work_root)
        if args.work_root
        else output_report.parent / "group_sensitivity_work"
    )
    work_root.mkdir(parents=True, exist_ok=True)

    try:
        profile = detect_profile(args.model)
    except Exception:
        profile = DefaultProfile()

    assignment = load_assignment(args.base_assignment)
    costs = _load_costs(args.costs)
    stats = _load_pickle_mapping(args.probe, "stats")
    candidate_payload = build_promotion_candidate_report(
        assignment,
        costs=costs,
        stats=stats,
        categories=_parse_categories(args.categories),
        target_format=args.target_format,
        group_by=args.group_by,
        metric=args.metric,
        profile=profile,
    )
    candidates = list(candidate_payload["candidates"])
    if int(args.max_groups) > 0:
        candidates = candidates[: int(args.max_groups)]
    overrides_by_key = candidate_payload["current_format_overrides"]
    overrides = [overrides_by_key[candidate.key] for candidate in candidates]

    production_cache = _load_production_cache(
        args.production_weight_cache,
        cache_dir_override=args.production_cache_dir_override,
        lru_gb=float(args.production_cache_lru_gb),
    )
    prefetch_stats = _prefetch_production_cache(
        production_cache,
        candidate_payload["assignment"],
        args,
    )

    dtype = _dtype_from_name(args.dtype)
    model = load_text_model_under_work_root(
        args.model,
        device=args.device,
        dtype=dtype,
        work_root=work_root,
        device_map=args.device_map,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=Path(args.model).exists(),
    )
    calib_ids = load_calibration(
        tokenizer,
        args.dataset,
        int(args.n_calib_samples),
        int(args.calib_seqlen),
        calib_seed=int(args.calib_seed),
    )

    started = time.monotonic()
    kls = measure_override_paired_kl_deltas(
        model,
        candidate_payload["assignment"],
        overrides,
        calib_ids,
        work_root=work_root,
        max_lanes_per_batch=int(args.max_lanes_per_batch),
        profile=profile,
        progress_callback=_progress,
        tail_only=not bool(args.no_tail_only),
        cache_tail_layer_inputs=not bool(args.no_cache_tail_layer_inputs),
        include_activation_quant=not bool(args.no_activation_quant),
        production_weight_cache=production_cache,
        strict_production_weight_cache=not bool(args.allow_rtn_fallback),
        use_frozen_context_cache=bool(args.frozen_context_cache),
    )
    elapsed = time.monotonic() - started

    rows = []
    for local_rank, (candidate, propagated_kl) in enumerate(
        zip(candidates, kls, strict=True),
        start=1,
    ):
        row = candidate.to_json()
        bits_delta = float(row["bits_delta"])
        local_mse = float(row["output_mse_removed"])
        row.update({
            "local_rank": int(local_rank),
            "candidate_lane_override": dict(overrides_by_key[candidate.key]),
            "propagated_kl": float(propagated_kl),
            "propagated_kl_per_added_bit": _finite_ratio(
                float(propagated_kl),
                bits_delta,
            ),
            "local_output_mse_per_added_bit": _finite_ratio(
                local_mse,
                bits_delta,
            ),
            "propagation_amplification_kl_per_output_mse": _finite_ratio(
                float(propagated_kl),
                local_mse,
            ),
        })
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["propagated_kl_per_added_bit"] or 0.0),
            -float(row["propagated_kl"]),
            row["key"],
        ),
    )
    propagated_rank = {row["key"]: idx for idx, row in enumerate(ranked, start=1)}
    for row in rows:
        row["propagated_rank"] = propagated_rank[row["key"]]

    report = {
        "schema": "prismaquant.propagated_group_sensitivity.v1",
        "model": args.model,
        "base_assignment": args.base_assignment,
        "costs": args.costs,
        "probe": args.probe,
        "production_weight_cache": args.production_weight_cache,
        "production_cache_dir_override": args.production_cache_dir_override,
        "production_cache_prefetch": prefetch_stats,
        "calibration": {
            "dataset": args.dataset,
            "n_samples": int(args.n_calib_samples),
            "seqlen": int(args.calib_seqlen),
            "seed": int(args.calib_seed),
        },
        "target_format": candidate_payload["target_format"],
        "categories": candidate_payload["categories"],
        "group_by": candidate_payload["group_by"],
        "metric": candidate_payload["metric"],
        "params": candidate_payload["params"],
        "base_bits": candidate_payload["base_bits"],
        "base_bpp": candidate_payload["base_bpp"],
        "candidate_count_total": len(candidate_payload["candidates"]),
        "measured_count": len(rows),
        "max_lanes_per_batch": int(args.max_lanes_per_batch),
        "tail_only": not bool(args.no_tail_only),
        "cache_tail_layer_inputs": not bool(args.no_cache_tail_layer_inputs),
        "include_activation_quant": not bool(args.no_activation_quant),
        "frozen_context_cache": bool(args.frozen_context_cache),
        "strict_production_weight_cache": not bool(args.allow_rtn_fallback),
        "elapsed_seconds": float(elapsed),
        "rows": rows,
        "top_by_propagated_per_bit": ranked[:50],
        "top_by_local_mse_per_bit": rows[:50],
    }
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {output_report}")
    print(
        f"measured={len(rows)}/{len(candidate_payload['candidates'])} "
        f"elapsed={elapsed:.1f}s base_bpp={float(candidate_payload['base_bpp']):.6f}",
        flush=True,
    )
    print("top propagated KL per added bit:")
    for row in ranked[: min(10, len(ranked))]:
        value = row["propagated_kl_per_added_bit"]
        print(
            f"  #{row['propagated_rank']} {row['key']}: "
            f"kl={row['propagated_kl']:.6g} "
            f"kl/bit={(0.0 if value is None else value):.6g} "
            f"local_rank={row['local_rank']} members={row['member_count']} "
            f"formats={row['current_formats']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
