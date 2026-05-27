#!/usr/bin/env python3
"""Measure budget-neutral swap candidates against the base assignment."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from prismaquant.calibration_data import _dtype_from_name
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.kl_measurement import (
    measure_assignment_kl,
    measure_override_paired_kl_deltas,
    measure_override_set_kl,
)
from prismaquant.layer_config import load_assignment
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.perturbed_x_cache import load_text_model_under_work_root
from prismaquant.sensitivity_probe import load_calibration


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
            log_prefix="[budget-swaps/prod-cache-files]",
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
            log_prefix="[budget-swaps/prod-cache]",
        )
    raise ValueError(f"unknown production prefetch mode {args.production_cache_prefetch!r}")


def _progress(event: dict) -> None:
    name = event.get("event", "event")
    if name.endswith("_start"):
        print(
            "[budget-swaps] "
            f"{name} {event.get('chunk_index')}/{event.get('chunk_count')} "
            f"lanes={event.get('lane_count')} overrides={event.get('override_count')}",
            flush=True,
        )
    elif name.endswith("_end"):
        print(
            "[budget-swaps] "
            f"{name} {event.get('chunk_index')}/{event.get('chunk_count')} "
            f"batches={event.get('batch_count')} "
            f"dt={float(event.get('elapsed_seconds', 0.0)):.1f}s",
            flush=True,
        )


def _select_swaps(payload: Mapping[str, object], limit: int) -> list[dict]:
    rows = [row for row in payload.get("swaps", ()) if isinstance(row, Mapping)]
    if int(limit) > 0:
        rows = rows[: int(limit)]
    return [dict(row) for row in rows]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-assignment", required=True)
    parser.add_argument("--swaps", required=True)
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
    parser.add_argument("--max-swaps", type=int, default=32)
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

    require_cuda_hot_path("measure_budget_neutral_swaps", args.device)

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    work_root = (
        Path(args.work_root)
        if args.work_root
        else output_report.parent / "budget_swap_work"
    )
    work_root.mkdir(parents=True, exist_ok=True)

    try:
        profile = detect_profile(args.model)
    except Exception:
        profile = DefaultProfile()

    assignment = load_assignment(args.base_assignment)
    swap_payload = json.loads(Path(args.swaps).read_text())
    swaps = _select_swaps(swap_payload, int(args.max_swaps))
    overrides = [dict(row["override"]) for row in swaps]

    production_cache = _load_production_cache(
        args.production_weight_cache,
        cache_dir_override=args.production_cache_dir_override,
        lru_gb=float(args.production_cache_lru_gb),
    )
    prefetch_stats = _prefetch_production_cache(production_cache, assignment, args)

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

    device = next(model.parameters()).device
    print("[budget-swaps] caching BF16 teacher logprobs", flush=True)
    ref_log_probs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(calib_ids.size(0)):
            batch = calib_ids[i:i + 1].to(device)
            logits = model(batch).logits[:, -1:, :]
            ref_log_probs.append(F.log_softmax(logits.float(), dim=-1).detach())
            del logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    started = time.monotonic()
    base_kl_work = work_root / "base_kl"
    swap_kl_work = work_root / "swap_kl"
    base_kl_work.mkdir(parents=True, exist_ok=True)
    swap_kl_work.mkdir(parents=True, exist_ok=True)
    base_kl = measure_assignment_kl(
        model,
        assignment,
        calib_ids,
        ref_log_probs,
        work_root=base_kl_work,
        profile=profile,
        use_frozen_weight_cache=bool(args.frozen_context_cache),
        production_weight_cache=production_cache,
        rng_seed=0,
        include_activation_quant=not bool(args.no_activation_quant),
        use_cuda_graphs=False,
    )
    swap_kl_vs_bf16 = measure_override_set_kl(
        model,
        assignment,
        overrides,
        calib_ids,
        ref_log_probs,
        work_root=swap_kl_work,
        max_lanes_per_batch=int(args.max_lanes_per_batch),
        profile=profile,
        include_activation_quant=not bool(args.no_activation_quant),
        use_cuda_graphs=False,
        production_weight_cache=production_cache,
        strict_production_weight_cache=not bool(args.allow_rtn_fallback),
        use_frozen_perturbed_cache=bool(args.frozen_context_cache),
    )
    swap_kl_vs_base = measure_override_paired_kl_deltas(
        model,
        assignment,
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
        baseline_mode="assignment",
    )
    elapsed = time.monotonic() - started

    rows = []
    for row, kl_bf16, kl_base in zip(
        swaps,
        swap_kl_vs_bf16,
        swap_kl_vs_base,
        strict=True,
    ):
        out = dict(row)
        out["base_kl_vs_bf16"] = float(base_kl)
        out["swap_kl_vs_bf16"] = float(kl_bf16)
        out["swap_delta_kl_vs_bf16"] = float(kl_bf16 - base_kl)
        out["swap_kl_vs_base_assignment"] = float(kl_base)
        out["empirical_score"] = -float(kl_bf16 - base_kl)
        rows.append(out)
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["swap_delta_kl_vs_bf16"]),
            float(row["swap_kl_vs_base_assignment"]),
            float(row.get("net_bits_delta", 0.0)),
            -float(row.get("promoted_propagated_kl", 0.0)),
            str(row.get("key", "")),
        ),
    )
    for idx, row in enumerate(ranked, start=1):
        row["measured_rank"] = idx

    report = {
        "schema": "prismaquant.budget_neutral_swap_measurement.v1",
        "model": args.model,
        "base_assignment": args.base_assignment,
        "swaps": args.swaps,
        "production_weight_cache": args.production_weight_cache,
        "production_cache_dir_override": args.production_cache_dir_override,
        "production_cache_prefetch": prefetch_stats,
        "calibration": {
            "dataset": args.dataset,
            "n_samples": int(args.n_calib_samples),
            "seqlen": int(args.calib_seqlen),
            "seed": int(args.calib_seed),
            "kl_scope": "last_token",
        },
        "baseline_mode": "assignment",
        "base_kl_vs_bf16": float(base_kl),
        "measured_count": len(rows),
        "max_lanes_per_batch": int(args.max_lanes_per_batch),
        "tail_only": not bool(args.no_tail_only),
        "cache_tail_layer_inputs": not bool(args.no_cache_tail_layer_inputs),
        "include_activation_quant": not bool(args.no_activation_quant),
        "frozen_context_cache": bool(args.frozen_context_cache),
        "strict_production_weight_cache": not bool(args.allow_rtn_fallback),
        "elapsed_seconds": float(elapsed),
        "rows": rows,
        "ranked": ranked,
    }
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"wrote {output_report} measured={len(rows)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    if ranked:
        best = ranked[0]
        print(
            "best swap: "
            f"{best['key']} delta_kl={best['swap_delta_kl_vs_bf16']:.8g} "
            f"swap_kl={best['swap_kl_vs_bf16']:.8g} "
            f"base_drift={best['swap_kl_vs_base_assignment']:.8g} "
            f"net_bits={float(best.get('net_bits_delta', 0.0)):.0f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
