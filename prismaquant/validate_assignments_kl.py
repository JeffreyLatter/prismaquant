"""Measure real last-token KL for one or more assignment JSON files."""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    kl_divergence,
    stage_multimodal,
)
from prismaquant.calibration_data import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.layer_config import canonicalize_format
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.kl_measurement import (
    assignment_bit_total,
    assignment_hash,
    measure_assignment_kl,
)
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    build_quantizable_map,
    calibration_data_hash,
)
from prismaquant.production_weight_cache import ProductionWeightCacheVariantView
from prismaquant.schemas import validate_cost_payload
from prismaquant.sensitivity_probe import load_calibration
from prismaquant.source_prefetch import prefetch_safetensors_checkpoint


def _load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def _load_probe_stats(path: str | Path) -> dict:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, Mapping) and isinstance(payload.get("stats"), Mapping):
        return dict(payload["stats"])
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError(f"probe file {path} does not contain a stats mapping")


def _load_costs(path: str | Path) -> dict:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    validate_cost_payload(payload, str(path))
    return dict(payload["costs"])


def load_assignment_json(path: str | Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    payload = _load_json(path)
    if isinstance(payload, Mapping) and isinstance(payload.get("assignment"), Mapping):
        assignment = {str(k): canonicalize_format(v) for k, v in payload["assignment"].items()}
    elif isinstance(payload, Mapping):
        assignment = {str(k): canonicalize_format(v) for k, v in payload.items()}
    else:
        raise ValueError(f"unsupported assignment JSON shape: {path}")
    if base is not None:
        merged = {str(k): canonicalize_format(v) for k, v in base.items()}
        merged.update(assignment)
        return merged
    return assignment


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.stem, path


def _load_cache_variant_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("--production-cache-variant-map must contain a JSON object")
    raw = payload.get("chosen_cache_variants")
    if raw is None:
        numeric = payload.get("numeric_variants")
        if isinstance(numeric, Mapping):
            raw = numeric.get("chosen_cache_variants")
    if raw is None:
        raw = payload
    if not isinstance(raw, Mapping):
        raise ValueError(
            "--production-cache-variant-map must be a qname->cache-format map "
            "or a probe payload containing chosen_cache_variants"
        )
    return {
        str(qname): str(fmt).strip().upper()
        for qname, fmt in raw.items()
        if str(qname).strip() and str(fmt).strip()
    }


def _assignment_bpp(stats: Mapping, assignment: Mapping[str, str], specs_by_name: Mapping[str, fr.FormatSpec]) -> float:
    names = [name for name in assignment if name in stats]
    total_params = sum(int(stats[name].get("n_params", 0) or 0) for name in names)
    if total_params <= 0:
        return 0.0
    return assignment_bit_total(stats, assignment, specs_by_name) / float(total_params)


def _lookup_cost_entry(costs: Mapping, name: str, fmt: str) -> Mapping | None:
    per_name = costs.get(name)
    if not isinstance(per_name, Mapping):
        return None
    candidates = [str(fmt)]
    try:
        candidates.extend(fr.aliases_for(str(fmt)))
    except Exception:
        candidates.append(fr.canonical_format_name(str(fmt)))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        entry = per_name.get(key)
        if isinstance(entry, Mapping) and "error" not in entry:
            return entry
    return None


def _assignment_cost_summary(
    costs: Mapping,
    assignment: Mapping[str, str],
) -> dict[str, object]:
    """Summarize local cost-table MSE for an assignment.

    These are local render/probe metrics, not end-to-end KL.  BF16 entries are
    counted as zero-error because they preserve the original Linear weights.
    """
    sums = {
        "weight_mse": 0.0,
        "output_mse": 0.0,
        "fisher_output_mse": 0.0,
        "rel_output_mse": 0.0,
        "predicted_dloss": 0.0,
    }
    counts = {
        "weight_mse": 0,
        "output_mse": 0,
        "fisher_output_mse": 0,
        "rel_output_mse": 0,
        "predicted_dloss": 0,
    }
    missing: list[str] = []
    unmeasured_output = 0
    format_counts: Counter[str] = Counter()
    for name, raw_fmt in assignment.items():
        fmt = fr.canonical_format_name(str(raw_fmt).strip().upper())
        format_counts[fmt] += 1
        if fmt == "BF16":
            for key in ("weight_mse", "output_mse", "rel_output_mse"):
                counts[key] += 1
            continue
        entry = _lookup_cost_entry(costs, str(name), fmt)
        if entry is None:
            missing.append(str(name))
            continue
        if entry.get("output_mse_measured") is False:
            unmeasured_output += 1
        for key in sums:
            value = entry.get(key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except Exception:
                continue
            sums[key] += value_f
            counts[key] += 1
    means = {
        key: (sums[key] / counts[key] if counts[key] else None)
        for key in sums
    }
    return {
        "objective": "local_cost_table_mse",
        "weight_mse_sum": float(sums["weight_mse"]),
        "weight_mse_mean": means["weight_mse"],
        "output_mse_sum": float(sums["output_mse"]),
        "output_mse_mean": means["output_mse"],
        "fisher_output_mse_sum": float(sums["fisher_output_mse"]),
        "fisher_output_mse_mean": means["fisher_output_mse"],
        "rel_output_mse_sum": float(sums["rel_output_mse"]),
        "rel_output_mse_mean": means["rel_output_mse"],
        "predicted_dloss_sum": float(sums["predicted_dloss"]),
        "predicted_dloss_mean": means["predicted_dloss"],
        "counts": dict(counts),
        "formats": dict(format_counts),
        "missing_count": int(len(missing)),
        "missing_sample": missing[:8],
        "output_mse_unmeasured_count": int(unmeasured_output),
    }


def _device_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


@contextmanager
def _temporary_env(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _materialize_assignment_inplace(
    model,
    assignment: Mapping[str, str],
    production_cache,
    *,
    progress: bool = False,
    log_prefix: str = "[validate-kl/inplace]",
) -> dict[str, object]:
    """Destructively copy one rendered assignment into the live model.

    This is intentionally one-way: reference logits must be cached before
    calling it, and callers should reload the model for another assignment.
    It avoids whole-assignment hook clone/restore overhead and keeps rendered
    weights flowing through ``ProductionWeightCache`` one tensor at a time.
    """
    quant_map = build_quantizable_map(model)
    copied = 0
    copied_bytes = 0
    format_counts: Counter[str] = Counter()
    missing_model: list[str] = []
    missing_cache: list[tuple[str, str]] = []
    shape_mismatch: list[dict[str, object]] = []
    seen_targets: set[tuple[int, str]] = set()
    start = time.time()
    total_non_bf16 = sum(
        1
        for fmt in assignment.values()
        if fr.canonical_format_name(str(fmt)) != "BF16"
    )
    if progress:
        print(
            f"{log_prefix} materializing {total_non_bf16} rendered weights "
            "into the live model",
            flush=True,
        )
    for name, fmt in assignment.items():
        fmt_canon = fr.canonical_format_name(str(fmt))
        if fmt_canon == "BF16":
            continue
        target = quant_map.get(str(name))
        if target is None:
            missing_model.append(str(name))
            continue
        module, attr = target
        target_key = (id(module), attr)
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        param = getattr(module, attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            missing_model.append(str(name))
            continue
        rendered = production_cache.get(str(name), fmt_canon)
        if rendered is None:
            missing_cache.append((str(name), fmt_canon))
            continue
        if tuple(rendered.shape) != tuple(param.shape):
            shape_mismatch.append(
                {
                    "name": str(name),
                    "format": fmt_canon,
                    "rendered_shape": list(rendered.shape),
                    "param_shape": list(param.shape),
                }
            )
            continue
        with torch.no_grad():
            rendered_device = rendered.to(
                device=param.device,
                dtype=param.dtype,
                non_blocking=True,
            )
            param.data.copy_(rendered_device)
        copied += 1
        copied_bytes += int(param.numel() * param.element_size())
        format_counts[fmt_canon] += 1
        if progress and (copied == total_non_bf16 or copied % 64 == 0):
            print(
                f"{log_prefix} materialized {copied}/{total_non_bf16} "
                f"weights ({copied_bytes / 1024**3:.2f} GiB copied)",
                flush=True,
            )
        del rendered
        if "rendered_device" in locals():
            del rendered_device
    if missing_model or missing_cache or shape_mismatch:
        raise RuntimeError(
            "in-place assignment materialization failed: "
            f"missing_model={len(missing_model)} sample={missing_model[:8]} "
            f"missing_cache={len(missing_cache)} sample={missing_cache[:8]} "
            f"shape_mismatch={len(shape_mismatch)} sample={shape_mismatch[:3]}"
        )
    elapsed = time.time() - start
    if progress:
        gib_s = (copied_bytes / 1024**3 / elapsed) if elapsed > 0 else 0.0
        print(
            f"{log_prefix} materialized {copied} weights, "
            f"{copied_bytes / 1024**3:.2f} GiB in {elapsed:.1f}s "
            f"({gib_s:.2f} GiB/s)",
            flush=True,
        )
    return {
        "copied": int(copied),
        "copied_bytes": int(copied_bytes),
        "elapsed_seconds": float(elapsed),
        "format_counts": dict(format_counts),
    }


def _activation_quant_assignment(
    assignment: Mapping[str, str],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, fmt in assignment.items():
        spec = fr.get_format(fmt)
        if spec.act_bits is not None and int(spec.act_bits) < 16:
            out[str(name)] = spec.name
    return out


@torch.no_grad()
def _measure_inplace_assignment_kl(
    model,
    assignment: Mapping[str, str],
    calib_ids: torch.Tensor,
    ref_log_probs,
    *,
    work_root: Path,
    profile,
    production_cache,
    kl_scope: str,
    use_cuda_graphs: bool | None,
) -> tuple[float, dict[str, object]]:
    device = next(model.parameters()).device
    cal_hash = calibration_data_hash(calib_ids)
    calib_ids = calib_ids.to(device)
    full_sequence = kl_scope == "full_sequence"
    materialize_stats = _materialize_assignment_inplace(
        model,
        assignment,
        production_cache,
        progress=True,
    )
    hook_assignment = _activation_quant_assignment(assignment)
    hooks = PerturbedActivationCache(
        model,
        hook_assignment,
        Path(tempfile.mkdtemp(prefix="prismaquant_inplace_kl_", dir=str(work_root))),
        input_rows=0,
        cal_hash=cal_hash,
        profile=profile,
        production_weight_cache=production_cache,
        include_activation_quant=True,
        capture_inputs=False,
    )
    missing = [
        name for name in hooks.missing
        if fr.canonical_format_name(hook_assignment.get(name, "BF16")) != "BF16"
    ]
    if missing:
        raise RuntimeError(
            "assignment contains non-BF16 qnames that do not resolve on "
            f"the live model; missing={len(missing)} sample={missing[:8]}"
        )
    if hooks.skipped:
        raise RuntimeError(
            "assignment has conflicting activation-quant formats within at "
            f"least one module; sample={hooks.skipped[:3]}"
        )
    if use_cuda_graphs is None:
        # The in-place path is already one stable model graph, but CUDA graph
        # capture on 27B can exceed the GPU budget. Keep auto conservative.
        use_cuda_graphs = False
    values: list[float] = []
    graph_key = (
        id(model),
        "inplace",
        assignment_hash(assignment),
        kl_scope,
        cal_hash,
    )
    registry = None
    if use_cuda_graphs:
        from prismaquant.kl_measurement import _KL_CUDA_GRAPH_REGISTRY

        registry = _KL_CUDA_GRAPH_REGISTRY

    with _temporary_env("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1"):
        hooks.install()
        try:
            for i in range(calib_ids.size(0)):
                batch = calib_ids[i:i + 1]

                def _forward(batch_ids):
                    logits = model(batch_ids).logits
                    if not full_sequence:
                        logits = logits[:, -1:, :]
                    return logits.clone()

                if registry is not None:
                    logits = registry.run(
                        "assignment-kl-inplace-forward",
                        graph_key,
                        _forward,
                        batch,
                        enabled=True,
                        device=device,
                        keepalive=(hooks,),
                    )
                else:
                    logits = _forward(batch)
                teacher = ref_log_probs[i]
                if not full_sequence:
                    teacher = teacher[:, -1:, :]
                teacher = teacher.to(device, non_blocking=True)
                values.append(float(kl_divergence(logits, teacher).item()))
        finally:
            hooks.remove()
    stats = {
        "materialized": materialize_stats,
        "activation_hooks": {
            "plans": len(hooks.plans),
            "capture_inputs": False,
            "external_weight_management": True,
        },
        "cuda_graphs": bool(use_cuda_graphs),
    }
    return sum(values) / max(len(values), 1), stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate assignment JSONs with real KL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument(
        "--costs",
        default=None,
        help="Optional measure_quant_cost pickle. When supplied, each result "
        "includes assignment-level local MSE / predicted-Δloss summaries "
        "from the same cost table the allocator optimized.",
    )
    parser.add_argument("--base-assignment", required=True)
    parser.add_argument(
        "--assignment",
        action="append",
        required=True,
        help="Assignment path or label=path. Solve-result JSONs are overlaid on base.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--formats", default="NVFP4,MXFP8_E4M3,FP8_E4M3,BF16")
    parser.add_argument("--n-calib-samples", type=int, default=2)
    parser.add_argument("--calib-seqlen", type=int, default=128)
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional calibration source accepted by sensitivity_probe "
        "(HF dataset id, .jsonl, or .txt). When omitted, preserves the "
        "historical wikitext-2 windowed loader.",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--kl-scope",
        choices=("last_token", "full_sequence"),
        default="last_token",
        help="Token scope for KL. Default last_token matches production probe "
        "gates and avoids full-sequence reference tensor residency on 27B.",
    )
    parser.add_argument(
        "--kl-cuda-graphs",
        choices=("auto", "off", "on"),
        default="auto",
        help=(
            "CUDA graph mode for assignment KL replay. Use 'off' for large "
            "resident production-cache validations where graph capture would "
            "exceed the GPU memory budget."
        ),
    )
    parser.add_argument(
        "--assignment-materialization",
        choices=("auto", "hooks", "inplace"),
        default="auto",
        help=(
            "How to replay production-rendered assignments. 'auto' uses the "
            "in-place path for a single production-cache assignment and the "
            "legacy hook path otherwise."
        ),
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--source-prefetch",
        choices=("off", "auto", "require"),
        default="require",
        help=(
            "Prefetch local BF16 source safetensors before loading the teacher "
            "model. Default 'require' fails instead of allowing first-forward "
            "NVMe page faults on production KL validation."
        ),
    )
    parser.add_argument(
        "--source-prefetch-max-gb",
        type=float,
        default=0.0,
        help=(
            "Resident byte budget for source safetensors prefetch. 0 derives "
            "the budget from available memory minus --source-prefetch-headroom-gb."
        ),
    )
    parser.add_argument(
        "--source-prefetch-headroom-gb",
        type=float,
        default=16.0,
    )
    parser.add_argument("--source-prefetch-workers", type=int, default=2)
    parser.add_argument("--disable-frozen-weight-cache", action="store_true")
    parser.add_argument(
        "--production-weight-cache",
        default=None,
        help="Optional pickled ProductionWeightCache. When supplied, KL is "
        "measured on the same production-rendered W_tilde path used by export.",
    )
    parser.add_argument(
        "--production-cache-variant-map",
        default=None,
        help="JSON qname->internal cache-format map, or a probe payload "
        "containing chosen_cache_variants.",
    )
    parser.add_argument(
        "--production-cache-dir-override",
        default=None,
        help="Relocate disk-backed production cache entries to this directory.",
    )
    parser.add_argument(
        "--production-cache-lru-gb",
        type=float,
        default=4.0,
        help="Resident tensor budget for disk-backed production cache use.",
    )
    parser.add_argument(
        "--production-cache-prefetch",
        choices=("auto", "off", "require"),
        default="require",
        help="Preload assignment-required rendered weights before KL replay. "
             "'require' fails instead of allowing an NVMe-bound validation.",
    )
    parser.add_argument(
        "--production-cache-prefetch-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--production-cache-file-prefetch-max-gb",
        type=float,
        default=0.0,
        help=(
            "Resident byte budget for production cache file-page prefetch in "
            "the in-place replay path. 0 derives the budget from available "
            "memory minus --production-cache-file-prefetch-headroom-gb."
        ),
    )
    parser.add_argument(
        "--production-cache-file-prefetch-headroom-gb",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "--halo-mode",
        choices=("off", "random"),
        default="off",
        help="Apply HALO to the BF16 model before reference/candidate KL. "
        "Required when measuring a HALO-rendered production cache.",
    )
    parser.add_argument("--halo-seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.disable_frozen_weight_cache:
        import os

        os.environ["PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE"] = "0"

    stats = _load_probe_stats(args.probe)
    costs = _load_costs(args.costs) if args.costs else None
    specs = [fr.get_format(part.strip()) for part in args.formats.split(",") if part.strip()]
    specs_by_name = {spec.name: spec for spec in specs}
    specs_by_name.update({fr.canonical_format_name(spec.name): spec for spec in specs})

    base_assignment = load_assignment_json(args.base_assignment)
    labeled_paths = [_parse_labeled_path(value) for value in args.assignment]
    assignments = [
        (label, load_assignment_json(path, base=base_assignment), str(path))
        for label, path in labeled_paths
    ]

    device_str = _device_arg(args.device)
    device = require_cuda_hot_path("validate_assignments_kl", device_str)
    device_str = str(device)
    if args.device_map not in (None, "cuda"):
        raise RuntimeError(
            "validate_assignments_kl requires a CUDA-resident model. CPU/offload "
            f"device_map={args.device_map!r} is not allowed."
        )
    dtype = _dtype_from_name(args.dtype)
    staged, cleanup = stage_multimodal(args.model)
    work_root = Path(args.work_dir or tempfile.mkdtemp(prefix="prismaquant_validate_kl_"))
    work_root.mkdir(parents=True, exist_ok=True)
    remove_work_root = args.work_dir is None
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        tokenizer_kwargs = {
            "trust_remote_code": True,
            "local_files_only": bool(args.local_files_only or Path(staged).exists()),
        }
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": bool(args.local_files_only or Path(staged).exists()),
        }
        if args.device_map:
            load_kwargs["device_map"] = args.device_map
        elif device.type == "cuda":
            load_kwargs["device_map"] = device_str
        tokenizer = AutoTokenizer.from_pretrained(staged, **tokenizer_kwargs)
        if args.dataset:
            calib_ids = load_calibration(
                tokenizer,
                args.dataset,
                args.n_calib_samples,
                args.calib_seqlen,
            )
        else:
            calib_ids = load_wikitext_calibration_windowed(
                tokenizer,
                args.n_calib_samples,
                args.calib_seqlen,
                split=args.calib_split,
                seed=args.calib_seed,
            )
        source_prefetch_stats = prefetch_safetensors_checkpoint(
            staged,
            mode=args.source_prefetch,
            max_resident_bytes=(
                int(float(args.source_prefetch_max_gb) * 1024**3)
                if float(args.source_prefetch_max_gb) > 0
                else None
            ),
            headroom_gb=float(args.source_prefetch_headroom_gb),
            workers=int(args.source_prefetch_workers),
            progress=True,
            log_prefix="[validate-kl/source]",
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        except ValueError as exc:
            if (
                "requires `accelerate`" not in str(exc)
                and "requires accelerate" not in str(exc)
            ):
                raise
            load_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
            if device.type == "cuda":
                model.to(device)
        if not args.device_map and device.type != "cuda":
            model.to(device)
        model.eval()
        model_device = next(model.parameters()).device
        production_cache = None
        if args.production_weight_cache:
            import pickle

            with Path(args.production_weight_cache).open("rb") as fh:
                production_cache = pickle.load(fh)
            if args.production_cache_dir_override:
                production_cache.relocate(args.production_cache_dir_override)
            if (
                args.production_cache_lru_gb
                and float(args.production_cache_lru_gb) > 0
                and hasattr(production_cache, "enable_lru")
            ):
                production_cache.enable_lru(
                    int(float(args.production_cache_lru_gb) * 1024**3)
                )
            variant_map = _load_cache_variant_map(args.production_cache_variant_map)
            if variant_map:
                production_cache = ProductionWeightCacheVariantView(
                    production_cache,
                    variant_map,
                )
        materialization_mode = args.assignment_materialization
        if materialization_mode == "auto":
            if production_cache is not None and len(assignments) == 1:
                materialization_mode = "inplace"
            else:
                materialization_mode = "hooks"
        if materialization_mode == "inplace":
            if production_cache is None:
                raise RuntimeError(
                    "--assignment-materialization=inplace requires "
                    "--production-weight-cache"
                )
            if len(assignments) != 1:
                raise RuntimeError(
                    "--assignment-materialization=inplace is destructive and "
                    "supports exactly one assignment per model load; run "
                    "multiple assignments as separate validator invocations."
                )
            if float(args.production_cache_lru_gb) <= 0:
                raise RuntimeError(
                    "in-place production-cache validation requires a bounded "
                    "--production-cache-lru-gb budget"
                )

        try:
            profile = detect_profile(args.model)
        except Exception:
            profile = DefaultProfile()
        cache_halo = dict(
            (getattr(production_cache, "metadata", {}) or {}).get("halo", {}) or {}
        ) if production_cache is not None else {"mode": "off"}
        cache_halo_mode = str(cache_halo.get("mode", "off"))
        if args.halo_mode == "off":
            if cache_halo_mode != "off":
                raise RuntimeError(
                    "[validate-kl] production cache was rendered with "
                    f"HALO mode={cache_halo_mode!r}; re-run validation with "
                    "matching --halo-mode/--halo-seed."
                )
        else:
            from prismaquant.halo import apply_random_halo_to_model

            if production_cache is not None:
                if cache_halo_mode != args.halo_mode:
                    raise RuntimeError(
                        "[validate-kl] production cache HALO mode mismatch: "
                        f"cache={cache_halo_mode!r} requested={args.halo_mode!r}")
                if int(cache_halo.get("seed", -1)) != int(args.halo_seed):
                    raise RuntimeError(
                        "[validate-kl] production cache HALO seed mismatch: "
                        f"cache={cache_halo.get('seed')!r} "
                        f"requested={args.halo_seed!r}")
            cfg = AutoConfig.from_pretrained(staged, **tokenizer_kwargs)
            _, halo_meta = apply_random_halo_to_model(
                model,
                profile,
                cfg,
                seed=args.halo_seed,
                verbose=True,
            )
            if production_cache is not None:
                for key in ("dim", "rotation_hash", "profile"):
                    expected = halo_meta.get(key)
                    actual = cache_halo.get(key)
                    if expected is not None and actual != expected:
                        raise RuntimeError(
                            "[validate-kl] production cache HALO metadata "
                            f"mismatch for {key}: cache={actual!r} "
                            f"expected={expected!r}")
        ref_log_probs = cache_reference_log_probs(
            model,
            calib_ids,
            model_device,
            kl_scope=args.kl_scope,
        )

        results = []
        for label, assignment, path in assignments:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            prefetch_stats = None
            if (
                production_cache is not None
                and args.production_cache_prefetch != "off"
            ):
                if materialization_mode == "inplace":
                    prefetch_stats = production_cache.prefetch_assignment_file_pages(
                        assignment,
                        mode=args.production_cache_prefetch,
                        max_resident_bytes=(
                            int(float(args.production_cache_file_prefetch_max_gb) * 1024**3)
                            if float(args.production_cache_file_prefetch_max_gb) > 0
                            else None
                        ),
                        headroom_gb=float(
                            args.production_cache_file_prefetch_headroom_gb
                        ),
                        max_workers=args.production_cache_prefetch_workers,
                        progress=True,
                        log_prefix="[validate-kl/prod-cache-files]",
                    )
                else:
                    preload_budget = (
                        getattr(production_cache, "_lru_max_bytes", 0) or None
                    )
                    prefetch_stats = production_cache.prefetch_assignment(
                        assignment,
                        max_resident_bytes=preload_budget,
                        max_workers=args.production_cache_prefetch_workers,
                        require=args.production_cache_prefetch == "require",
                        progress=True,
                        log_prefix="[validate-kl]",
                    )
            if materialization_mode == "inplace":
                kl, replay_stats = _measure_inplace_assignment_kl(
                    model,
                    assignment,
                    calib_ids,
                    ref_log_probs,
                    work_root=work_root,
                    profile=profile,
                    production_cache=production_cache,
                    kl_scope=args.kl_scope,
                    use_cuda_graphs=(
                        None if args.kl_cuda_graphs == "auto"
                        else args.kl_cuda_graphs == "on"
                    ),
                )
            else:
                kl = measure_assignment_kl(
                    model,
                    assignment,
                    calib_ids,
                    ref_log_probs,
                    work_root=work_root,
                    profile=profile,
                    use_frozen_weight_cache=not args.disable_frozen_weight_cache,
                    production_weight_cache=production_cache,
                    use_cuda_graphs=(
                        None if args.kl_cuda_graphs == "auto"
                        else args.kl_cuda_graphs == "on"
                    ),
                    kl_scope=args.kl_scope,
                    stream_ref_log_probs=args.kl_scope == "full_sequence",
                )
                replay_stats = {"mode": "hooks"}
            counts = dict(Counter(assignment.values()))
            changed = sum(
                1
                for name, fmt in assignment.items()
                if base_assignment.get(name) != fmt
            )
            result = {
                "label": label,
                "path": path,
                "last_token_kl": float(kl),
                "bpp": _assignment_bpp(stats, assignment, specs_by_name),
                "format_counts": counts,
                "changed_vs_base": int(changed),
                "assignment_entries": len(assignment),
                "kl_scope": args.kl_scope,
                "assignment_materialization": materialization_mode,
                "replay": replay_stats,
            }
            if costs is not None:
                result["mse"] = _assignment_cost_summary(costs, assignment)
            if prefetch_stats is not None:
                result["production_cache_prefetch"] = prefetch_stats
            results.append(result)
            mse_msg = ""
            if costs is not None:
                mse = result["mse"]
                mse_msg = (
                    f" output_mse={mse['output_mse_sum']:.6g}"
                    f" pred_dloss={mse['predicted_dloss_sum']:.6g}"
                    f" mse_missing={mse['missing_count']}"
                )
            print(
                f"[validate-kl] {label}: KL={kl:.8g} "
                f"bpp={result['bpp']:.6f}{mse_msg} "
                f"changed={changed} counts={counts}",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out = {
            "model": args.model,
            "probe": args.probe,
            "costs": args.costs,
            "base_assignment": args.base_assignment,
            "formats": [spec.name for spec in specs],
            "calibration": {
                "n_calib_samples": int(args.n_calib_samples),
                "calib_seqlen": int(args.calib_seqlen),
                "calib_split": args.calib_split,
                "calib_seed": int(args.calib_seed),
                "dataset": args.dataset,
                "kl_scope": args.kl_scope,
            },
            "kl_cuda_graphs": args.kl_cuda_graphs,
            "assignment_materialization": materialization_mode,
            "production_cache": {
                "path": args.production_weight_cache,
                "cache_dir_override": args.production_cache_dir_override,
                "lru_gb": float(args.production_cache_lru_gb),
                "prefetch": args.production_cache_prefetch,
                "prefetch_workers": int(args.production_cache_prefetch_workers),
                "file_prefetch_max_gb": float(
                    args.production_cache_file_prefetch_max_gb
                ),
                "file_prefetch_headroom_gb": float(
                    args.production_cache_file_prefetch_headroom_gb
                ),
            } if args.production_weight_cache else None,
            "source_prefetch": source_prefetch_stats,
            "halo": {
                "mode": args.halo_mode,
                "seed": int(args.halo_seed),
            },
            "results": results,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[validate-kl] wrote {output}", flush=True)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        if remove_work_root:
            shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
