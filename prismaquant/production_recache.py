"""Production-faithful activation re-cache helpers.

The initial production cache is rendered from BF16-upstream activations.
At runtime, upstream Linears are quantized, so downstream activation ranges can
shift.  This module replays calibration with production weights installed and
re-fits the per-Linear ``activation_max_abs`` values consumed by export and
perturbed-X measurement.
"""
from __future__ import annotations

import json
import pickle
import tempfile
import time
import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant.decision_units import fused_group_key
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    calibration_data_hash,
    iter_calibration_forwards,
)


def _model_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        if not p.is_meta:
            return p.device
    return torch.device("cpu")


def _unify_fused_max_abs(
    max_abs: Mapping[str, float],
    *,
    profile=None,
) -> dict[str, float]:
    groups: dict[str, list[str]] = {}
    for qname, value in max_abs.items():
        if value <= 0:
            continue
        try:
            group = fused_group_key(profile, qname) if profile else qname
        except Exception:
            group = qname
        groups.setdefault(group, []).append(qname)

    unified: dict[str, float] = {}
    for members in groups.values():
        shared = max(float(max_abs[m]) for m in members)
        for member in members:
            unified[member] = shared
    return unified


@torch.no_grad()
def measure_production_activation_max_abs(
    model: nn.Module,
    calibration_data,
    assignment: Mapping[str, str],
    production_weight_cache,
    *,
    profile=None,
    include_activation_quant: bool = True,
    microbatch_size: int = 1,
    progress: bool = True,
) -> dict[str, float]:
    """Measure activation max-abs under quantized upstream weights.

    ``assignment`` must be the concrete production assignment that will be
    exported.  Candidate caches with multiple possible formats per Linear are
    intentionally not accepted here because the upstream graph would be
    ambiguous.
    """
    if production_weight_cache is None:
        raise ValueError("production_weight_cache is required for recache")
    if not assignment:
        return {}

    started = time.monotonic()
    device = _model_device(model)
    cal_hash = calibration_data_hash(calibration_data)
    with tempfile.TemporaryDirectory(prefix="prismaquant_recache_") as tmp:
        builder = PerturbedActivationCache(
            model,
            assignment,
            tmp,
            input_rows=0,
            cal_hash=cal_hash,
            profile=profile,
            production_weight_cache=production_weight_cache,
            include_activation_quant=include_activation_quant,
        )
        builder.install()
        try:
            for args, kwargs in iter_calibration_forwards(
                calibration_data,
                device,
                microbatch_size=microbatch_size,
            ):
                call_kwargs = dict(kwargs)
                call_kwargs.setdefault("use_cache", False)
                try:
                    model(*args, **call_kwargs)
                except TypeError:
                    model(*args, **kwargs)
        finally:
            builder.remove()
        measured = dict(builder.max_abs)

    unified = _unify_fused_max_abs(measured, profile=profile)
    if progress:
        print(
            "[prod-recache] measured activation max_abs for "
            f"{len(unified)} Linears in {time.monotonic() - started:.1f}s",
            flush=True,
        )
    return unified


def apply_activation_max_abs_to_cache(
    production_weight_cache,
    activation_max_abs: Mapping[str, float],
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Update a ProductionWeightCache with re-fitted activation ranges."""
    values = {str(k): float(v) for k, v in activation_max_abs.items() if v > 0}
    production_weight_cache.activation_max_abs = values or None
    production_weight_cache.activation_scales = production_weight_cache.activation_max_abs
    meta = dict(getattr(production_weight_cache, "metadata", {}) or {})
    recache_meta = dict(metadata or {})
    recache_meta.setdefault("status", "applied")
    recache_meta.setdefault("n_activation_max_abs", len(values))
    meta["activation_recache"] = recache_meta
    production_weight_cache.metadata = meta


def assignment_digest(assignment: Mapping[str, str]) -> str:
    """Stable digest for the concrete assignment used during re-cache."""
    payload = json.dumps(
        {str(k): str(v) for k, v in sorted(assignment.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def recache_production_weight_cache(
    model: nn.Module,
    calibration_data,
    assignment: Mapping[str, str],
    production_weight_cache,
    *,
    profile=None,
    include_activation_quant: bool = True,
    microbatch_size: int = 1,
    progress: bool = True,
    write_sidecar: bool = True,
) -> dict[str, float]:
    """Measure and apply production-faithful activation max-abs values."""
    max_abs = measure_production_activation_max_abs(
        model,
        calibration_data,
        assignment,
        production_weight_cache,
        profile=profile,
        include_activation_quant=include_activation_quant,
        microbatch_size=microbatch_size,
        progress=progress,
    )
    apply_activation_max_abs_to_cache(
        production_weight_cache,
        max_abs,
        metadata={
            "include_activation_quant": bool(include_activation_quant),
            "microbatch_size": int(microbatch_size),
            "assignment_sha256": assignment_digest(assignment),
            "assignment_entries": int(len(assignment)),
        },
    )
    cache_dir = getattr(production_weight_cache, "cache_dir", None)
    if write_sidecar and cache_dir:
        sidecar = Path(cache_dir) / "activation_max_abs.json"
        sidecar.write_text(json.dumps(max_abs, indent=2))
    return max_abs


def _strip_weight(name: str) -> str:
    return name[:-7] if name.endswith(".weight") else name


def _canonicalize_layer_format(entry: dict | str | int) -> str:
    if isinstance(entry, dict):
        dt = entry.get("data_type")
        bits = int(entry.get("bits", 0))
        if dt == "nv_fp" and bits == 4:
            return "NVFP4"
        if dt == "mx_fp" and bits == 4:
            return "MXFP4"
        if dt == "mx_fp" and bits == 8:
            elt = str(entry.get("weight_element_dtype", "fp8_e4m3")).lower()
            if elt == "fp8_e5m2":
                raise ValueError("MXFP8_E5M2 is not exportable on the vLLM path")
            return "MXFP8"
        if dt in ("float", "bfloat16") and bits in (16, 0):
            return "BF16"
        if dt == "fp8_e4m3" and bits == 8:
            group_size = int(entry.get("group_size", 0))
            if group_size == 128:
                return "FP8_SOURCE"
            if group_size == 32:
                return "MXFP8"
            if group_size in (0, -1):
                return "FP8_E4M3"
            return "MXFP8"
        if dt == "fp8_e5m2" and bits == 8:
            raise ValueError("FP8_E5M2 is not exportable on the vLLM path")
        if dt == "mx_fp" and bits == 6:
            elt = str(entry.get("weight_element_dtype", "fp6_e3m2")).lower()
            return "MXFP6_E2M3" if elt == "fp6_e2m3" else "MXFP6_E3M2"
        if dt == "fp6_e3m2" and bits == 6:
            return "MXFP6_E3M2"
        if dt == "fp6_e2m3" and bits == 6:
            return "MXFP6_E2M3"
        raise ValueError(f"unsupported layer-config entry: {entry!r}")
    if isinstance(entry, str):
        value = entry.lower()
        if value in ("nvfp4", "fp4", "4"):
            return "NVFP4"
        if value in ("mxfp4", "mx_fp4"):
            return "MXFP4"
        if value in ("mxfp8", "mxfp8_e4m3", "8"):
            return "MXFP8"
        if value in ("fp8", "fp8_e4m3", "fp8_e4m3fn"):
            return "FP8_E4M3"
        if value in ("mxfp8_e5m2", "fp8_e5m2"):
            raise ValueError("E5M2 FP8 formats are not exportable on the vLLM path")
        if value in ("bf16", "bfloat16", "16"):
            return "BF16"
    if isinstance(entry, int):
        if entry <= 4:
            return "NVFP4"
        if entry <= 8:
            return "MXFP8"
        return "BF16"
    raise ValueError(f"unrecognized layer-config entry: {entry!r}")


def _load_assignment(layer_config: str | Path) -> dict[str, str]:
    from prismaquant.schemas import validate_layer_config_payload

    path = Path(layer_config)
    payload = json.loads(path.read_text())
    validate_layer_config_payload(payload, str(path))
    return {
        _strip_weight(str(name)): _canonicalize_layer_format(entry)
        for name, entry in payload.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-fit ProductionWeightCache activation scales under "
        "quantized upstream weights."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-config", required=True)
    parser.add_argument("--production-weight-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir-override", default=None)
    parser.add_argument("--production-cache-lru-gb", type=float, default=24.0)
    parser.add_argument(
        "--dataset",
        default="/home/rob/dq-runs/calibration/diverse-v1.jsonl",
    )
    parser.add_argument("--n-calib-samples", type=int, default=8)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument(
        "--no-activation-quant",
        action="store_true",
        help="Measure ranges with production weights installed but without "
        "activation quantization in the replay hooks.",
    )
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from prismaquant.calibration_data import _dtype_from_name
    from prismaquant.model_profiles import DefaultProfile, detect_profile
    from prismaquant.perturbed_x_cache import load_text_model_under_work_root
    from prismaquant.sensitivity_probe import load_calibration

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.production_weight_cache, "rb") as fh:
        cache = pickle.load(fh)
    if args.cache_dir_override:
        cache.relocate(args.cache_dir_override)
    if args.production_cache_lru_gb > 0 and hasattr(cache, "enable_lru"):
        cache.enable_lru(int(float(args.production_cache_lru_gb) * 1024**3))

    work_root = Path(args.work_root) if args.work_root else output.parent / "recache_work"
    work_root.mkdir(parents=True, exist_ok=True)
    dtype = _dtype_from_name(args.dtype)
    model = load_text_model_under_work_root(
        args.model,
        device=args.device,
        dtype=dtype,
        work_root=work_root,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=Path(args.model).exists(),
    )
    calib_ids = load_calibration(
        tokenizer,
        args.dataset,
        args.n_calib_samples,
        args.calib_seqlen,
    )
    assignment = _load_assignment(args.layer_config)
    try:
        profile = detect_profile(args.model)
    except Exception:
        profile = DefaultProfile()
    max_abs = recache_production_weight_cache(
        model,
        calib_ids,
        assignment,
        cache,
        profile=profile,
        include_activation_quant=not args.no_activation_quant,
        microbatch_size=args.microbatch_size,
        progress=True,
    )
    with open(output, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[prod-recache] wrote {output} with {len(max_abs)} activation scales",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
