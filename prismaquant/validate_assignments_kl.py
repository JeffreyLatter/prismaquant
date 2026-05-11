"""Measure real last-token KL for one or more assignment JSON files."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import cache_reference_log_probs, stage_multimodal
from prismaquant.calibration_data import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.layer_config import canonicalize_format
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.kl_measurement import assignment_bit_total, measure_assignment_kl
from prismaquant.production_weight_cache import ProductionWeightCacheVariantView
from prismaquant.sensitivity_probe import load_calibration


def _load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def _load_probe_stats(path: str | Path) -> dict:
    import pickle

    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, Mapping) and isinstance(payload.get("stats"), Mapping):
        return dict(payload["stats"])
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError(f"probe file {path} does not contain a stats mapping")


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


def _device_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate assignment JSONs with real KL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", required=True)
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
    parser.add_argument("--work-dir", default=None)
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
        default=64.0,
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
        ref_log_probs = cache_reference_log_probs(model, calib_ids, model_device)

        results = []
        for label, assignment, path in assignments:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            prefetch_stats = None
            if (
                production_cache is not None
                and args.production_cache_prefetch != "off"
            ):
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
            kl = measure_assignment_kl(
                model,
                assignment,
                calib_ids,
                ref_log_probs,
                work_root=work_root,
                profile=profile,
                use_frozen_weight_cache=not args.disable_frozen_weight_cache,
                production_weight_cache=production_cache,
            )
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
            }
            if prefetch_stats is not None:
                result["production_cache_prefetch"] = prefetch_stats
            results.append(result)
            print(
                f"[validate-kl] {label}: KL={kl:.8g} "
                f"bpp={result['bpp']:.6f} changed={changed} counts={counts}",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out = {
            "model": args.model,
            "probe": args.probe,
            "base_assignment": args.base_assignment,
            "formats": [spec.name for spec in specs],
            "calibration": {
                "n_calib_samples": int(args.n_calib_samples),
                "calib_seqlen": int(args.calib_seqlen),
                "calib_split": args.calib_split,
                "calib_seed": int(args.calib_seed),
                "dataset": args.dataset,
            },
            "production_cache": {
                "path": args.production_weight_cache,
                "cache_dir_override": args.production_cache_dir_override,
                "lru_gb": float(args.production_cache_lru_gb),
                "prefetch": args.production_cache_prefetch,
                "prefetch_workers": int(args.production_cache_prefetch_workers),
            } if args.production_weight_cache else None,
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
