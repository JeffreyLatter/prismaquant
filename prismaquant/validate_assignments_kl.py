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
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.kl_measurement import assignment_bit_total, measure_assignment_kl


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
        assignment = {str(k): fr.canonical_format_name(str(v)) for k, v in payload["assignment"].items()}
    elif isinstance(payload, Mapping):
        assignment = {str(k): fr.canonical_format_name(str(v)) for k, v in payload.items()}
    else:
        raise ValueError(f"unsupported assignment JSON shape: {path}")
    if base is not None:
        merged = {str(k): fr.canonical_format_name(str(v)) for k, v in base.items()}
        merged.update(assignment)
        return merged
    return assignment


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.stem, path


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
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--disable-frozen-weight-cache", action="store_true")
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
    device = torch.device(device_str)
    dtype = _dtype_from_name(args.dtype)
    staged, cleanup = stage_multimodal(args.model)
    work_root = Path(args.work_dir or tempfile.mkdtemp(prefix="prismaquant_validate_kl_"))
    work_root.mkdir(parents=True, exist_ok=True)
    remove_work_root = args.work_dir is None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        calib_ids = load_wikitext_calibration_windowed(
            tokenizer,
            args.n_calib_samples,
            args.calib_seqlen,
            split=args.calib_split,
            seed=args.calib_seed,
        )
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        if not args.device_map and device.type != "cuda":
            model.to(device)
        model.eval()
        model_device = next(model.parameters()).device
        try:
            profile = detect_profile(args.model)
        except Exception:
            profile = DefaultProfile()
        ref_log_probs = cache_reference_log_probs(model, calib_ids, model_device)

        results = []
        for label, assignment, path in assignments:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            kl = measure_assignment_kl(
                model,
                assignment,
                calib_ids,
                ref_log_probs,
                work_root=work_root,
                profile=profile,
                use_frozen_weight_cache=not args.disable_frozen_weight_cache,
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
