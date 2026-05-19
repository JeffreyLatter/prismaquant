"""Multi-shot recalibration of per-Linear costs.

Re-runs the calibration forwards through a partially-quantized model so the
downstream cost step measures `weight_mse` against activations that reflect
the previous shot's format assignment, rather than the BF16 baseline. This is
the cheap variant of the LLM-Surgeon-style T-shot recalibration loop
(arXiv:2312.17244 §3.5): we refresh activation-conditioned cost numbers but
keep the Fisher diagonal trace (``probe.pkl``) and inner quantizer knobs
(JSO clip grids, GPTQ damp) frozen across shots.

The output of this module is an activation cache directory in the same on-disk
format as ``incremental_probe`` writes — per-Linear ``{name}.pt`` files with
payload ``{"inputs": X, "name": name}`` plus a sidecar ``metadata.json``. The
``incremental_measure_quant_cost --activation-cache-dir <DIR>`` CLI consumes
the directory unchanged.

Design notes
------------
- Reuses ``PerturbedActivationCache`` from ``perturbed_x_cache`` for the
  quantized-upstream forward path. No parallel cache mechanism.
- Reuses ``ProductionWeightCache`` for the rendered weights. Optionally
  preloads + fails fast if the assignment cannot be made resident, matching
  ``production_recache`` semantics.
- Reuses ``SharedRowSubsampler`` semantics via the existing ``cal_hash``
  threading — passing the same calibration data produces the same per-Linear
  row subsample across shots, so shot-N costs are apples-to-apples with
  shot-1 costs.
- Writes activations in FP32 by default, matching the probe's
  ``PRISMAQUANT_ACT_CACHE_FP32=1`` default. Set ``PRISMAQUANT_ACT_CACHE_FP32=0``
  in the environment to drop to BF16.
- Writes a ``metadata.json`` sidecar carrying ``calibration_hash``, ``model``,
  ``profile``, ``assignment_sha256``, ``input_rows``, ``activation_dtype``,
  ``include_activation_quant``. The cost step does not yet validate this
  sidecar, but the file lets ``run-pipeline.sh`` (and humans) confirm that a
  given activation dir matches the layer_config it was recached for.

Promotion gate
--------------
Ships in Research state. The orchestration in ``run-pipeline.sh`` is gated by
``MULTI_SHOT_PASSES`` (default 1 = vanilla path is byte-identical). Promotes
to Candidate only after a measured KL win on Qwen3-4B across at least three
bpp budgets vs. the 1-shot baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.calibration_data import _dtype_from_name
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.layer_config import load_assignment as _load_assignment
from prismaquant.memory_management import model_device as _model_device
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    activation_cache_filename,
    calibration_data_hash,
    iter_calibration_forwards,
    load_text_model_under_work_root,
)
from prismaquant.production_recache import (
    assignment_digest,
    preload_production_cache_for_assignment,
)


_DEFAULT_CALIBRATION = "/home/rob/dq-runs/calibration/diverse-v1.jsonl"


def _activation_dtype_from_env() -> torch.dtype:
    """Match the probe's PRISMAQUANT_ACT_CACHE_FP32 default."""
    if os.environ.get("PRISMAQUANT_ACT_CACHE_FP32", "1") != "0":
        return torch.float32
    return torch.bfloat16


def _write_metadata(
    dest_cache_dir: Path,
    *,
    model: str,
    calibration_hash: str,
    assignment_sha256: str,
    input_rows: int,
    activation_dtype: torch.dtype,
    include_activation_quant: bool,
    profile_name: str,
    n_linears_written: int,
    seqlen: int | None,
    n_samples: int | None,
    shot_index: int | None,
    source_layer_config: str | None,
) -> Path:
    payload: dict[str, Any] = {
        "schema": "prismaquant.multi_shot.activation_recache.v1",
        "model": str(model),
        "calibration_hash": str(calibration_hash),
        "assignment_sha256": str(assignment_sha256),
        "input_rows": int(input_rows),
        "activation_dtype": str(activation_dtype).replace("torch.", ""),
        "include_activation_quant": bool(include_activation_quant),
        "profile": str(profile_name),
        "n_linears_written": int(n_linears_written),
        "wall_time_unix": time.time(),
    }
    if seqlen is not None:
        payload["calib_seqlen"] = int(seqlen)
    if n_samples is not None:
        payload["n_calib_samples"] = int(n_samples)
    if shot_index is not None:
        payload["shot_index"] = int(shot_index)
    if source_layer_config is not None:
        payload["source_layer_config"] = str(source_layer_config)
    target = dest_cache_dir / "metadata.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def _drain_snaps_to_probe_format(
    builder: PerturbedActivationCache,
    dest_cache_dir: Path,
    *,
    activation_dtype: torch.dtype,
    progress: bool,
) -> list[str]:
    """Write builder._snaps to dest_cache_dir in incremental_probe format.

    Each per-Linear file is ``{activation_cache_filename(name)}`` with payload
    ``{"inputs": X, "name": name}``. ``X`` is concatenated across all captured
    microbatches, truncated to ``input_rows``, moved to CPU and cast to
    ``activation_dtype``. No ``row_indices`` are written: the cheap variant
    does not refresh h-detail / per-token Fisher; row order is preserved
    deterministically by ``SharedRowSubsampler`` so apples-to-apples with the
    probe's row policy is maintained via ``cal_hash`` rather than via
    explicit indices.
    """
    dest_cache_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, snaps in builder._snaps.items():
        if not snaps:
            continue
        X = torch.cat(snaps, dim=0)[:builder.input_rows].contiguous()
        X = X.to(device="cpu", dtype=activation_dtype).contiguous()
        payload = {"inputs": X, "name": name}
        path = dest_cache_dir / activation_cache_filename(name)
        torch.save(payload, path)
        written.append(name)
    if progress:
        print(
            f"[multi-shot] drained {len(written)} per-Linear activations to "
            f"{dest_cache_dir}",
            flush=True,
        )
    return sorted(written)


@torch.no_grad()
def recache_calibration_activations_for_cost(
    model: nn.Module,
    calibration_data,
    assignment: Mapping[str, str],
    production_weight_cache,
    dest_cache_dir: str | Path,
    *,
    profile=None,
    input_rows: int = 256,
    microbatch_size: int = 1,
    activation_dtype: torch.dtype | None = None,
    include_activation_quant: bool = True,
    preload_production_cache: bool = True,
    preload_max_bytes: int | None = None,
    preload_max_workers: int = 4,
    require_preload: bool = True,
    progress: bool = True,
    source_layer_config: str | None = None,
    shot_index: int | None = None,
    n_samples: int | None = None,
    seqlen: int | None = None,
) -> dict[str, Any]:
    """Run calibration forwards under ``assignment`` and write per-Linear inputs.

    Replays ``calibration_data`` through ``model`` with ``PerturbedActivationCache``
    installed against ``assignment`` + ``production_weight_cache``. The
    upstream Linears in ``assignment`` are rendered to their target formats
    via the production cache; their downstream Linears see the resulting
    quantized-context activations, which are captured and written to
    ``dest_cache_dir`` in the on-disk format ``incremental_measure_quant_cost``
    expects.

    Parameters mirror ``production_recache.measure_production_activation_max_abs``
    with two additions: ``dest_cache_dir`` (where the full activations land)
    and ``activation_dtype`` (defaults to the probe's
    ``PRISMAQUANT_ACT_CACHE_FP32`` convention).

    Returns a manifest dict with ``written``, ``missing``, ``cache_dir``,
    ``metadata_path``, and a few diagnostic counts.
    """
    if production_weight_cache is None:
        raise ValueError(
            "production_weight_cache is required: the whole point of "
            "multi-shot recache is to apply the previous shot's rendered "
            "weights during the forward."
        )
    if not assignment:
        raise ValueError("assignment must be non-empty")
    activation_dtype = activation_dtype or _activation_dtype_from_env()
    dest_cache_dir = Path(dest_cache_dir)
    dest_cache_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    device = _model_device(model)
    cal_hash = calibration_data_hash(calibration_data)
    asn_digest = assignment_digest(assignment)

    if preload_production_cache:
        preload_production_cache_for_assignment(
            production_weight_cache,
            assignment,
            max_resident_bytes=preload_max_bytes,
            max_workers=preload_max_workers,
            require=require_preload,
            progress=progress,
        )

    builder = PerturbedActivationCache(
        model,
        assignment,
        dest_cache_dir,
        input_rows=int(input_rows),
        cal_hash=cal_hash,
        profile=profile,
        production_weight_cache=production_weight_cache,
        include_activation_quant=bool(include_activation_quant),
        capture_inputs=True,
    )
    builder.install()
    try:
        for args, kwargs in iter_calibration_forwards(
            calibration_data,
            device,
            microbatch_size=int(microbatch_size),
        ):
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault("use_cache", False)
            try:
                model(*args, **call_kwargs)
            except TypeError:
                model(*args, **kwargs)
    finally:
        builder.remove()

    written = _drain_snaps_to_probe_format(
        builder,
        dest_cache_dir,
        activation_dtype=activation_dtype,
        progress=progress,
    )

    profile_name = (
        getattr(profile, "name", "default") if profile is not None else "default"
    )
    metadata_path = _write_metadata(
        dest_cache_dir,
        model=str(getattr(model, "name_or_path", "")) or "unknown",
        calibration_hash=cal_hash,
        assignment_sha256=asn_digest,
        input_rows=int(input_rows),
        activation_dtype=activation_dtype,
        include_activation_quant=bool(include_activation_quant),
        profile_name=str(profile_name),
        n_linears_written=len(written),
        seqlen=seqlen,
        n_samples=n_samples,
        shot_index=shot_index,
        source_layer_config=source_layer_config,
    )

    elapsed = time.monotonic() - started
    if progress:
        print(
            f"[multi-shot] recache complete: {len(written)} Linears in "
            f"{elapsed:.1f}s; metadata={metadata_path}",
            flush=True,
        )

    return {
        "cache_dir": str(dest_cache_dir),
        "metadata_path": str(metadata_path),
        "written": written,
        "missing": sorted(builder.missing),
        "skipped_activation_quant": list(builder.skipped),
        "n_linears": len(written),
        "calibration_hash": cal_hash,
        "assignment_sha256": asn_digest,
        "elapsed_s": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-shot recalibration: replay calibration through a "
            "partially-quantized model and write per-Linear input activations "
            "for the next cost re-measurement."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recache = subparsers.add_parser(
        "recache",
        help="Recache calibration activations under a layer_config assignment.",
    )
    recache.add_argument("--model", required=True)
    recache.add_argument(
        "--layer-config",
        required=True,
        help="Assignment to render upstream during replay (previous shot).",
    )
    recache.add_argument(
        "--production-weight-cache",
        required=True,
        help=(
            "Pickled ProductionWeightCache that holds rendered weights for "
            "the assignment formats (built by build_production_cache for the "
            "previous shot)."
        ),
    )
    recache.add_argument(
        "--output",
        required=True,
        help="Destination activation cache directory.",
    )
    recache.add_argument("--cache-dir-override", default=None)
    recache.add_argument("--production-cache-lru-gb", type=float, default=64.0)
    recache.add_argument("--production-cache-prefetch-workers", type=int, default=4)
    recache.add_argument(
        "--production-cache-prefetch",
        choices=("auto", "off", "require"),
        default="require",
    )
    recache.add_argument("--dataset", default=_DEFAULT_CALIBRATION)
    recache.add_argument("--n-calib-samples", type=int, default=32)
    recache.add_argument("--calib-seqlen", type=int, default=1024)
    recache.add_argument("--dtype", default="bf16")
    recache.add_argument("--device", default="cuda")
    recache.add_argument("--device-map", default=None)
    recache.add_argument("--work-root", default=None)
    recache.add_argument("--input-rows", type=int, default=256)
    recache.add_argument("--microbatch-size", type=int, default=1)
    recache.add_argument(
        "--no-activation-quant",
        action="store_true",
        help=(
            "Disable activation quantization in the replay hooks. Default is "
            "to match production behaviour (W4A4 activations get clamped + "
            "act-quantized)."
        ),
    )
    recache.add_argument(
        "--shot-index",
        type=int,
        default=None,
        help="Optional shot index (k); written to metadata.json.",
    )

    args = parser.parse_args(argv)

    if args.command != "recache":
        parser.error(f"unknown command {args.command!r}")

    from prismaquant.model_profiles import DefaultProfile, detect_profile
    from prismaquant.sensitivity_probe import load_calibration
    from transformers import AutoTokenizer

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with open(args.production_weight_cache, "rb") as fh:
        cache = pickle.load(fh)
    if args.cache_dir_override:
        cache.relocate(args.cache_dir_override)
    if args.production_cache_lru_gb > 0 and hasattr(cache, "enable_lru"):
        cache.enable_lru(int(float(args.production_cache_lru_gb) * 1024**3))

    work_root = (
        Path(args.work_root)
        if args.work_root
        else output.parent / "multi_shot_work"
    )
    work_root.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    require_cuda_hot_path("multi_shot_recache", args.device)
    if args.device_map not in (None, "cuda"):
        raise RuntimeError(
            "multi_shot recache requires a CUDA-resident model. CPU/offload "
            f"device_map={args.device_map!r} is not allowed."
        )

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

    manifest = recache_calibration_activations_for_cost(
        model,
        calib_ids,
        assignment,
        cache,
        output,
        profile=profile,
        input_rows=int(args.input_rows),
        microbatch_size=int(args.microbatch_size),
        include_activation_quant=not args.no_activation_quant,
        preload_production_cache=args.production_cache_prefetch != "off",
        preload_max_bytes=(
            getattr(cache, "_lru_max_bytes", 0) or None
        ),
        preload_max_workers=int(args.production_cache_prefetch_workers),
        require_preload=args.production_cache_prefetch == "require",
        progress=True,
        source_layer_config=args.layer_config,
        shot_index=args.shot_index,
        n_samples=int(args.n_calib_samples),
        seqlen=int(args.calib_seqlen),
    )

    summary_path = output / "recache_manifest.json"
    summary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"[multi-shot] wrote manifest {summary_path} "
        f"({manifest['n_linears']} Linears, {len(manifest['missing'])} missing)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
