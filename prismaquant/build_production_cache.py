"""Build a production-faithful δw cache for a model checkpoint.

Renders W_tilde[name, fmt] using the export pipeline's activation-aware
passes (GPTQ damp-sweep + scale_sweep on NVFP4; activation-weighted scale
search on MXFP8/FP8; passthrough on BF16) and saves a pickle that
PerturbedActivationCache can load via ``production_weight_cache=...``.

By default this standalone CLI renders the explicit ``--formats`` menu for
all quantizable Linears. Pipeline callers should pass ``--render-scope
assignment --render-layer-config layer_config.json`` to render only the
concrete non-BF16 entries the export assignment will consume.

This CLI is GPU-or-bust and refuses CPU execution.

Usage:

    python -m prismaquant.build_production_cache \\
        --model /path/to/model \\
        --output /work/production_cache.pkl \\
        --formats NVFP4 \\
        --n-calib-samples 8 \\
        --calib-seqlen 256

The output pickle is a ``ProductionWeightCache`` keyed by
``(qname, fmt_canonical)``. Payloads are either resident tensors or references
into the configured streaming cache directory.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant.build_rtn_cache import (
    iter_quantizable_tensors,
    stage_multimodal,
)
from prismaquant.calibration_data import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.production_recache import _load_assignment
from prismaquant.production_weight_cache import (
    fill_production_weight_cache,
)
from prismaquant.sensitivity_probe import load_calibration


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build production δw cache")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--formats",
        default="NVFP4",
        help="Comma-separated formats to render. MXFP8 / FP8 / BF16 cache "
        "is cheap compared with NVFP4, but MXFP8 and FP8 still benefit "
        "from activation-weighted scale search when scale_sweep is enabled.",
    )
    p.add_argument(
        "--render-scope",
        choices=("format-menu", "assignment"),
        default="format-menu",
        help="format-menu renders every requested format for every eligible "
        "Linear. assignment renders only the concrete non-BF16 entries from "
        "--render-layer-config. The pipeline defaults to assignment to avoid "
        "wasting compute on unused cache entries.",
    )
    p.add_argument(
        "--render-layer-config",
        default=None,
        help="Concrete layer_config.json assignment used when "
        "--render-scope=assignment. Non-BF16 entries are rendered exactly; "
        "BF16 entries are ignored because they do not need cache weights.",
    )
    p.add_argument("--n-calib-samples", type=int, default=8)
    p.add_argument("--calib-seqlen", type=int, default=256)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument(
        "--dataset",
        default=None,
        help="Optional calibration source accepted by sensitivity_probe "
        "(HF dataset id, .jsonl, or .txt). When omitted, preserves the "
        "historical wikitext-2 windowed loader.",
    )
    p.add_argument("--dtype", default="bf16")
    p.add_argument(
        "--max-act-rows",
        type=int,
        default=512,
        help="Max activation rows kept per Linear for GPTQ covariance. "
        "GPTQ is O(in_features^2); rows just need to span the input "
        "subspace well.",
    )
    p.add_argument(
        "--enable",
        default="gptq,joint_scale_opt,scale_sweep",
        help="Comma-separated levers to enable. Currently honored: "
        "{none, gptq, scale_sweep, fisher_gptq, awq_round, "
        "static_act_order, joint_scale_opt}.  "
        "Use none for RTN-only rendering with no local production levers. "
        "AWQ, SmoothQuant, and BlockOrtho-G are archived under "
        "archive/foldscale_orthog_2026-05-13 and are rejected by the "
        "production cache path. PrismaClip (act_clip_solver) and "
        "PrismaFisherClip (fisher_clip) are archived under "
        "archive/prismaclip_2026-05-14. Joint NVFP4 "
        "sibling globals + calibrated "
        "input_global_scale are computed unconditionally when NVFP4 is in "
        "the format menu. NVFP4 block scaling follows "
        "PRISMAQUANT_NVFP4_SCALE_RULE.",
    )
    p.add_argument(
        "--disable",
        default="",
        help="Comma-separated levers to FORCE off (overrides defaults). "
        "Use this to run RTN-only ablations, e.g. --disable gptq,scale_sweep.",
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write the cache even if validate_coverage finds missing "
        "(qname, fmt) entries.  Default: fail loudly.  Downstream "
        "consumers running with PRISMAQUANT_STRICT_PRODUCTION_CACHE=1 "
        "will refuse to use an incomplete cache anyway.",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to stream per-Linear weight tensors to (one .pt "
        "per (qname, fmt)).  When set, fill peak memory is bounded by "
        "the largest single render rather than the full cache size.  "
        "The pickle becomes a small manifest; PerturbedActivationCache "
        "lazy-loads each weight on first access at hook time.  Required "
        "for arbitrarily-large models (e.g. 27B+ on a 121 GB UMA box).",
    )
    p.add_argument(
        "--h-detail-dir",
        default=None,
        help="Optional h-detail directory from incremental_probe. When "
        "'fisher_gptq' is enabled, g2_per_token vectors from this directory "
        "weight NVFP4 GPTQ/scale-sweep and MXFP8 scale-sweep objectives.",
    )
    p.add_argument(
        "--skip-qnames",
        nargs="*",
        default=["lm_head"],
        help="Substrings on qname components that should be EXCLUDED from "
        "the cache fill.  Default: lm_head — we always pin it to BF16 in "
        "polish (vLLM ParallelLMHead constraint), so a NVFP4 cache entry "
        "is unused.  Excluding lm_head also avoids the OOM-prone last "
        "render on big models with linear-attention forward fallbacks.",
    )
    p.add_argument(
        "--recache-layer-config",
        default=None,
        help="Optional concrete layer_config.json assignment. When set, "
        "after rendering the cache, replay calibration with those production "
        "weights installed and re-fit activation_max_abs for export.",
    )
    p.add_argument(
        "--recache-microbatch-size",
        type=int,
        default=1,
        help="Calibration microbatch size for the production activation "
        "re-cache replay.",
    )
    p.add_argument(
        "--no-recache-activation-quant",
        action="store_true",
        help="During re-cache, install production weights but leave activation "
        "quantization disabled in replay hooks.",
    )
    p.add_argument(
        "--halo-mode",
        choices=("off", "random"),
        default="off",
        help="Apply HALO before rendering production weights. A HALO cache is "
        "only valid with matching export --halo-mode/--halo-seed.",
    )
    p.add_argument(
        "--halo-seed",
        type=int,
        default=0,
        help="RNG seed for HALO random Hadamard sign diagonal.",
    )
    args = p.parse_args(argv)

    # Opt-in deterministic CUDA path. The default lever ablations on small
    # models show ~2-4% per-Linear weight variance across re-runs of the
    # same configuration, driven by non-deterministic CUDA reduction order
    # inside the GPTQ Cholesky + U-update. Enable
    # PRISMAQUANT_DETERMINISTIC=1 to force bit-reproducible builds at a
    # ~5-15% throughput cost. Required for any A/B comparison where the
    # expected gain is comparable to or smaller than the noise floor.
    if os.environ.get("PRISMAQUANT_DETERMINISTIC", "0").lower() in {"1", "true", "yes"}:
        import torch as _torch_det
        # cuBLAS workspace fix is the main lever for deterministic matmul
        # reductions; torch.use_deterministic_algorithms(True) additionally
        # forces some kernels into slow paths that produce numerically
        # different results — those differences then cascade into NaN KL
        # downstream even though per-Linear MSE looks healthy. Keep only
        # the cuBLAS workspace + seeded RNGs for now.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        _torch_det.manual_seed(int(args.calib_seed))
        if _torch_det.cuda.is_available():
            _torch_det.cuda.manual_seed_all(int(args.calib_seed))
        print(
            f"[build-prod-cache] deterministic mode ON "
            f"(CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}, "
            f"seed={args.calib_seed}, use_deterministic_algorithms=False)",
            flush=True,
        )

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formats = [f.strip().upper() for f in args.formats.split(",") if f.strip()]
    levers = {
        name: True for name in (
            x.strip() for x in args.enable.split(",")
        ) if name
    }
    for name in (x.strip() for x in args.disable.split(",")):
        if name:
            levers[name] = False

    dtype = _dtype_from_name(args.dtype)
    staged, cleanup = stage_multimodal(args.model)
    device = require_cuda_hot_path("build_production_cache")
    print(f"[build-prod-cache] device={device}", flush=True)
    try:
        local_only = Path(staged).exists()
        tokenizer = AutoTokenizer.from_pretrained(
            staged, trust_remote_code=True, local_files_only=local_only,
        )
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
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": local_only,
        }
        if device.type == "cuda":
            load_kwargs["device_map"] = "cuda"
        try:
            model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        except ValueError as exc:
            if "requires `accelerate`" not in str(exc) and "requires accelerate" not in str(exc):
                raise
            load_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
            model.to(device)
        if device.type != "cuda":
            model.to(device)
        model.eval()
        try:
            profile = detect_profile(args.model)
        except Exception:
            profile = DefaultProfile()
        halo_meta = {"mode": "off"}
        if args.halo_mode == "random":
            from prismaquant.halo import apply_random_halo_to_model

            cfg = AutoConfig.from_pretrained(
                staged,
                trust_remote_code=True,
                local_files_only=local_only,
            )
            print(
                f"[build-prod-cache] applying HALO mode=random "
                f"seed={args.halo_seed}",
                flush=True,
            )
            _, halo_meta = apply_random_halo_to_model(
                model,
                profile,
                cfg,
                seed=args.halo_seed,
                verbose=True,
            )
            print(
                "[build-prod-cache] HALO applied: "
                f"dim={halo_meta['dim']} "
                f"blocks={halo_meta['block_sizes']} "
                f"hash={halo_meta['rotation_hash']}",
                flush=True,
            )

        skip_tokens = list(args.skip_qnames or [])
        qnames: list[str] = []
        skipped: list[str] = []
        for full_name, mod, attr in iter_quantizable_tensors(model):
            if attr != "weight" or not isinstance(mod, nn.Linear):
                continue
            qname = full_name[:-7] if full_name.endswith(".weight") else full_name
            # Exact dotted-token match against --skip-qnames substrings.
            tokens = qname.split(".")
            if any(s in tokens for s in skip_tokens):
                skipped.append(qname)
                continue
            qnames.append(qname)
        print(
            f"[build-prod-cache] {len(qnames)} quantizable Linears, "
            f"formats={formats}, levers={sorted(levers)}",
            flush=True,
        )
        if skipped:
            print(
                f"[build-prod-cache] skipped {len(skipped)} qnames matching "
                f"{skip_tokens} (typically pinned-BF16 in polish): "
                f"{skipped if len(skipped) <= 5 else skipped[:5] + ['...']}",
                flush=True,
            )

        recache_assignment = (
            _load_assignment(args.recache_layer_config)
            if args.recache_layer_config else None
        )
        render_assignment = None
        if args.render_scope == "assignment":
            layer_config = args.render_layer_config or args.recache_layer_config
            if not layer_config:
                print(
                    "[build-prod-cache] FAIL: --render-scope=assignment "
                    "requires --render-layer-config",
                    flush=True,
                )
                return 2
            render_assignment = _load_assignment(layer_config)
            non_bf16 = sum(
                1 for fmt in render_assignment.values()
                if str(fmt).strip().upper() != "BF16"
            )
            print(
                f"[build-prod-cache] assignment render scope: "
                f"{non_bf16} non-BF16 entries from {layer_config}",
                flush=True,
            )
        t0 = time.monotonic()
        cache = fill_production_weight_cache(
            model, calib_ids, qnames,
            formats=formats,
            render_assignment=render_assignment,
            levers=levers,
            max_act_rows=args.max_act_rows,
            cache_dir=args.cache_dir,
            recache_pass=recache_assignment is not None,
            recache_assignment=recache_assignment,
            recache_profile=profile,
            recache_include_activation_quant=not args.no_recache_activation_quant,
            recache_microbatch_size=args.recache_microbatch_size,
            h_detail_dir=args.h_detail_dir,
        )
        elapsed = time.monotonic() - t0
        meta = dict(getattr(cache, "metadata", {}) or {})
        meta["halo"] = halo_meta
        cache.metadata = meta

        # Strict coverage validation: every (qname, NVFP4) must be present
        # before we ship.  Catches naming-alias mismatches, GPTQ Cholesky
        # failures, and any other silent gaps that would otherwise fall
        # through to RTN at hook time.
        try:
            if render_assignment is not None:
                _, missing = cache.assignment_keys(render_assignment)
                failed = list((cache.failed or {}).keys())
                if missing or failed:
                    samples = missing[:5] + failed[:5]
                    raise RuntimeError(
                        f"ProductionWeightCache assignment coverage failure: "
                        f"{len(missing)} misses, {len(failed)} failed "
                        f"renders; sample={samples}"
                    )
            else:
                cache.validate_coverage(qnames, formats)
            print("[build-prod-cache] coverage check passed", flush=True)
        except RuntimeError as e:
            if args.allow_incomplete:
                print(f"[build-prod-cache] WARNING: {e}", flush=True)
                print(
                    "[build-prod-cache] --allow-incomplete: writing cache "
                    "anyway.  Downstream consumers running with "
                    "PRISMAQUANT_STRICT_PRODUCTION_CACHE=1 will refuse "
                    "this cache.",
                    flush=True,
                )
            else:
                print(f"[build-prod-cache] FAIL: {e}", flush=True)
                print(
                    "[build-prod-cache] aborting.  Pass --allow-incomplete "
                    "to write the cache anyway, or fix the underlying "
                    "render failures.",
                    flush=True,
                )
                return 2

        compacted = (
            cache.compact_for_pickle()
            if hasattr(cache, "compact_for_pickle")
            else 0
        )
        if compacted:
            print(
                f"[build-prod-cache] compacted {compacted} resident cache "
                "tensors back to path references before writing",
                flush=True,
            )
        with open(output_path, "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[build-prod-cache] wrote {len(cache)} entries to "
            f"{output_path} ({elapsed:.1f}s)",
            flush=True,
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
