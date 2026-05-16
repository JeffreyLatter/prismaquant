"""Build a production-faithful δw cache for a model checkpoint.

Renders W_tilde[name, fmt] using the export pipeline's activation-aware
passes (GPTQ damp-sweep + scale_sweep on NVFP4; activation-weighted scale
search on MXFP8; passthrough on BF16) and saves a pickle that
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


def _estimate_model_bytes_from_index(model_path: str) -> int:
    """Return total param bytes from the safetensors shard index, 0 if unknown."""
    import json
    import os
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        try:
            with open(idx_path) as fh:
                d = json.load(fh)
            total = int((d.get("metadata") or {}).get("total_size", 0))
            if total > 0:
                return total
        except Exception:
            pass
    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        return os.path.getsize(single)
    return 0


def _fill_production_cache_streaming(
    *,
    model_path: str,
    staged: str,
    calib_ids: "torch.Tensor",
    device: "torch.device",
    dtype: "torch.dtype",
    formats: "list[str]",
    levers: dict,
    max_act_rows: int,
    cache_dir: "str | None",
    render_assignment: "dict | None",
    skip_qnames: "list[str]",
    h_detail_dir: "str | None" = None,
    progress: bool = True,
) -> "tuple":
    """Streaming production cache fill for models too large to fit in CUDA.

    Builds the model skeleton once (head modules resident, decoder layers
    streamed from disk), then for each layer: install → collect activations
    from all calibration samples → compute NVFP4 joint globals → render GPTQ
    weight → unload.  Peak CUDA memory is approximately one layer plus the
    hidden-state buffer instead of the full model.

    Returns ``(cache, qnames, skipped)`` where ``qnames`` is the list of
    eligible nn.Linear qnames derived from the skeleton (same as the
    non-streaming path would derive from the live model) and ``skipped`` lists
    qnames excluded by ``skip_qnames``.

    Limitations compared to ``fill_production_weight_cache``:
      - AWQ, SmoothQuant, HALO, and recache passes are not supported (all
        require the full model resident at once).
      - ``h_detail_dir`` (fisher_gptq/fisher_clip) is silently ignored.
    """
    import gc
    import json
    import re
    import tempfile
    from pathlib import Path

    import torch
    import torch.nn as nn

    from prismaquant.build_rtn_cache import iter_quantizable_tensors
    from prismaquant.layer_streaming import (
        _call_layer,
        _compute_position_embeddings,
        _make_causal_mask,
    )
    from prismaquant.production_weight_cache import (
        PRISMACLIP_FORMAT,
        ProductionWeightCache,
        _LinearActivationCollector,
        _cache_weight_filename,
        _render_base_format,
        _store_rendered_weight_entry,
        render_production_weight,
    )
    from prismaquant.streaming_model import _build_streaming_context
    from prismaquant import format_registry as fr
    from prismaquant.export_native_compressed import resolve_nvfp4_scale_rule

    # Normalise levers. AWQ, SmoothQuant, and fisher passes are unsupported.
    levers = dict(levers)
    levers.setdefault("gptq", True)
    levers.setdefault("gptq_damp_sweep", bool(levers.get("gptq", True)))
    levers.setdefault("scale_sweep", True)
    levers.setdefault("act_clip_solver", False)
    levers.setdefault("nvfp4_scale_rule", resolve_nvfp4_scale_rule())
    for _unsup in ("awq", "awq_round", "smoothquant", "fisher_gptq", "fisher_clip"):
        if levers.pop(_unsup, False) and progress:
            print(
                f"[prod-cache-stream] WARNING: '{_unsup}' not supported in "
                "streaming path; disabled",
                flush=True,
            )

    def _canon(fmt: str) -> str:
        fmt_u = str(fmt).strip().upper()
        if fmt_u == PRISMACLIP_FORMAT:
            return PRISMACLIP_FORMAT
        return fr.canonical_format_name(fmt_u)

    requested_formats = tuple(
        dict.fromkeys(_canon(f) for f in formats if str(f).strip())
    )

    offload_folder = tempfile.mkdtemp(prefix="prod_cache_stream_offload_")
    try:
        ctx = _build_streaming_context(
            model_path,
            device=device,
            dtype=dtype,
            offload_folder=offload_folder,
            log_prefix="[prod-cache-stream]",
        )
        skeleton = ctx.model
        base_model = ctx.base_model
        layers = ctx.layers
        num_layers = ctx.num_layers
        layers_prefix = ctx.layers_prefix
        profile = ctx.profile

        # Build qnames from skeleton (meta tensors have correct shapes/strides).
        skip_tokens_set = set(skip_qnames or [])
        qnames: list[str] = []
        skipped_list: list[str] = []
        qname_to_module: dict[str, nn.Module] = {}
        for full_name, mod, attr in iter_quantizable_tensors(skeleton):
            if attr != "weight" or not isinstance(mod, nn.Linear):
                continue
            qname = full_name[:-7] if full_name.endswith(".weight") else full_name
            tokens = qname.split(".")
            if any(s in tokens for s in skip_tokens_set):
                skipped_list.append(qname)
                continue
            qnames.append(qname)
            qname_to_module[qname] = mod
        eligible_qnames = set(qnames)

        if progress:
            print(
                f"[prod-cache-stream] {len(qnames)} quantizable Linears from "
                f"skeleton; formats={list(requested_formats)}",
                flush=True,
            )

        # Resolve per-qname format lists.
        if render_assignment is not None:
            render_formats_by_qname: dict[str, list[str]] = {}
            for q, fmt in render_assignment.items():
                if q not in eligible_qnames:
                    continue
                fmt_canon = _canon(fmt)
                if fmt_canon == "BF16":
                    continue
                render_formats_by_qname[q] = [fmt_canon]
            render_scope = "assignment"
        else:
            non_bf16_formats = [f for f in requested_formats if f != "BF16"]
            render_formats_by_qname = {q: non_bf16_formats for q in eligible_qnames}
            render_scope = "format-menu"

        qname_set: set[str] = set(render_formats_by_qname)
        if not qname_set:
            return (
                ProductionWeightCache(
                    weights={},
                    levers=dict(levers),
                    metadata={"render_scope": render_scope, "streaming": True},
                ),
                qnames,
                skipped_list,
            )

        activation_aware_formats = {"NVFP4", PRISMACLIP_FORMAT}
        if bool(levers.get("scale_sweep", True)):
            activation_aware_formats.update({"MXFP8", "MXFP8_E4M3"})
        qnames_needing_activation: set[str] = {
            q for q, fmts in render_formats_by_qname.items()
            if any(f in activation_aware_formats for f in fmts)
        }
        needs_nvfp4_render = any(
            any(_render_base_format(fmt) == "NVFP4" for fmt in fmts)
            for fmts in render_formats_by_qname.values()
        )

        if progress:
            total_entries = sum(
                len(render_formats_by_qname.get(q, []))
                for q in qname_set
            )
            print(
                f"[prod-cache-stream] render_scope={render_scope} "
                f"qnames={len(qname_set)} entries={total_entries} "
                f"levers={dict(sorted(levers.items()))}",
                flush=True,
            )

        cache_dir_path: Path | None = None
        if cache_dir is not None:
            cache_dir_path = Path(cache_dir)
            cache_dir_path.mkdir(parents=True, exist_ok=True)

        sidecar_path: Path | None = (
            cache_dir_path / "activation_max_abs.json"
            if cache_dir_path is not None else None
        )

        # Partition render qnames by decoder-layer index.
        layer_pat = re.compile(rf'^{re.escape(layers_prefix)}(\d+)\.')
        layer_qnames: dict[int, list[str]] = {L: [] for L in range(num_layers)}
        head_qnames: list[str] = []
        for qname in qnames:
            m = layer_pat.match(qname)
            if m:
                L = int(m.group(1))
                if 0 <= L < num_layers:
                    layer_qnames[L].append(qname)
            else:
                head_qnames.append(qname)

        if head_qnames and progress:
            sample = head_qnames[:3]
            print(
                f"[prod-cache-stream] {len(head_qnames)} head qnames outside "
                f"decoder layers (always resident): {sample}"
                f"{'...' if len(head_qnames) > 3 else ''}",
                flush=True,
            )

        # Install activation collector on skeleton once.  Hooks fire per-sample
        # as each layer is called; other layers' hooks are silent (never called).
        collector = _LinearActivationCollector(
            skeleton,
            qnames=eligible_qnames,
            max_rows=max_act_rows,
            store_qnames=qnames_needing_activation,
            store_device=device,
            store_dtype=torch.float32,
        )
        collector.install()

        # Embed all N calibration samples (embed_tokens is always resident).
        N = calib_ids.size(0)
        T = calib_ids.size(1)
        with torch.no_grad():
            embed = base_model.embed_tokens
            hidden_states: list[torch.Tensor] = [
                embed(calib_ids[i:i + 1].to(device)).to(dtype=dtype)
                for i in range(N)
            ]

        position_ids = torch.arange(T, device=device).unsqueeze(0)
        position_embeddings = _compute_position_embeddings(
            base_model, hidden_states[0], position_ids)
        attention_mask = _make_causal_mask(T, device=device, dtype=dtype)

        # Load max_abs sidecar for resume support.
        activation_max_abs: dict[str, float] = {}
        if sidecar_path is not None and sidecar_path.is_file():
            try:
                activation_max_abs.update(json.loads(sidecar_path.read_text()))
                if progress:
                    print(
                        f"[prod-cache-stream] resume: loaded "
                        f"{len(activation_max_abs)} max_abs entries from sidecar",
                        flush=True,
                    )
            except Exception as e:
                if progress:
                    print(
                        f"[prod-cache-stream] sidecar load failed ({e}); recomputing",
                        flush=True,
                    )

        weights: dict = {}
        failed: dict = {}
        done = 0
        n_total = sum(
            len(render_formats_by_qname.get(q, []))
            for L in range(num_layers)
            for q in layer_qnames[L]
            if q in qname_set
        )

        for L in range(num_layers):
            layer_q = [q for q in layer_qnames.get(L, []) if q in qname_set]

            ctx.install(L)

            # Run all N samples through layer L; hooks collect activations.
            with torch.no_grad():
                for i in range(N):
                    extra_kw = (
                        profile.extra_layer_kwargs(
                            input_ids=calib_ids[i:i + 1].to(device))
                        if profile is not None else {}
                    )
                    out = _call_layer(
                        layers[L], hidden_states[i],
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **extra_kw,
                    )
                    hidden_states[i] = out

            if layer_q:
                # Joint NVFP4 globals — computed while layer weights are on GPU.
                layer_joint_globals: dict[str, torch.Tensor] = {}
                if needs_nvfp4_render:
                    from prismaquant.export_native_compressed import (
                        _compute_nvfp4_joint_global,
                    )
                    try:
                        layer_joint_globals = _compute_nvfp4_joint_global(
                            skeleton, {q: "NVFP4" for q in layer_q})
                    except Exception as e:
                        if progress:
                            print(
                                f"[prod-cache-stream] L{L}: joint global "
                                f"failed ({e!r}); using per-Linear globals",
                                flush=True,
                            )

                # Extract activations accumulated for this layer's qnames.
                layer_activations: dict[str, torch.Tensor] = {}
                for q in layer_q:
                    parts = collector.activations.get(q, [])
                    if parts:
                        layer_activations[q] = torch.cat(parts, dim=0)

                # Compute activation_max_abs with sibling unification.
                if needs_nvfp4_render:
                    from prismaquant.decision_units import fused_group_key
                    per_qname_max: dict[str, float] = {}
                    for q in layer_q:
                        if q in activation_max_abs:
                            per_qname_max[q] = activation_max_abs[q]
                            continue
                        t = collector._max_abs_tensors.get(q)
                        mx = float(t.item()) if t is not None else 0.0
                        if mx <= 0:
                            acts = layer_activations.get(q)
                            if acts is not None:
                                mx = float(acts.abs().max().item())
                        if mx > 0:
                            per_qname_max[q] = mx
                    grps: dict[str, list[str]] = {}
                    for q in per_qname_max:
                        try:
                            gk = fused_group_key(profile, q) if profile else q
                        except Exception:
                            gk = q
                        grps.setdefault(gk, []).append(q)
                    for gk, members in grps.items():
                        shared = max(per_qname_max[m] for m in members)
                        for m in members:
                            activation_max_abs[m] = shared

                # Render production weights — layer weights still on GPU.
                for q in layer_q:
                    mod = qname_to_module.get(q)
                    if mod is None:
                        continue
                    weight = mod.weight.data
                    joint = layer_joint_globals.get(q)
                    max_abs_val = activation_max_abs.get(q)
                    export_scale = (
                        (6.0 / max_abs_val)
                        if (max_abs_val is not None and max_abs_val > 0)
                        else None
                    )
                    for fmt in render_formats_by_qname.get(q, ()):
                        fmt_key = str(fmt).upper()
                        key = (q, fmt_key)
                        if key in weights:
                            done += 1
                            continue
                        if cache_dir_path is not None:
                            fname = _cache_weight_filename(q, fmt_key)
                            if (cache_dir_path / fname).is_file():
                                weights[key] = fname
                                done += 1
                                continue
                        try:
                            w_dq = render_production_weight(
                                weight,
                                _render_base_format(fmt_key),
                                qname=q,
                                activations=layer_activations,
                                levers=levers,
                                joint_global_real=joint,
                                input_global_scale=export_scale,
                            )
                        except Exception as e:
                            failed[key] = str(e)
                            if progress:
                                print(
                                    f"[prod-cache-stream] FAILED {q} @ {fmt}: {e}",
                                    flush=True,
                                )
                            done += 1
                            continue
                        _store_rendered_weight_entry(
                            weights=weights,
                            cache_dir_path=cache_dir_path,
                            qname=q,
                            fmt=fmt_key,
                            tensor=w_dq,
                            weight_dtype=weight.dtype,
                        )
                        del w_dq
                        done += 1
                        if progress and done % 25 == 0:
                            print(
                                f"[prod-cache-stream] {done}/{n_total}",
                                flush=True,
                            )

            ctx.unload(L)

            # Free this layer's activation entries from the accumulator.
            for q in layer_qnames.get(L, []):
                collector.activations.pop(q, None)
                collector._max_abs_tensors.pop(q, None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        collector.remove()

        # Render always-resident head qnames (lm_head is typically skipped).
        for qname in head_qnames:
            if qname not in qname_set:
                continue
            mod = qname_to_module.get(qname)
            if mod is None:
                continue
            weight = mod.weight.data
            for fmt in render_formats_by_qname.get(qname, ()):
                fmt_key = str(fmt).upper()
                key = (qname, fmt_key)
                if key in weights:
                    continue
                if cache_dir_path is not None:
                    fname = _cache_weight_filename(qname, fmt_key)
                    if (cache_dir_path / fname).is_file():
                        weights[key] = fname
                        continue
                try:
                    w_dq = render_production_weight(
                        weight,
                        _render_base_format(fmt_key),
                        qname=qname,
                        activations={},
                        levers=levers,
                        joint_global_real=None,
                        input_global_scale=None,
                    )
                except Exception as e:
                    failed[(qname, fmt_key)] = str(e)
                    continue
                _store_rendered_weight_entry(
                    weights=weights,
                    cache_dir_path=cache_dir_path,
                    qname=qname,
                    fmt=fmt_key,
                    tensor=w_dq,
                    weight_dtype=weight.dtype,
                )
                del w_dq

        # Persist max_abs sidecar for future resume runs.
        if sidecar_path is not None and activation_max_abs:
            sidecar_path.write_text(json.dumps(activation_max_abs, indent=2))

        if progress:
            print(
                f"[prod-cache-stream] rendered {len(weights)} (qname, fmt) entries "
                f"({len(failed)} failures)",
                flush=True,
            )

        cache = ProductionWeightCache(
            weights=weights,
            levers=dict(levers),
            activation_max_abs=activation_max_abs or None,
            failed=failed,
            cache_dir=str(cache_dir_path) if cache_dir_path is not None else None,
            metadata={
                "render_scope": render_scope,
                "requested_formats": list(requested_formats),
                "streaming": True,
                "num_layers": num_layers,
            },
        )
        return cache, qnames, skipped_list

    finally:
        shutil.rmtree(offload_folder, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build production δw cache")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--formats",
        default="NVFP4",
        help="Comma-separated formats to render. MXFP8 / BF16 cache is "
        "cheap compared with NVFP4, but MXFP8 still benefits from "
        "activation-weighted scale search when scale_sweep is enabled.",
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
        default="gptq,scale_sweep",
        help="Comma-separated levers to enable. Currently honored: "
        "{gptq, scale_sweep, act_clip_solver, fisher_gptq, fisher_clip, "
        "awq, smoothquant}.  "
        "act_clip_solver is PrismaClip, the production-rendered NVFP4 "
        "activation clipping solver. Requesting the internal "
        "NVFP4_CLIPPED cache format renders PrismaClip as a separate "
        "cache variant; layer configs still use ordinary NVFP4. "
        "fisher_clip is PrismaFisherClip: it "
        "uses h-detail per-token Fisher weights to audit clip candidates "
        "without enabling Fisher-weighted GPTQ; set "
        "PRISMAQUANT_PRISMAFISHERCLIP_MODE=veto or score for explicit "
        "ablations. "
        "AWQ and SmoothQuant require --render-scope=assignment because the "
        "fold scale is tied to the concrete format assignment.  Joint NVFP4 "
        "sibling globals + calibrated "
        "input_global_scale are computed unconditionally when NVFP4 is in "
        "the format menu. NVFP4 block scaling follows "
        "PRISMAQUANT_NVFP4_SCALE_RULE.",
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
        "weight NVFP4 GPTQ/scale-sweep and MXFP8 scale-sweep objectives. "
        "When 'fisher_clip' is enabled, they weight PrismaClip candidate "
        "scoring only.",
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

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formats = [f.strip().upper() for f in args.formats.split(",") if f.strip()]
    levers = {
        name: True for name in (
            x.strip() for x in args.enable.split(",")
        ) if name
    }

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
        # Route to streaming path when model exceeds available CUDA memory.
        _use_streaming = False
        if device.type == "cuda":
            _model_bytes = _estimate_model_bytes_from_index(staged)
            if _model_bytes > 0:
                _cuda_total = torch.cuda.get_device_properties(device).total_memory
                if _model_bytes > _cuda_total - 4 * 1024 ** 3:
                    _use_streaming = True
                    print(
                        f"[build-prod-cache] model {_model_bytes / (1024**3):.1f} GB > "
                        f"CUDA {_cuda_total / (1024**3):.1f} GB - 4 GB headroom; "
                        "routing to streaming production cache path",
                        flush=True,
                    )

        # render_assignment resolution is shared across both paths.
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
        if _use_streaming:
            halo_meta = {"mode": "off"}
            if args.halo_mode != "off":
                print(
                    "[build-prod-cache] WARNING: HALO not supported in streaming "
                    "production cache path; --halo-mode ignored",
                    flush=True,
                )
            if recache_assignment is not None:
                print(
                    "[build-prod-cache] WARNING: --recache-layer-config not "
                    "supported in streaming production cache path; ignoring",
                    flush=True,
                )
            cache, qnames, skipped = _fill_production_cache_streaming(
                model_path=args.model,
                staged=staged,
                calib_ids=calib_ids,
                device=device,
                dtype=dtype,
                formats=formats,
                levers=levers,
                max_act_rows=args.max_act_rows,
                cache_dir=args.cache_dir,
                render_assignment=render_assignment,
                skip_qnames=list(args.skip_qnames or []),
                h_detail_dir=args.h_detail_dir,
            )
            print(
                f"[build-prod-cache] {len(qnames)} quantizable Linears, "
                f"formats={formats}, levers={sorted(levers)}",
                flush=True,
            )
            if skipped:
                print(
                    f"[build-prod-cache] skipped {len(skipped)} qnames matching "
                    f"{list(args.skip_qnames or [])} (typically pinned-BF16 in "
                    f"polish): "
                    f"{skipped if len(skipped) <= 5 else skipped[:5] + ['...']}",
                    flush=True,
                )
        else:
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
        #
        # Packed MoE experts (3D tensors) are intentionally excluded from the
        # production weight cache — the export pipeline quantizes them directly
        # via _quantize_2d without reading the cache.  Filter the assignment
        # to only the eligible_qnames (nn.Linear modules) before checking
        # coverage so expert qnames don't produce false-positive misses.
        try:
            if render_assignment is not None:
                eligible_set = set(qnames)
                cacheable_assignment = {
                    q: fmt for q, fmt in render_assignment.items()
                    if q in eligible_set
                }
                n_skipped_experts = len(render_assignment) - len(cacheable_assignment)
                if n_skipped_experts:
                    print(
                        f"[build-prod-cache] coverage check: skipping "
                        f"{n_skipped_experts} packed-expert assignment entries "
                        f"(export handles them directly, not via cache)",
                        flush=True,
                    )
                _, missing = cache.assignment_keys(cacheable_assignment)
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
