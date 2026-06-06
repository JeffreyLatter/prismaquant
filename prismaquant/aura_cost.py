"""Aura cost: KL-adjoint per-Linear sensitivity surrogate.

Produces an allocator-compatible ``cost.pkl`` whose per-(Linear, format)
``predicted_dloss`` is the second-order KL contribution of quantizing that
Linear, measured against the **KL/Gauss-Newton Fisher** (not the CE empirical
Fisher) and the **production-rendered** weight error:

    predicted_dloss[i, f] = 0.5 * mean_k ( <gW_i^(k), dW_{i,f}> )^2

    gW_i^(k) = d/dW_i [ fisher_probe_scalar(logits; seed=k) ]   (kl_fisher probe;
               E_k[gW_i gW_i^T] = the layer Fisher w.r.t. the model KL)
    dW_{i,f} = Q_f(W_i) - W_i  (production-rendered error from ProductionWeight
               Cache when available, else the format-registry RTN error)

Why this is the right cost (rung-0 validated, 2026-06-04):
  * end-KL is locally a Fisher quadratic in the logit displacement, and the
    per-Linear unary KLs are **additive in fp32** (cross-terms ~0), so summing
    these per-Linear costs is a faithful end-KL surrogate -- the additive
    knapsack is sound once each per-Linear term is the KL-Fisher quantity.
  * <gW_i^(k), dW> = r_k . (J_i dY_i) is the probe projection of the propagated
    logit displacement; 0.5*mean_k(.)^2 is the unbiased estimator of
    0.5 * dY_i^T (J_i^T F J_i) dY_i = the unary KL contribution.
  * This is the analytic O(N) generalization of the validated 35B serving-unit
    propagated-sensitivity win (no hand-tuned scale, covers all Linears).

Reuses kl_fisher (probe), ProductionWeightCache (dW), format_registry (RTN
fallback), schemas (cost.pkl contract). Sets output_mse_measured=False so
allocator_candidates.cost_entry_predicted_dloss consumes predicted_dloss
directly. Measurement defaults to fp32 (the precision the additivity result
requires); memory-safe (one autograd graph at a time, watchdog-gated).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

import prismaquant.format_registry as fr
from prismaquant.kl_fisher import fisher_probe_scalar

SCHEMA = "prismaquant.aura_cost.v1"

# Passthrough formats -> zero predicted_dloss. This is the *passthrough rule*
# (see allocator_candidates.PASSTHROUGH_SOURCE_REQUIREMENTS): zero cost is
# correct only when the source weight already has the target precision --
#   BF16        is lossless iff the source weight dtype is bf16 (or lower);
#   FP8_SOURCE  is lossless iff the source weight is native fp8 (verbatim copy).
# Production models load bf16, so BF16 here is a true passthrough (0 error) and
# the zero-cost is exact. The only unsafe case is an fp32-source model loaded
# with --dtype float32: then BF16 is a *downcast* (~half a bf16-ulp of error),
# not a passthrough, and the unconditional zero would let the allocator pick
# BF16 as "free" when it is not. That case is opt-in guarded by
# compute_aura_cost(assert_bf16_passthrough=True); the default stays a no-op so
# the documented bit-identical regression output is unchanged. FP8_SOURCE has
# no source tensor in a bf16/fp32-loaded model, so its legality is gated by the
# allocator's passthrough-integrity check, not here; aura only declines to
# double-count it.
_ZERO_COST_FORMATS = {"BF16", "FP8_SOURCE"}


def _log(msg: str) -> None:
    print(f"[aura {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _free_gib() -> float:
    """Reclaimable-inclusive free memory in GiB.

    On the GB10/DGX Spark unified-memory box, CUDA and host share one physical
    pool, and clean page cache (model safetensors, cache shards) counts as
    'used' in ``torch.cuda.mem_get_info()`` even though the kernel reclaims it
    on demand. ``/proc/meminfo`` ``MemAvailable`` is the true 'can still
    allocate' headroom and is what should gate the watchdog -- gating on CUDA
    free aborts spuriously whenever a large file was just read. Fall back to
    the CUDA figure off-Linux / if /proc is unreadable."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)  # kB -> GiB
    except Exception:
        pass
    try:
        return torch.cuda.mem_get_info()[0] / (1024 ** 3)
    except Exception:
        return float("inf")


def _target_linears(
    model: nn.Module, *, include_lm_head: bool = False,
) -> dict[str, nn.Linear]:
    """Quantizable nn.Linear targets. lm_head is EXCLUDED by default (the
    profile pins it BF16). include_lm_head adds it so Aura can MEASURE its
    KL-sensitivity and let the allocator choose its format as a budget
    decision rather than a hardcoded pin -- the KL probe gradient flows
    directly into lm_head (it produces the logits), so its cost is
    measured the same way as any body Linear."""
    out: dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "lm_head" in name and not include_lm_head:
            continue
        if mod.weight.dim() == 2 and min(mod.weight.shape) >= 16:
            out[name] = mod
    return out


def _delta_w(
    name: str,
    fmt: str,
    weight: torch.Tensor,
    cache: object | None,
    *,
    strict: bool = False,
) -> torch.Tensor | None:
    """Q_f(W)-W: production-rendered error if the cache has it, else RTN.

    ``strict`` (require_production_cache): when a cache is supplied but lacks the
    rendered (name, fmt), fail fast with a clear coverage error instead of
    silently falling back to RTN -- so a 'production-faithful' run cannot quietly
    mix RTN deltas into the cost. Default off preserves the RTN fallback used by
    non-production ablations."""
    if cache is not None:
        try:
            rendered = cache.get(name, fr.canonical_format_name(fmt))
        except Exception:
            rendered = None
        if rendered is not None:
            return (rendered.to(weight.device, torch.float32) - weight.float())
        if strict:
            raise RuntimeError(
                f"require_production_cache: production-rendered weight missing "
                f"for ({name!r}, {fmt!r}); refusing silent RTN fallback. Build the "
                f"cache for this (Linear, format) or drop --require-production-cache.")
    spec = fr.get_format(fmt)
    qdq = getattr(spec, "quantize_dequantize", None)
    if qdq is None:
        return None
    try:
        return qdq(weight.float()) - weight.float()
    except Exception:
        return None


def _auto_n_chunks(
    linears: dict[str, nn.Linear],
    names: Sequence[str],
    min_free_gib: float,
    *,
    n_nonzero_fmts: int = 1,
    dw_bytes: int = 2,
    accurate_chunk_bytes: bool = False,
) -> int:
    """Pick the number of Linear chunks so peak memory stays under budget.

    Per chunk we hold dW_chunk (one bf16 delta per *nonzero* format, ~W/G each)
    + retained grads (one per weight at the model's param dtype, ~W/G) on top of
    the resident model, where W is the chunk's target-weight footprint. We size
    G so the per-chunk peak fits in (free - headroom), headroom covering the
    autograd graph and the watchdog floor. G=1 reproduces the legacy
    single-pass path exactly.

    ``_free_gib`` reads ``/proc/meminfo`` ``MemAvailable``, the correct 'can
    still allocate' signal on this GB10/DGX Spark *unified*-memory box (CUDA and
    host share one physical pool). On a *discrete* GPU MemAvailable is host RAM
    only and says nothing about VRAM headroom -- this sizing would be wrong
    there and would have to gate on ``torch.cuda.mem_get_info`` instead.

    Legacy (default) accounting hardcodes 2 bytes/weight and a single ~W/G dW
    term -- it silently assumes a bf16 model with one nonzero format, and
    under-counts by ~2x on the default fp32 load (4-byte weights+grads) or with
    multiple nonzero formats (one bf16 dW each), picking too few chunks and
    tripping the watchdog mid-run. ``accurate_chunk_bytes`` switches to the real
    footprint: grad bytes from the model param ``element_size()`` (4 for fp32,
    2 for bf16) plus ``n_nonzero_fmts * dw_bytes`` for the per-format bf16
    deltas. It only changes how many memory-bounded passes are taken; the
    numerical payload is bit-identical for any G, so it is purely an opt-in
    safety knob and never perturbs the cost output."""
    free = _free_gib()
    if free == float("inf"):
        return 1
    import math
    numel = sum(linears[n].weight.numel() for n in names)
    budget = max(free - (min_free_gib + 12.0), 4.0)
    if not accurate_chunk_bytes:
        # Legacy path, preserved bit-for-bit: 2 bytes/weight, peak ~ 2*W/G.
        wgib = numel * 2 / (1024 ** 3)
        return max(1, min(math.ceil(2.0 * wgib / budget), len(names)))
    # Accurate: grad/weight footprint follows the model param dtype; dW is one
    # bf16 (``dw_bytes``) delta per nonzero format. Peak over the resident model
    # per chunk = numel/G * (grad_bytes + n_nonzero_fmts * dw_bytes).
    grad_bytes = (
        next(iter(linears.values())).weight.element_size() if linears else 4
    )
    per_weight_bytes = grad_bytes + max(1, n_nonzero_fmts) * max(1, dw_bytes)
    peak_gib = numel * per_weight_bytes / (1024 ** 3)
    return max(1, min(math.ceil(peak_gib / budget), len(names)))


def compute_aura_cost(
    model: nn.Module,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    n_probes: int = 16,
    token_scope: str = "all",
    temperature: float = 1.0,
    production_cache: object | None = None,
    min_free_gib: float = 20.0,
    seed_base: int = 7000,
    n_linear_chunks: int = 0,
    assert_bf16_passthrough: bool = False,
    accurate_chunk_bytes: bool = False,
    require_production_cache: bool = False,
    dw_dtype: str = "bfloat16",
    include_lm_head: bool = False,
) -> dict:
    """Return a cost.pkl payload dict (stats + costs) for the allocator.

    ``n_linear_chunks`` bounds peak memory for large resident models: the
    target Linears are partitioned into G groups, and dW + retained grads are
    held for only one group at a time (peak ~ model + 2*model/G instead of
    3*model). The probe seeds and forwards are deterministic, so the per-Linear
    gradient a Linear receives is identical regardless of which group it lands
    in -- the chunked result is bit-identical to the single-pass (G=1) path,
    just computed in G memory-bounded passes. 0 = auto-size from free memory."""
    if n_probes < 1:
        raise ValueError(f"n_probes must be >= 1, got {n_probes!r}")
    _dw_torch_dtype = torch.float32 if str(dw_dtype) == "float32" else torch.bfloat16
    device = next(model.parameters()).device
    linears = _target_linears(model, include_lm_head=include_lm_head)
    names = list(linears.keys())
    fmts = [fr.canonical_format_name(f) for f in formats]
    nonzero_fmts = [f for f in fmts if f not in _ZERO_COST_FORMATS]
    # Passthrough-rule guard (opt-in; default off keeps the output byte-for-byte
    # identical). BF16 zero-cost is only valid when the source weight is already
    # bf16/fp16 -- on an fp32-source model loaded as fp32, casting W to BF16 is a
    # real downcast and the unconditional zero-cost is wrong. Catch that here
    # rather than silently mis-cost the format. (fp8 source can't be loaded as a
    # plain Linear weight, so an fp32 resident dtype never legitimizes BF16
    # zero-cost.)
    if assert_bf16_passthrough and "BF16" in fmts:
        src_dtype = next(model.parameters()).dtype
        if src_dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(
                f"assert_bf16_passthrough: BF16 zero-cost requires a bf16/fp16 "
                f"source weight (passthrough rule), but model params are "
                f"{src_dtype}. Loading as float32 makes BF16 a downcast, not a "
                f"passthrough -- drop BF16 from --formats or load the model as "
                f"bfloat16.")
    if n_linear_chunks <= 0:
        n_linear_chunks = _auto_n_chunks(
            linears, names, min_free_gib,
            n_nonzero_fmts=len(nonzero_fmts),
            dw_bytes=_dw_torch_dtype.itemsize,
            accurate_chunk_bytes=accurate_chunk_bytes,
        )
    n_linear_chunks = max(1, min(n_linear_chunks, len(names)))
    _log(f"targets={len(names)} formats={fmts} probes={n_probes} "
         f"dtype={next(model.parameters()).dtype} chunks={n_linear_chunks} "
         f"free={_free_gib():.1f}")

    for p in model.parameters():
        p.requires_grad_(False)

    # Partition Linears into G contiguous chunks. For each chunk we enable grad
    # on that chunk only, precompute its dW, run all K probes, project, free.
    chunks: list[list[str]] = [
        names[i::n_linear_chunks] for i in range(n_linear_chunks)
    ]
    chunks = [c for c in chunks if c]
    s2: dict[tuple[str, str], float] = {}
    g_trace: dict[str, float] = {}  # KL-Fisher weight-grad energy
    inv = 1.0 / float(n_probes)

    for ci, chunk in enumerate(chunks):
        for n in chunk:
            linears[n].weight.requires_grad_(True)
        # Precompute dW_{i,f} (fp32 delta, stored bf16) for this chunk only.
        dW: dict[tuple[str, str], torch.Tensor] = {}
        with torch.no_grad():
            for f in nonzero_fmts:
                for n in chunk:
                    d = _delta_w(n, f, linears[n].weight.data, production_cache,
                                 strict=require_production_cache)
                    if d is not None:
                        dW[(n, f)] = d.to(_dw_torch_dtype)  # dot upcasts to fp32
        for key in dW:
            s2.setdefault(key, 0.0)
        for n in chunk:
            g_trace.setdefault(n, 0.0)
        # dW is now materialized for this chunk; the cache's LRU-resident
        # rendered weights are no longer needed. Evict them (back to disk
        # paths) so they don't accumulate across chunks -- otherwise the
        # cache LRU holds chunk 1+2+3's weights on top of the model and the
        # watchdog trips by the last chunk. compact_for_pickle() resets the
        # LRU; empty_cache returns the freed segments to the OS pool.
        compact = getattr(production_cache, "compact_for_pickle", None)
        if callable(compact):
            try:
                compact()
            except Exception:
                pass
        elif production_cache is not None and ci == 0:
            # No disk-backed eviction (in-memory cache): rendered tensors the
            # cache holds in RAM persist across chunks, so the per-chunk memory
            # bound is NOT guaranteed. Warn once; a --cache-dir-backed cache is
            # required for large resident models.
            _log("WARNING: production cache has no compact_for_pickle "
                 "(in-memory); cross-chunk memory bound not guaranteed -- use a "
                 "disk-backed (--cache-dir) cache for large resident models.")
        torch.cuda.empty_cache()
        if len(chunks) > 1:
            _log(f"chunk {ci+1}/{len(chunks)}: {len(chunk)} Linears, "
                 f"dW pairs={len(dW)}; free={_free_gib():.1f}")

        # K probe backward passes; one autograd graph alive at a time (fresh
        # forward per probe). Grads retained for this chunk only.
        for k in range(n_probes):
            if _free_gib() < min_free_gib:
                raise RuntimeError(
                    f"free UMA {_free_gib():.1f} < floor {min_free_gib}; abort")
            for n in chunk:
                linears[n].weight.grad = None
            logits = model(calib_ids).logits
            probe = fisher_probe_scalar(
                logits, seed=seed_base + k, token_scope=token_scope,
                temperature=temperature, distribution="rademacher",
            )
            probe.backward()
            with torch.no_grad():
                for n in chunk:
                    g = linears[n].weight.grad
                    if g is None:
                        continue
                    gf = g.float()
                    g_trace[n] += float((gf * gf).sum().item())
                    for f in nonzero_fmts:
                        key = (n, f)
                        if key in dW:
                            s2[key] += float(
                                (gf * dW[key].float()).sum().item()) ** 2
                    linears[n].weight.grad = None
            del logits, probe
            torch.cuda.empty_cache()
            if (k + 1) % 8 == 0:
                _log(f"  chunk {ci+1}/{len(chunks)} probe {k+1}/{n_probes}; "
                     f"free={_free_gib():.1f}")
        # Release this chunk's dW + grad enablement before the next chunk.
        del dW
        for n in chunk:
            linears[n].weight.grad = None
            linears[n].weight.requires_grad_(False)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Assemble payload.
    inv = 1.0 / float(n_probes)
    stats: dict[str, dict] = {}
    costs: dict[str, dict] = {}
    for n in names:
        mod = linears[n]
        stats[n] = {
            "h_trace": g_trace[n] * inv,  # KL-Fisher weight-grad energy
            "n_params": int(mod.weight.numel()),
            "in_features": int(getattr(mod, "in_features", mod.weight.shape[1])),
            "out_features": int(getattr(mod, "out_features", mod.weight.shape[0])),
            "n_probes": int(n_probes),
        }
        costs[n] = {}
        for f in fmts:
            if f in _ZERO_COST_FORMATS:
                # Passthrough rule: zero error iff the source already has this
                # precision (bf16 source for BF16, fp8 source for FP8_SOURCE).
                # See _ZERO_COST_FORMATS and the assert_bf16_passthrough guard
                # above for the fp32-source downcast caveat.
                costs[n][f] = {
                    "predicted_dloss": 0.0,
                    "output_mse_measured": False,
                    "cost_source": "aura_passthrough_zero",
                }
                continue
            key = (n, f)
            if key not in s2:
                continue  # format illegal / no dW for this Linear
            costs[n][f] = {
                "predicted_dloss": 0.5 * inv * s2[key],
                "output_mse_measured": False,
                "cost_source": "aura",
            }
    return {
        "schema": SCHEMA,
        "n_probes": n_probes,
        "formats": fmts,
        "token_scope": token_scope,
        "stats": stats,
        "costs": costs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aura KL-adjoint allocator cost")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--formats", default="NVFP4,FP8_DYNAMIC,BF16")
    p.add_argument("--production-cache", default=None,
                   help="ProductionWeightCache pickle for production-faithful dW")
    p.add_argument("--n-probes", type=int, default=16)
    p.add_argument("--n-calib-samples", type=int, default=4)
    p.add_argument("--calib-seqlen", type=int, default=256)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--token-scope", default="all")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--n-linear-chunks", type=int, default=0,
                   help="Partition Linears into G memory-bounded groups "
                        "(peak ~ model + 2*model/G). 0 = auto-size from free "
                        "UMA. G=1 is the legacy single-pass path. Required >1 "
                        "for large resident models (e.g. 27B on a 121GB box).")
    p.add_argument("--min-free-gib", type=float, default=20.0)
    p.add_argument("--seed-base", type=int, default=7000,
                   help="Base seed for the Rademacher KL probes. Vary it "
                        "(same calibration) to test probe-direction stability "
                        "of the allocation -- i.e. whether K probes suffice.")
    p.add_argument("--assert-bf16-passthrough", action="store_true",
                   help="Fail fast if BF16 is in --formats but the model is "
                        "loaded fp32 (BF16 would be a downcast, not a lossless "
                        "passthrough, so its zero-cost would be wrong). Off by "
                        "default; current behavior is unchanged when omitted.")
    p.add_argument("--accurate-chunk-bytes", action="store_true",
                   help="Size --n-linear-chunks=0 auto-chunking from the real "
                        "per-weight footprint: grad bytes from the model param "
                        "element_size() (4 for fp32, 2 for bf16) + one bf16 dW "
                        "per nonzero format. The legacy default assumes 2 "
                        "bytes/weight and a single dW, under-counting ~2x on the "
                        "default fp32 load and tripping the watchdog. Off by "
                        "default; only changes the pass count, never the output "
                        "(bit-identical for any G).")
    p.add_argument("--require-production-cache", action="store_true",
                   help="Fail fast if the production cache lacks a rendered "
                        "(Linear, format); refuse silent RTN fallback. Off by "
                        "default. Use for production-faithful cost runs.")
    p.add_argument("--dw-dtype", default="bfloat16",
                   choices=["bfloat16", "float32"],
                   help="Storage dtype for the dW=Q_f(W)-W error vector. Default "
                        "bfloat16 (validated: bf16-vs-fp32 Aura Spearman 0.997); "
                        "float32 for exact fidelity at 2x dW memory.")
    p.add_argument("--include-lm-head", action="store_true",
                   help="Also measure lm_head (normally pinned BF16) so the "
                        "allocator can choose its format by budget-value rather "
                        "than a hardcoded pin. dW falls back to RTN if the cache "
                        "lacks a rendered lm_head.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.calibration_data import load_wikitext_calibration_windowed
    # Reuse the pipeline's text-only stager so multimodal (VL) checkpoints
    # (e.g. Qwen3.6-27B Qwen3_5ForConditionalGeneration) load as a CausalLM
    # with tensor names that match the production cache keys. No-op on
    # pure-text checkpoints.
    from prismaquant.build_rtn_cache import stage_multimodal

    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    staged, _cleanup = stage_multimodal(args.model)
    local_only = Path(staged).exists()
    _log(f"loading {args.model} (staged={staged}) dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=local_only)
    load_kwargs = dict(
        dtype=dt, trust_remote_code=True, local_files_only=local_only,
        attn_implementation="eager",
    )
    if args.device.startswith("cuda"):
        load_kwargs["device_map"] = args.device
    try:
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
    except ValueError as exc:
        if "accelerate" not in str(exc):
            raise
        load_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        model.to(args.device)
    model.eval()
    calib = load_wikitext_calibration_windowed(
        tok, args.n_calib_samples, args.calib_seqlen, split=args.calib_split,
    ).to(args.device)

    cache = None
    if args.production_cache:
        with open(args.production_cache, "rb") as fh:
            cache = pickle.load(fh)
        _log(f"loaded production cache: {args.production_cache}")

    payload = compute_aura_cost(
        model, calib, [f.strip() for f in args.formats.split(",") if f.strip()],
        n_probes=args.n_probes, token_scope=args.token_scope,
        temperature=args.temperature, production_cache=cache,
        min_free_gib=args.min_free_gib, n_linear_chunks=args.n_linear_chunks,
        seed_base=args.seed_base,
        assert_bf16_passthrough=args.assert_bf16_passthrough,
        accurate_chunk_bytes=args.accurate_chunk_bytes,
        require_production_cache=args.require_production_cache,
        dw_dtype=args.dw_dtype,
        include_lm_head=args.include_lm_head,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fh:
        pickle.dump(payload, fh)
    nz = sum(1 for n in payload["costs"] for f in payload["costs"][n]
             if payload["costs"][n][f].get("predicted_dloss", 0.0) > 0)
    _log(f"wrote {args.output}: {len(payload['costs'])} Linears, {nz} non-zero cost entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
