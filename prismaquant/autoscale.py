"""Hardware-adaptive memory sizing for the PrismaQuant pipeline.

Two knobs the probe/cost passes care about:

  1. `layers_per_shard` — how many decoder layers get their Fisher stats
     accumulated in a single reverse sweep. Bigger shards = fewer sweeps
     through the full model = faster probe, but each shard needs more
     gradient + retained-activation memory.

  2. `cache_headroom_gb` — safety margin subtracted from available RAM
     when sizing the streaming layer cache. Lower headroom = bigger
     cache = fewer evictions = fewer `torch.cuda.empty_cache()` stalls
     on UMA hosts, but less slack for autograd spikes.

Both defaults were historically tuned for a 35B-A3B MoE on a 128 GB
Spark. Dense-27B / 122B-A10B / etc. want different values. This module
derives them from the actual checkpoint + host at runtime.

The heuristic is deliberately simple:

    per_layer_bytes(shard) ≈ weight + activations + gradients
    available = free_RAM - safety
    reserved_for_cache = num_layers * per_layer_weight   # hold all layers ⇒ no evictions
    layers_per_shard = (available - reserved_for_cache) / per_layer_bytes

and clamped to [1, num_layers]. Explicit env overrides always win.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


DEFAULT_SAFETY_GB = 20.0     # slack above the committed estimate. NEVER rely on
                             # swap — kernel OOM-kills BEFORE swap fills on many
                             # Linux configs.
DEFAULT_ACT_MULT = 12        # multiplier in (N*T*hidden*dtype*K) per tracked
                             # layer. Captures backward transient scratch.
# Prefetch-window size for the streaming tier of pick_layers_per_shard:
# 4 concurrent prefetch reads + a completed-ahead margin of 4.
STREAMING_CACHE_WINDOW_LAYERS = 8

DEFAULT_DTYPE_BYTES = 2      # bf16
# Observed on Qwen3.6-27B dense: gradient checkpointing retains activations
# at ~sqrt(n_layers) boundaries, so the full autograd graph adds a
# per-layer-mix overhead independent of how many layers are tracked. Plus
# HF transformers wrappers, tokenizer caches, and Python heap contribute a
# roughly model-independent floor. Empirically ~35 GB at nsamples=32,
# seqlen=1024, hidden=5120. Scale by N*T*hidden so the term tracks
# calibration size.
DEFAULT_FULL_GRAPH_ACT_MULT = 48   # 64 layers × sqrt ≈ 8 × 6 (per-layer-mix overshoot)
DEFAULT_FIXED_OVERHEAD_GB = 15.0   # HF transformers + tokenizer + Python heap floor


def _num_layers(cfg: dict) -> int:
    tc = cfg.get("text_config") or cfg
    return int(tc.get("num_hidden_layers")
               or tc.get("n_layer")
               or cfg.get("num_hidden_layers", 0))


def _hidden_size(cfg: dict) -> int:
    tc = cfg.get("text_config") or cfg
    return int(tc.get("hidden_size")
               or tc.get("n_embd")
               or cfg.get("hidden_size", 0))


def _act_width(cfg: dict) -> int:
    """Widest per-Linear activation the probe/cost retains.

    The retained-activation estimate must track the *widest* projection
    activation a layer holds, not just ``hidden_size``. A transformer MLP's
    ``down_proj`` reads an ``intermediate_size``-wide input, and the cost
    step's batched render materializes ``intermediate_size``-wide outputs
    (gate/up) in fp32 scratch. On models where ``intermediate_size`` ≫
    ``hidden_size`` (Gemma4-31B: 21504 vs 5376, 4×) sizing on ``hidden_size``
    undershoots host RAM ~4× and the watchdog aborts the shard.

    Returns ``max(hidden, ffn, moe_ffn)`` so the estimate is governed by the
    true widest activation. Collapses to ``hidden_size`` when no FFN width is
    declared (== hidden for plain models)."""
    tc = cfg.get("text_config") or cfg
    hidden = _hidden_size(cfg)
    widths = [hidden]
    for key in ("intermediate_size", "moe_intermediate_size",
                "ffn_dim", "n_inner", "shared_expert_intermediate_size"):
        v = tc.get(key) or cfg.get(key)
        if v:
            try:
                widths.append(int(v))
            except (TypeError, ValueError):
                pass
    return max(w for w in widths if w > 0) if any(w > 0 for w in widths) else hidden


# Per-expert routed-expert tensor qnames (`...experts.<id>....`). Matches
# both live (`model.layers.N.mlp.experts.7.gate_proj.weight`) and DSv4
# checkpoint (`layers.N.ffn.experts.7.w1.weight`) naming.
_EXPERT_TENSOR_RE = re.compile(r"\.experts\.\d+\.")


def declared_expert_dtype_covers(name: str) -> bool:
    """Whether the checkpoint's `expert_dtype` declaration covers `name`.

    **ROUTED experts only — verified against the real checkpoint.** The
    declaration reads like a statement about all of a layer's experts, and
    this predicate used to widen to `mlp.shared_experts.*` on that
    reasoning. The real `deepseek-ai/DeepSeek-V4-Flash` headers say
    otherwise (safetensors metadata, four shards spanning the model):

        layers.N.ffn.experts.{i}.w{1,2,3}.weight   I8        <- nibble-packed
        layers.N.ffn.experts.{i}.w{1,2,3}.scale    F8_E8M0
        layers.N.ffn.shared_experts.w{1,2,3}.weight  F8_E4M3 <- block-FP8
        layers.N.ffn.shared_experts.w{1,2,3}.scale   F8_E8M0

    i.e. the shared expert is ordinary block-FP8, 2304/2304 routed-expert
    weights are I8 and 9/9 shared-expert weights are F8_E4M3. The authors'
    own converter agrees and is the tie-breaker: `inference/convert.py`
    gates the fp4 path on ``"experts" in name and dtype == torch.int8``, so
    an F8_E4M3 shared expert never enters it.

    Widening to shared experts would therefore send a block-FP8 tensor into
    the MXFP4 decode, where `_check_mxfp4_packed_grid` refuses a non-int8
    weight — a hard DSv4 load failure. Keep this routed-only.

    Nothing here inspects a tensor's shape or dtype: the trigger stays the
    config declaration (`declared_fp4_expert_dtype`) and the packed layout
    stays a hard assertion after the fact
    (`layer_streaming._check_mxfp4_packed_grid`). Non-expert tensors
    (attention projections, the router gate, norms) and the shared expert
    keep the block-FP8 dequant path and its `_check_fp8_scale_grid`
    assertion.
    """
    return bool(_EXPERT_TENSOR_RE.search(name))


# safetensors dtype names for a 1-byte integer plane — what a nibble-packed
# MXFP4 expert weight ships as. Both spellings must be priced the same way
# the decode treats them: `layer_streaming._check_mxfp4_packed_grid` accepts
# int8 *and* uint8 nibble-packs, so sizing only "I8" would leave a U8
# checkpoint undercounted 4x.
_PACKED_BYTE_DTYPES = frozenset({"I8", "U8"})


def declared_fp4_expert_dtype(model_path: str) -> bool:
    """True when the checkpoint config *explicitly* declares packed-FP4
    routed experts (DSv4-Flash: top-level `expert_dtype: "fp4"` alongside a
    block-FP8 `quantization_config`; the MXFP4 scale siblings are E8M0).

    This declaration — never a tensor-shape heuristic — is what gates the
    streaming loader's MXFP4 decode (`layer_streaming` step 3b) and what
    the resident-size estimators key on: a nibble-packed I8 expert byte
    dequants to 2 logical elements of the execution dtype."""
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return False
        tc = cfg.get("text_config")
        tc = tc if isinstance(tc, dict) else {}
        val = cfg.get("expert_dtype") or tc.get("expert_dtype") or ""
    except Exception:
        # Absent, unreadable, or unexpectedly-shaped config: not declared.
        # Sizing wrongly here silently mis-budgets the streaming cache, so
        # every failure mode resolves to "verbatim" rather than to a guess.
        return False
    return str(val).lower() in {"fp4", "mxfp4", "mx_fp4"}


def _safetensors_source_float_bytes(dtype_name: str) -> int | None:
    """On-disk bytes/element for a safetensors *floating* dtype name;
    None for non-float dtypes (kept verbatim by the streaming loader)."""
    dt = str(dtype_name).upper()
    if dt.startswith("F8"):
        return 1
    if dt in ("F16", "BF16"):
        return 2
    if dt == "F32":
        return 4
    if dt == "F64":
        return 8
    return None


def _shard_resident_bytes(path: Path, dtype_bytes: int,
                          fp4_experts: bool = False) -> int:
    """Resident bytes for one safetensors shard after streaming load.

    `_read_layer_to_device` casts every floating tensor to the execution
    dtype (native-FP8 weights are block-dequanted to bf16), so resident
    bytes per float element = ``dtype_bytes`` regardless of on-disk
    element size — the same rule `streaming_model._estimate_layer_cache_bytes`
    applies per tensor. fp8-native checkpoints (1 byte/elem on disk)
    therefore occupy 2x their disk size in the layer cache; sizing from
    raw file size undercounts them 2x and blows the memory budget.

    ``fp4_experts`` is the checkpoint's explicit packed-FP4 expert
    declaration (`declared_fp4_expert_dtype`): expert I8/U8 tensors (routed
    and shared alike, see `declared_expert_dtype_covers`) are then MXFP4
    nibble-packs that dequant to TWO logical elements of the execution
    dtype per on-disk byte (a 4x undercount at bf16 if sized verbatim).
    Other non-float dtypes stay verbatim.

    Parses the safetensors JSON header directly (stdlib-only; no tensor
    data is read). Raises on malformed files; the caller falls back to
    the raw file size."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        if header_len <= 0 or header_len > 512 * 1024 ** 2:
            raise ValueError(
                f"implausible safetensors header length {header_len} in {path}"
            )
        header = json.loads(f.read(header_len))
    total = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        off = meta["data_offsets"]
        nbytes = int(off[1]) - int(off[0])
        dtype_name = str(meta.get("dtype", "")).upper()
        if (fp4_experts and dtype_name in _PACKED_BYTE_DTYPES
                and declared_expert_dtype_covers(key)):
            total += nbytes * 2 * int(dtype_bytes)
            continue
        src_bytes = _safetensors_source_float_bytes(dtype_name)
        if src_bytes is None:
            total += nbytes
        else:
            total += (nbytes // src_bytes) * int(dtype_bytes)
    return total


def _model_resident_weight_bytes(model_path: str, dtype_bytes: int) -> int:
    """Sum of resident (post-cast/dequant) bytes across all *.safetensors
    blobs — dtype-aware, see `_shard_resident_bytes`. Falls back to the
    raw blob size per shard when a header can't be parsed, and to 0 if
    the dir doesn't exist yet."""
    p = Path(model_path)
    if not p.exists():
        return 0
    fp4_experts = declared_fp4_expert_dtype(model_path)
    total = 0
    for f in p.glob("*.safetensors"):
        try:
            total += _shard_resident_bytes(f, dtype_bytes, fp4_experts)
        except Exception:
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _available_ram_bytes() -> int:
    """Free RAM right now. On UMA (Grace-Blackwell) this is the shared
    LPDDR5X pool that both CPU and GPU draw from — same number matters
    for CUDA and host work."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 64 * 1024 ** 3  # conservative fallback


def estimate_per_layer_bytes(
    model_path: str,
    num_layers: int,
    hidden_size: int,
    nsamples: int,
    seqlen: int,
    dtype_bytes: int = DEFAULT_DTYPE_BYTES,
    act_mult: int = DEFAULT_ACT_MULT,
    act_width: int | None = None,
) -> tuple[int, int]:
    """Return `(per_layer_weight_bytes, per_layer_active_shard_bytes)`.

    - weight bytes: *resident* size / num_layers, minus head/embed
      approximation. Resident is dtype-aware: floating checkpoint tensors
      cast to the execution dtype at load, so fp8-native sources dequant
      to bf16 in the layer cache (2 bytes/elem resident, not the 1
      byte/elem on disk — sizing from disk undercounted those 2x).
    - active_shard bytes: gradients (~weight) + retained activations
      (N·T·act_width·dtype·act_mult)

    `act_width` is the widest per-Linear activation the layer retains —
    `max(hidden, intermediate)` (see `_act_width`). Defaults to `hidden_size`
    for back-compat; pass the FFN-aware width so large-MLP models size right.
    """
    total_resident = _model_resident_weight_bytes(model_path, dtype_bytes)
    if total_resident > 0 and num_layers > 0:
        # subtract a conservative 10% for non-layer weights (embed, lm_head, norms)
        body_bytes = int(total_resident * 0.90)
        per_layer_weight = body_bytes // num_layers
    else:
        per_layer_weight = 1 * 1024 ** 3  # 1 GB fallback

    grad_bytes = per_layer_weight  # same shape, same dtype
    width = act_width if act_width else hidden_size
    act_bytes = nsamples * seqlen * width * dtype_bytes * act_mult
    per_layer_active = grad_bytes + act_bytes
    return per_layer_weight, per_layer_active


def pick_layers_per_shard(
    model_path: str,
    *,
    nsamples: int,
    seqlen: int,
    dtype_bytes: int = DEFAULT_DTYPE_BYTES,
    act_mult: int = DEFAULT_ACT_MULT,
    safety_gb: float = DEFAULT_SAFETY_GB,
    full_graph_act_mult: int = DEFAULT_FULL_GRAPH_ACT_MULT,
    fixed_overhead_gb: float = DEFAULT_FIXED_OVERHEAD_GB,
    available_ram_bytes: int | None = None,
    hold_all_layers_in_cache: bool = True,
    default: int = 2,
) -> tuple[int, dict]:
    """Pick LAYERS_PER_SHARD from host memory + model size.

    Returns `(lps, diagnostics)` so callers can log the derivation.

    `hold_all_layers_in_cache=True` reserves enough RAM for the layer
    cache to fit every decoder layer (zero evictions → no empty_cache
    stalls). Falls back to holding half the layers if that leaves
    too little for shard work.
    """
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return default, {"reason": "no config.json", "lps": default}
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_layers = _num_layers(cfg)
    hidden = _hidden_size(cfg)
    if n_layers <= 0 or hidden <= 0:
        return default, {"reason": "missing layer/hidden in config", "lps": default}

    per_layer_weight, per_layer_active = estimate_per_layer_bytes(
        model_path, n_layers, hidden, nsamples, seqlen,
        dtype_bytes=dtype_bytes, act_mult=act_mult, act_width=_act_width(cfg),
    )
    avail = available_ram_bytes if available_ram_bytes is not None else _available_ram_bytes()
    safety = int(safety_gb * 1024 ** 3)

    if hold_all_layers_in_cache:
        cache_reserve = n_layers * per_layer_weight
    else:
        # Streaming tier. LRU under a cyclic layer sweep yields ZERO
        # reuse whenever the cache cannot hold the full cycle — a
        # half-model reserve buys ~nothing (measured 9-11% hit rate on
        # Laguna-117B) while starving shard width down to lps=1, which
        # multiplies the number of full-model sweeps. Reserve only a
        # prefetch window deep enough to overlap reads with compute
        # (workers in flight + completed-ahead margin) and spend the
        # rest of RAM on layers-per-shard: each extra layer per shard
        # removes an entire model sweep from the phase-3 schedule.
        cache_reserve = (
            min(n_layers, STREAMING_CACHE_WINDOW_LAYERS) * per_layer_weight)

    # Full-graph checkpointed activations: autograd retains activations
    # at ~sqrt(n_layers) boundaries across ALL layers, not just tracked
    # ones — this is fixed memory that any shard incurs. Plus HF /
    # tokenizer / Python overhead floor.
    full_graph_act = nsamples * seqlen * hidden * dtype_bytes * full_graph_act_mult
    overhead = int(fixed_overhead_gb * 1024 ** 3)

    shard_budget = avail - safety - cache_reserve - full_graph_act - overhead
    # If reserving the full cache leaves too little, fall back to half-cache
    if shard_budget < per_layer_active and hold_all_layers_in_cache:
        return pick_layers_per_shard(
            model_path, nsamples=nsamples, seqlen=seqlen,
            dtype_bytes=dtype_bytes, act_mult=act_mult,
            safety_gb=safety_gb,
            full_graph_act_mult=full_graph_act_mult,
            fixed_overhead_gb=fixed_overhead_gb,
            available_ram_bytes=avail,
            hold_all_layers_in_cache=False, default=default,
        )
    shard_budget = max(shard_budget, per_layer_active)  # never below 1 layer

    lps = max(1, int(shard_budget // per_layer_active))
    lps = min(lps, n_layers)

    return lps, {
        "lps": lps,
        "n_layers": n_layers,
        "hidden": hidden,
        "per_layer_weight_gb": per_layer_weight / 1024 ** 3,
        "per_layer_active_gb": per_layer_active / 1024 ** 3,
        "full_graph_act_gb": full_graph_act / 1024 ** 3,
        "fixed_overhead_gb": fixed_overhead_gb,
        "available_gb": avail / 1024 ** 3,
        "safety_gb": safety_gb,
        "cache_reserve_gb": cache_reserve / 1024 ** 3,
        "shard_budget_gb": shard_budget / 1024 ** 3,
        "hold_all_layers": hold_all_layers_in_cache,
    }


def pick_cache_headroom_gb(
    model_path: str,
    *,
    safety_gb: float = DEFAULT_SAFETY_GB,
    layers_per_shard: int = 1,
    nsamples: int = 32,
    seqlen: int = 1024,
    fixed_overhead_gb: float = DEFAULT_FIXED_OVERHEAD_GB,
    dtype_bytes: int = DEFAULT_DTYPE_BYTES,
    default: float = 75.0,
) -> tuple[float, dict]:
    """Pick `cache_headroom_gb` so the layer cache gets (available - headroom)
    bytes for fitting decoder layers. Returns `(headroom_gb, diagnostics)`.

    Headroom is the max of two non-overlapping phases:

    - Phase-1 forward pass: device_acts list (all layer outputs on device,
      (n_layers+1) tensors) plus the simultaneous host copy being built by
      [t.cpu() for t in device_acts].  On UMA both draw from the same pool.
    - Shard phase (backward sweeps): shard_working + full_graph_act overhead.

    The binding constraint (larger headroom) wins.
    """
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return default, {"reason": "no config.json", "headroom_gb": default}
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_layers = _num_layers(cfg)
    hidden = _hidden_size(cfg)
    if n_layers <= 0 or hidden <= 0:
        return default, {"reason": "missing layer/hidden", "headroom_gb": default}

    _, per_layer_active = estimate_per_layer_bytes(
        model_path, n_layers, hidden, nsamples, seqlen, act_width=_act_width(cfg),
    )
    shard_working_bytes = layers_per_shard * per_layer_active

    # Phase-1 peak: device_acts (accumulates all layer outputs on device)
    # + host copy being built simultaneously in [t.cpu() for t in device_acts].
    # Factor of 2 captures both sides alive at the same time.
    phase1_single = (n_layers + 1) * nsamples * seqlen * hidden * dtype_bytes
    phase1_peak_bytes = 2 * phase1_single

    fixed_bytes = int(fixed_overhead_gb * 1024 ** 3)
    safety_bytes = int(safety_gb * 1024 ** 3)

    shard_headroom = shard_working_bytes + fixed_bytes + safety_bytes
    phase1_headroom = phase1_peak_bytes + fixed_bytes + safety_bytes
    headroom_bytes = max(shard_headroom, phase1_headroom)
    headroom_gb = headroom_bytes / 1024 ** 3
    return headroom_gb, {
        "headroom_gb": headroom_gb,
        "shard_working_gb": shard_working_bytes / 1024 ** 3,
        "phase1_device_acts_gb": phase1_single / 1024 ** 3,
        "phase1_peak_gb": phase1_peak_bytes / 1024 ** 3,
        "fixed_overhead_gb": fixed_overhead_gb,
        "safety_gb": safety_gb,
        "layers_per_shard": layers_per_shard,
        "binding_constraint": "phase1" if phase1_headroom >= shard_headroom else "shard",
    }


def autoscale(
    model_path: str,
    *,
    nsamples: int,
    seqlen: int,
    layers_per_shard_env: str | int | None = None,
    cache_headroom_gb_env: str | float | None = None,
    safety_gb: float = DEFAULT_SAFETY_GB,
) -> tuple[int, float, dict]:
    """Compute `(layers_per_shard, cache_headroom_gb)` from model + host.

    Explicit env overrides win:
      - `layers_per_shard_env` (int or stringified int) skips LPS autoscale
      - `cache_headroom_gb_env` (float or stringified float) skips headroom autoscale

    Use `"auto"` or `None` to request autoscale.
    """
    diag: dict = {}

    # Parse LPS override
    lps: int
    if layers_per_shard_env in (None, "", "auto", "AUTO"):
        lps, lps_diag = pick_layers_per_shard(
            model_path, nsamples=nsamples, seqlen=seqlen, safety_gb=safety_gb,
        )
        diag["lps_autoscaled"] = lps_diag
    else:
        lps = int(layers_per_shard_env)
        diag["lps_source"] = f"explicit={lps}"

    # Parse headroom override
    headroom: float
    if cache_headroom_gb_env in (None, "", "auto", "AUTO"):
        headroom, hr_diag = pick_cache_headroom_gb(
            model_path, safety_gb=safety_gb,
            layers_per_shard=lps, nsamples=nsamples, seqlen=seqlen,
        )
        diag["headroom_autoscaled"] = hr_diag
    else:
        headroom = float(cache_headroom_gb_env)
        diag["headroom_source"] = f"explicit={headroom}"

    return lps, headroom, diag


if __name__ == "__main__":
    # CLI usage: python -m prismaquant.autoscale <model_path> [--nsamples N --seqlen T]
    import argparse
    ap = argparse.ArgumentParser(description="Print autoscaled memory knobs.")
    ap.add_argument("model_path")
    ap.add_argument("--nsamples", type=int, default=int(os.environ.get("NSAMPLES", 32)))
    ap.add_argument("--seqlen", type=int, default=int(os.environ.get("SEQLEN", 1024)))
    ap.add_argument("--safety-gb", type=float, default=DEFAULT_SAFETY_GB)
    ap.add_argument("--lps", default=os.environ.get("LAYERS_PER_SHARD"))
    ap.add_argument("--headroom", default=os.environ.get("CACHE_HEADROOM_GB"))
    args = ap.parse_args()

    lps, hr, diag = autoscale(
        args.model_path,
        nsamples=args.nsamples, seqlen=args.seqlen,
        layers_per_shard_env=args.lps,
        cache_headroom_gb_env=args.headroom,
        safety_gb=args.safety_gb,
    )
    print(f"LAYERS_PER_SHARD={lps}")
    print(f"CACHE_HEADROOM_GB={hr:.1f}")
    print(json.dumps(diag, indent=2))
