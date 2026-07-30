"""Exact on-disk artifact footprint for a quantization assignment.

The fit-the-card bit-rate selector (see ``saturation_select.select_under_byte
_budget``) needs to map an allocation -> the GB the exported compressed-tensors
checkpoint will occupy on disk, *before* paying for an export. This module is
that map, and it is exact (not the handover's hand-fit ``fixed_floor + 3.04*bpp``
linear approximation): it reproduces the real exported
``model.safetensors.index.json`` ``metadata.total_size`` to 0.00% on the
Qwen3.6-27B aura / baseline / aura_525 artifacts.

The accounting is the same identity the streaming exporter ships:

    artifact_bytes = floor_bytes + Σ_reencoded memory_bytes_for_shape(shape, fmt)
    floor_bytes    = source_checkpoint_total_bytes
                     − Σ_reencoded n_params · source_bytes_per_param

i.e. every quantizable Linear in the assignment is *re-encoded* from its source
precision into its chosen format (``Σ memory_bytes_for_shape`` == the allocator's
``bits_total_with_aux / 8``), and everything else — embeddings, lm_head (when
pinned), every norm/bias/rotary buffer, vision/MTP sidecars kept at source
precision — stays verbatim at its source byte size. The floor is therefore the
*residual* (source total minus the source size of the re-encoded tensors), which
is why it needs **no checkpoint-name matching**: the multimodal / vLLM name
remap (``model.language_model.layers...`` on disk vs ``model.layers...`` in the
layer_config) never has to be reconciled. Only two scalars are read from the
checkpoint — the grand total bytes and the source bytes-per-param — both via the
safetensors header (``data_offsets``), with no weight load and no torch.

``memory_bytes_for_shape`` already counts scale/zero-point overhead per format
(NVFP4's fp8 block scale per group-of-16, FP8's per-row fp32 scale, FP8_SOURCE's
128×128 block scale), so the body term is exact per shape rather than via a
nominal scalar bpp. Packed-MoE experts (3D shape) are handled by feeding the
``(num_experts, out, in)`` shape through the same primitive. NVFP4 additionally
ships two fp32 global sidecars per emitted 2-D Linear (``weight_global_scale``
+ ``input_global_scale``, 8 bytes) that ``memory_bytes_for_shape`` does not
count; :func:`nvfp4_global_sidecar_bytes` adds them (per expert × on-disk
projection for packed 3-D tensors).
"""
from __future__ import annotations

import glob
import json
import os
import struct
from typing import Iterable, Mapping

from . import format_registry as fr
from .allocator_solver import _shape_from_stats

# safetensors header dtype -> bytes per element (header carries the source
# dtype string; we only need it to derive source-bytes-per-param when the
# checkpoint is not uniformly one dtype).
_ST_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1,
}

GB = 1_000_000_000.0  # decimal GB, matching index.json total_size reporting


def _read_safetensors_header(path: str) -> dict:
    """Return the JSON header of a .safetensors file (no weight load)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def source_checkpoint_bytes(model_path: str) -> tuple[int, dict[str, int]]:
    """(total_bytes, {dtype: bytes}) over all *.safetensors shards.

    Sums the safetensors ``data_offsets`` spans — the exact on-disk byte size
    of every tensor — so it is precise regardless of dtype, sharding, or
    name remapping. Reads only the headers; no tensor data is materialized.
    """
    total = 0
    by_dtype: dict[str, int] = {}
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"no *.safetensors shards under {model_path!r}; cannot size the "
            "non-quantizable floor")
    for shard in shards:
        header = _read_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            nb = int(b) - int(a)
            total += nb
            by_dtype[meta["dtype"]] = by_dtype.get(meta["dtype"], 0) + nb
    return total, by_dtype


def dominant_source_bytes_per_param(by_dtype: Mapping[str, int]) -> int:
    """Bytes/element of the dtype holding the most bytes (the body precision).

    Production sources are uniform bf16 (-> 2) or native fp8 (-> 1). The body
    Linears dominate the byte mass, so the largest-by-bytes dtype is the source
    precision the re-encoded weights are read from. Unknown dtype -> 2 (bf16).

    NB: prefer ``source_regime`` for the re-encoded-source-bytes accounting --
    on a large-vocab FP8 model the bf16 embed+lm_head can outmass the fp8 body
    and this would mis-pick bf16. ``source_regime`` keys off the *presence* of
    fp8 (which only the body has), so it is robust to that.
    """
    if not by_dtype:
        return 2
    dt = max(by_dtype.items(), key=lambda kv: kv[1])[0]
    return _ST_DTYPE_BYTES.get(dt, 2)


def source_regime(by_dtype: Mapping[str, int]) -> str:
    """Classify the source checkpoint's *body* weight precision: 'bf16' | 'fp8'.

    The non-quantizable floor (embeddings, lm_head, norms) is always bf16/fp16,
    so the *presence* of any fp8 dtype in the checkpoint is an unambiguous tell
    that the re-encoded body weights are native fp8 (DeepSeek-V4 / MiniMax) --
    robust to the large-vocab case where bf16 embed+lm_head outmass the fp8 body
    (which fools ``dominant_source_bytes_per_param``). A re-encoded fp8 Linear
    occupies its full FP8_SOURCE layout on disk (fp8 weight + fp32 128x128
    block ``weight_scale_inv``), so its source bytes are
    ``memory_bytes_for_shape("FP8_SOURCE", shape)`` -- see
    ``reencoded_source_bytes_for_shape``. Returns 'fp8' if any F8_* dtype carries
    bytes, else 'bf16'. (A genuinely mixed bf16+fp8 *body* is not a production
    shape; it is reported via ``source_total``/floor as fp8 and should be
    cross-checked against the export.)
    """
    if any(dt.startswith("F8") and nb > 0 for dt, nb in by_dtype.items()):
        return "fp8"
    return "bf16"


# Quantization sidecar suffixes summed into their base tensor's manifest
# entry: the export removes these together with the weight when it
# re-encodes a Linear (DSv4 ``.scale`` MXFP4/E8M0 group scales, DeepSeek /
# MiniMax fp8 ``.weight_scale_inv`` 128x128 block scales, compressed-tensors
# ``.weight_scale``). A standalone sidecar with no base tensor is never
# re-encoded and stays priced in the floor.
_SIDECAR_SUFFIXES = (".scale", ".weight_scale_inv", ".weight_scale")


def _default_expert_parent_for_projection(projection_name: str) -> str | None:
    """No-profile fallback for the per-expert -> packed projection mapping.

    Mirrors ``ModelProfile.packed_expert_parent_for_projection``'s legacy
    fallback: per-expert ``gate_proj``/``up_proj`` fuse into the packed
    ``gate_up_proj`` (output-axis cat, the transformers packed-FusedMoE
    convention); ``down_proj`` packs 1:1. Anything else (e.g. MiniMax's
    per-expert ``w1``/``w2``/``w3`` modules, which stay per-expert live)
    has no packed parent here — callers with a profile should pass its
    ``packed_expert_parent_for_projection`` instead.
    """
    if projection_name in ("gate_proj", "up_proj"):
        return "gate_up_proj"
    if projection_name == "down_proj":
        return "down_proj"
    return None


def packed_expert_alias(qname: str, parent_for_projection=None) -> str | None:
    """Packed live qname a per-expert Linear aggregates into, or None.

    ``...experts.{i}.{proj}`` -> ``...experts.{parent}`` when
    ``parent_for_projection(proj)`` names a packed parent
    (``ModelProfile.packed_expert_parent_for_projection``; the legacy
    gate/up/down fallback when None). Non-expert names and unrecognized
    projections return None. This is the same structural
    ``experts.{idx}.{leaf}`` detection the profile layer uses for
    packed-format grouping (``packed_expert_format_group``).
    """
    parts = str(qname).split(".")
    if len(parts) < 3 or parts[-3] != "experts" or not parts[-2].isdigit():
        return None
    fn = (parent_for_projection if parent_for_projection is not None
          else _default_expert_parent_for_projection)
    parent = fn(parts[-1])
    if not parent:
        return None
    return ".".join(parts[:-2] + [str(parent)])


def source_tensor_bytes_manifest(
    model_path: str,
    name_map=None,
    expert_parent_for_projection=None,
) -> dict[str, int]:
    """Exact on-disk source bytes per weight tensor, keyed by live qname base.

    Walks the safetensors headers and, for every weight tensor, sums its
    byte span with its quantization sidecars (``<base>.scale``,
    ``<base>.weight_scale_inv``, ``<base>.weight_scale``) — exactly the
    bytes the export removes from the checkpoint when it re-encodes that
    Linear. ``name_map`` maps a checkpoint key to the live transformers
    parameter name (``ModelProfile.checkpoint_to_live_name``); identity
    when None. Keys are stored without the ``.weight`` suffix to match
    allocator qnames.

    Both packed-MoE on-disk layouts resolve to the packed allocator names
    (``...experts.gate_up_proj`` / ``...experts.down_proj``):

    - **Packed 3-D on disk** (LFM2.5, Qwen3.6-35B): the expert param is a
      checkpoint key with NO ``.weight`` suffix. Suffix-less keys are kept
      (only sidecar keys are folded into their base), so the packed tensor
      lands in the manifest under its own name.
    - **Per-expert 2-D on disk** (``...experts.{i}.{proj}.weight``): each
      per-expert span is ALSO accumulated into the packed parent name via
      :func:`packed_expert_alias` (gate+up fuse into gate_up), driven by
      ``expert_parent_for_projection``
      (``ModelProfile.packed_expert_parent_for_projection``; legacy
      gate/up/down fallback when None). The per-expert entries are kept
      alongside the packed aggregate so per-expert-named allocations
      resolve too — a ``reencoded_names`` list must use ONE naming scheme
      per tensor (any consistent probe does), never both.

    This is the per-tensor replacement for the regime-wide
    ``reencoded_source_bytes_for_shape`` accounting, which charges EVERY
    re-encoded Linear at the FP8_SOURCE layout (1 B/param + fp32 block
    scales) as soon as any F8 dtype is present in the checkpoint. On a
    mixed-precision source that is wrong per tensor class — e.g. the
    MXFP4-packed routed experts of a DSv4-Flash checkpoint (I8 nibble
    weights + E8M0 group scales, ~0.53 B/param) were charged 1 B/param,
    "removing" 279.9 GB from a 166.9 GB checkpoint and driving the
    non-quantizable floor to −113 GB. Summing actual header byte spans can
    never exceed the checkpoint total, so a floor computed from this
    manifest is >= 0 by construction (a negative floor is always an
    accounting bug — rejected at the consumers).
    """
    spans: dict[str, int] = {}
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"no *.safetensors shards under {model_path!r}; cannot build the "
            "per-tensor source-byte manifest")
    for shard in shards:
        header = _read_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            spans[name] = spans.get(name, 0) + (int(b) - int(a))
    out: dict[str, int] = {}

    def _add(live: str, nb: int) -> None:
        out[live] = out.get(live, 0) + nb

    for name, nb in spans.items():
        if any(name.endswith(s) for s in _SIDECAR_SUFFIXES):
            continue  # folded into its base tensor's entry below
        # Packed 3-D expert params have no ".weight" suffix; the key IS the
        # base (and its sidecars still hang off `<base>.scale` etc.).
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        total = nb + sum(spans.get(base + s, 0) for s in _SIDECAR_SUFFIXES)
        # A live-graph mapper declining a key does NOT mean the tensor has no
        # source bytes, so it must not drop out of the manifest: MTP sidecars
        # are the live case (transformers v5 removed the module, so
        # `checkpoint_to_live_name("mtp.fc.weight")` is None on the Qwen
        # profiles) while the exporter still re-encodes `mtp.*` from exactly
        # these bytes, and the allocator assigns them under their raw names.
        # Fall back to the checkpoint key so such a name resolves. This is
        # inert for tensors nothing re-encodes: the floor is
        # `checkpoint_total - sum(resolved re-encoded spans)`, so a manifest
        # entry no `reencoded_names` member references never moves it.
        live = (name_map(name) if name_map is not None else name) or name
        if live.endswith(".weight"):
            live = live[: -len(".weight")]
        _add(live, total)
        packed = packed_expert_alias(live, expert_parent_for_projection)
        if packed is not None:
            _add(packed, total)
    return out


def resolve_reencoded_source_bytes(
    manifest: Mapping[str, int],
    reencoded_names: Iterable[str],
    *,
    context: str,
) -> dict[str, int]:
    """Look up each re-encoded Linear's actual source bytes in the manifest.

    A name the manifest cannot resolve is a HARD ERROR, not a warning: an
    unresolved Linear's source bytes stay in the floor while its quantized
    body bytes are still added, silently inflating the artifact estimate —
    on a packed-MoE model by the full expert mass, at which point every
    rung reads "below the floor". Raising here, before any selection
    numbers are computed, puts the offending tensor names in front of the
    operator instead of a fatal below-the-floor exit with a trailing
    warning.
    """
    out: dict[str, int] = {}
    missing: list[str] = []
    for qname in reencoded_names:
        nb = manifest.get(qname)
        if nb is None and qname.endswith(".weight"):
            nb = manifest.get(qname[: -len(".weight")])
        if nb is None:
            missing.append(qname)
        else:
            out[qname] = int(nb)
    if missing:
        shown = sorted(missing)[:10]
        raise ValueError(
            f"[footprint] {len(missing)} re-encoded Linear(s) not resolvable "
            f"in the source checkpoint manifest ({context}): "
            + ", ".join(shown)
            + (", …" if len(missing) > len(shown) else "")
            + ". Their source bytes would stay in the floor while their "
            "quantized bytes are still added (artifact over-count; on a "
            "packed-MoE model the entire expert mass is double-counted and "
            "every rung reads 'below the floor'). Fix the profile's name "
            "resolution (checkpoint_to_live_name / "
            "packed_expert_parent_for_projection) so every re-encoded "
            "tensor resolves; do not consume these numbers.")
    return out


def reencoded_source_bytes_for_shape(shape: tuple[int, ...], regime: str) -> int:
    """On-disk source bytes of ONE re-encoded Linear, by source regime.

    bf16 source: ``n_params * 2`` (no scale sibling). fp8 source: the weight is
    native fp8 and ships with an fp32 128x128 ``weight_scale_inv`` block scale --
    exactly the FP8_SOURCE on-disk layout -- so its bytes are
    ``memory_bytes_for_shape("FP8_SOURCE", shape)`` (fp8 weight + fp32 block
    scale). This is what makes the floor exact for fp8-native sources: every
    re-encoded Linear's *full* source footprint (weight + scale_inv) is removed
    from the floor, not just the weight bytes.

    WARNING: regime-wide accounting is only correct when the body is
    uniformly bf16 or uniformly fp8. For mixed-precision sources (e.g.
    MXFP4-packed experts, I8 + E8M0 scales) use
    :func:`source_tensor_bytes_manifest`, which charges each tensor its
    actual header byte span. Consumers must reject a negative floor.
    """
    if regime == "fp8":
        return int(fr.get_format("FP8_SOURCE").memory_bytes_for_shape(shape))
    n = 1
    for d in shape:
        n *= int(d)
    return n * 2  # bf16/fp16 source weight, no scale sibling


def _tensor_class(qname: str) -> str:
    """Coarse tensor-class label for floor-accounting diagnostics."""
    if ".shared_experts." in qname:
        return "shared_experts"
    if ".experts." in qname:
        return "routed_experts"
    if ".self_attn." in qname or ".attn." in qname:
        return "attention"
    if ".mlp." in qname or ".ffn." in qname:
        return "mlp"
    return "other"


def check_floor_non_negative(
    floor_bytes: float,
    source_total_bytes: float,
    reencoded_by_name: Mapping[str, int],
    *,
    context: str,
) -> None:
    """A negative non-quantizable floor is ALWAYS an accounting bug.

    It means the source bytes 'removed' for re-encoding exceed the bytes the
    checkpoint actually holds — e.g. MXFP4-packed experts (~0.53 B/param on
    disk) charged at the FP8_SOURCE 1 B/param layout can drive the floor
    negative and let an artifact more than twice the budget 'fit' it. Raises
    with the per-tensor-class byte breakdown so the offending class is
    named, never rationalized.
    """
    if floor_bytes >= 0:
        return
    by_class: dict[str, int] = {}
    for qname, nb in reencoded_by_name.items():
        cls = _tensor_class(qname)
        by_class[cls] = by_class.get(cls, 0) + int(nb)
    detail = ", ".join(
        f"{cls}={nb / GB:.2f}GB"
        for cls, nb in sorted(by_class.items(), key=lambda kv: -kv[1]))
    raise ValueError(
        f"[footprint] negative non-quantizable floor in {context}: "
        f"floor={floor_bytes / GB:.3f}GB (source_total="
        f"{source_total_bytes / GB:.3f}GB, reencoded_source="
        f"{(source_total_bytes - floor_bytes) / GB:.3f}GB). Removed source "
        f"bytes by tensor class: {detail}. The per-class source-byte rate is "
        "wrong (mixed-precision source charged at a uniform regime?). Use "
        "source_tensor_bytes_manifest() for per-tensor accounting; do not "
        "ship a selection computed from this floor.")


# fp32 weight_global_scale + fp32 input_global_scale per emitted NVFP4
# 2-D Linear (verified against shipped-artifact safetensors headers:
# both are F32 scalars, 4 bytes each).
_NVFP4_GLOBAL_SIDECAR_BYTES_PER_LINEAR = 8

# On-disk projection count for packed 3-D expert tensors, keyed by the
# assignment key's leaf name. Mirrors the exporter's
# ``ModelProfile.packed_expert_projection_names`` DefaultProfile fallback
# (a packed ``gate_up_proj`` splits into gate_proj + up_proj per-expert
# Linears on disk; every other packed param emits one Linear per expert).
# footprint deliberately carries no profile/torch dependency, so a
# profile that *declares* a differently-named multi-projection packed
# param would under-count 8·E·(P−1) bytes here — no such profile exists
# in the tree today.
_PACKED_LEAF_PROJECTIONS = {"gate_up_proj": 2}


def nvfp4_global_sidecar_bytes(qname: str, shape: tuple[int, ...]) -> int:
    """Bytes of the fp32 NVFP4 global sidecars the export emits.

    Every emitted NVFP4 2-D Linear ships ``weight_global_scale`` +
    ``input_global_scale`` (fp32 scalars, 8 bytes). A packed 3-D expert
    tensor ``(E, out, in)`` is split into E × P per-expert 2-D Linears on
    disk (P = on-disk projection count, 2 for ``gate_up_proj``), each with
    its own pair — 8·E·P bytes. ``memory_bytes_for_shape`` counts weight +
    group-scale bytes only, so this is additive.
    """
    if len(shape) == 3:
        leaf = qname.rsplit(".", 1)[-1]
        n_proj = _PACKED_LEAF_PROJECTIONS.get(leaf, 1)
        return _NVFP4_GLOBAL_SIDECAR_BYTES_PER_LINEAR * int(shape[0]) * n_proj
    return _NVFP4_GLOBAL_SIDECAR_BYTES_PER_LINEAR


def assignment_artifact_bytes(
    assignment: Mapping[str, str],
    stats: Mapping[str, dict],
    *,
    source_total_bytes: int,
    regime: str = "bf16",
    canonicalize: bool = True,
    source_manifest: Mapping[str, int] | None = None,
) -> dict:
    """Exact on-disk bytes of the exported artifact for ``assignment``.

    ``assignment`` maps Linear qname -> format name (the allocator's *expanded*,
    post-promotion per-Linear assignment, so fused-sibling / packed-MoE coupling
    is already reflected). ``stats`` is the probe's per-Linear stats (carries
    ``n_params`` and the ``in/out_features`` / ``num_experts`` the byte formula
    needs). Names absent from ``stats`` are *not* a problem: they are simply not
    subtracted from the floor, so they remain counted at source precision — which
    is correct for any tensor that ships verbatim (and explains why a handful of
    fused super-names / pins can be missing yet the total stays exact).

    ``source_manifest`` (from :func:`source_tensor_bytes_manifest`) is the
    preferred source-byte accounting and the one
    :func:`floor_bytes_for_model` and the allocator's byte-budget selector
    use: each re-encoded Linear is charged its ACTUAL header byte span
    (weight + scale siblings), so the two paths agree exactly. A priced
    Linear the manifest cannot resolve is a hard error
    (:func:`resolve_reencoded_source_bytes`).

    Without a manifest, ``regime`` ('bf16' | 'fp8', from source_regime) sets
    each re-encoded Linear's *source* byte size removed from the floor:
    bf16 -> 2 bytes/param; fp8 -> the full FP8_SOURCE layout (fp8 weight +
    fp32 128x128 weight_scale_inv), so the source scale sibling is removed
    too (else it is double-counted: left in the floor and re-added by the
    export). This is exact ONLY for a uniformly-bf16 or uniformly-fp8
    (128x128-block-scaled) body; on any other source pass a manifest — a
    mixed source drives the floor negative and is rejected
    (``check_floor_non_negative``), never silently shipped.

    Returns a dict: ``artifact_bytes``, ``floor_bytes``, ``body_quant_bytes``,
    ``reencoded_source_bytes``, ``n_reencoded``, ``n_missing_stats``,
    ``regime``, ``source_accounting``.
    """
    body_quant = 0
    reenc_by_name: dict[str, int] = {}
    priced: list[str] = []
    n_missing = 0
    for qname, fmt in assignment.items():
        entry = stats.get(qname)
        if entry is None and qname.endswith(".weight"):
            entry = stats.get(qname[: -len(".weight")])
        if not isinstance(entry, dict):
            n_missing += 1
            continue
        shape = _shape_from_stats(entry)
        name = fr.canonical_format_name(fmt) if canonicalize else fmt
        body_quant += fr.get_format(name).memory_bytes_for_shape(shape)
        if name == "NVFP4":
            body_quant += nvfp4_global_sidecar_bytes(qname, shape)
        if source_manifest is None:
            reenc_by_name[qname] = reencoded_source_bytes_for_shape(
                shape, regime)
        priced.append(qname)
    if source_manifest is not None:
        reenc_by_name = resolve_reencoded_source_bytes(
            source_manifest, priced, context="assignment_artifact_bytes")
    reenc_src = sum(reenc_by_name.values())
    floor = int(source_total_bytes) - reenc_src
    check_floor_non_negative(
        floor, int(source_total_bytes), reenc_by_name,
        context="assignment_artifact_bytes")
    return {
        "artifact_bytes": floor + body_quant,
        "floor_bytes": floor,
        "body_quant_bytes": body_quant,
        "reencoded_source_bytes": reenc_src,
        "n_reencoded": len(priced),
        "n_missing_stats": n_missing,
        "regime": regime,
        "source_accounting": (
            "per_tensor_manifest" if source_manifest is not None else "regime"),
    }


def assignment_artifact_gb(
    assignment: Mapping[str, str],
    stats: Mapping[str, dict],
    *,
    source_total_bytes: int,
    regime: str = "bf16",
    source_manifest: Mapping[str, int] | None = None,
) -> float:
    """Convenience: just the artifact size in GB (decimal, matches index.json)."""
    return assignment_artifact_bytes(
        assignment, stats,
        source_total_bytes=source_total_bytes,
        regime=regime,
        source_manifest=source_manifest,
    )["artifact_bytes"] / GB


def floor_bytes_for_model(
    model_path: str,
    reencoded_names: Iterable[str],
    stats: Mapping[str, dict],
    *,
    regime: str | None = None,
    name_map=None,
    expert_parent_for_projection=None,
) -> dict:
    """Compute the non-quantizable floor (and the scalars to reuse) from a model.

    Convenience wrapper that reads the checkpoint headers once and returns
    ``{source_total_bytes, regime, source_bytes_per_param, floor_bytes,
    reencoded_source_bytes, source_manifest, source_dtype_bytes}``. The floor
    is constant across formats (only the re-encoded *format* varies, not which
    tensors are re-encoded), so callers sweeping many allocations compute this
    once and pass ``source_total_bytes`` + ``source_manifest`` to
    ``assignment_artifact_bytes`` per candidate. Each re-encoded Linear is
    charged its actual header byte span from
    :func:`source_tensor_bytes_manifest` (this function has the model path,
    so it never needs the regime-wide per-param rate); a name the manifest
    cannot resolve is a hard error (:func:`resolve_reencoded_source_bytes`) —
    an unresolved name would silently over-count the artifact.
    ``name_map`` / ``expert_parent_for_projection`` are the profile's
    ``checkpoint_to_live_name`` / ``packed_expert_parent_for_projection``
    (pass them for any packed-MoE architecture; defaults handle the
    identity naming and the legacy gate/up/down packing). ``regime``
    defaults to :func:`source_regime` (robust fp8/bf16 detection) and is
    returned for reporting. ``stats`` is retained for call compatibility
    (shapes are no longer needed to price source bytes).
    """
    total, by_dtype = source_checkpoint_bytes(model_path)
    reg = regime if regime is not None else source_regime(by_dtype)
    manifest = source_tensor_bytes_manifest(
        model_path, name_map=name_map,
        expert_parent_for_projection=expert_parent_for_projection)
    reenc_by_name = resolve_reencoded_source_bytes(
        manifest, reencoded_names, context="floor_bytes_for_model")
    reenc_src = sum(reenc_by_name.values())
    check_floor_non_negative(
        int(total) - reenc_src, total, reenc_by_name,
        context="floor_bytes_for_model")
    return {
        "source_total_bytes": total,
        "regime": reg,
        "source_bytes_per_param": dominant_source_bytes_per_param(by_dtype),
        "floor_bytes": int(total) - reenc_src,
        "reencoded_source_bytes": reenc_src,
        "source_manifest": manifest,
        "source_dtype_bytes": by_dtype,
    }
