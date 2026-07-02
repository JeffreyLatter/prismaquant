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


def reencoded_source_bytes_for_shape(shape: tuple[int, ...], regime: str) -> int:
    """On-disk source bytes of ONE re-encoded Linear, by source regime.

    bf16 source: ``n_params * 2`` (no scale sibling). fp8 source: the weight is
    native fp8 and ships with an fp32 128x128 ``weight_scale_inv`` block scale --
    exactly the FP8_SOURCE on-disk layout -- so its bytes are
    ``memory_bytes_for_shape("FP8_SOURCE", shape)`` (fp8 weight + fp32 block
    scale). This is what makes the floor exact for fp8-native sources: every
    re-encoded Linear's *full* source footprint (weight + scale_inv) is removed
    from the floor, not just the weight bytes.
    """
    if regime == "fp8":
        return int(fr.get_format("FP8_SOURCE").memory_bytes_for_shape(shape))
    n = 1
    for d in shape:
        n *= int(d)
    return n * 2  # bf16/fp16 source weight, no scale sibling


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

    ``regime`` ('bf16' | 'fp8', from source_regime) sets each re-encoded Linear's
    *source* byte size removed from the floor: bf16 -> 2 bytes/param; fp8 -> the
    full FP8_SOURCE layout (fp8 weight + fp32 128x128 weight_scale_inv), so the
    source scale sibling is removed too (else it is double-counted: left in the
    floor and re-added by the export).

    Returns a dict: ``artifact_bytes``, ``floor_bytes``, ``body_quant_bytes``,
    ``reencoded_source_bytes``, ``n_reencoded``, ``n_missing_stats``, ``regime``.
    """
    body_quant = 0
    reenc_src = 0
    n_reencoded = 0
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
        reenc_src += reencoded_source_bytes_for_shape(shape, regime)
        n_reencoded += 1
    floor = int(source_total_bytes) - reenc_src
    return {
        "artifact_bytes": floor + body_quant,
        "floor_bytes": floor,
        "body_quant_bytes": body_quant,
        "reencoded_source_bytes": reenc_src,
        "n_reencoded": n_reencoded,
        "n_missing_stats": n_missing,
        "regime": regime,
    }


def assignment_artifact_gb(
    assignment: Mapping[str, str],
    stats: Mapping[str, dict],
    *,
    source_total_bytes: int,
    regime: str = "bf16",
) -> float:
    """Convenience: just the artifact size in GB (decimal, matches index.json)."""
    return assignment_artifact_bytes(
        assignment, stats,
        source_total_bytes=source_total_bytes,
        regime=regime,
    )["artifact_bytes"] / GB


def floor_bytes_for_model(
    model_path: str,
    reencoded_names: Iterable[str],
    stats: Mapping[str, dict],
    *,
    regime: str | None = None,
) -> dict:
    """Compute the non-quantizable floor (and the scalars to reuse) from a model.

    Convenience wrapper that reads the checkpoint headers once and returns
    ``{source_total_bytes, regime, source_bytes_per_param, floor_bytes,
    reencoded_source_bytes, source_dtype_bytes}``. The floor is constant across
    formats (only the re-encoded *format* varies, not which tensors are
    re-encoded), so callers sweeping many allocations compute this once and pass
    ``source_total_bytes`` + ``regime`` to ``assignment_artifact_bytes`` per
    candidate. ``regime`` defaults to :func:`source_regime` (robust fp8/bf16
    detection); each re-encoded Linear's source bytes follow the regime
    (fp8 removes the weight_scale_inv sibling too).
    """
    total, by_dtype = source_checkpoint_bytes(model_path)
    reg = regime if regime is not None else source_regime(by_dtype)
    reenc_src = 0
    for qname in reencoded_names:
        entry = stats.get(qname)
        if entry is None and qname.endswith(".weight"):
            entry = stats.get(qname[: -len(".weight")])
        if isinstance(entry, dict):
            reenc_src += reencoded_source_bytes_for_shape(
                _shape_from_stats(entry), reg)
    return {
        "source_total_bytes": total,
        "regime": reg,
        "source_bytes_per_param": dominant_source_bytes_per_param(by_dtype),
        "floor_bytes": int(total) - reenc_src,
        "reencoded_source_bytes": reenc_src,
        "source_dtype_bytes": by_dtype,
    }
