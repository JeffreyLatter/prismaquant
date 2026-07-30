"""CB-family sidecar-aware byte accountant (Phase-0 measurement harness).

Covers both concurrently-authored CB families:

  * ``NVFP4_CB_K{k}`` — FP4-grid codewords, group-16 E4M3 scale plane
    (``k/8 + 0.5`` bpw, docs/lanes/nvfp4-cb/format-pipeline.md §1.2);
  * ``FP8_CB_K{k}`` — FP8/E4M3-grid codewords, NO group scale plane
    (``k/8`` bpw body) + per-output-channel fp32 scales.

The registry ``FormatSpec.memory_bytes_for_shape`` is byte-exact for the
**fixed-lattice** variants (no sidecar, exactly like GGUF). For the
**learned** variant it *understates* true bytes by the per-tensor codebook
sidecar. This accountant adds the terms a stock FormatSpec does not model:

  * learned-codebook sidecar: ``2^k`` entries × 4 bytes (NVFP4_CB, 8×4-bit
    codes) or × 8 bytes (FP8_CB, 8×8-bit codes) per tensor, charged once per
    shared group;
  * per-tensor FP32 global scale (NVFP4_CB): ``4 bytes`` / tensor;
  * per-output-channel fp32 scales (FP8_CB): ``4 × out_rows`` bytes / tensor.

so that no arm can hide sidecar cost. ``body_bpw`` (over quantizable params)
is registry-exact and reproduces the §1.2 ``k/8 + 0.5`` table for the
fixed-lattice NVFP4_CB case; ``total_bytes`` adds sidecar + scale terms.
"""

from __future__ import annotations

import math
import re
from typing import Mapping

from . import format_registry as fr

# Per-tensor FP32 global scale (NVFP4-style), §1.3 "+ negligible" term.
# Applies to the NVFP4_CB family only; FP8_CB carries per-output-channel
# fp32 scales instead (see _CHANNEL_SCALE_BYTES below).
_GLOBAL_SCALE_BYTES = 4
# Learned codebook entry bytes are family-dependent:
#   NVFP4_CB: 8 FP4 (E2M1) 4-bit codes = 4 bytes/entry (§1.4);
#   FP8_CB:   8 FP8 (E4M3) 8-bit codes = 8 bytes/entry.
_CODEBOOK_ENTRY_BYTES = {"nvfp4": 4, "fp8": 8}
# FP8_CB per-output-channel fp32 scale = 4 bytes per output row.
_CHANNEL_SCALE_BYTES = 4

_CB_NAME_RE = re.compile(r"^(NVFP4|FP8)_CB_K(\d+)$")


def _cb_info(format_name: str) -> tuple[str | None, int | None]:
    """Return ``(cb_family, k)`` for a CB format name, else ``(None, None)``.

    ``cb_family`` is ``"nvfp4"`` (NVFP4_CB_K*) or ``"fp8"`` (FP8_CB_K*).
    """
    m = _CB_NAME_RE.match(str(format_name).strip().upper())
    if not m:
        return None, None
    return m.group(1).lower(), int(m.group(2))


def _cb_k(format_name: str) -> int | None:
    """Return the CB index width ``k`` for a CB format name, else None."""
    return _cb_info(format_name)[1]


def _resolve_spec(format_name: str) -> fr.FormatSpec:
    """Resolve a format to a FormatSpec.

    Uses the registry when available. For CB rungs that the (separately
    authored) format module has not registered yet, synthesise the byte-exact
    spec so this accountant is format-agnostic and independent of
    registration timing:

      * NVFP4_CB_K{k}: §2 factory parameters (``weight_bits=0``,
        ``group_size=256``, ``scale_bits=32k+128`` — index stream + the
        group-16 E4M3 scale plane).
      * FP8_CB_K{k}: ``weight_bits=0``, ``group_size=256``,
        ``scale_bits=32k`` — index stream only. FP8_CB has NO group-16 scale
        plane; its per-output-channel fp32 scales are accounted separately
        (``channel_scale_bytes``), because ``scale_count_for_shape`` groups
        along the input dim and cannot express a per-row plane on top of the
        superblock stream.

    The registered NVFP4_CB spec carries the same body parameters as the
    fallback. The registered FP8_CB spec instead folds the channel-scale
    plane into its body bytes (``weight_bits=k/8, group_size=0,
    scale_bits=32`` → ``k/8 + 32/in`` exactly); ``cb_footprint`` detects that
    layout and skips the separate ``channel_scale_bytes`` charge so the plane
    is counted exactly once on either path.
    """
    name = str(format_name).strip()
    try:
        return fr.get_format(name)
    except KeyError:
        cb_family, k = _cb_info(name)
        if k is None:
            raise
        if cb_family == "fp8":
            return fr.FormatSpec(
                name=f"FP8_CB_K{k}",
                weight_bits=0,
                group_size=256,
                scale_bits=32 * k,
                scale_dtype_name="fp8_cb_vq",
                weight_element_dtype=f"fp8_cb_k{k}",
                family="fp8_cb",
            )
        return fr.FormatSpec(
            name=f"NVFP4_CB_K{k}",
            weight_bits=0,
            group_size=256,
            scale_bits=32 * k + 128,
            scale_dtype_name="nvfp4_cb_vq",
            weight_element_dtype=f"nvfp4_cb_k{k}",
            family="nvfp4_cb",
        )


def _codebook_source_kind(entry) -> str:
    """Classify a codebook_source entry as 'learned', 'lattice', or 'none'."""
    if entry is None:
        return "none"
    if isinstance(entry, str):
        return "learned" if entry.strip().lower() == "learned" else "lattice"
    if isinstance(entry, Mapping):
        if "learned" in entry:
            return "learned"
        if "lattice" in entry:
            return "lattice"
        kind = str(entry.get("kind", entry.get("source", ""))).strip().lower()
        if kind in ("learned", "lattice"):
            return kind
    return "none"


def _codebook_group(entry) -> str | None:
    """Return a shared-codebook group id, if this learned entry names one.

    Learned codebooks that name the same group are charged their ``2^k * 4``
    sidecar exactly once (a shared per-(model, role) codebook amortises to ~0
    bpw). A learned entry with no group id is charged per-tensor.
    """
    if isinstance(entry, Mapping):
        for key in ("group", "shared_group", "codebook_group"):
            val = entry.get(key)
            if val:
                return str(val)
    return None


def cb_footprint(
    assignment: Mapping[str, str],
    shapes: Mapping[str, tuple[int, ...]],
    *,
    codebook_sources: Mapping[str, object] | None = None,
) -> dict:
    """Exact shipped-bytes accountant for an NVFP4-CB (or mixed) assignment.

    Args:
      assignment: ``{qname: format_name}``.
      shapes: ``{qname: (out_features, in_features)}`` (or any tensor shape).
      codebook_sources: optional ``{qname: source}`` where ``source`` is
        ``"lattice"`` / ``"learned"`` or a dict ``{"lattice"|"learned": id,
        "group": <shared_id>}``. Absent / lattice ⇒ no sidecar. Learned ⇒
        ``2^k * 4`` bytes, once per named shared group (else per-tensor).

    Returns a dict with total bytes and a ``body_bpw`` (registry-exact, over
    quantizable params) that reproduces the §1.2 ``k/8 + 0.5`` table for the
    fixed-lattice case.
    """
    codebook_sources = codebook_sources or {}

    body_bytes = 0
    global_scale_bytes = 0
    channel_scale_bytes = 0
    n_params = 0
    per_tensor: dict[str, dict] = {}
    # Charge each shared learned codebook exactly once.
    charged_groups: dict[str, int] = {}
    sidecar_bytes = 0

    for qname, format_name in assignment.items():
        if qname not in shapes:
            raise KeyError(f"cb_footprint: no shape for '{qname}'")
        shape = tuple(int(d) for d in shapes[qname])
        spec = _resolve_spec(format_name)
        params = int(math.prod(shape)) if shape else 1
        tensor_body = int(spec.memory_bytes_for_shape(shape))
        body_bytes += tensor_body
        n_params += params

        cb_family, k = _cb_info(format_name)
        kind = _codebook_source_kind(codebook_sources.get(qname))
        # Per-tensor FP32 global scale: NVFP4_CB family only.
        g_bytes = _GLOBAL_SCALE_BYTES if cb_family == "nvfp4" else 0
        global_scale_bytes += g_bytes
        # FP8_CB: per-output-channel fp32 scales (4 bytes × output rows).
        # The REGISTERED FP8_CB spec (group_size=0, scale_bits=32) already
        # folds this plane into memory_bytes_for_shape — charging it again
        # would double-count. Only the pre-registration fallback spec
        # (group_size=256, index stream only) needs the separate charge.
        c_bytes = 0
        if cb_family == "fp8" and not (
                spec.group_size == 0 and spec.scale_bits > 0):
            out_rows = int(math.prod(shape[:-1])) if len(shape) > 1 else 1
            c_bytes = _CHANNEL_SCALE_BYTES * out_rows
        channel_scale_bytes += c_bytes

        tensor_sidecar = 0
        if kind == "learned":
            if k is None:
                raise ValueError(
                    f"cb_footprint: learned codebook_source on non-CB format "
                    f"'{format_name}' for '{qname}'")
            cb_bytes = (1 << k) * _CODEBOOK_ENTRY_BYTES[cb_family]
            group = _codebook_group(codebook_sources.get(qname))
            if group is None:
                tensor_sidecar = cb_bytes
                sidecar_bytes += cb_bytes
            else:
                if group not in charged_groups:
                    charged_groups[group] = cb_bytes
                    sidecar_bytes += cb_bytes
                # shared: this tensor adds nothing further.

        per_tensor[qname] = {
            "format": str(format_name),
            "k": k,
            "cb_family": cb_family,
            "params": params,
            "body_bytes": tensor_body,
            "global_scale_bytes": g_bytes,
            "channel_scale_bytes": c_bytes,
            "sidecar_bytes": tensor_sidecar,
            "codebook_source": kind,
            "body_bpw": 8.0 * tensor_body / max(params, 1),
        }

    total_bytes = (body_bytes + global_scale_bytes + channel_scale_bytes
                   + sidecar_bytes)
    return {
        "total_bytes": int(total_bytes),
        "body_bytes": int(body_bytes),
        "sidecar_bytes": int(sidecar_bytes),
        "global_scale_bytes": int(global_scale_bytes),
        "channel_scale_bytes": int(channel_scale_bytes),
        "n_params": int(n_params),
        # Registry-exact bpw over quantizable params (excludes sidecar +
        # global/channel scales) — reproduces the §1.2 k/8+0.5 table for the
        # fixed-lattice NVFP4_CB case and k/8 for FP8_CB.
        "body_bpw": 8.0 * body_bytes / max(n_params, 1),
        # True shipped bpw including sidecar + global/channel scales.
        "total_bpw": 8.0 * total_bytes / max(n_params, 1),
        "per_tensor": per_tensor,
    }
