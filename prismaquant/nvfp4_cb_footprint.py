"""NVFP4-CB sidecar-aware byte accountant (Phase-0 measurement harness).

The registry ``FormatSpec.memory_bytes_for_shape`` is byte-exact for the
**fixed-lattice** NVFP4-CB variant (no sidecar, exactly like GGUF). For the
**learned** variant it *understates* true bytes by the per-tensor codebook
sidecar. This accountant adds the two terms a stock FormatSpec does not model:

  * learned-codebook sidecar: ``2^k * 8 * 4 bits = 2^k * 4 bytes`` per tensor
    (or once per shared group), and
  * optional per-tensor FP32 global scale: ``4 bytes`` / tensor.

so that no arm can hide sidecar cost. See docs/nvfp4-cb-plan/format-pipeline.md
§1.2 (registry-exact bpw table) and §1.4 (codebook sidecar).

``body_bpw`` (over quantizable params) is registry-exact and reproduces the
§1.2 ``k/8 + 0.5`` table for the fixed-lattice case; ``total_bytes`` adds the
sidecar + global-scale terms on top.
"""

from __future__ import annotations

import math
import re
from typing import Mapping

from . import format_registry as fr

# Per-tensor FP32 global scale (NVFP4-style), §1.3 "+ negligible" term.
_GLOBAL_SCALE_BYTES = 4
# Learned codebook entry = 8 FP4 (E2M1) 4-bit codes = 4 bytes/entry, §1.4.
_CODEBOOK_ENTRY_BYTES = 4

_CB_NAME_RE = re.compile(r"^NVFP4_CB_K(\d+)$")


def _cb_k(format_name: str) -> int | None:
    """Return the CB index width ``k`` for an NVFP4-CB format name, else None."""
    m = _CB_NAME_RE.match(str(format_name).strip().upper())
    return int(m.group(1)) if m else None


def _resolve_spec(format_name: str) -> fr.FormatSpec:
    """Resolve a format to a FormatSpec.

    Uses the registry when available. For NVFP4-CB rungs that the (separately
    authored) format module has not registered yet, synthesise the byte-exact
    spec from the §2 factory parameters (``weight_bits=0``,
    ``group_size=256``, ``scale_bits=32k+128``) so this accountant is
    format-agnostic and independent of registration timing. A real registered
    spec carries identical parameters, so byte accounting is unchanged.
    """
    name = str(format_name).strip()
    try:
        return fr.get_format(name)
    except KeyError:
        k = _cb_k(name)
        if k is None:
            raise
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

        k = _cb_k(format_name)
        kind = _codebook_source_kind(codebook_sources.get(qname))
        # Per-tensor FP32 global scale applies to CB-family tensors.
        g_bytes = _GLOBAL_SCALE_BYTES if k is not None else 0
        global_scale_bytes += g_bytes

        tensor_sidecar = 0
        if kind == "learned":
            if k is None:
                raise ValueError(
                    f"cb_footprint: learned codebook_source on non-CB format "
                    f"'{format_name}' for '{qname}'")
            cb_bytes = (1 << k) * _CODEBOOK_ENTRY_BYTES
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
            "params": params,
            "body_bytes": tensor_body,
            "global_scale_bytes": g_bytes,
            "sidecar_bytes": tensor_sidecar,
            "codebook_source": kind,
            "body_bpw": 8.0 * tensor_body / max(params, 1),
        }

    total_bytes = body_bytes + global_scale_bytes + sidecar_bytes
    return {
        "total_bytes": int(total_bytes),
        "body_bytes": int(body_bytes),
        "sidecar_bytes": int(sidecar_bytes),
        "global_scale_bytes": int(global_scale_bytes),
        "n_params": int(n_params),
        # Registry-exact bpw over quantizable params (excludes sidecar +
        # global scale) — reproduces the §1.2 k/8+0.5 table for fixed lattice.
        "body_bpw": 8.0 * body_bytes / max(n_params, 1),
        # True shipped bpw including sidecar + global scale.
        "total_bpw": 8.0 * total_bytes / max(n_params, 1),
        "per_tensor": per_tensor,
    }
