"""Runtime-backed serving shape validators.

Serving-profile JSON specs reference these by dotted callable path.  Keep
backend probes here so allocator and serving-profile core stay config-driven.
"""
from __future__ import annotations

from . import format_registry as fr


def flashinfer_mxfp8_problem_size_accepts(
    fmt: str,
    *,
    in_features: int,
    out_features: int,
) -> bool | None:
    """Return FlashInfer's MXFP8_E4M3/MXFP8_E5M2 GEMM shape verdict."""
    try:
        canonical = fr.canonical_format_name(fmt)
    except Exception:
        return None
    if not canonical.startswith("MXFP8"):
        return None
    try:
        import torch
        from flashinfer.gemm.gemm_base import _check_mm_mxfp8_problem_size
        from flashinfer.gemm.gemm_base import _mxfp8_swizzled_scale_len
        from flashinfer.gemm.gemm_base import SfLayout
    except Exception:
        return None

    try:
        a = torch.empty((1, in_features), dtype=torch.float8_e4m3fn)
        b = torch.empty((in_features, out_features), dtype=torch.float8_e4m3fn)
        a_desc_len = _mxfp8_swizzled_scale_len(
            a.shape[0],
            a.shape[1],
            SfLayout.layout_8x4,
        )
        b_desc_len = _mxfp8_swizzled_scale_len(
            b.shape[1],
            b.shape[0],
            SfLayout.layout_8x4,
        )
        a_desc = torch.empty((a_desc_len,), dtype=torch.uint8)
        b_desc = torch.empty((b_desc_len,), dtype=torch.uint8)
    except Exception:
        return None
    try:
        return _check_mm_mxfp8_problem_size(a, b, a_desc, b_desc) is True
    except Exception:
        return False
