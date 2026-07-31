"""ROCm/HIP dispatch for :class:`gridbook.linear.PrismaQuantCBLinearMethod`.

Deliberately a SEPARATE module with a single, cheap entry point
(:func:`maybe_apply`).  ``linear.py`` gains one guarded import and one
``is not None`` check, so on a CUDA box — where ``_HIP`` is ``None`` — the
shipping code path is unchanged and no HIP code is imported, parsed or built.

The layer state consumed here is exactly what
``PrismaQuantCBLinearMethod.process_weights_after_loading`` already builds for
the CUDA path (``_cb_qw_padded``, ``_cb_flat_fp8`` / ``_cb_flat``,
``_cb_row_offset``, ``_cb_scale``, ``_cb_compose``).  Nothing new is made
resident: INV-1 holds for the same reason it holds on CUDA.

Dispatch, with the evidence for each boundary:

* ``M <= HIP_GEMV_M_MAX`` (16) -> the decode GEMV.  Measured on gfx1151 at
  N=5120, K=4096, K44: 0.17 ms at M=1 rising to 0.56 ms at M=16, i.e. the GEMV
  is still ahead of the GEMM's fixed tile cost through M=16.
* ``M > 16``, fp8 rungs -> the bf16 WMMA prefill GEMM.  At M=512 it is ~7.8x
  faster than looping the GEMV would be (2.31 ms vs 32 x 0.56 ms).
* fp4 rungs at any M, and anything the kernels do not cover, return ``None``
  and fall through to the caller's existing Triton path.  The fp4 WMMA GEMM is
  not written: the fp4 decode-to-bf16 prologue is the same shape as the fp8 one
  but has no measured demand yet, and an unmeasured second GEMM is exactly the
  kind of thing this repo does not ship.

The 17..31 band is served by the GEMM's smallest tile (32 rows) and is
UNTUNED — the GEMV/GEMM crossover there has not been measured, only bounded.
"""
from __future__ import annotations

import os

import torch

from . import codec

_ENABLED = None


def hip_enabled() -> bool:
    """True when a ROCm device is present and the HIP extension is loaded.

    Cached: the answer cannot change within a process, and the first call is
    what triggers the (one-time, ~1 min) JIT build.
    """
    global _ENABLED
    if _ENABLED is None:
        if os.environ.get("PRISMAQUANT_CB_HIP", "1") == "0":
            _ENABLED = False
        else:
            from .hip_ext import get_ext, is_rocm
            _ENABLED = bool(is_rocm() and get_ext() is not None)
    return _ENABLED


# Same meaning as linear.CUDA_GEMV_M_MAX, re-measured for this device rather
# than inherited: on GB10 the CUDA GEMV loses at M=16, on gfx1151 it does not.
HIP_GEMV_M_MAX = int(os.environ.get("PRISMAQUANT_CB_HIP_M_MAX", "16"))


def _ext():
    from .hip_ext import get_ext
    return get_ext()


def maybe_apply(method, layer, x, bias=None):
    """Run the HIP path for this Linear, or return None to fall through.

    Returning None rather than raising is the contract: every unsupported
    rung, shape or missing-toolchain case must degrade to the caller's Triton
    path silently, exactly as the CUDA entry points do.
    """
    if not hip_enabled():
        return None
    ext = _ext()
    if ext is None:
        return None

    N, K = layer._cb_N, layer._cb_K
    if K % codec.SUPERBLOCK != 0:
        return None
    M = x.reshape(-1, K).shape[0]

    if method.is_fp4:
        # fp4 two-tier v2, product or signed, decode regime only.
        if not method.is_v2 or method.n_sub not in (1, 2):
            return None
        if M > HIP_GEMV_M_MAX:
            return None
        xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
        y = ext.cb_gemv_fp4_v2(xq, layer._cb_qw_padded, layer._cb_flat,
                               layer._cb_row_offset, layer._cb_compose,
                               N, K, method.k, method.n_sub, method.type_size)
        return y + bias if bias is not None else y

    if method.n_sub != 4 or method.type_size != 4 * method.k:
        return None
    cb8 = getattr(layer, "_cb_flat_fp8", None)
    if cb8 is None:
        return None

    if M <= HIP_GEMV_M_MAX:
        # The GEMV fuses the fp8 dynamic per-token activation QDQ (qdq_input),
        # so raw x goes in — matching the CUDA fp8 GEMV's contract exactly.
        y = ext.cb_gemv_fp8(x, layer._cb_qw_padded, cb8, layer._cb_row_offset,
                            layer._cb_scale, N, K, method.k, method.n_sub,
                            method.type_size, True)
        return y + bias if bias is not None else y

    # Prefill: decode-in-prologue bf16 WMMA.  No fp8 matrix instruction exists
    # on RDNA 3.5 (measured; csrc_hip/README.md), so fp8 here is storage only
    # and the compute is bf16 with f32 accumulate — strictly more accurate than
    # the fp8-MMA path the CUDA lane uses, not less.
    y = ext.cb_gemm_fp8(x, layer._cb_qw_padded, cb8, layer._cb_row_offset,
                        layer._cb_scale, N, K, method.k, method.n_sub,
                        method.type_size, True)
    return y + bias if bias is not None else y
