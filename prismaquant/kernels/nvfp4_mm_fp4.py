"""Hardware NVFP4 GEMM for GB10 / sm_121 via flashinfer ``mm_fp4``.

GB10 (consumer/workstation Blackwell, sm_121) has no ``tcgen05``/TMEM; its
FP4 path is the SM120-family block-scaled warp ``mma``, exposed by CUTLASS
4.5.0 and wrapped by ``flashinfer.mm_fp4``. This module is the thin adapter
PrismaQuant uses to route NVFP4 measurement-loop GEMMs onto that path.

OPT-IN / EXPERIMENTAL. Gated by ``PRISMAQUANT_FP4_GEMM`` (see
perturbed_x_cache). It is NOT a transparent drop-in for PrismaQuant's RTN
cost model: PrismaQuant's NVFP4 uses continuous fp32 block scales, the
hardware path uses fp8_e4m3 block scales (the faithful, vLLM-served form).
A micro-benchmark put the per-layer output shift at ~10%. Promotion to
default-on requires an apples-to-apples KL/bpp A/B per docs/design_guidelines.

Backend notes (measured on this GB10):
  - backend='cutlass' works (CUTLASS C++ block-scaled sm_121 kernels).
  - backend='b12x'  -> DSL ICE (broken nvidia-cutlass-dsl 4.5.0, issue #3227).
  - backend='auto'  -> selects cuDNN, which has no sm_121 FP4 engine -> fails.
  So we pin backend='cutlass'.
"""
from __future__ import annotations

import os

import torch

_FLASHINFER = None
_AVAILABLE: bool | None = None


def _flashinfer():
    global _FLASHINFER
    if _FLASHINFER is None:
        import flashinfer  # noqa: PLC0415
        _FLASHINFER = flashinfer
    return _FLASHINFER


def is_available() -> bool:
    """True when a hardware NVFP4 GEMM can run on this device.

    Requires CUDA, a Blackwell-class capability (sm_120+ — the consumer
    block-scaled path), and an importable flashinfer.
    """
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    avail = False
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap >= (12, 0):
                _flashinfer()
                avail = True
    except Exception:
        avail = False
    _AVAILABLE = avail
    return avail


def min_problem_size() -> int:
    """Minimum M*N*K below which the FP4 GEMM is not worth it.

    The micro-benchmark showed sub-~2e9-flop GEMMs are launch-bound and
    FP4 gives no reliable win (the smallest probe shape was ambiguous,
    0.9-3.5x across runs). Tunable via ``PRISMAQUANT_FP4_GEMM_MIN_MNK``.
    """
    try:
        v = int(os.environ.get("PRISMAQUANT_FP4_GEMM_MIN_MNK", "").strip())
        if v > 0:
            return v
    except (ValueError, AttributeError):
        pass
    return 2_000_000_000


def quantize(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a ``[..., K]`` bf16/fp16 tensor to NVFP4.

    Returns ``(packed, block_scale, global_sf)`` ready for :func:`gemm`.
    ``global_sf`` maps the tensor amax into the e4m3 block-scale range
    (e2m1 max 6, e4m3 max 448).
    """
    fi = _flashinfer()
    flat = t.reshape(-1, t.shape[-1]).contiguous()
    amax = flat.abs().amax().clamp_min(1e-8).float()
    global_sf = ((448.0 * 6.0) / amax).reshape(1).to(torch.float32).to(flat.device)
    packed, block_scale = fi.nvfp4_quantize(flat, global_sf, sf_vec_size=16)
    return packed, block_scale, global_sf


def gemm(
    xq: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    wq: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    out_features: int,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run ``y = x @ w.T`` from pre-quantized NVFP4 operands.

    ``xq`` is :func:`quantize` of the activation ``[M, K]``; ``wq`` is
    :func:`quantize` of the weight ``[N, K]`` (nn.Linear layout). Result is
    ``[M, out_features]``.
    """
    fi = _flashinfer()
    xpacked, xsf, xg = xq
    wpacked, wsf, wg = wq
    alpha = (1.0 / (xg * wg)).to(torch.float32)
    # b must be (k_packed, n) column-major: wpacked is [N, K/2] row-major,
    # so wpacked.t() is exactly the [K/2, N] column-major operand.
    return fi.mm_fp4(
        xpacked, wpacked.t(), xsf, wsf,
        alpha=alpha, out_dtype=out_dtype, block_size=16,
        backend="cutlass", use_nvfp4=True,
    )


def aw_matmul(
    x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Convenience: quantize ``x`` and ``w`` and run the NVFP4 GEMM.

    ``x``: ``[..., K]`` activation; ``w``: ``[N, K]`` weight. Returns
    ``[..., N]`` bf16, or ``None`` when the call is not viable (unavailable,
    bad dtype/shape, or below :func:`min_problem_size` — caller falls back).
    Re-quantizes the weight every call; callers in a hot loop should cache
    :func:`quantize` of the weight and call :func:`gemm` directly.
    """
    if not is_available():
        return None
    if x.dtype not in (torch.bfloat16, torch.float16) or not x.is_cuda:
        return None
    K = x.shape[-1]
    N = w.shape[0]
    if w.ndim != 2 or w.shape[1] != K or K % 16 != 0:
        return None
    flat = x.reshape(-1, K)
    if flat.shape[0] * N * K < min_problem_size():
        return None
    try:
        out = gemm(quantize(flat), quantize(w), N)
    except Exception:
        return None
    out = out.reshape(*x.shape[:-1], N)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out
