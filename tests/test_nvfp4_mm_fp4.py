"""Tests for the GB10 / sm_121 hardware NVFP4 GEMM adapter.

These exercise ``prismaquant.kernels.nvfp4_mm_fp4`` — the flashinfer
``mm_fp4`` wrapper used by the opt-in ``PRISMAQUANT_FP4_GEMM`` path. They
require a Blackwell-class GPU (sm_120+) and flashinfer; they skip cleanly
on any other host.
"""
import pytest
import torch

from prismaquant.kernels import nvfp4_mm_fp4

pytestmark = pytest.mark.skipif(
    not nvfp4_mm_fp4.is_available(),
    reason="hardware NVFP4 GEMM unavailable (needs sm_120+ and flashinfer)",
)


def _relerr(y, ref):
    return (
        (y.float() - ref.float()).norm()
        / ref.float().norm().clamp_min(1e-9)
    ).item()


def test_aw_matmul_matches_bf16_within_nvfp4_tolerance():
    torch.manual_seed(0)
    M, N, K = 256, 5120, 5120
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    y = nvfp4_mm_fp4.aw_matmul(x, w)
    assert y is not None
    assert y.shape == (M, N)
    assert y.dtype == torch.bfloat16
    ref = x.float() @ w.float().t()
    # NVFP4 is 4-bit; ~13% relative error vs the bf16 truth is expected.
    assert _relerr(y, ref) < 0.25


def test_below_threshold_returns_none():
    # Tiny GEMM, below min_problem_size — caller must fall back.
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    assert nvfp4_mm_fp4.aw_matmul(x, w) is None


def test_quantize_gemm_roundtrip_matches_aw_matmul():
    torch.manual_seed(1)
    M, N, K = 256, 4096, 4096
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    y_conv = nvfp4_mm_fp4.aw_matmul(x, w)
    assert y_conv is not None
    y_split = nvfp4_mm_fp4.gemm(
        nvfp4_mm_fp4.quantize(x), nvfp4_mm_fp4.quantize(w), N,
    )
    assert torch.equal(y_conv, y_split)


def test_bad_inputs_return_none():
    # fp32 activation is unsupported -> None.
    x = torch.randn(256, 4096, device="cuda", dtype=torch.float32)
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    assert nvfp4_mm_fp4.aw_matmul(x, w) is None
    # K not divisible by the block size (16) -> None.
    x = torch.randn(256, 4095, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 4095, device="cuda", dtype=torch.bfloat16)
    assert nvfp4_mm_fp4.aw_matmul(x, w) is None


def test_3d_activation_shape_preserved():
    torch.manual_seed(2)
    B, T, K, N = 8, 256, 4096, 4096
    x = torch.randn(B, T, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    y = nvfp4_mm_fp4.aw_matmul(x, w)
    assert y is not None
    assert y.shape == (B, T, N)
