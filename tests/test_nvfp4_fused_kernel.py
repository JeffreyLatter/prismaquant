import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.kernels.nvfp4_fused import (
    nvfp4_dequantize_weight,
    nvfp4_fused_aw_matmul,
    nvfp4_pack_weight,
)


def _nvfp4_quant_then_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    nvfp4 = fr.get_format("NVFP4")
    qx = nvfp4.activation_quantize_dequantize(x)
    qw = nvfp4.quantize_dequantize(weight)
    return qx @ qw.t()


def test_nvfp4_pack_weight_matches_format_registry_reference():
    torch.manual_seed(0)
    weight = (torch.randn(17, 64) * 0.2).to(torch.bfloat16)

    w_packed, w_scales, w_global_scale = nvfp4_pack_weight(weight)
    dequant = nvfp4_dequantize_weight(
        w_packed,
        w_scales,
        w_global_scale,
        dtype=weight.dtype,
    )
    reference = fr.get_format("NVFP4").quantize_dequantize(weight)

    assert w_packed.dtype == torch.uint8
    assert w_packed.shape == (17, 32)
    assert w_scales.shape == (17, 4)
    torch.testing.assert_close(dequant, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")
@pytest.mark.parametrize(
    ("M", "N", "K"),
    [
        (4, 32, 64),
        (17, 96, 128),
        (16, 2560, 2560),
    ],
)
def test_nvfp4_fused_matches_unfused_path(M, N, K):
    torch.manual_seed(1234 + M + N + K)
    device = torch.device("cuda")
    x = (torch.randn(M, K, device=device) * 0.05).to(torch.bfloat16)
    weight = (torch.randn(N, K, device=device) * 0.05).to(torch.bfloat16)

    w_packed, w_scales, w_global_scale = nvfp4_pack_weight(weight)
    out_fused = nvfp4_fused_aw_matmul(x, w_packed, w_scales, w_global_scale)
    out_reference = _nvfp4_quant_then_matmul(x, weight)
    max_abs = (out_fused.float() - out_reference.float()).abs().max().item()

    assert torch.allclose(out_fused, out_reference, atol=6e-3, rtol=2e-2), (
        f"max_abs_diff={max_abs:.6g}"
    )
