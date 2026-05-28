from __future__ import annotations

import torch

from prismaquant import format_registry as fr
from prismaquant.export_native_compressed import (
    _mxfp8_dequantize_2d,
    quantize_dequantize_fp8_dynamic,
    quantize_dequantize_mxfp8,
)


def test_plain_fp8_rtn_uses_eager_path(monkeypatch):
    compile_calls = []

    def fake_compile(fn, *args, **kwargs):
        compile_calls.append((fn, args, kwargs))

        def compiled(*_args, **_kwargs):
            raise AssertionError("plain FP8 RTN should not use torch.compile")

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    quantize = fr._make_rtn("fp8_e4m3", 0)
    x = torch.linspace(-3.1, 3.1, steps=64, dtype=torch.float32).reshape(2, 32)
    y = quantize(x)

    assert compile_calls == []
    assert y.shape == x.shape
    assert not torch.equal(y, x)


def test_plain_fp8_weight_matches_compressed_tensors_fp8_dynamic():
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC
    from compressed_tensors.quantization.utils.helpers import calculate_qparams

    vals = torch.tensor(
        [
            0.0,
            1e-12,
            -1e-12,
            1e-8,
            -1e-8,
            1e-6,
            -1e-6,
            1.0 / 1024.0,
            -1.0 / 1024.0,
            1.0 / 512.0,
            -1.0 / 512.0,
            240.0,
            -240.0,
            448.0,
            -448.0,
        ],
        dtype=torch.float32,
    )
    w = vals.repeat(4, 1)
    w[1] *= 1e-6
    w[2] = 0.0

    registry = fr.get_format("FP8_E4M3").quantize_dequantize(w)

    args = FP8_DYNAMIC["weights"]
    scale, zero_point = calculate_qparams(w.amin(dim=1), w.amax(dim=1), args)
    compressed_tensors = fake_quantize(
        w,
        scale.reshape(-1, 1),
        zero_point.reshape(-1, 1),
        args,
    )

    export_q, export_scale = quantize_dequantize_fp8_dynamic(w)
    export = export_q.float() * export_scale

    assert torch.allclose(registry, compressed_tensors, atol=0.0, rtol=0.0)
    assert torch.allclose(registry, export, atol=0.0, rtol=0.0)


def test_plain_fp8_rank3_activation_matches_compressed_tensors_fp8_dynamic():
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC
    from compressed_tensors.quantization.utils.helpers import (
        compute_dynamic_scales_and_zp,
    )

    torch.manual_seed(19)
    x = torch.randn(2, 5, 32, dtype=torch.float32) * 3.0
    x[0, 0, :8] = torch.tensor(
        [
            0.0,
            1.0 / 1024.0,
            -1.0 / 1024.0,
            1.0 / 512.0,
            -1.0 / 512.0,
            240.0,
            448.0,
            -448.0,
        ],
        dtype=torch.float32,
    )

    registry = fr.get_format("FP8_E4M3").activation_quantize_dequantize(x)

    args = FP8_DYNAMIC["input_activations"]
    dummy = torch.nn.Linear(x.shape[-1], 1, bias=False)
    scale, zero_point = compute_dynamic_scales_and_zp(x, args, dummy)
    compressed_tensors = fake_quantize(x, scale, zero_point, args)

    assert torch.allclose(registry, compressed_tensors, atol=0.0, rtol=0.0)


def test_plain_fp8_rank2_activation_matches_vllm_dynamic_token_reference():
    x = torch.tensor(
        [
            [
                0.0,
                1e-12,
                -1e-12,
                1e-8,
                -1e-8,
                1e-6,
                -1e-6,
                1.0 / 1024.0,
                -1.0 / 1024.0,
                1.0 / 512.0,
                -1.0 / 512.0,
                1.0,
                -1.0,
                448.0,
                -448.0,
                0.25,
            ],
            torch.linspace(-3.0, 3.0, steps=16, dtype=torch.float32),
        ],
        dtype=torch.float32,
    )
    registry = fr.get_format("FP8_E4M3").activation_quantize_dequantize(x)

    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    min_scale = 1.0 / (fp8_max * 512.0)
    scale = (x.abs().amax(dim=-1, keepdim=True) / fp8_max).clamp_min(
        min_scale,
    )
    quant = (x / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    reference = quant.float() * scale

    assert torch.allclose(registry, reference, atol=0.0, rtol=0.0)


def test_mx_e8m0_rtn_matches_export_scale_rounding():
    cases = [torch.linspace(-3.7, 3.7, steps=64, dtype=torch.float32).reshape(2, 32)]
    for seed in range(5):
        torch.manual_seed(seed)
        cases.append(torch.randn(16, 64, dtype=torch.float32) * 10 ** (seed - 2))

    for w in cases:
        registry = fr.get_format("MXFP8_E4M3").quantize_dequantize(w)
        export_q, export_scales = quantize_dequantize_mxfp8(w)
        export = _mxfp8_dequantize_2d(export_q, export_scales)

        assert torch.allclose(registry, export, atol=0.0, rtol=0.0)


def test_mxfp8_exported_scales_match_compressed_tensors():
    from compressed_tensors.quantization.utils.mxfp_utils import generate_mx_scales

    torch.manual_seed(11)
    w = torch.randn(16, 96, dtype=torch.float32) * 7.0

    _, export_scales = quantize_dequantize_mxfp8(w)
    grouped = w.reshape(16, 3, 32)
    expected_scales = generate_mx_scales(
        grouped.abs().amax(dim=-1),
        num_bits=8,
    ).to(torch.uint8)

    assert torch.equal(export_scales, expected_scales)


def test_mxfp8_activation_quantizer_matches_vllm_runtime_reference():
    x = torch.randn(9, 64, dtype=torch.float32) * 17.0
    registry = fr.get_format("MXFP8_E4M3").activation_quantize_dequantize(x)

    blocked = x.reshape(9, 2, 32)
    amax = blocked.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    max_pos = float(torch.finfo(torch.float8_e4m3fn).max)
    scale_unbiased = torch.ceil(torch.log2(amax / max_pos)).clamp(-127, 127)
    descale = torch.exp2(scale_unbiased)
    quant = (
        blocked / descale.unsqueeze(-1)
    ).clamp(-max_pos, max_pos).reshape_as(x).to(torch.float8_e4m3fn)
    reference = (
        quant.float().reshape_as(blocked) * descale.unsqueeze(-1)
    ).reshape_as(x)

    assert torch.allclose(registry, reference, atol=0.0, rtol=0.0)


def test_mxfp8_activation_quantizer_uses_e4m3_range():
    block = torch.cat([
        torch.tensor([14.0], dtype=torch.float32),
        torch.linspace(-0.02, 0.02, steps=31, dtype=torch.float32),
    ])
    x = block.repeat(2).reshape(1, 64)

    corrected = fr.get_format("MXFP8_E4M3").activation_quantize_dequantize(x)

    blocked = x.reshape(1, 2, 32)
    amax = blocked.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    raw_amax_scale = torch.exp2(torch.floor(torch.log2(amax)).clamp(-127, 127))
    raw_amax_quant = (
        blocked / raw_amax_scale.unsqueeze(-1)
    ).reshape_as(x).to(torch.float8_e4m3fn)
    raw_amax_reference = (
        raw_amax_quant.float().reshape_as(blocked) * raw_amax_scale.unsqueeze(-1)
    ).reshape_as(x)

    corrected_mse = torch.mean((corrected.float() - x) ** 2)
    raw_amax_mse = torch.mean((raw_amax_reference.float() - x) ** 2)

    assert corrected_mse < raw_amax_mse * 0.01
