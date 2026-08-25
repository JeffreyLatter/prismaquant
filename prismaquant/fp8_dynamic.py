"""Shared FP8_DYNAMIC quantization adapters.

Use compressed-tensors primitives for FP8 cast/dequant behavior and qparam
generation where that library is the exporter/load-time authority. Activation
QDQ follows vLLM's served Linear path: flatten to rows and compute one dynamic
FP8 scale per row/token.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC
from compressed_tensors.quantization.utils.helpers import calculate_qparams


@dataclass(frozen=True)
class FP8DynamicResult:
    quant: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


def _fallback_fp8_qdq(
    values: torch.Tensor,
    scale: torch.Tensor,
    *,
    element_dtype: torch.dtype,
    element_max: float,
) -> FP8DynamicResult:
    values_f = values.to(torch.float32)
    scale_f = scale.to(device=values_f.device, dtype=torch.float32).clamp_min(
        2.0 ** -127,
    )
    quant = (
        values_f / scale_f
    ).clamp(-float(element_max), float(element_max)).to(element_dtype)
    dequant = quant.to(torch.float32) * scale_f
    return FP8DynamicResult(quant=quant, scale=scale_f, dequant=dequant)


def fp8_dynamic_weight_qdq(
    weight: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = 448.0,
) -> FP8DynamicResult:
    """Compressed-tensors FP8_DYNAMIC static per-channel weight QDQ."""
    weight_f = weight.to(torch.float32)
    rows = weight_f.reshape(-1, weight_f.shape[-1])

    if element_dtype != torch.float8_e4m3fn:
        scale = (
            rows.abs().amax(dim=-1, keepdim=True).clamp_min(2.0 ** -127)
            / float(element_max)
        )
        result = _fallback_fp8_qdq(
            rows,
            scale,
            element_dtype=element_dtype,
            element_max=element_max,
        )
        return FP8DynamicResult(
            quant=result.quant.reshape(weight.shape),
            scale=result.scale.reshape(*weight.shape[:-1], 1),
            dequant=result.dequant.reshape(weight.shape),
        )

    args = FP8_DYNAMIC["weights"]
    scale, zero_point = calculate_qparams(
        rows.amin(dim=-1),
        rows.amax(dim=-1),
        args,
    )
    scale = scale.reshape(-1, 1)
    zero_point = zero_point.reshape(-1, 1)
    quant = quantize(rows, scale, zero_point, args, dtype=element_dtype)
    dequant = quant.to(torch.float32) * scale
    return FP8DynamicResult(
        quant=quant.reshape(weight.shape),
        scale=scale.reshape(*weight.shape[:-1], 1),
        dequant=dequant.reshape(weight.shape),
    )


def fp8_dynamic_activation_qdq_vllm(
    activation: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = 448.0,
) -> FP8DynamicResult:
    """vLLM dynamic-token FP8 activation QDQ for served Linear inputs."""
    act_f = activation.to(torch.float32)
    rows = act_f.reshape(-1, act_f.shape[-1])
    min_scale = 1.0 / (float(element_max) * 512.0)
    scale = (
        rows.abs().amax(dim=-1, keepdim=True) / float(element_max)
    ).clamp_min(min_scale)

    if element_dtype != torch.float8_e4m3fn:
        result = _fallback_fp8_qdq(
            rows,
            scale,
            element_dtype=element_dtype,
            element_max=element_max,
        )
    else:
        args = FP8_DYNAMIC["input_activations"]
        zero_point = torch.zeros_like(scale)
        quant = quantize(rows, scale, zero_point, args, dtype=element_dtype)
        dequant = quant.to(torch.float32) * scale
        result = FP8DynamicResult(quant=quant, scale=scale, dequant=dequant)

    return FP8DynamicResult(
        quant=result.quant.reshape(activation.shape),
        scale=result.scale.reshape(*activation.shape[:-1], 1),
        dequant=result.dequant.reshape(activation.shape),
    )
