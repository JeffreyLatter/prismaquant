"""Shared OCP-MX quantization adapters backed by compressed-tensors.

MXFP8_E4M3 is a commodity compressed-tensors/vLLM format: grouped FP8
values with E8M0 power-of-two scales. Keep the qparam and cast semantics in
one place and defer scale generation to compressed-tensors, which is the
load/export authority for this on-disk representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from compressed_tensors.compressors.mx_utils import (
    compress_mx_scale,
    decompress_mx_scale,
)
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.quant_scheme import MXFP8
from compressed_tensors.quantization.utils.helpers import calculate_qparams

from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm


@dataclass(frozen=True)
class MXFP8Result:
    quant: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


def e8m0_to_scale(
    e8m0_uint8: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Decode compressed-tensors E8M0 scale bytes to float32 powers of two."""
    target_device = device if device is not None else e8m0_uint8.device
    return decompress_mx_scale(e8m0_uint8.to(device=target_device)).to(
        device=target_device,
        dtype=torch.float32,
    )


def mxfp8_e4m3_qdq(
    values: torch.Tensor,
    *,
    group_size: int = 32,
    fallback_plain_activation: bool = False,
) -> MXFP8Result:
    """Quantize/dequantize values as MXFP8_E4M3 using compressed-tensors.

    ``values`` may be any rank as long as the final dimension is the feature
    dimension. Scales are generated per final-dimension group and returned as
    uint8 E8M0 metadata with shape ``values.shape[:-1] + (K / group_size,)``.
    """
    orig_shape = values.shape
    if len(orig_shape) == 0:
        raise ValueError("MXFP8_E4M3 requires at least one tensor dimension")
    cols = int(orig_shape[-1])
    if cols % group_size != 0:
        if fallback_plain_activation:
            result = fp8_dynamic_activation_qdq_vllm(
                values,
                element_dtype=torch.float8_e4m3fn,
                element_max=float(torch.finfo(torch.float8_e4m3fn).max),
            )
            return MXFP8Result(
                quant=result.quant,
                scale=result.scale,
                dequant=result.dequant,
            )
        raise ValueError(
            f"MXFP8_E4M3 group_size={group_size} does not divide K={cols}"
        )

    rows = values.to(torch.float32).reshape(-1, cols)
    grouped = rows.reshape(rows.shape[0], cols // group_size, group_size)
    args = MXFP8["weights"]
    scale, zero_point = calculate_qparams(
        grouped.amin(dim=-1),
        grouped.amax(dim=-1),
        args,
    )
    quant_rows = quantize(
        rows,
        scale,
        zero_point,
        args,
        dtype=torch.float8_e4m3fn,
    )
    e8m0 = compress_mx_scale(scale, torch.uint8)
    scale_f = e8m0_to_scale(e8m0, device=values.device)
    dequant = (
        quant_rows.reshape_as(grouped).to(torch.float32)
        * scale_f.unsqueeze(-1)
    )
    scale_shape = (*orig_shape[:-1], cols // group_size)
    return MXFP8Result(
        quant=quant_rows.reshape(orig_shape),
        scale=e8m0.reshape(scale_shape),
        dequant=dequant.reshape(orig_shape),
    )


def mxfp8_e4m3_weight_qdq(
    weight: torch.Tensor,
    *,
    group_size: int = 32,
) -> MXFP8Result:
    return mxfp8_e4m3_qdq(weight, group_size=group_size)


def mxfp8_e4m3_activation_qdq_vllm(
    activation: torch.Tensor,
    *,
    group_size: int = 32,
) -> MXFP8Result:
    return mxfp8_e4m3_qdq(
        activation,
        group_size=group_size,
        fallback_plain_activation=True,
    )
