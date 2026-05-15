"""Fold-only DuQuant++-inspired microscale preconditioning.

Full DuQuant++ applies block rotations at activation quantization time. That
requires runtime/kernel support, so it is not part of PrismaQuant's vanilla
vLLM production path. This module implements the fold-only subset: choose a
per-input-channel scale that reduces microscale block outlier pressure, then
let the existing AWQ/SmoothQuant fold machinery materialize the identity

    (x / s) @ Q(W * s)^T

with no new runtime operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from prismaquant.awq import AwqSearchTarget, normalize_awq_scale
from prismaquant import format_registry as fr


@dataclass(frozen=True)
class DuQuantFormatTraits:
    format: str
    family: str
    weight_bits: int
    group_size: int
    scale_dtype_name: str


def duquant_format_traits(fmt: str) -> DuQuantFormatTraits | None:
    """Return microscale traits for formats this fold-only path can target."""

    try:
        spec = fr.get_format(str(fmt).strip().upper())
    except Exception:
        return None
    if spec.group_size <= 0:
        return None
    if spec.family not in {"nv", "mx"}:
        return None
    return DuQuantFormatTraits(
        format=spec.name,
        family=spec.family,
        weight_bits=int(spec.weight_bits),
        group_size=int(spec.group_size),
        scale_dtype_name=str(spec.scale_dtype_name),
    )


def supports_duquant_fold(fmt: str) -> bool:
    return duquant_format_traits(fmt) is not None


def _block_geomean(values: torch.Tensor, group_size: int, eps: float) -> torch.Tensor:
    vals = values.detach().to(torch.float32).reshape(-1).clamp_min(eps)
    n = int(vals.numel())
    g = max(1, int(group_size))
    pad = (-n) % g
    if pad:
        vals = torch.cat([vals, torch.ones(pad, device=vals.device, dtype=vals.dtype)])
    grouped = vals.reshape(-1, g)
    center = grouped.log().mean(dim=1, keepdim=True).exp().clamp_min(eps)
    out = center.expand_as(grouped).reshape(-1)[:n]
    return out


def duquant_block_scale_from_stats(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    group_size: int,
    alpha: float,
    eps: float = 1e-4,
    clamp_ratio: float = 10.0,
) -> torch.Tensor:
    """Build one block-aware fold scale candidate.

    Large activation columns get a larger ``s`` so runtime activations
    ``x/s`` have less per-block outlier pressure. Large weight columns push
    ``s`` back down so ``W*s`` does not become harder to quantize. Statistics
    are normalized inside each microscale block, matching the shared-scale
    failure mode this method is meant to address.
    """

    if weight.dim() != 2:
        raise ValueError("DuQuant fold scale expects a 2D Linear weight")
    cols = int(weight.shape[1])
    x = activations.detach().to(device=weight.device, dtype=torch.float32)
    x = x.reshape(-1, cols)
    w = weight.detach().to(torch.float32)
    x_stat = x.abs().amax(dim=0).clamp_min(eps)
    w_stat = w.abs().amax(dim=0).clamp_min(eps)
    x_center = _block_geomean(x_stat, group_size, eps)
    w_center = _block_geomean(w_stat, group_size, eps)
    x_rel = (x_stat / x_center).clamp_min(eps)
    w_rel = (w_stat / w_center).clamp_min(eps)
    a = float(alpha)
    raw = x_rel.pow(a) / w_rel.pow(1.0 - a)
    raw = raw / _block_geomean(raw, group_size, eps).clamp_min(eps)
    return normalize_awq_scale(raw, eps=eps, clamp_ratio=clamp_ratio)


def duquant_scale_from_targets(
    targets: Sequence[AwqSearchTarget],
    *,
    alpha: float,
    clamp_ratio: float = 10.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Combine per-target block scales into one shared predecessor scale."""

    if not targets:
        raise ValueError("DuQuant fold scale requires at least one target")
    cols = int(targets[0].weight.shape[1])
    device = targets[0].weight.device
    logs: list[torch.Tensor] = []
    for target in targets:
        if int(target.weight.shape[1]) != cols:
            raise ValueError("all DuQuant targets must share input width")
        traits = duquant_format_traits(target.fmt)
        if traits is None:
            continue
        scale = duquant_block_scale_from_stats(
            target.weight.to(device=device, dtype=torch.float32),
            target.activations.to(device=device, dtype=torch.float32),
            group_size=traits.group_size,
            alpha=float(alpha),
            eps=eps,
            clamp_ratio=clamp_ratio,
        )
        logs.append(scale.clamp_min(eps).log())
    if not logs:
        return torch.ones(cols, device=device, dtype=torch.float32)
    merged = torch.stack(logs, dim=0).mean(dim=0).exp()
    return normalize_awq_scale(merged, eps=eps, clamp_ratio=clamp_ratio)
