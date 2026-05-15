"""Activation-aware weight quantization scale search.

This module contains the shared AWQ-v2 numerical core used by production
cache fill and export.  It deliberately does not know about model topology or
compressed-tensors metadata; callers provide a renderer for the target format.

The convention is the standard AWQ identity:

    W' = W * s
    x' = x / s

The renderer quantizes ``W'`` in the target format and the scorer evaluates the
effective original-coordinate weight ``Q(W') / s`` on the original activation
samples.  Callers that can fold ``1/s`` into a predecessor at export time store
``Q(W')`` in the artifact; cache/probe paths store ``Q(W') / s`` so the original
unfolded model sees the same outputs.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch

from prismaquant.render_score import gate_render_candidate, score_render_error


DEFAULT_AWQ_MIN_GAIN = 0.03


@dataclass(frozen=True)
class AwqSearchTarget:
    """One Linear participating in a shared AWQ scale search."""

    name: str
    fmt: str
    weight: torch.Tensor
    activations: torch.Tensor
    group_size: int = 0
    score_weight: float = 1.0
    row_weights: torch.Tensor | None = None


@dataclass
class AwqSearchResult:
    scale: torch.Tensor
    selected_label: str
    selected_ratio: float | None
    baseline_score: float
    best_score: float
    relative_gain: float
    n_candidates: int
    trace: list[dict[str, float | str | None]]
    gate_reason: str = "improved"


RenderScaledFn = Callable[
    [int, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def _flatten_activations(
    activations: torch.Tensor,
    cols: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if activations.shape[-1] != cols:
        raise ValueError(
            f"AWQ activation width {activations.shape[-1]} != weight.in {cols}"
        )
    return activations.detach().to(device=device, dtype=torch.float32).reshape(-1, cols)


def awq_activation_mean(
    activations: Sequence[torch.Tensor],
    cols: int,
    *,
    device: torch.device,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Return per-input-channel mean absolute activation."""

    total = torch.zeros(cols, device=device, dtype=torch.float32)
    count = 0
    for a in activations:
        x = _flatten_activations(a, cols, device=device)
        total += x.abs().sum(dim=0)
        count += int(x.shape[0])
    if count <= 0:
        return torch.ones(cols, device=device, dtype=torch.float32)
    return (total / float(count)).clamp_min(eps)


def _normalized_weight_abs(
    weight: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    w = weight.detach().to(torch.float32)
    rows, cols = w.shape
    if group_size > 0 and cols % group_size == 0:
        grouped = w.abs().reshape(rows, cols // group_size, group_size)
        denom = grouped.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return (grouped / denom).reshape(rows, cols)
    denom = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    return w.abs() / denom


def awq_weight_mean(
    weights: Sequence[torch.Tensor],
    group_sizes: Sequence[int],
    cols: int,
    *,
    device: torch.device,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Return AutoAWQ-style per-input-channel normalized weight mean."""

    total = torch.zeros(cols, device=device, dtype=torch.float32)
    count = 0
    for weight, group_size in zip(weights, group_sizes):
        if weight.shape[1] != cols:
            raise ValueError(
                f"AWQ weight width {weight.shape[1]} != expected {cols}"
            )
        norm = _normalized_weight_abs(weight.to(device), int(group_size))
        total += norm.sum(dim=0)
        count += int(norm.shape[0])
    if count <= 0:
        return torch.ones(cols, device=device, dtype=torch.float32)
    return (total / float(count)).clamp_min(eps)


def normalize_awq_scale(
    scale: torch.Tensor,
    *,
    eps: float = 1e-4,
    clamp_ratio: float = 10.0,
) -> torch.Tensor:
    """Geomean-normalize and log-symmetrically clamp an AWQ scale."""

    s = scale.detach().to(torch.float32)
    s = torch.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)
    finite = s[torch.isfinite(s) & (s > 0)]
    if finite.numel() == 0:
        return torch.ones_like(s)
    norm = (finite.max() * finite.min()).sqrt().clamp_min(eps)
    s = s / norm
    s = s.clamp(1.0 / float(clamp_ratio), float(clamp_ratio))
    return torch.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0)


def awq_scale_from_stats(
    x_mean: torch.Tensor,
    w_mean: torch.Tensor,
    ratio: float,
    *,
    duo_scaling: bool = True,
    eps: float = 1e-4,
    clamp_ratio: float = 10.0,
) -> torch.Tensor:
    """Build one AWQ candidate scale from activation/weight statistics."""

    r = float(ratio)
    if duo_scaling:
        raw = x_mean.clamp_min(eps).pow(r) / w_mean.clamp_min(eps).pow(1.0 - r)
    else:
        raw = x_mean.clamp_min(eps).pow(r)
    return normalize_awq_scale(raw, eps=eps, clamp_ratio=clamp_ratio)


def legacy_activation_awq_scale(
    activations: torch.Tensor,
    *,
    eps: float = 1e-4,
    clamp_ratio: float = 10.0,
) -> torch.Tensor:
    """Legacy activation-only alpha=0.5 AWQ scale kept for tests/ablation."""

    cols = int(activations.shape[-1])
    x_mean = awq_activation_mean(
        [activations],
        cols,
        device=activations.device,
        eps=eps,
    )
    return normalize_awq_scale(
        x_mean.pow(0.5),
        eps=eps,
        clamp_ratio=clamp_ratio,
    )


def awq_output_mse(
    weight: torch.Tensor,
    rendered_effective: torch.Tensor,
    activations: torch.Tensor,
    *,
    row_weights: torch.Tensor | None = None,
    row_chunk: int = 128,
) -> float:
    """Mean output-space MSE for ``rendered_effective`` on ``activations``."""
    return score_render_error(
        weight,
        rendered_effective,
        activations,
        row_weights=row_weights,
        row_chunk=row_chunk,
    )


def _candidate_scales(
    targets: Sequence[AwqSearchTarget],
    *,
    n_grid: int,
    duo_scaling: bool,
    eps: float,
    clamp_ratio: float,
) -> list[tuple[str, float | None, torch.Tensor]]:
    if not targets:
        raise ValueError("AWQ search requires at least one target")
    cols = int(targets[0].weight.shape[1])
    device = targets[0].weight.device
    x_mean = awq_activation_mean(
        [t.activations for t in targets],
        cols,
        device=device,
        eps=eps,
    )
    w_mean = awq_weight_mean(
        [t.weight for t in targets],
        [t.group_size for t in targets],
        cols,
        device=device,
        eps=eps,
    )

    candidates: list[tuple[str, float | None, torch.Tensor]] = [
        ("identity", None, torch.ones(cols, device=device, dtype=torch.float32)),
        (
            "legacy_x_alpha_0.5",
            0.5,
            normalize_awq_scale(x_mean.pow(0.5), eps=eps, clamp_ratio=clamp_ratio),
        ),
    ]
    grid = max(2, int(n_grid))
    for i in range(grid):
        ratio = i / float(grid)
        candidates.append((
            f"duo_ratio_{ratio:.4f}" if duo_scaling else f"x_ratio_{ratio:.4f}",
            ratio,
            awq_scale_from_stats(
                x_mean,
                w_mean,
                ratio,
                duo_scaling=duo_scaling,
                eps=eps,
                clamp_ratio=clamp_ratio,
            ),
        ))

    deduped: list[tuple[str, float | None, torch.Tensor]] = []
    seen: set[bytes] = set()
    for label, ratio, scale in candidates:
        key = (
            torch.round(scale.detach().cpu() * 1_000_000)
            .to(torch.int64)
            .numpy()
            .tobytes()
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, ratio, scale))
    return deduped


def search_awq_scale(
    targets: Sequence[AwqSearchTarget],
    render_scaled: RenderScaledFn,
    *,
    n_grid: int = 20,
    duo_scaling: bool = True,
    eps: float = 1e-4,
    clamp_ratio: float = 10.0,
    min_gain: float = 0.0,
    row_chunk: int = 128,
) -> AwqSearchResult:
    """Search AWQ scales by rendering and scoring real quantized weights.

    ``render_scaled`` must return the dequantized weight in scaled coordinates
    for target ``i``.  This function divides by ``scale`` before scoring so the
    metric is always in the original model coordinates.
    """

    if not targets:
        raise ValueError("AWQ search requires at least one target")
    cols = int(targets[0].weight.shape[1])
    device = targets[0].weight.device
    for t in targets:
        if int(t.weight.shape[1]) != cols or int(t.activations.shape[-1]) != cols:
            raise ValueError("all AWQ targets must share input width")

    candidates = _candidate_scales(
        targets,
        n_grid=n_grid,
        duo_scaling=duo_scaling,
        eps=eps,
        clamp_ratio=clamp_ratio,
    )

    best_scale = candidates[0][2]
    best_label = candidates[0][0]
    best_ratio = candidates[0][1]
    best_score = float("inf")
    baseline_score = float("inf")
    trace: list[dict[str, float | str | None]] = []

    with torch.no_grad():
        for label, ratio, scale in candidates:
            scale = scale.to(device=device, dtype=torch.float32).clamp_min(eps)
            total = 0.0
            for idx, target in enumerate(targets):
                w = target.weight.detach().to(device=device, dtype=torch.float32)
                x = target.activations.detach().to(device=device, dtype=torch.float32)
                w_scaled = w * scale.unsqueeze(0)
                x_scaled = x.reshape(-1, cols) / scale.unsqueeze(0)
                rendered_scaled = render_scaled(idx, w_scaled, x_scaled, scale)
                effective = rendered_scaled.to(device=device, dtype=torch.float32)
                effective = effective / scale.unsqueeze(0)
                score = awq_output_mse(
                    w,
                    effective,
                    target.activations,
                    row_weights=target.row_weights,
                    row_chunk=row_chunk,
                )
                total += float(target.score_weight) * score
            if label == "identity":
                baseline_score = float(total)
            if total < best_score:
                best_score = float(total)
                best_scale = scale.detach().clone()
                best_label = label
                best_ratio = ratio
            trace.append({
                "label": label,
                "ratio": None if ratio is None else float(ratio),
                "score": float(total),
                "best_score": float(best_score),
            })

    if not math.isfinite(baseline_score):
        baseline_score = best_score
    gate = gate_render_candidate(
        baseline_score=float(baseline_score),
        candidate_score=float(best_score),
        metric="output_mse",
        min_relative_gain=float(min_gain),
    )
    relative_gain = float(gate.relative_gain)
    gate_reason = str(gate.reason)
    if best_label == "identity":
        gate_reason = "identity"
        relative_gain = 0.0
    if best_label != "identity" and not gate.accepted:
        best_scale = candidates[0][2].to(device=device, dtype=torch.float32)
        best_label = "identity"
        best_ratio = None
        best_score = float(baseline_score)
        relative_gain = 0.0

    return AwqSearchResult(
        scale=best_scale.detach(),
        selected_label=best_label,
        selected_ratio=None if best_ratio is None else float(best_ratio),
        baseline_score=float(baseline_score),
        best_score=float(best_score),
        relative_gain=float(relative_gain),
        n_candidates=len(candidates),
        trace=trace,
        gate_reason=gate_reason,
    )
