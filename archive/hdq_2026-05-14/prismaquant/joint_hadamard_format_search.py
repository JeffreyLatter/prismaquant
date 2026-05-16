"""Joint (rotation, format) search for Hadamard-DuQuant++.

Offline search producing the per-cluster cost sidecar the allocator
consumes via ``--hadamard-duquant-cost`` (Phase 5 wiring). For each
fused-sibling cluster × format combination, measures Fisher-weighted
output MSE under the local render stack and — for formats that support
microscale rotation — learns the ``(G, G)`` orthogonal+permutation
rotation that minimizes it.

Per Locked Decision 9, this module **does not** ask the cache-fill to
re-derive a rotation. Each format-specific rotation is solved once, the
composed matrix ``M = R[perm][:, perm]`` is stored to safetensors keyed
by ``{cluster_key}/{format}/composed_matrix``, and the sidecar JSON
references it. Cache-fill (Phase 3) reads the stored matrix verbatim.

Per-format rotation:
  For maximum accuracy the rotation is optimized *per format*. A NVFP4
  cluster's R is tuned for the E2M1 codebook; the MXFP8 candidate gets
  its own R tuned for the E4M3 codebook. Storage overhead is trivial
  (~1 KB per cluster per applicable format).

Output artifacts:
  - **sidecar JSON** (``sidecar_path``): ``{cluster_key: {insertion_kind,
    candidates: {label: {fisher_mse, bpp, rotation_key}}}}``. Allocator
    consumes this via the override flag.
  - **rotation safetensors** (``rotation_safetensors_path``): stored
    composed matrices keyed by ``{cluster_key}/{format}/composed_matrix``.
    Read by cache-fill on install and by the exporter on transforms_config
    emission.
  - **decision log JSONL** (optional, ``decision_log_path``): per-cluster
    ``ClusterDecisionRecord`` entries for inspection and attribution.

The renderer interface is pluggable so the same module can be driven by
both a fast STE surrogate (used in Phase 1/2 tests) and the production
render stack (production_weight_cache renderer; Phase 3 integration).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import save_file as save_safetensors

from prismaquant.hadamard_duquant import (
    HadamardDuQuantSpec,
    MXFP8_GROUP_SIZE,
    NVFP4_GROUP_SIZE,
    ClusterRotationResult,
    ClusterRotationTarget,
    ClusterDecisionRecord,
    apply_block_rotation_input,
    emit_cluster_decision,
    solve_cluster_rotation,
    sylvester_hadamard,
    ste_rtn_mxfp8_per_group,
    ste_rtn_nvfp4_per_group,
)
from prismaquant.render_score import normalize_row_weights, score_render_error


__all__ = [
    "CandidateScore",
    "ClusterInputs",
    "ClusterJointResult",
    "RendererFn",
    "DEFAULT_BPP",
    "DEFAULT_FORMAT_MENU",
    "DEFAULT_FORMATS_WITH_ROTATION",
    "default_renderer",
    "search_cluster",
    "run_joint_search",
]


# ---------------------------------------------------------------------------
# Format menu defaults — match the locked-in brief
# ---------------------------------------------------------------------------

DEFAULT_FORMAT_MENU: tuple[str, ...] = (
    "NVFP4", "MXFP8_E4M3", "FP8_E4M3", "BF16",
)

# Formats where the microscale block aligns with our rotation block size.
# FP8_E4M3 is per-tensor / per-channel with no microscale block; BF16 has
# no quantization. Rotation has no leverage on either.
DEFAULT_FORMATS_WITH_ROTATION: tuple[str, ...] = ("NVFP4", "MXFP8_E4M3")

# Group size per format (None ⇒ no microscale block ⇒ no rotation).
_GROUP_SIZE_BY_FORMAT: dict[str, int | None] = {
    "NVFP4": NVFP4_GROUP_SIZE,
    "MXFP8_E4M3": MXFP8_GROUP_SIZE,
    "FP8_E4M3": None,
    "BF16": None,
}


def _runtime_hadamard_head_dim(
    format_label: str,
    quant_group_size: int,
    input_dim: int,
) -> int | None:
    """Return the vLLM runtime Hadamard block size for an online candidate.

    The production-compatible path uses the same block width as the
    microscale format. For NVFP4 that is 16: the Hadamard transform and the
    NVFP4 scale group align exactly. Local vLLM builds must include the
    native-Hadamard selector patch so this does not route through the dense
    random-matrix Qutlass transform wrapper.
    """
    head_dim = int(quant_group_size)
    if head_dim < 1 or (head_dim & (head_dim - 1)) != 0:
        return None
    if int(input_dim) % head_dim != 0:
        return None
    return head_dim

# Approximate per-format bits-per-parameter including microscale overhead.
# The allocator may refine these via per-Linear bpp accounting; these
# defaults are sufficient for the search-time cost matrix.
DEFAULT_BPP: dict[str, float] = {
    "NVFP4": 4.5,        # 4-bit weight + per-block FP8 scale overhead
    "MXFP8_E4M3": 8.25,  # 8-bit weight + per-block E8M0 scale
    "FP8_E4M3": 8.0,
    "BF16": 16.0,
}


# ---------------------------------------------------------------------------
# Renderer interface
# ---------------------------------------------------------------------------

# Renderer takes (weight, activations, format, group_size) and returns the
# dequantized rendered weight in the input's coordinate system. The
# returned tensor has the same shape and is suitable for direct use in
# output-MSE scoring against the original weight.
RendererFn = Callable[
    [torch.Tensor, torch.Tensor, str, int | None],
    torch.Tensor,
]


# E4M3FN max representable value (matches torch.float8_e4m3fn).
_E4M3_MAX_ABS = 448.0


def default_renderer(
    weight: torch.Tensor,
    activations: torch.Tensor,
    format_label: str,
    group_size: int | None,
) -> torch.Tensor:
    """STE-based renderer for the joint search.

    NVFP4 → per-group RTN to the E2M1 codebook.
    MXFP8_E4M3 → per-group RTN to the E4M3 codebook.
    FP8_E4M3 → per-tensor RTN to the E4M3 codebook.
    BF16 → identity (no quantization).

    Activations are accepted but unused — this renderer is weight-only.
    Production callers should swap in the cache-fill renderer (Phase 3)
    which applies the full local stack (four_over_six → GPTQ → scale_sweep).
    """
    w = weight.float()
    if format_label == "NVFP4":
        return ste_rtn_nvfp4_per_group(
            w, group_size=group_size or NVFP4_GROUP_SIZE
        )
    if format_label == "MXFP8_E4M3":
        return ste_rtn_mxfp8_per_group(
            w, group_size=group_size or MXFP8_GROUP_SIZE
        )
    if format_label == "FP8_E4M3":
        # Per-tensor symmetric FP8 quantization.
        max_abs = w.abs().max().clamp_min(1e-12)
        scale = max_abs / _E4M3_MAX_ABS
        normalized = w / scale
        quant = normalized.to(torch.float8_e4m3fn).to(normalized.dtype)
        return quant * scale
    if format_label == "BF16":
        return w
    raise ValueError(f"unknown format_label {format_label!r}")


def _quantize_activation_for_score(
    activations: torch.Tensor,
    format_label: str,
    group_size: int | None,
) -> torch.Tensor:
    """Activation quantization used by W4A4 candidate scoring.

    This is intentionally local to the shared joint-search scorer: the
    solver and sidecar accounting must evaluate the same runtime convention,
    namely ``Q_a(x M^T) @ Q_w(W M^T)^T`` for rotated microscale formats.
    Formats without activation quantization modeled here fall back to the
    identity so BF16 remains a zero-error reference.
    """
    x = activations.float()
    if format_label == "NVFP4":
        return ste_rtn_nvfp4_per_group(
            x, group_size=group_size or NVFP4_GROUP_SIZE
        )
    if format_label == "MXFP8_E4M3":
        return ste_rtn_mxfp8_per_group(
            x, group_size=group_size or MXFP8_GROUP_SIZE
        )
    if format_label == "FP8_E4M3":
        max_abs = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = max_abs / _E4M3_MAX_ABS
        normalized = x / scale
        quant = normalized.to(torch.float8_e4m3fn).to(normalized.dtype)
        return (quant * scale).to(dtype=x.dtype)
    return x


def _score_w4a4_render_error(
    reference_weight: torch.Tensor,
    rendered_storage_weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    format_label: str,
    group_size: int | None,
    row_weights: torch.Tensor | None,
    row_chunk: int,
    reference_activations: torch.Tensor | None = None,
) -> float:
    """Output MSE for the artifact/runtime W4A4 path.

    ``rendered_storage_weight`` is already in the coordinate system that the
    runtime matmul sees. For rotated candidates that means ``Q_w(W M^T)``;
    for no-rotation candidates it is simply ``Q_w(W)``. ``activations`` must
    be in the matching runtime coordinate system. ``reference_activations``
    remains in the original model coordinate system; when omitted it defaults
    to ``activations`` for the no-rotation case.
    """
    w_ref = reference_weight.detach().to(dtype=torch.float32)
    w_rt = rendered_storage_weight.detach().to(dtype=torch.float32)
    x_rt = activations.detach().to(dtype=torch.float32).reshape(-1, w_ref.shape[1])
    if reference_activations is None:
        x_ref = x_rt
    else:
        x_ref = reference_activations.detach().to(dtype=torch.float32).reshape(
            -1, w_ref.shape[1]
        )
    if int(x_ref.shape[0]) != int(x_rt.shape[0]):
        raise ValueError(
            "reference_activations and activations must have the same row count"
        )
    if w_rt.shape != w_ref.shape:
        raise ValueError(
            f"rendered_storage_weight shape {tuple(w_rt.shape)} does not "
            f"match reference_weight shape {tuple(w_ref.shape)}"
        )
    err_sum = torch.zeros((), device=x_rt.device, dtype=torch.float32)
    n_rows = 0
    rw_all = normalize_row_weights(row_weights, x_rt.shape[0], device=x_rt.device)
    for s in range(0, x_rt.shape[0], int(row_chunk)):
        x_chunk = x_rt[s:s + int(row_chunk)]
        x_ref_chunk = x_ref[s:s + int(row_chunk)]
        x_q = _quantize_activation_for_score(
            x_chunk, format_label, group_size
        )
        y_ref = x_ref_chunk @ w_ref.t()
        y_rt = x_q @ w_rt.t()
        diff = y_ref - y_rt
        if rw_all is not None:
            rw = rw_all[s:s + diff.shape[0]].to(device=diff.device, dtype=diff.dtype)
            err = (diff.pow(2) * rw.unsqueeze(1)).sum()
        else:
            err = diff.pow(2).sum()
        err_sum = err_sum + err
        n_rows += int(diff.shape[0])
    denom = max(1, n_rows * int(w_ref.shape[0]))
    return float((err_sum / denom).item())


# ---------------------------------------------------------------------------
# Cluster-level data structures
# ---------------------------------------------------------------------------


@dataclass
class ClusterInputs:
    """Calibration data for one fused-sibling cluster's joint search.

    Attributes:
        cluster_key: must match the ``cluster_key`` of a
            :class:`HadamardDuQuantSpec` from
            :mod:`prismaquant.hadamard_duquant`.
        targets: one ``ClusterRotationTarget`` per Linear in the cluster.
            All targets must share the same input width.
    """

    cluster_key: str
    targets: list[ClusterRotationTarget]


@dataclass(frozen=True)
class CandidateScore:
    """Cost-table entry for one (rotation, format) candidate."""

    label: str                       # e.g., "rot+NVFP4" or "no_rot+BF16"
    fisher_mse: float                # Fisher-weighted output MSE under render
    bpp: float                       # bits-per-parameter for this candidate
    rotation_key: str | None = None  # safetensors key, None if no rotation
    runtime_transform_type: str | None = None
    runtime_head_dim: int | None = None
    train_fisher_mse: float | None = None
    validation_fisher_mse: float | None = None
    rotation_accepted: bool | None = None
    rejection_reason: str | None = None
    train_relative_gain: float | None = None
    validation_relative_gain: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "fisher_mse": float(self.fisher_mse),
            "bpp": float(self.bpp),
        }
        if self.rotation_key is not None:
            d["rotation_key"] = str(self.rotation_key)
        if self.runtime_transform_type is not None:
            d["runtime_transform_type"] = str(self.runtime_transform_type)
        if self.runtime_head_dim is not None:
            d["runtime_head_dim"] = int(self.runtime_head_dim)
        if self.train_fisher_mse is not None:
            d["train_fisher_mse"] = float(self.train_fisher_mse)
        if self.validation_fisher_mse is not None:
            d["validation_fisher_mse"] = float(self.validation_fisher_mse)
        if self.rotation_accepted is not None:
            d["rotation_accepted"] = bool(self.rotation_accepted)
        if self.rejection_reason is not None:
            d["rejection_reason"] = str(self.rejection_reason)
        if self.train_relative_gain is not None:
            d["train_relative_gain"] = float(self.train_relative_gain)
        if self.validation_relative_gain is not None:
            d["validation_relative_gain"] = float(self.validation_relative_gain)
        return d


@dataclass
class ClusterJointResult:
    """Joint-search output for one cluster."""

    cluster_key: str
    insertion_kind: str
    candidates: dict[str, CandidateScore]
    # Per-format rotation results (keyed by format label). Only formats that
    # support and accepted rotation appear here. Each entry has its own
    # composed_matrix tuned for that format's quantization grid.
    rotations: dict[str, ClusterRotationResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-cluster search
# ---------------------------------------------------------------------------


def _score_candidate(
    targets: Sequence[ClusterRotationTarget],
    format_label: str,
    group_size: int | None,
    renderer: RendererFn,
    *,
    composed_matrix: torch.Tensor | None,
    score_loss: Literal["w_only", "w4a4"],
    row_chunk: int,
) -> float:
    """Score one (rotation, format) candidate over all cluster siblings.

    If ``composed_matrix`` is ``None``: score format quant on the raw weights.
    Otherwise: rotate weights via ``W @ M^T``, render, and map back via
    ``rendered @ M`` (the same effective-weight construction the runtime
    composition produces under the input-side rotation).

    ``score_loss="w4a4"`` scores the actual runtime activation path:
    ``Q_a(x M^T) @ Q_w(W M^T)^T``. ``score_loss="w_only"`` preserves the
    legacy effective-weight score but still renders rotated weights against
    rotated activations so activation-aware renderers see the right basis.

    Returns sum of per-target (Fisher-weighted) output MSE values, weighted
    by ``target.score_weight``.
    """
    total = 0.0
    for t in targets:
        w = t.weight.detach().to(dtype=torch.float32)
        x = t.activations.detach().to(dtype=torch.float32).reshape(-1, w.shape[1])
        if composed_matrix is not None:
            M = composed_matrix.to(device=w.device, dtype=torch.float32)
            w_for_quant = apply_block_rotation_input(w, M.t())
            x_for_quant = apply_block_rotation_input(x, M.t())
            w_q = renderer(w_for_quant, x_for_quant, format_label, group_size)
            if score_loss == "w4a4":
                score = _score_w4a4_render_error(
                    w,
                    w_q,
                    x_for_quant,
                    format_label=format_label,
                    group_size=group_size,
                    row_weights=t.row_weights,
                    row_chunk=row_chunk,
                    reference_activations=x,
                )
            else:
                w_eff = apply_block_rotation_input(w_q, M)
                score = score_render_error(
                    w, w_eff, x,
                    row_weights=t.row_weights,
                    row_chunk=row_chunk,
                )
        else:
            w_q = renderer(w, x, format_label, group_size)
            if score_loss == "w4a4":
                score = _score_w4a4_render_error(
                    w,
                    w_q,
                    x,
                    format_label=format_label,
                    group_size=group_size,
                    row_weights=t.row_weights,
                    row_chunk=row_chunk,
                )
            else:
                score = score_render_error(
                    w, w_q, x,
                    row_weights=t.row_weights,
                    row_chunk=row_chunk,
                )
        total = total + float(t.score_weight) * float(score)
    return float(total)


def _relative_gain(baseline: float, candidate: float) -> float:
    if not math.isfinite(float(baseline)) or not math.isfinite(float(candidate)):
        return float("-inf")
    return (
        (float(baseline) - float(candidate))
        / max(abs(float(baseline)), 1e-12)
    )


def _rotation_gate_status(
    *,
    train_baseline: float,
    train_rotated: float,
    validation_baseline: float | None,
    validation_rotated: float | None,
    min_train_relative_gain: float,
    min_validation_relative_gain: float,
) -> tuple[bool, str | None, float, float | None]:
    """Return whether a solved rotation clears the train/validation gate."""
    train_gain = _relative_gain(train_baseline, train_rotated)
    validation_gain = (
        None if validation_baseline is None or validation_rotated is None
        else _relative_gain(validation_baseline, validation_rotated)
    )
    if not math.isfinite(float(train_rotated)):
        return False, "nonfinite_train_score", train_gain, validation_gain
    if train_gain < float(min_train_relative_gain):
        return False, "train_gain_below_margin", train_gain, validation_gain
    if validation_rotated is not None and not math.isfinite(float(validation_rotated)):
        return False, "nonfinite_validation_score", train_gain, validation_gain
    if (
        validation_gain is not None
        and validation_gain < float(min_validation_relative_gain)
    ):
        return False, "validation_gain_below_margin", train_gain, validation_gain
    return True, None, train_gain, validation_gain


def search_cluster(
    spec: HadamardDuQuantSpec,
    inputs: ClusterInputs,
    *,
    validation_inputs: ClusterInputs | None = None,
    format_menu: Sequence[str] = DEFAULT_FORMAT_MENU,
    formats_with_rotation: Sequence[str] = DEFAULT_FORMATS_WITH_ROTATION,
    renderer: RendererFn = default_renderer,
    bpp_per_format: dict[str, float] | None = None,
    solver_n_iters: int = 60,
    solver_lr: float = 5e-3,
    solver_init: str = "identity",
    solver_loss: Literal["w_only", "w4a4"] = "w_only",
    score_loss: Literal["w_only", "w4a4"] | None = None,
    solver_weight_decay: float = 0.0,
    solver_multi_init: tuple[str, ...] = (),
    solver_n_random_probes: int = 0,
    solver_early_stop_patience: int | None = 100,
    online_rotation_mode: Literal["hadamard", "learned"] = "hadamard",
    rotation_min_train_gain: float = float("-inf"),
    rotation_min_validation_gain: float = float("-inf"),
    row_chunk: int = 256,
) -> ClusterJointResult:
    """Run the joint (rotation, format) search for one fused-sibling cluster.

    For each ``fmt`` in ``format_menu``:
      - Score the no-rotation candidate.
      - If ``fmt`` is in ``formats_with_rotation`` *and* the format's
        microscale block size matches ``spec.group_size``, solve a
        format-specific rotation and score the rotated candidate.

    The format-specific solve is what makes the result the per-format
    accuracy ceiling: each rotation is tuned for the quantization grid it
    will ship under.

    Args:
        spec: insertion-point specification from
            :func:`prismaquant.hadamard_duquant.default_insertion_specs`.
        inputs: calibration data for this cluster.
        format_menu: list of formats to evaluate. Default covers the
            production menu: ``{NVFP4, MXFP8_E4M3, FP8_E4M3, BF16}``.
        formats_with_rotation: subset of ``format_menu`` to attempt rotation
            on. Defaults to the two microscale formats.
        renderer: scoring renderer. Default is the STE surrogate; Phase 3
            swaps in the production render stack.
        bpp_per_format: bits-per-parameter table. Defaults to :data:`DEFAULT_BPP`.
        solver_n_iters / solver_lr / solver_init: rotation solver settings.
        row_chunk: row-batch size for the scoring loop.

    Returns:
        :class:`ClusterJointResult` with one ``CandidateScore`` per
        ``(rotation_choice, format)`` actually evaluated, plus per-format
        ``ClusterRotationResult`` entries for the rotations that were learned.
    """
    bpp = dict(DEFAULT_BPP)
    if bpp_per_format:
        bpp.update(bpp_per_format)

    targets = inputs.targets
    if not targets:
        raise ValueError(
            f"cluster {inputs.cluster_key!r} has no targets to score"
        )
    validation_targets = (
        validation_inputs.targets
        if validation_inputs is not None and validation_inputs.targets
        else None
    )

    candidates: dict[str, CandidateScore] = {}
    rotations: dict[str, ClusterRotationResult] = {}
    score_loss_resolved: Literal["w_only", "w4a4"] = (
        solver_loss if score_loss is None else score_loss
    )

    for fmt in format_menu:
        gs = _GROUP_SIZE_BY_FORMAT.get(fmt)
        fmt_bpp = float(bpp.get(fmt, 0.0))

        # No-rotation candidate.
        try:
            train_mse = _score_candidate(
                targets, fmt, gs, renderer,
                composed_matrix=None,
                score_loss=score_loss_resolved,
                row_chunk=row_chunk,
            )
        except (ValueError, RuntimeError):
            train_mse = float("inf")
        validation_mse: float | None = None
        if validation_targets is not None:
            try:
                validation_mse = _score_candidate(
                    validation_targets, fmt, gs, renderer,
                    composed_matrix=None,
                    score_loss=score_loss_resolved,
                    row_chunk=row_chunk,
                )
            except (ValueError, RuntimeError):
                validation_mse = float("inf")
        sidecar_mse = train_mse if validation_mse is None else validation_mse
        candidates[f"no_rot+{fmt}"] = CandidateScore(
            label=f"no_rot+{fmt}",
            fisher_mse=float(sidecar_mse),
            bpp=fmt_bpp,
            train_fisher_mse=(
                float(train_mse) if validation_mse is not None else None
            ),
            validation_fisher_mse=(
                float(validation_mse) if validation_mse is not None else None
            ),
        )

        # With-rotation candidate: only if format's group_size matches the
        # cluster's spec.group_size. Different group_size means the cluster
        # was sized for a different microscale format.
        rot_eligible = (
            fmt in formats_with_rotation
            and gs is not None
            and int(spec.group_size) == int(gs)
        )
        if not rot_eligible:
            continue

        runtime_transform_type: str | None = None
        if spec.online and online_rotation_mode == "hadamard":
            runtime_head_dim = _runtime_hadamard_head_dim(
                fmt,
                int(gs),
                int(targets[0].weight.shape[1]),
            )
            if runtime_head_dim is None:
                candidates[f"rot+{fmt}"] = CandidateScore(
                    label=f"rot+{fmt}",
                    fisher_mse=float("inf"),
                    bpp=fmt_bpp,
                    runtime_transform_type="hadamard",
                )
                continue
            H = sylvester_hadamard(
                runtime_head_dim,
                device=targets[0].weight.device,
                dtype=torch.float32,
            )
            train_rot_mse = _score_candidate(
                targets, fmt, gs, renderer,
                composed_matrix=H,
                score_loss=score_loss_resolved,
                row_chunk=row_chunk,
            )
            validation_rot_mse: float | None = None
            if validation_targets is not None:
                validation_rot_mse = _score_candidate(
                    validation_targets, fmt, gs, renderer,
                    composed_matrix=H,
                    score_loss=score_loss_resolved,
                    row_chunk=row_chunk,
                )
            baseline_train_mse = float(train_mse)
            baseline_validation_mse = (
                None if validation_mse is None else float(validation_mse)
            )
            accepted, rejection_reason, train_gain, validation_gain = (
                _rotation_gate_status(
                    train_baseline=baseline_train_mse,
                    train_rotated=float(train_rot_mse),
                    validation_baseline=baseline_validation_mse,
                    validation_rotated=validation_rot_mse,
                    min_train_relative_gain=rotation_min_train_gain,
                    min_validation_relative_gain=rotation_min_validation_gain,
                )
            )
            rr = ClusterRotationResult(
                R=H.detach().clone(),
                permutation=torch.arange(runtime_head_dim, device=H.device),
                composed_matrix=H.detach().clone(),
                baseline_score=baseline_train_mse,
                rotated_score=float(train_rot_mse),
                relative_gain=float(train_gain),
                solver_seconds=0.0,
                orthogonality_err=float(
                    (H @ H.t() - torch.eye(
                        runtime_head_dim, device=H.device, dtype=H.dtype,
                    )).norm().item()
                ),
                init_strategy="hadamard",
                n_iters=0,
                best_iter=0,
                final_iter_loss=float(train_rot_mse),
                still_improving=False,
                early_stopped=False,
            )
            runtime_transform_type = "hadamard"
            sidecar_rot_mse = (
                float(train_rot_mse)
                if validation_rot_mse is None else float(validation_rot_mse)
            )
            candidates[f"rot+{fmt}"] = CandidateScore(
                label=f"rot+{fmt}",
                fisher_mse=float(sidecar_rot_mse) if accepted else float("inf"),
                bpp=fmt_bpp,
                rotation_key=_rotation_key(spec.cluster_key, fmt) if accepted else None,
                runtime_transform_type=runtime_transform_type,
                runtime_head_dim=runtime_head_dim,
                train_fisher_mse=(
                    float(train_rot_mse) if validation_rot_mse is not None else None
                ),
                validation_fisher_mse=(
                    float(validation_rot_mse)
                    if validation_rot_mse is not None else None
                ),
                rotation_accepted=accepted,
                rejection_reason=rejection_reason,
                train_relative_gain=float(train_gain),
                validation_relative_gain=(
                    None if validation_gain is None else float(validation_gain)
                ),
            )
            if accepted:
                rotations[fmt] = rr
            continue

        runtime_transform_type = (
            "random-matrix" if spec.online else None
        )

        # Adaptive basin search: build a list of (init_strategy, seed)
        # candidates and run the solver from each, then commit argmin over
        # final W4A4 scores. Three sources of candidates:
        #
        # 1. solver_init (always): the named init strategy. Identity by
        #    default — produces the same single-basin result the legacy
        #    solver would, so disabling all probes preserves prior behavior.
        # 2. solver_multi_init: explicit extra named inits (sylvester,
        #    svd_v, etc.). Useful for testing landscape structure.
        # 3. solver_n_random_probes: N random-orthogonal inits with seeds
        #    0..N-1. The per-cluster random-orthogonal diagnostic showed
        #    ~17% of clusters have multimodal W4A4 landscape where random
        #    inits find basins identity-Adam can't reach; the probe
        #    catches those without thresholds — the calibration loss is
        #    shared across all candidates so argmin is the unambiguous
        #    basin selector. Ties broken by lower orthogonality_err.
        candidate_specs: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for init_name in [solver_init, *solver_multi_init]:
            key = (str(init_name), 0)
            if key in seen:
                continue
            seen.add(key)
            candidate_specs.append(key)
        for s in range(int(solver_n_random_probes)):
            key = ("random", int(s))
            if key in seen:
                continue
            seen.add(key)
            candidate_specs.append(key)

        rr_candidates: list[ClusterRotationResult] = []
        last_exc: BaseException | None = None
        for (init_strat, init_seed) in candidate_specs:
            try:
                rr_i = solve_cluster_rotation(
                    targets,
                    group_size=int(gs),
                    format_label=fmt,
                    init_strategy=init_strat,
                    seed=int(init_seed),
                    loss_kind=solver_loss,
                    n_iters=solver_n_iters,
                    lr=solver_lr,
                    weight_decay=float(solver_weight_decay),
                    row_chunk=row_chunk,
                    early_stop_patience=solver_early_stop_patience,
                )
                rr_candidates.append(rr_i)
            except (ValueError, RuntimeError) as exc:
                last_exc = exc
                continue
        if not rr_candidates:
            _solver_exc = last_exc if last_exc is not None else RuntimeError(
                "all multi-init candidates failed"
            )
        else:
            # Pick winner by the SAME production-renderer score the sidecar
            # will report (not by in-solver STE-loss). The two can disagree:
            # a random-seed candidate that wins on STE-loss may lose on the
            # production renderer because the STE quantizer is a piecewise-
            # constant proxy, while the renderer applies the exact NVFP4
            # scale-rounding rules. Selecting on the renderer score is the
            # only way to keep candidate-selection and sidecar-reporting
            # consistent — without it, the adaptive probe can regress
            # clusters where identity's STE-loss is worse but its
            # production-renderer score is better. Score each candidate
            # once; argmin gives the explicit per-cluster basin choice.
            scored: list[tuple[float, ClusterRotationResult]] = []
            for rr_i in rr_candidates:
                try:
                    train_rmse = _score_candidate(
                        targets, fmt, gs, renderer,
                        composed_matrix=rr_i.composed_matrix,
                        score_loss=score_loss_resolved,
                        row_chunk=row_chunk,
                    )
                except (ValueError, RuntimeError):
                    train_rmse = float("inf")
                validation_rmse: float | None = None
                if validation_targets is not None:
                    try:
                        validation_rmse = _score_candidate(
                            validation_targets, fmt, gs, renderer,
                            composed_matrix=rr_i.composed_matrix,
                            score_loss=score_loss_resolved,
                            row_chunk=row_chunk,
                        )
                    except (ValueError, RuntimeError):
                        validation_rmse = float("inf")
                selector_score = (
                    float(train_rmse)
                    if validation_rmse is None else float(validation_rmse)
                )
                scored.append((selector_score, rr_i))
            # Ties broken by lower orthogonality_err
            rmse_best, rr = min(scored, key=lambda t: (t[0], t[1].orthogonality_err))
            try:
                _train_score_cached = _score_candidate(
                    targets, fmt, gs, renderer,
                    composed_matrix=rr.composed_matrix,
                    score_loss=score_loss_resolved,
                    row_chunk=row_chunk,
                )
            except (ValueError, RuntimeError):
                _train_score_cached = float("inf")
            _validation_score_cached: float | None = None
            if validation_targets is not None:
                try:
                    _validation_score_cached = _score_candidate(
                        validation_targets, fmt, gs, renderer,
                        composed_matrix=rr.composed_matrix,
                        score_loss=score_loss_resolved,
                        row_chunk=row_chunk,
                    )
                except (ValueError, RuntimeError):
                    _validation_score_cached = float("inf")
            _renderer_score_cached: float = (
                float(_train_score_cached)
                if _validation_score_cached is None
                else float(_validation_score_cached)
            )
            _solver_exc = None

        if _solver_exc is not None:
            # Surface the failure so silent rotation drop-outs are
            # diagnosable. The stderr line includes cluster key + format
            # so the operator can correlate with sidecar inf-cost entries.
            import sys as _sys
            print(
                f"[joint-search] solve_cluster_rotation FAILED for "
                f"cluster={spec.cluster_key!r} fmt={fmt!r}: "
                f"{type(_solver_exc).__name__}: {_solver_exc}",
                file=_sys.stderr, flush=True,
            )
            candidates[f"rot+{fmt}"] = CandidateScore(
                label=f"rot+{fmt}",
                fisher_mse=float("inf"),
                bpp=fmt_bpp,
            )
            continue

        # The winner's production-renderer score was computed above as part
        # of candidate selection; reuse it rather than re-scoring.
        baseline_train_mse = float(train_mse)
        baseline_validation_mse = (
            None if validation_mse is None else float(validation_mse)
        )
        accepted, rejection_reason, train_gain, validation_gain = (
            _rotation_gate_status(
                train_baseline=baseline_train_mse,
                train_rotated=float(_train_score_cached),
                validation_baseline=baseline_validation_mse,
                validation_rotated=_validation_score_cached,
                min_train_relative_gain=rotation_min_train_gain,
                min_validation_relative_gain=rotation_min_validation_gain,
            )
        )
        candidates[f"rot+{fmt}"] = CandidateScore(
            label=f"rot+{fmt}",
            fisher_mse=(
                float(_renderer_score_cached) if accepted else float("inf")
            ),
            bpp=fmt_bpp,
            rotation_key=_rotation_key(spec.cluster_key, fmt) if accepted else None,
            runtime_transform_type=runtime_transform_type,
            train_fisher_mse=(
                float(_train_score_cached)
                if _validation_score_cached is not None else None
            ),
            validation_fisher_mse=(
                float(_validation_score_cached)
                if _validation_score_cached is not None else None
            ),
            rotation_accepted=accepted,
            rejection_reason=rejection_reason,
            train_relative_gain=float(train_gain),
            validation_relative_gain=(
                None if validation_gain is None else float(validation_gain)
            ),
        )
        if accepted:
            rotations[fmt] = rr

    return ClusterJointResult(
        cluster_key=inputs.cluster_key,
        insertion_kind=spec.kind.value,
        candidates=candidates,
        rotations=rotations,
    )


def _rotation_key(cluster_key: str, format_label: str) -> str:
    """Canonical safetensors key for one cluster's per-format rotation."""
    return f"{cluster_key}/{format_label}/composed_matrix"


# ---------------------------------------------------------------------------
# Whole-model driver
# ---------------------------------------------------------------------------


def run_joint_search(
    specs: Sequence[HadamardDuQuantSpec],
    cluster_inputs: dict[str, ClusterInputs],
    *,
    validation_cluster_inputs: dict[str, ClusterInputs] | None = None,
    sidecar_path: Path | str,
    rotation_safetensors_path: Path | str,
    decision_log_path: Path | str | None = None,
    format_menu: Sequence[str] = DEFAULT_FORMAT_MENU,
    formats_with_rotation: Sequence[str] = DEFAULT_FORMATS_WITH_ROTATION,
    renderer: RendererFn = default_renderer,
    bpp_per_format: dict[str, float] | None = None,
    solver_n_iters: int = 60,
    solver_lr: float = 5e-3,
    solver_init: str = "identity",
    solver_loss: Literal["w_only", "w4a4"] = "w_only",
    score_loss: Literal["w_only", "w4a4"] | None = None,
    solver_weight_decay: float = 0.0,
    solver_multi_init: tuple[str, ...] = (),
    solver_n_random_probes: int = 0,
    solver_early_stop_patience: int | None = 100,
    online_rotation_mode: Literal["hadamard", "learned"] = "hadamard",
    rotation_min_train_gain: float = float("-inf"),
    rotation_min_validation_gain: float = float("-inf"),
    row_chunk: int = 256,
) -> dict[str, ClusterJointResult]:
    """Run the joint search across all specs and write the three artifacts.

    Specs with no entry in ``cluster_inputs`` are skipped (they may exist in
    architectures where the cluster wasn't captured during calibration).

    Args:
        specs: insertion-point specifications for the whole model.
        cluster_inputs: per-cluster calibration data, keyed by
            ``spec.cluster_key``.
        sidecar_path: where to write the cost-table JSON for the allocator.
        rotation_safetensors_path: where to write the stored rotation
            matrices. Keys are ``{cluster_key}/{format}/composed_matrix``.
        decision_log_path: optional JSONL emission target for per-cluster
            decision records. The file is truncated on entry.
        format_menu / formats_with_rotation / renderer / bpp_per_format /
        solver_n_iters / solver_lr / solver_init / row_chunk: forwarded to
            :func:`search_cluster` per cluster.

    Returns:
        Mapping ``{cluster_key: ClusterJointResult}`` for every cluster that
        was searched.
    """
    sidecar_path = Path(sidecar_path)
    rotation_safetensors_path = Path(rotation_safetensors_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    rotation_safetensors_path.parent.mkdir(parents=True, exist_ok=True)

    decision_log_path_obj: Path | None = None
    if decision_log_path is not None:
        decision_log_path_obj = Path(decision_log_path)
        decision_log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        # Truncate prior log so re-runs aren't appended.
        decision_log_path_obj.write_text("")

    results: dict[str, ClusterJointResult] = {}
    rotation_tensors: dict[str, torch.Tensor] = {}

    for spec in specs:
        inputs = cluster_inputs.get(spec.cluster_key)
        if inputs is None:
            continue
        result = search_cluster(
            spec, inputs,
            validation_inputs=(
                (validation_cluster_inputs or {}).get(spec.cluster_key)
            ),
            format_menu=format_menu,
            formats_with_rotation=formats_with_rotation,
            renderer=renderer,
            bpp_per_format=bpp_per_format,
            solver_n_iters=solver_n_iters,
            solver_lr=solver_lr,
            solver_init=solver_init,
            solver_loss=solver_loss,
            score_loss=score_loss,
            solver_weight_decay=solver_weight_decay,
            solver_multi_init=solver_multi_init,
            solver_n_random_probes=solver_n_random_probes,
            solver_early_stop_patience=solver_early_stop_patience,
            online_rotation_mode=online_rotation_mode,
            rotation_min_train_gain=rotation_min_train_gain,
            rotation_min_validation_gain=rotation_min_validation_gain,
            row_chunk=row_chunk,
        )
        results[spec.cluster_key] = result

        for fmt, rr in result.rotations.items():
            key = _rotation_key(spec.cluster_key, fmt)
            rotation_tensors[key] = rr.composed_matrix.detach().cpu().contiguous()

        if decision_log_path_obj is not None:
            _emit_decision_log_for_cluster(
                spec, result, decision_log_path_obj
            )

    specs_by_cluster = {s.cluster_key: s for s in specs}
    _write_sidecar(results, sidecar_path, specs_by_cluster=specs_by_cluster)
    if rotation_tensors:
        save_safetensors(rotation_tensors, str(rotation_safetensors_path))

    return results


def _write_sidecar(
    results: dict[str, ClusterJointResult],
    sidecar_path: Path,
    *,
    specs_by_cluster: dict[str, HadamardDuQuantSpec] | None = None,
) -> None:
    """Serialize the cost-table JSON consumed by the allocator override.

    Per cluster, emits:
      - ``insertion_kind`` (residual / v_o / attn_out / down_proj)
      - ``group_size``, ``input_dim``, ``online``
      - ``consumer_qnames`` and ``producer_qnames`` (so the allocator can
        map per-cluster rotation picks to per-Linear cost overrides
        without needing a separate specs file)
      - ``candidates``: ``{label: {fisher_mse, bpp, rotation_key?}}``
    """
    specs_by_cluster = specs_by_cluster or {}
    cluster_entries: dict[str, dict[str, Any]] = {}
    for cluster_key, result in sorted(results.items()):
        spec = specs_by_cluster.get(cluster_key)
        entry: dict[str, Any] = {
            "insertion_kind": result.insertion_kind,
            "candidates": {
                label: cs.to_dict()
                for label, cs in sorted(result.candidates.items())
            },
        }
        if spec is not None:
            entry["group_size"] = int(spec.group_size)
            entry["input_dim"] = int(spec.input_dim)
            entry["online"] = bool(spec.online)
            entry["consumer_qnames"] = list(spec.consumer_qnames)
            entry["producer_qnames"] = list(spec.producer_qnames)
        cluster_entries[cluster_key] = entry
    payload: dict[str, Any] = {
        "version": "1",
        "clusters": cluster_entries,
    }
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _emit_decision_log_for_cluster(
    spec: HadamardDuQuantSpec,
    result: ClusterJointResult,
    log_path: Path,
) -> None:
    """Append one ClusterDecisionRecord per cluster to the JSONL log.

    The ``allocator_pick`` field is empty at this stage — it's filled in
    by the allocator phase. ``render_gates`` is also empty here; the
    render-gate decisions are emitted during cache-fill, not joint search.
    """
    # Summarize rotation state. If any format learned a rotation, capture
    # its log dict (the per-format result is small and useful for attribution).
    if result.rotations:
        rotation_summary: dict[str, Any] = {
            "applied": True,
            "G": spec.group_size,
            "per_format": {
                fmt: rr.to_log_dict() for fmt, rr in sorted(result.rotations.items())
            },
        }
    else:
        rotation_summary = {"applied": False, "G": spec.group_size}

    candidates_for_log = {
        label: {"fisher_mse": cs.fisher_mse, "bpp": cs.bpp}
        for label, cs in result.candidates.items()
    }
    record = ClusterDecisionRecord(
        cluster_key=spec.cluster_key,
        insertion_kind=spec.kind.value,
        rotation=rotation_summary,
        candidates=candidates_for_log,
        allocator_pick="",  # filled by allocator phase
        render_gates=[],
    )
    emit_cluster_decision(record, log_path, append=True)
