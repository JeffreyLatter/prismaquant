"""Hadamard-DuQuant++ — learned-orthogonal block-diagonal rotation with
calibrated zigzag permutation for microscale quantization formats.

Ports DuQuant-v2's offline pipeline (insertion-point identification,
zigzag permutation, double-rotation-with-permutation, LET fold) into
PrismaQuant. The Givens rotation primitive of the original DuQuant++
is replaced by a learned-orthogonal dense ``(G, G)`` matrix for folded
sites and a vLLM-native Sylvester Hadamard transform for production online
sites.

Pipeline summary per fused-sibling cluster:
  1. Calibrate within-block zigzag permutation ``P`` from per-channel
     activation magnitudes (snake order — adjacent positions have very
     different average magnitudes, which mixes well under rotation).
  2. Solve a ``(G, G)`` orthogonal rotation ``R`` via Cayley-parameterized
     Adam, minimizing Fisher-weighted output MSE under format-specific
     STE quantization summed across all sibling Linears.
  3. Compose into a single matrix ``M[i, j] = R[perm[i], perm[j]]``.
     Folded sites absorb ``M`` into adjacent weights. Production online
     sites use a fixed Sylvester Hadamard matrix and emit a compressed-
     tensors ``type="hadamard"`` input transform; dense ``random-matrix``
     online transforms are kept research-only until vLLM supports them on
     NVFP4.

The four DuQuant++ insertion points:
  - ``RESIDUAL``: residual-stream rotation. Offline-fold via WEIGHT_OUTPUT
    on producers (previous-layer's output projections + embedding) and
    WEIGHT_INPUT on consumers (q/k/v_proj, gate_proj/up_proj).
  - ``V_O``: V → O rotation. Offline-fold via V_proj's WEIGHT_OUTPUT and
    o_proj's WEIGHT_INPUT (independent of the residual rotation).
  - ``ATTN_OUT``: attn out_proj input rotation. Distinct from V_O when
    the model fuses V_proj and o_proj differently; otherwise subsumed.
  - ``DOWN_PROJ``: down_proj input rotation. **Online** — SwiGLU's
    elementwise multiplication breaks offline fold algebra
    (``(a ⊙ b) @ R ≠ (a @ R) ⊙ (b @ R)``). Uses INPUT location at
    runtime; ``apply_block_rotation_input`` runs as a forward_pre_hook
    before ``Linear.forward``.

Render-mechanism registration lives in ``prismaquant/render_score.py``
at phase 20 with ``exclusive_group="activation_weight_fold"``, mutually
exclusive with the archived ``awq``/``smoothquant``/``block_rotation``
levers.

Format support:
  - NVFP4 (``G=16``): STE-RTN simulator with per-block scale
    ``max(|w|) / 6`` against the E2M1 codebook.
  - MXFP8_E4M3 (``G=32``): STE-RTN simulator with per-block scale
    ``max(|w|) / 448`` against the ``torch.float8_e4m3fn`` codebook.
  - FP8_E4M3 (per-channel) and BF16: no rotation; the allocator picks
    these for clusters where rotation can't tame outliers.

Note on the compressed-tensors algebra used by the runtime: online
transforms apply ``x @ M^T`` while the production cache stores
``Q(W @ M^T)`` in the artifact. The solver/search scores the same effective
algebra ``Q(W @ M^T) @ M`` as the archived BlockOrtho-G. For NVFP4 online
Hadamard, the runtime Hadamard block can be wider than the 16-wide NVFP4
quantization group to stay on vLLM's compatible path.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import time
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from prismaquant.render_score import score_render_error


__all__ = [
    # Constants
    "NVFP4_GROUP_SIZE",
    "MXFP8_GROUP_SIZE",
    "SUPPORTED_FORMATS",
    # Algebra primitives
    "sylvester_hadamard",
    "random_orthogonal",
    "cayley_orthogonal",
    "apply_block_rotation_input",
    "rotate_weight_for_storage",
    "effective_weight_for_scoring",
    # STE quantizers
    "ste_rtn_nvfp4_per_group",
    "ste_rtn_mxfp8_per_group",
    # Permutation + composition
    "calibrate_zigzag_permutation",
    "compose_rotation_with_permutation",
    # Insertion-point identification
    "InsertionPointKind",
    "HadamardDuQuantSpec",
    "insertion_specs_for_layer",
    "default_insertion_specs",
    # Solver
    "ClusterRotationTarget",
    "ClusterRotationResult",
    "solve_cluster_rotation",
    # Log emission
    "ClusterDecisionRecord",
    "ShipSummaryRecord",
    "emit_cluster_decision",
    "emit_ship_summary",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NVFP4_GROUP_SIZE = 16
MXFP8_GROUP_SIZE = 32

# Formats for which rotation is candidate. FP8_E4M3 (per-channel) and BF16
# do not have a microscale block to align rotation to; the allocator picks
# them for clusters where rotation has no leverage.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"NVFP4", "MXFP8_E4M3"})


# ---------------------------------------------------------------------------
# Algebra primitives (adapted from archive/foldscale_orthog_2026-05-13)
# ---------------------------------------------------------------------------


def sylvester_hadamard(
    group_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized Sylvester Hadamard at ``(group_size, group_size)``.

    ``group_size`` must be a power of two. The returned matrix satisfies
    ``H @ H.T == I`` and ``H @ H == I`` (it is its own inverse — useful for
    compressed-tensors' INPUT+WEIGHT_INPUT pairing when the inverse flag
    is not used).
    """
    g = int(group_size)
    if g < 1 or (g & (g - 1)) != 0:
        raise ValueError(
            f"Sylvester Hadamard requires group_size to be a power of two, got {g}"
        )
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < g:
        H = torch.cat(
            [
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )
    return H / math.sqrt(g)


def random_orthogonal(
    group_size: int,
    *,
    generator: torch.Generator,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Random ``(group_size, group_size)`` orthogonal via QR of a Gaussian.

    The QR runs in the requested ``dtype`` (the archived BlockOrtho-G version
    fixed QR at float32 and downcast; we honor the caller's precision here so
    float64 callers get full float64 precision).
    """
    g = int(group_size)
    if g < 1:
        raise ValueError(f"group_size must be >= 1, got {g}")
    work_device = device if device is not None else torch.device("cpu")
    a = torch.randn(g, g, generator=generator, device=work_device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    d = torch.diag(r).sign()
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q * d.unsqueeze(0)
    return q


def cayley_orthogonal(A: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric ``A → R = exp((A − A^T) / 2)`` via matrix exponential.

    The canonical Lie-group exponential map for the special-orthogonal
    group: every orthogonal matrix with determinant +1 is the matrix
    exponential of some skew-symmetric matrix.
    ``torch.linalg.matrix_exp`` uses scaling-and-squaring with Padé
    approximation — well-conditioned for any magnitude of ``A``, no
    ``linalg.solve`` step, and empirically superior to the literal Cayley
    map on the Qwen3.5-0.8B smoke (+2.121%/+11.14% median/max gain vs
    Cayley's +2.080%/+9.40% at identical 60/60 coverage).

    Naming note: this function used to implement the literal Cayley map
    ``(I − A_skew)^{−1} (I + A_skew)``, kept the name for back-compat
    across tests and call sites. The Adam optimizer that drives it works
    identically — at A=0 both maps return ``I``, and for small ``A``
    the two agree to within float-precision noise.
    """
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("orthogonal parametrization input must be a square matrix")
    A_skew = (A - A.t()) / 2
    return torch.linalg.matrix_exp(A_skew)


def _inverse_cayley(R: torch.Tensor) -> torch.Tensor:
    """Inverse Cayley: orthogonal ``R → A`` (skew-symmetric).

    Returns zero matrix if ``R`` has ``−1`` as an eigenvalue (the Cayley
    map's singular set). Used to initialize the optimization at a
    non-trivial ``R`` while reparameterizing via skew-symmetric ``A``.
    """
    g = int(R.shape[0])
    identity = torch.eye(g, device=R.device, dtype=R.dtype)
    try:
        A = torch.linalg.solve(R + identity, R - identity)
    except RuntimeError:
        return torch.zeros_like(R)
    return (A - A.t()) / 2


def apply_block_rotation_input(
    matrix: torch.Tensor, R: torch.Tensor
) -> torch.Tensor:
    """Right-multiply each G-block on the last axis of ``matrix`` by ``R``.

    Equivalent to ``matrix @ block_diag(R, R, ..., R)`` where the block size
    is ``R.shape[0]``. Works on weights ``(out, in)`` and activations
    ``(..., in)`` identically — the operation acts on the last axis.
    """
    if R.dim() != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be square")
    g = int(R.shape[0])
    in_features = int(matrix.shape[-1])
    if in_features % g != 0:
        raise ValueError(
            f"input width {in_features} is not divisible by group_size {g}"
        )
    R = R.to(device=matrix.device, dtype=matrix.dtype)
    grouped = matrix.reshape(*matrix.shape[:-1], in_features // g, g)
    out = grouped @ R
    return out.reshape(matrix.shape)


def rotate_weight_for_storage(
    weight: torch.Tensor, R: torch.Tensor
) -> torch.Tensor:
    """Compute the weight to be stored when ``R`` is the runtime input rotation.

    Runtime applies ``R`` to ``x`` per G-block. Storing ``W @ block_diag(R^T)``
    cancels it back out in float-precision composition, so quantization error
    is the only deviation from the original function.
    """
    return apply_block_rotation_input(weight, R.t())


def effective_weight_for_scoring(
    rendered_rotated: torch.Tensor, R: torch.Tensor
) -> torch.Tensor:
    """Map a rendered rotated weight back to original input-axis coordinates.

    If ``rendered_rotated`` is ``Q(W R^T)``, returns ``Q(W R^T) @ block_diag(R)``,
    which is the weight that — applied to original activations — reproduces
    the runtime computation under the input-side rotation.
    """
    return apply_block_rotation_input(rendered_rotated, R)


# ---------------------------------------------------------------------------
# STE quantizers — differentiable surrogates for the rotation solver
# ---------------------------------------------------------------------------


_NVFP4_E2M1_LEVELS: tuple[float, ...] = (
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
)


def _nvfp4_codebook(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_NVFP4_E2M1_LEVELS, device=device, dtype=dtype)


def _ste_round_e4m3_group_scale(scale: torch.Tensor) -> torch.Tensor:
    """Round positive group scales through the NVFP4 FP8-scale path with STE."""
    scale_f = scale.float()
    global_real = (scale_f.amax() / _E4M3_MAX_ABS).clamp_min(1e-12)
    fp8_scale = (scale_f / global_real).clamp(0, _E4M3_MAX_ABS)
    rounded = fp8_scale.to(torch.float8_e4m3fn).to(fp8_scale.dtype) * global_real
    rounded = rounded.to(dtype=scale.dtype)
    return scale + (rounded - scale).detach()


def ste_rtn_nvfp4_per_group(
    weight: torch.Tensor, group_size: int = NVFP4_GROUP_SIZE
) -> torch.Tensor:
    """NVFP4 per-group symmetric RTN with straight-through estimator.

    Forward: round each ``(group_size,)`` block to the E2M1 codebook with
    microscale = ``max(|w|_block) / 6`` after the same FP8 E4M3 scale
    round-trip used by compressed-tensors NVFP4. Backward: STE bypasses the
    discrete codebook and scale-rounding steps so the operation remains
    differentiable through ``weight``.

    Used only as the differentiable surrogate inside the rotation solver.
    Production cost measurement uses the format-registry's real NVFP4
    renderer, which dispatches through the same numerical path the
    exporter uses.
    """
    if weight.dim() != 2:
        raise ValueError("STE RTN expects a 2D weight")
    rows, cols = weight.shape
    g = int(group_size)
    if cols % g != 0:
        raise ValueError(f"input width {cols} not divisible by group_size {g}")
    grouped = weight.reshape(rows, cols // g, g)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = _ste_round_e4m3_group_scale(max_abs / 6.0)
    normalized = grouped / scale
    codebook = _nvfp4_codebook(weight.device, weight.dtype)
    distances = (normalized.unsqueeze(-1) - codebook).abs()
    nearest = codebook[distances.argmin(dim=-1)]
    quantized = normalized + (nearest - normalized).detach()
    return (quantized * scale).reshape(rows, cols)


# E4M3FN max finite representable value (special encoding excludes ±inf,
# uses 0xFF/0x7F for NaN). Matches torch.float8_e4m3fn.
_E4M3_MAX_ABS = 448.0


def ste_rtn_mxfp8_per_group(
    weight: torch.Tensor, group_size: int = MXFP8_GROUP_SIZE
) -> torch.Tensor:
    """MXFP8 E4M3 per-group symmetric RTN with STE.

    Forward: per ``(group_size,)`` block, scale by ``448 / max(|w|)`` to bring
    the maximum-magnitude element to the E4M3 limit, round-trip through
    ``torch.float8_e4m3fn`` to apply the E4M3 codebook, then scale back.
    Backward: STE through the rounding step.

    Real MXFP8 constrains the per-block scale to a power of two (E8M0).
    This simulator uses an unconstrained scale — a strict upper bound on
    quantization error. The solver's gradient signal stays correct; the
    final per-format cost is re-scored under the production renderer
    before allocator handoff so the power-of-two constraint is captured.
    """
    if weight.dim() != 2:
        raise ValueError("STE RTN expects a 2D weight")
    rows, cols = weight.shape
    g = int(group_size)
    if cols % g != 0:
        raise ValueError(f"input width {cols} not divisible by group_size {g}")
    grouped = weight.reshape(rows, cols // g, g)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = max_abs / _E4M3_MAX_ABS
    normalized = grouped / scale
    # Round-trip through E4M3 to apply the codebook
    quant = normalized.to(torch.float8_e4m3fn).to(normalized.dtype)
    out = normalized + (quant - normalized).detach()
    return (out * scale).reshape(rows, cols)


_QUANTIZER_BY_FORMAT: dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    "NVFP4": ste_rtn_nvfp4_per_group,
    "MXFP8_E4M3": ste_rtn_mxfp8_per_group,
}


# ---------------------------------------------------------------------------
# Zigzag permutation calibration + composition with rotation
# ---------------------------------------------------------------------------


def calibrate_zigzag_permutation(
    activation_magnitudes: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Compute within-block zigzag (snake) permutation.

    Algorithm:
      1. Average per-channel magnitudes by within-block position
         (averaging across the ``in_features / group_size`` blocks).
      2. Sort positions by mean magnitude (ascending).
      3. Interleave: smallest, largest, 2nd-smallest, 2nd-largest, ...
         so adjacent positions within each block have very different
         magnitudes — a known-good initialization for Hadamard / orthogonal
         mixing because the rotation can pair high-magnitude channels
         with low-magnitude ones for variance equalization.

    The same permutation applies to every G-block (compressed-tensors'
    HadamardTransform similarly uses a single shared perm). For our
    learned-orthogonal case we fold this permutation directly into the
    stored matrix via :func:`compose_rotation_with_permutation`, so the
    runtime sees one ``(G, G)`` matrix.

    Args:
        activation_magnitudes: shape ``(in_features,)`` — per-channel
            magnitudes. Typically the mean absolute activation over a
            calibration batch.
        group_size: ``G``, the within-block dimension.

    Returns:
        ``LongTensor`` of shape ``(group_size,)`` giving the within-block
        permutation. Deterministic given the input magnitudes.
    """
    g = int(group_size)
    mags = activation_magnitudes.detach().to(dtype=torch.float64).reshape(-1)
    in_features = int(mags.numel())
    if in_features % g != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {g}"
        )
    per_position = mags.reshape(in_features // g, g).mean(dim=0)
    sorted_positions = torch.argsort(per_position, descending=False)

    snake = torch.empty(g, dtype=torch.long)
    lo, hi = 0, g - 1
    for k in range(g):
        if k % 2 == 0:
            snake[k] = int(sorted_positions[lo].item())
            lo += 1
        else:
            snake[k] = int(sorted_positions[hi].item())
            hi -= 1
    return snake


def compose_rotation_with_permutation(
    R: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """Compose a ``(G, G)`` rotation with a ``(G,)`` within-block permutation.

    Returns ``M[i, j] = R[perm[i], perm[j]]`` — the matrix that the
    cache/export path materializes once so folded weights and optional
    runtime transforms use the same coordinate system without needing a
    separate permutation field at runtime.

    Algebraically this is a similarity transformation ``P @ R @ P^T`` where
    ``P`` is the permutation matrix whose row ``i`` is the ``perm[i]``-th
    standard basis vector.
    """
    if R.dim() != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be square")
    if perm.dim() != 1 or int(perm.shape[0]) != int(R.shape[0]):
        raise ValueError(
            f"perm shape {perm.shape} must be (G,) matching R shape {R.shape}"
        )
    perm = perm.to(device=R.device)
    return R[perm][:, perm]


# ---------------------------------------------------------------------------
# Insertion-point identification (HALO-pattern adapted for DuQuant)
# ---------------------------------------------------------------------------


class InsertionPointKind(str, Enum):
    """The four DuQuant++ rotation insertion points."""

    # Residual-stream rotation — offline-fold via producer WEIGHT_OUTPUT
    # + consumer WEIGHT_INPUT.
    RESIDUAL = "residual"

    # V → O rotation — offline-fold via V WEIGHT_OUTPUT + O WEIGHT_INPUT.
    V_O = "v_o"

    # attn out_proj input rotation — offline-fold via O WEIGHT_INPUT
    # (distinct from V_O when V and O have different fused-tensor layouts).
    ATTN_OUT = "attn_out"

    # down_proj input rotation — online. SwiGLU's elementwise multiplication
    # breaks offline fold algebra; INPUT location applies M at runtime,
    # WEIGHT_INPUT (with inverse=True) folds M^{-1} into the weight.
    DOWN_PROJ = "down_proj"


@dataclass(frozen=True)
class HadamardDuQuantSpec:
    """One DuQuant++ rotation insertion site.

    A spec is a *recipe* for where to apply a rotation; the rotation matrix
    itself is solved separately per cluster by
    :func:`solve_cluster_rotation`.

    Attributes:
        cluster_key: Unique identifier for the fused-sibling cluster
            (used as the key in cost sidecars and decision records).
        kind: Which of the four DuQuant++ insertion points this is.
        input_dim: Dimension of the rotated activation axis. Must be
            divisible by ``group_size``.
        group_size: ``G`` — 16 for NVFP4, 32 for MXFP8. Set by the
            cluster's per-format choice.
        consumer_qnames: Linears whose input axis receives the rotation.
            For folded producer/consumer clusters such as ``V_O`` the
            location is ``WEIGHT_INPUT``. For runtime-input clusters such
            as residual-stream and ``DOWN_PROJ`` rotations the location is
            ``INPUT`` (online runtime hook) and the same Linear also gets
            its input-axis weight rotated.
        producer_qnames: Linears whose output axis is in the rotated
            basis. Location is ``WEIGHT_OUTPUT`` (offline fold). Empty
            for ``DOWN_PROJ`` (the producer is SwiGLU, not a Linear).
        online: ``True`` for runtime input transforms; ``False`` only when
            every consumer has a matching Linear producer fold.
    """

    cluster_key: str
    kind: InsertionPointKind
    input_dim: int
    group_size: int
    consumer_qnames: tuple[str, ...]
    producer_qnames: tuple[str, ...]
    online: bool

    def __post_init__(self) -> None:
        if self.input_dim % self.group_size != 0:
            raise ValueError(
                f"{self.cluster_key}: input_dim {self.input_dim} not divisible "
                f"by group_size {self.group_size}"
            )
        if self.online and self.producer_qnames:
            raise ValueError(
                f"{self.cluster_key}: online insertion points have no producer "
                f"Linears, got {self.producer_qnames}"
            )
        if not self.online and not self.producer_qnames:
            raise ValueError(
                f"{self.cluster_key}: offline rotations require producer "
                "Linears so the basis change is folded at both ends; use "
                "online=True or skip this insertion point"
            )
        if self.kind == InsertionPointKind.DOWN_PROJ and not self.online:
            raise ValueError(
                f"{self.cluster_key}: down_proj rotations must be online; "
                "SwiGLU has no Linear producer to fold"
            )


def _has_attr_path(parent: nn.Module, qname: str) -> bool:
    """Check whether a dotted path resolves to a submodule of ``parent``."""
    parts = [p for p in qname.split(".") if p]
    cur: Any = parent
    for p in parts:
        try:
            if p.isdigit():
                cur = cur[int(p)]
            else:
                cur = getattr(cur, p)
        except (AttributeError, IndexError, KeyError, TypeError):
            return False
    return True


def _get_module_by_relpath(parent: nn.Module, relpath: str) -> nn.Module:
    parts = [p for p in relpath.split(".") if p]
    cur: Any = parent
    for p in parts:
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur


def insertion_specs_for_layer(
    layer_mod: nn.Module,
    layer_qname: str,
    *,
    group_size: int,
    hidden_dim: int,
    include_folded: bool = True,
    include_online: bool = False,
    include_residual_online: bool = False,
) -> list[HadamardDuQuantSpec]:
    """Enumerate DuQuant++ insertion points for one transformer layer.

    Detects standard projections by qname convention:
      - ``self_attn.q_proj``, ``self_attn.k_proj``, ``self_attn.v_proj``
      - ``self_attn.o_proj`` (Qwen / Llama) or ``self_attn.out_proj`` (Gemma)
      - ``mlp.gate_proj``, ``mlp.up_proj``, ``mlp.down_proj``

    Linear-attention variants (Qwen3.5/3.6 ``linear_attn.in_proj_*``) and
    MoE expert tensors are deferred — handled by the production cache's
    cluster-discovery path which already exposes ``_awq_group_key_from_qname``.

    Args:
        layer_mod: the layer module (e.g., ``model.layers[5]``).
        layer_qname: dotted path to the layer (e.g., ``"model.layers.5"``).
        group_size: ``G`` — must divide ``hidden_dim`` and the FFN
            intermediate dim.
        hidden_dim: residual-stream dimension.

        include_folded: include algebraically folded producer/consumer
            clusters such as V→O. This is production-safe by default.
        include_online: include runtime-input transform clusters such as
            ``down_proj``. These are opt-in until the compressed-tensors /
            vLLM transform gate is proven for the target runtime.
        include_residual_online: include residual-stream input rotations
            as ONLINE transforms. Offline residual rotations without a
            producer fold are invalid and are never emitted.

    Returns:
        Zero to four ``HadamardDuQuantSpec`` entries per layer. Skips
        insertion kinds whose projections aren't present or whose
        dimensions aren't divisible by ``group_size``.
    """
    specs: list[HadamardDuQuantSpec] = []

    # Attention projections
    have_qkv = all(
        _has_attr_path(layer_mod, f"self_attn.{name}")
        for name in ("q_proj", "k_proj", "v_proj")
    )
    o_proj_rel: str | None = None
    for cand in ("self_attn.o_proj", "self_attn.out_proj"):
        if _has_attr_path(layer_mod, cand):
            o_proj_rel = cand
            break

    if have_qkv and o_proj_rel is not None:
        q_qname = f"{layer_qname}.self_attn.q_proj"
        k_qname = f"{layer_qname}.self_attn.k_proj"
        v_qname = f"{layer_qname}.self_attn.v_proj"
        o_qname = f"{layer_qname}.{o_proj_rel}"

        # RESIDUAL — q/k/v share the residual stream input. This is only
        # valid as an online input transform; there is no in-layer Linear
        # producer to fold the residual basis change into.
        if include_residual_online and hidden_dim % group_size == 0:
            specs.append(
                HadamardDuQuantSpec(
                    cluster_key=f"{layer_qname}.attn.residual",
                    kind=InsertionPointKind.RESIDUAL,
                    input_dim=hidden_dim,
                    group_size=group_size,
                    consumer_qnames=(q_qname, k_qname, v_qname),
                    producer_qnames=(),
                    online=True,
                )
            )

        # V_O — V_proj's output axis is rotated; o_proj's input axis matches.
        v_mod = _get_module_by_relpath(layer_mod, "self_attn.v_proj")
        v_out = int(getattr(v_mod, "out_features"))
        if include_folded and v_out % group_size == 0:
            specs.append(
                HadamardDuQuantSpec(
                    cluster_key=f"{layer_qname}.attn.v_o",
                    kind=InsertionPointKind.V_O,
                    input_dim=v_out,
                    group_size=group_size,
                    consumer_qnames=(o_qname,),
                    producer_qnames=(v_qname,),
                    online=False,
                )
            )

    # MLP — dense path only (MoE deferred to cache-side cluster discovery)
    have_gate_up = all(
        _has_attr_path(layer_mod, f"mlp.{name}")
        for name in ("gate_proj", "up_proj")
    )
    have_down = _has_attr_path(layer_mod, "mlp.down_proj")
    if have_gate_up and have_down:
        gate_qname = f"{layer_qname}.mlp.gate_proj"
        up_qname = f"{layer_qname}.mlp.up_proj"
        down_qname = f"{layer_qname}.mlp.down_proj"

        # RESIDUAL — gate/up share the post-attention residual input. Like
        # attention residual, this has no Linear producer inside the layer,
        # so it is only valid as an online input transform.
        if include_residual_online and hidden_dim % group_size == 0:
            specs.append(
                HadamardDuQuantSpec(
                    cluster_key=f"{layer_qname}.mlp.residual",
                    kind=InsertionPointKind.RESIDUAL,
                    input_dim=hidden_dim,
                    group_size=group_size,
                    consumer_qnames=(gate_qname, up_qname),
                    producer_qnames=(),
                    online=True,
                )
            )

        # DOWN_PROJ — online rotation at down_proj input. SwiGLU breaks fold.
        down_mod = _get_module_by_relpath(layer_mod, "mlp.down_proj")
        down_in = int(getattr(down_mod, "in_features"))
        if include_online and down_in % group_size == 0:
            specs.append(
                HadamardDuQuantSpec(
                    cluster_key=f"{layer_qname}.mlp.down",
                    kind=InsertionPointKind.DOWN_PROJ,
                    input_dim=down_in,
                    group_size=group_size,
                    consumer_qnames=(down_qname,),
                    producer_qnames=(),
                    online=True,
                )
            )

    return specs


def default_insertion_specs(
    model: nn.Module,
    *,
    group_size: int,
    body_layer_prefix: str = "model.layers",
    hidden_dim: int | None = None,
    include_folded: bool = True,
    include_online: bool = False,
    include_residual_online: bool = False,
) -> list[HadamardDuQuantSpec]:
    """Aggregate insertion specs across all transformer layers in ``model``.

    Walks ``model.layers`` (or the alternative ``body_layer_prefix``) and
    invokes :func:`insertion_specs_for_layer` per layer.

    Args:
        model: the transformer model.
        group_size: ``G``. Same value applied across all layers.
        body_layer_prefix: dotted path to the layer-list container.
        hidden_dim: residual-stream dim. If ``None``, inferred from the
            first layer's ``self_attn.q_proj.in_features``.
        include_folded / include_online / include_residual_online: forwarded
            to :func:`insertion_specs_for_layer`. Defaults are production-safe:
            folded producer/consumer rotations only.
    """
    layers: Any = model
    for p in [seg for seg in body_layer_prefix.split(".") if seg]:
        if not hasattr(layers, p):
            raise ValueError(
                f"layer container '{body_layer_prefix}' not found in model "
                f"(missing segment '{p}')"
            )
        layers = getattr(layers, p)

    if hidden_dim is None:
        # Walk layers looking for any Linear whose in_features is the
        # residual-stream dim. Models with mixed attention types (e.g.,
        # Qwen3.5 interleaves linear_attn and self_attn) may not have
        # self_attn on layer 0; try several candidates per layer.
        candidate_relpaths = (
            "self_attn.q_proj",
            "linear_attn.in_proj_qkv",
            "mlp.gate_proj",
            "mlp.up_proj",
        )
        for layer_idx, layer_mod in enumerate(layers):
            for rel in candidate_relpaths:
                if _has_attr_path(layer_mod, rel):
                    try:
                        mod = _get_module_by_relpath(layer_mod, rel)
                    except (AttributeError, IndexError, KeyError):
                        continue
                    in_feat = getattr(mod, "in_features", None)
                    if in_feat:
                        hidden_dim = int(in_feat)
                        break
            if hidden_dim is not None:
                break
        if hidden_dim is None:
            raise ValueError(
                "could not infer hidden_dim from any layer's standard "
                "projection (tried self_attn.q_proj, linear_attn.in_proj_qkv, "
                "mlp.gate_proj, mlp.up_proj) — pass --hidden-dim explicitly"
            )

    all_specs: list[HadamardDuQuantSpec] = []
    for i, layer_mod in enumerate(layers):
        layer_qname = f"{body_layer_prefix}.{i}"
        all_specs.extend(
            insertion_specs_for_layer(
                layer_mod,
                layer_qname,
                group_size=group_size,
                hidden_dim=hidden_dim,
                include_folded=include_folded,
                include_online=include_online,
                include_residual_online=include_residual_online,
            )
        )
    return all_specs


# ---------------------------------------------------------------------------
# Cluster rotation solver
# ---------------------------------------------------------------------------


@dataclass
class ClusterRotationTarget:
    """One Linear participating in a cluster's joint rotation solve.

    All targets in a cluster share their input axis (the axis the rotation
    acts on); each target may have its own row weights (Fisher importance).

    Attributes:
        qname: dotted name of the Linear in the model.
        weight: ``(out_features, in_features)`` weight tensor.
        activations: ``(N, in_features)`` calibration activations.
        score_weight: cluster-internal weighting for the joint loss
            (default 1.0). Caller can use this to upweight high-importance
            Linears (e.g., q_proj over k_proj if grad-norm-weighted).
        row_weights: optional ``(N,)`` Fisher/output-importance weights for
            per-row reweighting of the output MSE.
    """

    qname: str
    weight: torch.Tensor
    activations: torch.Tensor
    score_weight: float = 1.0
    row_weights: torch.Tensor | None = None


@dataclass
class ClusterRotationResult:
    """Result of solving the rotation for one cluster."""

    R: torch.Tensor                  # (G, G) learned orthogonal
    permutation: torch.Tensor        # (G,) zigzag permutation
    composed_matrix: torch.Tensor    # (G, G) — R[perm][:, perm], the shipped matrix
    baseline_score: float            # no-rotation Fisher-weighted MSE
    rotated_score: float             # learned-rotation Fisher-weighted MSE
    relative_gain: float             # (baseline - rotated) / |baseline|
    solver_seconds: float
    orthogonality_err: float         # ||R R^T - I||_F sanity check
    init_strategy: str
    n_iters: int
    # Convergence diagnostics:
    #   best_iter: 1-based iter index where best_loss was achieved. 0 means
    #     the initial state (Cayley(0) @ init_R) was the best — no Adam step
    #     beat it.
    #   final_iter_loss: loss at the last iter (post-step), regardless of
    #     whether it was the best — useful to compare against best_loss to
    #     see how much the optimizer drifted after its best iterate.
    #   still_improving: True when best_iter lies in the last 20% of the
    #     run; signals "would likely benefit from more iters."
    best_iter: int = 0
    final_iter_loss: float = float("nan")
    still_improving: bool = False
    # early_stopped: True if the loop exited before n_iters because no
    # improvement was seen for ``early_stop_patience`` consecutive iters.
    # Useful for verifying that the patience budget is reasonable.
    early_stopped: bool = False

    def to_log_dict(self) -> dict[str, Any]:
        """Compact serializable representation for log records.

        Tensors are dropped (those go to safetensors); only scalar metadata
        is preserved.
        """
        return {
            "applied": bool(self.relative_gain > 0.0),
            "G": int(self.R.shape[0]),
            "permutation_swaps": int((self.permutation != torch.arange(
                self.permutation.shape[0], device=self.permutation.device
            )).sum().item()),
            "solver_seconds": float(self.solver_seconds),
            "orthogonality_err": float(self.orthogonality_err),
            "init_strategy": str(self.init_strategy),
            "n_iters": int(self.n_iters),
            "baseline_score": float(self.baseline_score),
            "rotated_score": float(self.rotated_score),
            "relative_gain": float(self.relative_gain),
            "best_iter": int(self.best_iter),
            "final_iter_loss": float(self.final_iter_loss),
            "still_improving": bool(self.still_improving),
            "early_stopped": bool(self.early_stopped),
        }


def _compute_cluster_loss(
    M: torch.Tensor,
    weights_f32: Sequence[torch.Tensor],
    acts_f32: Sequence[torch.Tensor],
    targets: Sequence[ClusterRotationTarget],
    quantizer: Callable[[torch.Tensor, int], torch.Tensor],
    group_size: int,
    row_chunk: int,
) -> torch.Tensor:
    """Cluster output-MSE loss under shipped matrix ``M`` (post-permutation).

    For each target Linear:
      - rotated weight under storage convention: ``W_stored = W @ M^T``
      - quantize: ``W_stored_q = Q(W_stored)``
      - effective weight in original coords: ``W_eff = W_stored_q @ M``
      - output diff vs original: ``(W - W_eff) @ x^T`` reshaped via
        ``x @ (W - W_eff)^T`` for activations of shape ``(N, in)``.

    Optional Fisher row-weights reweight the per-row contributions.
    """
    total = torch.zeros((), device=M.device, dtype=torch.float32)
    g = int(group_size)
    for t, w, x in zip(targets, weights_f32, acts_f32):
        w_stored = apply_block_rotation_input(w, M.t())
        w_stored_q = quantizer(w_stored, g)
        w_eff = apply_block_rotation_input(w_stored_q, M)
        diff_t = (w - w_eff).t().contiguous()
        err_sum = torch.zeros((), device=M.device, dtype=torch.float32)
        n_rows = 0
        for s in range(0, x.shape[0], int(row_chunk)):
            y = x[s:s + int(row_chunk)] @ diff_t
            if t.row_weights is not None:
                rw = t.row_weights[s:s + y.shape[0]].to(
                    device=y.device, dtype=y.dtype
                )
                err = (y.pow(2) * rw.unsqueeze(1)).sum()
            else:
                err = y.pow(2).sum()
            err_sum = err_sum + err
            n_rows += int(y.shape[0])
        denom = max(1, n_rows * int(w.shape[0]))
        total = total + float(t.score_weight) * (err_sum / denom)
    return total


def _compute_cluster_loss_w4a4(
    M: torch.Tensor,
    weights_f32: Sequence[torch.Tensor],
    acts_f32: Sequence[torch.Tensor],
    targets: Sequence[ClusterRotationTarget],
    quantizer: Callable[[torch.Tensor, int], torch.Tensor],
    group_size: int,
    row_chunk: int,
) -> torch.Tensor:
    """Joint W4A4 loss: ``||x w^T - Q_a(x M^T) @ Q_w(W M^T)^T||^2``.

    Models the runtime exactly (W4A4 path): both activations and weights
    pass through the STE quantizer with shared per-G-block scales. At M = I
    this equals the standard per-Linear W4A4 reconstruction MSE; for any
    orthogonal M with Q = identity the inner product closes back to the
    target.

    The W-only loss in :func:`_compute_cluster_loss` only sees the weight
    quantization error and so cannot credit a rotation whose primary
    benefit is reshaping per-block ACTIVATION distributions. For NVFP4
    W4A4 in particular, the activation side carries roughly half the
    total quantization error (see tools/w4a4_geodesic_sweep.py).
    """
    total = torch.zeros((), device=M.device, dtype=torch.float32)
    g = int(group_size)
    for t, w, x in zip(targets, weights_f32, acts_f32):
        w_stored = apply_block_rotation_input(w, M.t())
        w_stored_q = quantizer(w_stored, g)
        err_sum = torch.zeros((), device=M.device, dtype=torch.float32)
        n_rows = 0
        for s in range(0, x.shape[0], int(row_chunk)):
            x_chunk = x[s : s + int(row_chunk)]
            y_ref = x_chunk @ w.t()
            x_rot_chunk = apply_block_rotation_input(x_chunk, M.t())
            x_rot_q = quantizer(x_rot_chunk, g)
            y_rt = x_rot_q @ w_stored_q.t()
            y_diff = y_ref - y_rt
            if t.row_weights is not None:
                rw = t.row_weights[s : s + y_diff.shape[0]].to(
                    device=y_diff.device, dtype=y_diff.dtype
                )
                err = (y_diff.pow(2) * rw.unsqueeze(1)).sum()
            else:
                err = y_diff.pow(2).sum()
            err_sum = err_sum + err
            n_rows += int(y_diff.shape[0])
        denom = max(1, n_rows * int(w.shape[0]))
        total = total + float(t.score_weight) * (err_sum / denom)
    return total


def solve_cluster_rotation(
    targets: Sequence[ClusterRotationTarget],
    *,
    group_size: int,
    format_label: str,
    init_strategy: str = "sylvester",
    loss_kind: Literal["w_only", "w4a4"] = "w_only",
    seed: int = 0,
    n_iters: int = 60,
    lr: float = 5e-3,
    weight_decay: float = 0.0,
    row_chunk: int = 256,
    permutation: torch.Tensor | None = None,
    early_stop_patience: int | None = 100,
) -> ClusterRotationResult:
    """Optimize a ``(G, G)`` orthogonal rotation for a fused-sibling cluster.

    Process:
      1. Calibrate zigzag permutation ``P`` from aggregate per-channel
         activation magnitudes across cluster siblings (unless an explicit
         ``permutation`` is provided).
      2. Initialize the rotation from ``{sylvester, random, identity}``.
      3. Cayley-parametrize ``R`` via a free skew-symmetric ``A`` and run
         Adam against an STE-quantized output MSE loss summed across all
         cluster siblings. The loss is computed against the *composed*
         matrix ``M = P^T R P`` so the learned rotation is optimized in
         the post-permutation basis the runtime will actually see.
      4. Compose and return ``M``.

    Args:
        targets: cluster siblings (must share input width).
        group_size: ``G`` — must divide the cluster input width.
        format_label: one of :data:`SUPPORTED_FORMATS` — selects the STE
            quantizer.
        init_strategy: how to initialize ``R``.
        seed: RNG seed for random initialization.
        n_iters: Adam optimization steps.
        lr: Adam learning rate.
        row_chunk: minibatch size for the row-loop in the loss.
        permutation: optional pre-computed permutation; if ``None``, the
            permutation is calibrated from ``targets`` activation magnitudes.

    Returns:
        :class:`ClusterRotationResult` with the learned ``R``, the
        permutation, the composed matrix ``M``, and scoring metadata
        suitable for the render-gate decision in
        :mod:`prismaquant.render_score`.
    """
    if not targets:
        raise ValueError("solve_cluster_rotation requires at least one target")
    if format_label not in SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported format_label {format_label!r} — "
            f"must be one of {sorted(SUPPORTED_FORMATS)}"
        )
    g = int(group_size)
    cols = int(targets[0].weight.shape[1])
    if cols % g != 0:
        raise ValueError(f"input_dim {cols} not divisible by group_size {g}")
    for t in targets:
        if int(t.weight.shape[1]) != cols:
            raise ValueError("all cluster targets must share input width")
        if int(t.activations.shape[-1]) != cols:
            raise ValueError(
                "activation width must equal weight input width "
                f"({t.activations.shape[-1]} vs {cols} for {t.qname})"
            )
    device = targets[0].weight.device
    quantizer = _QUANTIZER_BY_FORMAT[format_label]
    if loss_kind == "w_only":
        loss_fn = _compute_cluster_loss
    elif loss_kind == "w4a4":
        loss_fn = _compute_cluster_loss_w4a4
    else:
        raise ValueError(f"unknown loss_kind {loss_kind!r}")

    # Step 1: zigzag permutation
    if permutation is None:
        with torch.no_grad():
            total_mag = torch.zeros(cols, device=device, dtype=torch.float64)
            n_seen = 0
            for t in targets:
                x = t.activations.detach().to(device=device, dtype=torch.float64)
                x_flat = x.reshape(-1, cols)
                total_mag = total_mag + x_flat.abs().mean(dim=0)
                n_seen += 1
            avg_mag = total_mag / max(1, n_seen)
            perm = calibrate_zigzag_permutation(avg_mag, g).to(device=device)
    else:
        perm = permutation.to(device=device)
        if int(perm.shape[0]) != g:
            raise ValueError(
                f"permutation length {perm.shape[0]} must equal group_size {g}"
            )

    # Step 2: initialize R (kept fixed; the Cayley parametrization optimizes
    # a multiplicative correction).
    if init_strategy == "sylvester":
        if (g & (g - 1)) != 0:
            # Sylvester requires power-of-two; fall back to identity
            init_R = torch.eye(g, device=device, dtype=torch.float32)
        else:
            init_R = sylvester_hadamard(g, device=device, dtype=torch.float32)
    elif init_strategy == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        init_R = random_orthogonal(g, generator=gen, device=device, dtype=torch.float32)
    elif init_strategy == "identity":
        init_R = torch.eye(g, device=device, dtype=torch.float32)
    elif init_strategy.startswith("sylvester_t"):
        # Fractional Sylvester: R = orthogonalize((1 - t) I + t H_sylvester).
        # Geodesic sweep on Qwen3-4B (tools/w4a4_geodesic_sweep.py) showed
        # W4A4 minima land at t ∈ {0.1, 0.3, 0.5, 0.7, 1.0} per cluster.
        # Useful as multi-init candidates: gives Adam a "warm start" inside
        # SO(g) along the geodesic to Sylvester without locking into the
        # full-strength rotation that often regresses on small models.
        suffix = init_strategy[len("sylvester_t"):]
        try:
            t = float(suffix.replace("p", "."))
        except ValueError as exc:
            raise ValueError(
                f"unparseable sylvester_t fraction {suffix!r}; "
                "expected e.g. sylvester_t0p3"
            ) from exc
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"sylvester_t fraction must lie in [0,1], got {t}")
        if (g & (g - 1)) != 0:
            init_R = torch.eye(g, device=device, dtype=torch.float32)
        else:
            H = sylvester_hadamard(g, device=device, dtype=torch.float32)
            I_g = torch.eye(g, device=device, dtype=torch.float32)
            mix = (1.0 - t) * I_g + t * H
            q, r = torch.linalg.qr(mix)
            d = torch.diag(r).sign()
            d = torch.where(d == 0, torch.ones_like(d), d)
            init_R = (q * d.unsqueeze(0)).contiguous()
    elif init_strategy == "givens_balance":
        # Data-inferred constructive init: pair columns of W within each
        # G-block by descending column-norm² (largest with smallest, etc.)
        # and apply a 2×2 Givens rotation per pair that equalizes the
        # post-rotation column norms.
        #
        # Derivation: for a 2x2 rotation by angle θ on cols (a, b) of W,
        # let c_aa = Σ_i W[i,a]², c_bb = Σ_i W[i,b]², c_ab = Σ_i W[i,a] W[i,b].
        # The new column norms satisfy:
        #   new_aa = cos²θ · c_aa - 2 sinθ cosθ · c_ab + sin²θ · c_bb
        # Trace is preserved (orthogonal rotation), so we want
        #   new_aa = new_bb = (c_aa + c_bb) / 2.
        # Solving:  tan(2θ) = (c_aa - c_bb) / (2 c_ab).
        # Using atan2 for quadrant correctness and numerical stability.
        #
        # Why this beats sylvester / sylvester_tX: the rotation is
        # tailored to the actual weight matrix's outlier profile. For a
        # uniformly-magnitude block (no outliers), all pair angles → 0
        # and R → I (no perturbation). For an outlier-heavy block,
        # outlier channel gets paired with its weakest companion and
        # spread by exactly the angle that balances them, no more.
        # Adam then refines from this near-optimal starting point.
        GtG = torch.zeros(g, g, device=device, dtype=torch.float64)
        for t_target in targets:
            w64 = t_target.weight.detach().to(device=device, dtype=torch.float64)
            wb = w64.view(w64.shape[0], -1, g)
            GtG = GtG + torch.einsum("obg,obh->gh", wb, wb)
        # Symmetrize (kills numerical asymmetry)
        GtG = (GtG + GtG.t()) / 2
        # Pair by sorting: greedy pair largest-norm with smallest, etc.
        col_n2 = GtG.diagonal().clone()
        sorted_idx = torch.argsort(col_n2, descending=True).tolist()
        pairs = []
        for k in range(g // 2):
            pairs.append((sorted_idx[k], sorted_idx[g - 1 - k]))
        # Compose disjoint Givens rotations into a single G×G R. For each
        # candidate pair, EXPLICITLY measure the change in the actual
        # quantization-error surrogate (Σ_block max²(|W_block|)) and
        # commit only if it strictly reduces. No heuristic thresholds —
        # the data itself decides which pairs to rotate, and the test
        # uses the same per-block-max² quantity that drives the NVFP4
        # E2M1 scale = max_abs/6 and thus the per-element quant error.
        def _per_block_max_sq_sum(W_block_view: torch.Tensor) -> float:
            # W_block_view shape: (out, n_blocks, g)
            return float(W_block_view.abs().amax(dim=-1).pow(2).sum().item())

        # Stack all targets' weights once for cost evaluation
        W_concat = torch.cat(
            [t_target.weight.detach().to(device=device, dtype=torch.float64)
             for t_target in targets], dim=0,
        )  # shape: (total_out, in)
        # We'll apply column rotations to W_concat to evaluate the cost
        # without writing back to targets[*].weight.
        W_view = W_concat.view(W_concat.shape[0], -1, g)
        cost_pre = _per_block_max_sq_sum(W_view)

        R64 = torch.eye(g, device=device, dtype=torch.float64)
        for (a, b) in pairs:
            c_aa = float(GtG[a, a].item())
            c_bb = float(GtG[b, b].item())
            c_ab = float(GtG[a, b].item())
            # tan(2θ) = (c_aa - c_bb) / (2 c_ab); atan2 for quadrant safety.
            # When both arguments vanish (identical columns), atan2 returns
            # 0 → θ = 0 → no rotation. Floating-point near-zero cases give
            # arbitrary θ; we'd reject those at the explicit cost check.
            two_theta = math.atan2(c_aa - c_bb, 2.0 * c_ab)
            theta = 0.5 * two_theta
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            # Trial: apply rotation, evaluate cost, revert if not improved.
            col_a_W = W_concat[:, a].clone()
            col_b_W = W_concat[:, b].clone()
            W_concat[:, a] = cos_t * col_a_W - sin_t * col_b_W
            W_concat[:, b] = sin_t * col_a_W + cos_t * col_b_W
            cost_post = _per_block_max_sq_sum(W_view)
            if cost_post < cost_pre:
                # Accept: update R and the running cost; keep W_concat as-is
                col_a_R = R64[:, a].clone()
                col_b_R = R64[:, b].clone()
                R64[:, a] = cos_t * col_a_R - sin_t * col_b_R
                R64[:, b] = sin_t * col_a_R + cos_t * col_b_R
                cost_pre = cost_post
            else:
                # Reject: revert the trial mutation on W_concat
                W_concat[:, a] = col_a_W
                W_concat[:, b] = col_b_W
        init_R = R64.to(dtype=torch.float32).contiguous()
    elif init_strategy == "svd_v":
        # Aggregate per-block input covariance C = Σ_b W_b^T W_b across all
        # blocks of all cluster siblings, then eigendecompose. Columns of V
        # (descending eigenvalue) point along the directions in which W has
        # the most per-block energy on average. Rotating inputs by V^T aligns
        # them with those principal directions, concentrating per-block W
        # energy onto the leading axes — the opposite of Hadamard spreading.
        # Concentration is what NVFP4's E2M1 grid prefers (dense near 0,
        # sparse far), so this init points the solver into the basin that
        # matches the format's quantization geometry.
        cov = torch.zeros(g, g, device=device, dtype=torch.float64)
        for t in targets:
            w64 = t.weight.detach().to(device=device, dtype=torch.float64)
            w_blocks = w64.view(w64.shape[0], -1, g)
            cov = cov + torch.einsum("obg,obh->gh", w_blocks, w_blocks)
        cov = (cov + cov.t()) / 2
        _eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvecs = eigvecs.flip(dims=[1])  # descending eigenvalue
        init_R = eigvecs.t().to(dtype=torch.float32).contiguous()
    else:
        raise ValueError(f"unknown init_strategy {init_strategy!r}")
    init_R = init_R.detach()

    # Pre-bake target tensors once
    weights_f32 = [
        t.weight.detach().to(device=device, dtype=torch.float32) for t in targets
    ]
    acts_f32 = [
        t.activations.detach().to(device=device, dtype=torch.float32).reshape(
            -1, cols
        )
        for t in targets
    ]

    # Step 3: Cayley-parameterized Adam over a multiplicative correction.
    #
    # Parametrize R = Cayley(A) @ init_R with A starting at zero. This avoids
    # _inverse_cayley(init_R), which is ill-conditioned for any init_R with
    # eigenvalue -1 — in particular for the normalized Sylvester Hadamard at
    # G=16 (a symmetric involution with eigenvalues ±1). Product of two
    # orthogonals is orthogonal, so R stays on the manifold at every step.
    # The loss uses M = R[perm][:, perm].
    #
    # Best-so-far tracking: Adam at non-trivial lr can overshoot — the final
    # ``cayley(A)`` may have higher loss than an intermediate iterate. We
    # track the best M seen across the whole loop and return that, so the
    # solver's output is monotone-non-regressive against its own initial
    # state. If the optimizer's bouncing around makes every step worse than
    # the initial M (Cayley(0) @ init_R), we return that initial M.
    start_time = time.time()
    A = torch.nn.Parameter(
        torch.zeros(g, g, device=device, dtype=torch.float32)
    )
    # Adam at flat lr. We deliberately do NOT use quasi-Newton methods
    # (L-BFGS) or LR schedules here — the loss flows through a
    # straight-through estimator (STE) on the format quantizer, which
    # zeroes the analytic gradient w.r.t. the rotation in the unquantized
    # limit: the backward pass sees ``W - W @ M^T @ M = 0`` for orthogonal
    # M, so any optimizer that trusts the gradient magnitude (L-BFGS,
    # strong-Wolfe line search, etc.) terminates at A=0 immediately.
    # Adam works under STE because momentum + per-parameter scale
    # normalization push finite-size steps regardless of gradient
    # magnitude, and best-so-far tracking keeps us at the local optimum
    # if Adam overshoots. Flat lr is more reliable than cosine decay for
    # the bimodal cluster distribution (most converge in <30 iters; a
    # few V→O clusters keep descending past iter 100). Early-stop with
    # patience prunes the wasted iters on the converged clusters without
    # cutting off the slow learners.
    # AdamW with weight decay on A_skew biases the optimizer toward smaller
    # rotations (closer to identity). With small calibration (<5K tokens),
    # unregularized Adam can overfit a per-cluster rotation that beats the
    # calib loss but doesn't generalize — wd>0 trades some calib-set fit for
    # better generalization. weight_decay=0 (default) preserves the legacy
    # vanilla-Adam behavior.
    if float(weight_decay) > 0.0:
        optimizer = torch.optim.AdamW(
            [A], lr=float(lr), weight_decay=float(weight_decay)
        )
    else:
        optimizer = torch.optim.Adam([A], lr=float(lr))

    best_loss = float("inf")
    best_iter = 0  # 0 means initial state was best (no step beat it)
    final_iter_loss = float("nan")
    # Initialize best_R / best_M to the starting state (Cayley(0) @ init_R).
    # For identity init this is the identity rotation; for sylvester init
    # this is the normalized Sylvester. Either way the candidate is
    # well-defined before any optimization step runs and serves as the
    # "iter 0" baseline against which the Adam loop competes.
    with torch.no_grad():
        best_R = init_R.detach().clone()
        best_M = best_R[perm][:, perm].detach().clone()
        # Score the initial state so best_loss starts at a meaningful value.
        # This also makes "iter 0" comparable to any later iter's loss.
        best_loss = float(
            loss_fn(
                best_M, weights_f32, acts_f32, targets, quantizer, g, row_chunk
            ).item()
        )

    # Early-stop: break when no improvement is seen for ``early_stop_patience``
    # consecutive iters. Convergence is bimodal in practice — most clusters
    # find their best by iter ~30, then drift; a minority keep improving
    # past iter 100. Patience=50 lets the fast ones exit early (saving
    # solver wall-clock) while the slow ones keep their full budget.
    early_stopped_flag = False
    iters_since_improvement = 0
    actual_steps_run = 0
    for step in range(int(n_iters)):
        optimizer.zero_grad()
        delta_R = cayley_orthogonal(A)
        R = delta_R @ init_R
        M = R[perm][:, perm]
        loss = loss_fn(
            M, weights_f32, acts_f32, targets, quantizer, g, row_chunk
        )
        loss_value = float(loss.item())
        iter_idx = step + 1
        if loss_value < best_loss:
            best_loss = loss_value
            best_iter = iter_idx
            best_R = R.detach().clone()
            best_M = M.detach().clone()
            iters_since_improvement = 0
        else:
            iters_since_improvement += 1
        loss.backward()
        optimizer.step()
        final_iter_loss = loss_value
        actual_steps_run = iter_idx
        if (
            early_stop_patience is not None
            and int(early_stop_patience) > 0
            and iters_since_improvement >= int(early_stop_patience)
        ):
            early_stopped_flag = True
            break

    elapsed = time.time() - start_time

    with torch.no_grad():
        R_final = best_R.detach()
        M_final = best_M.detach()
        identity = torch.eye(g, device=device, dtype=torch.float32)
        orth_err = float((R_final @ R_final.t() - identity).norm().item())

        # Final rotated score: re-evaluate at M_final so the reported number
        # exactly matches what the cache-fill will install. Should agree with
        # best_loss to within floating-point noise.
        final_loss = float(
            loss_fn(
                M_final, weights_f32, acts_f32, targets, quantizer, g, row_chunk
            ).item()
        )

        # Baseline (no rotation) under the same STE quantizer AND the same
        # loss kind as the solver — apples-to-apples with rotated_score.
        # For loss_kind="w4a4" we evaluate the same joint W4A4 loss at M=I;
        # for loss_kind="w_only" we use the legacy per-Linear weight-only
        # path (matches the prior baseline_score reported in the sidecar).
        if loss_kind == "w4a4":
            baseline_total = float(
                loss_fn(
                    identity, weights_f32, acts_f32, targets, quantizer, g, row_chunk
                ).item()
            )
        else:
            baseline_total = 0.0
            for t, w, x in zip(targets, weights_f32, acts_f32):
                w_q = quantizer(w, g)
                score = score_render_error(
                    w, w_q, x,
                    row_weights=t.row_weights,
                    row_chunk=row_chunk,
                )
                baseline_total = baseline_total + float(t.score_weight) * score

    baseline_score = float(baseline_total)
    rotated_score = float(final_loss)
    if baseline_score > 0.0:
        relative_gain = (baseline_score - rotated_score) / abs(baseline_score)
    else:
        relative_gain = 0.0

    # "Still improving" iff the best iter landed in the last 20% of the
    # iters actually run — the most useful signal for "should I have run
    # longer?" An n_iters=0 (or early-stop on iter 0) run reports
    # still_improving=False because there's no "last 20%" by definition.
    iters_actually_run = max(actual_steps_run, 1) if actual_steps_run > 0 else 0
    still_improving_threshold = max(1, int(0.8 * iters_actually_run))
    still_improving_flag = bool(
        iters_actually_run > 0 and best_iter >= still_improving_threshold
    )

    return ClusterRotationResult(
        R=R_final,
        permutation=perm,
        composed_matrix=M_final,
        baseline_score=baseline_score,
        rotated_score=rotated_score,
        relative_gain=float(relative_gain),
        solver_seconds=float(elapsed),
        orthogonality_err=float(orth_err),
        init_strategy=str(init_strategy),
        n_iters=int(actual_steps_run),  # iters actually run (may be < requested)
        best_iter=int(best_iter),
        final_iter_loss=float(final_iter_loss),
        still_improving=still_improving_flag,
        early_stopped=bool(early_stopped_flag),
    )


# ---------------------------------------------------------------------------
# Log emission — schemas as the single source of truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterDecisionRecord:
    """Per-fused-sibling-cluster decision record.

    Emitted as one JSON line per cluster during the joint search. Captures
    the rotation choice, all measured ``(rotation, format)`` candidates,
    the allocator's pick, and any render-gate decisions for downstream
    inspection.
    """

    cluster_key: str
    insertion_kind: str
    rotation: dict[str, Any]                  # ClusterRotationResult.to_log_dict()
    candidates: dict[str, dict[str, float]]   # {label: {fisher_mse, bpp}}
    allocator_pick: str
    render_gates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_key": self.cluster_key,
            "insertion_kind": self.insertion_kind,
            "rotation": dict(self.rotation),
            "candidates": {k: dict(v) for k, v in self.candidates.items()},
            "allocator_pick": self.allocator_pick,
            "render_gates": list(self.render_gates),
        }


@dataclass(frozen=True)
class ShipSummaryRecord:
    """Per-ship summary record.

    Captures the cross-cluster aggregate state of the final artifact:
    format distribution, rotation coverage, eval-suite metrics, and the
    deltas vs the no-rotation baseline that the production gate consumes.
    """

    model: str
    target_bpp: float
    actual_bpp: float
    format_distribution: dict[str, int]
    rotation_distribution: dict[str, int]
    insertion_coverage: dict[str, int]
    metrics: dict[str, Any]
    vs_baseline_no_rotation: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "target_bpp": float(self.target_bpp),
            "actual_bpp": float(self.actual_bpp),
            "format_distribution": dict(self.format_distribution),
            "rotation_distribution": dict(self.rotation_distribution),
            "insertion_coverage": dict(self.insertion_coverage),
            "metrics": dict(self.metrics),
            "vs_baseline_no_rotation": dict(self.vs_baseline_no_rotation),
        }


def emit_cluster_decision(
    record: ClusterDecisionRecord,
    output_path: Path | str,
    *,
    append: bool = True,
) -> None:
    """Append a per-cluster decision JSON line to ``output_path``.

    Output format is JSON-lines (one record per line). Created with
    sort_keys for determinism so diffs across runs are meaningful.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode) as f:
        f.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def emit_ship_summary(
    record: ShipSummaryRecord,
    output_path: Path | str,
) -> None:
    """Write the per-ship summary JSON to ``output_path``.

    Output is pretty-printed (indent=2) since this is a single record
    intended for human inspection as well as machine consumption.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")
