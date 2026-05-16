"""Block-diagonal-at-G orthogonal rotation candidate search for microscale formats.

Applies a ``(G, G)`` orthogonal matrix ``R`` block-diagonally along the input
feature axis of a Linear. For NVFP4 ``G = 16`` so the rotation lives entirely
inside one microscale group; the microscale layout that NVFP4 already imposes
is therefore preserved exactly.

Runtime path
------------

vLLM 0.19+ ships a compressed-tensors transform mechanism that applies a
stored ``(G, G)`` matrix as block-diagonal-at-``head_dim`` via unflatten /
matmul / flatten on the input activation. No new vLLM kernel is required;
the artifact just needs ``transforms_config`` plus the matching weight files
written into ``input_transform`` / ``output_transform`` parameter slots. The
exporter writes those.

Math
----

For one input-axis G-block, the original computation is
``y_block = W[:, block] @ x[block]``. Under block rotation with ``R`` of size
``(G, G)``, runtime sees:

    x[block]_runtime  = R @ x[block]_original
    W_stored[:, block] = W[:, block] @ R^T          (so W_stored R = W)
    y_block_runtime    = Q(W_stored[:, block]) @ (R @ x[block])

In original coordinates the *effective* dequantized weight is
``Q(W R^T) R`` applied per G-block on the input axis. Search scores that
effective weight against the original ``W`` on the original activations using
``score_render_error`` (the same output-MSE objective AWQ uses).

The candidate set is intentionally small for v0: identity, Sylvester Hadamard
at ``G``, and a few random orthogonals seeded for reproducibility. Learned
optimization over Stiefel can come later if the static candidates leave gain
on the table.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch

from prismaquant.render_score import gate_render_candidate, score_render_error


DEFAULT_BLOCK_ROTATION_MIN_GAIN = 0.0
NVFP4_GROUP_SIZE = 16
MXFP8_GROUP_SIZE = 32


@dataclass(frozen=True)
class BlockRotationSearchTarget:
    """One Linear participating in a shared block-rotation candidate search."""

    name: str
    fmt: str
    weight: torch.Tensor
    activations: torch.Tensor
    group_size: int
    score_weight: float = 1.0
    row_weights: torch.Tensor | None = None


@dataclass
class BlockRotationSearchResult:
    matrix: torch.Tensor
    selected_label: str
    baseline_score: float
    best_score: float
    relative_gain: float
    n_candidates: int
    trace: list[dict[str, float | str | None]]
    gate_reason: str = "improved"


RenderRotatedFn = Callable[
    [int, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def sylvester_hadamard(
    group_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a normalized Sylvester Hadamard of size ``group_size``.

    ``group_size`` must be a power of two. The returned matrix satisfies
    ``H @ H.T == I``.
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
    """Random orthogonal ``(group_size, group_size)`` via QR of a Gaussian."""

    g = int(group_size)
    if g < 1:
        raise ValueError(f"group_size must be >= 1, got {g}")
    work_device = device if device is not None else torch.device("cpu")
    a = torch.randn(g, g, generator=generator, device=work_device, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    d = torch.diag(r).sign()
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q * d.unsqueeze(0)
    return q.to(dtype)


def apply_block_rotation_input(
    matrix: torch.Tensor, R: torch.Tensor
) -> torch.Tensor:
    """Right-multiply each G-block on the last axis of ``matrix`` by ``R``.

    Equivalent to ``matrix @ block_diag(R, R, ..., R)`` where the block size
    is ``R.shape[0]``. Works on weights ``[out, in]`` and activations
    ``[..., in]`` identically.
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
    """Compute the weight to be quantized when ``R`` is the runtime rotation.

    Runtime applies ``R`` to ``x`` per G-block. Storing ``W @ block_diag(R^T)``
    cancels it back out in the float-precision composition, so quantization
    error is the only deviation from the original function.
    """

    return apply_block_rotation_input(weight, R.t())


def effective_weight_for_scoring(
    rendered_rotated: torch.Tensor, R: torch.Tensor
) -> torch.Tensor:
    """Map a rendered rotated weight back to original input-axis coordinates.

    If ``rendered_rotated`` is ``Q(W R^T)``, returns ``Q(W R^T) @ block_diag(R)``,
    which is the weight that — applied to original activations — reproduces the
    runtime computation under the input-side rotation.
    """

    return apply_block_rotation_input(rendered_rotated, R)


_NVFP4_E2M1_LEVELS_TENSOR: torch.Tensor | None = None


def _nvfp4_codebook(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """NVFP4 E2M1 codebook for STE-rounded surrogate quantization."""

    global _NVFP4_E2M1_LEVELS_TENSOR
    if (
        _NVFP4_E2M1_LEVELS_TENSOR is None
        or _NVFP4_E2M1_LEVELS_TENSOR.device != device
        or _NVFP4_E2M1_LEVELS_TENSOR.dtype != dtype
    ):
        _NVFP4_E2M1_LEVELS_TENSOR = torch.tensor(
            [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
             0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            device=device,
            dtype=dtype,
        )
    return _NVFP4_E2M1_LEVELS_TENSOR


def _ste_rtn_nvfp4_per_group(
    weight: torch.Tensor, group_size: int = NVFP4_GROUP_SIZE
) -> torch.Tensor:
    """Per-group NVFP4 RTN with straight-through estimator for the rounding.

    Forward path matches per-group symmetric RTN against the E2M1 codebook
    with microscale = ``max(|w|_block) / 6``. Backward path skips the
    discrete codebook step (STE), so the operation is differentiable end to
    end through ``weight``.
    """
    if weight.dim() != 2:
        raise ValueError("STE RTN expects a 2D weight")
    rows, cols = weight.shape
    g = int(group_size)
    if cols % g != 0:
        raise ValueError(f"input width {cols} not divisible by group_size {g}")
    grouped = weight.reshape(rows, cols // g, g)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = max_abs / 6.0
    normalized = grouped / scale
    codebook = _nvfp4_codebook(weight.device, weight.dtype)
    distances = (normalized.unsqueeze(-1) - codebook).abs()
    nearest = codebook[distances.argmin(dim=-1)]
    quantized = normalized + (nearest - normalized).detach()
    return (quantized * scale).reshape(rows, cols)


def cayley_orthogonal(A: torch.Tensor) -> torch.Tensor:
    """Cayley map: skew-symmetric ``A`` -> orthogonal ``R = (I - A)^-1 (I + A)``.

    The mapping is a smooth bijection between the skew-symmetric Lie algebra
    and the special-orthogonal group (det +1). For ``A = 0`` the map returns
    ``I``. The matrix solve is differentiable via ``torch.linalg.solve``.
    """
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Cayley input must be a square matrix")
    g = int(A.shape[0])
    identity = torch.eye(g, device=A.device, dtype=A.dtype)
    A_skew = (A - A.t()) / 2
    return torch.linalg.solve(identity - A_skew, identity + A_skew)


def _inverse_cayley(R: torch.Tensor) -> torch.Tensor:
    """Inverse Cayley: orthogonal ``R`` -> skew-symmetric ``A``.

    Fails (returns zero) when ``R`` has ``-1`` as an eigenvalue (rare for
    well-conditioned R). Used to initialize the optimization at a non-trivial
    R while reparameterizing via skew-symmetric ``A``.
    """
    g = int(R.shape[0])
    identity = torch.eye(g, device=R.device, dtype=R.dtype)
    try:
        A = torch.linalg.solve(R + identity, R - identity)
    except RuntimeError:
        return torch.zeros_like(R)
    return (A - A.t()) / 2


def learn_block_rotation_cayley(
    targets: "Sequence[BlockRotationSearchTarget]",
    *,
    init_R: torch.Tensor,
    group_size: int = NVFP4_GROUP_SIZE,
    steps: int = 60,
    lr: float = 5e-3,
    score_chunk: int = 256,
) -> tuple[torch.Tensor, float]:
    """Optimize a ``(G, G)`` orthogonal via Cayley parameterization + STE RTN.

    The loss is the sum over targets of activation-weighted output MSE between
    the original weight and the dequantized rotated weight returned to original
    coordinates:

        loss(R) = Σ_t ||(W_t - Q(W_t R^T) R) x_t||²

    Q is the per-group symmetric NVFP4 RTN with straight-through estimator.
    The optimization runs ``steps`` Adam updates over the skew-symmetric
    parameter ``A``, with R = Cayley(A).

    Returns ``(R_final, final_loss)``. ``R_final`` is the optimized rotation
    in original (non-storage) coordinates — apply the same ``sqrt(G)``
    compensation as the static candidates when exporting.
    """
    if not targets:
        raise ValueError("learn_block_rotation_cayley requires targets")
    g = int(group_size)
    device = targets[0].weight.device

    A_init = _inverse_cayley(init_R.to(device=device, dtype=torch.float32))
    A = torch.nn.Parameter(A_init.detach().clone())
    optimizer = torch.optim.Adam([A], lr=float(lr))

    weights_f32 = [
        t.weight.detach().to(device=device, dtype=torch.float32) for t in targets
    ]
    acts_f32 = []
    for t in targets:
        x = t.activations.detach().to(device=device, dtype=torch.float32)
        x = x.reshape(-1, x.shape[-1])
        acts_f32.append(x)

    final_loss = float("inf")
    for _step in range(int(steps)):
        optimizer.zero_grad()
        R = cayley_orthogonal(A)
        total = torch.zeros((), device=device, dtype=torch.float32)
        for t, w, x in zip(targets, weights_f32, acts_f32):
            w_rot = apply_block_rotation_input(w, R.t())
            w_rot_q = _ste_rtn_nvfp4_per_group(w_rot, group_size=g)
            effective = apply_block_rotation_input(w_rot_q, R)
            diff_t = (w - effective).t().contiguous()
            err_sum = torch.zeros((), device=device, dtype=torch.float32)
            for start in range(0, x.shape[0], int(score_chunk)):
                y = x[start:start + int(score_chunk)] @ diff_t
                err_sum = err_sum + y.pow(2).sum()
            denom = max(1, int(x.shape[0]) * int(w.shape[0]))
            total = total + float(t.score_weight) * (err_sum / denom)
        total.backward()
        optimizer.step()
        final_loss = float(total.item())

    with torch.no_grad():
        R_final = cayley_orthogonal(A).detach()
    return R_final, final_loss


def _candidate_matrices(
    group_size: int,
    device: torch.device,
    *,
    n_random_orthogonal: int,
    seed: int,
) -> list[tuple[str, torch.Tensor]]:
    g = int(group_size)
    cands: list[tuple[str, torch.Tensor]] = []
    cands.append(("identity", torch.eye(g, device=device, dtype=torch.float32)))
    if g >= 2 and (g & (g - 1)) == 0:
        cands.append(
            (
                "sylvester_hadamard",
                sylvester_hadamard(g, device=device, dtype=torch.float32),
            )
        )
    n_random = max(0, int(n_random_orthogonal))
    if n_random > 0:
        gen = torch.Generator(device=device)
        for k in range(n_random):
            gen.manual_seed(int(seed) + k)
            cands.append(
                (
                    f"random_orthogonal_{k}",
                    random_orthogonal(
                        g,
                        generator=gen,
                        device=device,
                        dtype=torch.float32,
                    ),
                )
            )
    return cands


def search_block_rotation(
    targets: Sequence[BlockRotationSearchTarget],
    render_rotated: RenderRotatedFn,
    *,
    group_size: int = NVFP4_GROUP_SIZE,
    n_random_orthogonal: int = 4,
    seed: int = 0,
    min_gain: float = 0.0,
    row_chunk: int = 128,
    cayley_steps: int = 0,
    cayley_lr: float = 5e-3,
) -> BlockRotationSearchResult:
    """Pick the block-rotation candidate that minimizes output-MSE under Q.

    ``render_rotated(idx, w_rotated, R)`` must return ``Q(w_rotated)`` in the
    target format. The caller decides whether to score against original
    activations or pre-rotated activations (this routine evaluates against
    original ``activations`` on each ``BlockRotationSearchTarget`` and maps
    the rendered tensor back to original coordinates via
    ``effective_weight_for_scoring``).
    """

    if not targets:
        raise ValueError("block-rotation search requires at least one target")
    g = int(group_size)
    if g <= 0:
        raise ValueError("group_size must be > 0")
    cols = int(targets[0].weight.shape[1])
    if cols % g != 0:
        raise ValueError(
            f"input width {cols} not divisible by group_size {g}"
        )
    device = targets[0].weight.device
    for t in targets:
        if int(t.weight.shape[1]) != cols:
            raise ValueError("all block-rotation targets must share input width")
        if int(t.activations.shape[-1]) != cols:
            raise ValueError("activation width must equal weight input width")
        if int(t.group_size) != g:
            raise ValueError("all targets must share the search group_size")

    candidates = _candidate_matrices(
        g,
        device=device,
        n_random_orthogonal=n_random_orthogonal,
        seed=seed,
    )

    best_R = candidates[0][1]
    best_label = candidates[0][0]
    best_score = float("inf")
    baseline_score = float("inf")
    trace: list[dict[str, float | str | None]] = []

    def _score_candidate(R: torch.Tensor) -> float:
        with torch.no_grad():
            total = 0.0
            for idx, target in enumerate(targets):
                w = target.weight.detach().to(device=device, dtype=torch.float32)
                w_rotated = rotate_weight_for_storage(w, R)
                rendered = render_rotated(idx, w_rotated, R)
                rendered = rendered.to(device=device, dtype=torch.float32)
                effective = effective_weight_for_scoring(rendered, R)
                score = score_render_error(
                    w,
                    effective,
                    target.activations,
                    row_weights=target.row_weights,
                    row_chunk=row_chunk,
                )
                total += float(target.score_weight) * score
            return float(total)

    with torch.no_grad():
        for label, R in candidates:
            R = R.to(device=device, dtype=torch.float32)
            total = _score_candidate(R)
            if label == "identity":
                baseline_score = float(total)
            if total < best_score:
                best_score = float(total)
                best_R = R.detach().clone()
                best_label = label
            trace.append(
                {
                    "label": label,
                    "score": float(total),
                    "best_score": float(best_score),
                }
            )

    # Optional learned-rotation refinement: initialize at the best static
    # candidate and run a small number of Cayley/Stiefel optimization steps
    # against an STE-NVFP4 surrogate loss. The final R is re-scored using
    # the actual ``render_rotated`` so the gate stays comparable.
    if int(cayley_steps) > 0 and best_label != "identity":
        init_R = best_R.detach().to(device=device, dtype=torch.float32)
        R_learned, learned_loss = learn_block_rotation_cayley(
            targets,
            init_R=init_R,
            group_size=g,
            steps=int(cayley_steps),
            lr=float(cayley_lr),
        )
        # Re-score under the production renderer for apples-to-apples gate
        learned_score = _score_candidate(R_learned)
        trace.append(
            {
                "label": "cayley_learned",
                "score": float(learned_score),
                "best_score": float(min(best_score, learned_score)),
            }
        )
        if learned_score < best_score:
            best_score = float(learned_score)
            best_R = R_learned.detach().clone()
            best_label = "cayley_learned"

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
        best_R = candidates[0][1].to(device=device, dtype=torch.float32)
        best_label = "identity"
        best_score = float(baseline_score)
        relative_gain = 0.0

    return BlockRotationSearchResult(
        matrix=best_R.detach(),
        selected_label=best_label,
        baseline_score=float(baseline_score),
        best_score=float(best_score),
        relative_gain=float(relative_gain),
        n_candidates=len(candidates),
        trace=trace,
        gate_reason=gate_reason,
    )
