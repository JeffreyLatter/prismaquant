"""Batched NVFP4 GPTQ + scale-sweep for the export hot path.

Per-Linear GPTQ on a 256-expert MoE layer fires ~770 sequential calls,
each with its own Cholesky + per-column update loop. Most of the
compute is the same shape — every expert's `gate_up_proj` has shape
`[2*intermediate, hidden]`, every `down_proj` has shape `[hidden,
intermediate]`. The per-call kernel-launch overhead and Python-side
coordination dominate the wall time.

This module batches GPTQ and scale-sweep across same-shape Linears so
the GPU sees `(E_chunk, out, in)` tensors instead of `E_chunk` separate
small ops. Memory is bounded by chunking the expert dimension; the
default `E_chunk = 32` keeps the activation-covariance peak under
~5 GB even for `in_features = 6144`.

Expected speedup on a 256-expert MiniMax-style layer: 3–8× wall on
the GPTQ + scale-sweep portion, depending on how launch-overhead-bound
the per-Linear path was. The compute itself is identical to the
per-Linear path — we are amortizing kernel launches, not skipping work.

The math is bitwise-equivalent to the per-Linear functions in
`export_native_compressed`. Equivalence tests live in
`tests/test_export_batched_gptq.py`.
"""
from __future__ import annotations

import torch

# Re-import codebook helpers from the main export module to avoid
# duplicating the FP4 grid definition.
from .export_native_compressed import (
    _activation_matrix_for_gptq,
    _nvfp4_codebook,
    _rtn_dequant_nvfp4,
    _round_to_codebook,
    _select_nvfp4_group_scales,
    NVFP4_MAX,
    FP8_E4M3_MAX,
)


def _build_H_stack(
    activations_list: list[torch.Tensor],
    in_features: int,
    device: torch.device,
    damp: float = 0.01,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    row_weights_list: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a stacked activation covariance tensor across E Linears.

    Each entry of ``activations_list`` is a per-Linear activation
    tensor (any shape collapsing to `[*, in_features]`). Returns:

      - ``H_stack`` of shape ``[E, in, in]`` with damping already added,
        ready for batched Cholesky.
      - ``dead_mask`` of shape ``[E, in]`` flagging columns whose H
        diagonal was non-positive (so the caller can zero those weight
        columns).

    Computing H per-Linear in a small Python loop is cheap relative to
    the column-update loop further down; we don't try to batch the
    `X.T @ X` itself because per-Linear `X` shapes vary in row count
    (routed-token-count differs per expert).
    """
    E = len(activations_list)
    H_stack = torch.zeros(
        (E, in_features, in_features), dtype=torch.float32, device=device,
    )
    dead_mask = torch.zeros((E, in_features), dtype=torch.bool, device=device)
    for e in range(E):
        a = activations_list[e]
        if a is None or a.numel() == 0:
            # No activations for this Linear — fall back to identity
            # so the batched Cholesky doesn't fail. Caller will see all
            # columns "dead" and weights zero out.
            H_stack[e] = torch.eye(
                in_features, dtype=torch.float32, device=device)
            dead_mask[e] = True
            continue
        # v23 fix: explicitly move activations to the target device
        # before the matmul. _LazyActivationCache.get() returns CPU
        # tensors; the per-Linear path inherits this and runs the
        # H = X^T X matmul on CPU (slow). Batching across E experts
        # on CPU doesn't help — the algorithm is bandwidth-bound on
        # the host. Moving X to the GPU lets the bmm/matmul run as
        # a real CUDA kernel and is what delivers the projected
        # speedup at production scale.
        row_weights = (
            row_weights_list[e]
            if row_weights_list is not None and e < len(row_weights_list)
            else None
        )
        X = _activation_matrix_for_gptq(
            a,
            in_features,
            device=device,
            clip_threshold=clip_threshold,
            clip_rescale=clip_rescale,
            row_weights=row_weights,
        )
        H = X.t() @ X
        diag_mean = torch.diagonal(H).mean().clamp_min(1e-12)
        H.diagonal().add_(damp * diag_mean)
        # Dead columns: zero diagonal (can happen if the activation is
        # all zeros on a particular channel).
        dead = torch.diagonal(H) <= 0
        if dead.any():
            H[dead, dead] = 1.0
            dead_mask[e] = dead
        H_stack[e] = H
    return H_stack, dead_mask


def _batched_round_to_codebook(in_grid: torch.Tensor) -> torch.Tensor:
    """Round any-rank tensor to NVFP4 codebook indices (same encoding
    as `_round_to_codebook` in the per-Linear path)."""
    # The single-Linear `_round_to_codebook` already operates element-
    # wise on its input — re-using it here is a no-op cost since it is
    # itself batched over the leading dimensions.
    return _round_to_codebook(in_grid)


def gptq_obs_rounding_nvfp4_batched(
    weights: torch.Tensor,
    activations_list: list[torch.Tensor],
    *,
    group_size: int = 16,
    damp: float = 0.01,
    global_real_overrides: torch.Tensor | None = None,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    row_weights_list: list[torch.Tensor | None] | None = None,
    expert_chunk: int = 32,
) -> torch.Tensor:
    """Batched NVFP4 GPTQ across E same-shape Linears.

    Args:
      weights: ``[E, out, in]`` float32 stack of weight matrices. All
        Linears in the batch must share `(out, in)`.
      activations_list: length-E list of per-Linear activation tensors,
        each shape ``[*, in]`` (T may differ per Linear for routed-MoE
        experts). Use a zero-numel tensor for Linears that should be
        treated as "no activations available" (dead columns).
      group_size: NVFP4 group size (always 16 in production).
      damp: ridge applied to the activation covariance diagonal for
        Cholesky stability. Default 0.01 matches the per-Linear path.
      global_real_overrides: optional ``[E]`` tensor of per-Linear
        global_real values from fused-sibling joint global. ``None``
        falls back to per-Linear computation.
      expert_chunk: maximum E processed in one batched op. Caps GPU
        memory at roughly ``E_chunk × in² × 4 B`` (covariance) +
        ``E_chunk × out × in × 4 B`` (weight stack). Default 32 keeps
        peak under ~5 GB for in=6144 / out=3072.

    Returns:
      ``[E, out, in]`` float32 stack of dequantized error-propagated
      weights, ready for the standard NVFP4 packer.
    """
    if weights.dim() != 3:
        raise ValueError(f"weights must be [E, out, in]; got {weights.shape}")
    E, out_features, in_features = weights.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"GPTQ requires group_size={group_size} ∤ {in_features}")
    if len(activations_list) != E:
        raise ValueError(
            f"len(activations_list)={len(activations_list)} ≠ E={E}")
    if global_real_overrides is not None and global_real_overrides.numel() != E:
        raise ValueError(
            f"global_real_overrides must have shape [E={E}]; "
            f"got {global_real_overrides.shape}")

    device = weights.device
    cb = _nvfp4_codebook(device, dtype=torch.float32)
    out_buf = torch.empty_like(weights)

    for e_start in range(0, E, expert_chunk):
        e_end = min(e_start + expert_chunk, E)
        Ec = e_end - e_start
        W = weights[e_start:e_end].clone()  # [Ec, out, in]
        # 1. Build H_stack + dead_mask for this E-chunk.
        H_stack, dead_mask = _build_H_stack(
            activations_list[e_start:e_end],
            in_features,
            device,
            damp=damp,
            clip_threshold=clip_threshold,
            clip_rescale=clip_rescale,
            row_weights_list=(
                row_weights_list[e_start:e_end]
                if row_weights_list is not None else None
            ),
        )
        # Zero out dead weight columns up front (matches per-Linear path).
        W = torch.where(
            dead_mask.unsqueeze(1).expand_as(W), torch.zeros_like(W), W,
        )

        # 2. Batched Cholesky + cholesky_inverse + upper Cholesky.
        # Failure handling: any single Linear with a degenerate H aborts
        # the batched call. v26 retries per-Linear so only the actually-
        # bad ones lose GPTQ — the rest still get the activation-aware
        # update. The previous behavior reverted the ENTIRE chunk to
        # unchanged weights, which on a 32-expert chunk is a 32× quality
        # loss for a single bad Linear.
        try:
            L = torch.linalg.cholesky(H_stack)            # [Ec, in, in]
            Hinv = torch.cholesky_inverse(L)              # [Ec, in, in]
            U = torch.linalg.cholesky(Hinv, upper=True)   # [Ec, in, in]
            failed_mask = torch.zeros(Ec, dtype=torch.bool, device=device)
        except Exception:
            # Per-Linear retry. Build U one-Linear-at-a-time and mark
            # any failures; the column-update loop below will preserve
            # the input weight for failed Linears.
            U = torch.zeros_like(H_stack)
            Hinv = torch.zeros_like(H_stack)
            failed_mask = torch.zeros(Ec, dtype=torch.bool, device=device)
            for j in range(Ec):
                try:
                    Lj = torch.linalg.cholesky(H_stack[j])
                    Hinv[j] = torch.cholesky_inverse(Lj)
                    U[j] = torch.linalg.cholesky(Hinv[j], upper=True)
                except Exception:
                    failed_mask[j] = True
            n_failed = int(failed_mask.sum().item())
            if n_failed:
                # Identity-ish U for failed Linears so the bmm in the
                # propagation step doesn't NaN; combined with the
                # per-Linear preserve-input we apply at the end of each
                # block, the failed Linear's weight passes through as
                # the un-error-propagated original (NVFP4 RTN at pack
                # time).
                eye = torch.eye(in_features, device=device, dtype=torch.float32)
                for j in range(Ec):
                    if failed_mask[j]:
                        U[j] = eye

        # 3. Per-Linear global_real (or override).
        if global_real_overrides is not None:
            global_real = global_real_overrides[e_start:e_end].to(
                device, dtype=torch.float32).clamp_min(1e-12)  # [Ec]
        else:
            grouped_full = W.reshape(
                Ec, out_features, in_features // group_size, group_size,
            )
            s_g_real_full = _select_nvfp4_group_scales(grouped_full)
            global_real = (
                s_g_real_full.reshape(Ec, -1).amax(dim=-1) / FP8_E4M3_MAX
            ).clamp_min(1e-12)  # [Ec]
        global_real_b = global_real.view(Ec, 1, 1)  # [Ec, 1, 1]

        U_diag = torch.diagonal(U, dim1=-2, dim2=-1)  # [Ec, in]

        # 4. Block-by-block column update. Within a block, batch over
        # the expert dimension so each column update issues one bmm
        # for all Ec experts instead of Ec separate matmuls.
        for block_start in range(0, in_features, group_size):
            block_end = min(block_start + group_size, in_features)
            block = W[:, :, block_start:block_end]      # [Ec, out, gs]

            # Per-Linear per-row max within the block → scale.
            s_g_real = _select_nvfp4_group_scales(block).unsqueeze(-1)
            fp8_scale_real = (s_g_real / global_real_b).clamp(0, FP8_E4M3_MAX)
            eff_scale = (fp8_scale_real * global_real_b).clamp_min(1e-12)
            in_grid = (block / eff_scale).clamp(-NVFP4_MAX, NVFP4_MAX)
            fp4_idx = _batched_round_to_codebook(in_grid)
            abs_idx = fp4_idx & 0x7
            sign = -((fp4_idx >> 3).to(torch.float32) * 2 - 1)
            q_vals = sign * cb[abs_idx]                  # [Ec, out, gs]
            block_dq = q_vals * eff_scale
            block_err = block - block_dq                 # [Ec, out, gs]

            # 5. Propagate error to the remaining columns.
            if block_end < in_features:
                U_block_diag = U_diag[:, block_start:block_end].clamp_min(
                    1e-12)                              # [Ec, gs]
                U_offdiag = U[:, block_start:block_end, block_end:]
                # ^ [Ec, gs, rest]
                # err_scaled = block_err / U_block_diag (broadcast on out dim)
                err_scaled = block_err / U_block_diag.unsqueeze(1)
                # Batched matmul across experts: each Ec gets its own
                # [out, gs] @ [gs, rest] = [out, rest].
                prop = torch.bmm(err_scaled, U_offdiag)  # [Ec, out, rest]
                W[:, :, block_end:] = W[:, :, block_end:] - prop

            W[:, :, block_start:block_end] = block_dq

        # v26 per-Linear failure handling: any Linear whose Cholesky
        # failed gets the un-error-propagated input weight back. The
        # column-update loop above touched W[failed_idx] with identity-
        # like U, so the propagation was a no-op there — but block_dq
        # still ran. To avoid even that pass for failed Linears, restore
        # them to weights[e_start:e_end][failed_idx]. Downstream NVFP4
        # pack will RTN-quantize as if no GPTQ ran.
        if failed_mask.any():
            failed_idx = failed_mask.nonzero(as_tuple=True)[0]
            for j in failed_idx.tolist():
                override = (
                    global_real[j]
                    if global_real_overrides is not None
                    else None
                )
                W[j] = _rtn_dequant_nvfp4(
                    weights[e_start + j],
                    group_size=group_size,
                    global_real_override=override,
                )

        out_buf[e_start:e_end] = W

    return out_buf


def scale_sweep_nvfp4_batched(
    weights: torch.Tensor,
    activations_list: list[torch.Tensor],
    *,
    reference_weights: torch.Tensor,
    group_size: int = 16,
    global_real_overrides: torch.Tensor | None = None,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    row_weights_list: list[torch.Tensor | None] | None = None,
    grid: int = 32,
    span: tuple[float, float] = (0.5, 1.5),
    expert_chunk: int = 32,
) -> torch.Tensor:
    """Batched NVFP4 scale sweep across E same-shape Linears.

    Same closed-form joint-(scale, rounding) search as the per-Linear
    `_scale_sweep_nvfp4`, but the per-group MSE evaluation is run
    across all E in a chunk in one pass. The expensive intermediate
    ``[chunk, n_g, grid, gs, len(cb)]`` peak is bounded by the existing
    row-chunking in the per-Linear path; we add an outer expert chunk
    so memory stays under control on 256-expert layers.

    Args:
      weights: ``[E, out, in]`` post-GPTQ (or pre-pass) weights.
      activations_list: length-E list, same semantics as the GPTQ path.
      reference_weights: ``[E, out, in]`` pre-pass weights (used as the
        MSE baseline). Pass ``weights`` here when no GPTQ was run.
      grid / span: scale sweep bounds and resolution.
      global_real_overrides / expert_chunk: same semantics as the GPTQ
        batched path.
    """
    if weights.shape != reference_weights.shape:
        raise ValueError(
            f"scale-sweep batched: weights {tuple(weights.shape)} ≠ "
            f"reference {tuple(reference_weights.shape)}")
    E, out_features, in_features = weights.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"scale-sweep requires group_size={group_size} ∤ {in_features}")

    device = weights.device
    cb_pos = _nvfp4_codebook(device, dtype=torch.float32)
    cb = torch.cat([-cb_pos.flip(0), cb_pos[1:]], dim=0)  # [15] signed
    n_g = in_features // group_size
    out_buf = torch.empty_like(weights)

    for e_start in range(0, E, expert_chunk):
        e_end = min(e_start + expert_chunk, E)
        Ec = e_end - e_start
        W_in = weights[e_start:e_end].to(torch.float32).contiguous()
        W_ref = reference_weights[e_start:e_end].to(
            torch.float32).contiguous()

        # Per-Linear column importance from cached activations.
        col_imp = torch.empty(
            (Ec, in_features), device=device, dtype=torch.float32)
        for j in range(Ec):
            a = activations_list[e_start + j]
            if a is None or a.numel() == 0:
                col_imp[j] = 1.0  # no activation signal → uniform
                continue
            # Same v23 device-fix as in the GPTQ path: move X to the
            # target device so the per-column statistics run as CUDA
            # ops rather than on the host.
            row_weights = (
                row_weights_list[e_start + j]
                if row_weights_list is not None
                and (e_start + j) < len(row_weights_list)
                else None
            )
            a32 = _activation_matrix_for_gptq(
                a,
                in_features,
                device=device,
                clip_threshold=clip_threshold,
                clip_rescale=clip_rescale,
                clip_quantile=0.0 if clip_threshold is None else None,
                row_weights=row_weights,
            )
            col_imp[j] = a32.pow(2).mean(dim=0).clamp_min(1e-12)
        col_imp_g = col_imp.reshape(Ec, 1, n_g, group_size)  # [Ec, 1, n_g, gs]

        ref_g = W_ref.reshape(Ec, out_features, n_g, group_size)
        in_g = W_in.reshape(Ec, out_features, n_g, group_size)
        s_g_real = _select_nvfp4_group_scales(ref_g)
        if global_real_overrides is not None:
            global_real = global_real_overrides[e_start:e_end].to(
                device, dtype=torch.float32).clamp_min(1e-12)
        else:
            global_real = (
                s_g_real.reshape(Ec, -1).amax(dim=-1) / FP8_E4M3_MAX
            ).clamp_min(1e-12)
        fp8_scale_real = (
            s_g_real / global_real.view(Ec, 1, 1)
        ).clamp(0, FP8_E4M3_MAX)
        eff_scale0 = (
            fp8_scale_real * global_real.view(Ec, 1, 1)
        ).unsqueeze(-1).clamp_min(1e-12)  # [Ec, out, n_g, 1]

        init_mse = (col_imp_g * (ref_g - in_g).pow(2)).sum(dim=-1)
        # ^ [Ec, out, n_g]

        mults = torch.linspace(
            span[0], span[1], grid, device=device, dtype=torch.float32)

        # Chunk over the OUT dimension to bound the
        # ``[Ec, out_chunk, n_g, grid, gs, 15]`` intermediate.
        bytes_per_row = n_g * grid * group_size * cb.numel() * 4
        out_chunk = max(
            1, (2 * 1024 * 1024 * 1024) // max(1, Ec * bytes_per_row))
        out_chunk = min(out_features, int(out_chunk))

        result_g = torch.empty_like(ref_g)
        for r0 in range(0, out_features, out_chunk):
            r1 = min(r0 + out_chunk, out_features)
            scales_c = (
                eff_scale0[:, r0:r1].squeeze(-1).unsqueeze(-1) * mults
            )  # [Ec, oc, n_g, grid]
            ref_c = ref_g[:, r0:r1]  # [Ec, oc, n_g, gs]
            in_c = in_g[:, r0:r1]
            init_c = init_mse[:, r0:r1]

            # gexp: [Ec, oc, n_g, 1, gs]; sexp: [Ec, oc, n_g, grid, 1]
            gexp = ref_c.unsqueeze(3)
            sexp = scales_c.unsqueeze(4)
            v = gexp / sexp  # [Ec, oc, n_g, grid, gs]
            d = (v.unsqueeze(-1) - cb).abs()  # [Ec, oc, n_g, grid, gs, 15]
            idx = d.argmin(dim=-1)
            del d, v
            Wq_cand = cb[idx] * sexp  # [Ec, oc, n_g, grid, gs]
            del idx
            # col_imp_g is [Ec, 1, n_g, gs]; broadcast over (oc, grid).
            # Add a "grid" axis at position 3 → [Ec, 1, n_g, 1, gs] which
            # multiplies cleanly with [Ec, oc, n_g, grid, gs].
            err = col_imp_g.unsqueeze(3) * (gexp - Wq_cand).pow(2)
            mse = err.sum(dim=-1)  # [Ec, oc, n_g, grid]
            del err
            best = mse.argmin(dim=-1)  # [Ec, oc, n_g]
            bidx = best.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, 1, group_size)
            chosen_Wq = Wq_cand.gather(3, bidx).squeeze(3)
            chosen_mse = mse.gather(3, best.unsqueeze(-1)).squeeze(-1)
            del Wq_cand, mse, best, bidx

            use_new = chosen_mse < init_c
            result_g[:, r0:r1] = torch.where(
                use_new.unsqueeze(-1).expand(-1, -1, -1, group_size),
                chosen_Wq,
                in_c,
            )

        out_buf[e_start:e_end] = result_g.reshape(
            Ec, out_features, in_features)

    return out_buf
