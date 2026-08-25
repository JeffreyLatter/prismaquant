"""Block-wise output matching for NVFP4 quantization (quality win #12).

Per-Linear scale optimization (the existing `_scale_sweep_nvfp4`) picks
each Linear's per-group scale to minimize that Linear's reconstruction
MSE. This ignores the FACT that downstream Linears in the same block
(attention or MLP/MoE) compose their errors. A small per-Linear
reconstruction error in q_proj can blow up after the attention dot
product with k_proj. Per-Linear optimization can't see this.

Block-wise output matching takes a calibration block-input, forwards it
through the FP16 block to get a reference output, then refines per-
Linear scales by greedy coordinate descent: for each Linear in the
block, pick the scale that minimizes the BLOCK's output MSE against
the FP16 reference. Captures inter-Linear interaction effects that
per-Linear MSE can't see.

The implementation here is an MVP greedy variant. Full Pareto-optimal
joint optimization over all Linears in a block would be combinatorial
(scale_grid^n_linears) — we instead iterate Linears in topological
order, choose each one's best scale given the current state of the
others, and (optionally) re-iterate until no improvement.

Decoupled from the streaming export so it can be tested in isolation.
Integration into export_native_compressed is via an env flag — when
PRISMAQUANT_BLOCK_OUTPUT_MATCH=1 and an activation cache is supplied,
each layer's attention block + MLP block run a block-output refinement
pass after the per-Linear scale_sweep.

Cost: per block, n_linears × scale_grid forward passes through the
block. For attention (4 Linears, scale_grid=16) on hidden=4096: ~64
forward passes × ~10 ms = ~0.6 sec/block. Across 43 layers × 2 blocks
= ~50 sec total on Spark. Acceptable.

Quality: expected ~0.05-0.10 PPL gain on top of per-Linear scale_sweep.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BlockSpec:
    """Declares the Linears that compose one transformer block.

    `linears`: ordered list of (qname, dequantize_fn) pairs. The
    dequantize_fn takes a per-group fp8 scale tensor and returns the
    quantized weight to assign to the Linear. This abstracts away
    NVFP4 / MXFP8_E4M3 / etc. format-specifics so the refiner doesn't need
    to know which format a particular Linear uses.

    `forward_fn`: callable `(input_tensor) -> output_tensor` that runs
    the block forward (using the current weights of the listed Linears).

    `scale_setter`: callable `(qname, group_scales) -> None` that
    installs a candidate set of group scales onto the named Linear's
    weight. Used by the refiner to test scale candidates.

    `scale_getter`: callable `(qname) -> group_scales` returning the
    Linear's current group scales (so we can revert if a candidate
    doesn't improve).
    """
    linears: list[str]
    forward_fn: callable
    scale_setter: callable
    scale_getter: callable


def make_attention_block_spec(layer_mod: nn.Module, layer_qname: str,
                              candidate_perturbations: tuple[float, ...]
                              = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
                              ) -> BlockSpec | None:
    """Build a BlockSpec for the attention block of a standard transformer
    layer.

    Linears: q_proj, k_proj, v_proj (input-side) and o_proj (output-side).
    Returns None if the layer doesn't have the expected attention
    structure (e.g., Mamba layer, or DSv4 compressor — those need
    architecture-specific spec builders).

    The BlockSpec uses LIVE module references via attribute lookup, so
    the refiner can install candidate weights in-place. Candidate scales
    are multiplicative perturbations of the current weight (the
    refinement objective is to find which perturbation scale gives the
    smallest block-output MSE; this is a discrete proxy for the proper
    NVFP4 per-group scale search but cheap enough to run greedily).
    """
    if not hasattr(layer_mod, "self_attn"):
        return None
    sa = layer_mod.self_attn
    o_attr = "o_proj" if hasattr(sa, "o_proj") else (
        "out_proj" if hasattr(sa, "out_proj") else None)
    if o_attr is None:
        return None
    required = ["q_proj", "k_proj", "v_proj"]
    for r in required:
        if not hasattr(sa, r):
            return None

    saved: dict[str, torch.Tensor] = {}
    for name in (*required, o_attr):
        saved[name] = getattr(sa, name).weight.data.clone()

    def setter(qname: str, scale_tensor: torch.Tensor):
        # qname is "q_proj" / "o_proj" / etc. — short name.
        lin = getattr(sa, qname)
        s = float(scale_tensor)
        lin.weight.data = saved[qname] * s

    def getter(qname: str) -> torch.Tensor:
        lin = getattr(sa, qname)
        # Recover scale by inverse — robust to fp dtype shenanigans.
        ref = saved[qname]
        # Use the max-|·| element to estimate scale. Divide by the
        # SIGNED value: it is the max-magnitude element by construction
        # (never ~0 for a real weight), and clamping a negative
        # denominator to 1e-12 would explode the recovered scale to
        # ~cur·1e12 whenever the max-|·| element is negative.
        idx = ref.abs().argmax()
        flat_ref = ref.reshape(-1)
        flat_cur = lin.weight.data.reshape(-1)
        ref_v = flat_ref[idx]
        if float(ref_v) == 0.0:
            # All-zero reference weight — no scale is recoverable.
            return torch.tensor(1.0)
        s = float(flat_cur[idx] / ref_v)
        return torch.tensor(s)

    def fwd(x: torch.Tensor) -> torch.Tensor:
        # Single-head-style forward: just exercise q/k/v/o through
        # softmax(QK^T)V → o_proj. This isn't faithful to the model's
        # multi-head attention, but it's sufficient for measuring
        # reconstruction sensitivity of the four Linears as a coupled
        # system (the qk-product is the inter-Linear coupling we want
        # to capture). Faithful reproduction would call the model's
        # actual attention forward — that's an integration upgrade.
        Q = sa.q_proj(x)
        K = sa.k_proj(x)
        V = sa.v_proj(x)
        d = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / (d ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return getattr(sa, o_attr)(attn @ V)

    return BlockSpec(
        linears=[*required, o_attr],
        forward_fn=fwd,
        scale_setter=setter,
        scale_getter=getter,
    )


def make_mlp_block_spec(layer_mod: nn.Module, layer_qname: str
                        ) -> BlockSpec | None:
    """Build a BlockSpec for a dense MLP block (gate_proj + up_proj +
    down_proj with SiLU activation, the Llama/Qwen pattern).

    Returns None for MoE layers — those need expert-aware spec builders
    that can refine the experts as a fused tensor or per-expert.
    """
    if not hasattr(layer_mod, "mlp"):
        return None
    mlp = layer_mod.mlp
    required = ["gate_proj", "up_proj", "down_proj"]
    for r in required:
        if not hasattr(mlp, r):
            return None

    saved: dict[str, torch.Tensor] = {}
    for name in required:
        saved[name] = getattr(mlp, name).weight.data.clone()

    def setter(qname: str, scale_tensor: torch.Tensor):
        lin = getattr(mlp, qname)
        s = float(scale_tensor)
        lin.weight.data = saved[qname] * s

    def getter(qname: str) -> torch.Tensor:
        lin = getattr(mlp, qname)
        ref = saved[qname]
        # Signed division by the max-|·| element (see the attention
        # getter): clamp_min on a negative denominator exploded the
        # recovered scale by ~1e12.
        idx = ref.abs().argmax()
        flat_ref = ref.reshape(-1)
        flat_cur = lin.weight.data.reshape(-1)
        ref_v = flat_ref[idx]
        if float(ref_v) == 0.0:
            return torch.tensor(1.0)
        s = float(flat_cur[idx] / ref_v)
        return torch.tensor(s)

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return mlp.down_proj(
            torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))

    return BlockSpec(
        linears=required,
        forward_fn=fwd,
        scale_setter=setter,
        scale_getter=getter,
    )


def block_output_mse(spec: BlockSpec, calib_input: torch.Tensor,
                     reference_output: torch.Tensor) -> float:
    """Forward the block with current weights and compute MSE against
    the FP16 reference output."""
    with torch.no_grad():
        out = spec.forward_fn(calib_input)
    diff = (out.float() - reference_output.float())
    return float(diff.pow(2).mean())


def refine_block_scales(spec: BlockSpec,
                        calib_input: torch.Tensor,
                        reference_output: torch.Tensor,
                        scale_candidates_per_linear: dict[str, list[torch.Tensor]],
                        max_passes: int = 2,
                        verbose: bool = False) -> float:
    """Greedy block-wise scale refinement.

    For each Linear in `spec.linears` (in topological order), for each
    candidate scale tensor in `scale_candidates_per_linear[qname]`,
    install the candidate, forward the block, measure MSE against
    `reference_output`, and keep the candidate with smallest MSE.

    Iterates over Linears `max_passes` times so a Linear's choice can
    be reconsidered after later Linears have been refined.

    Returns the final block-output MSE.
    """
    best_mse = block_output_mse(spec, calib_input, reference_output)
    if verbose:
        print(f"  initial block MSE = {best_mse:.6e}")

    for pass_idx in range(max_passes):
        improved = False
        for qname in spec.linears:
            candidates = scale_candidates_per_linear.get(qname, [])
            if not candidates:
                continue
            current_scales = spec.scale_getter(qname).clone()
            best_for_this = current_scales
            best_mse_for_this = best_mse
            for cand in candidates:
                spec.scale_setter(qname, cand)
                m = block_output_mse(spec, calib_input, reference_output)
                if m < best_mse_for_this:
                    best_mse_for_this = m
                    best_for_this = cand
            if best_mse_for_this < best_mse:
                spec.scale_setter(qname, best_for_this)
                best_mse = best_mse_for_this
                improved = True
                if verbose:
                    print(f"  pass {pass_idx} {qname}: "
                          f"MSE → {best_mse:.6e}")
            else:
                # Revert to current_scales (no candidate improved).
                spec.scale_setter(qname, current_scales)
        if not improved:
            if verbose:
                print(f"  pass {pass_idx}: no improvement — converged")
            break

    return best_mse
