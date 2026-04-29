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
    NVFP4 / MXFP8 / etc. format-specifics so the refiner doesn't need
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
