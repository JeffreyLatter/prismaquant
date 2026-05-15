# ReSpinQuant Paper-Fidelity Audit — 2026-05-13

Source: https://arxiv.org/abs/2604.11080

## Paper Requirements

ReSpinQuant is not just a low-rank residual adapter. The paper's method has
four essential parts:

- Train distinct full layer-wise orthogonal rotations for the attention and
  FFN blocks, initialized from Hadamard matrices and optimized with the Cayley
  optimizer.
- Preserve the paper's rotation topology: per-layer residual/input-output
  rotations for MHSA and FFN, an intermediate attention rotation such as the
  value/output-projection rotation, and the structured Fast Hadamard rotations
  used for specific online activation flows.
- Fuse those rotations into the surrounding Linear weights offline.
- For residual-basis mismatches, compute `T = R_out R_in^T`, take the top-r
  singular subspace of `T - I`, polar-orthogonalize `Q^T T Q`, and serve the
  residual transition as `x -> x + Q(R_sub - I)Q^T x`.
- Quantize after transformation optimization. The paper uses GPTQ plus fixed
  weight clipping, 100 optimization steps, cosine LR, WikiText-2 calibration,
  and rank `r=32` residual-transition approximation by default.

## Current PrismaQuant State

The standalone vLLM plugin is a valid runtime substrate for the paper's
low-rank residual transition because it applies:

```text
x -> x + (x @ U) @ V
```

For the paper approximation, PrismaQuant now represents this as `U=Q` and
`V=(R_sub - I)Q^T` in row-vector convention.

The `.8B` artifact tested on 2026-05-13 was not paper-faithful ReSpinQuant. It
used an untrained rank-16 alternating disjoint-Givens basis only to validate
runtime hooks and attribution. That result must not be used to conclude that
ReSpinQuant itself fails on NVFP4.

## Code Changes From This Audit

- `tools/create_respin_equivalent_variant.py` now exposes the paper SVD/polar
  residual-transition tensorization with `--transition-mode paper-svd`.
- `prismaquant/respinquant_core.py` contains the reusable paper-family math:
  Hadamard initialization, Cayley orthogonal updates, activation fake
  quantization, residual transition construction, and SVD/polar adapter
  tensors.
- `tools/train_respinquant_rotations.py` is a GPU-only rotation trainer. It
  trains Hadamard-initialized dense rotations with CE loss and activation fake
  quantization, then writes `respin_rotations.pt` plus metadata.
- `tools/create_respin_equivalent_variant.py --rotation-checkpoint ...` can
  consume that checkpoint, rotate weights with the trained bases, rotate the
  embedding table when needed, and write SVD/polar residual adapters between
  consecutive bases.
- Artifact metadata now records `paper_faithful=false`,
  `basis_source=random_disjoint_givens_untrained`, and the transition
  approximation mode.
- Tests cover the SVD/polar residual-transition math and the plugin `U,V`
  representation.

The trainer's first topology is `single_boundary_basis`: one learned residual
basis per decoder-layer input. That is a SpinQuant/ReSpin-family trained
rotation checkpoint, but it is not the full paper topology. Metadata therefore
sets `paper_faithful=false`.

## Missing For A True ReSpinQuant Benchmark

Before running a paper-faithful quality benchmark, complete the topology pass:

1. Map the trained rotations to Qwen3.5/Qwen3.6 attention, linear-attention,
   and FFN block boundaries without hardcoding Qwen-specific assumptions.
2. Include the paper's intermediate attention rotation and decide whether the
   Fast Hadamard online rotations are compatible with our vLLM/no-custom-kernel
   deployment contract.
3. Fuse trained rotations into weights, write rank-32 SVD/polar residual
   adapters, then re-run probe, cost, allocator, production cache, KL, and vLLM
   validation.

Until those steps exist, ReSpinQuant remains a research/plugin path, not a
production recipe lever.
