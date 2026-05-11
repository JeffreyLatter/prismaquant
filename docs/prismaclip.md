# PrismaClip

PrismaClip is PrismaQuant's production-rendered NVFP4 activation clipping
solver.

It is not a new runtime format and does not require a custom kernel. It chooses
one scalar render-time activation clamp for each eligible Linear or
fused-sibling group, then stores only the selected thresholds in
`ProductionWeightCache` metadata. Export consumes those thresholds through the
normal compressed-tensors path.

## What It Optimizes

For each candidate threshold, PrismaClip renders the actual production NVFP4
weight path:

```text
GPTQ + damp sweep + scale sweep + FourOverSix + candidate activation clamp
```

It scores the rendered candidate on original, unclipped calibration activations
using local output MSE. This prevents the solver from appearing to win simply
because it hid outliers from its own evaluator.

The current implementation searches in log space with a small evaluation
budget controlled by `PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS`. A candidate must
clear `PRISMAQUANT_ACT_CLIP_SOLVER_MIN_GAIN` (default `0.002`) before it is
selected; this avoids accepting numerically tiny local-MSE wins that can move
end-to-end KL in the wrong direction.

## Scope

- Format: NVFP4.
- Granularity: per Linear or fused-sibling group.
- Cache path: existing `ProductionWeightCache`.
- Runtime: no additional serving mechanism beyond the exported activation
  scale metadata.
- Default: off unless `PRISMAQUANT_ACT_CLIP_SOLVER=1`.

## Naming

Activation clipping, calibration, and learned clipping have substantial prior
art. PrismaClip names this implementation's specific production contract:
fused production-error clipping inside PrismaQuant's per-Linear, vLLM-compatible
mixed-format pipeline.

Related prior art includes TensorRT calibration, ACIQ, PACT, and OmniQuant.
