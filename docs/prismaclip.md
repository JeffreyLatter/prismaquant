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
clear `PRISMAQUANT_ACT_CLIP_SOLVER_MIN_GAIN` before it is selected. The default
is `0.0`: the 4B FourOverSix+PrismaClip run showed that many individually tiny
local-MSE wins can be collectively useful, so a nonzero floor is an ablation
knob rather than a production default. Cache metadata records every threshold
evaluation so convergence can be audited after the run.

The solver writes the baseline NVFP4 rendering to `ProductionWeightCache` when
it scores the baseline. If no threshold clears the gain floor, the final cache
fill reuses that prewritten baseline instead of rendering the same Linear a
second time.

Same-shape fused groups can also try the existing batched NVFP4
GPTQ/scale-sweep path when `PRISMAQUANT_ACT_CLIP_SOLVER_BATCHED=1`. This is
currently opt-in: a 4B validation run on 2026-05-11 improved over the broken
batched damp selector but still measured `0.060248224` KL versus the scalar
`0.056417932` reference, so scalar rendering remains the production default.

## Scope

- Format: NVFP4.
- Granularity: per Linear or fused-sibling group.
- Cache path: existing `ProductionWeightCache`.
- GPU path: scalar CUDA rendering by default; same-shape batched rendering is
  opt-in while parity is under validation.
- Runtime: no additional serving mechanism beyond the exported activation
  scale metadata.
- Default: off unless `PRISMAQUANT_ACT_CLIP_SOLVER=1`.

## Validation Notes

4B rerun, 2026-05-11:

- Reference scalar FourOverSix+PrismaClip cache:
  `/home/rob/dq-runs/fouroversix-smoke-20260510T225344Z/qwen3-4b`,
  KL `0.056417932`.
- `PRISMAQUANT_ACT_CLIP_SOLVER_MIN_GAIN=0.002` selected only 30 qnames and
  regressed to KL `0.094699124`.
- Fixed batched damp-sweep evaluation selected 161 qnames and improved to KL
  `0.060248224`, but did not match scalar quality, so
  `PRISMAQUANT_ACT_CLIP_SOLVER_BATCHED` remains opt-in.
- Baseline prewrite engaged safely in both reruns, with 225 NVFP4 entries
  prewritten by the solver before final cache fill.

## Naming

Activation clipping, calibration, and learned clipping have substantial prior
art. PrismaClip names this implementation's specific production contract:
fused production-error clipping inside PrismaQuant's per-Linear, vLLM-compatible
mixed-format pipeline.

Related prior art includes TensorRT calibration, ACIQ, PACT, and OmniQuant.
