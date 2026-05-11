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
budget controlled by `PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS`. Candidate
thresholds are selected using the full local activation output-error score. A
candidate must clear `PRISMAQUANT_ACT_CLIP_SOLVER_MIN_GAIN` before it is
selected. The default is `0.0`: the 4B FourOverSix+PrismaClip run showed that
many individually tiny local-MSE wins can be collectively useful, so a nonzero
floor is an ablation knob rather than a production default. Cache metadata
records every threshold evaluation so convergence can be audited after the run.

An experimental holdout-veto gate is available with
`PRISMAQUANT_ACT_CLIP_SOLVER_HOLDOUT=1`. It selects thresholds by the full
local score, then requires the same rendered weight to improve deterministic
held-out activation rows. This is off by default: an .8B smoke on
2026-05-11 measured KL `0.34022548` with the holdout veto versus `0.12600227`
for the no-prewrite PrismaClip baseline.

The solver can optionally write the baseline NVFP4 rendering to
`ProductionWeightCache` when it scores the baseline. This is controlled by
`PRISMAQUANT_ACT_CLIP_SOLVER_PREWRITE_BASELINE=1` and is currently off by
default: .8B validation on 2026-05-11 regressed from KL `0.12600227` with
prewrite off to `0.21575844` with prewrite on.

That .8B result does not make no-prewrite universally safe. A Qwen3-4B
rerun on 2026-05-11 with the same FourOverSix+PrismaClip assignment,
`PRISMAQUANT_ACT_CLIP_SOLVER_PREWRITE_BASELINE=0`, and resident cache
prefetch regressed to KL `0.16388227`. The cache differed from the
known-good `0.056417932` cache in exactly three NVFP4 sidecars:

- `model.layers.8.self_attn.o_proj`
- `model.layers.14.mlp.gate_proj`
- `model.layers.14.mlp.up_proj`

Hardlink-patching only those three files back from the known-good cache
restored KL exactly to `0.056417932`, so the issue is PrismaClip's local
threshold selection/render stability for a few groups, not assignment,
bpp accounting, cache prefetch, or validation.

The holdout gate was added after that finding. It is not a proof of global KL
monotonicity, and the first .8B smoke regressed badly, so it remains an opt-in
diagnostic rather than part of PrismaClip's default acceptance rule.

## PrismaFisherClip

PrismaFisherClip is an experimental scorer for the same PrismaClip threshold
search. It does not add a new cache, runtime format, or export path. It reuses
the probe's h-detail `g2_per_token` vectors and scores each threshold as
Fisher-weighted local output error:

```text
E_t[ normalized_g2_t * ||X_t @ (W - W_rendered)^T||^2 ]
```

This targets the observed failure mode where a threshold can have a small
average local-MSE win while hurting the tokens that matter most to the loss.
PrismaFisherClip is enabled with `PRISMAFISHERCLIP=1` in `run-pipeline.sh`, or
with the lower-level `fisher_clip` production-cache lever plus an h-detail
directory. It forces `act_clip_solver` on and requires h-detail. It does not
enable Fisher-weighted GPTQ unless `fisher_gptq` / `FISHER_WEIGHTED_GPTQ=1` is
also set.

Default: off until it beats or matches fixed clipping on both .8B and 4B under
the same calibration contract.

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
- Baseline prewrite is not production-safe yet. A .8B isolation rerun showed
  `0.21575844` KL with prewrite on versus `0.12600227` with prewrite off, so
  prewrite remains opt-in.
- No-prewrite is also not sufficient as a 4B stability fix. The
  `/home/rob/dq-runs/qwen3-4b-prismaclip-no-prewrite-20260511T170732Z`
  rerun measured KL `0.16388227`. Restoring only the three differing NVFP4
  sidecars from the known-good scalar cache restored KL to `0.056417932`.
- The experimental holdout-veto gate is not a fix: an .8B smoke measured KL
  `0.34022548` versus `0.12600227` for the no-prewrite PrismaClip baseline.
  Keep it opt-in unless a later variant validates on both .8B and 4B.

## Naming

Activation clipping, calibration, and learned clipping have substantial prior
art. PrismaClip names this implementation's specific production contract:
fused production-error clipping inside PrismaQuant's per-Linear, vLLM-compatible
mixed-format pipeline.

Related prior art includes TensorRT calibration, ACIQ, PACT, and OmniQuant.
