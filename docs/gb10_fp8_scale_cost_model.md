# Design note — fp8_e4m3-scale NVFP4 cost-model migration

Status: **Future work, not scheduled.** Captured 2026-05-17 after the
GB10 hardware FP4 GEMM A/B (see `docs/gb10_fp4_gemm.md`).

## Problem

PrismaQuant's NVFP4 cost model
(`format_registry._make_rtn("fp4_e2m1", 16, mx_scale=False)`) uses
**continuous fp32 block scales**. Every measurement loop computes NVFP4
outputs that way: the cost step (`measure_quant_cost` batched/unbatched
bmm), the perturbed-x measurement (`perturbed_x_cache` reference +
Triton fused kernel), and frontier-selection KL.

The **served artifact** — the compressed-tensors export that vLLM runs —
uses **fp8_e4m3 block scales** + a per-tensor fp32 global. The GB10
hardware FP4 tensor cores (`flashinfer.mm_fp4`) can *only* consume
fp8_e4m3 scales; there is no hardware path for continuous fp32 scales.

This mismatch is exactly why the 2026-05-17 FP4-GEMM A/B failed:
dropping `mm_fp4` into the measurement loop shifted measured KL because
it changed the numerical method (continuous-fp32 → fp8_e4m3 block
scales). That is not a bug — it is two different number systems being
compared.

Consequence: the GB10 tensor cores cannot accelerate any
cost-model-faithful stage, **including the single hottest GPU loop**
(`measure_quant_cost.measure_batched_gpu`). The hardware path is locked
out by a numerics wall, not by a performance limit.

## Proposal

Migrate the NVFP4 cost model from continuous fp32 block scales to
fp8_e4m3 block scales — make the surrogate use the *same* number system
as the served artifact.

Effect:

- `flashinfer.mm_fp4` becomes the **faithful** GEMM everywhere — cost
  step, measurement loop, validation all become eligible for hardware
  acceleration with zero numerics drift.
- The surrogate stops being marginally optimistic (13.31% vs the
  artifact's 13.44% error in the micro-benchmark) and instead matches
  the artifact.
- The ~10% A/B gap disappears: there is no longer a method mismatch.

## Touch points

- `format_registry._make_rtn` / the NVFP4 `FormatSpec` — add or switch
  to an fp8_e4m3-block-scale RTN variant.
- `measure_quant_cost` (`measure_batched_gpu`, `measure_unbatched`) —
  the `_batched_quantize` helpers.
- `perturbed_x_cache` reference path + the Triton fused kernel
  (`kernels/nvfp4_fused.py`) — the kernel dequants continuous fp32
  scales today; it would need an fp8_e4m3-scale variant, or the path
  routes directly to `mm_fp4`.
- `production_weight_cache` / `export_native_compressed` already use
  fp8_e4m3 — they become the reference, not the outliers.

## Risk & gate

This is a numerical-method change to the **core cost model**, so the
`docs/design_guidelines.md` measurement-discipline gate applies. It
requires a KL/bpp A/B across the standard .8B / 4B / 27B validation set
before adoption: rebuild probe → cost → allocate → export under both
scale systems and compare final exported-artifact KL at matched bpp.
Adopt only if KL is preserved or improved.

Expected outcome: KL ~neutral (the exported artifact is unchanged; only
the surrogate's accuracy improves), with the allocator making slightly
different per-layer choices because per-layer costs shift. The win is
speed + faithfulness, not compression.

## Payoff estimate

The cost step is the wall-time-dominant GPU phase on large models. The
micro-benchmark put `mm_fp4` at 3.3–5.9× on medium/large GEMMs vs bf16.
The cost step is already batched bf16 bmm, so the realistic end-to-end
speedup is smaller than the raw GEMM ratio — but cost + measurement are
the dominant GPU phases, so even a 2× on those is material on 27B+
runs. This is the only path that converts the GB10 tensor cores from a
rejected opt-in lever into a faithful, pipeline-wide speedup.
