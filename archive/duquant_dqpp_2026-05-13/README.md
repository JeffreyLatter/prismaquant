# DQ-Fold / DuQuant++ Fold-Only — Archived 2026-05-13

This directory archives the fold-only DuQuant++-inspired microscale
preconditioner that briefly lived as a production candidate during the
PrismaQuant `.8B`/`4B` smoke ladder.

## What it was

A per-input-channel scale search that folded the chosen scale through the
predecessor normalization, so the runtime saw `(x / s) @ Q(W * s)^T` with no
new operator. It searched alphas in `{0.25, 0.5, 0.75}` over microscale-block
statistics, then accepted only if the local output MSE under the active
downstream package (GPTQ / Fisher-GPTQ / scale-sweep) improved.

It was **not** full DuQuant++. The original method applies block rotations at
activation quantization time and requires runtime/kernel support. This
implementation stayed within the existing fold-scale machinery and added no
runtime cost.

## Why it was archived

- The cheap pre-GPTQ proxy that the first revision used was too permissive:
  it accepted `47/48` groups (`100` linears) on Qwen3.5-0.8B and shipped
  `+13.00%` KL regression at `n=16` despite a `-9.98%` local proxy
  improvement (run `qwen35-0p8b-dqfold-smoke-20260513T170840Z`).
- After switching the candidate scorer to the active downstream package, the
  Qwen3.5-0.8B result was positive: `25/48` groups accepted, local score
  `-0.61%`, KL `n=16` `-32.62%`, KL `n=64` `-12.19%`. vLLM loaded the
  artifact with `FlashInferCutlassNvFp4LinearKernel` and generated coherent
  text (run `qwen35-0p8b-dqfold-fullgate-20260513T171847Z`).
- But the full-gate cache fill went from `165.1s` to `352.7s` (`~2.1x`) on
  `.8B`, and a separate `4B` smoke without PrismaClip was mixed. The user
  decided the rotation-shaped problem deserved an actual runtime rotation
  (block-diagonal at the microscale group size) rather than continuing to
  rasterize it through fold-only scales.

## Critical bug fixed before archive

`_render_awq_scaled_for_cache` in `production_weight_cache.py` had been
scoring NVFP4 `compute_only` pre-pack tensors. With GPTQ disabled inside the
candidate search loop, the identity candidate could score as exact BF16,
beating every real fold candidate.

The fix scored the actual packed/dequantized NVFP4 tensor via
`enc._rtn_dequant_nvfp4(result["_w_dq"], group_size=16,
global_real_override=joint_global_real)`. Anyone reviving fold-scale
preconditioning research should preserve that lesson.

## What was kept live

- `prismaquant/render_score.py` — the progressive render plugin framework is
  general infrastructure, not DQ-specific.
- AWQ-v2 + SmoothQuant — opt-in fold-scale levers; the joint `awq_scales`
  field on `ProductionWeightCache` is reused for either.
- `_render_fold_scaled_for_cache` — the generic format-aware fold renderer.
  Its docstring was scrubbed of DQ-specific wording.

## Files in this archive

- `duquant.py` — the fold-only preconditioner module.
- `test_duquant.py` — DQ-specific unit tests.
- `duquant_fold_smoke_2026-05-13.md` — the `.8B` smoke writeup.

## Pointer for the rotation revival

The path forward is a **runtime** block-diagonal rotation at the microscale
group size, using the compressed-tensors `transforms_config` already wired in
vLLM 0.19+ (`HadamardTransform`, `input_transform`/`output_transform` on
`CompressedTensorsLinearTransformMethod`). That mechanism applies a stored
`(G, G)` matrix as block-diagonal-at-`head_dim` via either fused
`hadacore_transform` (fast Sylvester) or a dense GEMM (learned orthogonal).
PrismaQuant's BlockOrtho-G builds on that infrastructure.
