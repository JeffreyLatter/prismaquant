# DQ-Fold Smoke Results - 2026-05-13

Fold-only DuQuant++-inspired preconditioning was tested on the existing
Qwen3.5-0.8B progressive-gates assignment:

- model: `/home/rob/.cache/huggingface/qwen35-0p8b-bf16`
- assignment: `/home/rob/dq-runs/qwen35-0p8b-progressive-gates-v2-20260512T224854Z/artifacts/layer_config.json`
- calibration for cache fill: `n=8`, `seqlen=512`,
  `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- validation: `validate_assignments_kl`, production-cache in-place replay,
  `last_token`, `seqlen=512`
- bpp: `4.966982922201138` over quantizable parameters

The implementation is not full DuQuant++. It folds microscale-aware column
scales through existing weights/norms only. It has no runtime block rotation,
no residual adapter, and no custom vLLM kernel requirement.

## Important Fix

The first DQ scorer path exposed a bug shared by fold-scale candidate scoring:
the NVFP4 `compute_only` tensor was scored before final pack/dequant. With
GPTQ disabled in the candidate search, identity could score as exact BF16.
`production_weight_cache._render_awq_scaled_for_cache` now scores the actual
packed/dequantized NVFP4 value.

After that fix, a cheap pre-GPTQ proxy still accepted too many DQ folds:

- run: `/home/rob/dq-runs/qwen35-0p8b-dqfold-smoke-20260513T170840Z`
- DQ accepted: `47/48` groups, `100` linears
- local proxy score: `0.3106623573 -> 0.2796653693` (`-9.98%`)
- KL at `n=16`: baseline `0.2012080634`, DQ `0.2273565184`
  (`+13.00%`, regression)

That proxy is not production-faithful. DQ-fold now defaults its candidate
search to the active downstream package: GPTQ, Fisher-GPTQ, and scale-sweep
when those cache levers are enabled.

## Full-Gate Result

Matched no-DQ baseline:

- run: `/home/rob/dq-runs/qwen35-0p8b-dqfold-baseline-20260513T171330Z`
- levers: FourOverSix, Fisher-GPTQ, GPTQ, scale-sweep
- cache fill: `165.1s`

DQ full-gate candidate:

- run: `/home/rob/dq-runs/qwen35-0p8b-dqfold-fullgate-20260513T171847Z`
- levers: DQ-fold, FourOverSix, Fisher-GPTQ, GPTQ, scale-sweep
- cache fill: `352.7s`
- DQ accepted: `25/48` groups, `54` linears
- selected alphas: `0.25` for 8 groups, `0.5` for 17 groups
- gated local score: `0.2174830869 -> 0.2161643869` (`-0.61%`)

KL validation:

| validation | baseline KL | DQ-fold KL | relative change |
| --- | ---: | ---: | ---: |
| `n=16`, `seqlen=512` | `0.2012080634` | `0.1355714227` | `-32.62%` |
| `n=64`, `seqlen=512` | `0.2182590449` | `0.1916450168` | `-12.19%` |

The result is positive on `.8B`, but it is still a candidate, not a default.
Next gate is the same full-gate smoke on Qwen3.6-4B.

## Export And vLLM Smoke

The full-gated DQ cache exported successfully:

- artifact:
  `/home/rob/dq-runs/qwen35-0p8b-dqfold-fullgate-20260513T171847Z/exported`
- export log:
  `/home/rob/dq-runs/qwen35-0p8b-dqfold-fullgate-20260513T171847Z/logs/export.log`
- vLLM smoke log:
  `/home/rob/dq-runs/qwen35-0p8b-dqfold-fullgate-20260513T171847Z/logs/validate_native_export.log`

vLLM loaded the artifact with `FlashInferCutlassNvFp4LinearKernel` and
generated on `The capital of France is`. The environment still reports a
nonfatal stale `prismaquant_residual_adapter` plugin import error from the
archived ReSpin work; that should be cleaned from the container environment,
but it did not block load or generation.

