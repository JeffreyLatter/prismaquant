# AWQ-v2 Integration

AWQ is implemented as an opt-in production-cache/export preconditioner.

## Role in the Pipeline

AWQ complements the existing production stack. It does not supersede GPTQ,
damp sweep, scale sweep, clipping, FourOverSix, Fisher weighting, or per-Linear
format allocation.

Order:

1. Solve AWQ scale `s` for a normalization-predecessor group.
2. Fold the runtime identity:
   - predecessor norm gamma `gamma <- gamma / s`
   - every reader weight `W <- W * s`
3. Run GPTQ, damp sweep, scale sweep, clipping, and Fisher-weighted objectives
   in the transformed activation coordinates `x / s`.

This makes AWQ a coordinate-system change before the existing numerical
passes. It can reduce outlier pressure and may reduce how often clipping
matters, but the post-AWQ rounding/scale problem still needs GPTQ and
scale_sweep.

Scale search uses a cheap rendered search by default, then the selected scale
is rendered once through the full production stack. This avoids multiplying
the GPTQ/damp sweep cost by every AWQ candidate. Set
`PRISMAQUANT_AWQ_SEARCH_GPTQ=1` only for small ablations where a fully
production-faithful candidate search is worth the extra compute.

## Production Cache Contract

`ProductionWeightCache` stores AWQ-rendered weights in measurement
coordinates:

```text
cache weight = Q(W * s) / s
```

This lets KL, polish, and recache install the cached Linear into the original
unfolded model and see the same Linear function that the folded export serves.

The cache also stores `awq_scales[qname] = s`. Export uses those scales to:

```text
artifact weight = Q(W * s)
gamma <- gamma / s
```

This keeps one cache mechanism. There is no AWQ side cache.

AWQ cache fill is GPU-resident by default when the model is on CUDA:
activation tensors are captured on the model device, scalar `max_abs` syncs
are deferred until after calibration forward, and the fill raises if CUDA
AWQ captures fall back to CPU.

Production call sites apply `PRISMAQUANT_AWQ_MIN_GAIN` with a default of
`0.03`. Groups whose rendered-weight output-MSE gain is below that threshold
fall back to identity. This is not just a speed knob: small-model smokes showed
ungated AWQ can improve local MSE while worsening end-to-end KL.

Search controls:

- `PRISMAQUANT_AWQ_GRID` controls the candidate ratio grid.
- `PRISMAQUANT_AWQ_SEARCH_GPTQ=0` by default; final selected renders still
  use the caller's GPTQ setting.
- `PRISMAQUANT_AWQ_SEARCH_SCALE_SWEEP` defaults to the caller's scale-sweep
  setting, so MXFP8/FP8 searches still use their applicable scale objective.

## Scope

Current implementation supports normalization-predecessor folds:

- `input_layernorm -> q_proj/k_proj/v_proj`
- `input_layernorm -> linear_attn in_proj_*`
- `post_attention_layernorm -> gate_proj/up_proj/gate_up_proj/w1/w3/router`

Post-nonlinearity folds such as `v_proj -> o_proj` and `up_proj -> down_proj`
are not enabled. They require coupled upstream/downstream transforms that are
not independently representable by the current per-Linear cache contract when
the upstream Linear may itself be quantized. Add them only with explicit cache
transform support.

## Gates

AWQ remains experimental until measured.

Small-model gate:

```bash
AWQ=1 PRODUCTION_CACHE_RENDER_SCOPE=assignment ./prismaquant/run-pipeline.sh
```

Compare against the same run with `AWQ=0`, same calibration data, same bpp,
same production recache setting, and same vLLM load/generation smoke.

Promotion requires non-regressing KL on `.8B` and `4B`, then a 27B same-
calibration comparison. If AWQ regresses on the current GPTQ + scale_sweep +
clipper + FourOverSix stack, leave it opt-in.

Smoke note, 2026-05-11: ungated AWQ regressed Qwen3.5-0.8B KL on a tiny
`2 x 128` validation slice (`0.05619` vs current `0.02142`) while improving
Qwen3-4B (`0.08961` vs current `0.12141`). Applying only groups with local
gain >= `0.03` made 0.8B non-regressive on that slice (`0.02108`) and improved
4B further (`0.05304`); `0.05` improved 4B more (`0.04432`) but regressed
0.8B (`0.02349`). Default stays at `0.03` until a larger calibration confirms
a better cross-model threshold.
