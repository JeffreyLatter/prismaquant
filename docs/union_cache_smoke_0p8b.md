# Smart Union Cache — 0.8B Smoke Results

Date: 2026-05-15
Tool: `tools/build_union_cache.py`
Run artifact: `/home/rob/dq-runs/union-smoke-0p8b-v2-20260515T051000Z/`

## Motivation

The kneedle path (`SELECTION_MODE=validated-surrogate`) currently builds a
**full format-menu** production cache: every NVFP4-eligible Linear is
rendered in NVFP4, MXFP8, and FP8 regardless of whether any Pareto
target's assignment will actually use those formats. At low bit budgets
the vast majority of Linears stay at NVFP4; MXFP8/FP8 renders for
NVFP4-clean Linears are wasted compute.

The smart-union approach renders selectively, using the per-Linear
output_mse already measured in cost.pkl as the suspect signal:

- All NVFP4-eligible Linears: NVFP4 rendered (always).
- Linears whose NVFP4 `output_mse > T_mxfp8`: MXFP8 fallback rendered.
- Linears whose NVFP4 `output_mse > T_fp8`: FP8 fallback rendered.
- BF16 passthrough always available at runtime (no render needed).

Thresholds default to percentile-based (p50 / p75 of the NVFP4
`output_mse` distribution); supports absolute thresholds via flags.

## Smoke result on Qwen3.5-0.8B

- Input layer_config: 146 NVFP4-eligible Linears
- NVFP4 `output_mse` distribution: T_mxfp8 (p50) = 2.4e-3, T_fp8 (p75) = 8.4e-3

| Format    | Count | Note |
|-----------|------:|------|
| NVFP4     |  146  | always |
| MXFP8     |   76  | top 50% suspect (incl. 4 from base config) |
| FP8       |   36  | top 25% very-suspect |
| **Total** |  258  | **58% of full format-menu (438)** |

Build time: 212.2s (Phase A NVFP4) + 22.5s (Phase B MXFP8) + 7.6s
(Phase C FP8) = **242s end-to-end**. Phase B / C are inexpensive because
the heavy NVFP4 GPTQ+JSO+damp_sweep work isn't repeated for those
Linears that are already in cache.

## Per-kind triage

Triage correctly identifies which kinds are NVFP4-clean vs
NVFP4-suspect:

| Kind | NVFP4 only | + MXFP8 | + MXFP8 + FP8 | Reading |
|---|---|---|---|---|
| linear_attn.out_proj | 18/18 | 0 | 0 | clean |
| mlp.down_proj | 24/24 | 0 | 0 | clean |
| self_attn.o_proj | 6/6 | 0 | 0 | clean |
| mlp.up_proj | 20/23 | 3 | 0 | mostly clean |
| linear_attn.in_proj_qkv | 0 | 4 | 13 | uniformly suspect |
| linear_attn.in_proj_z | 0 | 4 | 13 | uniformly suspect |
| mlp.gate_proj | 4/23 | 12 | 7 | mixed |

Attention input projections (`in_proj_qkv`, `in_proj_z`) are the
universally-suspect kinds — they get every higher-precision fallback
rendered. Output projections + `down_proj` are NVFP4-clean and skip
all fallback rendering. This matches our prior findings — `out_proj`
and `down_proj` quantize cleanly under NVFP4, while attention input
projections often need more precision.

## Caveats

- The cost.pkl used here only measured NVFP4 + BF16 (the cost phase
  was run with a limited format menu). Suspect Linears get MXFP8 / FP8
  rendered *unconditionally* without verifying MXFP8/FP8 would actually
  reduce the error. For a more measured rule, the cost phase should
  measure all four formats; the tool can then compare across formats
  directly.
- The union approach assumes the allocator's per-target assignments
  will not pick a (Linear, format) pair that's outside the rendered
  union. The thresholds are generous enough on 0.8B (50% MXFP8 cover)
  that this is unlikely, but for the kneedle path on a wider Pareto
  sweep, a fallback to BF16 passthrough would activate for any
  mis-prediction. Belt-and-suspenders fix: take the union of (MSE-band
  fallbacks) ∪ (allocator picks across all Pareto targets).

## Next step

Integrate into `run-pipeline.sh` validated-surrogate path: replace the
single `build_production_cache --render-scope format-menu` invocation
with a call to `build_union_cache.py`. Env-gate behind
`PRODUCTION_CACHE_UNION=1` so the format-menu path remains the
fallback. Then run the kneedle on Qwen3-4B with the union scope and
compare end-to-end build time + selected-frontier KL vs the format-menu
path.
