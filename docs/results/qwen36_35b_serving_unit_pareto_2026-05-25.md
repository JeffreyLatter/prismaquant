# Qwen3.6-35B Serving-Unit Pareto, 2026-05-25

Run root:
`/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z`

This reran the 35B allocator Pareto using the serving-unit propagated
sensitivity report with scale 10 and `local_mse_ratio` extrapolation.

Inputs:

- Model: `/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409`
- Probe: `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/probe.pkl`
- Costs: `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/cost.pkl`
- Production cache: `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/production_weight_cache_frontier_raw.pkl`
- Serving-unit report: `/home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/mse_promotion/propagated_serving_unit_sensitivity_all_n4s512.json`

Allocator command shape:

```bash
python -m prismaquant.allocator \
  --formats NVFP4,MXFP8_E4M3,BF16 \
  --propagated-sensitivity-report <serving-unit-report> \
  --propagated-sensitivity-scale 10 \
  --propagated-sensitivity-format-extrapolation local_mse_ratio \
  --pareto-targets 4.5,4.6,4.7,4.75,4.85,5.0,5.15,5.25,5.31,5.5,5.75,6.0,6.5,7.0,8.25
```

## Pareto

The allocator surrogate knee was target 4.85, but the measured hook-KL kneedle
selected the 4.7028-bpp point. The validated hook screen is useful for finding
a cheap point, but the 5.5329-bpp point was also materialized because it is the
closest useful new "5.5" replacement candidate.

| Target | Achieved bpp | Predicted dloss | DP BF16 | DP MXFP8 | DP NVFP4 |
|---:|---:|---:|---:|---:|---:|
| 4.70 | 4.702826 | 4.47473 | 88 | 85 | 97 |
| 4.75 | 4.752337 | 3.09264 | 111 | 71 | 88 |
| 5.25 | 5.249425 | 0.18813 | 187 | 5 | 78 |
| 5.31 | 5.257401 | 0.18239 | 192 | 0 | 78 |
| 5.50 | 5.257401 | 0.18239 | 192 | 0 | 78 |
| 5.75 | 5.532853 | 0.14846 | 194 | 0 | 76 |

Validated hook-KL kneedle:

- Assignment: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/artifacts/layer_config_validated_kneedle.json`
- Achieved bpp: 4.702826
- Hook KL: 0.0169124
- Hook-screen expanded mix: BF16 174, MXFP8 130, NVFP4 97
- Export: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/exported_validated_kneedle_4p70`

The 5.53 candidate was materialized from:

- Solve result: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/artifacts/pareto_assignments/allocator_target_5p7500_achieved_5p5329_ca04046c97f9.json`
- Layer config: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/artifacts/layer_config_5p53.json`
- Recached PWC: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/artifacts/production_weight_cache_5p53_recached.pkl`
- Export: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/exported_5p53`

## Export Mixes

The 4.70 kneedle uses MXFP8 on linears plus NVFP4 on packed experts:

```text
linear/MXFP8_E4M3_PRODUCTION_CACHE: 130
linear/BF16: 203
linear/NVFP4_PRODUCTION_CACHE: 17
packed_moe_per_expert/NVFP4: 80
packed_moe_per_expert/BF16: 4
layer_passthrough/BF16: 260
mtp_linear/BF16: 9
mtp_packed_moe/BF16: 2
mtp_passthrough/BF16: 8
head_passthrough/BF16: 3
```

The 5.53 candidate drops MXFP8 entirely and spends BF16 on all linears, leaving
only 76 packed experts in NVFP4:

```text
linear/BF16: 350
packed_moe_per_expert/NVFP4: 76
packed_moe_per_expert/BF16: 4
layer_passthrough/BF16: 260
mtp_linear/BF16: 9
mtp_packed_moe/BF16: 2
mtp_passthrough/BF16: 8
head_passthrough/BF16: 3
```

Both native vLLM smokes passed under `--quantization compressed-tensors`.
The 4.70 artifact used `FlashInferCutlassMxfp8LinearKernel` and
`FlashInferCutlassNvFp4LinearKernel`. The 5.53 artifact used the NVFP4 MoE
backend and the FlashInfer CUTLASS unquantized MoE backend for BF16 experts.

## vLLM Metrics

All rows use the same vLLM compressed-tensors path and the same WikiText
contracts:

- Full-vocab KL: BF16 teacher payload, n=8, seqlen=512, max-logprobs 248320.
- PPL: WikiText test, 8192 requested tokens, seqlen=512.

| Artifact | bpp | KL vs BF16 | PPL | Mean NLL |
|---|---:|---:|---:|---:|
| shipped 4.75 | 4.75 | 0.0671039 | 9.639520 | 2.265871 |
| current kneedle 5.1569 | 5.1569 | 0.0642787 | 9.511862 | 2.252540 |
| serving-unit 4.75 scale-10 | 4.752337 | 0.0361860 | 9.454371 | 2.246477 |
| new measured kneedle 4.70 | 4.702826 | 0.0451274 | 9.494644 | 2.250728 |
| new 5.53 candidate | 5.532853 | 0.0326770 | 9.354678 | 2.235877 |

The new 5.53 candidate is the best materialized 35B artifact from this pass on
both full-vocab KL and PPL. It improves over shipped 4.75 by about 51% KL and
0.285 PPL, and it improves over the earlier serving-unit 4.75 by about 10% KL
and 0.100 PPL.

Metric files:

- `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/metrics/candidate_5p53_vllm_kl_vs_bf16_wikitext_n8_s512.json`
- `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/metrics/candidate_5p53_vllm_wikitext_ppl_8192_s512.json`
- `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/metrics/kneedle_4p70_vllm_kl_vs_bf16_wikitext_n8_s512.json`
- `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/metrics/kneedle_4p70_vllm_wikitext_ppl_8192_s512.json`

Logs:

- Allocator: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/logs/allocator_servingunit_scale10.log`
- Hook KL validation: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/logs/validated_frontier_kl.log`
- 4.70 export: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/logs/export_validated_kneedle_4p70.log`
- 5.53 recache: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/logs/production_recache_5p53.log`
- 5.53 export: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/logs/export_5p53.log`
- 5.53 vLLM smoke: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/logs/validate_native_export_5p53.log`
- 5.53 KL: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/logs/measure_kl_5p53.log`
- 5.53 PPL: `/home/rob/dq-runs/qwen36-35b-servingunit-pareto-20260525T000000Z/logs/measure_ppl_5p53.log`

## Notes

The first 5.53 recache attempt pointed at the solve-result JSON rather than a
plain layer assignment and failed schema validation before writing output. The
assignment was extracted into `layer_config_5p53.json` and recache/export then
completed.

For KL/PPL, the host's default `/home/rob/.triton/cache` is root-owned. The
successful metric runs used a run-local `TRITON_CACHE_DIR` and
`FLASHINFER_DISABLE_VERSION_CHECK=1`, matching the environment policy already
used by `validate_native_export.py`. PPL also required
`--dataset-cache-dir /home/rob/.cache/huggingface/datasets`; `/hfcache/datasets`
is not a valid local cache path on this host.

## Takeaway

The 4.70 measured kneedle is an efficient low-bpp point, but it is not the best
replacement candidate. The 5.53 artifact is the current best 35B result from
this run and is the first fresh serving-unit Pareto artifact here that beats
the shipped baseline, the current 5.1569 candidate, and the earlier
serving-unit 4.75 on both measured KL and PPL.
