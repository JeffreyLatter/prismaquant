# Qwen3.6 27B Current vs Shipped, 2026-05-25

This note records the 27B served-vLLM comparison between the current
allocator outputs and the shipped 5.5 / 5.31 artifacts. Metrics use the
same calibration contract as the prior 27B comparisons:

- KL teacher payload:
  `/home/rob/dq-runs/qwen3p6-27b-rerun/kl_shipped_5p5_20260503T012223Z/teacher_logprobs.pt`
- WikiText ids:
  `/home/rob/dq-runs/qwen36-27b-grouped-5p5-vs-shipped-20260521T134058Z/wikitext_test_8192_ids.pt`
- vLLM flags: `--quantization compressed-tensors --gpu-memory-utilization 0.70 --enforce-eager`
- PPL scored tokens: 8176

## Served vLLM Results

| Artifact | Model path | KL mean vs BF16 | WikiText PPL | Mean NLL |
| --- | --- | ---: | ---: | ---: |
| Current 5.5 | `/home/rob/dq-runs/qwen36-27b-current-5p5-materialize-20260523T012423Z/exported` | 0.0344416238 | 9.3206082674 | 2.2322278913 |
| Shipped 5.5 | `/home/rob/.cache/huggingface/rdtand-Qwen3.6-27B-PrismaQuant-5.5bit-vllm` | 0.0474975146 | 9.4982818549 | 2.2511109249 |
| Shipped 5.31 | `/home/rob/.cache/huggingface/rdtand-Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm-shipped-5p31` | 0.0550810285 | 9.6277936759 | 2.2646540901 |
| Current 5.0 kneedle | `/home/rob/dq-runs/qwen36-27b-codec-kneedle-20260522T183316Z/exported` | 0.0781948268 | 9.5770206819 | 2.2593665501 |

The current materialized 5.5 beats shipped 5.5 by 27.5% on KL and 1.9%
on WikiText PPL. It beats shipped 5.31 by 37.5% on KL and 3.2% on PPL.

The current 5.0 kneedle candidate does not reproduce its hook-screen KL in
served vLLM. It is worse than shipped 5.5 on both KL and PPL, and worse
than shipped 5.31 on KL, despite having slightly lower PPL than shipped
5.31. Treat the 5.0 served export as not shippable until the hook-vs-served
drift is reconciled.

## Current 5.31 MSE Comparison

The current 5.31 assignment was not materialized for served vLLM in this
comparison. Its available evidence is the language-body MSE comparison
against the shipped 5.31 artifact:

| Artifact | Body counts | Output MSE sum | Output MSE element-weighted | Weight MSE param-weighted |
| --- | --- | ---: | ---: | ---: |
| Current 5.31 assignment | BF16 82, FP8_E4M3 126, NVFP4 288 | 140308.17340366542 | 0.0002807780749785185 | 9.850120292396546e-07 |
| Shipped 5.31 reference | BF16 117, NVFP4 379 | 313073.03117986023 | 0.000626506930351603 | 1.2853533751019287e-06 |

This is a 55.2% reduction in output MSE sum and a 23.4% reduction in
param-weighted weight MSE for the current 5.31 assignment. Caveat: under
the current quantizable-parameter accounting, the public "5.31" artifact's
language-body bpp is about 4.76, so the label is not budget-matched against
the current 5.31 body assignment.

## Metric Files

- Current 5.0 KL:
  `/home/rob/dq-runs/qwen36-27b-compare-current-vs-shipped-20260525T000000Z/vllm_kl_current_kneedle_5p0_vs_bf16.json`
- Current 5.0 PPL:
  `/home/rob/dq-runs/qwen36-27b-compare-current-vs-shipped-20260525T000000Z/vllm_ppl_current_kneedle_5p0_wikitext_test_8192_from_ids.json`
- Current 5.5 KL:
  `/home/rob/dq-runs/qwen36-27b-current-5p5-materialize-20260523T012423Z/vllm_kl_current_5p5_vs_bf16.json`
- Current 5.5 PPL:
  `/home/rob/dq-runs/qwen36-27b-current-5p5-materialize-20260523T012423Z/vllm_ppl_current_5p5_wikitext_test_8192_from_ids.json`
- Shipped 5.5 KL:
  `/home/rob/dq-runs/qwen3p6-27b-rerun/kl_shipped_5p5_20260503T012223Z/shipped_5p5_kl.json`
- Shipped 5.5 PPL:
  `/home/rob/dq-runs/qwen36-27b-grouped-5p5-vs-shipped-20260521T134058Z/vllm_ppl_shipped_5p5_wikitext_test_8192_from_ids.json`
- Shipped 5.31 KL:
  `/home/rob/dq-runs/qwen36-27b-compare-current-vs-shipped-20260525T000000Z/vllm_kl_shipped_5p31_vs_bf16.json`
- Shipped 5.31 PPL:
  `/home/rob/dq-runs/qwen36-27b-compare-current-vs-shipped-20260525T000000Z/vllm_ppl_shipped_5p31_wikitext_test_8192_from_ids.json`
- Current 5.31 MSE summary:
  `/home/rob/dq-runs/qwen36-27b-mxfp8-rerender-20260522T235213Z/artifacts/current_5p31_vs_shipped_5p31_mse_summary_v2.json`

The shipped 5.31 PPL command wrote a complete JSON result and then aborted
during vLLM process shutdown. The metric above is taken from the completed
JSON output.
