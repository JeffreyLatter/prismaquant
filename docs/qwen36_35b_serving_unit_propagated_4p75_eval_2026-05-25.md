# Qwen3.6-35B Serving-Unit Propagated 4.75 Evaluation

Date: 2026-05-25

## Code Change

The MSE-promotion path now supports `group_by=serving_unit` / `fused_unit`.
When a model profile exposes `fused_sibling_group(name)`, q/k/v,
linear-attn qkv/z, and shared-expert gate/up siblings are grouped as one
serving unit. Non-fused siblings such as `o_proj`, `out_proj`, and
`shared_expert.down_proj` stay independent singletons.

This matters because the previous `layer_category` grouping was too coarse:
it tied unrelated tensors in the same layer/category together and hid the
fact that the most sensitive units were mostly shared-expert down projections.

## Propagated Sensitivity Report

Report:
`/home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/mse_promotion/propagated_serving_unit_sensitivity_all_n4s512.json`

Command shape:

```bash
PYTHONPATH=. PRISMAQUANT_L3_CUDA_GRAPHS=0 \
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  tools/sensitivity_propagated_group_report.py \
  --group-by serving_unit \
  --n-calib-samples 4 \
  --calib-seqlen 512 \
  --max-lanes-per-batch 4 \
  ...
```

Measured 124 / 124 serving units in 1094.4s.

Top propagated KL per added bit:

| Rank | Unit | KL | Current |
|---:|---|---:|---|
| 1 | `tensor:model.layers.5.mlp.shared_expert.down_proj` | 0.020284 | NVFP4 |
| 2 | `tensor:model.layers.20.mlp.shared_expert.down_proj` | 0.018727 | NVFP4 |
| 3 | `tensor:model.layers.18.mlp.shared_expert.down_proj` | 0.005736 | MXFP8 |
| 4 | `tensor:model.layers.14.mlp.shared_expert.down_proj` | 0.005534 | MXFP8 |
| 5 | `tensor:model.layers.10.mlp.shared_expert.down_proj` | 0.007141 | NVFP4 |
| 7 | `fused:model.layers.8.mlp.shared_expert.gate_up_proj` | 0.008027 | MXFP8 x2 |

## Cost Sweep

Added checked-in builder:

```bash
PYTHONPATH=. /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  tools/build_propagated_sensitivity_cost_sweep.py \
  --costs /home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/cost.pkl \
  --probe /home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/probe.pkl \
  --sensitivity-report /home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/mse_promotion/propagated_serving_unit_sensitivity_all_n4s512.json \
  --output-dir /home/rob/dq-runs/qwen36-35b-propagated-servingunit-4p75-20260525T000000Z/artifacts \
  --scales 0.25,0.5,1,2,5,10
```

The propagated penalty is counted once per serving unit. For fused units it is
distributed across siblings by added-bit share, then each candidate format is
scaled by its local output-MSE ratio to the measured current format.

Scale-10 was selected for materialization because it had the best hook KL and
the strongest top-unit coverage.

| Scale | Hook KL n4/s512 | bpp | output_mse | BF16 | MXFP8 | NVFP4 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.018150 | 4.752891 | 0.004023 | 261 | 123 | 127 |
| 0.5 | 0.017650 | 4.752961 | 0.004169 | 262 | 128 | 121 |
| 1 | 0.037191 | 4.752906 | 0.004093 | 270 | 127 | 114 |
| 2 | 0.021197 | 4.752571 | 0.004106 | 301 | 99 | 111 |
| 5 | 0.012730 | 4.752501 | 0.004324 | 314 | 96 | 101 |
| 10 | 0.011384 | 4.752337 | 0.005144 | 322 | 101 | 88 |

## Materialized Artifact

Run root:
`/home/rob/dq-runs/qwen36-35b-propagated-servingunit-4p75-20260525T000000Z`

Export:
`/home/rob/dq-runs/qwen36-35b-propagated-servingunit-4p75-20260525T000000Z/exported_scale_10p0`

Export mix:

```text
BF16: 322
MXFP8_E4M3: 101
NVFP4: 88
```

vLLM smoke passed and used the intended kernels:

```text
FlashInferCutlassMxfp8LinearKernel
FlashInferCutlassNvFp4LinearKernel
```

## Shipped 4.75 Comparison

Same vLLM compressed-tensors path, same WikiText contracts used for the
previous shipped comparison.

| Artifact | KL vs BF16 n8/s512 | PPL 8192/s512 | mean NLL |
|---|---:|---:|---:|
| shipped 4.75 | 0.0671039 | 9.639520 | 2.265871 |
| serving-unit propagated 4.75 scale-10 | 0.0361860 | 9.454371 | 2.246477 |

This is a clear improvement over the shipped 4.75 on both checks:
KL is down about 46%, and PPL is down about 1.9%.

Metric files:

- `/home/rob/dq-runs/qwen36-35b-propagated-servingunit-4p75-20260525T000000Z/metrics/servingunit_4p75_scale_10p0_vllm_kl_vs_bf16_wikitext_n8_s512.json`
- `/home/rob/dq-runs/qwen36-35b-propagated-servingunit-4p75-20260525T000000Z/metrics/servingunit_4p75_scale_10p0_vllm_wikitext_ppl_8192_s512.json`
- `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/metrics/shipped_4p75_vllm_kl_vs_bf16_wikitext_n8_s512.json`
- `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/metrics/shipped_4p75_vllm_wikitext_ppl_8192_s512.json`

## Interpretation

The useful signal was not a blanket attention passthrough heuristic. It was a
serving-unit-level propagated sensitivity measurement. The resulting allocation
protects fused siblings atomically where vLLM serves them fused, but still lets
singletons like shared-expert down projections compete independently.

The improvement suggests this is the right direction for phase 1. The next
decision should be whether to replace the current shipped 4.75 with this
artifact or run the same serving-unit propagated sweep at the desired 5.x
shipping point.
