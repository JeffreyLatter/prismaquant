# Qwen3.6 35B Propagated 4.75 Evaluation

Date: 2026-05-25

Purpose: test whether the propagated-sensitivity allocation can produce an
equal-budget 4.75-bpp replacement that beats the shipped 4.75 artifact.

## Candidate

- Run root: `/home/rob/dq-runs/qwen36-35b-propagated-4p75-20260525T041500Z`
- Source sensitivity report: `/home/rob/dq-runs/qwen36-35b-current-4p75-strategic-20260524T144947Z/mse_promotion/propagated_group_sensitivity_all_n4s512.json`
- Base cost table: `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/cost.pkl`
- Probe: `/home/rob/dq-runs/qwen36-35b-current-kneedle-20260524T015443Z/artifacts/probe.pkl`
- Selected layer config: `/home/rob/dq-runs/qwen36-35b-propagated-4p75-20260525T041500Z/artifacts/scale_5p0/layer_config.json`
- Exported model: `/home/rob/dq-runs/qwen36-35b-propagated-4p75-20260525T041500Z/exported_scale_5p0`

The sweep adjusted non-BF16 cost entries for 75 measured propagated groups
covering 175 Linear members.  Scale 5.0 was selected for materialization because
it had the best in-repo last-token KL screen at the target budget.

## Allocation Screen

| Candidate | bpp | Hook KL | Output MSE sum | BF16 | MXFP8 | NVFP4 |
|---|---:|---:|---:|---:|---:|---:|
| scale 5.0 | 4.751581 | 0.0319779 | 0.00403461 | 317 | 62 | 132 |
| scale 10.0 | 4.750754 | 0.0328062 | 0.00487161 | 333 | 61 | 117 |
| scale 25.0 | 4.750302 | 0.0360454 | 0.00598793 | 337 | 69 | 105 |

Allocation sensitivity coverage, counting top propagated groups that were all
BF16:

| Candidate | Top 10 | Top 20 | Top 40 | linear_attn.9 qkv/z/out | self_attn.11 q/k/v/o |
|---|---:|---:|---:|---|---|
| shipped 4.75 | 9 | 19 | 35 | MXFP8 / MXFP8 / NVFP4 | BF16 / BF16 / BF16 / BF16 |
| previous siblingfix 4.75 | 0 | 0 | 0 | BF16 / BF16 / NVFP4 | BF16 / BF16 / BF16 / NVFP4 |
| propagated scale 5.0 | 10 | 14 | 14 | BF16 / BF16 / BF16 | BF16 / BF16 / BF16 / NVFP4 |
| propagated scale 10.0 | 10 | 20 | 33 | MXFP8 / MXFP8 / BF16 | BF16 / BF16 / BF16 / BF16 |
| propagated scale 25.0 | 10 | 20 | 40 | MXFP8 / MXFP8 / BF16 | BF16 / BF16 / BF16 / BF16 |

## Materialization

The scale-5.0 candidate was recached and exported with the production cache path:

- Recached cache: `/home/rob/dq-runs/qwen36-35b-propagated-4p75-20260525T041500Z/artifacts/production_weight_cache_scale_5p0_recached.pkl`
- Recache log: `/home/rob/dq-runs/qwen36-35b-propagated-4p75-20260525T041500Z/logs/production_recache_scale_5p0.log`
- Export log: `/home/rob/dq-runs/qwen36-35b-propagated-4p75-20260525T041500Z/logs/export_scale_5p0.log`

Native vLLM smoke passed and loaded the expected kernels:

- `FlashInferCutlassMxfp8LinearKernel`
- `FlashInferCutlassNvFp4LinearKernel`
- FlashAttention

## Results

| Artifact | bpp | KL mean | PPL | Mean NLL |
|---|---:|---:|---:|---:|
| Shipped 4.75 | 4.75 | 0.0671039 | 9.63952 | 2.26587 |
| Propagated 4.75 scale 5.0 | 4.751581 | 0.0769279 | 9.87240 | 2.28974 |
| Propagated 5.15 | 5.149831 | 0.0487880 | 9.37146 | 2.23767 |

The equal-budget propagated 4.75 candidate does not beat shipped 4.75.  It is
14.6% worse on full-vocab vLLM KL and 0.233 worse on WikiText PPL.

## Logs

- Hook KL screen: `/home/rob/dq-runs/qwen36-35b-propagated-4p75-20260525T041500Z/logs/validate_kl_scale_sweep.log`
- vLLM smoke: `/home/rob/dq-runs/qwen36-35b-propagated-4p75-20260525T041500Z/logs/validate_native_export_scale_5p0.log`
- vLLM KL: `/home/rob/dq-runs/qwen36-35b-propagated-4p75-20260525T041500Z/logs/measure_kl_scale_5p0.log`
- WikiText PPL: `/home/rob/dq-runs/qwen36-35b-propagated-4p75-20260525T041500Z/logs/measure_ppl_scale_5p0.log`

## Interpretation

The new propagated signal corrected the obvious failure in the recent current
4.75 allocation: the previous siblingfix candidate protected none of the top
20 propagated groups, while the scale-5.0 candidate protected all top 10 and
14 of the top 20.  That was still not enough to beat shipped, because the
shipped 4.75 artifact already protects 19 of the top 20 and 35 of the top 40
propagated groups.

At equal budget, shipped remains the better artifact.  The propagated signal is
still useful at higher budget: the materialized 5.15 propagated candidate beats
shipped 4.75 on both KL and PPL, but it spends about 0.40 additional bpp.
