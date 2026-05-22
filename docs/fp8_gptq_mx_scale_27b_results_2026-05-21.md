# 27B FP8 GPTQ and Microscaled FP8 Scale-Optimized Tail Results

Date: 2026-05-21

## Code Path Tested

- `FP8_E4M3`: GPTQ one-shot OBS rounding.
- `MXFP8_E4M3`: GPTQ plus legal E8M0 joint scale optimization.
- `MXFP8_E5M2`: GPTQ plus legal E8M0 joint scale optimization, measured for research but blocked by the current `vllm_packed_moe` dense-Linears profile.
- Legacy bare `MXFP8` inputs remain accepted at parser/cache boundaries, but allocator/export outputs are canonicalized to explicit format names.

## Run Artifacts

- Run dir: `/home/rob/dq-runs/qwen36-27b-fp8-mx-gptq-tail-20260521T222923Z`
- Score CSVs:
  - `/home/rob/prismaquant/tmp/qwen36-27b-fp8-mx-gptq-tail-20260521T222923Z/tail_per_linear_scores.csv`
  - `/home/rob/prismaquant/tmp/qwen36-27b-fp8-mx-gptq-tail-20260521T222923Z/tail_export_unit_scores.csv`
  - `/home/rob/prismaquant/tmp/qwen36-27b-fp8-mx-gptq-tail-20260521T222923Z/qwen36-27b-fp8-mx-gptq-tail-csvs.zip`
- Allocator output:
  - `/home/rob/dq-runs/qwen36-27b-fp8-mx-gptq-tail-20260521T222923Z/artifacts/layer_config_fp8_mx_gptq_tail_5p5.json`
  - `/home/rob/dq-runs/qwen36-27b-fp8-mx-gptq-tail-20260521T222923Z/artifacts/pareto_fp8_mx_gptq_tail.csv`
- Validation logs:
  - `/home/rob/dq-runs/qwen36-27b-fp8-mx-gptq-tail-20260521T222923Z/artifacts/validation_fp8_mx_gptq_tail_5p5_direct_2048_nographs.log`
  - `/home/rob/dq-runs/qwen36-27b-fp8-mx-gptq-tail-20260521T222923Z/artifacts/validation_fp8_mx_gptq_tail_5p5_direct_2048_nographs_kl.log`

## Commands

Score render used the production cache builder over the 149-line tail qname list, with:

```bash
PRISMAQUANT_GPTQ_DAMP_SWEEP=1
PRISMAQUANT_MXFP8_JOINT_SCALE_SHIFTS=-4,-3,-2,-1,0,1,2
python3 -m prismaquant.build_production_cache \
  --model /hfcache/qwen36-27b-bf16 \
  --dataset /work/calibration/diverse-v1.jsonl \
  --formats FP8_E4M3,MXFP8_E4M3,MXFP8_E5M2 \
  --include-qnames-file artifacts/production_render_score_tail_qnames.txt
```

Allocator command used:

```bash
python3 -m prismaquant.allocator \
  --probe artifacts/probe.pkl \
  --costs artifacts/cost_fp8_mx_gptq_tail.pkl \
  --formats NVFP4,FP8_E4M3,MXFP8_E4M3,MXFP8_E5M2,BF16 \
  --target-bits 5.5 \
  --target-profile vllm_packed_moe
```

Validation was run with a combined production cache, CUDA graphs disabled, 2048 WikiText tokens, and 8 x 512 end-KL calibration:

```bash
PRISMAQUANT_VALIDATION_PROD_CACHE=artifacts/combined_selected_prod_cache.pkl
PRISMAQUANT_VALIDATION_PROD_CACHE_LRU_GB=8
PRISMAQUANT_VALIDATION_CUDA_GRAPHS=0
python3 -m prismaquant.validation_harness \
  --model /hfcache/qwen36-27b-bf16 \
  --layer-config artifacts/layer_config_fp8_mx_gptq_tail_5p5.json \
  --n-wikitext-tokens 2048 \
  --n-mmlu-questions 0 \
  --calib-seqlen 512 \
  --calib-n-samples 8
```

## Local Score Results

149 tail Linears were rendered. `FP8_E4M3` was the best legal local MSE for all 149 Linears and all 90 fused export units.

Per-Linear local MSE ranges:

| format | n | min | p50 | p90 | p99 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `NVFP4` | 149 | 0.002708 | 0.009372 | 0.014366 | 0.030486 | 0.033639 | 0.009128 |
| `FP8_E4M3` | 149 | 0.000170 | 0.000811 | 0.001376 | 0.002827 | 0.003770 | 0.000818 |
| `MXFP8_E4M3` | 149 | 0.000961 | 0.008148 | 0.031037 | 0.081594 | 0.117688 | 0.013286 |
| `MXFP8_E5M2` | 149 | 0.001411 | 0.011040 | 0.035358 | 0.092361 | 0.121289 | 0.015911 |

Format deltas:

- `FP8_E4M3` GPTQ improved over old FP8 tail scores: mean new/old = 0.7226, p50 = 0.6552.
- `MXFP8_E4M3` GPTQ plus E8M0 JSO improved over old scale-sweep tail scores: mean new/old = 0.4794, p50 = 0.4865.
- `MXFP8_E4M3` was still worse than `FP8_E4M3` on every measured tail Linear.
- `MXFP8_E5M2` was worse than `MXFP8_E4M3` on every measured tail Linear and is profile-mismatched for the current dense path.

## Allocator Result

Pareto core rows before fused sibling expansion:

| target bpp | achieved bpp | NVFP4 | FP8_E4M3 | BF16 | predicted dloss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5.0 | 4.9990 | 270 | 34 | 0 | 5.6638e6 |
| 5.5 | 5.4957 | 248 | 56 | 0 | 4.0923e6 |
| 6.0 | 5.9966 | 242 | 49 | 13 | 3.6737e6 |

Final 5.5 layer config after expansion/passthrough:

| format | count |
| --- | ---: |
| `NVFP4` | 386 |
| `FP8_E4M3` | 110 |
| `BF16` | 118 |

No `MXFP8_E4M3` or `MXFP8_E5M2` entries were selected.

## Validation

Direct production-cache validation, 2048 WikiText tokens:

| artifact | ppl_wikitext | end_kl | notes |
| --- | ---: | ---: | --- |
| New FP8 GPTQ allocation | 6.9887 | 0.062815 | CUDA graphs disabled, direct production cache |
| Prior staged 5.5 direct validation | 8.9567 | n/a | same 2048-token direct harness, end-KL skipped |
| Prior 5.5 direct baseline | 8.3294 | n/a | 8192-token direct harness, end-KL skipped |
| Shipped 5.5 vLLM artifact | n/a | 0.047498 | 8 x 512 calibration KL, vLLM-served artifact |

## Recommendation

Do not ship this as the production allocator replacement yet. The local score is much cleaner than the previous proxy and the direct 2048-token PPL is better than the prior staged run, but the end-KL is worse than the shipped 5.5 artifact on the same 8 x 512 scale. Local post-quantization MSE is useful for candidate generation, but it still does not capture cumulative layer interactions well enough to replace validation-gated selection.

The immediate next production path should use this renderer/scorer to build a small high-quality candidate set, then run a sparse empirical KL gate before finalizing promotions.
