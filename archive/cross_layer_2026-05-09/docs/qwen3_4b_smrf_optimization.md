# Qwen3-4B SMRF optimization

Date: 2026-05-17
Branch: `clado-plugin-integration`
Run: `/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z`

Status: SMRF is wired as an opt-in research component and produced a
materialized Qwen3-4B compressed-tensors artifact. The best-KL artifact
loads in vLLM eager mode and selects performant NVFP4/FP8 kernels.

## Method

This run resurrects SMRF as a proposal generator over the existing
PrismaQuant allocator/probe cost table. The bridge builds SMRF decision
units from `probe.pkl` and `cost.pkl`, then filters candidate formats
through the active serving profile.

For this Qwen3-4B run:

- Model profile: `qwen3`
- Target serving profile: `vllm_qwen3_5_packed_moe`
- Formats: `NVFP4`, `MXFP8_E4M3`, `FP8_E4M3`, `BF16`
- Fused siblings: aggregated before optimization, including q/k/v and
  gate/up runtime groups
- Objective source: existing local cost-table MSE, primarily `output_mse`
- Candidate generation: 375 archive rows, 15 validation candidates

Important caveat: this is not a full propagated-cost or L3 SMRF run yet.
It uses the existing local output-MSE allocator costs as the SMRF value
surface. That makes it directly compatible with the current cache and
validation path, but the candidate frontier should still be treated as
research until larger validation and graph-mode serving checks are run.

## Validation contract

The measured KL comparison used the same Qwen3-4B baseline artifacts from:

`/home/rob/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z`

Validation inputs:

- Model: `/hfcache/Qwen3-4B`
- Probe: `/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/probe.pkl`
- Costs: `/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/cost.pkl`
- Baseline PQ validation:
  `/home/rob/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/validated_frontier_kl.json`
- Dataset: `/dq-runs/calibration/diverse-v1.jsonl`
- Samples: 32
- Sequence length: 1024
- Split/seed: `train` / `42`
- KL scope: `last_token`
- Assignment materialization: `hooks`
- Source prefetch: `require`
- Production cache prefetch: `require`
- Production cache:
  `/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/production_weight_cache_frontier_raw.pkl`
- Production cache dir:
  `/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/production_weight_cache_frontier`
- Production cache LRU: 64 GiB

Log:
`/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/logs/smrf_validate_kl.log`

The validator prefetched 3 source shards, 7.49 GiB, and preloaded the
required production cache entries for each candidate.

## SMRF frontier

| assignment | bpp | KL | formats |
|---|---:|---:|---|
| smrf_000_bpp_4.5000_19c5586bb07d | 4.500000 | 0.182746 | 252 NVFP4 |
| smrf_001_bpp_4.7830_39e4de5c9f12 | 4.783031 | 0.139386 | 228 NVFP4, 20 FP8_E4M3, 4 BF16 |
| smrf_002_bpp_4.8072_13c380ad93e0 | 4.807170 | 0.149714 | 227 NVFP4, 21 FP8_E4M3, 4 BF16 |
| smrf_003_bpp_5.0600_13284f7009f4 | 5.059970 | 0.133757 | 204 NVFP4, 38 FP8_E4M3, 10 BF16 |
| smrf_004_bpp_5.3405_c98fe606c384 | 5.340494 | 0.090950 | 186 NVFP4, 52 FP8_E4M3, 14 BF16 |
| smrf_005_bpp_5.6677_d78683c5aa32 | 5.667677 | 0.122692 | 165 NVFP4, 70 FP8_E4M3, 17 BF16 |
| smrf_006_bpp_5.9440_259df68d4e74 | 5.943989 | 0.179269 | 149 NVFP4, 83 FP8_E4M3, 20 BF16 |
| smrf_007_bpp_6.2252_9dca35e4571b | 6.225239 | 0.138405 | 135 NVFP4, 97 FP8_E4M3, 20 BF16 |
| smrf_008_bpp_6.5264_c46a08efcf5f | 6.526398 | 0.089961 | 122 NVFP4, 105 FP8_E4M3, 25 BF16 |
| smrf_009_bpp_6.7883_3107be59a8f7 | 6.788316 | 0.103629 | 102 NVFP4, 128 FP8_E4M3, 22 BF16 |
| smrf_010_bpp_7.0675_84091e3283e5 | 7.067492 | 0.075809 | 99 NVFP4, 118 FP8_E4M3, 35 BF16 |
| smrf_011_bpp_7.3880_c975f73df27c | 7.387960 | 0.052747 | 81 NVFP4, 136 FP8_E4M3, 35 BF16 |
| smrf_012_bpp_7.6583_a441dbb70773 | 7.658315 | 0.017559 | 78 NVFP4, 128 FP8_E4M3, 46 BF16 |
| smrf_013_bpp_7.9876_3ae4ad22384e | 7.987590 | 0.018496 | 62 NVFP4, 144 FP8_E4M3, 46 BF16 |
| smrf_014_bpp_8.2381_e92dfd5a6846 | 8.238104 | 0.017414 | 55 NVFP4, 144 FP8_E4M3, 53 BF16 |

Full validation output:
`/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_validated_kl.json`

## Selection

Kneedle selected:

- Assignment:
  `/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_assignment_kneedle.json`
- Layer config:
  `/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_layer_config_kneedle.json`
- Summary:
  `/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_selection_kneedle.json`
- Candidate: `smrf_004_bpp_5.3405_c98fe606c384`
- Bpp: 5.340494
- KL: 0.090950
- Formats: 186 NVFP4, 52 FP8_E4M3, 14 BF16

Best KL selected:

- Assignment:
  `/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_assignment_best_kl.json`
- Layer config:
  `/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_layer_config_best_kl.json`
- Summary:
  `/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_selection_best_kl.json`
- Candidate: `smrf_014_bpp_8.2381_e92dfd5a6846`
- Bpp: 8.238104
- KL: 0.017414
- Formats: 55 NVFP4, 144 FP8_E4M3, 53 BF16

## Comparison to standard PQ

Comparison summary:
`/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_summary.json`

| comparison | SMRF bpp | SMRF KL | PQ bpp | PQ KL | relative KL delta |
|---|---:|---:|---:|---:|---:|
| kneedle vs nearest PQ | 5.340494 | 0.090950 | 5.250681 | 0.101920 | -10.76% |
| best-KL vs nearest PQ | 8.238104 | 0.017414 | 8.250347 | 0.018402 | -5.37% |
| lower-bpp SMRF vs 8.25 PQ | 7.658315 | 0.017559 | 8.250347 | 0.018402 | -4.58% |

The current SMRF frontier therefore has at least one high-bpp point that
beats the closest standard PQ point while using slightly fewer bits. The
kneedle point also improves KL versus the nearest lower-bpp PQ point, but
that claim should be rerun with a 256-sample validation pass before being
promoted beyond research.

## Materialization and vLLM smoke

The best-KL SMRF assignment was exported:

`/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/exported_smrf_best_kl`

Export log:
`/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/logs/export_smrf_best_kl.log`

Export details:

- Recipe: 252 quantizable entries
- Mix: 55 NVFP4, 144 FP8_E4M3, 53 BF16
- Production-weight-cache direct path: 199 entries
- LRU: 64 GiB
- vLLM metadata: `compressed-tensors`
- Checkpoint size: 4.21 GiB

Serve command from the exporter:

```bash
vllm serve /dq-runs/qwen3-4b-smrf-20260517T000429Z/exported_smrf_best_kl \
  --quantization compressed-tensors
```

Eager vLLM smoke log:
`/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/logs/vllm_smoke_smrf_best_kl.log`

vLLM selected:

- `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM
- `CutlassFP8ScaledMMLinearKernel` for FP8

The smoke generated text successfully from:
`Explain PrismaQuant in one sentence.`

The log includes a non-blocking warning for a missing residual-adapter
plugin entrypoint. The SMRF artifact does not require that adapter, and
vLLM continued to load and generate.

Graph-mode vLLM smoke and WikiText PPL have not been run for this SMRF
artifact yet.

## Rigorous matched validation

Run:
`/home/rob/dq-runs/qwen3-4b-smrf-rigorous-l3v-20260517T021446Z`

The initial SMRF table used 32 calibration samples and compared against
nearby existing PQ points. The rigorous follow-up generated fresh standard
PQ assignments targeted to the same bpp as the selected SMRF points, then
validated SMRF and PQ under the same n=256 contract:

- Dataset: `/dq-runs/calibration/diverse-v1.jsonl`
- Samples: 256
- Sequence length: 1024
- Split/seed: `train` / `42`
- KL scope: `last_token`
- Assignment materialization: `hooks`
- Source prefetch: `require`
- Production cache prefetch: `require`
- Production cache LRU: 64 GiB

Summary:
`/home/rob/dq-runs/qwen3-4b-smrf-rigorous-l3v-20260517T021446Z/artifacts/smrf_vs_pq_matched_n256_summary.json`

Validation output:
`/home/rob/dq-runs/qwen3-4b-smrf-rigorous-l3v-20260517T021446Z/artifacts/smrf_vs_pq_matched_n256_kl.json`

Log:
`/home/rob/dq-runs/qwen3-4b-smrf-rigorous-l3v-20260517T021446Z/logs/smrf_vs_pq_matched_n256_kl.log`

| target | SMRF bpp | SMRF KL | matched PQ bpp | matched PQ KL | relative KL delta |
|---|---:|---:|---:|---:|---:|
| 5.34 | 5.340494 | 0.116258 | 5.339344 | 0.135154 | -13.98% |
| 7.66 | 7.658315 | 0.051592 | 7.654216 | 0.050425 | +2.32% |
| 8.24 | 8.238104 | 0.040593 | 8.235511 | 0.040364 | +0.57% |

Conclusion: the robust SMRF win is currently the 5.34 bpp point. The
higher-bpp points did not survive matched n=256 validation; they are PQ
ties or slight regressions.

## L3V expansion

The L3V expansion anchored on the validated SMRF 5.34 bpp assignment:

`/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_candidates/assignments/smrf_004_bpp_5.3405_c98fe606c384.json`

L3V measured a bounded, fused-sibling-complete neighborhood:

- Selection: selective L3 neighborhood, expanded to full serving groups
- Measured entries: 33 linears
- Serving groups: 15
- Format entries: 132
- Calibration: 16 samples, sequence length 1024
- Runtime: 365.1 seconds
- Source prefetch: `require`

L3V propagated-cost payload:
`/home/rob/dq-runs/qwen3-4b-smrf-rigorous-l3v-20260517T021446Z/artifacts/l3v_smrf_5p34_propagated_costs.pkl`

L3V measurement summary:
`/home/rob/dq-runs/qwen3-4b-smrf-rigorous-l3v-20260517T021446Z/artifacts/l3v_smrf_5p34_measure_summary.json`

L3V SMRF candidate manifest:
`/home/rob/dq-runs/qwen3-4b-smrf-rigorous-l3v-20260517T021446Z/artifacts/l3v_smrf_5p34_candidates/manifest.json`

L3V validation output:
`/home/rob/dq-runs/qwen3-4b-smrf-rigorous-l3v-20260517T021446Z/artifacts/l3v_smrf_5p34_candidates_n256_kl.json`

L3V summary:
`/home/rob/dq-runs/qwen3-4b-smrf-rigorous-l3v-20260517T021446Z/artifacts/l3v_smrf_5p34_summary.json`

| candidate | bpp | KL | formats |
|---|---:|---:|---|
| l3v_000 | 5.142902 | 0.155188 | 219 NVFP4, 19 FP8_E4M3, 14 BF16 |
| l3v_001 | 5.235998 | 0.157041 | 214 NVFP4, 20 FP8_E4M3, 18 BF16 |
| l3v_002 | 5.322664 | 0.146246 | 204 NVFP4, 25 FP8_E4M3, 23 BF16 |
| l3v_003 | 5.412067 | 0.126635 | 202 NVFP4, 21 FP8_E4M3, 29 BF16 |
| l3v_004 | 5.441604 | 0.138085 | 201 NVFP4, 19 FP8_E4M3, 32 BF16 |
| l3v_005 | 5.491387 | 0.124739 | 198 NVFP4, 19 FP8_E4M3, 35 BF16 |
| l3v_006 | 5.574360 | 0.138475 | 194 NVFP4, 19 FP8_E4M3, 39 BF16 |
| l3v_007 | 5.667456 | 0.118179 | 189 NVFP4, 20 FP8_E4M3, 43 BF16 |
| l3v_008 | 5.790088 | 0.123765 | 186 NVFP4, 19 FP8_E4M3, 47 BF16 |

Best L3V point was `l3v_007` at 5.667456 bpp / 0.118179 KL. It is
`+0.001921` KL worse than the original SMRF 5.34 anchor, a `1.65%`
relative regression while using more bits. This L3V attempt therefore
does not replace the current SMRF 5.34 candidate.

Implementation note: `smrf_runtime` now accepts L3 propagated-cost
pickles through `--l3-costs` plus `--baseline-assignment`. Unmeasured
linears are frozen to the baseline assignment so bpp remains reported
over the same quantizable parameter set, and profile-owned fused siblings
are aggregated before SMRF search.

## Commands

Candidate generation ran inside the CUDA docket image
`vllm-eugr-v020:latest`:

```bash
python3 -m prismaquant.research_components.smrf_runtime generate \
  --probe /dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/probe.pkl \
  --costs /dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/cost.pkl \
  --model /hfcache/Qwen3-4B \
  --formats NVFP4,MXFP8_E4M3,FP8_E4M3,BF16 \
  --target-profile vllm_qwen3_5_packed_moe \
  --bpp-min 4.5 \
  --bpp-max 8.25 \
  --n-lambdas 41 \
  --bit-precision-bpp 0.02 \
  --beam-per-bin 2 \
  --validation-candidates 17 \
  --write-payload /dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_cost_payload.json \
  --output-dir /dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_candidates
```

Validation used:

```bash
python3 -m prismaquant.validate_assignments_kl \
  --model /hfcache/Qwen3-4B \
  --probe /dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/probe.pkl \
  --costs /dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/cost.pkl \
  --base-assignment /dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/layer_config.json \
  --formats NVFP4,MXFP8_E4M3,FP8_E4M3,BF16 \
  --assignment smrf_004=/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_candidates/assignments/smrf_004_bpp_5.3405_c98fe606c384.json \
  --output /dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_validated_kl.json \
  --dataset /dq-runs/calibration/diverse-v1.jsonl \
  --n-calib-samples 32 \
  --calib-seqlen 1024 \
  --calib-split train \
  --calib-seed 42 \
  --kl-scope last_token \
  --assignment-materialization hooks \
  --production-weight-cache /dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/production_weight_cache_frontier_raw.pkl \
  --production-cache-dir-override /dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/production_weight_cache_frontier \
  --production-cache-lru-gb 64 \
  --production-cache-prefetch require \
  --production-cache-prefetch-workers 4 \
  --source-prefetch require \
  --source-prefetch-headroom-gb 24
```

Final summary:
`/home/rob/dq-runs/qwen3-4b-smrf-20260517T000429Z/artifacts/smrf_summary.json`
