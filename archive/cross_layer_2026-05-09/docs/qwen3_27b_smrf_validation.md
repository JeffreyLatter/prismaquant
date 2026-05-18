# Qwen3.6-27B SMRF validation

Date: 2026-05-17
Branch: `clado-plugin-integration`
Run: `/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z`

Status: 27B was validated before 35B. The low-budget SMRF point is now
a credible 5.24 bpp candidate, but SMRF should not be promoted as a
general 27B/default method. A stronger n=256 in-place KL rerun favored
the low-bpp SMRF point over matched PQ, vLLM ToolEvalBench tied matched
PQ, and held-out WikiText PPL slightly favored matched PQ. High-bpp SMRF
still lost to standard PQ.

## Inputs

- Model: `/hfcache/qwen36-27b-bf16`
- Probe:
  `/dq-runs/qwen36-27b-halooff-prismaclip-frontier-20260513T005948Z/artifacts/probe.pkl`
- Costs:
  `/dq-runs/qwen36-27b-halooff-prismaclip-frontier-20260513T005948Z/artifacts/cost.pkl`
- Base assignment:
  `/dq-runs/qwen36-27b-halooff-prismaclip-frontier-20260513T005948Z/artifacts/layer_config_nvfp4_mxfp8_only.json`
- Production cache:
  `/dq-runs/qwen3p6-27b-kl-probe-triad-n64-production-20260508T032958Z-directpy/production_weight_cache_nvfp4_mxfp8.pkl`
- Production cache dir:
  `/dq-runs/qwen3p6-27b-kl-probe-triad-n64-production-20260508T032958Z-directpy/production_weight_cache`
- Dataset: `/dq-runs/calibration/diverse-v1.jsonl`
- Sequence length: 1024
- Split/seed: `train` / `42`
- KL scope: `last_token`
- Production/source prefetch: `require`

## Fixes made during validation

Two accounting/legalization issues were fixed before interpreting the
27B results:

- Allocator Pareto assignment JSONs now apply the final MTP BF16 override.
  The previous Pareto seed artifacts could quantize MTP even when the
  final `layer_config` did not.
- Allocator bpp budgeting now excludes forced BF16 MTP and pinned names
  from the DP denominator. MTP BF16 remains in assignment JSONs for export
  coverage; `lm_head` is excluded from budgeting and omitted from
  assignments.
- `validate_assignments_kl` now reports bpp over quantizable parameters
  only, excluding pinned names and BF16 source-passthrough entries.

Tests:

```bash
python3 -m pytest tests/test_allocator_pareto_seed_export.py \
  tests/test_validate_assignments_kl_bpp.py \
  tests/test_smrf_runtime.py -q
```

Result: `9 passed`.

## Candidate generation

SMRF candidates:

```bash
python3 -m prismaquant.research_components.smrf_runtime \
  --probe /dq-runs/qwen36-27b-halooff-prismaclip-frontier-20260513T005948Z/artifacts/probe.pkl \
  --costs /dq-runs/qwen36-27b-halooff-prismaclip-frontier-20260513T005948Z/artifacts/cost.pkl \
  --model /hfcache/qwen36-27b-bf16 \
  --formats NVFP4,MXFP8_E4M3,BF16 \
  --target-profile vllm_packed_moe
```

SMRF manifest:
`/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/artifacts/smrf_candidates_profile_filtered/manifest.json`

Corrected matched PQ generation:

```bash
python3 -m prismaquant.allocator \
  --probe /dq-runs/qwen36-27b-halooff-prismaclip-frontier-20260513T005948Z/artifacts/probe.pkl \
  --costs /dq-runs/qwen36-27b-halooff-prismaclip-frontier-20260513T005948Z/artifacts/cost.pkl \
  --model-override /hfcache/qwen36-27b-bf16 \
  --formats NVFP4,MXFP8_E4M3,BF16 \
  --target-profile vllm_packed_moe \
  --pareto-output-dir /dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/artifacts/pq_matched_profile_filtered_budgetlegal2/assignments \
  --visual-format BF16 \
  --mtp-format BF16
```

PQ manifest:
`/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/artifacts/pq_matched_profile_filtered_budgetlegal2/assignments/manifest.json`

Generation log:
`/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/logs/pq_matched_generate_budgetlegal2.log`

## n=16 screen

The broad n=16 hook screen was useful for candidate discovery but was not
stable enough to call a winner.

- SMRF output:
  `/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/artifacts/smrf_vs_pq_n16_kl_no_frozen.json`
- Corrected PQ output:
  `/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/artifacts/pq_budgetlegal2_n16_kl.json`
- Logs:
  `/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/logs/validate_smrf_vs_pq_n16_no_frozen.log`
  and
  `/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/logs/validate_pq_budgetlegal2_n16.log`

The hook path required `--disable-frozen-weight-cache`; otherwise the
frozen hook cache exceeded the 27B GPU budget after production cache
prefetch.

## n=64 in-place finalists

Validation used:

```bash
python3 -m prismaquant.validate_assignments_kl \
  --assignment-materialization inplace \
  --n-calib-samples 64 \
  --calib-seqlen 1024 \
  --production-cache-prefetch require \
  --source-prefetch require
```

Output dir:
`/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/artifacts/finalists_n64_inplace`

Log dir:
`/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/logs/finalists_n64_inplace`

| assignment | bpp | KL | notes |
|---|---:|---:|---|
| smrf_003 | 5.241438 | 0.015877 | SMRF beat PQ at same bpp under n=64 |
| pq_003 | 5.241576 | 0.023054 | matched PQ |
| smrf_004 | 5.422778 | 0.016954 | matched 5.42 SMRF |
| pq_004 | 5.422327 | 0.014639 | best low-budget n=64 point |
| smrf_010 | 7.128593 | 0.012152 | high-budget SMRF |
| pq_010 | 7.125740 | 0.009745 | best n=64 point |
| smrf_014 | 8.248304 | 0.011492 | high-budget SMRF |
| pq_014 | 8.247820 | 0.015358 | matched PQ |

## n=128 in-place finalists

Validation used the same command shape with `--n-calib-samples 128`.

Output dir:
`/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/artifacts/finalists_n128_inplace`

Log dir:
`/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/logs/finalists_n128_inplace`

| assignment | bpp | KL | notes |
|---|---:|---:|---|
| smrf_003 | 5.241438 | 0.023765 | practical-budget SMRF did not hold up |
| pq_003 | 5.241576 | 0.016902 | matched practical-budget PQ |
| pq_004 | 5.422327 | 0.023149 | regressed at n=128 |
| pq_010 | 7.125740 | 0.013182 | best tested n=128 point |
| smrf_012 | 7.665986 | 0.018325 | SMRF high-budget check |
| pq_012 | 7.665841 | 0.011841 | matched high-budget PQ |

## n=256 in-place finalists

Validation used the same production-rendered in-place path with
`--n-calib-samples 256`. The first attempt failed fast because the
required production-cache file prefetch size was 42.54 GiB and the
derived budget was 42.45 GiB. The rerun set
`--production-cache-file-prefetch-max-gb 48`, preserving required
resident prefetch while avoiding the near-miss budget failure.

Output dir:
`/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/artifacts/finalists_n256_inplace`

Log dir:
`/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/logs/finalists_n256_inplace`

| assignment | bpp | KL | notes |
|---|---:|---:|---|
| smrf_003 | 5.241438 | 0.019005 | low-bpp SMRF beat matched PQ by KL |
| pq_003 | 5.241576 | 0.021408 | matched low-bpp PQ |
| pq_004 | 5.422327 | 0.027676 | higher-bpp PQ bracket regressed |
| pq_010 | 7.125740 | 0.015735 | best tested n=256 KL point |
| smrf_012 | 7.665986 | 0.017916 | high-bpp SMRF lost despite more bits than pq_010 |
| pq_012 | 7.665841 | 0.015035 | matched high-bpp PQ beat SMRF |

At the practical 5.24 bpp target, SMRF improved KL by 0.002403
absolute, or about 11.23% relative to matched PQ. That KL win did not
hold for the high-bpp point: matched PQ improved KL by 0.002881 absolute,
or about 19.16% relative to SMRF.

## vLLM export and PPL

The low-bpp SMRF and matched PQ assignments were expanded over the same
base assignment and checked for runtime legality before export:

- SMRF layer config:
  `/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/artifacts/vllm_low_bpp_inputs/smrf_003_layer_config_full.json`
- PQ layer config:
  `/home/rob/dq-runs/qwen36-27b-smrf-standard-20260517T000000Z/artifacts/vllm_low_bpp_inputs/pq_003_layer_config_full.json`
- Runtime coercions: `0` for both.

Export run:
`/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z`

Exports used the same production weight cache direct path:

- SMRF: 370 NVFP4, 4 MXFP8, 240 BF16 assignment entries.
- PQ: 364 NVFP4, 250 BF16 assignment entries.
- Each artifact wrote 6 safetensors shards, about 21.25 GiB.
- Export logs:
  `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/logs/export_smrf_003.log`
  and
  `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/logs/export_pq_003.log`

Both artifacts passed vLLM eager and graph-mode load/generation with
`quantization=compressed-tensors`. SMRF selected
`FlashInferCutlassNvFp4LinearKernel` and
`FlashInferCutlassMxfp8LinearKernel`; PQ selected
`FlashInferCutlassNvFp4LinearKernel`. A stale optional
`prismaquant_residual_adapter` plugin entry logged an import error, but
vLLM continued and completed generation in all four smokes.

Smoke logs:

| artifact | eager log | graph log |
|---|---|---|
| smrf_003 | `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/logs/validate_native_export_smrf_003_eager.log` | `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/logs/validate_native_export_smrf_003_graph.log` |
| pq_003 | `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/logs/validate_native_export_pq_003_eager.log` | `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/logs/validate_native_export_pq_003_graph.log` |

WikiText PPL was measured through vLLM with compressed-tensors, sequence
length 512, graph mode, and `FLASHINFER_DISABLE_VERSION_CHECK=1`.
The vLLM container did not include `datasets`, so the ephemeral
validation container installed `datasets==4.6.0` before running PPL.

| artifact | tokens scored | mean NLL | PPL | output |
|---|---:|---:|---:|---|
| smrf_003 32k | 32,704 | 2.135831 | 8.464077 | `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/smrf_003/wikitext_ppl_32k.json` |
| pq_003 32k | 32,704 | 2.132949 | 8.439722 | `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/pq_003/wikitext_ppl_32k.json` |
| smrf_003 64k | 65,408 | 2.151340 | 8.596369 | `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/smrf_003/wikitext_ppl_64k.json` |
| pq_003 64k | 65,408 | 2.149144 | 8.577513 | `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/pq_003/wikitext_ppl_64k.json` |

The 64k PPL run favored PQ by 0.018857 PPL, a 0.22% relative SMRF
regression. The smaller 32k run also favored PQ by 0.024355 PPL, a
0.29% relative SMRF regression.

## vLLM ToolEvalBench

ToolEvalBench is now part of the standard vLLM downstream suite alongside
PPL and log-likelihood style checks. The benchmark was run against both
materialized low-bpp artifacts using the same sequential hardmode shape
as the stored 27B ToolEvalBench runs:

```bash
tool-eval-bench \
  --backend vllm \
  --base-url http://localhost:8000/v1 \
  --temperature 0 \
  --seed 1234 \
  --timeout 180 \
  --no-think \
  --hardmode \
  --parallel 1
```

Serving settings matched the previous 27B ToolEvalBench convention:
`vllm-fresh-b12x-fla:latest`, compressed-tensors, graph mode,
`--max-model-len 32768`, FP8 KV cache, prefix caching, tool choice
enabled, `qwen3_coder` tool parser, `qwen3` reasoning parser, and
safetensors prefetch. Both artifacts selected performant kernels:
SMRF used `FlashInferCutlassNvFp4LinearKernel` and
`FlashInferCutlassMxfp8LinearKernel`; PQ used
`FlashInferCutlassNvFp4LinearKernel`.

| artifact | score | points | pass / partial / fail | responsiveness | deployability | report |
|---|---:|---:|---:|---:|---:|---|
| smrf_003 | 87 / 100 | 129 / 148 | 60 / 9 / 5 | 32 / 100 | 70 / 100 | `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/tooleval_smrf_003/runs/2026/05/2026-05-17T18-06-25Z_7e3bd4.md` |
| pq_003 | 87 / 100 | 129 / 148 | 60 / 9 / 5 | 29 / 100 | 70 / 100 | `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/tooleval_pq_003/runs/2026/05/2026-05-17T18-31-35Z_a9259a.md` |

Logs:

- SMRF ToolEvalBench log:
  `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/tooleval_smrf_003/logs/tooleval_full_hardmode_seq.log`
- SMRF vLLM server log:
  `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/tooleval_smrf_003/logs/vllm_server_final.log`
- PQ ToolEvalBench log:
  `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/tooleval_pq_003/logs/tooleval_full_hardmode_seq.log`
- PQ vLLM server log:
  `/home/rob/dq-runs/qwen36-27b-smrf-vllm-20260517T124500Z/tooleval_pq_003/logs/vllm_server_final.log`

The headline ToolEvalBench result is a tie. The failures are not
identical:

- SMRF was better on `TC-03`, `TC-31`, `TC-50`, and `TC-57`.
- PQ was better on `TC-29`, `TC-42`, `TC-52`, and `TC-74`.
- Both failed `TC-48`, `TC-60`, and `TC-72`.
- Both triggered the critical `TC-60` cross-turn sleeper-injection
  failure.

## Conclusion

Do not promote SMRF as a general/default Qwen3.6-27B method yet, but keep
the low-budget 5.24 bpp SMRF point alive as a serious candidate.

At the low-budget point, SMRF improved n=256 KL by 0.002403 absolute
against matched PQ, or about 11.23% relative. Its held-out WikiText PPL
was slightly worse than matched PQ, but the 64k regression was 0.22% and
the 32k regression was 0.29%, both below the usual 0.5% preservation
tolerance. ToolEvalBench tied matched PQ exactly at 87 / 100 and
129 / 148 points.

That is enough evidence to say the low-budget SMRF point is competitive.
It is not enough evidence to say SMRF is globally better: the high-bpp
SMRF point lost to matched PQ and to a lower-bpp standard PQ point, and
both low-budget artifacts still failed the same critical sleeper-injection
ToolEvalBench case. Before 35B, use corrected standard PQ as the default
baseline and treat low-budget SMRF as a candidate arm that must be
validated with the full standard suite: n=256-or-stronger KL, PPL/mean
NLL, log-likelihood downstream tasks, ToolEvalBench, and vLLM eager/graph
materialization checks.

## Refined solver rerun

Run:
`/home/rob/dq-runs/qwen36-27b-smrf-refined-20260517T204108Z`

After the initial 27B result, SMRF candidate generation was updated to:

- weight Lagrangian penalties by each unit's true bpp contribution rather
  than raw `bits_per_param`;
- use slope-derived lambda coverage instead of a coarse linear lambda grid;
- re-check serving legality per fused member after aggregation;
- keep profile-pinned or passthrough members out of aggregated SMRF units;
- emit richer archive diagnostics, block format histograms, hamming distance
  from injected baselines, and included matched-PQ rows;
- support KL repeat summaries and UCB fields in validation output;
- support practical-knee, UCB, rank-correlation, and leave-one-out kneedle
  diagnostics in measured-frontier selection.

Targeted tests:

```bash
python3 -m pytest tests/test_smrf_runtime.py \
  tests/test_select_validated_frontier.py \
  tests/test_validate_assignments_kl_bpp.py -q
```

Result: `17 passed`.

Refined candidate generation used the same 27B probe/cost/base inputs as
above, with `n_lambdas=81`, `bit_precision_bpp=0.01`,
`beam_per_bin=4`, `validation_candidates=33`, and injected matched PQ
assignments for `pq_003`, `pq_004`, and `pq_010`.

Archive:
`/home/rob/dq-runs/qwen36-27b-smrf-refined-20260517T204108Z/artifacts/smrf_candidates_refined/smrf_archive.json`

Generation log:
`/home/rob/dq-runs/qwen36-27b-smrf-refined-20260517T204108Z/logs/smrf_generate_refined.log`

Generation summary:

- Generated archive rows: 1,530.
- Surrogate frontier points: 583.
- Max surrogate-frontier bpp gap: 0.026712.
- Mean surrogate-frontier bpp gap: 0.006440.
- Validation manifest rows: 34, including three matched PQ baselines.
- DP retained states: 4,596 across 1,149 final bins.

The refined search selected a low-budget lambda candidate,
`smrf_005`, at 5.233916 bpp. A 2x64 screen showed high variance for
this candidate:

| assignment | bpp | KL repeats | mean KL | UCB |
|---|---:|---:|---:|---:|
| smrf_004 | 5.214804 | 0.017660, 0.051021 | 0.034341 | 0.051021 |
| smrf_005 | 5.233916 | 0.015236, 0.026851 | 0.021044 | 0.026851 |
| pq_003 | 5.241576 | 0.017640, 0.016271 | 0.016955 | 0.017640 |

The final apples-to-apples check used the same no-FLA CUDA docket image
as the earlier KL logs, `vllm-eugr-v020:latest`, because the FLA-enabled
container takes a different model execution path. Validation used
`--n-calib-samples 256`, sequence length 1024, `last_token` KL,
in-place materialization, and required source/production-cache prefetch.

Final n=256 result:

| assignment | bpp | KL | output MSE | formats | log |
|---|---:|---:|---:|---|---|
| smrf_005 | 5.233916 | 0.022896 | 2.452648 | 250 BF16, 364 NVFP4 | `/home/rob/dq-runs/qwen36-27b-smrf-refined-20260517T204108Z/logs/final_n256_eugr_inplace/smrf_005.log` |
| pq_003 | 5.241576 | 0.021408 | 2.396772 | 250 BF16, 364 NVFP4 | `/home/rob/dq-runs/qwen36-27b-smrf-refined-20260517T204108Z/logs/final_n256_eugr_inplace/pq_003.log` |

Selection summary:
`/home/rob/dq-runs/qwen36-27b-smrf-refined-20260517T204108Z/artifacts/final_n256_eugr_inplace/selected_best_summary.json`

Result: the refined low-budget SMRF candidate did not beat matched PQ.
`pq_003` improved KL by 0.001488 absolute versus `smrf_005`, a 6.50%
relative reduction from `smrf_005`. The corrected/wider solver therefore
does not replace the previous low-budget SMRF result, and it weakens the
case for promoting SMRF as a default 27B method.

## MXFP8 ablation and warmed throughput

Run:
`/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z`

The original low-budget `smrf_003` used four MXFP8 linears:

- `model.layers.13.mlp.down_proj`
- `model.layers.19.self_attn.o_proj`
- `model.layers.27.self_attn.o_proj`
- `model.layers.31.self_attn.o_proj`

An all-NVFP4 ablation replaced those four entries with NVFP4. The full
export layer config then had `240 BF16 / 374 NVFP4` entries and no MXFP8.
The exporter runtime-legality audit reported zero coercions.

Artifacts:

- Assignment:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/artifacts/smrf_003_all_nvfp4_assignment_e7ab016bc017.json`
- Layer config:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/artifacts/smrf_003_all_nvfp4_layer_config_aec02484caa4.json`
- Export:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/smrf_003_all_nvfp4/exported`

n=256 no-FLA KL replay used the same production-cache in-place path as the
earlier final checks:

| assignment | bpp | KL | output MSE | formats |
|---|---:|---:|---:|---|
| smrf_003 mixed | 5.241438 | 0.019005 | - | 240 BF16, 370 NVFP4, 4 MXFP8 |
| smrf_003 all-NVFP4 | 5.213179 | 0.020139 | 2.772540 | 240 BF16, 374 NVFP4 |
| pq_003 | 5.241576 | 0.021408 | 2.396772 | 250 BF16, 364 NVFP4 |

The all-NVFP4 ablation gives back 0.001134 KL versus mixed SMRF, but it
still beats matched PQ by 0.001269 absolute while using slightly fewer
bits. This keeps the low-budget SMRF neighborhood interesting, but it also
shows that the four MXFP8 choices were not just runtime noise; they carried
some KL value.

The all-NVFP4 export completed through the production-cache direct path in
five safetensors shards. It passed vLLM eager and graph-mode load/generation
with `quantization=compressed-tensors` and selected only
`FlashInferCutlassNvFp4LinearKernel`.

Logs:

- KL:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/logs/kl_n256_eugr_smrf_003_all_nvfp4.log`
- Export:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/logs/export_smrf_003_all_nvfp4.log`
- vLLM eager smoke:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/logs/validate_native_export_smrf_003_all_nvfp4_eager.log`
- vLLM graph smoke:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/logs/validate_native_export_smrf_003_all_nvfp4_graph.log`

The earlier PPL wall-clock comparison made SMRF look much slower, but that
measurement mixed model init, profiling, and first-run behavior into the
elapsed time. A separate warmed graph-mode throughput benchmark used
case-specific warmups and reported median steady-state throughput:

| artifact | case | prompt tokens | median wall s | decode tok/s | total tok/s |
|---|---|---:|---:|---:|---:|
| pq_003 | decode_128 | 12 | 11.204 | 11.425 | 12.496 |
| smrf_003 mixed | decode_128 | 12 | 11.214 | 11.415 | 12.485 |
| smrf_003 all-NVFP4 | decode_128 | 12 | 11.091 | 11.541 | 12.623 |
| pq_003 | prefill_1k_decode_32 | 1540 | 3.305 | 9.682 | 475.633 |
| smrf_003 mixed | prefill_1k_decode_32 | 1540 | 3.301 | 9.693 | 476.156 |
| smrf_003 all-NVFP4 | prefill_1k_decode_32 | 1540 | 3.273 | 9.776 | 480.247 |
| pq_003 | prefill_2k_decode_32 | 3080 | 3.884 | 8.239 | 801.250 |
| smrf_003 mixed | prefill_2k_decode_32 | 3080 | 3.941 | 8.119 | 789.617 |
| smrf_003 all-NVFP4 | prefill_2k_decode_32 | 3080 | 3.920 | 8.164 | 793.918 |

Throughput summary:
`/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/artifacts/throughput_graph/summary.json`

Interpretation: the 2x slowdown did not reproduce in steady-state serving.
Mixed SMRF has extra init/autotune overhead from the MXFP8 kernel path, but
steady-state throughput is within about 1.5% of PQ on these shapes.
All-NVFP4 removes the MXFP8 init/autotune path and is slightly faster than
PQ on short decode and the 1.5k prompt case, but still about 0.9% slower
than PQ on the 3k prompt case.

Held-out WikiText 64k PPL for all-NVFP4:

| artifact | tokens scored | mean NLL | PPL |
|---|---:|---:|---:|
| pq_003 64k | 65,408 | 2.149144 | 8.577513 |
| smrf_003 mixed 64k | 65,408 | 2.151340 | 8.596369 |
| smrf_003 all-NVFP4 64k | 65,408 | 2.150811 | 8.591824 |

All-NVFP4 improves PPL slightly versus mixed SMRF, but still trails PQ by
0.014311 PPL, a 0.17% relative regression. That is under the usual 0.5%
preservation tolerance, but it is directionally worse despite better KL.

Recommendation: do not promote SMRF as a default allocator replacement yet.
The low-budget SMRF neighborhood remains viable because all-NVFP4 beats PQ
on n=256 KL at lower bpp and no longer has a material steady-state runtime
problem. The next useful SMRF step is validation-guided local search around
the old mixed `smrf_003`, the all-NVFP4 ablation, and matched `pq_003`,
with a gate that considers both repeat-stable KL and held-out PPL. The
allocator should also keep serving-kernel mix as an explicit runtime cost or
constraint rather than treating MXFP8 and NVFP4 as interchangeable once both
are shape-legal.

## MXFP8 subset search

The next search enumerated all 16 subsets of the four MXFP8 choices in the
original low-budget `smrf_003`:

- bit 0: `model.layers.13.mlp.down_proj`
- bit 1: `model.layers.19.self_attn.o_proj`
- bit 2: `model.layers.27.self_attn.o_proj`
- bit 3: `model.layers.31.self_attn.o_proj`

Artifacts:

- Manifest:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/artifacts/mxfp8_subset_assignments/manifest.json`
- n=64 hook screen:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/artifacts/mxfp8_subset_screen_n64_hooks/results.json`
- n=256 in-place confirmations:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/artifacts/mxfp8_subset_n256_eugr_remaining/`
- n=256 summary:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/artifacts/mxfp8_subset_n256_summary.json`

The n=64 hook screen was not predictive enough for promotion. It ranked
`mxsubset_05_2mxfp8` first at KL 0.014636, but the n=256 in-place
confirmation for that same assignment was KL 0.025287. Because in-place
materialization is destructive, each n=256 subset was run as a separate
validator invocation with a clean model load.

n=256 no-FLA subset results:

| assignment | enabled MXFP8 bits | bpp | KL | output MSE |
|---|---:|---:|---:|---:|
| `mxsubset_14_3mxfp8` | 1,2,3 | 5.227712 | 0.018457 | 2.772040 |
| `smrf_003 mixed` | 0,1,2,3 | 5.241438 | 0.019005 | - |
| `mxsubset_11_3mxfp8` | 0,1,3 | 5.236594 | 0.020127 | 2.772220 |
| `smrf_003 all-NVFP4` | none | 5.213179 | 0.020139 | 2.772540 |
| `mxsubset_01_1mxfp8` | 0 | 5.226905 | 0.020659 | 2.772530 |
| `mxsubset_09_2mxfp8` | 0,3 | 5.231749 | 0.021033 | 2.772240 |
| `mxsubset_13_3mxfp8` | 0,2,3 | 5.236594 | 0.021131 | 2.772060 |
| `pq_003` | matched PQ | 5.241576 | 0.021408 | 2.396772 |
| `mxsubset_10_2mxfp8` | 1,3 | 5.222868 | 0.022382 | 2.772230 |
| `mxsubset_04_1mxfp8` | 2 | 5.218023 | 0.022515 | 2.772350 |
| `mxsubset_07_3mxfp8` | 0,1,2 | 5.236594 | 0.022650 | 2.772330 |
| `mxsubset_12_2mxfp8` | 2,3 | 5.222868 | 0.023074 | 2.772060 |
| `mxsubset_03_2mxfp8` | 0,1 | 5.231749 | 0.023240 | 2.772510 |
| `mxsubset_02_1mxfp8` | 1 | 5.218023 | 0.024589 | 2.772510 |
| `mxsubset_05_2mxfp8` | 0,2 | 5.231749 | 0.025287 | 2.772347 |
| `mxsubset_08_1mxfp8` | 3 | 5.218023 | 0.031349 | 2.772250 |
| `mxsubset_06_2mxfp8` | 1,2 | 5.222868 | 0.031406 | 2.772330 |

`mxsubset_14_3mxfp8` is the best n=256 KL point in this neighborhood. It
removes the layer 13 MLP MXFP8 choice from the original mixed SMRF point,
keeps MXFP8 on the three attention `o_proj` entries, reduces bpp by
0.013726, and improves KL by 0.000548 absolute versus original mixed SMRF.

Runtime legality and export:

- Layer config:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/artifacts/mxsubset_14_3mxfp8_layer_config.json`
- Export:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/mxsubset_14_3mxfp8/exported`
- Export log:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/logs/export_mxsubset_14_3mxfp8.log`

The runtime-legality audit used the `qwen3_5_dense` model profile and
reported zero coercions. The exported recipe was `240 BF16 / 371 NVFP4 /
3 MXFP8`, with 371 NVFP4 and 3 MXFP8 linears served from the existing
production weight cache.

vLLM eager and graph-mode smokes both passed with
`quantization=compressed-tensors`. The logs show
`FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM and
`FlashInferCutlassMxfp8LinearKernel` for MXFP8 GEMM.

- Eager smoke:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/logs/validate_native_export_mxsubset_14_3mxfp8_eager.log`
- Graph smoke:
  `/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/logs/validate_native_export_mxsubset_14_3mxfp8_graph.log`

Held-out WikiText 64k PPL rejected the KL-only winner:

| artifact | tokens scored | mean NLL | PPL |
|---|---:|---:|---:|
| pq_003 64k | 65,408 | 2.149144 | 8.577513 |
| smrf_003 mixed 64k | 65,408 | 2.151340 | 8.596369 |
| smrf_003 all-NVFP4 64k | 65,408 | 2.150811 | 8.591824 |
| mxsubset_14_3mxfp8 64k | 65,408 | 2.152858 | 8.609431 |

`mxsubset_14_3mxfp8` worsens PPL by 0.031918 versus matched PQ, about
0.37% relative. It also worsens PPL versus both the original mixed SMRF
artifact and the all-NVFP4 ablation. This is still inside a loose 0.5%
preservation bound, but the direction is wrong and the degradation is
larger than the earlier SMRF variants.

Warmed graph-mode throughput for `mxsubset_14_3mxfp8` was also weaker than
PQ on the tested shapes:

| artifact | case | prompt tokens | median wall s | decode tok/s | total tok/s |
|---|---|---:|---:|---:|---:|
| mxsubset_14_3mxfp8 | decode_128 | 12 | 11.422 | 11.206 | 12.257 |
| mxsubset_14_3mxfp8 | prefill_1k_decode_32 | 1540 | 3.354 | 9.540 | 468.637 |
| mxsubset_14_3mxfp8 | prefill_2k_decode_32 | 3080 | 3.990 | 8.021 | 780.042 |

Throughput result:
`/home/rob/dq-runs/qwen36-27b-smrf-mxfp8-ablation-20260518T000000Z/artifacts/throughput_graph/mxsubset_14_3mxfp8.json`

Conclusion: do not promote `mxsubset_14_3mxfp8` despite its n=256 KL win.
This is a concrete example where last-token KL on the calibration contract
is not sufficient as the sole acceptance metric. The SMRF acceptance gate
should require a repeat-stable KL improvement and a non-regressing held-out
PPL/mean-NLL result before any ToolEvalBench spend or 27B-to-35B scale-up.
For low-budget 27B, the most defensible SMRF artifact remains the
all-NVFP4 ablation as a research checkpoint: it keeps lower bpp and a KL
edge over matched PQ without the extra MXFP8 serving path, but it still does
not beat PQ on held-out PPL.
