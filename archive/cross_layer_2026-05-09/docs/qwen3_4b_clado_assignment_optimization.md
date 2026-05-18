# Qwen3-4B CLADO assignment optimization

Date: 2026-05-16
Branch: `clado-plugin-integration`
Run: `/home/rob/dq-runs/qwen3-4b-clado-assign-opt-20260516T053432Z`

Note: the initial results below predate runtime-legal fused-sibling
generation. The legal rerun in `Legal rerun vs standard PQ` supersedes
the apparent CLADO win.

## Gate choice

For CLADO promotion, use measured teacher-student KL on the production
cache path as the gate.  The original CLADO paper gates against task
loss / top-1 accuracy on a sensitivity set; for PrismaQuant text models
the closest production contract is last-token KL on the same calibration
set used by the allocator and validator.

Local MSE and the CLADO four-term surrogate are proposal generators, not
promotion gates.  In this run the surrogate frontier was jagged enough
that adjacent candidates at similar bpp had very different real KL.

Calibration contract used here:

- Dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- Sequence length: 1024
- Split/seed: `train` / `42`
- KL scope: `last_token`
- Production cache:
  `/home/rob/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/production_weight_cache_frontier_raw.pkl`
- Cache dir override:
  `/home/rob/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/production_weight_cache_frontier`
- Production cache LRU: 64 GiB
- Source/cache prefetch: required

## Result

Best measured assignment:

- Assignment:
  `/home/rob/dq-runs/qwen3-4b-clado-assign-opt-20260516T053432Z/artifacts/assignment_clado_confirm_n64_best_kl.json`
- Layer config:
  `/home/rob/dq-runs/qwen3-4b-clado-assign-opt-20260516T053432Z/artifacts/layer_config_clado_confirm_n64_best_kl.json`
- Validation summary:
  `/home/rob/dq-runs/qwen3-4b-clado-assign-opt-20260516T053432Z/artifacts/clado_confirm_n64_best_kl_selection.json`

On the n=64 confirmation run:

| assignment | bpp | KL | formats |
|---|---:|---:|---|
| current_4p700 | 4.700077 | 0.193891 | 229 NVFP4, 17 FP8_E4M3, 6 BF16 |
| clado_r2_4p711 | 4.711039 | 0.070469 | 232 NVFP4, 20 MXFP8 |
| clado_best_4p737 | 4.736742 | 0.067779 | 231 NVFP4, 21 MXFP8 |

On the n=32 search run, the same selected assignment measured:

| assignment | bpp | KL | formats |
|---|---:|---:|---|
| clado_best_4p737 | 4.736742 | 0.048746 | 231 NVFP4, 21 MXFP8 |

## Search trace

Starting point:

- Old Block-CLADO sweep:
  `/home/rob/dq-runs/qwen3-4b-clado-assign-opt-20260516T053432Z/artifacts/sweep_from_20260506_payload.json`
- Current-format kneedle candidates:
  `/home/rob/dq-runs/qwen3-4b-clado-assign-opt-20260516T053432Z/artifacts/kneedle_candidates_current`
- First validation:
  `/home/rob/dq-runs/qwen3-4b-clado-assign-opt-20260516T053432Z/artifacts/validated_clado_candidates_current_kl.json`

Best candidate from the initial CLADO frontier:

| assignment | bpp | KL | formats |
|---|---:|---:|---|
| clado_4p7236 | 4.747565 | 0.075515 | 230 NVFP4, 22 MXFP8 |

Single-removal ablation then found one promotion to undo:

| change from clado_4p7236 | bpp | KL |
|---|---:|---:|
| drop `model.layers.8.self_attn.q_proj` | 4.736742 | 0.048746 |
| drop `model.layers.8.self_attn.k_proj` | 4.744859 | 0.075184 |

Second-round ablation from the improved center did not find a better
single change.  The nearest lower-bpp alternative was:

| assignment | bpp | KL |
|---|---:|---:|
| also drop `model.layers.13.mlp.down_proj` | 4.711039 | 0.069729 |

## vLLM materialization

Run:
`/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z`

The unmodified measured CLADO assignments exported successfully, but the
4.711 artifact failed vLLM eager load because layer 8 mixed formats inside
the q/k/v fused sibling group:

- Export:
  `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p711/exported`
- Log:
  `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p711/logs/validate_native_export_eager.log`
- Failure: `KeyError: 'layers.8.self_attn.qkv_proj.weight'`

For vLLM-serving tests, both assignments were coerced with the same
fused-sibling rule: promote `model.layers.8.self_attn.q_proj` from
`NVFP4` to `MXFP8_E4M3`.

| artifact | exported path | formats after vLLM coercion |
|---|---|---|
| clado_4p711 | `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p711/exported_vllm_fused` | 231 NVFP4, 21 MXFP8 |
| clado_4p737 | `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p737/exported_vllm_fused` | 230 NVFP4, 22 MXFP8 |

Coercion manifests:

- `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p711/artifacts/vllm_fused_manifest.json`
- `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p737/artifacts/vllm_fused_manifest.json`

Both fused artifacts pass eager and graph-mode vLLM smokes with
`quantization=compressed-tensors`.  vLLM selected the expected performant
FlashInfer kernels:

- `FlashInferCutlassMxfp8LinearKernel` for MXFP8 GEMM
- `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM

Smoke logs:

| artifact | eager log | graph log |
|---|---|---|
| clado_4p711 | `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p711/logs/validate_native_export_eager_fused.log` | `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p711/logs/validate_native_export_graph_fused.log` |
| clado_4p737 | `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p737/logs/validate_native_export_eager_fused.log` | `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p737/logs/validate_native_export_graph_fused.log` |

WikiText perplexity was measured through vLLM with compressed-tensors on
the fused artifacts:

| artifact | tokens scored | seq len | mean NLL | PPL | output |
|---|---:|---:|---:|---:|---|
| clado_4p711 fused | 4,088 | 512 | 3.181346 | 24.079143 | `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p711/logs/vllm_wikitext_ppl_4k.json` |
| clado_4p737 fused | 4,088 | 512 | 3.177760 | 23.992950 | `/home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p737/logs/vllm_wikitext_ppl_4k.json` |

The first PPL attempt hit a FlashInfer package-version guard
(`0.6.8` versus `0.6.8.post1`).  The successful PPL runs used
`FLASHINFER_DISABLE_VERSION_CHECK=1`; this did not change the vLLM model
artifact or compressed-tensors metadata.

## Legal rerun vs standard PQ

Run:
`/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z`

Summary artifact:
`/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/artifacts/legal_vs_pq_summary.json`

This rerun generated CLADO candidates with the Qwen3 profile active, so
q/k/v and gate/up fused siblings were solved as legal runtime groups
instead of being coerced after the fact. The selected CLADO kneedle, the
lower-bpp PQ point, and the higher-bpp PQ point all had zero mixed fused
groups across 72 profile-owned fused groups.

Legal candidate generation:

- Sweep:
  `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/artifacts/clado_legal_sweep.json`
- Kneedle:
  `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/artifacts/clado_legal_candidates/kneedle.json`
- Logs:
  `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/logs/clado_legal_sweep.log`,
  `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/logs/clado_legal_kneedle.log`

Standard PQ baselines were reallocated with the same probe/cost table,
the same Qwen3 serving profile, and fused-sibling aggregation enabled.
Because the final validator bpp differs slightly from the CLADO payload
bpp, the final comparison uses PQ points that bracket the measured CLADO
bpp.

KL validation contract:

- Dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- Samples: 256
- Sequence length: 1024
- Split/seed: `train` / `42`
- Scope: `last_token`
- Materialization: `hooks`
- Production cache:
  `/home/rob/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/production_weight_cache_frontier_raw.pkl`
- Cache dir override:
  `/home/rob/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/production_weight_cache_frontier`
- Prefetch: `source=require`, `production-cache=require`, 64 GiB LRU,
  4 prefetch workers
- KL log:
  `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/logs/kl_seed42_n256_selected.log`

Measured KL:

| assignment | bpp | KL | formats |
|---|---:|---:|---|
| PQ lower bracket | 4.743854 | 0.125966 | 229 NVFP4, 19 FP8_E4M3, 4 BF16 |
| CLADO legal kneedle | 4.747565 | 0.143922 | 230 NVFP4, 22 MXFP8 |
| PQ upper bracket | 4.752661 | 0.117171 | 231 NVFP4, 17 FP8_E4M3, 4 BF16 |

Linearly interpolating PQ KL to CLADO's 4.747565 bpp gives `0.122260`.
CLADO is therefore `+0.021662` KL worse, or a `17.72%` relative
regression. This is also bounded without interpolation: CLADO has higher
KL than both bracketing standard PQ assignments.

The same three assignments were exported and loaded through vLLM with
`quantization=compressed-tensors`. vLLM selected performant kernels:

- CLADO: `FlashInferCutlassMxfp8LinearKernel` and
  `FlashInferCutlassNvFp4LinearKernel`
- PQ: `CutlassFP8ScaledMMLinearKernel` and
  `FlashInferCutlassNvFp4LinearKernel`

Export logs:

- `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/logs/export_clado_4p7476.log`
- `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/logs/export_pq_low_4p7439.log`
- `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/logs/export_pq_high_4p7527.log`

WikiText perplexity was measured through vLLM on 32,768 requested tokens,
32,704 scored tokens, sequence length 512:

| assignment | bpp | mean NLL | PPL | output |
|---|---:|---:|---:|---|
| PQ lower bracket | 4.743854 | 2.975143 | 19.592430 | `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/vllm/pq_low_4p7439/wikitext_ppl_32k.json` |
| CLADO legal kneedle | 4.747565 | 3.066320 | 21.462766 | `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/vllm/clado_4p7476/wikitext_ppl_32k.json` |
| PQ upper bracket | 4.752661 | 2.985381 | 19.794036 | `/home/rob/dq-runs/qwen3-4b-clado-legal-rigorous-20260516T133901Z/vllm/pq_high_4p7527/wikitext_ppl_32k.json` |

Linearly interpolating PQ PPL to CLADO's bpp gives `19.677387`.
CLADO is therefore `+1.785379` PPL worse, or a `9.07%` relative
regression. It is also worse than the lower-bpp PQ point by `9.55%`
and worse than the higher-bpp PQ point by `8.43%`.

Conclusion: legal CLADO does not improve Qwen3-4B over standard PQ at
the same bit rate. The earlier apparent win was not robust to runtime
legalization and larger KL/PPL validation.

Follow-up implementation: CLADO sweeps can now solve archived payloads
over model-structure units before kneedle expansion. Use
`--structure-scope runtime` for profile-owned serving groups, `subblock`
for attention/MLP groups, `layer` for whole transformer layers, or `none`
to preserve the archived flat unit space. The default CLI scope is
`runtime`, and `kneedle` refuses a sweep generated with a different scope.

## Structured rerun vs standard PQ

Run:
`/home/rob/dq-runs/qwen3-4b-clado-structured-20260516T150203Z`

Summary artifact:
`/home/rob/dq-runs/qwen3-4b-clado-structured-20260516T150203Z/artifacts/structured_vs_pq_summary.json`

The archived Qwen3-4B CLADO payload was re-solved with two stricter
structure scopes:

- `subblock`: 145 input units coarsened to 73 attention/MLP units.
- `layer`: 145 input units coarsened to 37 layer units.

Sweep and kneedle artifacts:

- Subblock sweep:
  `/home/rob/dq-runs/qwen3-4b-clado-structured-20260516T150203Z/artifacts/clado_subblock_sweep.json`
- Subblock candidates:
  `/home/rob/dq-runs/qwen3-4b-clado-structured-20260516T150203Z/artifacts/clado_subblock_candidates`
- Layer sweep:
  `/home/rob/dq-runs/qwen3-4b-clado-structured-20260516T150203Z/artifacts/clado_layer_sweep.json`
- Layer candidates:
  `/home/rob/dq-runs/qwen3-4b-clado-structured-20260516T150203Z/artifacts/clado_layer_candidates`

First-pass KL used the same production-cache contract as the legal rerun:
`diverse-v1.jsonl`, 128 samples, sequence length 1024, train seed 42,
last-token KL, `hooks` materialization, required source and production-cache
prefetch, and the Qwen3-4B frontier production cache.

First-pass results:

| structured candidate | bpp | KL | standard PQ reference | PQ bpp | PQ KL | result |
|---|---:|---:|---|---:|---:|---:|
| subblock_4p6919 | 4.712392 | 0.121701 | pq_4p6890 | 4.688988 | 0.117099 | 3.93% worse |
| subblock_4p7859 | 4.816558 | 0.181323 | pq_4p7831 | 4.783072 | 0.129836 | 39.66% worse |
| subblock_kneedle_4p9811 | 5.032558 | 0.163196 | pq_4p9806 | 4.980574 | 0.133552 | 22.20% worse |
| layer_4p6882 | 4.708333 | 0.150782 | pq_4p6890 | 4.688988 | 0.117099 | 28.76% worse |
| layer_4p7823 | 4.812500 | 0.186065 | pq_4p7831 | 4.783072 | 0.129836 | 43.31% worse |
| layer_kneedle_5p3531 | 5.444444 | 0.138960 | pq_5p3531 | 5.353107 | 0.123812 | 12.24% worse |

First-pass output:
`/home/rob/dq-runs/qwen3-4b-clado-structured-20260516T150203Z/artifacts/kl_seed42_n128_structured_candidates.json`

First-pass log:
`/home/rob/dq-runs/qwen3-4b-clado-structured-20260516T150203Z/logs/kl_seed42_n128_structured_candidates.log`

The best structured point was `subblock_4p6919`, so it was rechecked
against a standard-PQ assignment targeted to the same bpp with 256
calibration samples:

| assignment | bpp | KL | formats |
|---|---:|---:|---|
| subblock_best_4p712 | 4.712392 | 0.154827 | 229 NVFP4, 23 MXFP8 |
| pq_matched_4p709 | 4.709276 | 0.123714 | 229 NVFP4, 22 FP8_E4M3, 1 BF16 |

At matched bpp, the best structured CLADO point is `+0.031113` KL worse,
a `25.15%` relative regression.

Confirmation output:
`/home/rob/dq-runs/qwen3-4b-clado-structured-20260516T150203Z/artifacts/kl_seed42_n256_structured_best_vs_pq.json`

Confirmation log:
`/home/rob/dq-runs/qwen3-4b-clado-structured-20260516T150203Z/logs/kl_seed42_n256_structured_best_vs_pq.log`

Conclusion: respecting subblock/layer structure makes CLADO more interpretable
and runtime-safe, but it still does not beat standard PQ on Qwen3-4B. Since
every structured candidate regressed KL, no vLLM PPL export was run for this
round.

## Commands

Candidate generation:

```bash
python3 -m prismaquant.research_components.block_clado_runtime sweep \
  --payload /home/rob/dq-runs/qwen3-4b-block-clado-low-bpp-20260506T082141Z/iter_0/block_clado.json \
  --n-lambdas 81 \
  --output /tmp/pq_4b_sweep_test.json

python3 -m prismaquant.research_components.block_clado_runtime kneedle \
  --payload /home/rob/dq-runs/qwen3-4b-block-clado-low-bpp-20260506T082141Z/iter_0/block_clado.json \
  --sweep /tmp/pq_4b_sweep_test.json \
  --output-dir /home/rob/dq-runs/qwen3-4b-clado-assign-opt-20260516T053432Z/artifacts/kneedle_candidates_current \
  --n-neighbors 4
```

Validation ran inside the CUDA docket container image
`vllm-fresh-b12x-fla:latest` because the host Python is CPU-only.  The
validator command used:

```bash
python3 -m prismaquant.validate_assignments_kl \
  --model /home/rob/.cache/huggingface/Qwen3-4B \
  --probe /home/rob/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/probe.pkl \
  --base-assignment /home/rob/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/pareto_assignments/allocator_target_4p5000_achieved_4p5000_19c5586bb07d.json \
  --formats NVFP4,MXFP8_E4M3,FP8_E4M3,BF16 \
  --dataset /home/rob/dq-runs/calibration/diverse-v1.jsonl \
  --n-calib-samples 64 \
  --calib-seqlen 1024 \
  --calib-split train \
  --calib-seed 42 \
  --kl-scope last_token \
  --kl-cuda-graphs off \
  --assignment-materialization hooks \
  --production-weight-cache /home/rob/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/production_weight_cache_frontier_raw.pkl \
  --production-cache-dir-override /home/rob/dq-runs/qwen3-4b-frontier-kneedle-20260515T040000Z/artifacts/production_weight_cache_frontier \
  --production-cache-lru-gb 64 \
  --production-cache-prefetch require \
  --production-cache-prefetch-workers 4 \
  --source-prefetch require
```

vLLM materialization and quality testing used the same CUDA docket image:
`vllm-fresh-b12x-fla:latest`.

Representative quality command:

```bash
python3 tools/measure_vllm_wikitext_ppl.py \
  --model /home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p737/exported_vllm_fused \
  --quantization compressed-tensors \
  --output /home/rob/dq-runs/qwen3-4b-clado-vllm-20260516T085554Z/clado_4p737/logs/vllm_wikitext_ppl_4k.json \
  --dataset-cache-dir /home/rob/.cache/huggingface/datasets \
  --n-tokens 4096 \
  --seqlen 512 \
  --gpu-memory-utilization 0.55
```
