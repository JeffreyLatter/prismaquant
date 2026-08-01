# Production-Render Staged Allocator: 27B Results, 2026-05-21

Work dir:

`/home/rob/dq-runs/qwen36-27b-production-render-staged-20260521T143914Z`

Model:

`/home/rob/.cache/huggingface/qwen36-27b-bf16`

Calibration/scoring contract:

- Production render scoring: 8 samples, sequence length 1024, seed 42.
- Render levers: `gptq,joint_scale_opt`.
- Stage 1 rendered NVFP4 for all 496 quantizable Linears.
- Stage 2 selected the top 30% NVFP4 local forward-error tail.
- Stage 3 rendered MXFP8_E4M3 only for that selected tail.
- Allocation target: 5.5 bpp over quantizable parameters.

## Render Results

NVFP4 all-Linears render:

- Output: `artifacts/production_render_score_staged_nvfp4_cache.pkl`
- Cache dir: `cache_dir/`
- Entries: 496
- Failures: 0
- Runtime: 9067.4 s

Tail selection:

- Output: `artifacts/production_render_score_tail_qnames.txt`
- Selected: 149 / 496 qnames
- Threshold `score_sum`: 23490.197265625
- Max `score_sum`: 211638.75

MXFP8_E4M3 tail render:

- Output: `artifacts/production_render_score_staged_mxfp8_cache.pkl`
- Entries: 149
- Failures: 0
- Runtime: 50.6 s

MXFP8 did not look reliably better than NVFP4 under the recorded score:

- MXFP8 better than NVFP4: 19 / 149 tail qnames
- MXFP8 worse or equal: 130 / 149 tail qnames
- `MXFP8/NVFP4 score_sum` ratio: min 0.5688, median 1.9850, mean 2.5882, max 10.6870

The worst regressions were linear-attention projections, for example
`model.layers.0.linear_attn.in_proj_qkv` scored 10.687x worse under MXFP8 than
NVFP4. Treat this as a scoring-path/activation-scale warning, not a shipping
signal.

## Allocation

Artifacts:

- Cost: `artifacts/cost.pkl`
- Layer config: `artifacts/layer_config_staged_5p5.json`
- Pareto CSV: `artifacts/pareto_staged.csv`

Allocator output at target 5.5:

- Achieved bpp: 5.500322962643987
- Predicted loss: 6.649306e6
- Raw layer-config counts: NVFP4 450, BF16 54, MXFP8_E4M3 0

Pareto rows:

| target | achieved | NVFP4 layers | BF16 layers | predicted loss |
|---:|---:|---:|---:|---:|
| 5.0 | 4.990257 | 339 | 13 | 7.823934e6 |
| 5.5 | 5.500323 | 329 | 23 | 6.649306e6 |
| 6.0 | 5.998008 | 321 | 31 | 5.577147e6 |

The allocator selected no MXFP8 at 5.0, 5.5, or 6.0. It spent extra budget on
BF16 rescues.

## Validation

In-tree assignment KL screen:

- Command class: `python3 -m prismaquant.validate_assignments_kl`
- Production cache: `artifacts/production_render_score_staged_nvfp4_cache.pkl`
- Cache dir override: `cache_dir/`
- LRU: 8 GiB
- Calibration: 8 samples, sequence length 512, seed 42
- KL scope: last token
- Output: `artifacts/kl_validate_assignments_staged_5p5.json`

Result:

- Staged 5.5 last-token KL: 0.02322502905735746
- Staged bpp: 5.500322962643987
- Format counts: NVFP4 450, BF16 54

Prior same-screen references:

- Shipped/baseline 5.5 last-token KL: 0.027974171278401627
- Grouped-KL 5.5 last-token KL: 0.07498542190296575

Direct validation-harness WikiText PPL, 8192 tokens, CUDA graphs off, LRU 8:

- Staged 5.5: 10.826764342210923
- Baseline 5.5: 8.329411116803316
- Staged output: `artifacts/validation_staged_5p5_direct_8192.log`
- Baseline output: `artifacts/validation_baseline_5p5_direct_8192.log`

Prior vLLM WikiText PPL references on 8192 fixed token IDs:

- Shipped 5.5: 9.49828185485716
- Grouped-KL 5.5: 9.740415491878867

## Decision

Do not ship this allocator as-is.

The staged production-render allocator improves the narrow last-token KL screen,
but it substantially regresses direct WikiText PPL versus the baseline under the
same validation harness. The MXFP8 scoring behavior is also suspicious: MXFP8 is
worse than NVFP4 on most of the measured high-error tail, with large regressions
on linear-attention projections. Before shipping, the MXFP8 activation scoring
path and the allocator's validation objective need another pass.
